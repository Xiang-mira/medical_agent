"""
Unit tests for medai core modules.
All tests are self-contained: no real CT files, no GPU, no network.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_nii(arr: np.ndarray, path: Path) -> Path:
    """Save a numpy array as a minimal NIfTI file."""
    import nibabel as nib
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(path))
    return path


def _sphere_mask(shape=(32, 32, 32), radius=8, offset=(0, 0, 0)):
    """Binary sphere mask centred in `shape`."""
    cx, cy, cz = [s // 2 + o for s, o in zip(shape, offset)]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2).astype(np.uint8)


# ─── Test 1 : Label Verifier — DSC routing ────────────────────────────────────

class TestLabelVerifier:
    def test_accept_when_dsc_above_threshold(self, tmp_path):
        from cli_anything.medai.core.label_verifier import verify_annotation
        mask = _sphere_mask()
        a = _make_nii(mask, tmp_path / "a.nii.gz")
        b = _make_nii(mask, tmp_path / "b.nii.gz")          # identical → DSC=1.0
        result = verify_annotation(a, b, "liver", dsc_replace_threshold=0.0, dsc_vlm_threshold=0.8)
        assert result["decision"] == "accept"
        assert abs(result["dice"] - 1.0) < 1e-4

    def test_send_to_vlm_when_dsc_below_threshold(self, tmp_path):
        from cli_anything.medai.core.label_verifier import verify_annotation
        a_mask = _sphere_mask(radius=10)
        b_mask = _sphere_mask(radius=5, offset=(8, 0, 0))   # shifted smaller → low DSC
        a = _make_nii(a_mask, tmp_path / "a.nii.gz")
        b = _make_nii(b_mask, tmp_path / "b.nii.gz")
        result = verify_annotation(a, b, "pancreas", dsc_replace_threshold=0.0, dsc_vlm_threshold=0.8)
        assert result["decision"] == "send_to_vlm_label_expert"
        assert result["dice"] < 0.8

    def test_auto_replace_when_annotation_empty(self, tmp_path):
        from cli_anything.medai.core.label_verifier import verify_annotation
        empty = np.zeros((32, 32, 32), dtype=np.uint8)
        pred  = _sphere_mask()
        a = _make_nii(empty, tmp_path / "a.nii.gz")
        b = _make_nii(pred,  tmp_path / "b.nii.gz")
        result = verify_annotation(a, b, "spleen", dsc_replace_threshold=0.0, dsc_vlm_threshold=0.5)
        assert result["decision"] == "auto_replace_candidate"

    def test_disjoint_nonempty_masks_route_to_vlm(self, tmp_path):
        from cli_anything.medai.core.label_verifier import verify_annotation
        # Both masks non-empty but placed at opposite corners → DSC=0, not disjoint empty.
        # Should go to VLM, NOT auto_replace (localization error, not missing annotation).
        a_mask = _sphere_mask(offset=(-10, -10, -10))
        b_mask = _sphere_mask(offset=(10, 10, 10))
        a = _make_nii(a_mask, tmp_path / "a.nii.gz")
        b = _make_nii(b_mask, tmp_path / "b.nii.gz")
        result = verify_annotation(a, b, "pancreas", dsc_replace_threshold=0.0, dsc_vlm_threshold=0.5)
        assert result["dice"] == 0.0
        assert result["decision"] == "send_to_vlm_label_expert"


# ─── Test 2 : AnnotationManager — versioned storage ───────────────────────────

class TestAnnotationManager:
    def test_save_raw_and_get_raw(self, tmp_path):
        from cli_anything.medai.core.annotation_manager import AnnotationManager
        mask = _sphere_mask()
        src = _make_nii(mask, tmp_path / "pancreas.nii.gz")
        mgr = AnnotationManager(tmp_path / "store", "case_001")
        mgr.save_raw("pancreas", src)
        got = mgr.get_raw("pancreas")
        assert got is not None and got.exists()

    def test_save_prediction_and_apply_update(self, tmp_path):
        from cli_anything.medai.core.annotation_manager import AnnotationManager
        mask = _sphere_mask()
        raw_src  = _make_nii(mask, tmp_path / "raw.nii.gz")
        pred_src = _make_nii(mask, tmp_path / "pred.nii.gz")
        mgr = AnnotationManager(tmp_path / "store", "case_001")
        mgr.save_raw("liver", raw_src)
        mgr.save_prediction("liver", pred_src, round_idx=0)
        # winner B → copy prediction to updated/
        updated = mgr.apply_update("liver", winner="B", round_idx=0)
        assert updated is not None and updated.exists()
        # get_current should now return updated/
        current = mgr.get_current("liver")
        assert current == updated

    def test_log_decision_and_get_history(self, tmp_path):
        from cli_anything.medai.core.annotation_manager import AnnotationManager
        mgr = AnnotationManager(tmp_path / "store", "case_001")
        mgr.log_decision("pancreas", 0, {"winner": "A", "confidence": 0.9})
        mgr.log_decision("pancreas", 1, {"winner": "B", "confidence": 0.7})
        history = mgr.get_history()
        assert len(history) == 2
        assert history[0]["organ"] == "pancreas"
        assert history[1]["winner"] == "B"


# ─── Test 3 : VLM Label Expert — response parsing ─────────────────────────────

class TestVLMParsing:
    def _parse(self, text):
        from cli_anything.medai.core.vlm_label_expert import _parse_vlm_response
        return _parse_vlm_response(text)

    def test_clean_json_parsed(self):
        raw = '{"winner": "A", "confidence": 0.9, "reason": "good boundaries"}'
        r = self._parse(raw)
        assert r["winner"] == "A"
        assert r["confidence"] == pytest.approx(0.9)
        assert r["parse_status"] == "success"

    def test_thinking_block_stripped_before_parse(self):
        raw = "<think>let me think...</think>\n{\"winner\": \"B\", \"confidence\": 0.8, \"reason\": \"model better\"}"
        r = self._parse(raw)
        assert r["winner"] == "B"
        assert r["parse_status"] == "success"

    def test_keyword_fallback_explicit_winner_a(self):
        # Unambiguous: "winner = A" triggers fallback correctly
        raw = 'After reviewing both panels, winner = A seems more anatomically correct.'
        r = self._parse(raw)
        assert r["winner"] == "A"
        assert r["parse_status"] == "keyword_fallback"

    def test_ambiguous_candidate_a_returns_uncertain(self):
        # "candidate a" alone is ambiguous: "Candidate A is worse than Candidate B"
        # should NOT select A. Conservative: return uncertain.
        raw = 'The best annotation is candidate a based on anatomy.'
        r = self._parse(raw)
        # Strict fallback: ambiguous phrasing → uncertain (safer for medical context)
        assert r["winner"] == "uncertain"
        assert r["parse_status"] == "unparseable"

    def test_unparseable_returns_uncertain(self):
        r = self._parse("I cannot determine which is better.")
        assert r["winner"] == "uncertain"
        assert r["parse_status"] == "unparseable"


# ─── Test 4 : RadThinking — negation-aware suspicious term detection ───────────

class TestRadThinkingNegation:
    def _check(self, text, tmp_path):
        from cli_anything.medai.core.radthinking import parse_report_sections
        p = tmp_path / "report.txt"
        p.write_text(text, encoding="utf-8")
        return parse_report_sections(p)["suspicious_terms_found"]

    def test_negated_suspicious_not_flagged(self, tmp_path):
        found = self._check("No suspicious lesion identified in the pancreas.", tmp_path)
        assert "suspicious" not in found

    def test_negated_growing_not_flagged(self, tmp_path):
        found = self._check("Not growing; mass is stable over the follow-up period.", tmp_path)
        assert "growing" not in found

    def test_without_cancer_not_flagged(self, tmp_path):
        found = self._check("Without cancer involvement of the bile duct.", tmp_path)
        assert "cancer" not in found

    def test_affirmed_suspicious_flagged(self, tmp_path):
        found = self._check("Suspicious mass in the liver with malignant features.", tmp_path)
        assert "suspicious" in found
        assert "malign" in found

    def test_affirmed_recurrence_flagged(self, tmp_path):
        found = self._check("Recurrence detected in the pancreatic bed.", tmp_path)
        assert "recurrence" in found


# ─── Test 5 : QC Checker — temporal ratio logic ───────────────────────────────

class TestQCChecker:
    def _run_qc(self, tmp_path):
        from cli_anything.medai.core.qc_checker import check_segmentation_quality
        seg_dir = tmp_path / "segmentations"
        seg_dir.mkdir()
        mask = _sphere_mask()
        _make_nii(mask, seg_dir / "liver.nii.gz")
        return check_segmentation_quality(seg_dir, expected_organs=["liver", "pancreas"])

    def test_missing_organ_flagged(self, tmp_path):
        result = self._run_qc(tmp_path)
        issues = result["issues"]
        organs_flagged = [i["organ"] for i in issues]
        assert "pancreas" in organs_flagged   # pancreas missing → flagged

    def test_present_organ_not_missing(self, tmp_path):
        result = self._run_qc(tmp_path)
        issues = result["issues"]
        missing_issues = [i for i in issues if i["organ"] == "liver" and i["type"] == "missing_organ"]
        assert len(missing_issues) == 0       # liver present → not missing


# ─── Test 6 : Projection Builder — missing file handling ──────────────────────

class TestProjectionBuilder:
    def test_missing_ct_returns_failed(self, tmp_path):
        from cli_anything.medai.core.projection_builder import build_projection
        result = build_projection(
            ct_image=tmp_path / "nonexistent.nii.gz",
            mask_a=None, mask_b=None,
            output_folder=tmp_path / "out",
            organ="liver",
        )
        assert result["status"] == "failed"

    def test_successful_projection_returns_expected_keys(self, tmp_path):
        from cli_anything.medai.core.projection_builder import build_projection
        ct   = _make_nii(np.random.randint(-100, 200, (32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask = _make_nii(_sphere_mask(), tmp_path / "mask.nii.gz")
        result = build_projection(ct, mask_a=mask, mask_b=None,
                                  output_folder=tmp_path / "out", organ="liver")
        assert result["stage"] == "projection_builder"
        assert "saved_projections" in result
        assert "output_folder" in result


# ─── Test 7 : EM Loop — dry-run end-to-end ────────────────────────────────────

class TestEmLoop:
    def test_dry_run_completes_and_saves_metrics(self, tmp_path):
        from cli_anything.medai.core.em_loop import run_em_loop
        result = run_em_loop(
            case_id="test_case",
            ct_image=tmp_path / "nonexistent_ct.nii.gz",   # dry_run skips inference
            annotation_folder=tmp_path / "anns",
            output_folder=tmp_path / "out",
            organs=["liver", "pancreas"],
            num_rounds=1,
            dry_run=True,
        )
        assert result["status"] == "dry_run"
        assert len(result["rounds"]) == 1
        assert result["rounds"][0]["m_step"]["status"] == "stub"
        metrics_path = Path(result["rounds_metrics_json"])
        assert metrics_path.exists()
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert saved["case_id"] == "test_case"

    def test_dry_run_annotation_summary_structure(self, tmp_path):
        from cli_anything.medai.core.em_loop import run_em_loop
        result = run_em_loop(
            case_id="case_x",
            ct_image=tmp_path / "ct.nii.gz",
            annotation_folder=tmp_path / "anns",
            output_folder=tmp_path / "out",
            organs=["liver"],
            num_rounds=1,
            dry_run=True,
        )
        summary = result["annotation_summary"]
        assert summary["case_id"] == "case_x"
        assert "rounds_completed" in summary
        assert "decisions" in summary
