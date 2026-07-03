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


def _make_nii_with_affine(arr: np.ndarray, path: Path, affine: np.ndarray) -> Path:
    """Save a numpy array as a NIfTI file with explicit geometry."""
    import nibabel as nib
    img = nib.Nifti1Image(arr, affine=affine)
    nib.save(img, str(path))
    return path


def _sphere_mask(shape=(32, 32, 32), radius=8, offset=(0, 0, 0)):
    """Binary sphere mask centred in `shape`."""
    cx, cy, cz = [s // 2 + o for s, o in zip(shape, offset)]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2).astype(np.uint8)




class TestLabelCritic373PromptBank:
    def test_organ_ct_appearance_373_covers_formal_targets(self):
        root = Path(__file__).resolve().parents[2]
        targets = json.loads((root / "configs" / "student_3d_prompt_target_organs.json").read_text(encoding="utf-8"))
        doc = json.loads((root / "configs" / "organ_ct_appearance_373.json").read_text(encoding="utf-8"))
        organs = targets["target_organs"]
        bank = doc["organ_ct_appearance"]
        entries = doc["entries"]

        assert doc["entry_count"] == 373
        assert len(entries) == 373
        assert len(organs) == 373
        assert set(bank) == set(organs)
        assert {e["canonical_organ"] for e in entries} == set(organs)
        required = {"ct_appearance", "expected_location", "common_failure_modes", "rejection_rules", "source", "requires_manual_review"}
        for organ, entry in bank.items():
            missing = [key for key in required if key not in entry or entry[key] in (None, "", [])]
            assert not missing, f"{organ} missing {missing}"
        assert bank["liver"]["source"] == "labelcritic_seed"
        assert bank["pancreas"]["source"] == "labelcritic_seed"
        non_seed = next(entry for organ, entry in bank.items() if organ not in {"liver", "pancreas", "aorta", "spleen", "stomach", "gall_bladder"})
        assert non_seed["requires_manual_review"] is False
        assert non_seed["automatic_failure_action"] in {
            "allow_class_agnostic_pairwise", "abstain_runtime_only_unverified"
        }

    def test_prompt_provenance_prefers_seed_then_373_bank(self):
        from cli_anything.medai.core import labelcritic_wrapper as lw

        liver = lw._organ_description_provenance("liver")
        colon = lw._organ_description_provenance("colon")

        assert liver["prompt_source"] == "labelcritic_seed"
        assert liver["requires_manual_review"] is False
        assert colon["organ_description_available"] is True
        assert colon["organ_description_hash"]
        assert colon["requires_manual_review"] is False

    def test_labelcreator_prompt_template_contains_candidate_ranking_contract(self):
        from cli_anything.medai.core.labelcreator_prompt_template import build_labelcreator_prompt

        root = Path(__file__).resolve().parents[2]
        doc = json.loads((root / "configs" / "organ_ct_appearance_373.json").read_text(encoding="utf-8"))
        entry = next(e for e in doc["entries"] if e["canonical_organ"] == "colon")
        prompt = build_labelcreator_prompt(
            entry=entry,
            case_id="debug_case",
            candidate_table="candidate_id: c1\nteacher: epai\nfamily: ePAI\n\ncandidate_id: c2\nteacher: vsmtrans\nfamily: VSmTrans",
            mode="ranking",
        )

        assert "Target organ/structure: colon" in prompt
        assert "CT appearance" in prompt
        assert "candidate_id: c1" in prompt and "candidate_id: c2" in prompt
        assert "Do not prefer a candidate only because" in prompt
        assert "selected_reason" in prompt and "rejected_reasons" in prompt
        assert "Return strict JSON only" in prompt


# ─── Test 0 : Label Merger — nearest-neighbor resampling ─────────────────────

class TestLabelMerger:
    def test_geometry_mismatch_is_resampled_to_base_grid(self, tmp_path):
        from cli_anything.medai.core.label_merger import merge_case_segmentations

        case_root = tmp_path / "case_001"
        (case_root / "per_model" / "model_a" / "segmentations").mkdir(parents=True)
        (case_root / "per_model" / "model_b" / "segmentations").mkdir(parents=True)

        liver = np.zeros((10, 10, 10), dtype=np.uint8)
        liver[1:4, 1:4, 1:4] = 1
        pancreas = np.zeros((5, 5, 5), dtype=np.uint8)
        pancreas[1:3, 1:3, 1:3] = 1

        _make_nii(liver, case_root / "per_model" / "model_a" / "segmentations" / "liver.nii.gz")
        _make_nii_with_affine(
            pancreas,
            case_root / "per_model" / "model_b" / "segmentations" / "pancreas.nii.gz",
            np.diag([2.0, 2.0, 2.0, 1.0]),
        )

        global_space = tmp_path / "global_label_space.json"
        global_space.write_text(json.dumps({
            "organ_to_id": {"liver": 1, "pancreas": 2},
            "id_to_organ": {"1": "liver", "2": "pancreas"},
        }), encoding="utf-8")

        route_result = {
            "requested_organs": ["liver", "pancreas"],
            "ranked_candidates": {
                "liver": [{"model_key": "model_a"}],
                "pancreas": [{"model_key": "model_b"}],
            },
        }
        registry = {"models": {"model_a": {}, "model_b": {}}}

        result = merge_case_segmentations(
            case_root,
            route_result,
            registry,
            global_label_space_path=global_space,
            alias_config_path=None,
        )

        assert result["status"] == "success"
        assert result["coverage_summary"]["merged_organs"] == 2
        assert result["coverage_summary"]["geometry_error_organs"] == 0
        assert result["selected_organs"]["pancreas"]["resampled_to_base"] is True
        assert result["selected_organs"]["pancreas"]["merged_shape"] == [10, 10, 10]


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


# ─── Test 6 : Model Registry — formal routing keys ───────────────────────────

class TestModelRegistryRouting:
    def test_candidate_models_resolve_cads_family_to_runnable_keys(self):
        from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry

        registry = load_registry(Path("configs/model_registry.yaml"))
        candidates = candidate_models_for_organs(
            registry,
            ["aorta", "liver", "pancreas", "submandibular_gland_left"],
        )
        all_model_keys = {
            model_key
            for organ_candidates in candidates.values()
            for model_key in organ_candidates
        }

        assert "cads" not in all_model_keys
        assert {"cads551", "cads558"} & all_model_keys
        assert all_model_keys <= set(registry["models"])


class TestTotalSegmentatorTimeoutContract:
    def test_totalseg_runner_defaults_to_no_subprocess_timeout(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import totalseg_runner as ts

        ct = tmp_path / "ct.nii.gz"
        ct.write_text("fake", encoding="utf-8")
        seen_timeouts: list[int | None] = []

        def fake_run(cmd, **kwargs):
            seen_timeouts.append(kwargs.get("timeout"))
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            _make_nii(np.ones((4, 4, 4), dtype=np.uint8), out_dir / "liver.nii.gz")
            return __import__("subprocess").CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(ts, "find_totalseg_executable", lambda: "TotalSegmentator")
        monkeypatch.setattr(ts.subprocess, "run", fake_run)

        result = ts.run_totalseg_with_contract(
            ct,
            tmp_path / "per_model" / "totalsegmentator",
            tmp_path / "case_out",
            subtasks=["total"],
            case_id="case_001",
        )

        assert result["status"] == "success"
        assert seen_timeouts == [None]

    def test_registered_totalseg_ignores_global_timeout_sec(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import registered_infer as ri

        ct = tmp_path / "ct.nii.gz"
        ct.write_text("fake", encoding="utf-8")
        captured: dict[str, object] = {}

        monkeypatch.setattr(ri, "load_registry", lambda _: {
            "models": {
                "totalsegmentator": {
                    "runner": "builtin_totalsegmentator",
                    "recipe": "official",
                }
            }
        })
        monkeypatch.setattr(ri, "_totalseg_subtasks_for_context", lambda *_: (["total"], []))

        def fake_contract(*args, **kwargs):
            captured.update(kwargs)
            seg = tmp_path / "case_out" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            return {"status": "success", "segmentation_output": str(seg)}

        monkeypatch.setattr(ri, "run_totalseg_with_contract", fake_contract)

        result = ri.run_registered_model(
            ct,
            tmp_path / "case_out",
            "totalsegmentator",
            registry_path=tmp_path / "registry.yaml",
            timeout_sec=600,
        )

        assert result["status"] == "success"
        assert captured["timeout_sec"] is None


class TestFormal373TargetAndAutoFineLabels:
    def test_target_validation_accepts_current_373_config(self):
        from cli_anything.medai.core.target_space import validate_formal_373_target_space

        result = validate_formal_373_target_space("configs/student_3d_prompt_target_organs.json", require_full_target=False)
        assert result["status"] == "success"
        assert result["counts"]["target_organs"] == 373
        assert result["counts"]["unique_target_organs"] == 373
        assert result["historical_count_explanation"]["373"].startswith("current accepted")

    def test_target_validation_rejects_non_target_request(self):
        from cli_anything.medai.core.target_space import validate_formal_373_target_space

        result = validate_formal_373_target_space(
            "configs/student_3d_prompt_target_organs.json",
            requested_organs=["liver", "not_a_real_organ"],
            require_full_target=False,
        )
        assert result["status"] == "failed"
        assert result["blocking"]["requested_non_target_organs"] == ["not_a_real_organ"]

    def test_label_passport_maps_grade_to_training_weight(self, tmp_path):
        from cli_anything.medai.core.auto_fine_label import build_label_passport

        mask = _make_nii(_sphere_mask(), tmp_path / "liver.nii.gz")
        passport = build_label_passport({
            "case_id": "case_001",
            "organ": "liver",
            "mask_path": str(mask),
            "selected_model": "teacher_a",
            "candidate_models": ["teacher_a", "teacher_b"],
            "selection_method": "label_critic",
            "selected_candidate_qc_status": "pass",
            "shapekit_status": "success",
            "selected_pseudo_consistency_dice": 0.9,
        })
        assert passport["grade"] == "D"
        assert passport["training_weight"] == 0.0
        assert passport["auto_fine_label_status"] == "unresolved"
        assert passport["scoring_schema_version"] == "autolabel_core_v2"


class TestVoxTellStudentContracts:
    def test_dry_run_records_batching_and_io_contract(self, tmp_path):
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = tmp_path / "ct.nii.gz"
        ct.write_text("fake", encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, device="cpu")
        result = student.segment(
            ct,
            tmp_path / "out",
            prompts=["liver", "spleen", "kidney_left"],
            dry_run=True,
            prompt_batch_size=2,
        )
        assert result["status"] == "dry_run"
        assert result["num_batches"] == 2
        assert result["expected_masks"]["liver"].endswith("liver.nii.gz")
        assert result["organ_to_prompt"]["kidney_left"] == "left kidney"
        assert result["official_output_masks"]["kidney_left"].endswith("ct_left_kidney.nii.gz")
        assert result["expected_masks"]["kidney_left"].endswith("kidney_left.nii.gz")
        assert result["io_contract"]["combined_multilabel_policy"].startswith("not used formally")


    def test_voxtell_segment_uses_prompt_overrides_in_dry_run(self, tmp_path):
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({"target_organs": ["liver"], "organ_to_prompt": {"liver": "liver"}}), encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        result = student.segment(
            ct,
            tmp_path / "out",
            prompts=["liver"],
            prompt_overrides={"liver": "segment the hepatic organ"},
            dry_run=True,
        )
        assert result["organ_to_prompt"]["liver"] == "segment the hepatic organ"
        assert "segment the hepatic organ" in result["command"]
        assert result["expected_masks"]["liver"].endswith("liver.nii.gz")
        assert "segment_the_hepatic_organ" in result["official_output_masks"]["liver"]

    def test_manifest_adds_positive_negative_and_prompt_variants(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDAI_NEGATIVE_PROMPT_RATIO", "1.0")
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": ["liver", "spleen"],
            "organ_to_prompt": {"liver": "liver", "spleen": "spleen"},
            "organ_to_student_id": {"liver": 1, "spleen": 2},
            "prompt_variants": {"liver": ["hepatic organ"]},
        }), encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case = tmp_path / "cases" / "case_001"
        upd = case / "updated"
        upd.mkdir(parents=True)
        mask = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "liver.nii.gz")
        (case / "selection_metadata.json").write_text(json.dumps({
            "case_id": "case_001",
            "ct_path": str(ct),
            "selected_organs": [{
                "organ": "liver", "mask_path": str(mask), "selected_model": "teacher_a",
                "teacher_lineage": ["teacher_a"], "grade": "A", "training_weight": 1.0,
                "distillation_eligible": True, "selection_method": "label_critic",
                "scoring_schema_version": "autolabel_core_v2",
            }],
        }), encoding="utf-8")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(case.parent, tmp_path / "manifest.json", case_list=case_list, require_images=True)

        roles = {item["supervision_type"] for item in manifest["items"]}
        assert {"positive", "negative"} <= roles
        positives = [item for item in manifest["items"] if item["supervision_type"] == "positive"]
        negatives = [item for item in manifest["items"] if item["supervision_type"] == "negative"]
        pos = next(item for item in positives if item["prompt"] == "liver")
        neg = negatives[0]
        assert manifest["num_prompt_variant_items"] == 0
        assert {item["prompt"] for item in positives} == {"liver"}
        assert pos["prompt_variant_index"] == 0
        assert pos["teacher_lineage"] == ["teacher_a"]
        assert neg["negative_reason"] == "nonmedical_object_not_present_in_medical_ct"
        assert neg["negative_source"] == "nonmedical_absent_object"
        assert neg["negative_prompt_category"] == "nonmedical_absent_object"
        assert neg["zero_mask_role"] == "negative_target_mask"
        assert neg["organ"].startswith("negative_nonmedical_")
        assert neg["prompt_family_id"] == f"case_001:{neg['organ']}:negative"
        assert Path(neg["mask"]).exists()
        assert manifest["zero_mask_targets"]
        assert "negative_quota_policy" in manifest
        assert "negative_source_shortfalls" in manifest


    def test_manifest_includes_absent_negative_from_selection_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDAI_NEGATIVE_PROMPT_RATIO", "0.0")
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": ["liver", "brain"],
            "organ_to_prompt": {"liver": "liver", "brain": "brain"},
            "organ_to_student_id": {"liver": 1, "brain": 2},
        }), encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case = tmp_path / "cases" / "case_001"
        upd = case / "updated"
        upd.mkdir(parents=True)
        liver = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "liver.nii.gz")
        (upd / "negative_targets").mkdir(parents=True)
        zero = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), upd / "negative_targets" / "zero_mask.nii.gz")
        (case / "selection_metadata.json").write_text(json.dumps({
            "case_id": "case_001",
            "ct_path": str(ct),
            "selected_organs": [
                {"organ": "liver", "mask_path": str(liver), "grade": "A", "training_weight": 1.0, "scoring_schema_version": "autolabel_core_v2"},
                {"organ": "brain", "mask_path": str(zero), "final_mask": str(zero), "target_type": "absent_negative", "grade": "A", "grade_scope": "absence_target", "training_weight": 0.1, "distillation_eligible": True, "scoring_schema_version": "autolabel_core_v3_absent_negative", "negative_source": "case_373_expected_absent"},
            ],
        }), encoding="utf-8")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(case.parent, tmp_path / "manifest.json", case_list=case_list, require_images=True)

        negatives = [item for item in manifest["items"] if item.get("target_type") == "negative_absent"]
        assert len(negatives) == 1
        neg = negatives[0]
        assert neg["organ"] == "brain"
        assert neg["supervision_type"] == "negative"
        assert neg["training_gate_decision"] == "include_absent_negative"
        assert manifest["expected_targets"] == 2
        assert manifest["absent_negative_targets"] == 1
        assert manifest["target_type_counts"]["negative_absent"] == 1

    def test_voxtell_manifest_rejects_legacy_and_hard_c_positive_labels(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDAI_NEGATIVE_PROMPT_RATIO", "0.0")
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": ["liver", "spleen", "pancreas"],
            "organ_to_prompt": {"liver": "liver", "spleen": "spleen", "pancreas": "pancreas"},
        }), encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case = tmp_path / "cases" / "case_001"
        upd = case / "updated"
        upd.mkdir(parents=True)
        liver = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "liver.nii.gz")
        spleen = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "spleen.nii.gz")
        pancreas = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "pancreas.nii.gz")
        probability = _make_nii(np.full((8, 8, 8), 0.25, dtype=np.float32), case / "probability_mask.nii.gz")
        (case / "selection_metadata.json").write_text(json.dumps({
            "case_id": "case_001",
            "ct_path": str(ct),
            "selected_organs": [
                {"organ": "liver", "mask_path": str(liver), "grade": "A", "training_weight": 1.0},
                {"organ": "spleen", "mask_path": str(spleen), "grade": "C", "training_weight": 0.1, "scoring_schema_version": "autolabel_core_v2", "target_type": "hard"},
                {"organ": "pancreas", "mask_path": str(pancreas), "grade": "C", "training_weight": 0.1, "scoring_schema_version": "autolabel_core_v2", "target_type": "soft", "probability_mask_path": str(probability)},
            ],
        }), encoding="utf-8")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(case.parent, tmp_path / "manifest.json", case_list=case_list, require_images=True)

        positives = [item for item in manifest["items"] if item["supervision_type"] == "positive" and not item.get("is_prompt_variant")]
        assert [item["organ"] for item in positives] == ["pancreas"]
        assert positives[0]["target_type"] == "positive_soft"
        assert positives[0]["mask"] == str(probability)
        reasons = {row["organ"]: row["reason"] for row in manifest["skipped_ineligible_positive"]}
        assert reasons["liver"] == "legacy_requires_autolabel_core_v2_rescoring"
        assert reasons["spleen"] == "grade_C_requires_soft_probability_target"


    def test_negative_sampling_requires_safe_sources_and_ignores_student_empty_retry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDAI_NEGATIVE_PROMPT_RATIO", "10.0")
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        organs = [
            "prostate", "adrenal_left", "adrenal_right", "aorta", "gallbladder", "kidney_left",
            "kidney_right", "liver", "ovary_left", "pancreas", "spleen", "uterus",
        ]
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": organs,
            "organ_to_prompt": {organ: organ for organ in organs},
        }), encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case = tmp_path / "cases" / "case_001"
        upd = case / "updated"
        upd.mkdir(parents=True)
        mask = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), upd / "prostate.nii.gz")
        (case / "selection_metadata.json").write_text(json.dumps({
            "scan_coverage": "abdomen,pelvis",
            "selected_organs": [
                {"organ": "prostate", "mask_path": str(mask), "ct_path": str(ct), "grade": "A", "training_weight": 1.0, "scoring_schema_version": "autolabel_core_v2"},
                {"organ": "liver", "ct_path": str(ct), "grade": "D", "training_weight": 0.0, "quality_flags": ["missing_final_mask"]},
            ]
        }), encoding="utf-8")
        prev = tmp_path / "prev_student" / "case_001"
        prev.mkdir(parents=True)
        (prev / "voxtell_student_result.json").write_text(json.dumps({
            "per_organ_status": {"pancreas": {"status": "empty", "empty_mask": True}}
        }), encoding="utf-8")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(
            case.parent, tmp_path / "manifest.json", case_list=case_list,
            require_images=True, student_prediction_root=tmp_path / "prev_student",
        )

        counts = manifest["negative_source_counts"]
        assert counts["nonmedical_absent_object"] > 0
        assert counts["out_of_scan_anatomy_with_coverage_evidence"] > 0
        unsafe_sources = {
            "same_case_absent_organ",
            "anatomically_incompatible_organ",
            "low_confidence_teacher_rejected_organ",
            "student_empty_retry_organ",
        }
        negatives = [item for item in manifest["items"] if item.get("supervision_type") == "negative"]
        assert not any(item.get("negative_source") in unsafe_sources for item in negatives)
        assert not any(item.get("organ") == "pancreas" for item in negatives)
        out_of_scan = [
            item for item in negatives
            if item.get("negative_source") == "out_of_scan_anatomy_with_coverage_evidence"
        ]
        assert out_of_scan
        assert {item["prompt"] for item in out_of_scan} & {"segment the head", "segment the brain", "segment the skull"}
        assert all(item.get("negative_evidence", {}).get("type") == "scan_coverage_metadata" for item in out_of_scan)


class TestRunEMTrainingVoxTellMstep:
    def _load_run_em_training(self, monkeypatch, tmp_path):
        import importlib.util
        import sys

        monkeypatch.setenv("MEDAI_OUTPUT_ROOT", str(tmp_path / "outputs"))
        monkeypatch.setenv("MEDAI_ENABLE_VOXTELL_TRAINING", "0")
        spec = importlib.util.spec_from_file_location("run_em_training_test", Path("scripts/run_em_training.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules.pop("run_em_training_test", None)
        spec.loader.exec_module(module)
        return module

    def test_default_voxtell_train_cmd_points_to_project_trainer(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        assert "scripts/train_voxtell_prompt_student.py" in module.VOXTELL_TRAIN_CMD
        assert module.ENABLE_VOXTELL_TRAINING is False

    def test_formal_prompt_mstep_defaults_are_one_full_epoch(self, monkeypatch, tmp_path):
        for name in (
            "MEDAI_FINETUNE_EPOCHS",
            "MEDAI_MSTEP_LR",
            "MEDAI_MAX_STEPS",
            "MEDAI_TRAINABLE_SCOPE",
            "MEDAI_BCE_POS_WEIGHT_CAP",
            "MEDAI_SAVE_EVERY",
        ):
            monkeypatch.delenv(name, raising=False)
        module = self._load_run_em_training(monkeypatch, tmp_path)
        assert module.FINETUNE_EPOCHS == 1
        assert module.PROMPT_MAX_STEPS == 0
        assert module.LEARNING_RATE == 1e-6
        assert module.PROMPT_TRAINABLE_SCOPE == "prompt_path"
        assert module.PROMPT_BCE_POS_CAP == 20
        assert module.PROMPT_SAVE_EVERY == 1000

    def test_quality_gate_defaults_include_c_soft_labels(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEDAI_STUDENT_QUALITY_GRADES", raising=False)
        monkeypatch.delenv("MEDAI_STUDENT_QUALITY_MIN_WEIGHT", raising=False)
        module = self._load_run_em_training(monkeypatch, tmp_path)
        image = tmp_path / "ct.nii.gz"
        mask = tmp_path / "probability_mask.nii.gz"
        image.touch()
        mask.touch()
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [{
            "case_id": "case_001",
            "image": str(image),
            "mask": str(mask),
            "organ": "liver",
            "prompt": "segment the liver",
            "supervision_type": "positive",
            "training_weight": 0.1,
            "grade": "C",
            "is_prompt_variant": False,
        }]}), encoding="utf-8")
        monkeypatch.setattr(module, "_mask_nonempty", lambda path: True)

        selected = module._select_quality_gate_items(manifest, max_cases=3, max_prompts=8)

        assert list(selected) == ["case_001"]
        assert selected["case_001"][0]["grade"] == "C"

    def test_quality_gate_prefers_core_anatomy_over_alphabetical_composites(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        image = tmp_path / "ct.nii.gz"
        mask = tmp_path / "mask.nii.gz"
        image.touch()
        mask.touch()
        manifest = tmp_path / "manifest.json"
        common = {
            "case_id": "case_001",
            "image": str(image),
            "mask": str(mask),
            "supervision_type": "positive",
            "training_weight": 0.1,
            "grade": "C",
            "is_prompt_variant": False,
        }
        manifest.write_text(json.dumps({"items": [
            {**common, "organ": "abdominal_cavity", "prompt": "abdominal cavity"},
            {**common, "organ": "liver", "prompt": "liver"},
        ]}), encoding="utf-8")
        monkeypatch.setattr(module, "_mask_nonempty", lambda path: True)

        selected = module._select_quality_gate_items(manifest, max_cases=1, max_prompts=1)

        assert selected["case_001"][0]["organ"] == "liver"

    def test_dice_uses_half_probability_threshold_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEDAI_DSC_MASK_THRESHOLD", raising=False)
        module = self._load_run_em_training(monkeypatch, tmp_path)
        pred = _make_nii(np.ones((4, 4, 4), dtype=np.float32), tmp_path / "pred.nii.gz")
        ref_data = np.full((4, 4, 4), 0.1, dtype=np.float32)
        ref = _make_nii(ref_data, tmp_path / "ref.nii.gz")

        dsc, pred_nonempty, ref_nonempty, reason = module._dice_for_masks(pred, ref)

        assert dsc == 0.0
        assert pred_nonempty is True
        assert ref_nonempty is False
        assert reason is None

    def test_round_teacher_cache_uses_hierarchical_predictions_only(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        monkeypatch.setattr(module, "ALL_TEACHERS", ["teacher_a", "teacher_b"])
        monkeypatch.setattr(module, "TEACHER_INFERENCE_MODE", "hierarchical_roi")
        case_root = tmp_path / "outputs" / "round1" / "estep" / "cases" / "case_001"
        case_root.mkdir(parents=True)
        (case_root / "hierarchical_inference_plan.json").write_text(json.dumps({
            "teacher_inference_mode": "hierarchical_roi",
            "hierarchical_plan_cache_key": {"requested_organs": ["liver_segment_1"]},
            "hierarchical_plan_cache_key_sha256": "fake",
        }), encoding="utf-8")
        hier_seg = case_root / "hierarchical_predictions" / "teacher_a" / "segmentations"
        hier_seg.mkdir(parents=True)
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), hier_seg / "liver_segment_1.nii.gz")
        legacy_seg = case_root / "raw_predictions" / "teacher_b" / "case_001" / "segmentations"
        legacy_seg.mkdir(parents=True)
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), legacy_seg / "liver_segment_1.nii.gz")

        cache = module._round_teacher_cache_dirs(1)

        assert cache == {
            "teacher_a": tmp_path / "outputs" / "round1" / "estep" / "cases" / "{case_id}" / "hierarchical_predictions" / "teacher_a" / "segmentations"
        }

    def test_completed_cases_require_hierarchical_manifest_in_hierarchical_mode(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        monkeypatch.setattr(module, "TEACHER_INFERENCE_MODE", "hierarchical_roi")
        updated = tmp_path / "outputs" / "round1" / "estep" / "annotation_versions" / "case_001" / "updated"
        updated.mkdir(parents=True)
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), updated / "liver.nii.gz")
        (updated.parent / "selection_metadata.json").write_text("{}", encoding="utf-8")

        assert module.completed_cases(1) == set()

        manifest = tmp_path / "outputs" / "round1" / "estep" / "cases" / "case_001" / "hierarchical_inference_plan.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({
            "teacher_inference_mode": "hierarchical_roi",
            "hierarchical_plan_cache_key": {"requested_organs": ["liver"]},
            "hierarchical_plan_cache_key_sha256": "fake",
        }), encoding="utf-8")
        assert module.completed_cases(1) == {"case_001"}

    def test_disabled_voxtell_training_keeps_manifest_but_no_checkpoint(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "num_items": 1,
            "num_cases": 1,
            "num_distillation_eligible_items": 1,
            "items": [{"training_weight": 1.0, "distillation_eligible": True}],
        }), encoding="utf-8")

        result = module.run_prompt_student_mstep(1, manifest)

        assert result["status"] == "manifest_ready"
        assert result["training_status"] == "manifest_ready_manifest_only"
        assert result["training_mode"] == "manifest_only"
        assert result["num_distillation_eligible_items"] == 1
        assert result["finetuned_checkpoint"] is None
        assert "train_cmd" in result



    def test_voxtell_sanity_check_runs_five_prompt_contract(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        model_dir = tmp_path / "outputs" / "round1" / "mstep" / "voxtell_finetuned_model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        organs = ["liver", "spleen", "pancreas", "kidney_left", "aorta"]
        target_config.write_text(json.dumps({"target_organs": organs, "organ_to_prompt": {o: o for o in organs}}), encoding="utf-8")
        monkeypatch.setattr(module, "CASE_LIST", case_list)
        monkeypatch.setattr(module, "PROMPT_TARGET_CONFIG", target_config)
        monkeypatch.setattr(module, "load_student_target_organs", lambda: organs)

        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(self, ct_image, output_dir, prompts=None, **kwargs):
            output_dir = Path(output_dir)
            official = {}
            for organ in prompts:
                off = output_dir / f"ct_{organ}.nii.gz"
                official[organ] = str(off)
                _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), off)
                _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), output_dir / f"{organ}.nii.gz")
            return {"status": "success", "official_output_masks": official}

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.run_voxtell_student_sanity_check(1, model_dir)

        assert result["status"] == "failed"
        assert result["prompts"] == organs
        assert result["official_outputs_ok"] is True
        assert result["project_outputs_ok"] is True
        assert result["empty_mask_ratio"] == 1.0
        assert result["positive_nonempty_ok"] is False


    def test_voxtell_sanity_check_requires_some_nonempty_positive_output(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        model_dir = tmp_path / "outputs" / "round1" / "mstep" / "voxtell_finetuned_model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")
        organs = ["liver", "spleen", "pancreas", "kidney_left", "aorta"]
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({"target_organs": organs, "organ_to_prompt": {o: o for o in organs}}), encoding="utf-8")
        monkeypatch.setattr(module, "CASE_LIST", case_list)
        monkeypatch.setattr(module, "PROMPT_TARGET_CONFIG", target_config)
        monkeypatch.setattr(module, "load_student_target_organs", lambda: organs)

        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(self, ct_image, output_dir, prompts=None, **kwargs):
            output_dir = Path(output_dir)
            official = {}
            for idx, organ in enumerate(prompts):
                arr = np.zeros((8, 8, 8), dtype=np.uint8)
                if idx == 0:
                    arr[2:4, 2:4, 2:4] = 1
                off = output_dir / f"ct_{organ}.nii.gz"
                official[organ] = str(off)
                _make_nii(arr, off)
                _make_nii(arr, output_dir / f"{organ}.nii.gz")
            return {"status": "success", "official_output_masks": official}

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.run_voxtell_student_sanity_check(1, model_dir)
        assert result["status"] == "success"
        assert result["empty_mask_ratio"] == 0.8
        assert result["positive_nonempty_ok"] is True

    def test_sanity_selects_manifest_absent_negative_semantic_sentinels(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        zero = _make_nii(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "zero.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [
            {
                "case_id": "case_001", "organ": organ, "mask": str(zero),
                "supervision_type": "negative", "target_type": "absent_negative",
                "is_prompt_variant": False,
            }
            for organ in ["oral_cavity", "brain_ventricle", "cerebrospinal_fluid", "other_absent"]
        ]}), encoding="utf-8")

        selected = module._manifest_absent_negative_organs(
            manifest,
            "case_001",
            ["brain_ventricle", "cerebrospinal_fluid", "oral_cavity"],
            limit=3,
        )

        assert selected == ["brain_ventricle", "cerebrospinal_fluid", "oral_cavity"]

    def test_voxtell_sanity_fails_nonempty_absent_negative(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        model_dir = tmp_path / "outputs" / "round1" / "mstep" / "voxtell_finetuned_model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        zero = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "zero.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")
        positives = ["liver", "spleen", "pancreas", "kidney_left", "aorta"]
        negatives = ["brain_ventricle", "cerebrospinal_fluid", "oral_cavity"]
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": positives + negatives,
            "organ_to_prompt": {o: o for o in positives + negatives},
        }), encoding="utf-8")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [
            {
                "case_id": "case_001", "organ": organ, "mask": str(zero),
                "supervision_type": "negative", "target_type": "absent_negative",
                "is_prompt_variant": False,
            }
            for organ in negatives
        ]}), encoding="utf-8")
        monkeypatch.setattr(module, "CASE_LIST", case_list)
        monkeypatch.setattr(module, "PROMPT_TARGET_CONFIG", target_config)
        monkeypatch.setattr(module, "load_student_target_organs", lambda: positives + negatives)

        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(self, ct_image, output_dir, prompts=None, **kwargs):
            output_dir = Path(output_dir)
            official = {}
            for organ in prompts:
                arr = np.zeros((8, 8, 8), dtype=np.uint8)
                if organ in {"liver", "brain_ventricle"}:
                    arr[2:4, 2:4, 2:4] = 1
                off = output_dir / f"ct_{organ}.nii.gz"
                official[organ] = str(off)
                _make_nii(arr, off)
                _make_nii(arr, output_dir / f"{organ}.nii.gz")
            return {"status": "partial_success", "official_output_masks": official}

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.run_voxtell_student_sanity_check(1, model_dir, manifest)

        assert result["status"] == "failed"
        assert result["absent_negative_ok"] is False
        assert result["nonempty_absent_negative_predictions"] == ["brain_ventricle"]

    def test_quality_gate_fails_low_dsc_and_recall(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        model_dir = tmp_path / "outputs" / "round1" / "mstep" / "voxtell_finetuned_model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        ref = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "liver.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [{
            "case_id": "case_001", "image": str(ct), "mask": str(ref), "organ": "liver",
            "prompt": "liver", "supervision_type": "positive", "training_weight": 1.0,
            "grade": "A", "is_prompt_variant": False,
        }]}), encoding="utf-8")
        organs = ["liver"]
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({"target_organs": organs, "organ_to_prompt": {"liver": "liver"}}), encoding="utf-8")
        monkeypatch.setattr(module, "PROMPT_TARGET_CONFIG", target_config)

        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(self, ct_image, output_dir, prompts=None, **kwargs):
            output_dir = Path(output_dir)
            official = {}
            for organ in prompts:
                arr = np.zeros((8, 8, 8), dtype=np.uint8)
                off = output_dir / f"ct_{organ}.nii.gz"
                official[organ] = str(off)
                _make_nii(arr, off)
                _make_nii(arr, output_dir / f"{organ}.nii.gz")
            return {"status": "success", "official_output_masks": official}

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.run_voxtell_student_quality_gate(1, model_dir, manifest)
        assert result["status"] == "failed"
        assert result["mean_dsc"] == 0.0
        assert result["nonempty_recall"] == 0.0


    def test_quality_gate_reports_median_empty_rate_and_group_summary(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        model_dir = tmp_path / "outputs" / "round1" / "mstep" / "voxtell_finetuned_model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        liver = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "liver.nii.gz")
        aorta = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=1), tmp_path / "aorta.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [
            {"case_id": "case_001", "image": str(ct), "mask": str(liver), "organ": "liver", "prompt": "liver", "supervision_type": "positive", "training_weight": 1.0, "grade": "A", "is_prompt_variant": False},
            {"case_id": "case_001", "image": str(ct), "mask": str(aorta), "organ": "aorta", "prompt": "aorta", "supervision_type": "positive", "training_weight": 1.0, "grade": "B", "is_prompt_variant": False},
            {"case_id": "case_001", "image": str(ct), "mask": str(aorta), "organ": "veins", "prompt": "veins", "supervision_type": "positive", "training_weight": 1.0, "grade": "C", "is_prompt_variant": False},
        ]}), encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({"target_organs": ["liver", "aorta", "veins"], "organ_to_prompt": {"liver": "liver", "aorta": "aorta", "veins": "veins"}}), encoding="utf-8")
        monkeypatch.setattr(module, "PROMPT_TARGET_CONFIG", target_config)
        monkeypatch.setenv("MEDAI_STUDENT_QUALITY_GRADES", "A,B")

        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(self, ct_image, output_dir, prompts=None, **kwargs):
            output_dir = Path(output_dir)
            official = {}
            for organ in prompts:
                arr = np.zeros((8, 8, 8), dtype=np.uint8)
                if organ == "liver":
                    arr = _sphere_mask(shape=(8, 8, 8), radius=2)
                off = output_dir / f"ct_{organ}.nii.gz"
                official[organ] = str(off)
                _make_nii(arr, off)
                _make_nii(arr, output_dir / f"{organ}.nii.gz")
            return {"status": "partial_success", "official_output_masks": official}

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.run_voxtell_student_quality_gate(1, model_dir, manifest)
        assert result["num_evaluations"] == 2
        assert result["median_dsc"] == 0.5
        assert result["empty_failure_count"] == 1
        assert result["empty_failure_rate"] == 0.5
        assert "large_or_common_organ" in result["organ_group_summary"]
        assert "vessel" in result["organ_group_summary"]
        assert result["top10_organs"][0]["organ"] == "liver"
        assert result["bottom10_organs"][0]["organ"] == "aorta"

    def test_prompt_mstep_keeps_capped_pilot_ineligible(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "num_items": 10,
            "num_cases": 1,
            "num_distillation_eligible_items": 10,
            "items": [{"training_weight": 1.0, "distillation_eligible": True}],
        }), encoding="utf-8")
        monkeypatch.setattr(module, "ENABLE_VOXTELL_TRAINING", True)
        monkeypatch.setattr(module, "VOXTELL_TRAIN_CMD", "fake-train")
        monkeypatch.setattr(module.sys, "argv", ["run_em_training.py", "--voxtell-mstep-mode", "project_distillation_experimental"])
        monkeypatch.setattr(module, "stop_vllm_for_mstep", lambda: None)
        monkeypatch.setattr(module, "restart_vllm_after_mstep", lambda: None)
        monkeypatch.setattr(module, "run_voxtell_student_sanity_check", lambda *args, **kwargs: {"status": "success"})
        monkeypatch.setattr(module, "run_voxtell_student_quality_gate", lambda *args, **kwargs: {"status": "success", "mean_dsc": 1.0, "nonempty_recall": 1.0})

        out_dir = tmp_path / "outputs" / "round1" / "mstep"
        def fake_run(*args, **kwargs):
            model_dir = out_dir / "voxtell_finetuned_model"
            (model_dir / "fold_0").mkdir(parents=True)
            (model_dir / "plans.json").write_text("{}", encoding="utf-8")
            (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
            (out_dir / "voxtell_prompt_train_result.json").write_text(json.dumps({
                "status": "success",
                "num_manifest_items": 2,
                "max_steps": 2,
                "inference_model_dir": str(model_dir),
                "finetuned_checkpoint": str(out_dir / "model_finetune.pth"),
            }), encoding="utf-8")
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        result = module.run_prompt_student_mstep(1, manifest)
        assert result["status"] == "success"
        assert result["training_status"] == "completed_pilot_quality_gated"
        assert result["pilot_short_training"] is True
        assert result["checkpoint_eligible_for_next_round"] is False
        assert "full M-step" in result["formal_round2_recommendation"]

    def test_save_prompt_student_predictions_skips_without_eligible_checkpoint(self, monkeypatch, tmp_path):
        module = self._load_run_em_training(monkeypatch, tmp_path)
        called = {"segment": False}
        from cli_anything.medai.core import voxtell_student as vs

        def fake_segment(*args, **kwargs):
            called["segment"] = True
            raise AssertionError("base fallback should not run")

        monkeypatch.setattr(vs.VoxTellStudent, "segment", fake_segment)
        result = module.save_prompt_student_predictions(1)
        assert result["status"] == "skipped_no_eligible_checkpoint"
        assert result["checkpoint_eligible_for_next_round"] is False
        assert called["segment"] is False


class TestVoxTellTrainerContracts:
    def test_all_zero_multiscale_loss_is_finite_and_has_gradients(self):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_loss_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        outputs = [
            torch.zeros((1, 1, 8, 8, 8), requires_grad=True),
            torch.zeros((1, 1, 4, 4, 4), requires_grad=True),
            torch.zeros((1, 1, 2, 2, 2), requires_grad=True),
        ]
        target = torch.zeros((1, 1, 8, 8, 8))
        loss = module.voxtell_supervision_loss(outputs, target, 1.0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in outputs)

    def test_small_target_nearest_downsample_is_binary(self):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_resize_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        target = torch.zeros((1, 1, 8, 8, 8))
        target[:, :, 3:5, 3:5, 3:5] = 1
        resized = module._resize_target_like(target, torch.zeros((1, 1, 4, 4, 4)))
        assert set(torch.unique(resized).tolist()) <= {0.0, 1.0}
        assert resized.shape == (1, 1, 4, 4, 4)

    def test_foreground_balanced_bce_prevents_sparse_target_background_dominance(self):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
        target[..., 0, 0, 0] = 1
        logits = torch.zeros_like(target)
        unweighted = module._foreground_balanced_bce(logits, target, 1.0)
        balanced = module._foreground_balanced_bce(logits, target, 20.0)
        assert balanced > unweighted

    def test_supervision_loss_handles_multiscale_and_empty_target(self):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
        outputs = [
            torch.randn((1, 1, 8, 8, 8), requires_grad=True),
            torch.randn((1, 1, 4, 4, 4), requires_grad=True),
        ]
        loss = module.voxtell_supervision_loss(outputs, target, 0.5)
        assert torch.isfinite(loss)
        loss.backward()
        assert outputs[0].grad is not None


    def test_load_manifest_expands_legacy_prompt_variants_and_weights_sampler(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        image = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask = _make_nii(np.ones((4, 4, 4), dtype=np.uint8), tmp_path / "liver.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [{
            "case_id": "case_001",
            "image": str(image),
            "mask": str(mask),
            "organ": "liver",
            "prompt": "liver",
            "prompt_variants": ["liver", "hepatic organ", "segment the liver"],
            "grade": "A",
            "training_weight": 1.0,
            "scoring_schema_version": "autolabel_core_v2",
            "target_type": "hard",
            "route_confidence": 1.0,
            "label_confidence": 1.0,
            "teacher_lineage": ["teacher_a", "teacher_b"],
            "student_training_priority": "A",
        }]}, indent=2), encoding="utf-8")

        rows = module.load_manifest(manifest)
        prompts = {row["prompt"] for row in rows}
        assert prompts == {"liver"}
        canonical = rows[0]
        assert canonical["is_prompt_variant"] is False
        assert canonical["canonical_prompt"] == "liver"
        assert canonical["sampling_weight"] > 1.0
        assert canonical["effective_loss_weight"] == canonical["sampling_weight"]
        assert canonical["sampling_repeat"] >= 1
        assert module.compute_sampling_weight({"training_weight": 1.0, "grade": "A"}) > module.compute_sampling_weight({"training_weight": 1.0, "grade": "D"})

    def test_load_manifest_skips_unsafe_legacy_negative_sources(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        image = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        zero = _make_nii(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "zero.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [
            {
                "case_id": "case_001",
                "image": str(image),
                "mask": str(zero),
                "organ": "pancreas",
                "prompt": "pancreas",
                "supervision_type": "negative",
                "negative_source": "student_empty_retry_organ",
                "training_weight": 0.1,
            },
            {
                "case_id": "case_001",
                "image": str(image),
                "mask": str(zero),
                "organ": "negative_nonmedical_cat",
                "prompt": "segment the cat",
                "supervision_type": "negative",
                "negative_source": "nonmedical_absent_object",
                "zero_mask_role": "negative_target_mask",
                "training_weight": 0.1,
            },
        ]}, indent=2), encoding="utf-8")

        rows = module.load_manifest(manifest)

        assert rows
        assert {row["organ"] for row in rows} == {"negative_nonmedical_cat"}
        assert all(row["negative_source"] == "nonmedical_absent_object" for row in rows)

    def test_runtime_sampler_ratio_and_negative_reason_contracts(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        assert module.parse_pos_neg_ratio("2:1") == (2, 1)
        assert module.parse_pos_neg_ratio("1:1") == (1, 1)
        assert module.parse_pos_neg_ratio("10:1") == (10, 1)
        pools = module.build_sample_pools([
            {"case_id": "c", "organ": "liver", "supervision_type": "positive", "training_weight": 1.0},
            {"case_id": "c", "organ": "brain", "supervision_type": "negative", "training_weight": 0.1, "negative_source": "case_373_expected_absent"},
        ])
        kinds = [module.sample_training_item(i, pools, module.parse_pos_neg_ratio("2:1"))["runtime_sample_kind"] for i in range(30)]
        assert kinds.count("positive") == 20
        assert kinds.count("negative") == 10
        neg = module.sample_training_item(2, pools, module.parse_pos_neg_ratio("2:1"))
        assert neg["negative_reason"] in {"absent_in_scan", "absent_in_crop"}
        negatives = [
            module.sample_training_item(
                i,
                pools,
                module.parse_pos_neg_ratio("2:1"),
                module.parse_pos_neg_ratio("1:1"),
            )
            for i in range(30)
            if module.choose_sample_kind(i, module.parse_pos_neg_ratio("2:1"), pools) == "negative"
        ]
        assert [row["negative_source_class"] for row in negatives].count("semantic_negative") == 5
        assert [row["negative_source_class"] for row in negatives].count("derived_crop_negative") == 5

    def test_official_retention_loss_has_zero_gradient_at_official_prediction(self):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        official = torch.tensor([[[[[2.0, -2.0]]]]])
        student = official.clone().requires_grad_(True)
        loss = module.official_retention_loss(student, official)
        loss.backward()

        assert torch.isfinite(loss)
        assert float(student.grad.abs().max()) < 1e-6

    def test_load_manifest_accepts_absent_negative_zero_mask_role(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        image = _make_nii(np.ones((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        zero = _make_nii(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "zero.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [{
            "case_id": "case_001",
            "image": str(image),
            "mask": str(zero),
            "organ": "brain",
            "prompt": "brain",
            "supervision_type": "negative",
            "target_type": "absent_negative",
            "negative_source": "case_373_expected_absent",
            "zero_mask_role": "absent_negative_target_mask",
            "training_weight": 0.1,
        }]}, indent=2), encoding="utf-8")

        rows = module.load_manifest(manifest)

        assert len(rows) == 1
        assert rows[0]["negative_reason"] == "absent_in_scan"

    def test_runtime_patch_sampler_positive_and_absent_in_crop_negative(self, tmp_path):
        import importlib.util
        import random

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        random.seed(7)
        image = _make_nii(np.ones((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask_arr = np.zeros((8, 8, 8), dtype=np.uint8)
        mask_arr[0, 0, 0] = 1
        mask = _make_nii(mask_arr, tmp_path / "tiny_organ.nii.gz")
        base = {"case_id": "case_001", "image": str(image), "mask": str(mask), "organ": "tiny", "prompt": "tiny", "target_type": "hard"}

        pos_image, pos_target, pos_meta = module.load_training_patch_with_metadata({**base, "runtime_sample_kind": "positive"}, (4, 4, 4), 0.0)
        neg_image, neg_target, neg_meta = module.load_training_patch_with_metadata({**base, "runtime_sample_kind": "negative", "derived_negative_from_positive": True, "negative_reason": "absent_in_crop"}, (4, 4, 4), 0.0)

        assert pos_image.shape == neg_image.shape == (1, 1, 4, 4, 4)
        assert float(pos_target.sum()) > 0.0
        assert pos_meta["sample_kind"] == "positive"
        assert pos_meta["case_id"] == "case_001"
        assert pos_meta["organ"] == "tiny"
        assert pos_meta["prompt"] == "tiny"
        assert float(neg_target.sum()) == 0.0
        assert neg_meta["sample_kind"] == "negative"
        assert neg_meta["negative_reason"] == "absent_in_crop"
        assert neg_meta["all_zero_target"] is True

    def test_dry_run_reports_candidate_pools_and_sampling_config(self, tmp_path, monkeypatch):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        image = _make_nii(np.ones((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask = _make_nii(np.ones((4, 4, 4), dtype=np.uint8), tmp_path / "liver.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [{
            "case_id": "case_001", "image": str(image), "mask": str(mask), "organ": "liver", "prompt": "liver",
            "grade": "A", "training_weight": 1.0, "scoring_schema_version": "autolabel_core_v2", "target_type": "hard",
        }]}, indent=2), encoding="utf-8")
        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "train_voxtell_prompt_student.py", "--manifest", str(manifest), "--model-dir", str(model_dir),
            "--output-dir", str(out), "--dry-run", "--device", "cpu", "--pos-neg-ratio", "2:1",
        ])

        assert module.main() == 0
        result = json.loads((out / "voxtell_prompt_train_result.json").read_text(encoding="utf-8"))
        assert result["training_profile"] == "paper_aligned"
        assert result["negative_prompt_sampling"] == "per_image_2_positive_1_volume_absent_negative"
        assert result["pos_neg_ratio_parsed"] == [2, 1]
        assert result["candidate_pool_positive_count"] == 1
        assert result["candidate_pool_derived_crop_negative_count"] == 0
        assert "num_positive_items" not in result

    def test_round1_negative_job_sampler_skips_unsafe_sources(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("round1_patch_experiments_test", Path("scripts/run_round1_10h_patch_experiments.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        image = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": [
            {
                "case_id": "case_001",
                "image": str(image),
                "organ": "pancreas",
                "prompt": "pancreas",
                "supervision_type": "negative",
                "negative_source": "same_case_absent_organ",
                "zero_mask_role": "negative_target_mask",
            },
            {
                "case_id": "case_001",
                "image": str(image),
                "organ": "negative_nonmedical_cat",
                "prompt": "segment the cat",
                "supervision_type": "negative",
                "negative_source": "nonmedical_absent_object",
                "zero_mask_role": "negative_target_mask",
            },
        ]}, indent=2), encoding="utf-8")

        jobs, audit = module.select_negative_jobs(manifest, max_cases=1, max_per_source=3)

        assert [job["organ"] for job in jobs] == ["negative_nonmedical_cat"]
        assert "same_case_absent_organ" in audit["skipped_unsafe_sources"]

    def test_poly_lr_and_optimizer_defaults(self):
        import importlib.util
        import argparse
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        p = torch.nn.Parameter(torch.ones(1))
        args = argparse.Namespace(optimizer="sgd", learning_rate=1e-4, weight_decay=3e-5)
        optim = module.build_optimizer([p], args)
        assert isinstance(optim, torch.optim.SGD)
        assert module.poly_lr(5, 10, 1e-4, 0.9) < 1e-4

    def test_prompt_embedding_cache_is_bound_to_text_model_and_prompt_hash(self, monkeypatch, tmp_path):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        prompts = ["segment the liver"]
        cache_path = tmp_path / "prompt_embeddings.pt"
        torch.save({
            "metadata": module._text_encoder_cache_meta(prompts, "qwen-a"),
            "embeddings": {prompts[0]: torch.ones((1, 1, 4))},
        }, cache_path)

        calls = {"model_loads": 0}

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                }

        class FakeBackbone:
            def eval(self):
                return self

            def to(self, device):
                return self

            def requires_grad_(self, value):
                self.frozen = value is False
                return self

            def __call__(self, **tokens):
                assert getattr(self, "frozen", False) is True
                return type("Encoded", (), {
                    "last_hidden_state": torch.zeros((1, 2, 4), dtype=torch.float32)
                })()

        def fake_tokenizer_from_pretrained(*args, **kwargs):
            return FakeTokenizer()

        def fake_model_from_pretrained(*args, **kwargs):
            calls["model_loads"] += 1
            return FakeBackbone()

        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
        monkeypatch.setattr(module.AutoModel, "from_pretrained", fake_model_from_pretrained)

        reused = module.build_prompt_embeddings(prompts, "qwen-a", torch.device("cpu"), cache_path)
        assert calls["model_loads"] == 0
        assert torch.equal(reused[prompts[0]], torch.ones((1, 1, 4)))

        rebuilt = module.build_prompt_embeddings(prompts, "qwen-b", torch.device("cpu"), cache_path)
        assert calls["model_loads"] == 1
        assert torch.equal(rebuilt[prompts[0]], torch.zeros((1, 1, 4)))

        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert saved["metadata"]["text_model_name"] == "qwen-b"
        assert saved["metadata"]["prompt_hash"] == module._text_encoder_cache_meta(prompts, "qwen-b")["prompt_hash"]

    def test_text_encoder_policy_records_qwen_as_frozen(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        policy = module.TEXT_ENCODER_POLICY
        assert policy["trainability"] == "frozen"
        assert "requires_grad=False" in policy["implementation"]
        assert "last_token_pool" in policy["token_flow"]
        assert "cross" in policy["fusion"].lower()

    def test_paper_aligned_case_pool_and_unit_are_strict_two_plus_one(self, tmp_path):
        import importlib.util
        import random

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        image = _make_nii(np.ones((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask_a = _make_nii(np.ones((8, 8, 8), dtype=np.uint8), tmp_path / "a.nii.gz")
        mask_b = _make_nii(np.ones((8, 8, 8), dtype=np.uint8), tmp_path / "b.nii.gz")
        zero = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "zero.nii.gz")
        rows = [
            {"case_id": "c", "image": str(image), "mask": str(mask_a), "organ": "a", "prompt": "a", "canonical_prompt": "a", "supervision_type": "positive"},
            {"case_id": "c", "image": str(image), "mask": str(mask_b), "organ": "b", "prompt": "b", "canonical_prompt": "b", "supervision_type": "positive"},
            {"case_id": "c", "image": str(image), "mask": str(zero), "organ": "brain", "prompt": "brain", "canonical_prompt": "brain", "supervision_type": "negative"},
        ]
        pools = module.build_paper_case_pools(rows)
        random.seed(1)
        image_t, target_t, prompts, meta = module.load_paper_training_unit(pools["c"], (4, 4, 4), 0.85)
        assert image_t.shape == (1, 4, 4, 4)
        assert target_t.shape == (3, 4, 4, 4)
        assert len(prompts) == 3
        assert float(target_t[0].sum()) > 0 and float(target_t[1].sum()) > 0
        assert float(target_t[2].sum()) == 0
        assert meta["sample_kind"] == "paper_aligned_2_positive_1_negative"

    def test_official_embedding_bank_is_used_without_loading_qwen(self, tmp_path, monkeypatch):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_test", Path("scripts/train_voxtell_prompt_student.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        bank_path = tmp_path / "bank.npz"
        np.savez(bank_path, labels=np.asarray(["liver"]), embeddings=np.ones((1, 2560), dtype=np.float16))
        bank, audit = module.load_official_embedding_bank(bank_path)
        monkeypatch.setattr(module.AutoModel, "from_pretrained", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Qwen must not load")))
        embeddings = module.build_prompt_embeddings(["liver"], "unused", torch.device("cpu"), None, bank)
        assert audit["embedding_shape"] == [1, 2560]
        assert embeddings["liver"].shape == (1, 1, 2560)


class TestBDMAPPanTSLayout:
    def test_pants_import_files_writes_standard_case_layout_and_mapping(self, tmp_path):
        from cli_anything.medai.core.pants_utils import import_pants_files

        ct = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "source_ct.nii.gz")
        labels = tmp_path / "source_labels"
        labels.mkdir()
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), labels / "pancreas.nii.gz")

        result = import_pants_files(ct, labels, tmp_path / "dataset", patient_id="case_001")

        case = tmp_path / "dataset" / "case_001"
        assert result["status"] == "success"
        assert (case / "image.nii.gz").exists()
        assert (case / "segmentations" / "pancreas.nii.gz").exists()
        mapping = json.loads((case / "label_mapping.json").read_text(encoding="utf-8"))
        assert mapping["layout"] == "bdmap_pants_style_binary_masks"
        assert mapping["mappings"][0]["teacher_output_name"] == "pancreas"
        assert mapping["mappings"][0]["canonical_organ_name"] == "pancreas"
        assert mapping["mappings"][0]["student_target_id"] is None

    def test_voxtell_manifest_reads_standard_case_layout_and_three_layer_mapping(self, tmp_path):
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": ["pancreas"],
            "organ_to_prompt": {"pancreas": "segment the pancreas"},
            "organ_to_student_id": {"pancreas": 7},
        }), encoding="utf-8")
        case = tmp_path / "standard_dataset" / "case_001"
        (case / "segmentations").mkdir(parents=True)
        _make_nii(np.zeros((4, 4, 4), dtype=np.int16), case / "image.nii.gz")
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), case / "segmentations" / "pancreas.nii.gz")
        (case / "label_mapping.json").write_text(json.dumps({
            "layout": "bdmap_pants_style_binary_masks",
            "mappings": [{
                "teacher_target_id": 3,
                "teacher_output_name": "pancreas_raw",
                "teacher_output_file": "pancreas_raw.nii.gz",
                "canonical_organ_name": "pancreas",
                "student_target_id": 7,
            }],
        }), encoding="utf-8")
        (case / "selection_metadata.json").write_text(json.dumps({"selected_organs": [{
            "organ": "pancreas", "identity_status": "valid",
            "requested_canonical_id": "pancreas", "resolved_canonical_id": "pancreas",
            "grade": "B", "training_weight": 0.5,
            "scoring_schema_version": "autolabel_core_v2",
        }]}), encoding="utf-8")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(case.parent, tmp_path / "manifest.json", require_images=True)

        item = next(row for row in manifest["items"] if row["supervision_type"] == "positive")
        assert item["image"].endswith("image.nii.gz")
        assert item["mask"].endswith("segmentations/pancreas.nii.gz")
        assert item["teacher_target_id"] == 3
        assert item["teacher_output_name"] == "pancreas_raw"
        assert item["canonical_organ_name"] == "pancreas"
        assert item["student_target_id"] == 7
        assert "teacher IDs are never reused as student IDs" in item["target_mapping_policy"]

    def test_voxtell_manifest_does_not_treat_image_file_as_direct_mask_folder(self, tmp_path):
        from cli_anything.medai.core.voxtell_student import VoxTellStudent

        model_dir = tmp_path / "model"
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_text("fake", encoding="utf-8")
        target_config = tmp_path / "targets.json"
        target_config.write_text(json.dumps({
            "target_organs": ["image"],
            "organ_to_prompt": {"image": "segment the image"},
            "organ_to_student_id": {"image": 1},
        }), encoding="utf-8")
        case = tmp_path / "standard_dataset" / "case_001"
        case.mkdir(parents=True)
        _make_nii(np.zeros((4, 4, 4), dtype=np.int16), case / "image.nii.gz")

        student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device="cpu")
        manifest = student.build_training_manifest(case.parent, tmp_path / "manifest.json", require_images=True)

        assert manifest["num_items"] == 0

    def test_standard_dataset_export_metadata_points_to_standard_paths(self, tmp_path):
        from cli_anything.medai.core.multimodel_loop import _export_standard_case_dataset

        ct = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        old_mask_path = tmp_path / "annotation_versions" / "case_001" / "updated" / "pancreas.nii.gz"
        old_mask_path.parent.mkdir(parents=True)
        old_mask = _make_nii(np.ones((4, 4, 4), dtype=np.uint8), old_mask_path)

        result = _export_standard_case_dataset(
            case_id="case_001",
            ct=ct,
            selected_metadata=[{"organ": "pancreas", "final_mask": str(old_mask), "selected_model": "teacher_a"}],
            out_root=tmp_path / "standard_dataset",
            student_target_ids={"pancreas": 7},
        )

        meta = json.loads((Path(result["case_folder"]) / "selection_metadata.json").read_text(encoding="utf-8"))
        exported = meta["selected_organs"][0]
        assert exported["ct_path"].endswith("standard_dataset/case_001/image.nii.gz")
        assert exported["mask_path"].endswith("standard_dataset/case_001/segmentations/pancreas.nii.gz")
        assert exported["student_target_id"] == 7

    def test_standard_dataset_export_excludes_d_grade_positive_masks(self, tmp_path):
        from cli_anything.medai.core.multimodel_loop import _export_standard_case_dataset

        ct = _make_nii(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask_path = tmp_path / "annotation_versions" / "case_001" / "updated" / "pancreatic_duct.nii.gz"
        mask_path.parent.mkdir(parents=True)
        mask = _make_nii(np.ones((4, 4, 4), dtype=np.uint8), mask_path)

        result = _export_standard_case_dataset(
            case_id="case_001",
            ct=ct,
            selected_metadata=[{
                "organ": "pancreatic_duct",
                "final_mask": str(mask),
                "selected_model": "teacher_a",
                "grade": "D",
                "training_weight": 0.0,
                "target_type": "hard",
            }],
            out_root=tmp_path / "standard_dataset",
            student_target_ids={"pancreatic_duct": 9},
        )

        case_folder = Path(result["case_folder"])
        assert not (case_folder / "segmentations" / "pancreatic_duct.nii.gz").exists()
        assert result["num_masks"] == 0
        meta = json.loads((case_folder / "selection_metadata.json").read_text(encoding="utf-8"))
        exported = meta["selected_organs"][0]
        assert exported["distillation_eligible"] is False
        assert exported["distillation_exclusion_reason"] == "grade_D_or_zero_weight"
        assert exported["standard_dataset_export_status"] == "metadata_only_excluded_from_positive_masks"


class TestRound1RepairAuditAndMstepContracts:
    def test_mstep_manifest_excludes_d_grade_zero_weight_and_ineligible_items(self, tmp_path):
        from cli_anything.medai.core.mstep_runner import build_training_manifest

        root = tmp_path / "annotation_versions"
        case = root / "case_001"
        updated = case / "updated"
        updated.mkdir(parents=True)
        for organ in ["liver", "pancreas", "kidney_left", "kidney_right"]:
            _make_nii(np.ones((4, 4, 4), dtype=np.uint8), updated / f"{organ}.nii.gz")
        pancreas_prob = _make_nii(np.full((4, 4, 4), 0.7, dtype=np.float32), tmp_path / "pancreas_probability.nii.gz")

        selected_organs = [
            {
                "organ": "liver",
                "ct_path": str(tmp_path / "ct.nii.gz"),
                "grade": "B",
                "training_weight": 0.5,
                "identity_status": "valid",
                "requested_canonical_id": "liver",
                "resolved_canonical_id": "liver",
                "comparison_family": "whole_organ",
                "scoring_schema_version": "autolabel_core_v2",
            },
            {
                "organ": "pancreas",
                "grade": "C",
                "training_weight": 0.1,
                "identity_status": "valid",
                "requested_canonical_id": "pancreas",
                "resolved_canonical_id": "pancreas",
                "comparison_family": "whole_organ",
                "scoring_schema_version": "autolabel_core_v2",
                "target_type": "soft",
                "probability_mask_path": str(pancreas_prob),
            },
            {
                "organ": "kidney_left",
                "grade": "D",
                "training_weight": 0.0,
                "identity_status": "valid",
                "requested_canonical_id": "kidney_left",
                "resolved_canonical_id": "kidney_left",
                "comparison_family": "whole_organ",
                "scoring_schema_version": "autolabel_core_v2",
            },
            {
                "organ": "kidney_right",
                "grade": "B",
                "training_weight": 0.5,
                "distillation_eligible": False,
                "distillation_exclusion_reason": "manual_review_hold",
                "identity_status": "valid",
                "requested_canonical_id": "kidney_right",
                "resolved_canonical_id": "kidney_right",
                "comparison_family": "whole_organ",
                "scoring_schema_version": "autolabel_core_v2",
            },
        ]
        (case / "selection_metadata.json").write_text(json.dumps({"selected_organs": selected_organs}), encoding="utf-8")

        result = build_training_manifest(root, tmp_path / "mstep_manifest.json")
        rows = json.loads((tmp_path / "mstep_manifest.json").read_text(encoding="utf-8"))
        organs = {row["organ"] for row in rows}

        assert result["num_items"] == 2
        assert organs == {"liver", "pancreas"}
        pancreas_row = next(row for row in rows if row["organ"] == "pancreas")
        assert pancreas_row["training_weight"] == 0.1
        assert pancreas_row["target_type"] == "soft"
        assert pancreas_row["training_gate_decision"] == "include_soft_c"
        exclusions = json.loads(Path(result["training_exclusions"]).read_text(encoding="utf-8"))
        assert {row["organ"] for row in exclusions} == {"kidney_left", "kidney_right"}
        assert next(row for row in exclusions if row["organ"] == "kidney_left")["distillation_exclusion_reason"] == "grade_D_or_zero_weight"
        assert next(row for row in exclusions if row["organ"] == "kidney_right")["distillation_exclusion_reason"] == "manual_review_hold"

    def test_liver_cd_visual_audit_writes_csv_json_and_zero_volume_diagnosis(self, tmp_path, monkeypatch):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("audit_liver_cd_visual_test", Path("scripts/audit_liver_cd_visual.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules.pop("audit_liver_cd_visual_test", None)
        spec.loader.exec_module(module)

        estep = tmp_path / "estep"
        meta_dir = estep / "annotation_versions" / "case_001"
        meta_dir.mkdir(parents=True)
        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        zero_liver = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "liver.nii.gz")
        meta = {
            "case_id": "case_001",
            "ct_path": str(ct),
            "selected_organs": [{
                "organ": "liver",
                "grade": "D",
                "training_weight": 0.0,
                "selected_model": "teacher_a",
                "mask_path": str(zero_liver),
                "selected_candidate_qc_flags": ["zero_volume_mask"],
                "identity_status": "valid",
                "candidate_models": ["teacher_a"],
                "candidate_predictions": [{"model": "teacher_a", "prediction": str(zero_liver)}],
            }],
        }
        (meta_dir / "selection_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        out = tmp_path / "audit"
        monkeypatch.setattr(sys, "argv", ["audit_liver_cd_visual.py", "--estep", str(estep), "--output-dir", str(out), "--max-cases", "20"])

        assert module.main() == 0
        summary = json.loads((out / "liver_cd_visual_audit.json").read_text(encoding="utf-8"))
        assert summary["num_cases"] == 1
        assert summary["rows"][0]["suspected_failure_type"] == "zero_volume_or_missing_misclassified"
        assert (out / "liver_cd_visual_audit.csv").exists()

    def test_mini_repair_summary_accepts_child_roi_and_flags_full_volume_child(self, tmp_path):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("mini_repair_check_test", Path("scripts/run_mini_hierarchical_repair_check.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules.pop("mini_repair_check_test", None)
        spec.loader.exec_module(module)

        out = tmp_path / "mini"
        estep = out / "estep"
        case_dir = estep / "cases" / "case_001"
        case_dir.mkdir(parents=True)
        (estep / "training_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (estep / "training_manifest.json").write_text(json.dumps([
            {"case_id": "case_001", "organ": "liver", "grade": "B", "training_weight": 0.5},
            {"case_id": "case_001", "organ": "pancreas", "grade": "A", "training_weight": 1.0},
            {"case_id": "case_001", "organ": "kidney", "grade": "B", "training_weight": 0.5},
            {"case_id": "case_001", "organ": "liver_segment_1", "grade": "C", "training_weight": 0.1},
        ]), encoding="utf-8")
        plan = {
            "case_id": "case_001",
            "roi_tasks": [{"organ": "liver_segment_1", "inference": {"inference_scope": "child_roi"}}],
            "blocked": [{"organ": "pancreas_tail", "status": "blocked_by_parent"}],
        }
        (case_dir / "hierarchical_inference_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        meta_dir = estep / "annotation_versions" / "case_001"
        meta_dir.mkdir(parents=True)
        (meta_dir / "selection_metadata.json").write_text(json.dumps({
            "case_id": "case_001",
            "selection_rows": [{
                "organ": "liver",
                "selected_model": "teacher_a",
                "candidate_models": ["teacher_a", "fusion_consensus"],
            }],
        }), encoding="utf-8")

        summary = module.summarize(out)
        assert summary["status"] == "success"
        assert summary["blocked"][0]["organ"] == "pancreas_tail"
        assert summary["fusion_rows"][0]["selected_model"] == "teacher_a"

        plan["roi_tasks"][0]["inference"]["inference_scope"] = "full_volume"
        (case_dir / "hierarchical_inference_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        failed = module.summarize(out)
        assert failed["status"] == "failed"
        assert failed["child_scope_failures"]

    def test_mini_repair_preseed_uses_raw_teacher_cache_not_old_selected_masks(self, tmp_path):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("mini_repair_check_test", Path("scripts/run_mini_hierarchical_repair_check.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules.pop("mini_repair_check_test", None)
        spec.loader.exec_module(module)

        old_estep = tmp_path / "old_round1" / "estep"
        # Old selected masks may include stale full-volume child outputs; they
        # must not become preseeded teacher roots for repair.
        selected = old_estep / "annotation_versions" / "case_001" / "updated"
        selected.mkdir(parents=True)
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), selected / "liver_segment_1.nii.gz")

        raw_seg = old_estep / "cases" / "case_001" / "raw_predictions" / "teacher_a" / "case_001" / "segmentations"
        raw_seg.mkdir(parents=True)
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), raw_seg / "liver.nii.gz")

        preseed = module.preseed_major_cache(old_estep)

        assert preseed == {
            "teacher_a": old_estep / "cases" / "{case_id}" / "raw_predictions" / "teacher_a"
        }
        assert all("annotation_versions" not in str(path) for path in preseed.values())


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
        lc_root = Path(__file__).resolve().parents[2] / "third_party" / "LabelCritic-main"
        result = build_projection(ct, mask_a=mask, mask_b=mask,
                                  output_folder=tmp_path / "out", organ="liver",
                                  projection_backend="labelcritic",
                                  labelcritic_root=str(lc_root),
                                  dry_run=True)
        assert result["stage"] in {"projection_builder", "labelcritic_projection_builder"}
        assert "saved_projections" in result
        assert "output_folder" in result


# ─── Test 7b : Multi-teacher label fusion ────────────────────────────────────

class TestLabelFusion:
    def test_staple_consensus_of_three_masks(self, tmp_path):
        from cli_anything.medai.core.label_fusion import fuse_candidate_masks

        cube = np.zeros((16, 16, 16), dtype=np.uint8)
        cube[4:10, 4:10, 4:10] = 1
        a = _make_nii(cube, tmp_path / "a.nii.gz")
        b = _make_nii(cube, tmp_path / "b.nii.gz")           # agrees with A
        c = _make_nii(np.zeros((16, 16, 16), dtype=np.uint8), tmp_path / "c.nii.gz")  # empty outlier

        out = tmp_path / "fused.nii.gz"
        meta = fuse_candidate_masks([a, b, c], out, method="auto")
        assert meta["status"] == "success"
        assert meta["method"] in {"staple", "weighted_vote"}
        assert out.exists()
        # 2/3 raters agree on the cube → consensus must be non-empty and bounded by the union.
        import nibabel as nib
        fused = np.asanyarray(nib.load(str(out)).dataobj) > 0
        assert 0 < int(fused.sum()) <= int(cube.sum())

    def test_weighted_vote_follows_weights(self, tmp_path):
        from cli_anything.medai.core.label_fusion import fuse_candidate_masks

        hi = np.zeros((16, 16, 16), dtype=np.uint8); hi[2:6, 2:6, 2:6] = 1
        lo = np.zeros((16, 16, 16), dtype=np.uint8); lo[10:14, 10:14, 10:14] = 1   # disjoint
        a = _make_nii(hi, tmp_path / "hi.nii.gz")
        b = _make_nii(lo, tmp_path / "lo.nii.gz")

        out = tmp_path / "fused.nii.gz"
        meta = fuse_candidate_masks([a, b], out, weights=[0.9, 0.1], method="weighted_vote")
        assert meta["status"] == "success" and meta["method"] == "weighted_vote"
        import nibabel as nib
        fused = np.asanyarray(nib.load(str(out)).dataobj) > 0
        # Only the high-weight region clears the 0.5 threshold.
        assert fused[2:6, 2:6, 2:6].all()
        assert not fused[10:14, 10:14, 10:14].any()

    def test_fusion_reuses_fingerprint_cache(self, tmp_path):
        from cli_anything.medai.core.label_fusion import fuse_candidate_masks

        a = _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), tmp_path / "a.nii.gz")
        b = _make_nii(_sphere_mask(shape=(16, 16, 16), radius=5), tmp_path / "b.nii.gz")
        out = tmp_path / "fused.nii.gz"

        first = fuse_candidate_masks([a, b], out, method="weighted_vote")
        second = fuse_candidate_masks([a, b], out, method="weighted_vote")

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert second["cache_status"] == "reused_fusion_cache"
        assert out.exists()

    def test_single_input_is_copied(self, tmp_path):
        from cli_anything.medai.core.label_fusion import fuse_candidate_masks

        cube = np.zeros((8, 8, 8), dtype=np.uint8); cube[2:5, 2:5, 2:5] = 1
        a = _make_nii(cube, tmp_path / "a.nii.gz")
        out = tmp_path / "fused.nii.gz"
        meta = fuse_candidate_masks([a, tmp_path / "missing.nii.gz"], out, method="auto")
        assert meta["status"] == "single"
        assert out.exists()


# ─── Test 8 : Teacher meeting pipeline contract ──────────────────────────────

class TestTeacherMeetingPipeline:
    def test_candidate_qc_marks_zero_volume_mask_without_calling_it_empty(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml
        monkeypatch.setenv("MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION", "1")

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.float32), tmp_path / "ct.nii.gz")
        good_arr = np.zeros((8, 8, 8), dtype=np.uint8)
        good_arr[2:5, 2:5, 2:5] = 1
        good = _make_nii(good_arr, tmp_path / "good.nii.gz")
        empty = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "zero.nii.gz")

        good_qc = ml._compute_candidate_qc(ct=ct, mask=good, organ="liver")
        empty_qc = ml._compute_candidate_qc(ct=ct, mask=empty, organ="liver")
        assert good_qc["status"] == "pass"
        assert empty_qc["status"] == "review"
        assert "zero_volume_mask" in empty_qc["flags"]
        assert "empty_mask" not in empty_qc["flags"]
        assert empty_qc["eligible_for_labelcritic"] is True

        critic_calls: list[str] = []

        def fake_critic(*args, **kwargs):
            critic_calls.append("called")
            return {"status": "success", "decision": {"winner": "a"}}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_critic)
        selected, selection = ml._select_candidate(
            ct=ct,
            organ="liver",
            candidates=[
                {
                    "model": "teacher_good",
                    "prediction": str(good),
                    "dice": None,
                    "candidate_qc": good_qc,
                    "candidate_qc_status": good_qc["status"],
                    "candidate_qc_score": good_qc["score"],
                    "candidate_qc_flags": good_qc["flags"],
                    "eligible_for_labelcritic": good_qc["eligible_for_labelcritic"],
                },
                {
                    "model": "teacher_zero_volume",
                    "prediction": str(empty),
                    "dice": None,
                    "candidate_qc": empty_qc,
                    "candidate_qc_status": empty_qc["status"],
                    "candidate_qc_score": empty_qc["score"],
                    "candidate_qc_flags": empty_qc["flags"],
                    "eligible_for_labelcritic": empty_qc["eligible_for_labelcritic"],
                },
            ],
            out=tmp_path,
            case_id="case_001",
            enable_critic=True,
            critic_backend="labelcritic",
            critic_base_url="http://localhost",
            critic_port=8000,
            timeout_sec=30,
            dry_run=False,
        )

        assert selected and selected["model"] == "teacher_good"
        assert selection["comparison_candidate_count"] == 2
        assert selection["comparison_candidate_models"] == ["teacher_good", "teacher_zero_volume"]
        assert critic_calls == ["called"]

    def test_reuses_zero_mask_raw_inference_cache(self, tmp_path):
        import cli_anything.medai.core.multimodel_loop as ml

        root = tmp_path / "case" / "raw_predictions" / "teacher_a" / "case_001"
        seg_dir = root / "segmentations"
        seg_dir.mkdir(parents=True)
        summary = {
            "status": "failed",
            "return_code": 0,
            "timed_out": False,
            "num_masks": 0,
            "segmentation_output": str(seg_dir),
        }
        (root / "inference_summary.json").write_text(json.dumps(summary), encoding="utf-8")

        cached = ml._load_reusable_raw_inference_cache(
            cached_summary=root / "inference_summary.json",
            cached_seg_dir=seg_dir,
            model_key="teacher_a",
        )

        assert cached is not None
        assert cached["num_masks"] == 0
        assert cached["cache_status"] == "reused_empty_raw_prediction"

    def test_multimodel_loop_writes_selection_manifest_gap_and_resume_metadata(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml
        monkeypatch.setenv("MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION", "1")

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(
            "case_id,ct_path,annotation_folder,scan_coverage\ncase_001,%s,,abdomen\n" % ct,
            encoding="utf-8",
        )
        out = tmp_path / "out"

        # Simulate a half-finished previous run: raw predictions exist, but no
        # selection metadata/ShapeKit/final masks.  Resume must not skip it.
        stale = out / "cases" / "case_001" / "raw_predictions" / "teacher_a" / "case_001" / "segmentations"
        stale.mkdir(parents=True)
        _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), stale / "liver.nii.gz")

        infer_calls: list[str] = []
        call_order: list[str] = []

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            infer_calls.append(model_key)
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            if model_key == "teacher_a":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), seg / "pancreas.nii.gz")
            if model_key == "teacher_b":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=5), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer",
                "status": "success",
                "model_key": model_key,
                "segmentation_output": str(seg),
                "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        def fake_critic(*args, **kwargs):
            call_order.append("critic")
            out_json = Path(args[4])
            out_json.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "status": "success",
                "decision": {"winner": "uncertain", "parse_status": "test_uncertain"},
            }
            out_json.write_text(json.dumps(result), encoding="utf-8")
            return result

        def fake_shapekit(*args, **kwargs):
            call_order.append("shapekit")
            return {
                "stage": "postprocess",
                "tool": "ShapeKit",
                "status": "failed",
                "reason": "No safe ShapeKit target organs detected",
            }

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_critic)
        monkeypatch.setattr(ml, "run_shapekit", fake_shapekit)

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list,
            output_folder=out,
            models=["teacher_a", "teacher_b"],
            organs=["liver", "pancreas", "spleen"],
            enable_critic=True,
            enable_shapekit=True,
            dry_run=False,
            resume=True,
            timeout_sec=30,
            teacher_inference_mode="full_volume",
            enable_fusion=False,  # this contract test targets the pick-one + shapekit-fallback plumbing
            enable_auto_arbitration=False,  # VLM grading is covered by its own tests
        )

        assert infer_calls, "resume must continue incomplete raw-only cases"
        assert "shapekit" in call_order and "critic" in call_order
        assert call_order.index("shapekit") < call_order.index("critic")
        assert result["status"] == "success"
        manifest = json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
        assert all(row["organ"] != "liver" for row in manifest)
        selection = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver = next(row for row in selection["selection_rows"] if row["organ"] == "liver")
        assert liver["selected_model"] is None
        assert liver["selection_method"] == "label_critic_inconclusive"
        assert liver["selection_status"] == "review_required"
        assert liver["should_enter_student_training"] is False
        assert liver["comparison_input_stage"] == "post_shapekit_candidate"
        assert liver["labelcritic_decision_path"]
        assert liver["label_critic_decision_path"]
        assert (out / "shapekit_report.json").exists()
        assert (out / "annotation_versions" / "case_001" / "shapekit_report.json").exists()
        exclusions = json.loads((out / "training_manifest.training_exclusions.json").read_text(encoding="utf-8"))
        assert isinstance(exclusions, list)
        gaps = json.loads((out / "pseudo_label_gap_report.json").read_text(encoding="utf-8"))
        pancreas = next(row for row in gaps["gap_rows"] if row["organ"] == "pancreas")
        assert pancreas["gap_type"] == "shapekit_unsupported_but_original_usable"
        gap_types = {row["gap_type"] for row in gaps["gap_rows"]}
        assert "missing_final_pseudo_label" in gap_types
        assert "shapekit_unsupported_but_original_usable" in gap_types
        resume_reasons = [row["reason"] for row in result["resume_audit"]]
        assert "selection_metadata_missing" in resume_reasons

    def test_resume_rebuilds_summary_consistently_with_incremental_cases(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        case_rows = []
        for idx in range(3):
            case_id = f"case_{idx+1:03d}"
            ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / f"{case_id}_ct.nii.gz")
            case_rows.append((case_id, ct))

        case_list_all = tmp_path / "cases_all.csv"
        case_list_all.write_text(
            "case_id,ct_path,annotation_folder\n" + "".join(f"{case_id},{ct},\n" for case_id, ct in case_rows),
            encoding="utf-8",
        )
        case_list_partial = tmp_path / "cases_partial.csv"
        case_list_partial.write_text(
            "case_id,ct_path,annotation_folder\n" + "".join(f"{case_id},{ct},\n" for case_id, ct in case_rows[:2]),
            encoding="utf-8",
        )

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            case_id = kwargs["case_id"]
            seg = Path(output_folder) / case_id / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), seg / "pancreas.nii.gz")
            return {
                "stage": "registered_infer",
                "status": "success",
                "model_key": model_key,
                "segmentation_output": str(seg),
                "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_shapekit", lambda *args, **kwargs: {"status": "skipped_debug_only"})

        out_resume = tmp_path / "out_resume"
        first = ml.run_multimodel_annotation_loop(
            case_list=case_list_partial,
            output_folder=out_resume,
            models=["teacher_a"],
            organs=["liver", "pancreas"],
            enable_critic=False,
            enable_shapekit=False,
            dry_run=False,
            resume=True,
            timeout_sec=30,
            teacher_inference_mode="full_volume",
            enable_fusion=False,
            enable_auto_arbitration=False,
        )
        assert first["status"] == "success"

        second = ml.run_multimodel_annotation_loop(
            case_list=case_list_all,
            output_folder=out_resume,
            models=["teacher_a"],
            organs=["liver", "pancreas"],
            enable_critic=False,
            enable_shapekit=False,
            dry_run=False,
            resume=True,
            timeout_sec=30,
            teacher_inference_mode="full_volume",
            enable_fusion=False,
            enable_auto_arbitration=False,
        )
        assert second["status"] == "success"

        out_full = tmp_path / "out_full"
        full = ml.run_multimodel_annotation_loop(
            case_list=case_list_all,
            output_folder=out_full,
            models=["teacher_a"],
            organs=["liver", "pancreas"],
            enable_critic=False,
            enable_shapekit=False,
            dry_run=False,
            resume=False,
            timeout_sec=30,
            teacher_inference_mode="full_volume",
            enable_fusion=False,
            enable_auto_arbitration=False,
        )
        assert full["status"] == "success"

        resume_summary = json.loads((out_resume / "run_summary.json").read_text(encoding="utf-8"))
        full_summary = json.loads((out_full / "run_summary.json").read_text(encoding="utf-8"))

        for key in ("num_cases", "selected_organs_rebuilt", "selection_rows_rebuilt", "total_updated"):
            assert resume_summary[key] == full_summary[key]
        assert resume_summary["summary_rebuilt_from_artifacts"] is True
        assert full_summary["summary_rebuilt_from_artifacts"] is True
        assert len(resume_summary["case_timing_breakdown"]) == len(full_summary["case_timing_breakdown"]) == 3

    def test_route_aware_execution_uses_existing_mapping_only(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(
            "case_id,ct_path,annotation_folder,scan_coverage\ncase_001,%s,,abdomen\n" % ct,
            encoding="utf-8",
        )
        out = tmp_path / "out"

        infer_calls = []

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            infer_calls.append(model_key)
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            if model_key == "teacher_a":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "candidate_models_for_organs", lambda registry, organs, include_mock=False: {"liver": ["teacher_a"], "pancreas": ["teacher_b"]})
        monkeypatch.setattr(ml, "_load_teacher_branch_map", lambda root: {
            "liver": {"teacher_model": "teacher_a", "fallback_teachers": []},
            "pancreas": {"teacher_model": "teacher_b", "fallback_teachers": []},
        })

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a", "teacher_b", "teacher_c"], organs=["liver"],
            enable_critic=False, enable_shapekit=False, dry_run=False, resume=False,
            timeout_sec=30, teacher_inference_mode="full_volume", candidate_mode="route_pruned_with_competition",
        )

        assert result["status"] == "success"
        assert infer_calls == ["teacher_a"]
        meta = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        assert meta["case_execution_plan"]["teacher_run_list"] == ["teacher_a"]

    def test_selection_metadata_records_compare_and_grade_policy(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml
        monkeypatch.setenv("MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION", "1")

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(
            "case_id,ct_path,annotation_folder,scan_coverage\ncase_001,%s,,abdomen\n" % ct,
            encoding="utf-8",
        )
        out = tmp_path / "out"

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4 if model_key == "teacher_a" else 5), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        def fake_compare(*args, **kwargs):
            out_json = Path(args[4])
            out_json.parent.mkdir(parents=True, exist_ok=True)
            result = {"status": "success", "decision": {"winner": "a", "parse_status": "success"}}
            out_json.write_text(json.dumps(result), encoding="utf-8")
            return result

        def fake_grade(*args, **kwargs):
            return {"status": "success", "grade": 0.9, "accept": True, "reason": "good"}

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        monkeypatch.setattr(ml, "run_labelcritic_grade", fake_grade)
        monkeypatch.setattr(ml, "candidate_models_for_organs", lambda registry, organs, include_mock=False: {"liver": ["teacher_a", "teacher_b"]})
        monkeypatch.setattr(ml, "_load_teacher_branch_map", lambda root: {"liver": {"teacher_model": "teacher_a", "fallback_teachers": ["teacher_b"]}})

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a", "teacher_b"], organs=["liver"],
            enable_critic=True, enable_shapekit=False, dry_run=False, resume=False,
            timeout_sec=30, teacher_inference_mode="full_volume", enable_fusion=False, candidate_mode="route_pruned_with_competition",
        )

        assert result["status"] == "success"
        meta = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver = meta["selected_organs"][0]
        assert liver["labelcritic_compare_used"] is True
        assert liver["labelcritic_compare_reason"] == "multi_candidate_conflict"
        assert liver["selected_candidate_id"] in liver["candidate_ids"]
        assert liver["selected_source_mask_sha256"] == liver["final_mask_sha256"]
        assert liver["mask_lineage_verified"] is True
        assert liver["labelcritic_grade_used"] is False
        assert liver["labelcritic_grade_skipped_reason"] == "policy_grade_disabled"
        assert liver["distillation_eligible"] is True
        assert liver["teacher_lineage"]

    def test_stable_single_candidate_runs_absolute_grade(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text("case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct, encoding="utf-8")
        out = tmp_path / "out"

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        grade_calls = []

        def fake_grade(*args, **kwargs):
            grade_calls.append(args[2])
            return {"status": "success", "grade": 0.9, "accept": True, "reason": "organ-specific absolute check passed"}

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_labelcritic_grade", fake_grade)
        monkeypatch.setattr(ml, "candidate_models_for_organs", lambda registry, organs, include_mock=False: {"liver": ["teacher_a"]})
        monkeypatch.setattr(ml, "_load_teacher_branch_map", lambda root: {"liver": {"teacher_model": "teacher_a", "fallback_teachers": []}})

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a"], organs=["liver"],
            enable_critic=True, enable_shapekit=False, dry_run=False, resume=False,
            timeout_sec=30, teacher_inference_mode="full_volume", candidate_mode="route_pruned_with_competition",
        )

        assert result["status"] == "success"
        meta = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver = meta["selected_organs"][0]
        assert liver["labelcritic_compare_used"] is False
        assert liver["labelcritic_compare_skipped_reason"] == "single_candidate_or_route_unique"
        assert liver["labelcritic_grade_used"] is False
        assert liver["labelcritic_grade_skipped_reason"] == "policy_grade_disabled"
        assert grade_calls == []

    def test_multimodel_loop_blocks_fusion_for_low_agreement_candidates(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml
        import csv as _csv

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text("case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct, encoding="utf-8")
        out = tmp_path / "out"

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            if model_key == "teacher_a":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), seg / "pancreas.nii.gz")
            if model_key == "teacher_b":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=5), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a", "teacher_b"], organs=["liver", "pancreas"],
            enable_critic=False, enable_shapekit=False, dry_run=False, resume=False,
            teacher_inference_mode="full_volume",
            timeout_sec=30,
        )
        assert result["status"] == "success"
        with open(out / "dice_metrics.csv") as f:
            rows = list(_csv.DictReader(f))
        assert rows
        assert set(rows[0]) >= {"metric_target", "metric_subject", "metric_comparison", "metric_interpretation"}
        assert {r["metric_target"] for r in rows} == {"pseudo-label"}
        liver_models = {r["model"] for r in rows if r["organ"] == "liver"}
        pancreas_models = {r["model"] for r in rows if r["organ"] == "pancreas"}
        # Conservative fusion v1: low-agreement teacher masks do not produce fusion.
        assert "fusion_consensus" not in liver_models
        assert {"teacher_a", "teacher_b"} <= liver_models
        # Single teacher for pancreas -> no fusion (needs >=2 candidates).
        assert "fusion_consensus" not in pancreas_models
        sel = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver_sel = next(r for r in sel["selection_rows"] if r["organ"] == "liver")
        assert "fusion_consensus" not in liver_sel["candidate_models"]
        assert all(p.get("is_fusion") is not True for p in liver_sel["candidate_predictions"])


    def test_materialize_case_373_targets_adds_absent_negative_only_for_expected_absent(self, tmp_path):
        from cli_anything.medai.core import multimodel_loop as ml
        import nibabel as nib

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        selection_rows = [{
            "case_id": "case_001", "organ": "liver", "selection_status": "selected",
            "target_type": "hard", "expected_presence": "expected_present",
        }]
        selected_metadata = [{
            "case_id": "case_001", "organ": "liver", "final_mask": str(ct), "target_type": "hard",
        }]
        summary = ml._materialize_case_373_targets(
            case_id="case_001",
            ct=ct,
            organs=["liver", "brain", "pancreas"],
            case_updated=tmp_path / "updated",
            selection_rows=selection_rows,
            selected_metadata=selected_metadata,
            presence_context={
                "has_region_evidence": True,
                "has_abdomen_coverage": True,
                "has_head_coverage": False,
            },
            negative_absent_training_weight=0.2,
        )

        by_organ = {row["organ"]: row for row in selection_rows}
        assert summary["expected_targets"] == 3
        assert by_organ["brain"]["target_type"] == "negative_absent"
        assert by_organ["brain"]["grade"] == "A"
        assert by_organ["brain"]["grade_scope"] == "absence"
        assert by_organ["brain"]["training_weight"] == 0.2
        assert by_organ["brain"]["labelcritic_called"] is False
        assert by_organ["brain"]["metric_target"] == "all-zero target"
        assert by_organ["brain"]["metric_interpretation"] == "negative_absence_quality"
        assert by_organ["pancreas"]["target_type"] == "unresolved_visible"
        assert by_organ["pancreas"]["training_weight"] == 0.0
        assert by_organ["pancreas"]["should_enter_student_training"] is False
        zero = nib.load(by_organ["brain"]["mask_path"])
        assert zero.shape[:3] == (8, 8, 8)
        assert np.asanyarray(zero.dataobj).sum() == 0

    def test_gap_rows_distinguish_expected_absent_from_route_failure(self, tmp_path):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        rows = ml._build_gap_rows(
            case_id="case_001",
            ct=ct,
            organs=["bladder", "liver"],
            selection_rows=[
                {
                    "organ": "bladder",
                    "expected_presence": "expected_absent",
                    "selection_status": "missing",
                    "candidate_count": 0,
                    "candidate_models": [],
                },
                {
                    "organ": "liver",
                    "expected_presence": "expected_present",
                    "selection_status": "missing",
                    "selected_candidate_qc_flags": ["missing_file"],
                    "candidate_count": 1,
                    "candidate_models": ["official_voxtell_pretrained"],
                },
            ],
            selected_metadata=[],
        )
        by_organ = {row["organ"]: row for row in rows}
        assert by_organ["bladder"]["reason"] == "organ_expected_absent_or_out_of_fov"
        assert by_organ["liver"]["reason"] == "model_or_route_failed_to_produce_usable_mask"
        assert by_organ["bladder"]["gap_severity"] == "informational_out_of_fov"
        assert by_organ["liver"]["gap_severity"] == "action_required_expected_present"

    def test_fov_hints_turn_head_and_extremity_gaps_into_out_of_fov(self, tmp_path):
        from cli_anything.medai.core import multimodel_loop as ml

        assert ml._expected_presence_for_organ(
            "brain",
            {"has_region_evidence": True, "has_abdomen_coverage": True, "has_head_coverage": False},
        ) == "expected_absent"
        assert ml._expected_presence_for_organ(
            "femur_left",
            {"has_region_evidence": True, "has_abdomen_coverage": True, "has_extremity_coverage": False},
        ) == "expected_absent"
        assert ml._expected_presence_for_organ(
            "pancreas",
            {"has_region_evidence": True, "has_abdomen_coverage": True},
        ) == "expected_present"
        assert ml._organ_region_hint("gall_bladder") == "pelvis"
        assert ml._expected_presence_for_organ(
            "gall_bladder",
            {"has_region_evidence": True, "has_abdomen_coverage": True, "has_pelvis_coverage": False},
        ) == "expected_present"
        assert ml._expected_presence_for_organ(
            "hip_right",
            {"has_region_evidence": True, "has_abdomen_coverage": True, "has_pelvis_coverage": False},
        ) == "unknown"


    def test_high_agreement_candidates_add_conservative_fusion_without_auto_winning(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml
        import csv as _csv

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text("case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct, encoding="utf-8")
        out = tmp_path / "out"
        identical = _sphere_mask(shape=(16, 16, 16), radius=4)

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            _make_nii(identical, seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a", "teacher_b"], organs=["liver"],
            enable_critic=False, enable_shapekit=False, dry_run=False, resume=False, timeout_sec=30,
            teacher_inference_mode="full_volume",
        )

        assert result["status"] == "success"
        with open(out / "dice_metrics.csv") as f:
            rows = list(_csv.DictReader(f))
        assert rows
        assert set(rows[0]) >= {"metric_target", "metric_subject", "metric_comparison", "metric_interpretation"}
        assert {r["metric_target"] for r in rows} == {"pseudo-label"}
        liver_models = {r["model"] for r in rows if r["organ"] == "liver"}
        assert "fusion_consensus" not in liver_models
        sel = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver_sel = next(r for r in sel["selection_rows"] if r["organ"] == "liver")
        assert "fusion_consensus" not in liver_sel["candidate_models"]
        assert liver_sel["selection_method"] == "critic_disabled_fallback"
        assert liver_sel["selection_status"] == "review_required"
        assert liver_sel["selected_model"] is None

    def test_labelcritic_grade_batch_dry_run_preserves_order(self, tmp_path):
        from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_grade_batch

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        m1 = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "m1.nii.gz")
        m2 = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=3), tmp_path / "m2.nii.gz")
        jobs = [
            {"ct_image": ct, "mask": m1, "organ": "liver", "output_json": tmp_path / "g1.json"},
            {"ct_image": ct, "mask": m2, "organ": "pancreas", "output_json": tmp_path / "g2.json"},
        ]

        results = run_labelcritic_grade_batch(jobs, dry_run=True, concurrency=2)

        assert [r["organ"] for r in results] == ["liver", "pancreas"]
        assert all(r["status"] == "dry_run" for r in results)
        assert (tmp_path / "g1.json").exists() and (tmp_path / "g2.json").exists()

    def test_labelcritic_grade_degrades_gracefully(self, tmp_path):
        from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_grade

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        cube = np.zeros((8, 8, 8), dtype=np.uint8); cube[2:5, 2:5, 2:5] = 1
        mask = _make_nii(cube, tmp_path / "liver.nii.gz")

        # dry-run never touches the VLM and yields no verdict (non-blocking).
        dr = run_labelcritic_grade(ct, mask, "liver", tmp_path / "g0.json", dry_run=True)
        assert dr["status"] == "dry_run" and dr["grade"] is None and dr["accept"] is None
        # missing mask -> skipped, still non-blocking.
        miss = run_labelcritic_grade(ct, tmp_path / "nope.nii.gz", "liver", tmp_path / "g1.json")
        assert miss["status"] == "skipped" and miss["accept"] is None

    def test_parse_grade_response_supports_semantic_grade_labels(self):
        from cli_anything.medai.core.labelcritic_wrapper import _parse_grade_response

        parsed = _parse_grade_response(
            '{"grade_label":"acceptable","grade":0.65,"accept":true,"reason":"plausible organ location","hard_failure_reason":"none"}',
            accept_grade=0.5,
        )
        assert parsed["parse_status"] == "success"
        assert parsed["grade"] == 0.65
        assert parsed["grade_label"] == "acceptable"
        assert parsed["accept"] is True
        assert parsed["hard_failure_reason"] == "none"

        labeled = _parse_grade_response(
            '{"grade":"bad","accept":false,"reason":"wrong organ","hard_failure_reason":"wrong_organ"}',
            accept_grade=0.5,
        )
        assert labeled["parse_status"] == "success"
        assert labeled["grade"] == 0.1
        assert labeled["grade_label"] == "bad"
        assert labeled["accept"] is False
        assert labeled["hard_failure_reason"] == "wrong_organ"

    def test_auto_arbitration_swaps_and_flags(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text("case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct, encoding="utf-8")
        out = tmp_path / "out"

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            if model_key == "teacher_a":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "liver.nii.gz")
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), seg / "pancreas.nii.gz")
            if model_key == "teacher_b":
                _make_nii(_sphere_mask(shape=(16, 16, 16), radius=5), seg / "liver.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        def fake_critic(*args, **kwargs):
            out_json = Path(args[4]); out_json.parent.mkdir(parents=True, exist_ok=True)
            res = {"status": "success", "decision": {"winner": "uncertain"}}
            out_json.write_text(json.dumps(res)); return res

        def fake_grade(ct_image, mask, organ, output_json, **kwargs):
            # Reject teacher_a's masks, accept anything else with a high grade.
            reject = "teacher_a" in str(mask)
            d = (
                {"status": "success", "grade": 0.1, "grade_label": "bad", "accept": False, "reason": "stub reject", "hard_failure_reason": "wrong_organ"}
                if reject else
                {"status": "success", "grade": 0.9, "grade_label": "good", "accept": True, "reason": "stub accept", "hard_failure_reason": None}
            )
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(output_json).write_text(json.dumps(d))
            return {"stage": "labelcritic_grade", "organ": organ, "mask": str(mask), **d}

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_critic)
        monkeypatch.setattr(ml, "run_labelcritic_grade", fake_grade)

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a", "teacher_b"], organs=["liver", "pancreas"],
            enable_critic=True, enable_shapekit=False, enable_fusion=False,
            enable_auto_arbitration=True, dry_run=False, resume=False, timeout_sec=30,
            teacher_inference_mode="full_volume",
        )
        assert result["status"] == "success"
        sel = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        rows = {r["organ"]: r for r in sel["selection_rows"]}
        # In strict mode an inconclusive comparison is withheld for review.
        assert rows["liver"]["labelcritic_grade_used"] is False
        assert rows["liver"]["labelcritic_grade_skipped_reason"] == "no_selected_candidate"
        assert rows["liver"]["selection_status"] == "review_required"
        # Single-teacher masks receive deterministic QC only; the deprecated
        # absolute single-mask grader is never invoked in the formal path.
        assert rows["pancreas"]["labelcritic_grade_used"] is False
        assert not (out / "auto_arbitration_log.jsonl").exists()

    def test_auto_arbitration_uses_batched_selected_grade_prefetch(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text("case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct, encoding="utf-8")
        out = tmp_path / "out"
        batch_calls = []

        def fake_infer(ct_image, output_folder, model_key, **kwargs):
            seg = Path(output_folder) / "case_001" / "segmentations"
            seg.mkdir(parents=True, exist_ok=True)
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=4), seg / "pancreas.nii.gz")
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), seg / "portal_vein_and_splenic_vein.nii.gz")
            return {
                "stage": "registered_infer", "status": "success", "model_key": model_key,
                "segmentation_output": str(seg), "num_masks": len(list(seg.glob("*.nii.gz"))),
            }

        def fake_grade_batch(jobs, **kwargs):
            batch_calls.append([job["organ"] for job in jobs])
            results = []
            for job in jobs:
                out_json = Path(job["output_json"])
                result = {
                    "stage": "labelcritic_grade",
                    "status": "success",
                    "organ": job["organ"],
                    "mask": str(job["mask"]),
                    "output_json": str(out_json),
                    "grade": 0.8,
                    "accept": True,
                    "reason": "batch stub",
                    "batch_status": "batched_grade",
                }
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(json.dumps(result), encoding="utf-8")
                results.append(result)
            return results

        def fail_single_grade(*args, **kwargs):
            raise AssertionError("selected-mask grades should come from the batch prefetch cache")

        monkeypatch.setattr(ml, "run_registered_model", fake_infer)
        monkeypatch.setattr(ml, "run_labelcritic_grade_batch", fake_grade_batch)
        monkeypatch.setattr(ml, "run_labelcritic_grade", fail_single_grade)
        fail_single_grade.__module__ = "cli_anything.medai.core.labelcritic_wrapper"
        monkeypatch.setattr(ml, "candidate_models_for_organs", lambda registry, organs, include_mock=False: {
            "pancreas": ["teacher_a"],
            "portal_vein_and_splenic_vein": ["teacher_a"],
        })
        monkeypatch.setattr(ml, "_load_teacher_branch_map", lambda root: {
            "pancreas": {"teacher_model": "teacher_a", "fallback_teachers": []},
            "portal_vein_and_splenic_vein": {"teacher_model": "teacher_a", "fallback_teachers": []},
        })

        result = ml.run_multimodel_annotation_loop(
            case_list=case_list, output_folder=out,
            models=["teacher_a"], organs=["pancreas", "portal_vein_and_splenic_vein"],
            enable_critic=True, enable_shapekit=False, enable_fusion=False,
            enable_auto_arbitration=True, dry_run=False, resume=False, timeout_sec=30,
            teacher_inference_mode="full_volume",
            candidate_mode="route_pruned_with_competition",
        )

        assert result["status"] == "success"
        assert batch_calls == []
        sel = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        rows = {r["organ"]: r for r in sel["selection_rows"]}
        assert rows["pancreas"]["labelcritic_grade_used"] is False
        assert rows["portal_vein_and_splenic_vein"]["labelcritic_grade_used"] is False

    def test_failure_mining_preserves_labelcritic_and_review_context(self, tmp_path):
        import importlib.util
        script = Path(__file__).resolve().parents[2] / "scripts" / "mine_student_failure_cases.py"
        spec = importlib.util.spec_from_file_location("mine_student_failure_cases", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        compare_pair = module.compare_pair

        student = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "student.nii.gz")
        ref_arr = np.zeros((8, 8, 8), dtype=np.uint8)
        ref_arr[1:4, 1:4, 1:4] = 1
        ref = _make_nii(ref_arr, tmp_path / "ref.nii.gz")
        row = compare_pair(
            round_idx=1,
            case_id="case_001",
            organ="liver",
            student_mask_path=student,
            reference_mask_path=ref,
            selection_meta={
                "selected_model": "round_prev_selected",
                "source_model": "round_prev_selected",
                "candidate_models": ["round_prev_selected", "student_prev"],
                "candidate_count": 2,
                "selection_method": "label_critic_fallback",
                "selection_status": "fallback",
                "fallback_reason": "LabelCritic inconclusive",
                "shapekit_status": "fallback_original",
                "review_flags": ["selection_fallback"],
                "quality_flags": ["shapekit_fallback"],
                "labelcritic_records": [{
                    "output_json": "/tmp/decision.json",
                    "decision": {"winner": "uncertain", "parse_status": "vlm_undecided"},
                }],
            },
            dice_threshold=0.5,
            volume_ratio_min=0.25,
            volume_ratio_max=4.0,
        )

        assert row["status"] == "review"
        assert row["metric_family"] == "pseudo_consistency"
        assert row["pseudo_consistency_dice"] == row["student_vs_pseudo_dice"]
        assert row["labelcritic_final_winner"] == "fallback"
        assert row["labelcritic_decision_paths"] == ["/tmp/decision.json"]
        assert row["labelcritic_parse_statuses"] == ["vlm_undecided"]
        assert "round1_selection_fallback" in row["review_reasons"]
        assert "round1_shapekit_fallback_original" in row["review_reasons"]

    def test_fine_label_eval_placeholder_uses_explicit_metric_family(self, tmp_path):
        import importlib.util
        script = Path(__file__).resolve().parents[2] / "scripts" / "mine_student_failure_cases.py"
        spec = importlib.util.spec_from_file_location("mine_student_failure_cases", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        student = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "student.nii.gz")
        fine = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "fine.nii.gz")
        row = module.compare_fine_label_pair(
            round_idx=1,
            case_id="case_001",
            organ="liver",
            student_mask_path=student,
            fine_label_mask_path=fine,
            dice_threshold=0.5,
            volume_ratio_min=0.25,
            volume_ratio_max=4.0,
        )
        assert row["metric_family"] == "fine_label_eval"
        assert row["metric_scope"] == "student_vs_expert_fine_label"
        assert row["ground_truth_status"] == "expert_fine_label"


# ─── Test 9 : Cross-round convergence auto-stop (Phase 3) ─────────────────────

class TestConvergenceAutoStop:
    def _load_module(self):
        import importlib.util
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_em_training.py"
        spec = importlib.util.spec_from_file_location("run_em_training", script)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_converged_when_delta_below_threshold(self):
        m = self._load_module()
        assert m.convergence_reached(0.80, 0.795, 0.01) is True
        assert m.convergence_reached(0.80, 0.80, 0.01) is True

    def test_not_converged_when_delta_large_or_missing(self):
        m = self._load_module()
        assert m.convergence_reached(0.80, 0.70, 0.01) is False     # still improving
        assert m.convergence_reached(None, 0.70, 0.01) is False     # no current metric
        assert m.convergence_reached(0.80, None, 0.01) is False     # no previous metric


# ─── Test 10 : Phase-2 verdict feeds Phase-3 reliability weight ───────────────

class TestAutoGradeRejectDownweights:
    def test_auto_grade_reject_lowers_training_weight(self):
        from cli_anything.medai.core.auto_fine_label import compute_reliability

        base = {
            "mask_path": "x/liver.nii.gz",
            "selected_candidate_qc_status": "pass",
            "shapekit_status": "success",
            "selection_method": "label_critic",
            "selected_pseudo_consistency_dice": 0.9,
            "review_flags": [],
            "quality_flags": [],
        }
        good = compute_reliability(base)
        rejected = compute_reliability({**base, "review_flags": ["auto_grade_reject"]})
        # Process/VLM signals cannot manufacture accuracy evidence in v2.
        assert rejected["training_weight"] == good["training_weight"] == 0.0
        assert rejected["auto_fine_label_reliability_score"] == good["auto_fine_label_reliability_score"]
        # And on an already-weak label it pushes all the way to the zero-weight floor.
        weak = {**base, "selected_candidate_qc_status": "review", "shapekit_status": "failed",
                "selection_method": "label_critic_fallback", "selected_pseudo_consistency_dice": 0.4}
        weak_rej = compute_reliability({**weak, "review_flags": ["auto_grade_reject", "selection_fallback"]})
        assert weak_rej["grade"] == "D" and weak_rej["training_weight"] == 0.0


class TestLabelCriticWrapperSafety:
    def test_labelcritic_grade_skips_zero_volume_mask_without_bad_grade(self, tmp_path):
        from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_grade

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        zero = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "liver.nii.gz")

        result = run_labelcritic_grade(ct, zero, "liver", tmp_path / "grade.json")

        assert result["status"] == "skipped"
        assert result["grade"] is None
        assert result["parse_status"] == "zero_volume_mask"

    def test_parse_labelcritic_log_refuses_stale_tail_when_run_id_missing(self, tmp_path):
        from cli_anything.medai.core.labelcritic_wrapper import _parse_labelcritic_log

        log = tmp_path / "comparison_summary.log"
        mask1 = tmp_path / "old_mask1"
        mask2 = tmp_path / "old_mask2"
        mask1.mkdir()
        mask2.mkdir()
        log.write_text(f"Run ID: old123\nBetter: {mask2}\n", encoding="utf-8")

        decision = _parse_labelcritic_log(log, mask1, mask2, run_id="new456")

        assert decision["winner"] == "uncertain"
        assert decision["parse_status"] == "run_id_not_found"

    def test_labelcritic_compare_batch_stub_writes_outputs(self, tmp_path):
        from cli_anything.medai.core import labelcritic_wrapper as lw

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask_a = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "a.nii.gz")
        mask_b = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=3), tmp_path / "b.nii.gz")
        jobs = [
            {
                "ct_image": ct, "mask_a": mask_a, "mask_b": mask_b, "organ": "liver",
                "output_json": tmp_path / "c1.json",
                "candidate_context": [
                    {"candidate_id": "c1", "teacher_name": "teacher_a", "qc_status": "pass"},
                    {"candidate_id": "c2", "teacher_name": "teacher_b", "qc_status": "pass"},
                ],
            },
            {"ct_image": ct, "mask_a": mask_b, "mask_b": mask_a, "organ": "pancreas", "output_json": tmp_path / "c2.json"},
        ]

        results = lw.run_labelcritic_compare_batch(jobs, backend="stub")

        assert [r["organ"] for r in results] == ["liver", "pancreas"]
        assert all(r["status"] == "stub" for r in results)
        assert all(r["batch_status"] == "batched_stub" for r in results)
        assert all(r["rendered_organ_prompt_hash"] for r in results)
        assert "ct_appearance" in results[0]["rendered_organ_prompt"]
        assert "teacher_a" in results[0]["rendered_organ_prompt"]
        assert (tmp_path / "c1.json").exists() and (tmp_path / "c2.json").exists()

    def test_labelcritic_compare_reuses_fingerprint_cache_with_reversed_pair(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import labelcritic_wrapper as lw

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
        mask_a = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=2), tmp_path / "a.nii.gz")
        mask_b = _make_nii(_sphere_mask(shape=(8, 8, 8), radius=3), tmp_path / "b.nii.gz")
        calls = {"projection": 0}

        def fake_projection(*args, **kwargs):
            calls["projection"] += 1
            out = Path(args[3])
            out.mkdir(parents=True, exist_ok=True)
            png = out / "stub.png"
            png.write_bytes(b"png")
            return {"status": "dry_run", "saved_projections": [str(png)]}

        monkeypatch.setattr(lw, "build_projection", fake_projection)

        first = lw.run_labelcritic_compare(ct, mask_a, mask_b, "liver", tmp_path / "first.json", backend="stub")
        second = lw.run_labelcritic_compare(ct, mask_b, mask_a, "liver", tmp_path / "second.json", backend="stub")

        assert first["status"] == "stub"
        assert second["cache_status"] == "reused_labelcritic_compare_fingerprint"
        assert second["decision"]["cache_orientation"] == "inverted"
        assert calls["projection"] == 1

    def test_labelcritic_invert_compare_decision_swaps_winner(self):
        from cli_anything.medai.core.labelcritic_wrapper import _invert_compare_decision

        assert _invert_compare_decision({"winner": "a"})["winner"] == "b"
        assert _invert_compare_decision({"winner": "b"})["winner"] == "a"
        assert _invert_compare_decision({"winner": "uncertain"})["winner"] == "uncertain"

    def test_prepare_mask_folder_does_not_mutate_source_directory(self, tmp_path):
        from cli_anything.medai.core.labelcritic_wrapper import _prepare_mask_folder

        src_dir = tmp_path / "raw_predictions"
        src_dir.mkdir()
        _make_nii(np.ones((4, 4, 4), dtype=np.uint8), src_dir / "kidney_left.nii.gz")

        prepared = _prepare_mask_folder(src_dir, "kidney_left", tmp_path / "work", "mask1")

        assert (prepared / "kidney_left.nii.gz").exists()
        assert (prepared / "kidney_right.nii.gz").exists()
        assert not (src_dir / "kidney_right.nii.gz").exists()


# ─── Test 11 : LabelCritic tournament robustness (inconclusive != abort) ──────

class TestLabelCriticTournament:
    @pytest.fixture(autouse=True)
    def _allow_uncalibrated_unit_stub(self, monkeypatch):
        monkeypatch.setenv("MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION", "1")

    def test_labelcritic_decisive_selection_sets_primary_metadata(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        mask_a = _make_nii(_sphere_mask(shape=(16, 16, 16), radius=3), tmp_path / "a.nii.gz")
        mask_b = _make_nii(_sphere_mask(shape=(16, 16, 16), radius=6), tmp_path / "b.nii.gz")
        cands = [
            {"model": "teacher_a", "prediction": str(mask_a), "eligible_for_labelcritic": True, "dice": 0.5},
            {"model": "teacher_b", "prediction": str(mask_b), "eligible_for_labelcritic": True, "dice": 0.7},
        ]

        def fake_compare(ct, a, b, organ, out_json, **kw):
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text("{}")
            winner = "a" if Path(a).name == "b.nii.gz" else "b"
            return {"status": "success", "decision": {"winner": winner, "confidence": 0.91}}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={},
        )

        assert selected["model"] == "teacher_b"
        assert sel["selection_method"] == "label_critic"
        assert sel["primary_selector"] == "official_labelcritic_pairwise_condorcet"
        assert sel["comparison_decisive_count"] == 1
        assert ml._labelcritic_locks_selection(sel) is True

    def test_labelcritic_lock_helper_rejects_fallback_methods(self):
        from cli_anything.medai.core import multimodel_loop as ml

        assert ml._labelcritic_locks_selection({"selection_method": "label_critic", "comparison_decisive_count": 1}) is True
        assert ml._labelcritic_locks_selection({"selection_method": "label_critic_inconclusive", "comparison_decisive_count": 0}) is False
        assert ml._labelcritic_locks_selection({"selection_method": "near_identical_agreement", "comparison_decisive_count": 0}) is False

    @pytest.mark.skip(reason="superseded by complete pairwise Condorcet abstention contract")
    def test_inconclusive_pair_does_not_abort_tournament(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        cands = [
            {"model": "fusion_consensus", "prediction": str(tmp_path / "f.nii.gz"),
             "is_fusion": True, "eligible_for_labelcritic": True, "dice": 0.6},
            {"model": "teacher_a", "prediction": str(tmp_path / "a.nii.gz"),
             "eligible_for_labelcritic": True, "dice": 0.5},
            {"model": "teacher_b", "prediction": str(tmp_path / "b.nii.gz"),
             "eligible_for_labelcritic": True, "dice": 0.7},
        ]
        calls = {"n": 0}

        def fake_compare(ct, a, b, organ, out_json, **kw):
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text("{}")
            calls["n"] += 1
            # 1st pair inconclusive, 2nd pair decisive winner = challenger (b).
            decision = {"winner": "uncertain"} if calls["n"] == 1 else {"winner": "b"}
            return {"status": "success", "decision": decision}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={},
        )
        assert calls["n"] == 2                              # both pairs compared (no early abort)
        assert sel["selection_method"] == "label_critic"
        assert sel["selection_status"] == "selected"
        assert sel["comparison_decisive_count"] == 1
        assert sel["comparison_inconclusive_count"] == 1
        assert selected["model"] == "teacher_b"            # decisive winner kept

    def test_all_inconclusive_is_withheld_in_strict_mode(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        cands = [
            {"model": "teacher_a", "prediction": str(tmp_path / "a.nii.gz"),
             "eligible_for_labelcritic": True, "dice": 0.5},
            {"model": "fusion_consensus", "prediction": str(tmp_path / "f.nii.gz"),
             "is_fusion": True, "eligible_for_labelcritic": True, "dice": 0.4},
        ]

        def fake_compare(ct, a, b, organ, out_json, **kw):
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text("{}")
            return {"status": "success", "decision": {"winner": "uncertain"}}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={},
        )
        assert sel["selection_method"] == "single_teacher_provisional"
        assert sel["selection_status"] == "provisional"
        assert selected["model"] == "teacher_a"
        assert sel["should_enter_student_training"] is False


    @pytest.mark.skip(reason="sequential tournament was intentionally replaced by complete Condorcet pairwise")
    def test_three_candidate_tournament_keeps_sequential_semantics(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        masks = [
            _make_nii(_sphere_mask(shape=(16, 16, 16), radius=r), tmp_path / f"m{r}.nii.gz")
            for r in (3, 5, 7)
        ]
        cands = [
            {"model": "fusion_consensus", "prediction": str(masks[0]), "is_fusion": True, "eligible_for_labelcritic": True, "dice": 0.6},
            {"model": "teacher_a", "prediction": str(masks[1]), "eligible_for_labelcritic": True, "dice": 0.5},
            {"model": "teacher_b", "prediction": str(masks[2]), "eligible_for_labelcritic": True, "dice": 0.4},
        ]
        compare_pairs = []

        def fake_compare(ct, mask_a, mask_b, organ, output_json, **kwargs):
            compare_pairs.append((Path(mask_a).name, Path(mask_b).name))
            decision = {"winner": "b" if len(compare_pairs) == 1 else "a", "parse_status": "success"}
            result = {"status": "success", "decision": decision, "output_json": str(output_json)}
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(output_json).write_text(json.dumps(result), encoding="utf-8")
            return result

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={}, compare_batch_enabled=True, compare_batch_max_candidates=2,
        )

        assert selected["model"] == "teacher_a"
        assert compare_pairs == [("m3.nii.gz", "m5.nii.gz"), ("m5.nii.gz", "m7.nii.gz")]
        assert len(sel["labelcritic_records"]) == 2


    def test_three_candidate_all_pair_high_agreement_skips_compare(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        m = _sphere_mask(shape=(16, 16, 16), radius=5)
        masks = [_make_nii(m, tmp_path / f"m{i}.nii.gz") for i in range(3)]
        cands = [
            {"model": "fusion_consensus", "prediction": str(masks[0]), "is_fusion": True, "eligible_for_labelcritic": True, "dice": 0.6},
            {"model": "teacher_a", "prediction": str(masks[1]), "eligible_for_labelcritic": True, "dice": 0.5},
            {"model": "teacher_b", "prediction": str(masks[2]), "eligible_for_labelcritic": True, "dice": 0.4},
        ]
        called = {"n": 0}

        def fake_compare(*args, **kwargs):
            called["n"] += 1
            return {"status": "success", "decision": {"winner": "b"}}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={},
        )

        assert called["n"] == 0
        assert selected["model"] in {"fusion_consensus", "teacher_a", "teacher_b"}
        assert sel["selection_method"] == "near_identical_agreement"
        assert sel["labelcritic_records"][0]["status"] == "skipped_near_identical"

    def test_near_identical_pair_skips_vlm(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        m = _sphere_mask(shape=(16, 16, 16), radius=5)
        a = _make_nii(m, tmp_path / "a.nii.gz")
        b = _make_nii(m, tmp_path / "b.nii.gz")   # identical -> 3D Dice 1.0
        cands = [
            {"model": "fusion_consensus", "prediction": str(a), "is_fusion": True,
             "eligible_for_labelcritic": True, "dice": 0.6},
            {"model": "teacher_a", "prediction": str(b),
             "eligible_for_labelcritic": True, "dice": 0.5},
        ]
        called = {"n": 0}

        def fake_compare(*args, **kw):
            called["n"] += 1
            return {"status": "success", "decision": {"winner": "b"}}

        monkeypatch.setattr(ml, "run_labelcritic_compare", fake_compare)
        selected, sel = ml._select_candidate(
            ct=tmp_path / "ct.nii.gz", organ="liver", candidates=cands, out=tmp_path / "out",
            case_id="c1", enable_critic=True, critic_backend="labelcritic",
            critic_base_url="http://localhost", critic_port=8000, timeout_sec=30,
            dry_run=False, labelcritic_options={},
        )
        assert called["n"] == 0                      # near-identical pair -> VLM not invoked
        assert sel["comparison_agreed_count"] == 0
        assert sel["selection_method"] == "single_teacher_provisional"
        assert sel["selection_status"] == "provisional"
        assert "single_teacher_provisional" in sel["quality_flags"]
        assert "selection_fallback" not in sel["review_flags"]
        assert selected["model"] == "teacher_a"
        assert sel["excluded_fusion_candidates"] == ["fusion_consensus"]
