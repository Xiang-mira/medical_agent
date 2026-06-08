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
        assert passport["grade"] == "A"
        assert passport["training_weight"] == 1.0
        assert passport["auto_fine_label_status"] == "auto_fine_label_accepted"


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
        assert result["official_output_masks"]["kidney_left"].endswith("ct_kidney_left.nii.gz")
        assert result["io_contract"]["combined_multilabel_policy"].startswith("not used formally")


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
    def test_candidate_qc_rejects_empty_mask_before_labelcritic(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((8, 8, 8), dtype=np.float32), tmp_path / "ct.nii.gz")
        good_arr = np.zeros((8, 8, 8), dtype=np.uint8)
        good_arr[2:5, 2:5, 2:5] = 1
        good = _make_nii(good_arr, tmp_path / "good.nii.gz")
        empty = _make_nii(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "empty.nii.gz")

        good_qc = ml._compute_candidate_qc(ct=ct, mask=good, organ="liver")
        empty_qc = ml._compute_candidate_qc(ct=ct, mask=empty, organ="liver")
        assert good_qc["status"] == "pass"
        assert empty_qc["status"] == "fail"
        assert "empty_mask" in empty_qc["flags"]
        assert empty_qc["eligible_for_labelcritic"] is False

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
                    "model": "teacher_empty",
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
        assert selection["selection_method"] == "single_teacher_default"
        assert selection["comparison_candidate_models"] == ["teacher_good"]
        assert selection["comparison_candidate_count"] == 1
        assert selection["qc_rejected_candidates"][0]["model"] == "teacher_empty"
        assert "candidate_qc_rejected" in selection["review_flags"]
        assert critic_calls == []

    def test_multimodel_loop_writes_selection_manifest_gap_and_resume_metadata(self, tmp_path, monkeypatch):
        from cli_anything.medai.core import multimodel_loop as ml

        ct = _make_nii(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
        case_list = tmp_path / "cases.csv"
        case_list.write_text(
            "case_id,ct_path,annotation_folder\ncase_001,%s,\n" % ct,
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
            enable_fusion=False,  # this contract test targets the pick-one + shapekit-fallback plumbing
        )

        assert infer_calls, "resume must continue incomplete raw-only cases"
        assert "shapekit" in call_order and "critic" in call_order
        assert call_order.index("shapekit") < call_order.index("critic")
        assert result["status"] == "success"
        manifest = json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest
        liver = next(row for row in manifest if row["organ"] == "liver")
        assert liver["dataset_type"] == "auto_fine_label_dataset"
        assert liver["ground_truth_status"] == "machine_generated_candidate"
        assert liver["label_passport_path"]
        assert liver["training_weight"] in {0.0, 0.1, 0.5, 1.0}
        assert liver["source_model"] == "teacher_a"
        assert liver["selection_method"] == "label_critic_fallback"
        assert liver["comparison_input_stage"] == "post_shapekit_candidate"
        assert liver["labelcritic_decision_path"]
        assert liver["label_critic_decision_path"]
        assert liver["shapekit_status"] == "unsupported_target"
        assert liver["quality_status"] == "postprocess_review"
        assert (out / "shapekit_report.json").exists()
        assert (out / "annotation_versions" / "case_001" / "shapekit_report.json").exists()
        pancreas = next(row for row in manifest if row["organ"] == "pancreas")
        assert pancreas["selection_method"] == "single_teacher_default"
        gaps = json.loads((out / "pseudo_label_gap_report.json").read_text(encoding="utf-8"))
        gap_types = {row["gap_type"] for row in gaps["gap_rows"]}
        assert "missing_final_pseudo_label" in gap_types
        assert "shapekit_not_success" in gap_types
        resume_reasons = [row["reason"] for row in result["resume_audit"]]
        assert "selection_metadata_missing" in resume_reasons

    def test_multimodel_loop_adds_fusion_candidate_for_multi_teacher_organ(self, tmp_path, monkeypatch):
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
            timeout_sec=30,  # enable_fusion defaults to True
        )
        assert result["status"] == "success"
        with open(out / "dice_metrics.csv") as f:
            rows = list(_csv.DictReader(f))
        liver_models = {r["model"] for r in rows if r["organ"] == "liver"}
        pancreas_models = {r["model"] for r in rows if r["organ"] == "pancreas"}
        # Two teachers for liver -> a fusion_consensus candidate is added and competes.
        assert "fusion_consensus" in liver_models
        assert {"teacher_a", "teacher_b"} <= liver_models
        # Single teacher for pancreas -> no fusion (needs >=2 candidates).
        assert "fusion_consensus" not in pancreas_models
        # The fused consensus mask is written and auditable in selection metadata.
        sel = json.loads((out / "annotation_versions" / "case_001" / "selection_metadata.json").read_text(encoding="utf-8"))
        liver_sel = next(r for r in sel["selection_rows"] if r["organ"] == "liver")
        assert "fusion_consensus" in liver_sel["candidate_models"]

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
