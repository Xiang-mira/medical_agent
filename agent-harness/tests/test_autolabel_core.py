from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cli_anything.medai.core.auto_label_core import (
    family_membership,
    score_candidate_set,
    score_evidence_record,
    stable_case_fold,
    write_family_probability_fusion,
)
from cli_anything.medai.core.organ_model_performance import OrganModelPerformance
from cli_anything.medai.core.longtail_critic import build_pairwise_dataset, train_pairwise_ranker


def _mask(path: Path, offset: int = 0) -> Path:
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[4 + offset:12 + offset, 5:13, 6:14] = 1
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _complete_record(**updates):
    record = {
        "mask_path": "/tmp/mask.nii.gz",
        "identity_status": "valid",
        "selected_candidate_qc_status": "pass",
        "independent_family_count": 2,
        "family_consensus_score": 1.0,
        "ct_support_score": 1.0,
        "anatomy_plausibility_score": 1.0,
        "perturbation_stability_score": 1.0,
        "cross_round_stability_score": 1.0,
        "loo_model_reliability_score": 1.0,
    }
    record.update(updates)
    return record


def test_labelcritic_unsupported_does_not_change_confidence():
    baseline = score_evidence_record(_complete_record())
    unsupported = score_evidence_record(_complete_record(
        labelcritic_supported=False,
        labelcritic_tiebreak_adjustment=-0.03,
        quality_flags=["labelcritic_unsupported_organ"],
    ))
    assert unsupported["evidence_confidence"] == baseline["evidence_confidence"]


def test_qc_pass_is_not_positive_accuracy_evidence():
    result = score_evidence_record({
        "mask_path": "/tmp/mask.nii.gz", "identity_status": "valid",
        "selected_candidate_qc_status": "pass", "independent_family_count": 1,
    })
    assert result["grade"] == "D"
    assert result["training_weight"] == 0.0


def test_single_family_is_grade_c_capped_even_with_complete_evidence():
    result = score_evidence_record(_complete_record(independent_family_count=1, probability_mask_path="/tmp/p.nii.gz"))
    assert result["grade"] == "C"
    assert result["grade_cap"] == "C"
    assert result["grade_cap_reason"] == "single_teacher_without_verified_oof"
    assert result["training_weight"] == 0.1


@pytest.mark.parametrize("flag", ["geometry_mismatch", "identity_mismatch", "left_right_mismatch", "zero_volume_mask"])
def test_hard_failures_are_grade_d(flag):
    result = score_evidence_record(_complete_record(quality_flags=[flag]))
    assert result["grade"] == "D"
    assert result["target_type"] == "rejected"
    assert result["training_weight"] == 0.0


def test_family_fusion_does_not_multiply_correlated_checkpoints(tmp_path):
    a = _mask(tmp_path / "a.nii.gz")
    a2 = _mask(tmp_path / "a2.nii.gz")
    b = _mask(tmp_path / "b.nii.gz", offset=1)
    candidates = [
        {"model": "cads551", "prediction": str(a), "candidate_qc_score": 1.0},
        {"model": "cads552", "prediction": str(a2), "candidate_qc_score": 1.0},
        {"model": "totalsegmentator", "prediction": str(b), "candidate_qc_score": 1.0},
    ]
    membership = family_membership(candidates)
    assert membership["cads"] == ["cads551", "cads552"]
    result = write_family_probability_fusion(candidates, tmp_path / "fusion")
    assert result["status"] == "success"
    assert result["family_fusion_count"] == 2
    probability = np.asanyarray(nib.load(result["probability_mask_path"]).dataobj)
    assert set(np.unique(probability)).issubset({0.0, 0.5, 1.0})


def test_legacy_tracker_update_is_disabled(tmp_path):
    tracker = OrganModelPerformance(tmp_path / "performance.json")
    with pytest.raises(RuntimeError, match="Legacy pseudo-reference Dice"):
        tracker.update("liver", "model_a", 0.9)


def test_leave_one_family_tracker_requires_two_other_families(tmp_path):
    tracker = OrganModelPerformance(tmp_path / "performance.json")
    changed = tracker.update_leave_one_family_out(
        organ="liver", organ_family="whole_organ", model="model_a", evidence_family="family_a",
        dice=0.9, nsd=0.9, ct_support=0.8, anatomy_plausibility=1.0,
        other_family_count=1, case_id="case-1",
    )
    assert changed is False
    assert tracker.get_stats("liver", "model_a") is None


def test_case_fold_is_stable():
    assert stable_case_fold("case-001") == stable_case_fold("case-001")
    assert 0 <= stable_case_fold("case-001") < 5


def test_in_sample_student_is_not_an_independent_family(tmp_path):
    ct = _mask(tmp_path / "ct.nii.gz")
    teacher = _mask(tmp_path / "teacher.nii.gz")
    student = _mask(tmp_path / "student.nii.gz")
    candidates = [
        {"model": "cads551", "prediction": str(teacher), "candidate_exists": True, "identity_status": "valid", "candidate_qc_status": "pass"},
        {"model": "student_prev", "prediction": str(student), "candidate_exists": True, "identity_status": "valid", "candidate_qc_status": "pass", "out_of_fold": False},
    ]
    decision = score_candidate_set("case-1", "liver", ct, candidates, selected_model="cads551", output_dir=tmp_path / "out")
    assert decision.independent_family_count == 1
    assert decision.grade != "A"


def test_unavailable_evidence_is_removed_from_denominator():
    result = score_evidence_record({
        "identity_status": "valid",
        "selected_candidate_qc_status": "pass",
        "independent_family_count": 1,
        "probability_mask_path": "/tmp/p.nii.gz",
        "evidence_details": {
            "ct_support": {"score": 0.8, "status": "available"},
            "anatomy_plausibility": {"score": 1.0, "status": "available"},
            "family_consensus": {"score": None, "status": "unavailable"},
            "perturbation_stability": {"score": None, "status": "unavailable"},
            "cross_round_stability": {"score": None, "status": "unavailable"},
            "loo_model_reliability": {"score": None, "status": "unavailable"},
            "teacher_student_oof": {"score": None, "status": "unavailable"},
        },
    })
    assert result["available_weight_sum"] == pytest.approx(0.35)
    assert result["normalized_evidence_confidence"] == pytest.approx((0.2 * 0.8 + 0.15) / 0.35)
    assert result["grade"] == "C"


def test_failed_evidence_stays_in_denominator_as_zero():
    result = score_evidence_record({
        "identity_status": "valid",
        "selected_candidate_qc_status": "pass",
        "independent_family_count": 1,
        "probability_mask_path": "/tmp/p.nii.gz",
        "evidence_details": {
            "ct_support": {"score": 1.0, "status": "available"},
            "anatomy_plausibility": {"score": None, "status": "failed"},
        },
    })
    assert result["available_weight_sum"] == pytest.approx(0.35)
    assert result["normalized_evidence_confidence"] == pytest.approx(0.2 / 0.35)
    assert "anatomy_plausibility" in result["failed_evidence"]


def test_verified_oof_can_promote_single_teacher_to_b_but_not_a():
    result = score_evidence_record({
        **_complete_record(independent_family_count=1),
        "out_of_fold_verified": True,
        "teacher_student_oof_score": 0.95,
        "cross_round_stability_score": 0.95,
        "perturbation_stability_score": 0.95,
        "ct_support_score": 0.9,
        "anatomy_plausibility_score": 1.0,
    })
    assert result["grade"] == "B"
    assert result["grade_cap"] == "B"
    assert result["oof_promotable"] is True


def test_single_teacher_can_promote_to_b_with_strict_non_oof_evidence():
    result = score_evidence_record({
        **_complete_record(independent_family_count=1),
        "expected_presence": "expected_present",
        "ct_support_score": 0.8,
        "anatomy_plausibility_score": 1.0,
        "perturbation_stability_score": 0.95,
        "labelcritic_calibrated_score": 0.8,
        "parent_containment": 0.95,
        "review_flags": [],
        "quality_flags": [],
    })
    assert result["grade"] == "B"
    assert result["grade_cap"] == "B"
    assert result["single_teacher_b_eligible"] is True


def test_expected_present_zero_volume_is_hard_fail():
    result = score_evidence_record({
        **_complete_record(independent_family_count=1),
        "expected_presence": "expected_present",
        "selected_candidate_qc_flags": ["zero_volume_mask"],
    })
    assert result["grade"] == "D"
    assert result["target_type"] == "rejected"


def test_oof_provenance_rejects_leakage_and_accepts_heldout_case():
    from cli_anything.medai.core.auto_label_core import verify_oof_student_provenance

    case_id = "case-oof"
    fold = stable_case_fold(case_id)
    good = verify_oof_student_provenance(case_id, {
        "student_held_out_fold": fold,
        "student_training_folds": [x for x in range(5) if x != fold],
        "student_training_case_ids": ["other-case"],
        "training_manifest_hash": "abc",
    }, expected_manifest_hash="abc")
    assert good["verified"] is True
    leaked = verify_oof_student_provenance(case_id, {
        "student_held_out_fold": fold,
        "student_training_folds": list(range(5)),
        "student_training_case_ids": [case_id],
        "training_manifest_hash": "wrong",
    }, expected_manifest_hash="abc")
    assert leaked["verified"] is False
    assert "case_present_in_student_training_manifest" in leaked["reasons"]


def test_formal_target_contract_is_exactly_373():
    root = Path(__file__).resolve().parents[2]
    doc = json.loads((root / "configs" / "student_3d_prompt_target_organs.json").read_text())
    assert len(doc["target_organs"]) == 373
    assert len(set(doc["target_organs"])) == 373


def test_voxtell_soft_target_loader_preserves_probabilities_contract():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "train_voxtell_prompt_student.py").read_text()
    assert "preserve_probabilities=target_type == \"soft\"" in source
    assert "Soft target must contain finite probabilities in [0,1]" in source
    assert "def is_allowed_positive_item" in source
    assert "scoring_schema_version" in source
    assert "target_type == \"soft\"" in source


def test_prompt_student_loader_rejects_legacy_and_hard_c(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "train_voxtell_prompt_student.py"
    spec = importlib.util.spec_from_file_location("train_voxtell_prompt_student_for_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    image = _mask(tmp_path / "image.nii.gz")
    hard = _mask(tmp_path / "hard.nii.gz")
    soft = tmp_path / "soft.nii.gz"
    nib.save(nib.Nifti1Image(np.full((20, 20, 20), 0.5, dtype=np.float32), np.eye(4)), str(soft))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "items": [
            {"case_id": "c", "organ": "legacy", "image": str(image), "mask": str(hard), "prompt": "legacy", "grade": "A", "training_weight": 1.0},
            {"case_id": "c", "organ": "hard_c", "image": str(image), "mask": str(hard), "prompt": "hard c", "grade": "C", "training_weight": 0.1, "scoring_schema_version": "autolabel_core_v3", "target_type": "hard"},
            {"case_id": "c", "organ": "soft_c", "image": str(image), "mask": str(soft), "probability_mask_path": str(soft), "prompt": "soft c", "grade": "C", "training_weight": 0.1, "scoring_schema_version": "autolabel_core_v3", "target_type": "soft"},
            {"case_id": "c", "organ": "hard_b", "image": str(image), "mask": str(hard), "prompt": "hard b", "grade": "B", "training_weight": 0.5, "scoring_schema_version": "autolabel_core_v3", "target_type": "hard"},
        ]
    }), encoding="utf-8")

    rows = module.load_manifest(manifest)
    organs = {row["organ"] for row in rows}
    assert "legacy" not in organs
    assert "hard_c" not in organs
    assert {"soft_c", "hard_b"} <= organs


def test_longtail_critic_pairs_keep_source_case_in_one_fold(tmp_path):
    source = _mask(tmp_path / "seed.nii.gz")
    dataset_path = tmp_path / "pairs.json"
    result = build_pairwise_dataset([{
        "case_id": "case-1", "organ": "liver", "mask_path": str(source),
        "grade": "A", "independent_family_count": 2,
    }], dataset_path)
    assert result["num_pairs"] >= 6
    pairs = json.loads(dataset_path.read_text())["pairs"]
    assert len({row["fold"] for row in pairs}) == 1
    trained = train_pairwise_ranker(dataset_path, tmp_path / "critic.json")
    assert trained["status"] == "success"
    model = json.loads((tmp_path / "critic.json").read_text())
    assert model["output_semantics"] == "structural_corruption_probability_not_segmentation_accuracy"



def test_candidate_identity_direct_and_shapekit_fallback_alias_are_valid():
    from cli_anything.medai.core.multimodel_loop import _candidate_identity_contract

    taxonomy = {
        "organs": {
            "liver": {"comparison_family": "whole_organ", "parent_ids": []},
        }
    }
    alias_config = {"models": {}}
    direct = _candidate_identity_contract(
        taxonomy=taxonomy,
        alias_config=alias_config,
        organ="liver",
        model_key="epai_20250421",
        alias_match="direct",
    )
    fallback = _candidate_identity_contract(
        taxonomy=taxonomy,
        alias_config=alias_config,
        organ="liver",
        model_key="epai_20250421",
        alias_match="post_shapekit_missing_fallback:direct",
    )
    assert direct["identity_status"] == "valid"
    assert direct["mapping_type"] == "exact_synonym"
    assert fallback["identity_status"] == "valid"
    assert fallback["mapping_type"] == "exact_synonym"
    assert fallback["alias_match"] == "post_shapekit_missing_fallback:direct"


def test_candidate_identity_missing_mask_is_not_identity_mismatch():
    from cli_anything.medai.core.multimodel_loop import _candidate_identity_contract

    taxonomy = {
        "organs": {
            "liver": {"comparison_family": "whole_organ", "parent_ids": []},
        }
    }
    result = _candidate_identity_contract(
        taxonomy=taxonomy,
        alias_config={"models": {}},
        organ="liver",
        model_key="epai_20250421",
        alias_match="missing",
    )
    assert result["identity_status"] == "missing_candidate"
    assert result["mapping_type"] == "missing_mask"
    assert result["identity_mismatch_reasons"] == ["candidate_mask_missing"]


def test_whole_kidney_alias_does_not_satisfy_left_kidney_candidate():
    from cli_anything.medai.core.multimodel_loop import _candidate_identity_contract

    taxonomy = {
        "organs": {
            "kidney_left": {"comparison_family": "whole_organ", "parent_ids": []},
        }
    }
    alias_config = {
        "models": {
            "official_voxtell_pretrained": {
                "local_to_global": {},
                "mapping_types": {},
            }
        }
    }
    result = _candidate_identity_contract(
        taxonomy=taxonomy,
        alias_config=alias_config,
        organ="kidney_left",
        model_key="official_voxtell_pretrained",
        alias_match="missing",
    )
    assert result["identity_status"] == "missing_candidate"
    assert result["resolved_canonical_id"] == "kidney_left"


def test_shapekit_fallback_gap_distinguishes_usable_original(tmp_path):
    from cli_anything.medai.core.multimodel_loop import _build_gap_rows

    mask = _mask(tmp_path / "liver.nii.gz")
    rows = _build_gap_rows(
        case_id="case-1",
        ct=tmp_path / "ct.nii.gz",
        organs=["liver", "pancreas", "spleen"],
        selection_rows=[{"organ": "spleen", "reason": "no candidate masks produced"}],
        selected_metadata=[
            {
                "organ": "liver",
                "mask_path": str(mask),
                "shapekit_status": "fallback_original",
                "shapekit_reason": "ShapeKit crashed",
                "candidate_count": 1,
                "candidate_models": ["epai_20250421"],
                "selection_method": "single_teacher_default",
                "selection_status": "selected",
                "selected_candidate_qc_status": "pass",
            },
            {
                "organ": "pancreas",
                "mask_path": str(tmp_path / "missing.nii.gz"),
                "shapekit_status": "failed",
                "candidate_count": 1,
                "candidate_models": ["vsmtrans"],
                "selection_method": "single_teacher_default",
                "selection_status": "selected",
            },
        ],
    )
    by_organ = {(row["organ"], row["gap_type"]): row for row in rows}
    assert by_organ[("liver", "shapekit_failed_but_original_usable")]["original_mask_usable"] is True
    assert by_organ[("pancreas", "shapekit_failed_and_no_usable_mask")]["original_mask_usable"] is False
    assert ("spleen", "missing_final_pseudo_label") in by_organ


def test_single_teacher_soft_c_is_low_weight_not_promoted():
    result = score_evidence_record({
        "identity_status": "valid",
        "selected_candidate_qc_status": "pass",
        "independent_family_count": 1,
        "probability_mask_path": "/tmp/probability.nii.gz",
        "evidence_details": {
            "ct_support": {"score": 0.8, "status": "available"},
            "anatomy_plausibility": {"score": 1.0, "status": "available"},
            "family_consensus": {"score": None, "status": "unavailable"},
            "perturbation_stability": {"score": None, "status": "unavailable"},
            "cross_round_stability": {"score": None, "status": "unavailable"},
            "loo_model_reliability": {"score": None, "status": "unavailable"},
            "teacher_student_oof": {"score": None, "status": "unavailable"},
        },
    })
    assert result["grade"] == "C"
    assert result["target_type"] == "soft"
    assert result["training_weight"] == 0.1
    assert result["distillation_eligible"] is True
    assert result["grade_cap"] == "C"
