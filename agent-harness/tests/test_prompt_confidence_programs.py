from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))

from scripts.build_confidence_formula_report import build_report
from scripts.build_prompt_governance_report import (
    FREEFORM_PROMPT_SAFETY_CONTRACT,
    PROMPT_TAXONOMY,
    build_prompt_taxonomy_rows,
    classify_prompt,
)


def test_prompt_taxonomy_has_three_human_readable_levels() -> None:
    assert set(PROMPT_TAXONOMY) == {
        "AI Researcher Prompt",
        "Clinical Expert Prompt",
        "User Natural Prompt",
    }
    assert "validate_left_right_laterality" in FREEFORM_PROMPT_SAFETY_CONTRACT["validation_steps"]
    assert "fallback_to_fixed_prompt_on_any_failure" in FREEFORM_PROMPT_SAFETY_CONTRACT["validation_steps"]


def test_prompt_classifier_maps_examples_to_expected_levels() -> None:
    assert classify_prompt("segment the liver", "segment the liver", "liver") == ("AI Researcher Prompt", "simple_instruction")
    assert classify_prompt("Segment the liver in the right upper abdomen adjacent to the diaphragm", "segment the liver", "liver") == ("Clinical Expert Prompt", "anatomy_location_prompt")
    assert classify_prompt("Please delineate the liver on this CT volume", "segment the liver", "liver") == ("User Natural Prompt", "natural_language_request")


def test_prompt_taxonomy_rows_are_generated_from_existing_config() -> None:
    rows = build_prompt_taxonomy_rows({
        "target_organs": ["liver"],
        "organ_to_prompt": {"liver": "segment the liver"},
        "prompt_variants": {"liver": [
            "outline the liver",
            "Segment the liver in the right upper abdomen adjacent to the diaphragm",
            "Please delineate the liver on this CT volume",
        ]},
    })
    levels = {row["prompt_level"] for row in rows}
    assert "AI Researcher Prompt" in levels
    assert "Clinical Expert Prompt" in levels
    assert "User Natural Prompt" in levels


def test_confidence_report_states_c_is_not_dice_and_is_per_pair() -> None:
    config = {
        "schema_version": "3.0",
        "missing_evidence_policy": "normalize_available",
        "evidence_weights": {"family_consensus": 0.25, "ct_support": 0.20},
        "thresholds": {"grade_a": 0.85, "grade_b": 0.65, "grade_c": 0.35},
        "training_weights": {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0},
        "soft_labels": {"enabled": True, "c_grade_training_weight": 0.1},
    }
    report = build_report(config, None)
    assert report["c_is_dice"] is False
    assert report["scope"] == "per_case_per_organ_mask_candidate_or_selected_pseudo_label"
    c_row = next(row for row in report["grade_policy"] if row["grade"] == "C")
    assert c_row["training_weight"] == "hard=0.0, soft=0.1"
    assert report["example_breakdown"]["status"] == "no_example_selection_provided"


from cli_anything.medai.core.auto_label_core import score_evidence_record
from cli_anything.medai.core.prompt_governance import (
    build_prompt_validation_context,
    prompt_record,
    validate_generated_prompt_record,
)


def _liver_context() -> dict:
    return build_prompt_validation_context(
        organ_token="liver",
        canonical_name="liver",
        region="right upper abdomen",
        ct_appearance="large solid soft-tissue organ",
        synonyms=["hepatic parenchyma"],
        laterality="none",
        allowed_adjacent_anatomy=["diaphragm", "gallbladder"],
        ct_scan_range="abdomen",
        forbidden_targets=["pancreas", "liver_segment_1"],
    )


def test_freeform_prompt_validation_falls_back_on_schema_identity_and_anatomy_errors() -> None:
    context = _liver_context()
    missing_schema = {"prompt": "segment the liver"}
    out = validate_generated_prompt_record(missing_schema, context, fallback_prompt="segment the liver")
    assert out["validation_status"] == "failed"
    assert out["prompt_source"] == "fallback_fixed"
    assert "schema_missing:organ_token" in out["validation_errors"]

    wrong_identity = prompt_record(
        prompt="segment the pancreas",
        organ_token="pancreas",
        canonical_name="pancreas",
        prompt_source="qwen_generated",
        category="simple_instruction",
    )
    out = validate_generated_prompt_record(wrong_identity, context, fallback_prompt="segment the liver")
    assert out["validation_status"] == "failed"
    assert any("mismatch" in err or "forbidden_target" in err for err in out["validation_errors"])

    hallucinated = prompt_record(
        prompt="segment the liver near the diaphragm and skull",
        organ_token="liver",
        canonical_name="liver",
        prompt_source="qwen_generated",
        category="anatomical_location",
        introduced_anatomy=["skull"],
    )
    out = validate_generated_prompt_record(hallucinated, context, fallback_prompt="segment the liver")
    assert out["validation_status"] == "failed"
    assert "introduced_anatomy_not_allowed:skull" in out["validation_errors"]


def test_freeform_prompt_validation_falls_back_on_laterality_and_scan_range() -> None:
    context = build_prompt_validation_context(
        organ_token="kidney_left",
        canonical_name="left kidney",
        region="left retroperitoneum",
        ct_appearance="bean-shaped renal organ",
        synonyms=["left renal parenchyma"],
        laterality="left",
        allowed_adjacent_anatomy=["spleen"],
        ct_scan_range="abdomen",
        forbidden_targets=["kidney_right"],
    )
    right = prompt_record(
        prompt="segment the right kidney",
        organ_token="kidney_left",
        canonical_name="left kidney",
        prompt_source="qwen_generated",
        category="simple_instruction",
        laterality="right",
    )
    out = validate_generated_prompt_record(right, context, fallback_prompt="segment the left kidney")
    assert out["validation_status"] == "failed"
    assert "laterality_mismatch_expected_left" in out["validation_errors"]

    head = prompt_record(
        prompt="segment the left kidney in the skull field of view",
        organ_token="kidney_left",
        canonical_name="left kidney",
        prompt_source="qwen_generated",
        category="anatomical_location",
        ct_scan_range_assumption="head skull CT",
    )
    out = validate_generated_prompt_record(head, context, fallback_prompt="segment the left kidney")
    assert out["validation_status"] == "failed"
    assert any(err.startswith("ct_scan_range_out_of_scope") for err in out["validation_errors"])


def test_confidence_scoring_uses_available_weight_normalization_hard_fail_and_lc_cap() -> None:
    config = {
        "missing_evidence_policy": "normalize_available",
        "evidence_weights": {"family_consensus": 0.5, "ct_support": 0.5},
        "thresholds": {"grade_a": 0.85, "grade_b": 0.65, "grade_c": 0.35, "severe_family_conflict_dice": 0.30, "labelcritic_tiebreak_cap": 0.03},
        "training_weights": {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0},
        "soft_labels": {"enabled": True, "c_grade_training_weight": 0.1},
        "single_teacher_grade_cap": "C",
        "single_teacher_with_oof_grade_cap": "B",
        "oof_promotion": {},
    }
    scored = score_evidence_record({
        "evidence_details": {
            "family_consensus": {"score": 0.8, "status": "available"},
            "ct_support": {"score": None, "status": "unavailable"},
        },
        "independent_family_count": 2,
        "labelcritic_supported": True,
        "labelcritic_tiebreak_adjustment": 0.5,
    }, config)
    assert scored["normalized_evidence_confidence"] == 0.8
    assert scored["labelcritic_tiebreak_adjustment_applied"] == 0.03
    assert scored["evidence_confidence"] == 0.83

    hard_fail = score_evidence_record({
        "evidence_details": {"family_consensus": {"score": 1.0, "status": "available"}},
        "independent_family_count": 2,
        "quality_flags": ["identity_mismatch"],
    }, config)
    assert hard_fail["grade"] == "D"
    assert hard_fail["training_weight"] == 0.0


def test_training_gate_experiment_plan_has_required_ablation_matrix() -> None:
    from scripts.build_training_gate_experiment_plan import EXPERIMENTS

    names = {item["name"] for item in EXPERIMENTS}
    assert {"all_pseudo_labels", "only_A_B", "A_B_plus_soft_C", "A_only"} <= names



def test_mstep_training_gate_excludes_c_hard_and_unsupported_schema(tmp_path) -> None:
    import numpy as np
    import nibabel as nib
    from cli_anything.medai.core.mstep_runner import build_training_manifest

    root = tmp_path / "annotation_versions"
    case = root / "case_001"
    updated = case / "updated"
    updated.mkdir(parents=True)

    def make_mask(name: str) -> str:
        path = updated / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)), str(path))
        return str(path)

    for organ in ["liver", "pancreas", "kidney_left", "spleen"]:
        make_mask(organ)
    prob = tmp_path / "kidney_left_probability.nii.gz"
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), 0.6, dtype=np.float32), np.eye(4)), str(prob))
    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), str(ct))

    selected = [
            {"organ": "liver", "selected_model": "teacher_a", "grade": "A", "training_weight": 1.0, "target_type": "hard", "identity_status": "valid", "requested_canonical_id": "liver", "resolved_canonical_id": "liver", "scoring_schema_version": "autolabel_core_v2"},
            {"organ": "pancreas", "selected_model": "teacher_a", "grade": "C", "training_weight": 0.1, "target_type": "hard", "identity_status": "valid", "requested_canonical_id": "pancreas", "resolved_canonical_id": "pancreas", "scoring_schema_version": "autolabel_core_v2"},
            {"organ": "kidney_left", "selected_model": "teacher_a", "grade": "C", "training_weight": 0.1, "target_type": "soft", "probability_mask_path": str(prob), "identity_status": "valid", "requested_canonical_id": "kidney_left", "resolved_canonical_id": "kidney_left", "scoring_schema_version": "autolabel_core_v2"},
            {"organ": "spleen", "selected_model": "teacher_a", "grade": "B", "training_weight": 0.5, "target_type": "hard", "identity_status": "valid", "requested_canonical_id": "spleen", "resolved_canonical_id": "spleen", "scoring_schema_version": "legacy"},
        ]
    for row in selected:
        row["ct_path"] = str(ct)
    (case / "selection_metadata.json").write_text(json.dumps({"selected_organs": selected}), encoding="utf-8")
    result = build_training_manifest(root, tmp_path / "manifest.json")
    rows = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {row["organ"] for row in rows} == {"liver", "kidney_left"}
    assert next(row for row in rows if row["organ"] == "kidney_left")["training_gate_decision"] == "include_soft_c"
    exclusions = json.loads(Path(result["training_exclusions"]).read_text(encoding="utf-8"))
    reasons = {row["organ"]: row["distillation_exclusion_reason"] for row in exclusions}
    assert reasons["pancreas"] == "grade_C_requires_soft_probability_target"
    assert reasons["spleen"] == "unsupported_scoring_schema_requires_rescoring"
    summary = result["training_gate_summary"]
    assert summary["num_c_soft_included"] == 1
    assert summary["num_c_hard_or_provisional_excluded"] == 1
    assert summary["num_unsupported_schema_excluded"] == 1
