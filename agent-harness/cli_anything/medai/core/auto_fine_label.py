from __future__ import annotations

from pathlib import Path
from typing import Any

from .auto_label_core import ACCEPTED_SCORING_SCHEMA_VERSIONS, SCORING_SCHEMA_VERSION


GRADE_TO_TRAINING_WEIGHT = {
    "A": 1.0,
    "B": 0.5,
    # Stage-1 hard/provisional C labels are audit-only. C receives a
    # non-zero weight only when AutoLabelCore emits an explicit soft target
    # and the soft-label training path consumes the probability mask.
    "C": 0.0,
    "D": 0.0,
}

GRADE_DEFINITIONS = {
    "A": "High-confidence pseudo-label; safe for direct student training.",
    "B": "Generally reliable pseudo-label; allowed for student training.",
    "C": "Risk-bearing pseudo-label; low-weight training and review recommended.",
    "D": "Rejected for training; retain only as review evidence.",
}

def grade_to_training_weight(grade: str | None) -> float:
    return float(GRADE_TO_TRAINING_WEIGHT.get(str(grade or "D").upper(), 0.0))


def infer_label_maturity(selection: dict[str, Any]) -> str:
    if selection.get("ground_truth_status") == "expert_verified":
        return "L5"
    if selection.get("auto_fine_label_status") == "auto_fine_label_accepted":
        return "L4"
    if selection.get("selection_method") in {"label_critic", "near_identical_agreement", "geometric_teacher_consensus", "single_teacher_default", "candidate_qc_fallback", "label_critic_fallback"}:
        return "L3"
    if selection.get("selected_candidate_qc_status") in {"pass", "review"}:
        return "L2"
    if selection.get("selected_prediction") or selection.get("mask_path") or selection.get("final_mask"):
        return "L1"
    return "L0"


def compute_reliability(selection: dict[str, Any]) -> dict[str, Any]:
    """Return AutoLabelCore v2 evidence reliability, never expert accuracy."""
    from .auto_label_core import score_evidence_record

    incoming_schema = str(selection.get("scoring_schema_version") or "")
    if incoming_schema in ACCEPTED_SCORING_SCHEMA_VERSIONS and selection.get("evidence_confidence") is not None:
        grade = str(selection.get("grade") or "D").upper()
        training_weight = float(selection.get("training_weight") or 0.0)
        status = (
            "auto_fine_label_accepted" if grade in {"A", "B"}
            else "auto_fine_label_candidate" if grade == "C"
            else "unresolved"
        )
        return {
            "evidence_confidence": float(selection.get("evidence_confidence") or 0.0),
            "auto_fine_label_reliability_score": float(selection.get("auto_fine_label_reliability_score", selection.get("evidence_confidence")) or 0.0),
            "evidence_scores": selection.get("evidence_scores", {}),
            "missing_evidence": selection.get("missing_evidence", []),
            "grade": grade,
            "training_weight": training_weight,
            "target_type": selection.get("target_type", "hard" if grade in {"A", "B"} else "provisional" if grade == "C" else "rejected"),
            "decision_status": selection.get("decision_status", "accepted" if grade in {"A", "B"} else "provisional" if grade == "C" else "rejected"),
            "decision_reasons": selection.get("decision_reasons", []),
            "scoring_schema_version": incoming_schema,
            "labelcritic_tiebreak_adjustment_applied": selection.get("labelcritic_tiebreak_adjustment_applied", 0.0),
            "distillation_eligible": selection.get("distillation_eligible", training_weight > 0.0),
            "auto_fine_label_status": status,
            "label_maturity_level": infer_label_maturity({**selection, "auto_fine_label_status": status}),
            "grade_definition": GRADE_DEFINITIONS[grade],
            "accuracy_warning": "Evidence confidence is not expert accuracy or ground-truth DSC.",
        }

    has_mask = bool(selection.get("mask_path") or selection.get("final_mask") or selection.get("mask"))
    record = dict(selection)
    if not has_mask:
        record["identity_status"] = "missing"
        record.setdefault("quality_flags", []).append("missing_final_mask")
    scored = score_evidence_record(record)
    status = (
        "auto_fine_label_accepted" if scored["grade"] in {"A", "B"}
        else "auto_fine_label_candidate" if scored["grade"] == "C"
        else "unresolved"
    )
    return {
        **scored,
        "auto_fine_label_status": status,
        "label_maturity_level": infer_label_maturity({**selection, "auto_fine_label_status": status}),
        "grade_definition": GRADE_DEFINITIONS[scored["grade"]],
        "accuracy_warning": "Evidence confidence is not expert accuracy or ground-truth DSC.",
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
    }


def build_label_passport(selection: dict[str, Any]) -> dict[str, Any]:
    reliability = compute_reliability(selection)
    labelcritic_records = selection.get("labelcritic_records") or selection.get("critic_records") or []
    labelcritic_status = "not_used"
    if labelcritic_records:
        uncertain = any(
            (record.get("decision") or {}).get("winner") == "uncertain"
            for record in labelcritic_records
            if isinstance(record, dict)
        )
        labelcritic_status = "uncertain" if uncertain else "used"
    source_model = selection.get("source_model") or selection.get("selected_model")
    source_type = "student" if str(source_model or "").startswith("student") else "teacher"
    if selection.get("selection_method") == "consensus":
        source_type = "consensus"
    if selection.get("selection_status") == "fallback":
        source_type = "fallback"

    mask_path = selection.get("mask_path") or selection.get("final_mask") or selection.get("mask")
    return {
        "case_id": selection.get("case_id"),
        "organ": selection.get("organ"),
        "mask_path": mask_path,
        "label_maturity_level": reliability["label_maturity_level"],
        "auto_fine_label_status": reliability["auto_fine_label_status"],
        "source_type": source_type,
        "source_model": source_model,
        "candidate_models": selection.get("candidate_models", []),
        "teacher_lineage": selection.get("teacher_lineage") or selection.get("candidate_models", []),
        "route_primary_teacher": selection.get("route_primary_teacher"),
        "route_backup_teachers": selection.get("route_backup_teachers", []),
        "route_competition_teachers": selection.get("route_competition_teachers", []),
        "route_confidence": selection.get("route_confidence"),
        "candidate_mode": selection.get("candidate_mode"),
        "shapekit_status": selection.get("shapekit_status"),
        "candidate_qc_status": selection.get("selected_candidate_qc_status"),
        "candidate_qc_flags": selection.get("selected_candidate_qc_flags", []),
        "labelcritic_status": labelcritic_status,
        "labelcritic_compare_used": selection.get("labelcritic_compare_used"),
        "labelcritic_compare_reason": selection.get("labelcritic_compare_reason"),
        "labelcritic_compare_skipped_reason": selection.get("labelcritic_compare_skipped_reason"),
        "labelcritic_grade_used": selection.get("labelcritic_grade_used"),
        "labelcritic_grade_reason": selection.get("labelcritic_grade_reason"),
        "labelcritic_grade_skipped_reason": selection.get("labelcritic_grade_skipped_reason"),
        "labelcritic_supported": selection.get("labelcritic_supported"),
        "labelcritic_tiebreak_adjustment": selection.get("labelcritic_tiebreak_adjustment", 0.0),
        "labelcritic_tiebreak_adjustment_applied": reliability.get("labelcritic_tiebreak_adjustment_applied", 0.0),
        "teacher_consensus_score": selection.get("teacher_consensus_score"),
        "family_consensus_score": selection.get("family_consensus_score"),
        "ct_support_score": selection.get("ct_support_score"),
        "anatomy_plausibility_score": selection.get("anatomy_plausibility_score"),
        "volume_plausibility_score": selection.get("volume_plausibility_score"),
        "cross_round_stability_score": selection.get("cross_round_stability_score"),
        "perturbation_stability_score": selection.get("perturbation_stability_score"),
        "loo_model_reliability_score": selection.get("loo_model_reliability_score"),
        "student_consistency_score": selection.get("student_consistency_score"),
        "selected_reference_quality_bucket": selection.get("selected_reference_quality_bucket"),
        "auto_fine_label_reliability_score": reliability["auto_fine_label_reliability_score"],
        "estimated_reliability": reliability["evidence_confidence"],
        "evidence_confidence": reliability["evidence_confidence"],
        "evidence_scores": reliability["evidence_scores"],
        "evidence_details": reliability.get("evidence_details", selection.get("evidence_details", {})),
        "available_weight_sum": reliability.get("available_weight_sum"),
        "raw_weighted_score": reliability.get("raw_weighted_score"),
        "normalized_evidence_confidence": reliability.get("normalized_evidence_confidence", reliability["evidence_confidence"]),
        "failed_evidence": reliability.get("failed_evidence", []),
        "missing_evidence": reliability["missing_evidence"],
        "grade_cap": reliability.get("grade_cap"),
        "grade_cap_reason": reliability.get("grade_cap_reason"),
        "reference_provenance": selection.get("reference_provenance"),
        "selected_pseudo_consistency_dice": selection.get("selected_pseudo_consistency_dice", selection.get("selected_dice")),
        "family_consensus_dice": selection.get("family_consensus_dice", selection.get("family_consensus_score")),
        "teacher_student_oof_dice": selection.get("teacher_student_oof_dice"),
        "case_fold": selection.get("case_fold"),
        "student_training_folds": (selection.get("oof_verification") or {}).get("student_training_folds"),
        "student_held_out_fold": (selection.get("oof_verification") or {}).get("student_held_out_fold"),
        "out_of_fold_verified": selection.get("out_of_fold_verified", False),
        "training_manifest_hash": (selection.get("oof_verification") or {}).get("training_manifest_hash"),
        "decision_status": reliability["decision_status"],
        "decision_reasons": reliability["decision_reasons"],
        "target_type": reliability["target_type"],
        "probability_mask_path": selection.get("probability_mask_path"),
        "voxel_uncertainty_path": selection.get("voxel_uncertainty_path"),
        "independent_family_count": selection.get("independent_family_count", 0),
        "family_membership": selection.get("family_membership", {}),
        "conflict_score": selection.get("conflict_score"),
        "winner_margin": selection.get("winner_margin"),
        "penalties": selection.get("penalties", {}),
        "scoring_schema_version": reliability["scoring_schema_version"],
        "grade": reliability["grade"],
        "grade_definition": reliability["grade_definition"],
        "training_weight": reliability["training_weight"],
        "distillation_eligible": reliability["training_weight"] > 0.0 and selection.get("distillation_eligible") is not False,
        "distillation_exclusion_reason": selection.get("distillation_exclusion_reason") or (None if reliability["training_weight"] > 0.0 else "autolabel_core_training_weight_zero"),
        "student_training_priority": selection.get("student_training_priority", reliability["grade"]),
        "label_confidence": selection.get("label_confidence", reliability["auto_fine_label_reliability_score"]),
        "metric_family": selection.get("metric_family", "pseudo_consistency"),
        "requested_canonical_id": selection.get("requested_canonical_id"),
        "source_local_label": selection.get("source_local_label"),
        "resolved_canonical_id": selection.get("resolved_canonical_id"),
        "comparison_family": selection.get("comparison_family"),
        "parent_ids": selection.get("parent_ids", []),
        "mapping_type": selection.get("mapping_type"),
        "mapping_source": selection.get("mapping_source"),
        "identity_status": selection.get("identity_status", "legacy_unverified"),
        "identity_mismatch_reasons": selection.get("identity_mismatch_reasons", []),
        "ground_truth_status": selection.get("ground_truth_status", "machine_generated_candidate"),
        "accuracy_warning": "This passport reports evidence reliability for an auto-generated pseudo-label, not expert accuracy.",
    }


def passport_path_for_mask(mask_path: str | Path) -> Path:
    path = Path(mask_path)
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return path.with_name(f"{name}.label_passport.json")
