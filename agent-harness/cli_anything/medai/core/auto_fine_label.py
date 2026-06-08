from __future__ import annotations

from pathlib import Path
from typing import Any


GRADE_TO_TRAINING_WEIGHT = {
    "A": 1.0,
    "B": 0.5,
    "C": 0.1,
    "D": 0.0,
}


def grade_to_training_weight(grade: str | None) -> float:
    return float(GRADE_TO_TRAINING_WEIGHT.get(str(grade or "D").upper(), 0.0))


def infer_label_maturity(selection: dict[str, Any]) -> str:
    if selection.get("ground_truth_status") == "expert_verified":
        return "L5"
    if selection.get("auto_fine_label_status") == "auto_fine_label_accepted":
        return "L4"
    if selection.get("selection_method") in {"label_critic", "single_teacher_default", "candidate_qc_fallback", "label_critic_fallback"}:
        return "L3"
    if selection.get("selected_candidate_qc_status") in {"pass", "review"}:
        return "L2"
    if selection.get("selected_prediction") or selection.get("mask_path") or selection.get("final_mask"):
        return "L1"
    return "L0"


def compute_reliability(selection: dict[str, Any]) -> dict[str, Any]:
    """Conservative reliability estimate for auto fine-label candidates.

    The score is an auditable training/filtering signal, not expert accuracy.
    """
    if not (selection.get("mask_path") or selection.get("final_mask") or selection.get("mask")):
        return {
            "auto_fine_label_reliability_score": 0.0,
            "grade": "D",
            "training_weight": 0.0,
            "auto_fine_label_status": "unresolved",
            "label_maturity_level": "L0",
        }

    score = 0.35
    flags = set(selection.get("review_flags", []) or []) | set(selection.get("quality_flags", []) or [])

    qc_status = selection.get("selected_candidate_qc_status")
    if qc_status == "pass":
        score += 0.25
    elif qc_status == "review":
        score += 0.1
    elif qc_status == "fail":
        score -= 0.35

    shapekit_status = selection.get("shapekit_status")
    if shapekit_status == "success":
        score += 0.15
    elif shapekit_status in {"fallback_original", "unsupported_target", "failed"}:
        score -= 0.1

    method = selection.get("selection_method")
    if method == "label_critic":
        score += 0.15
    elif method == "single_teacher_default":
        score += 0.05
    elif method and "fallback" in str(method):
        score -= 0.15

    dice = selection.get("selected_pseudo_consistency_dice", selection.get("selected_dice"))
    reference_quality_bucket = selection.get("selected_reference_quality_bucket") or selection.get("reference_quality_bucket")
    try:
        if dice is not None and reference_quality_bucket != "empty_reference_nonempty_prediction":
            dice_value = float(dice)
            if dice_value >= 0.8:
                score += 0.1
            elif dice_value < 0.5:
                score -= 0.15
    except Exception:
        pass

    hard_flags = {
        "missing_candidate",
        "missing_final_mask",
        "all_candidates_failed_qc",
        "candidate_qc_fail",
        "empty_mask",
        "shape_mismatch_ct",
        # Confident-bad verdict from the automated VLM acceptance gate (Phase 2):
        # down-weight it so reliability-weighted self-cleaning training (Phase 3)
        # learns less from labels the gate rejected.
        "auto_grade_reject",
    }
    if flags & hard_flags:
        score -= 0.35
    if "selection_fallback" in flags:
        score -= 0.15
    if "shapekit_fallback" in flags:
        score -= 0.1

    score = max(0.0, min(1.0, round(score, 4)))
    if score >= 0.85:
        grade = "A"
        status = "auto_fine_label_accepted"
    elif score >= 0.65:
        grade = "B"
        status = "auto_fine_label_accepted"
    elif score >= 0.35:
        grade = "C"
        status = "auto_fine_label_candidate"
    else:
        grade = "D"
        status = "unresolved"

    return {
        "auto_fine_label_reliability_score": score,
        "grade": grade,
        "training_weight": grade_to_training_weight(grade),
        "auto_fine_label_status": status,
        "label_maturity_level": infer_label_maturity({**selection, "auto_fine_label_status": status}),
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
        "shapekit_status": selection.get("shapekit_status"),
        "candidate_qc_status": selection.get("selected_candidate_qc_status"),
        "candidate_qc_flags": selection.get("selected_candidate_qc_flags", []),
        "labelcritic_status": labelcritic_status,
        "teacher_consensus_score": selection.get("teacher_consensus_score"),
        "anatomy_plausibility_score": selection.get("anatomy_plausibility_score"),
        "volume_plausibility_score": selection.get("volume_plausibility_score"),
        "cross_round_stability_score": selection.get("cross_round_stability_score"),
        "student_consistency_score": selection.get("student_consistency_score"),
        "selected_reference_quality_bucket": selection.get("selected_reference_quality_bucket"),
        "auto_fine_label_reliability_score": reliability["auto_fine_label_reliability_score"],
        "grade": reliability["grade"],
        "training_weight": reliability["training_weight"],
        "metric_family": selection.get("metric_family", "pseudo_consistency"),
        "ground_truth_status": selection.get("ground_truth_status", "machine_generated_candidate"),
        "accuracy_warning": "This passport describes an auto-generated fine-label candidate, not expert ground truth.",
    }


def passport_path_for_mask(mask_path: str | Path) -> Path:
    path = Path(mask_path)
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return path.with_name(f"{name}.label_passport.json")
