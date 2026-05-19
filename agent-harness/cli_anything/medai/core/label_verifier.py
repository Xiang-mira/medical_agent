from __future__ import annotations

from pathlib import Path

from .qc_checker import mask_volume_cm3


def _dice(mask_a: Path, mask_b: Path) -> float | None:
    try:
        import numpy as np
        import nibabel as nib
        a_img = nib.load(str(mask_a)); b_img = nib.load(str(mask_b))
        a = np.asanyarray(a_img.dataobj) > 0
        b = np.asanyarray(b_img.dataobj) > 0
        if a.shape != b.shape:
            return None
        intersection = int((a & b).sum())
        total = int(a.sum()) + int(b.sum())
        return round(2 * intersection / total, 6) if total > 0 else 0.0
    except Exception:
        return None


def verify_annotation(
    current_annotation: str | Path | None,
    model_prediction: str | Path | None,
    organ: str,
    dsc_replace_threshold: float = 0.0,
    dsc_vlm_threshold: float = 0.5,
    dsc_accept_threshold: float = 0.8,
) -> dict:
    """Compare reference/current annotation vs. model prediction for one organ.

    Teacher-task thresholds:
      DSC >= 0.80        -> accept
      0.50 <= DSC < 0.80 -> uncertain / manual sanity check recommended
      DSC < 0.50         -> LabelCritic/VLM review
      DSC = 0 with empty current annotation but non-empty prediction -> replacement candidate
    """
    result: dict = {
        "stage": "label_verifier",
        "organ": organ,
        "dsc_replace_threshold": dsc_replace_threshold,
        "dsc_vlm_threshold": dsc_vlm_threshold,
        "dsc_accept_threshold": dsc_accept_threshold,
    }

    ann_path = Path(current_annotation).resolve() if current_annotation else None
    pred_path = Path(model_prediction).resolve() if model_prediction else None

    ann_exists = ann_path is not None and ann_path.exists()
    pred_exists = pred_path is not None and pred_path.exists()

    result["current_annotation"] = str(ann_path) if ann_path else None
    result["model_prediction"] = str(pred_path) if pred_path else None
    result["current_annotation_exists"] = ann_exists
    result["model_prediction_exists"] = pred_exists

    if not pred_exists and not ann_exists:
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "both_missing", "reason": "Both reference and prediction are missing."})
        return result

    if not pred_exists:
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "missing_prediction", "reason": "Model prediction missing; cannot verify."})
        return result

    if not ann_exists:
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "no_reference", "reason": "No current/reference annotation; prediction becomes candidate.", "dice": None})
        return result

    dice = _dice(ann_path, pred_path)
    result["dice"] = dice

    ann_vol = mask_volume_cm3(ann_path)
    pred_vol = mask_volume_cm3(pred_path)
    result["current_voxels"] = ann_vol.get("voxel_count")
    result["prediction_voxels"] = pred_vol.get("voxel_count")

    if dice is None:
        result.update({"status": "failed", "decision": "review_queue", "quality_bucket": "dice_failed", "reason": "DSC computation failed; check read error or shape mismatch."})
        return result

    pred_nonempty = (pred_vol.get("voxel_count") or 0) > 10
    ann_empty = (ann_vol.get("voxel_count") or 0) <= 10
    ann_nonempty = not ann_empty

    if dice == 0.0 and pred_nonempty and ann_empty:
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "empty_reference_nonempty_prediction", "reason": "Current/reference annotation is empty but prediction is non-empty."})
    elif dice == 0.0 and pred_nonempty and ann_nonempty:
        result.update({"status": "warning", "decision": "send_to_vlm_label_expert", "quality_bucket": "critical_low_dice", "reason": "DSC=0 and both masks are non-empty; requires LabelCritic/VLM comparison."})
    elif dice < dsc_vlm_threshold:
        result.update({"status": "warning", "decision": "send_to_vlm_label_expert", "quality_bucket": "low_dice", "reason": f"DSC={dice} < {dsc_vlm_threshold}; send to LabelCritic/VLM."})
    elif dice < dsc_accept_threshold:
        result.update({"status": "warning", "decision": "uncertain_manual_check", "quality_bucket": "moderate_dice", "reason": f"{dsc_vlm_threshold} <= DSC={dice} < {dsc_accept_threshold}; keep best candidate but queue for sanity check."})
    else:
        result.update({"status": "success", "decision": "accept", "quality_bucket": "high_dice", "reason": f"DSC={dice} >= {dsc_accept_threshold}; accept."})

    return result


def verify_case(
    segmentation_folder: str | Path,
    prediction_folder: str | Path,
    organs: list[str],
    dsc_replace_threshold: float = 0.0,
    dsc_vlm_threshold: float = 0.5,
    dsc_accept_threshold: float = 0.8,
) -> dict:
    """Run label_verifier on multiple organs for one case."""
    seg_dir = Path(segmentation_folder).resolve()
    pred_dir = Path(prediction_folder).resolve()
    results = []
    for organ in organs:
        ann = seg_dir / f"{organ}.nii.gz"
        pred = pred_dir / f"{organ}.nii.gz"
        r = verify_annotation(ann if ann.exists() else None, pred if pred.exists() else None, organ, dsc_replace_threshold, dsc_vlm_threshold, dsc_accept_threshold)
        results.append(r)

    needs_vlm = [r for r in results if r.get("decision") == "send_to_vlm_label_expert"]
    needs_replace = [r for r in results if r.get("decision") == "auto_replace_candidate"]
    uncertain = [r for r in results if r.get("decision") == "uncertain_manual_check"]
    accepted = [r for r in results if r.get("decision") == "accept"]

    return {
        "stage": "label_verifier_case",
        "segmentation_folder": str(seg_dir),
        "prediction_folder": str(pred_dir),
        "organs_checked": organs,
        "dsc_vlm_threshold": dsc_vlm_threshold,
        "dsc_accept_threshold": dsc_accept_threshold,
        "num_accepted": len(accepted),
        "num_vlm_needed": len(needs_vlm),
        "num_uncertain_manual_check": len(uncertain),
        "num_replace_candidates": len(needs_replace),
        "organ_results": results,
        "vlm_organs": [r["organ"] for r in needs_vlm],
        "uncertain_organs": [r["organ"] for r in uncertain],
        "replace_candidate_organs": [r["organ"] for r in needs_replace],
    }
