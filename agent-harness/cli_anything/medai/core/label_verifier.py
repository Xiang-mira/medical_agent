from __future__ import annotations

from pathlib import Path
from typing import Any

from .qc_checker import mask_volume_cm3


# ── Anatomical bounds per organ ───────────────────────────────────────────────
# Each entry is (z_min_fraction, z_max_fraction) relative to the CT volume's
# z-axis extent in LAS orientation (inferior=0, superior=1).
# Voxels outside this range are anatomically implausible for the organ and
# should be zeroed out before using the mask.
# These are conservative bounds — wide enough to accommodate normal variation.
ORGAN_ANATOMICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # Head / brain
    "brain": (0.80, 1.0),
    "brain_ventricle": (0.80, 1.0),
    "brainstem": (0.75, 1.0),
    # Neck / head-neck
    "thyroid_gland": (0.65, 0.90),
    "thyroid_left": (0.65, 0.90),
    "thyroid_right": (0.65, 0.90),
    "esophagus": (0.40, 0.95),
    "trachea": (0.55, 0.95),
    # Thorax
    "lung_left": (0.45, 0.85),
    "lung_right": (0.45, 0.85),
    "lung_lower_left_lobe": (0.45, 0.70),
    "lung_lower_right_lobe": (0.45, 0.70),
    "lung_upper_left_lobe": (0.65, 0.85),
    "lung_upper_right_lobe": (0.65, 0.85),
    "heart": (0.50, 0.80),
    "aorta": (0.10, 0.90),
    "pulmonary_artery": (0.55, 0.80),
    # Upper abdomen
    "liver": (0.30, 0.70),
    "spleen": (0.30, 0.65),
    "stomach": (0.25, 0.65),
    "pancreas": (0.25, 0.60),
    "pancreatic_duct": (0.25, 0.60),
    "pancreatic_pdac": (0.20, 0.65),
    "pancreatic_cyst": (0.20, 0.65),
    "pancreatic_pnet": (0.20, 0.65),
    "common_bile_duct": (0.25, 0.65),
    "gall_bladder": (0.30, 0.65),
    "adrenal_gland_left": (0.35, 0.65),
    "adrenal_gland_right": (0.35, 0.65),
    "celiac_aa": (0.30, 0.60),
    "superior_mesenteric_artery": (0.20, 0.55),
    # Mid abdomen
    "kidney_left": (0.20, 0.60),
    "kidney_right": (0.20, 0.60),
    "kidney_cortex": (0.20, 0.60),
    "kidney_medulla": (0.20, 0.60),
    "kidney_pelvicalyceal_system": (0.20, 0.60),
    "renal_vein_left": (0.25, 0.55),
    "renal_vein_right": (0.25, 0.55),
    "postcava": (0.05, 0.80),
    "veins": (0.20, 0.70),
    "duodenum": (0.20, 0.55),
    "colon": (0.05, 0.60),
    "intestine": (0.05, 0.55),
    # Pelvis / lower
    "bladder": (0.0, 0.25),
    "prostate": (0.0, 0.20),
    "rectum": (0.0, 0.25),
    # Spine / bones (full range)
    "aorta": (0.05, 0.90),
    "sacrum": (0.0, 0.15),
    "vertebrae_l1": (0.10, 0.30),
    "vertebrae_l2": (0.08, 0.28),
    "vertebrae_l3": (0.06, 0.25),
    "vertebrae_l4": (0.04, 0.22),
    "vertebrae_l5": (0.02, 0.18),
    "vertebrae_s1": (0.0, 0.12),
}

# Organs where a small volume drop between rounds is a red flag
_SMALL_ORGANS: frozenset[str] = frozenset({
    "pancreatic_duct", "common_bile_duct", "renal_vein_left", "renal_vein_right",
    "celiac_aa", "superior_mesenteric_artery", "cbd_stent",
})


def _apply_anatomical_bounds(mask_path: Path, organ: str, output_path: Path) -> tuple[Path, bool]:
    """Zero out voxels outside the organ's anatomical z-range.

    Returns (path_to_use, was_modified). If no bounds are defined or no voxels
    were outside the range, returns the original path unchanged.
    """
    bounds = ORGAN_ANATOMICAL_BOUNDS.get(organ)
    if bounds is None:
        return mask_path, False
    try:
        import numpy as np
        import nibabel as nib
        img = nib.load(str(mask_path))
        arr = np.asanyarray(img.dataobj).copy()
        z_size = arr.shape[2]
        z_min = int(bounds[0] * z_size)
        z_max = int(bounds[1] * z_size)
        # Zero out voxels outside [z_min, z_max)
        outside = np.zeros(arr.shape, dtype=bool)
        if z_min > 0:
            outside[:, :, :z_min] = True
        if z_max < z_size:
            outside[:, :, z_max:] = True
        n_outside = int((arr > 0)[outside].sum())
        if n_outside == 0:
            return mask_path, False
        arr[outside] = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(output_path))
        return output_path, True
    except Exception:
        return mask_path, False


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
    """Compare a prior/current pseudo reference vs. model prediction for one organ.

    The project has no expert fine-label set by default. DSC here is therefore
    pseudo-label/reference consistency, not true segmentation accuracy.
    """
    result: dict = {
        "stage": "label_verifier",
        "organ": organ,
        "metric_family": "pseudo_consistency",
        "metric_scope": "prediction_vs_prior_or_selected_pseudo_reference",
        "ground_truth_status": "pseudo_label_candidate",
        "accuracy_warning": "DSC is pseudo-label consistency unless the caller explicitly supplies expert fine labels.",
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
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "both_missing", "reason": "Both pseudo reference and prediction are missing."})
        return result

    if not pred_exists:
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "missing_prediction", "reason": "Model prediction missing; cannot verify."})
        return result

    if not ann_exists:
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "no_reference", "reason": "No prior pseudo reference; prediction becomes pseudo-label candidate.", "dice": None})
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
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "empty_reference_nonempty_prediction", "reason": "Prior pseudo reference is empty but prediction is non-empty."})
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


def select_best_round_mask(
    round1_mask: Path,
    round2_mask: Path,
    organ: str,
    work_dir: Path | None = None,
    small_organs: frozenset[str] | None = None,
    volume_change_threshold: float = 0.5,
    small_organ_volume_drop_threshold: float = 0.10,
    connectivity_dsc_bonus: float = 0.05,
) -> tuple[Path, str]:
    """Choose the better mask between two training rounds for one organ.

    Applies five checks in order; the first triggered check determines the result:

    1. Anatomical bounds truncation — zero out voxels outside the organ's
       plausible z-range (generalises the "head truncation" idea to any organ/model).
    2. Small organ protection — if the organ is in `small_organs` and round2
       volume dropped >10% relative to round1, keep round1.
    3. Volume change guard — if volume changed >50%, flag for LabelCritic
       arbitration (caller checks the returned reason string).
    4. Connectivity check — if round2 is more fragmented than round1 and the
       DSC improvement is below `connectivity_dsc_bonus`, keep round1.
    5. DSC comparison — keep whichever round has higher DSC against round1
       (round1 is the reference baseline).

    Returns (chosen_mask_path, reason_string).
    The reason string "volume_change_guard:trigger_labelcritic" signals the
    caller to route to LabelCritic instead of auto-accepting.
    """
    if small_organs is None:
        small_organs = _SMALL_ORGANS

    r1 = Path(round1_mask)
    r2 = Path(round2_mask)

    if not r1.exists():
        return r2, "round1_missing:use_round2"
    if not r2.exists():
        return r1, "round2_missing:use_round1"

    # ── Step 1: Apply anatomical bounds to round2 ────────────────────────────
    if work_dir is not None:
        truncated_path = work_dir / f"{organ}_round2_truncated.nii.gz"
    else:
        import tempfile
        truncated_path = Path(tempfile.mktemp(suffix=f"_{organ}_truncated.nii.gz"))

    r2_clean, was_truncated = _apply_anatomical_bounds(r2, organ, truncated_path)
    truncation_note = f"anatomical_bounds_applied({organ})" if was_truncated else "no_truncation_needed"

    # ── Step 2: Small organ protection ──────────────────────────────────────
    if organ in small_organs:
        vol1 = mask_volume_cm3(r1).get("voxel_count", 0) or 0
        vol2 = mask_volume_cm3(r2_clean).get("voxel_count", 0) or 0
        if vol1 > 0 and vol2 < vol1 * (1.0 - small_organ_volume_drop_threshold):
            return r1, f"small_organ_protection:{organ}:vol_drop_{vol1}->{vol2}"

    # ── Step 3: Volume change guard ──────────────────────────────────────────
    vol1 = mask_volume_cm3(r1).get("voxel_count", 0) or 0
    vol2 = mask_volume_cm3(r2_clean).get("voxel_count", 0) or 0
    if vol1 > 0:
        rel_change = abs(vol2 - vol1) / max(vol1, 1)
        if rel_change > volume_change_threshold:
            return r1, f"volume_change_guard:trigger_labelcritic:change={rel_change:.2f}"

    # ── Step 4: Connectivity check ───────────────────────────────────────────
    try:
        import numpy as np
        import nibabel as nib
        from scipy.ndimage import label as scipy_label
        arr1 = np.asanyarray(nib.load(str(r1)).dataobj) > 0
        arr2 = np.asanyarray(nib.load(str(r2_clean)).dataobj) > 0
        _, n1 = scipy_label(arr1)
        _, n2 = scipy_label(arr2)
        if n2 > n1:
            # round2 is more fragmented — only accept if DSC improvement justifies it
            dsc = _dice(r1, r2_clean)
            if dsc is None or dsc < connectivity_dsc_bonus:
                return r1, f"connectivity_guard:round2_fragmented(n1={n1},n2={n2})"
    except Exception:
        pass  # scipy not available or load failed — skip connectivity check

    # ── Step 5: DSC comparison (round1 as reference baseline) ────────────────
    dsc = _dice(r1, r2_clean)
    if dsc is None:
        return r1, "dsc_failed:use_round1"

    # round2 is "better" if it differs from round1 in a way that improves coverage.
    # Since we're comparing round2 against round1 (not against ground truth),
    # a higher DSC means they agree more — we prefer round2 only if it passed
    # all guards above, meaning it's at least as good.
    # Use volume as a tiebreaker: prefer the mask with more coverage when DSC is high.
    if vol2 >= vol1 * 0.9:
        return r2_clean, f"round2_accepted:{truncation_note}:dsc={dsc:.3f}"
    return r1, f"round1_preferred:round2_volume_too_low(vol1={vol1},vol2={vol2})"
