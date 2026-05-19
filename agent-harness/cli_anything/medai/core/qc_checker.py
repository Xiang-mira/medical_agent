from __future__ import annotations

from pathlib import Path
from typing import Any
from .radthinking import compare_temporal_masks

DEFAULT_CORE_ORGANS = ["liver", "pancreas", "aorta", "postcava"]


def _issue(kind: str, severity: str, message: str, action: str, **extra) -> dict:
    d = {"type": kind, "severity": severity, "message": message, "suggested_action": action}
    d.update(extra)
    return d


def mask_volume_cm3(mask_path: str | Path | None) -> dict:
    if mask_path is None:
        return {"status": "missing", "voxel_count": 0, "volume_cm3": 0.0}
    try:
        import numpy as np, nibabel as nib
        p = Path(mask_path)
        if not p.exists():
            return {"status": "missing", "mask": str(p), "voxel_count": 0, "volume_cm3": 0.0}
        img = nib.load(str(p)); arr = np.asanyarray(img.dataobj) > 0
        vox = int(arr.sum()); z = img.header.get_zooms()[:3]
        return {"status": "success", "mask": str(p), "voxel_count": vox, "volume_cm3": round(float(vox * z[0] * z[1] * z[2] / 1000.0), 6)}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "mask": str(mask_path)}


def check_segmentation_quality(case_segmentation_folder: str | Path | None, expected_organs: list[str] | None = None, min_nonzero_voxels: int = 10) -> dict:
    if expected_organs is None: expected_organs = DEFAULT_CORE_ORGANS
    if case_segmentation_folder is None:
        return {"stage": "qc_segmentation", "status": "failed", "issues": [_issue("missing_folder", "high", "No segmentation folder was provided.", "rerun_inference_or_human_review")], "folder": None}
    folder = Path(case_segmentation_folder).resolve()
    issues = []; stats: dict[str, Any] = {}
    if not folder.exists():
        return {"stage": "qc_segmentation", "status": "failed", "issues": [_issue("missing_folder", "high", f"Segmentation folder does not exist: {folder}", "rerun_inference_or_human_review")], "folder": str(folder)}
    masks = sorted(folder.glob("*.nii.gz"))
    if not masks: issues.append(_issue("no_masks", "high", "No NIfTI masks were found.", "rerun_inference_or_human_review"))
    for organ in expected_organs:
        p = folder / f"{organ}.nii.gz"
        if not p.exists():
            issues.append(_issue("missing_core_organ", "medium", f"Expected core organ mask is missing: {organ}.nii.gz", "continue_with_warning_or_review", organ=organ))
            continue
        info = mask_volume_cm3(p); stats[organ] = info
        if info.get("status") == "failed": issues.append(_issue("mask_read_error", "high", f"Failed to read {organ}.nii.gz", "human_review", organ=organ, details=info))
        elif int(info.get("voxel_count") or 0) < min_nonzero_voxels: issues.append(_issue("empty_or_tiny_mask", "high", f"{organ}.nii.gz is empty or too small.", "rerun_or_review", organ=organ, details=info))
    high = any(i["severity"] == "high" for i in issues)
    return {"stage": "qc_segmentation", "status": "failed" if high else "warning" if issues else "success", "folder": str(folder), "num_masks": len(masks), "issues": issues, "mask_stats": stats}


def check_trace_quality(trace: dict | None) -> dict:
    if not trace:
        return {"stage": "qc_trace", "status": "failed", "issues": [_issue("missing_trace", "high", "Trace was not generated.", "review_queue")]}
    issues = []
    rt = trace.get("radthinking_trace", {})
    if not rt.get("observation"): issues.append(_issue("missing_observation", "medium", "Observation step is missing.", "review_queue"))
    if not rt.get("temporal_comparison"): issues.append(_issue("missing_temporal", "medium", "Temporal comparison step is missing.", "review_queue"))
    if not rt.get("clinical_context"): issues.append(_issue("missing_context", "medium", "Clinical context step is missing.", "review_queue"))
    conclusion = rt.get("diagnostic_conclusion") or {}
    if conclusion.get("status") == "not_provided": issues.append(_issue("missing_ground_truth", "medium", "No pathology/follow-up conclusion was provided.", "mark_trace_incomplete"))
    temp = rt.get("temporal_comparison") or {}
    label = temp.get("temporal_label")
    ratio = temp.get("volume_ratio")
    if ratio is not None and (ratio > 2.0 or ratio < 0.5):
        issues.append(_issue("extreme_temporal_ratio", "high", f"Extreme volume ratio detected: {ratio}.", "human_review", volume_ratio=ratio))
    elif label in {"NEW", "GROWING", "SHRINKING", "RESOLVED"}:
        issues.append(_issue("temporal_change_requires_attention", "medium", f"Temporal comparison produced label {label}.", "build_trace_and_review_queue", temporal_label=label, volume_ratio=ratio))
    high = any(i["severity"] == "high" for i in issues)
    return {"stage": "qc_trace", "status": "failed" if high else "warning" if issues else "success", "issues": issues}


def merge_qc_results(*results: dict | None) -> dict:
    issues = []
    stages = []
    for r in results:
        if not r: continue
        stages.append({"stage": r.get("stage"), "status": r.get("status")})
        issues.extend(r.get("issues", []) or [])
    high = any(i.get("severity") == "high" for i in issues)
    return {"stage": "qc_merged", "status": "failed" if high else "warning" if issues else "success", "issues": issues, "stage_statuses": stages}
