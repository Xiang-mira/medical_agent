"""Multi-teacher label fusion for the E-step.

When two or more teacher candidates produce a mask for the same organ on the
same case, fusing them into a consensus mask is the standard automated way to
denoise multiple noisy annotators toward a pseudo ground truth. This module
provides:

- ``staple``        : SimpleITK STAPLE — estimates each rater's
                      sensitivity/specificity and produces a probabilistic
                      consensus. Does not use external weights (it learns them).
- ``weighted_vote`` : reliability-weighted majority voting (numpy). Weights
                      come from OrganModelPerformance mean DSC per (organ, model).
- ``auto``          : try STAPLE, fall back to weighted_vote, then to the
                      single highest-weight candidate.

All inputs are resampled (nearest-neighbour) onto a common reference grid (the
CT when provided, else the candidate with the most foreground) so that teachers
emitting different resolutions can still be fused. The fused result is written
as a uint8 NIfTI and a metadata dict is returned for audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_sitk():
    try:
        import SimpleITK as sitk
    except Exception:  # pragma: no cover - environment without SimpleITK
        return None
    return sitk


def _same_geometry(a, b, tol: float = 1e-4) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    for av, bv in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(av - bv) > tol:
            return False
    for av, bv in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(av - bv) > tol:
            return False
    return True


def _resample_to(sitk, mask, reference):
    if _same_geometry(mask, reference):
        return mask
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(mask)


def fuse_candidate_masks(
    mask_paths: list[str | Path],
    output_path: str | Path,
    weights: list[float] | None = None,
    reference_image: str | Path | None = None,
    method: str = "auto",
    vote_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fuse same-organ binary masks into one consensus mask written to disk.

    Returns a metadata dict with at least: status, method, n_inputs,
    fused_voxels, output. ``status`` is ``success`` when a fused mask is
    written, ``single`` when only one usable input exists (it is copied), or
    ``failed`` with a ``reason``.
    """
    out = Path(output_path)
    paths = [Path(p) for p in mask_paths]
    existing = [p for p in paths if p.exists()]
    meta: dict[str, Any] = {
        "stage": "label_fusion",
        "requested_method": method,
        "n_inputs": len(existing),
        "inputs": [str(p) for p in existing],
        "output": str(out),
    }
    if not existing:
        meta.update({"status": "failed", "reason": "no input masks exist"})
        return meta

    sitk = _load_sitk()
    if sitk is None:
        meta.update({"status": "failed", "reason": "SimpleITK not available"})
        return meta

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        images = [sitk.ReadImage(str(p), sitk.sitkUInt8) for p in existing]
    except Exception as exc:  # pragma: no cover - corrupt mask
        meta.update({"status": "failed", "reason": f"cannot read masks: {exc}"})
        return meta

    # Binarise (any positive label -> 1) so multi-label maps fuse on this organ.
    images = [sitk.BinaryThreshold(im, 1, 255, 1, 0) for im in images]

    # Choose reference grid.
    ref = None
    if reference_image is not None and Path(reference_image).exists():
        try:
            ref = sitk.ReadImage(str(reference_image))
        except Exception:
            ref = None
    if ref is None:
        ref = max(images, key=lambda im: int(sitk.GetArrayViewFromImage(im).sum()))
    images = [_resample_to(sitk, im, ref) for im in images]

    if len(images) == 1:
        sitk.WriteImage(images[0], str(out))
        arr = sitk.GetArrayViewFromImage(images[0])
        meta.update({"status": "single", "method": "single_input", "fused_voxels": int((arr > 0).sum())})
        return meta

    if weights is not None and len(weights) == len(existing):
        meta["weights"] = [round(float(w), 6) for w in weights]

    fused = None
    used_method = None

    if method in ("auto", "staple"):
        try:
            prob = sitk.STAPLE(images, 1.0)  # foregroundValue=1 -> probability map
            fused = sitk.BinaryThreshold(prob, vote_threshold, 1.0001, 1, 0)
            fused = sitk.Cast(fused, sitk.sitkUInt8)
            used_method = "staple"
        except Exception as exc:
            meta["staple_error"] = str(exc)
            fused = None

    if fused is None and method in ("auto", "staple", "weighted_vote"):
        import numpy as np

        ws = weights if (weights and len(weights) == len(images)) else [1.0] * len(images)
        total_w = float(sum(ws)) or float(len(images))
        acc = None
        for im, w in zip(images, ws):
            a = (sitk.GetArrayFromImage(im) > 0).astype("float32") * float(w)
            acc = a if acc is None else acc + a
        vote = (acc / total_w) >= vote_threshold
        fused = sitk.GetImageFromArray(vote.astype("uint8"))
        fused.CopyInformation(images[0])
        used_method = "weighted_vote"

    if fused is None:  # pragma: no cover - defensive
        meta.update({"status": "failed", "reason": "no fusion method produced a result"})
        return meta

    sitk.WriteImage(fused, str(out))
    fused_voxels = int((sitk.GetArrayViewFromImage(fused) > 0).sum())
    meta.update({"status": "success", "method": used_method, "fused_voxels": fused_voxels})
    return meta
