"""Multi-teacher label fusion for the E-step.

When two or more teacher candidates produce a mask for the same organ on the
same case, fusing them into a consensus mask is the standard automated way to
denoise multiple noisy annotators toward a pseudo ground truth. This module
provides:

- ``staple``        : SimpleITK STAPLE — estimates each rater's
                      sensitivity/specificity and produces a probabilistic
                      consensus. Does not use external weights (it learns them).
- ``weighted_vote`` : reliability-weighted majority voting (numpy). V2 weights
                      come from leave-one-evidence-family-out estimated
                      reliability, never pseudo-reference mean DSC.
- ``auto``          : try STAPLE, fall back to weighted_vote, then to the
                      single highest-weight candidate.

All inputs are resampled (nearest-neighbour) onto a common reference grid (the
CT when provided, else the candidate with the most foreground) so that teachers
emitting different resolutions can still be fused. The fused result is written
as a uint8 NIfTI and a metadata dict is returned for audit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        p = Path(path)
        st = p.stat()
        return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except Exception:
        return None


def _fusion_cache_key(
    mask_paths: list[str | Path],
    weights: list[float] | None,
    reference_image: str | Path | None,
    method: str,
    vote_threshold: float,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "inputs": [_file_fingerprint(p) for p in mask_paths],
        "weights": [round(float(w), 8) for w in weights] if weights is not None else None,
        "reference": _file_fingerprint(reference_image),
        "method": method,
        "vote_threshold": float(vote_threshold),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], payload


def _load_fusion_cache(cache_meta: Path, output_path: Path, cache_key: str) -> dict[str, Any] | None:
    if not cache_meta.exists() or not output_path.exists():
        return None
    try:
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("fusion_cache_key") != cache_key or meta.get("status") not in {"success", "single"}:
        return None
    meta = dict(meta)
    meta["cache_status"] = "reused_fusion_cache"
    meta["output"] = str(output_path)
    return meta


def _write_fusion_cache(cache_meta: Path, output_path: Path, meta: dict[str, Any], cache_key: str, payload: dict[str, Any]) -> None:
    try:
        cache_meta.parent.mkdir(parents=True, exist_ok=True)
        cache_doc = {**meta, "fusion_cache_key": cache_key, "fusion_cache_payload": payload, "output": str(output_path)}
        cache_meta.write_text(json.dumps(cache_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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

    cache_key, cache_payload = _fusion_cache_key(existing, weights, reference_image, method, vote_threshold)
    cache_meta = out.parent / ".fusion_cache" / f"{out.stem}_{cache_key}.json"
    cached = _load_fusion_cache(cache_meta, out, cache_key)
    if cached is not None:
        return cached

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
        _write_fusion_cache(cache_meta, out, meta, cache_key, cache_payload)
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
    _write_fusion_cache(cache_meta, out, meta, cache_key, cache_payload)
    return meta
