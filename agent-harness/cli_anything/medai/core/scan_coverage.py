from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def infer_scan_coverage(ct_path: str | Path) -> dict[str, Any]:
    """Conservative, label-free CT coverage inference.

    The detector intentionally returns unknown rather than guessing.  It uses
    only CT intensities and geometry; annotation folders are never inspected.
    """
    image = nib.as_closest_canonical(nib.load(str(ct_path)))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D CT, got shape={data.shape}")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    body = data > -500
    slice_area = body.sum(axis=(0, 1))
    active = np.flatnonzero(slice_area > max(100, int(0.01 * data.shape[0] * data.shape[1])))
    if active.size:
        first, last = int(active[0]), int(active[-1])
    else:
        first, last = 0, data.shape[2] - 1
    length_mm = float((last - first + 1) * spacing[2])

    # Large internal air regions are a conservative thorax/lung cue.  Air
    # outside the body is excluded by requiring a body neighbourhood.
    internal_air = data < -650
    body_dilated = body.copy()
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes

        envelope = np.zeros_like(body)
        for z in range(body.shape[2]):
            closed = binary_closing(body[:, :, z], iterations=5)
            envelope[:, :, z] = binary_fill_holes(closed)
        internal_air &= envelope
    except Exception:
        internal_air &= body_dilated
    air_by_slice = internal_air.sum(axis=(0, 1))
    thorax_slices = int(np.count_nonzero(air_by_slice > max(200, int(0.005 * data.shape[0] * data.shape[1]))))
    has_thorax = thorax_slices >= 5

    # PanTS inputs are abdominal CTs.  Require a substantial body span rather
    # than declaring abdomen from dataset labels.
    has_abdomen = length_mm >= 120.0
    regions = ["abdomen"] if has_abdomen else []
    if has_thorax:
        regions.append("thorax")
    confidence = "high" if has_abdomen and active.size else "low"
    return {
        "source": "ct_intensity_geometry_v1",
        "uses_annotations": False,
        "scan_coverage": regions or ["unknown"],
        "coverage_regions": regions,
        "body_region": "multi_region" if len(regions) > 1 else (regions[0] if regions else "unknown"),
        "ct_region": regions,
        "confidence": confidence,
        "body_first_slice": first,
        "body_last_slice": last,
        "body_length_mm": round(length_mm, 3),
        "thorax_air_slices": thorax_slices,
        "has_abdomen_coverage": has_abdomen,
        "has_pelvis_coverage": False,
        "has_thorax_coverage": has_thorax,
        "has_head_coverage": False,
        "has_extremity_coverage": False,
        "unknown_regions": ["pelvis", "head_neck", "extremity"],
    }
