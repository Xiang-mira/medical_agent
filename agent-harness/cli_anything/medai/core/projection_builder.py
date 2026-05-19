from __future__ import annotations

from pathlib import Path
from typing import Any


def _build_slice_projection(
    ct_image: str | Path,
    mask_a: str | Path | None,
    mask_b: str | Path | None,
    output_folder: str | Path,
    organ: str = "organ",
    views: list[str] | None = None,
    strict_alignment: bool = False,
) -> dict[str, Any]:
    """Fallback projection using mask-centered slices.

    This is intentionally no longer the default for VLM comparison when
    LabelCritic is available.  It is kept as a lightweight fallback for manual
    debugging or environments without LabelCritic dependencies.
    """
    if views is None:
        views = ["axial", "coronal"]

    output_dir = Path(output_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import nibabel as nib
    except ImportError:
        return {"stage": "projection_builder", "status": "failed",
                "reason": "nibabel not installed"}

    try:
        from PIL import Image as PILImage
        _has_pil = True
    except ImportError:
        _has_pil = False

    ct_path = Path(ct_image).resolve()
    if not ct_path.exists():
        return {"stage": "projection_builder", "status": "failed",
                "reason": f"CT image not found: {ct_path}"}

    ct_img = nib.load(str(ct_path))
    ct_data = np.asanyarray(ct_img.dataobj).astype(float)

    shape_warnings: list[str] = []

    def _load_mask(p, label: str) -> "np.ndarray | None":
        if p is None:
            return None
        pp = Path(p).resolve()
        if not pp.exists():
            return None
        mask_img = nib.load(str(pp))
        arr = np.asanyarray(mask_img.dataobj) > 0
        if arr.shape != ct_data.shape:
            shape_warnings.append(
                f"{label} shape {arr.shape} != CT shape {ct_data.shape}; overlay may be misaligned"
            )
        if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-3):
            shape_warnings.append(
                f"{label} affine does not match CT affine; voxel registration may be off"
            )
        return arr

    mask_a_arr = _load_mask(mask_a, "mask_a")
    mask_b_arr = _load_mask(mask_b, "mask_b")

    if strict_alignment and shape_warnings:
        return {
            "stage": "projection_builder",
            "status": "failed",
            "organ": organ,
            "reason": "strict_alignment=True: mask/CT registration mismatch detected",
            "shape_warnings": shape_warnings,
        }

    ct_win = np.clip(ct_data, -150, 250)
    ct_norm = ((ct_win + 150) / 400 * 255).astype(np.uint8)

    def _get_slice(arr3d, view, idx=None):
        s = arr3d.shape
        if view == "axial":
            mid = s[2] // 2 if idx is None else idx
            return arr3d[:, :, mid]
        elif view == "coronal":
            mid = s[1] // 2 if idx is None else idx
            return arr3d[:, mid, :]
        else:
            mid = s[0] // 2 if idx is None else idx
            return arr3d[mid, :, :]

    def _find_mask_center(mask_arr, view):
        if mask_arr is None:
            return None
        nz = np.nonzero(mask_arr)
        if len(nz[0]) == 0:
            return None
        if view == "axial":
            return int(np.median(nz[2]))
        elif view == "coronal":
            return int(np.median(nz[1]))
        else:
            return int(np.median(nz[0]))

    saved_files = []
    for view in views:
        ref_mask = mask_a_arr if mask_a_arr is not None else mask_b_arr
        center_idx = _find_mask_center(ref_mask, view)
        ct_slice = _get_slice(ct_norm, view, center_idx)

        panels = []
        for label, mask_arr in [("candidate_A", mask_a_arr), ("candidate_B", mask_b_arr)]:
            if mask_arr is None:
                continue
            mask_slice = _get_slice(mask_arr.astype(np.uint8), view, center_idx)
            rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
            rgb[mask_slice > 0, 0] = np.clip(rgb[mask_slice > 0, 0].astype(int) + 120, 0, 255)
            rgb[mask_slice > 0, 2] = np.clip(rgb[mask_slice > 0, 2].astype(int) - 60, 0, 255)
            panels.append((label, rgb))

        if not panels:
            continue

        fname = output_dir / f"{organ}_{view}_slice_fallback.png"
        if _has_pil and panels:
            from PIL import ImageDraw
            label_height = 20
            if len(panels) == 2:
                h, w = panels[0][1].shape[:2]
                combined = np.zeros((h + label_height, w * 2 + 4, 3), dtype=np.uint8)
                combined[label_height:, :w] = panels[0][1]
                combined[label_height:, w + 4:] = panels[1][1]
                img = PILImage.fromarray(combined)
                draw = ImageDraw.Draw(img)
                draw.text((4, 2), "Candidate A", fill=(255, 220, 0))
                draw.text((w + 8, 2), "Candidate B", fill=(255, 220, 0))
                img.save(str(fname))
            else:
                h, w = panels[0][1].shape[:2]
                padded = np.zeros((h + label_height, w, 3), dtype=np.uint8)
                padded[label_height:] = panels[0][1]
                img = PILImage.fromarray(padded)
                draw = ImageDraw.Draw(img)
                draw.text((4, 2), panels[0][0].replace("candidate_", "Candidate "), fill=(255, 220, 0))
                img.save(str(fname))
            saved_files.append(str(fname))
        else:
            npy_fname = output_dir / f"{organ}_{view}_slice_fallback.npy"
            np.save(str(npy_fname), np.stack([p[1] for p in panels]))
            saved_files.append(str(npy_fname))

    return {
        "stage": "projection_builder",
        "status": "success" if saved_files else "failed",
        "organ": organ,
        "views": views,
        "ct_image": str(ct_path),
        "mask_a": str(mask_a) if mask_a else None,
        "mask_b": str(mask_b) if mask_b else None,
        "output_folder": str(output_dir),
        "saved_projections": saved_files,
        "projection_backend": "slice_fallback",
        "projection_mode": "mask_centered_2d_slice_fallback",
        "strict_alignment": strict_alignment,
        "shape_warnings": shape_warnings if shape_warnings else None,
        "note": "Fallback only. The teacher-requested default is LabelCritic projection, which uses ProjectDatasetFlex_single.py/projection.py rather than naive averaging.",
    }


def build_projection(
    ct_image: str | Path,
    mask_a: str | Path | None,
    mask_b: str | Path | None,
    output_folder: str | Path,
    organ: str = "organ",
    views: list[str] | None = None,
    strict_alignment: bool = False,
    projection_backend: str = "labelcritic",
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    axis: int = 1,
    device: str = "cpu",
    num_processes: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create 2D projection images from 3D CT and candidate masks.

    Default behavior is teacher-aligned: use LabelCritic's own 3D-to-2D
    projection pipeline (ProjectDatasetFlex_single.py + projection.py).  That
    pipeline performs CT windowing for soft tissue/organs and bone/skeleton and
    creates comparison composites.  This explicitly replaces the earlier naive
    average or single-slice VLM input.

    projection_backend:
      - labelcritic: use LabelCritic projection and fail if unavailable.
      - auto: try LabelCritic first, fallback to slice projection if it fails.
      - slice: use lightweight mask-centered slice fallback only.
    """
    projection_backend = (projection_backend or "labelcritic").lower()
    if projection_backend in {"labelcritic", "auto"}:
        from .labelcritic_projection_runner import build_labelcritic_projection
        lc = build_labelcritic_projection(
            ct_image=ct_image,
            mask_a=mask_a,
            mask_b=mask_b,
            output_folder=output_folder,
            organ=organ,
            labelcritic_root=labelcritic_root,
            axis=axis,
            device=device,
            num_processes=num_processes,
            dry_run=dry_run,
        )
        if lc.get("status") in {"success", "dry_run", "warning"} or projection_backend == "labelcritic":
            return lc
        # Otherwise fall through for auto fallback.
        fallback = _build_slice_projection(ct_image, mask_a, mask_b, output_folder, organ, views, strict_alignment)
        fallback["labelcritic_attempt"] = lc
        fallback["projection_backend"] = "auto_slice_fallback_after_labelcritic_failure"
        return fallback

    return _build_slice_projection(ct_image, mask_a, mask_b, output_folder, organ, views, strict_alignment)
