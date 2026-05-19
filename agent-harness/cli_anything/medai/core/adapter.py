from __future__ import annotations

import shutil
from pathlib import Path

from .presets import LUNG_LEFT_PARTS, LUNG_RIGHT_PARTS, SHAPEKIT_EXPECTED_ORGANS, TOTALSEG_TO_SHAPEKIT_RENAME


def _copy_if_needed(src: Path, dst: Path, actions: list[str]) -> None:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        actions.append(f"copy {src.name} -> {dst.name}")


def _union_masks(seg_dir: Path, parts: list[str], output_name: str, actions: list[str]) -> bool:
    output = seg_dir / f"{output_name}.nii.gz"
    if output.exists():
        return True
    paths = [seg_dir / f"{p}.nii.gz" for p in parts if (seg_dir / f"{p}.nii.gz").exists()]
    if not paths:
        return False
    try:
        import numpy as np
        import nibabel as nib
        ref = nib.load(str(paths[0]))
        union = np.zeros(ref.shape, dtype=np.uint8)
        for p in paths:
            arr = np.asanyarray(nib.load(str(p)).dataobj)
            union[arr > 0] = 1
        nib.save(nib.Nifti1Image(union, ref.affine, ref.header), str(output))
        actions.append(f"union {','.join([p.name for p in paths])} -> {output.name}")
        return True
    except Exception as exc:
        actions.append(f"cannot create {output.name}: {exc}")
        return False


def normalize_totalseg_to_shapekit(segmentation_folder: str | Path) -> dict:
    seg_dir = Path(segmentation_folder).resolve()
    if not seg_dir.exists():
        return {"stage": "adapter", "status": "failed", "reason": "segmentation folder does not exist", "segmentation_folder": str(seg_dir)}
    actions: list[str] = []
    for src, dst in TOTALSEG_TO_SHAPEKIT_RENAME.items():
        _copy_if_needed(seg_dir / f"{src}.nii.gz", seg_dir / f"{dst}.nii.gz", actions)
    _union_masks(seg_dir, LUNG_LEFT_PARTS, "lung_left", actions)
    _union_masks(seg_dir, LUNG_RIGHT_PARTS, "lung_right", actions)
    available = sorted([p.name[:-7] for p in seg_dir.glob("*.nii.gz")])
    ready = sorted([x for x in available if x in SHAPEKIT_EXPECTED_ORGANS or x.startswith("vertebrae_")])
    missing_core = [x for x in ["liver", "pancreas", "aorta", "postcava"] if x not in available]
    return {"stage": "adapter", "status": "success" if not missing_core else "warning", "segmentation_folder": str(seg_dir), "actions": actions, "num_masks_total": len(available), "num_shapekit_ready_masks": len(ready), "sample_shapekit_ready_masks": ready[:60], "missing_core_masks_for_shapekit": missing_core, "note": "This adapter keeps raw masks and creates ShapeKit/PanTS-compatible names."}
