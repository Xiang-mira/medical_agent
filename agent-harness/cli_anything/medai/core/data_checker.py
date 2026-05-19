from __future__ import annotations

import shutil
import sys
from pathlib import Path


def check_ct_image(image: str | Path) -> dict:
    p = Path(image).resolve()
    return {
        "status": "success" if p.exists() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")) else "failed",
        "image": str(p),
        "exists": p.exists(),
        "is_nifti": p.name.endswith(".nii.gz") or p.name.endswith(".nii"),
    }


def check_case_folder(input_folder: str | Path) -> dict:
    root = Path(input_folder).resolve()
    if not root.exists():
        return {"status": "failed", "reason": "input folder does not exist", "input_folder": str(root)}
    cases = [p for p in root.iterdir() if p.is_dir()]
    summaries = []
    for case in sorted(cases):
        seg = case / "segmentations"
        masks = sorted([p.name for p in seg.glob("*.nii.gz")]) if seg.exists() else []
        summaries.append({
            "case_id": case.name,
            "has_segmentations_folder": seg.exists(),
            "num_masks": len(masks),
            "has_liver_reference": "liver.nii.gz" in masks,
            "has_pancreas": "pancreas.nii.gz" in masks,
            "has_aorta": "aorta.nii.gz" in masks,
            "has_postcava": "postcava.nii.gz" in masks,
            "sample_masks": masks[:40],
        })
    failed = [c for c in summaries if not c["has_segmentations_folder"] or c["num_masks"] == 0]
    status = "success" if summaries and not failed else "warning" if summaries else "failed"
    return {"status": status, "input_folder": str(root), "num_cases": len(cases), "num_failed_cases": len(failed), "cases": summaries}


def check_environment(shapekit_root: str | Path = "third_party/ShapeKit-main") -> dict:
    root = Path(shapekit_root).resolve()
    exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
    return {
        "python": sys.executable,
        "totalsegmentator_command": exe,
        "shapekit": {
            "root": str(root),
            "exists": root.exists(),
            "has_main_py": (root / "main.py").exists(),
            "has_config_yaml": (root / "config.yaml").exists(),
            "has_requirements": (root / "requirements.txt").exists(),
        },
        "status": "success" if root.exists() and (root / "main.py").exists() else "warning",
        "note": "If TotalSegmentator is missing, run: pip install TotalSegmentator. The mock custom backend can still test the CLI loop.",
    }
