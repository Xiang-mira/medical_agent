from __future__ import annotations

from pathlib import Path
from .json_utils import write_json


def summarize_segmentation_folder(folder: str | Path) -> dict:
    root = Path(folder).resolve()
    if not root.exists():
        return {"status": "failed", "reason": "folder does not exist", "folder": str(root)}
    cases = [p for p in root.iterdir() if p.is_dir()]
    summaries = []
    for case in sorted(cases):
        seg = case / "segmentations"
        masks = sorted([p.name for p in seg.glob("*.nii.gz")]) if seg.exists() else []
        summaries.append({"case_id": case.name, "has_segmentations": seg.exists(), "num_masks": len(masks), "has_combined_labels": (case / "combined_labels.nii.gz").exists(), "sample_masks": masks[:40]})
    return {"status": "success", "folder": str(root), "num_cases": len(cases), "cases": summaries}


def write_run_summary(output_folder: str | Path, summary: dict) -> str:
    return write_json(Path(output_folder).resolve() / "medai_run_summary.json", summary)
