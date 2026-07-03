#!/usr/bin/env python3
"""Execute generated corruption pairs through the pinned official adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_compare


def summary(path: Path, candidate_id: str) -> dict:
    image = nib.load(str(path))
    mask = np.asanyarray(image.dataobj) > 0
    voxels = int(mask.sum())
    result = {
        "candidate_id": candidate_id,
        "foreground_voxel_count": voxels,
        "volume_mm3": float(voxels * abs(np.linalg.det(image.affine[:3, :3]))),
        "connected_components": int(label(mask)[1]) if voxels else 0,
    }
    if voxels:
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0)
        centroid = nib.affines.apply_affine(image.affine, coords.mean(0))
        result.update({
            "bbox_voxel": {"min": lo.tolist(), "max": hi.tolist()},
            "centroid_ras_mm": [round(float(x), 4) for x in centroid],
            "boundary_contacts": {
                "axis0_min": bool(lo[0] == 0), "axis0_max": bool(hi[0] == mask.shape[0] - 1),
                "axis1_min": bool(lo[1] == 0), "axis1_max": bool(hi[1] == mask.shape[1] - 1),
                "axis2_min": bool(lo[2] == 0), "axis2_max": bool(hi[2] == mask.shape[2] - 1),
            },
            "truncation_suspected": bool(lo[2] == 0 or hi[2] == mask.shape[2] - 1),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--corruptions", nargs="*", default=[])
    args = parser.parse_args()
    task_doc = json.loads(Path(args.tasks).read_text())
    tasks = list(task_doc.get("tasks") or [])
    if args.corruptions:
        tasks = [task for task in tasks if task.get("corruption") in set(args.corruptions)]
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, task in enumerate(tasks):
        mask_a = Path(task["candidate_a"])
        mask_b = Path(task["candidate_b"])
        pair_output = output.parent / "pairs" / f"{index:03d}_{task['organ']}_{task['corruption']}.json"
        context = [summary(mask_a, "candidate_a"), summary(mask_b, "candidate_b")]
        result = run_labelcritic_compare(
            task["ct"], mask_a, mask_b, task["organ"], pair_output,
            backend="labelcritic", base_url=args.base_url, port=args.port,
            dry_run=False, timeout_sec=900, candidate_context=context,
        )
        decision = result.get("decision") or {}
        rows.append({
            **task,
            "status": result.get("status"),
            "winner": decision.get("winner"),
            "passed": result.get("status") == "success" and decision.get("winner") == task.get("expected"),
            "pair_output": str(pair_output),
            "structured_assessment_present": bool(result.get("structured_assessment")),
            "centered_slice_fallback_used": (result.get("projection") or {}).get("centered_slice_fallback_used"),
        })
        output.write_text(json.dumps({"status": "running", "results": rows}, indent=2) + "\n")
    passed = bool(rows) and all(row["passed"] for row in rows)
    report = {
        "stage": "labelcritic_373_corruption_regression",
        "status": "passed" if passed else "failed",
        "task_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "results": rows,
        "accuracy_claim_allowed": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
