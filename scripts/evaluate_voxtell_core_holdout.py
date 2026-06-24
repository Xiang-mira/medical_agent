#!/usr/bin/env python3
"""Compare VoxTell checkpoints against repaired pseudo-labels on core holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_binary(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--organs", nargs="+", default=["liver", "pancreas", "kidney_left", "kidney_right"])
    ap.add_argument("--modes", nargs="+", default=["base", "repair_balanced"])
    args = ap.parse_args()

    rows = []
    for case in args.cases:
        for organ in args.organs:
            reference_path = args.experiment_root / "estep" / "annotation_versions" / case / "updated" / f"{organ}.nii.gz"
            reference = read_binary(reference_path)
            reference_voxels = int(reference.sum())
            for mode in args.modes:
                prediction_path = args.experiment_root / "mstep" / "holdout_eval" / mode / case / f"{organ}.nii.gz"
                prediction = read_binary(prediction_path)
                prediction_voxels = int(prediction.sum())
                intersection = int(np.logical_and(reference, prediction).sum())
                denominator = reference_voxels + prediction_voxels
                dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
                rows.append({
                    "case_id": case,
                    "organ": organ,
                    "mode": mode,
                    "dice_pseudo_consistency": round(float(dice), 6),
                    "reference_voxels": reference_voxels,
                    "prediction_voxels": prediction_voxels,
                    "nonempty": prediction_voxels > 0,
                })
                del prediction
            del reference

    summary = {}
    for mode in args.modes:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "num_predictions": len(selected),
            "num_nonempty": sum(row["nonempty"] for row in selected),
            "mean_dice_pseudo_consistency": round(float(np.mean([row["dice_pseudo_consistency"] for row in selected])), 6),
            "mean_by_organ": {
                organ: round(float(np.mean([row["dice_pseudo_consistency"] for row in selected if row["organ"] == organ])), 6)
                for organ in args.organs
            },
        }
    base = {(r["case_id"], r["organ"]): r for r in rows if r["mode"] == args.modes[0]}
    repaired = {(r["case_id"], r["organ"]): r for r in rows if r["mode"] == args.modes[1]}
    deltas = [repaired[key]["dice_pseudo_consistency"] - row["dice_pseudo_consistency"] for key, row in base.items()]
    report = {
        "metric_warning": "Pseudo-consistency against repaired machine labels; not expert-ground-truth accuracy.",
        "summary": summary,
        "comparison": {
            "mean_dice_delta": round(float(np.mean(deltas)), 6),
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "worsened": sum(delta < 0 for delta in deltas),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
