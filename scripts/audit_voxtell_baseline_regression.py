#!/usr/bin/env python3
"""Fail-fast regression gate comparing a finetuned VoxTell model to its baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_mask(path: Path) -> tuple[np.ndarray, float]:
    image = nib.load(str(path))
    mask = np.asanyarray(image.dataobj) > 0
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return mask, voxel_ml


def dice(left: np.ndarray, right: np.ndarray) -> float:
    denom = int(left.sum() + right.sum())
    return 1.0 if denom == 0 else 2.0 * int(np.logical_and(left, right).sum()) / denom


def manifest_refs(path: Path, case_id: str) -> dict[str, Path]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    refs: dict[str, Path] = {}
    for item in doc.get("items", []):
        if str(item.get("case_id") or "") != case_id or item.get("is_prompt_variant"):
            continue
        organ = str(item.get("organ") or "")
        mask = Path(str(item.get("mask") or item.get("mask_path") or ""))
        if organ and mask.exists() and organ not in refs:
            refs[organ] = mask
    return refs


def audit(
    *,
    baseline_root: Path,
    candidate_root: Path,
    manifest_path: Path,
    case_id: str,
    positive_organs: list[str],
    absent_organs: list[str],
    max_positive_dice_drop: float,
    min_positive_volume_ratio: float,
    max_positive_volume_ratio: float,
    max_absent_volume_ratio: float,
    absent_volume_slack_ml: float,
) -> dict[str, Any]:
    refs = manifest_refs(manifest_path, case_id)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for organ in positive_organs + absent_organs:
        baseline_path = baseline_root / f"{organ}.nii.gz"
        candidate_path = candidate_root / f"{organ}.nii.gz"
        if not baseline_path.exists() or not candidate_path.exists():
            failures.append({"organ": organ, "reason": "missing_prediction"})
            continue
        baseline, voxel_ml = load_mask(baseline_path)
        candidate, candidate_voxel_ml = load_mask(candidate_path)
        if baseline.shape != candidate.shape or not np.isclose(voxel_ml, candidate_voxel_ml):
            failures.append({"organ": organ, "reason": "geometry_mismatch"})
            continue
        baseline_ml = float(baseline.sum()) * voxel_ml
        candidate_ml = float(candidate.sum()) * candidate_voxel_ml
        row: dict[str, Any] = {
            "organ": organ,
            "role": "positive" if organ in positive_organs else "absent_negative",
            "baseline_volume_ml": round(baseline_ml, 6),
            "candidate_volume_ml": round(candidate_ml, 6),
        }
        if organ in positive_organs:
            ref_path = refs.get(organ)
            if ref_path is None:
                failures.append({"organ": organ, "reason": "missing_manifest_reference"})
                continue
            reference, _ = load_mask(ref_path)
            baseline_dice = dice(baseline, reference)
            candidate_dice = dice(candidate, reference)
            if baseline_ml > 0:
                ratio = candidate_ml / baseline_ml
            else:
                ratio = 1.0 if candidate_ml == 0 else float("inf")
            row.update({
                "baseline_reference_dice": round(baseline_dice, 6),
                "candidate_reference_dice": round(candidate_dice, 6),
                "dice_drop": round(baseline_dice - candidate_dice, 6),
                "volume_ratio": round(ratio, 6),
            })
            if baseline_dice - candidate_dice > max_positive_dice_drop:
                failures.append({"organ": organ, "reason": "positive_dice_regression"})
            if not min_positive_volume_ratio <= ratio <= max_positive_volume_ratio:
                failures.append({"organ": organ, "reason": "positive_volume_regression"})
        else:
            allowed_ml = baseline_ml * max_absent_volume_ratio + absent_volume_slack_ml
            row["allowed_candidate_volume_ml"] = round(allowed_ml, 6)
            if candidate_ml > allowed_ml:
                failures.append({"organ": organ, "reason": "absent_volume_regression"})
        rows.append(row)
    return {
        "stage": "voxtell_baseline_regression_gate",
        "status": "success" if not failures else "failed",
        "case_id": case_id,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "manifest": str(manifest_path),
        "thresholds": {
            "max_positive_dice_drop": max_positive_dice_drop,
            "positive_volume_ratio": [min_positive_volume_ratio, max_positive_volume_ratio],
            "max_absent_volume_ratio": max_absent_volume_ratio,
            "absent_volume_slack_ml": absent_volume_slack_ml,
        },
        "rows": rows,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--positive-organs", default="liver,spleen,pancreas,kidney_left,aorta")
    parser.add_argument("--absent-organs", default="brain_ventricle,cerebrospinal_fluid,oral_cavity")
    parser.add_argument("--max-positive-dice-drop", type=float, default=0.02)
    parser.add_argument("--min-positive-volume-ratio", type=float, default=0.8)
    parser.add_argument("--max-positive-volume-ratio", type=float, default=1.25)
    parser.add_argument("--max-absent-volume-ratio", type=float, default=1.1)
    parser.add_argument("--absent-volume-slack-ml", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        baseline_root=args.baseline_root.resolve(),
        candidate_root=args.candidate_root.resolve(),
        manifest_path=args.manifest.resolve(),
        case_id=args.case_id,
        positive_organs=[x.strip() for x in args.positive_organs.split(",") if x.strip()],
        absent_organs=[x.strip() for x in args.absent_organs.split(",") if x.strip()],
        max_positive_dice_drop=args.max_positive_dice_drop,
        min_positive_volume_ratio=args.min_positive_volume_ratio,
        max_positive_volume_ratio=args.max_positive_volume_ratio,
        max_absent_volume_ratio=args.max_absent_volume_ratio,
        absent_volume_slack_ml=args.absent_volume_slack_ml,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
