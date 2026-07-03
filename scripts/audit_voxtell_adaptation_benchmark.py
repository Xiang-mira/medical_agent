#!/usr/bin/env python3
"""Freeze and score a held-out VoxTell pseudo-consistency benchmark.

The PanTS LabelTr masks used here are historical pseudo labels, not expert
ground truth.  The benchmark therefore measures adaptation to the repaired
project label distribution; it must never be reported as segmentation
accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


PANCREAS_SUBREGIONS = {"pancreas_body", "pancreas_head", "pancreas_tail"}
ADRENALS = {"adrenal_gland_left", "adrenal_gland_right"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj) > 0


def _dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(reference.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.count_nonzero(prediction & reference)) / denominator


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def _family(organ: str) -> str:
    if organ in PANCREAS_SUBREGIONS:
        return "pancreas_subregion"
    if organ in ADRENALS:
        return "adrenal_gland"
    if organ == "pancreas":
        return "pancreas_whole"
    return organ


def _metrics(prediction_path: Path, reference_path: Path) -> dict:
    prediction_image = nib.load(prediction_path)
    reference_image = nib.load(reference_path)
    if prediction_image.shape != reference_image.shape:
        raise ValueError(
            f"shape mismatch: {prediction_path} {prediction_image.shape} != "
            f"{reference_path} {reference_image.shape}"
        )
    prediction = np.asarray(prediction_image.dataobj) > 0
    reference = np.asarray(reference_image.dataobj) > 0
    return {
        "dice_pseudo_consistency": _dice(prediction, reference),
        "prediction_voxels": int(prediction.sum()),
        "reference_voxels": int(reference.sum()),
        "reference_touches_volume_boundary": _touches_boundary(reference),
    }


def _macro_summary(rows: list[dict], metric: str) -> dict:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["organ_family"])].append(float(row[metric]))
    case_family = [
        {
            "case_id": case_id,
            "organ_family": family,
            "num_targets": len(values),
            metric: float(np.mean(values)),
        }
        for (case_id, family), values in sorted(grouped.items())
    ]
    case_values: dict[str, list[float]] = defaultdict(list)
    for row in case_family:
        case_values[row["case_id"]].append(float(row[metric]))
    per_case = [
        {
            "case_id": case_id,
            "num_families": len(values),
            metric: float(np.mean(values)),
        }
        for case_id, values in sorted(case_values.items())
    ]
    return {
        "target_micro_mean": float(np.mean([row[metric] for row in rows])),
        "case_family_macro_mean": float(
            np.mean([row[metric] for row in case_family])
        ),
        "case_macro_mean": float(np.mean([row[metric] for row in per_case])),
        "case_family": case_family,
        "per_case": per_case,
    }


def freeze(args: argparse.Namespace) -> dict:
    baseline_root = Path(args.baseline_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    excluded = set(args.exclude_organ)
    rows = []
    exclusions = defaultdict(int)
    for case_id in args.case:
        case_prediction_root = baseline_root / case_id
        case_reference_root = reference_root / case_id / "segmentations"
        for prediction_path in sorted(case_prediction_root.glob("*.nii.gz")):
            organ = prediction_path.name.removesuffix(".nii.gz")
            if organ.startswith("ct_"):
                continue
            reference_path = case_reference_root / prediction_path.name
            if not reference_path.exists():
                exclusions["missing_reference"] += 1
                continue
            metrics = _metrics(prediction_path, reference_path)
            reasons = []
            if organ in excluded:
                reasons.append("explicit_semantic_exclusion")
            if metrics["reference_voxels"] < args.min_reference_voxels:
                reasons.append("reference_too_small")
            if metrics["reference_touches_volume_boundary"]:
                reasons.append("reference_touches_volume_boundary")
            if metrics["dice_pseudo_consistency"] >= args.max_baseline_dice:
                reasons.append("baseline_not_weak")
            if reasons:
                for reason in reasons:
                    exclusions[reason] += 1
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "organ": organ,
                    "organ_family": _family(organ),
                    "reference_path": str(reference_path.resolve()),
                    "reference_sha256": _sha256(reference_path),
                    "baseline_prediction_path": str(prediction_path.resolve()),
                    "baseline_prediction_sha256": _sha256(prediction_path),
                    "baseline_dice_pseudo_consistency": metrics[
                        "dice_pseudo_consistency"
                    ],
                    "baseline_prediction_voxels": metrics["prediction_voxels"],
                    "reference_voxels": metrics["reference_voxels"],
                }
            )
    if not rows:
        raise RuntimeError("benchmark selection produced no eligible targets")
    summary_rows = [
        {**row, "dice_pseudo_consistency": row["baseline_dice_pseudo_consistency"]}
        for row in rows
    ]
    return {
        "schema_version": 1,
        "status": "descriptive_only",
        "metric_family": "historical_pseudo_consistency",
        "ground_truth_status": "historical_pseudo_not_expert_ground_truth",
        "accuracy_warning": (
            "This benchmark measures consistency with historical PanTS pseudo "
            "labels and cannot establish true segmentation accuracy."
        ),
        "selection_policy": {
            "held_out_cases": args.case,
            "minimum_reference_voxels": args.min_reference_voxels,
            "maximum_official_baseline_dice_exclusive": args.max_baseline_dice,
            "require_reference_not_touching_volume_boundary": True,
            "excluded_organs": sorted(excluded),
            "aggregation": (
                "Make decisions from paired case-organ rows with teacher "
                "provenance. Aggregate means are descriptive only and must not "
                "offset a target-level regression."
            ),
        },
        "decision_policy": {
            "automatic_aggregate_gate": False,
            "unit_of_review": "paired_case_organ_with_teacher_provenance",
            "requires_separate_official_baseline_regression_gate": True,
            "interpretation": (
                "Aggregate values are descriptive diagnostics only. This "
                "historical pseudo-label comparison cannot establish true "
                "segmentation accuracy or hide a per-target regression."
            ),
        },
        "num_targets": len(rows),
        "num_case_families": len(
            {(row["case_id"], row["organ_family"]) for row in rows}
        ),
        "selection_exclusion_counts_nonexclusive": dict(sorted(exclusions.items())),
        "targets": rows,
        "baseline_summary": _macro_summary(
            summary_rows, "dice_pseudo_consistency"
        ),
    }


def score(args: argparse.Namespace, benchmark: dict) -> dict:
    candidate_root = Path(args.candidate_root).resolve()
    rows = []
    missing = []
    for target in benchmark["targets"]:
        prediction_path = (
            candidate_root / target["case_id"] / f"{target['organ']}.nii.gz"
        )
        if not prediction_path.exists():
            missing.append(str(prediction_path))
            continue
        reference_path = Path(target["reference_path"])
        if _sha256(reference_path) != target["reference_sha256"]:
            raise RuntimeError(f"frozen reference changed: {reference_path}")
        metrics = _metrics(prediction_path, reference_path)
        candidate_dice = metrics["dice_pseudo_consistency"]
        rows.append(
            {
                "case_id": target["case_id"],
                "organ": target["organ"],
                "organ_family": target["organ_family"],
                "baseline_dice_pseudo_consistency": target[
                    "baseline_dice_pseudo_consistency"
                ],
                "candidate_dice_pseudo_consistency": candidate_dice,
                "dice_delta": candidate_dice
                - target["baseline_dice_pseudo_consistency"],
                "reference_voxels": target["reference_voxels"],
                "baseline_prediction_voxels": target[
                    "baseline_prediction_voxels"
                ],
                "candidate_prediction_voxels": metrics["prediction_voxels"],
            }
        )
    complete = len(rows) == len(benchmark["targets"]) and not missing
    candidate_summary = (
        _macro_summary(
            [
                {
                    **row,
                    "dice_pseudo_consistency": row[
                        "candidate_dice_pseudo_consistency"
                    ],
                }
                for row in rows
            ],
            "dice_pseudo_consistency",
        )
        if rows
        else None
    )
    primary_delta = (
        candidate_summary["case_family_macro_mean"]
        - benchmark["baseline_summary"]["case_family_macro_mean"]
        if candidate_summary
        else None
    )
    baseline_case_family = {
        (row["case_id"], row["organ_family"]): row["dice_pseudo_consistency"]
        for row in benchmark["baseline_summary"]["case_family"]
    }
    case_family_deltas = []
    if candidate_summary:
        for row in candidate_summary["case_family"]:
            key = (row["case_id"], row["organ_family"])
            case_family_deltas.append(
                {
                    "case_id": row["case_id"],
                    "organ_family": row["organ_family"],
                    "dice_delta": row["dice_pseudo_consistency"]
                    - baseline_case_family[key],
                }
            )
    minimum_case_family_delta = min(
        (row["dice_delta"] for row in case_family_deltas), default=None
    )
    minimum_target_delta = min(
        (row["dice_delta"] for row in rows), default=None
    )
    return {
        "status": "complete" if complete else "incomplete",
        "benchmark": str(Path(args.benchmark).resolve()),
        "candidate_root": str(candidate_root),
        "num_expected_targets": len(benchmark["targets"]),
        "num_scored_targets": len(rows),
        "missing_predictions": missing,
        "descriptive_case_family_macro_delta": primary_delta,
        "minimum_case_family_delta": minimum_case_family_delta,
        "minimum_target_delta": minimum_target_delta,
        "case_family_deltas": case_family_deltas,
        "decision_policy": benchmark["decision_policy"],
        "candidate_summary": candidate_summary,
        "targets": rows,
        "accuracy_warning": benchmark["accuracy_warning"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", help="Frozen benchmark JSON to score.")
    parser.add_argument("--candidate-root", help="Candidate root containing CASE/*.nii.gz.")
    parser.add_argument("--baseline-root")
    parser.add_argument("--reference-root")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--min-reference-voxels", type=int, default=500)
    parser.add_argument("--max-baseline-dice", type=float, default=0.85)
    parser.add_argument("--exclude-organ", action="append", default=["veins"])
    args = parser.parse_args()
    if bool(args.benchmark) != bool(args.candidate_root):
        parser.error("--benchmark and --candidate-root must be used together")
    if args.benchmark:
        benchmark = json.loads(Path(args.benchmark).read_text())
        result = score(args, benchmark)
    else:
        if not args.baseline_root or not args.reference_root or not args.case:
            parser.error(
                "freeze mode requires --baseline-root, --reference-root, and --case"
            )
        result = freeze(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in result if k in {
        "status", "num_targets", "num_case_families", "primary_metric_delta"
    }}, indent=2))


if __name__ == "__main__":
    main()
