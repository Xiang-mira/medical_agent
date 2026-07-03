#!/usr/bin/env python3
"""Audit independent LabelCritic grades for conflicting teacher candidates.

This is a calibration experiment.  It never rewrites pseudo labels or a
training manifest, and historical pseudo-label Dice is reported only as weak
diagnostic context rather than ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = PROJECT_ROOT / "agent-harness"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_grade_batch


def _dice(mask_a: Path, mask_b: Path) -> float:
    a = np.asarray(nib.load(mask_a).dataobj) > 0
    b = np.asarray(nib.load(mask_b).dataobj) > 0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {mask_a} {a.shape} != {mask_b} {b.shape}")
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.count_nonzero(a & b)) / denominator


def _usable_candidates(selection: dict) -> list[dict]:
    candidates = []
    for candidate in selection.get("candidate_predictions", []) or []:
        path = Path(str(candidate.get("prediction") or ""))
        hard_flags = {
            "missing_file",
            "unreadable_mask",
            "geometry_mismatch",
            "shape_mismatch",
            "shape_mismatch_ct",
            "zero_volume_mask",
        }
        flags = set(map(str, candidate.get("candidate_qc_flags", []) or []))
        if (
            path.exists()
            and candidate.get("eligible_for_labelcritic", True)
            and not (flags & hard_flags)
        ):
            candidates.append(candidate)
    return candidates


def _select_conflicts(
    selection_json: Path,
    max_organs: int,
    requested_organs: set[str] | None = None,
) -> list[dict]:
    document = json.loads(selection_json.read_text())
    eligible = []
    for selection in document.get("selection_rows", []):
        if requested_organs and selection.get("organ") not in requested_organs:
            continue
        candidates = _usable_candidates(selection)
        if len(candidates) < 2:
            continue
        pairwise = [
            {
                "teacher_a": a["model"],
                "teacher_b": b["model"],
                "dice_3d": _dice(Path(a["prediction"]), Path(b["prediction"])),
            }
            for a, b in combinations(candidates, 2)
        ]
        eligible.append(
            {
                "case_id": document["case_id"],
                "ct_path": document["ct_path"],
                "organ": selection["organ"],
                "candidates": candidates,
                "pairwise_teacher_dice": pairwise,
                "minimum_teacher_dice": min(row["dice_3d"] for row in pairwise),
                "selection_method": selection.get("selection_method"),
                "selected_model": selection.get("selected_model"),
            }
        )
    eligible.sort(key=lambda row: (row["minimum_teacher_dice"], row["organ"]))
    return eligible[:max_organs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-organs", type=int, default=12)
    parser.add_argument(
        "--organ",
        action="append",
        default=[],
        help="Optional explicit organ panel; repeat for multiple organs.",
    )
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vlm-model")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = _select_conflicts(
        Path(args.selection_json),
        args.max_organs,
        set(args.organ) or None,
    )
    if not conflicts:
        raise RuntimeError("no multi-teacher case-organ conflicts found")

    jobs = []
    candidate_context = []
    for conflict in conflicts:
        for candidate in conflict["candidates"]:
            output_json = (
                output_dir
                / "grades"
                / f"{conflict['organ']}__{candidate['model']}.json"
            )
            jobs.append(
                {
                    "ct_image": conflict["ct_path"],
                    "mask": candidate["prediction"],
                    "organ": conflict["organ"],
                    "output_json": str(output_json),
                }
            )
            candidate_context.append((conflict, candidate, output_json))

    results = run_labelcritic_grade_batch(
        jobs,
        base_url=args.base_url,
        port=args.port,
        vlm_model=args.vlm_model,
        dry_run=args.dry_run,
        timeout_sec=300,
        concurrency=args.concurrency,
    )
    result_by_output = {
        str(Path(result["output_json"]).resolve()): result for result in results
    }
    candidate_rows = []
    for conflict, candidate, output_json in candidate_context:
        grade = result_by_output.get(str(output_json.resolve()), {})
        candidate_rows.append(
            {
                "case_id": conflict["case_id"],
                "organ": conflict["organ"],
                "teacher": candidate["model"],
                "mask_path": str(Path(candidate["prediction"]).resolve()),
                "candidate_qc_status": candidate.get("candidate_qc_status"),
                "candidate_qc_flags": candidate.get("candidate_qc_flags", []),
                "historical_pseudo_consistency_dice": candidate.get("dice"),
                "labelcritic_status": grade.get("status"),
                "labelcritic_grade": grade.get("grade"),
                "labelcritic_grade_label": grade.get("grade_label"),
                "labelcritic_accept": grade.get("accept"),
                "labelcritic_reason": grade.get("reason"),
                "labelcritic_hard_failure_reason": grade.get("hard_failure_reason"),
                "raw_response": grade.get("raw_response"),
                "grade_output_json": str(output_json),
            }
        )

    successful = [
        row for row in candidate_rows if row["labelcritic_status"] == "success"
    ]
    grade_counts = Counter(
        str(row["labelcritic_grade"]) for row in successful
    )
    label_counts = Counter(
        str(row["labelcritic_grade_label"]) for row in successful
    )
    per_organ = []
    for conflict in conflicts:
        rows = [
            row for row in candidate_rows if row["organ"] == conflict["organ"]
        ]
        numeric = [
            float(row["labelcritic_grade"])
            for row in rows
            if row["labelcritic_grade"] is not None
        ]
        unique_grades = sorted(set(numeric))
        per_organ.append(
            {
                "organ": conflict["organ"],
                "minimum_teacher_dice": conflict["minimum_teacher_dice"],
                "pairwise_teacher_dice": conflict["pairwise_teacher_dice"],
                "num_candidates": len(rows),
                "num_successful_grades": sum(
                    row["labelcritic_status"] == "success" for row in rows
                ),
                "unique_labelcritic_grades": unique_grades,
                "labelcritic_discriminated_candidates": len(unique_grades) > 1,
            }
        )

    summary = {
        "stage": "labelcritic_teacher_candidate_calibration",
        "status": (
            "complete"
            if len(successful) == len(candidate_rows)
            else "partial"
        ),
        "case_id": conflicts[0]["case_id"],
        "selection_json": str(Path(args.selection_json).resolve()),
        "ground_truth_status": "historical_pseudo_not_expert_ground_truth",
        "accuracy_warning": (
            "Historical pseudo-consistency is weak diagnostic context only."
        ),
        "mutation_policy": "read_only_candidates_no_manifest_or_mask_rewrite",
        "selection_policy": (
            "Explicit conflict-stratified organ panel; within the panel, "
            "lowest teacher-to-teacher 3D Dice first."
            if args.organ
            else "Lowest teacher-to-teacher 3D Dice first among organs with "
            "at least two structurally usable teacher candidates."
        ),
        "requested_organs": args.organ,
        "num_organs": len(conflicts),
        "num_candidates": len(candidate_rows),
        "num_successful_grades": len(successful),
        "grade_counts": dict(sorted(grade_counts.items())),
        "grade_label_counts": dict(sorted(label_counts.items())),
        "num_organs_with_discriminative_grades": sum(
            row["labelcritic_discriminated_candidates"] for row in per_organ
        ),
        "all_successful_grades_identical": len(grade_counts) == 1,
        "decision": (
            "insufficient_discrimination_do_not_argmax"
            if successful and len(grade_counts) == 1
            else "review_per_organ_discrimination"
        ),
        "per_organ": per_organ,
        "candidate_rows": candidate_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": summary["status"],
        "num_organs": summary["num_organs"],
        "num_candidates": summary["num_candidates"],
        "grade_counts": summary["grade_counts"],
        "num_organs_with_discriminative_grades": summary[
            "num_organs_with_discriminative_grades"
        ],
        "decision": summary["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
