#!/usr/bin/env python3
"""Audit an emitted case-organ manifest against the strict 373 contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "positive_hard", "positive_soft", "negative_absent",
    "unresolved_visible", "partial_fov", "rejected",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.selection_metadata)
    doc = json.loads(source.read_text())
    rows = doc.get("selection_rows") or doc.get("selected_organs") or []
    organs = [str(row.get("organ") or "") for row in rows]
    counts = Counter(str(row.get("target_type") or "") for row in rows)
    bad_absent = [
        row for row in rows
        if row.get("target_type") == "negative_absent"
        and (
            row.get("absence_confidence") != "high"
            or row.get("fov_status") != "out_of_fov"
            or float(row.get("training_weight") or 0) < 0
        )
    ]
    leaked_placeholders = [
        row for row in rows
        if row.get("target_type") in {"unresolved_visible", "partial_fov"}
        and (
            float(row.get("training_weight") or 0) != 0
            or row.get("should_enter_student_training") is True
        )
    ]
    valid = (
        len(rows) == 373
        and len(set(organs)) == 373
        and set(counts) <= ALLOWED
        and not bad_absent
        and not leaked_placeholders
    )
    result = {
        "stage": "case_x_373_contract",
        "status": "passed" if valid else "failed",
        "source": str(source.resolve()),
        "record_count": len(rows),
        "unique_organ_count": len(set(organs)),
        "target_type_counts": dict(counts),
        "identity_sum": sum(counts.values()),
        "allowed_target_types": sorted(ALLOWED),
        "bad_negative_absent": bad_absent[:20],
        "training_placeholder_leaks": leaked_placeholders[:20],
        "complete_case_373_semantics": "record_and_output_contract_only_not_visibility_or_accuracy",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
