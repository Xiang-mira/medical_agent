#!/usr/bin/env python3
"""Audit the exact case×373 identity and training-safety contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED = {
    "positive_hard", "positive_soft", "negative_absent",
    "unresolved_visible", "partial_fov", "rejected",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.manifest)
    doc = json.loads(source.read_text())
    rows = doc.get("items") or []
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id") or "")].append(row)
    case_failures = []
    for case_id, case_rows in sorted(by_case.items()):
        organs = [str(row.get("organ") or "") for row in case_rows]
        if len(case_rows) != 373 or len(set(organs)) != 373:
            case_failures.append({
                "case_id": case_id,
                "records": len(case_rows),
                "unique_organs": len(set(organs)),
            })
    bad_types = [row for row in rows if str(row.get("target_type") or "") not in ALLOWED]
    bad_absent = [
        row for row in rows
        if row.get("target_type") == "negative_absent"
        and (
            row.get("absence_confidence") != "high"
            or row.get("fov_status") != "out_of_fov"
            or row.get("zero_mask_role") != "negative_absent_target_mask"
        )
    ]
    placeholder_leaks = [
        row for row in rows
        if row.get("target_type") in {"unresolved_visible", "partial_fov"}
        and (
            float(row.get("training_weight") or 0) != 0
            or row.get("should_enter_student_training") is True
        )
    ]
    case_count = len(by_case)
    expected = case_count * 373
    counts = Counter(str(row.get("target_type") or "") for row in rows)
    passed = (
        case_count > 0 and len(rows) == expected and not case_failures
        and not bad_types and not bad_absent and not placeholder_leaks
    )
    result = {
        "stage": "full_case_x_373_manifest_audit",
        "status": "passed" if passed else "failed",
        "source": str(source.resolve()),
        "case_count": case_count,
        "class_count": 373,
        "expected_records": expected,
        "actual_records": len(rows),
        "identity_sum": sum(counts.values()),
        "target_type_counts": dict(counts),
        "case_failures": case_failures,
        "bad_target_type_count": len(bad_types),
        "bad_negative_absent_count": len(bad_absent),
        "placeholder_training_leak_count": len(placeholder_leaks),
        "contract_only_cases": sorted({
            str(row.get("case_id"))
            for row in rows
            if row.get("contract_only_no_teacher_evidence") is True
        }),
        "accuracy_claim_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
