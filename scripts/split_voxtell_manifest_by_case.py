#!/usr/bin/env python3
"""Create deterministic case-disjoint VoxTell train/validation manifests."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def summarize(base: dict[str, Any], rows: list[dict[str, Any]], split: str, case_ids: list[str]) -> dict[str, Any]:
    doc = {key: value for key, value in base.items() if key != "items"}
    doc.update({
        "status": "success",
        "split": split,
        "split_policy": "case_disjoint_16_train_4_validation_quartile_holdout",
        "case_ids": case_ids,
        "num_cases": len(case_ids),
        "num_items": len(rows),
        "num_positive_items": sum(row.get("supervision_type") == "positive" for row in rows),
        "num_negative_items": sum(row.get("supervision_type") == "negative" for row in rows),
        "num_prompt_variant_items": sum(bool(row.get("is_prompt_variant")) for row in rows),
        "grade_counts": dict(Counter(str(row.get("grade")) for row in rows)),
        "items": rows,
    })
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-cases", type=int, default=20)
    args = ap.parse_args()

    source = Path(args.manifest).resolve()
    base = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in base.get("items", []) if isinstance(row, dict)]
    with Path(args.case_list).resolve().open(encoding="utf-8-sig", newline="") as handle:
        ordered = [str(row["case_id"]) for row in list(csv.DictReader(handle))[: args.num_cases]]
    if len(ordered) != 20:
        raise SystemExit(f"Expected 20 case IDs, found {len(ordered)}")
    validation_ids = [ordered[index] for index in (4, 9, 14, 19)]
    train_ids = [case_id for case_id in ordered if case_id not in set(validation_ids)]
    row_cases = {str(row.get("case_id")) for row in rows}
    missing = [case_id for case_id in ordered if case_id not in row_cases]
    if missing:
        raise SystemExit(f"Manifest is missing cases: {missing}")
    train_rows = [row for row in rows if str(row.get("case_id")) in set(train_ids)]
    validation_rows = [row for row in rows if str(row.get("case_id")) in set(validation_ids)]
    overlap = set(train_ids) & set(validation_ids)
    if overlap or set(row["case_id"] for row in train_rows) & set(row["case_id"] for row in validation_rows):
        raise SystemExit(f"Case leakage detected: {sorted(overlap)}")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_doc = summarize(base, train_rows, "train", train_ids)
    validation_doc = summarize(base, validation_rows, "validation", validation_ids)
    (out / "train_manifest.json").write_text(json.dumps(train_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "validation_manifest.json").write_text(json.dumps(validation_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "status": "success",
        "source_manifest": str(source),
        "train_cases": train_ids,
        "validation_cases": validation_ids,
        "train_items": len(train_rows),
        "validation_items": len(validation_rows),
        "case_overlap": [],
        "all_prompt_variants_case_locked": True,
        "train_grade_counts": train_doc["grade_counts"],
        "validation_grade_counts": validation_doc["grade_counts"],
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
