#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import read_csv_fieldnames, read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    source = row.get("source_name") or row.get("current_name") or ""
    target = row.get("target_name") or ""
    return source, target


def compare_mappings(base: Path, others: list[Path]) -> dict[str, Any]:
    base_rows = read_csv_rows(base)
    base_by_key = {_row_key(row): row for row in base_rows}
    base_header = read_csv_fieldnames(base)
    reports = []
    diff_rows: list[dict[str, Any]] = []
    for other in others:
        rows = read_csv_rows(other)
        by_key = {_row_key(row): row for row in rows}
        header = read_csv_fieldnames(other)
        missing = sorted(set(base_by_key) - set(by_key))
        added = sorted(set(by_key) - set(base_by_key))
        changed = []
        for key in sorted(set(base_by_key) & set(by_key)):
            base_row = base_by_key[key]
            other_row = by_key[key]
            fields = sorted(set(base_row) | set(other_row))
            changed_fields = [field for field in fields if base_row.get(field, "") != other_row.get(field, "")]
            if changed_fields:
                changed.append({"source_name": key[0], "target_name": key[1], "changed_fields": changed_fields})
                for field in changed_fields:
                    diff_rows.append({
                        "mapping": str(other),
                        "source_name": key[0],
                        "target_name": key[1],
                        "diff_type": "field_changed",
                        "field": field,
                        "base_value": base_row.get(field, ""),
                        "other_value": other_row.get(field, ""),
                    })
        for source, target in missing:
            diff_rows.append({"mapping": str(other), "source_name": source, "target_name": target, "diff_type": "missing_row", "field": "", "base_value": "", "other_value": ""})
        for source, target in added:
            diff_rows.append({"mapping": str(other), "source_name": source, "target_name": target, "diff_type": "added_row", "field": "", "base_value": "", "other_value": ""})
        report = {
            "mapping": str(other),
            "sha256": sha256_file(other),
            "header": header,
            "header_matches": header == base_header,
            "row_count": len(rows),
            "row_count_delta": len(rows) - len(base_rows),
            "missing_rows": [{"source_name": s, "target_name": t} for s, t in missing],
            "added_rows": [{"source_name": s, "target_name": t} for s, t in added],
            "changed_rows": changed,
            "status_changes": [item for item in changed if "status" in item["changed_fields"]],
            "reason_notes_changes": [item for item in changed if {"reason", "notes"} & set(item["changed_fields"])],
        }
        reports.append(report)
    status = "success" if all(
        item["header_matches"] and item["row_count_delta"] == 0 and not item["missing_rows"] and not item["added_rows"] and not item["changed_rows"]
        for item in reports
    ) else "different"
    return {
        "status": status,
        "base_mapping": str(base),
        "base_sha256": sha256_file(base),
        "base_header": base_header,
        "base_row_count": len(base_rows),
        "comparisons": reports,
        "diff_count": len(diff_rows),
        "diff_rows": diff_rows,
    }


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Mapping Comparison",
        "",
        f"- status: `{report['status']}`",
        f"- base: `{report['base_mapping']}`",
        f"- base rows: `{report['base_row_count']}`",
        f"- diff rows: `{report['diff_count']}`",
    ]
    for item in report["comparisons"]:
        lines.extend([
            "",
            f"## {item['mapping']}",
            f"- header_matches: `{item['header_matches']}`",
            f"- row_count: `{item['row_count']}`",
            f"- row_count_delta: `{item['row_count_delta']}`",
            f"- missing_rows: `{len(item['missing_rows'])}`",
            f"- added_rows: `{len(item['added_rows'])}`",
            f"- changed_rows: `{len(item['changed_rows'])}`",
            f"- status_changes: `{len(item['status_changes'])}`",
            f"- reason_notes_changes: `{len(item['reason_notes_changes'])}`",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Task 1 rename mapping copies without modifying them.")
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--compare", required=True, action="append", type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", type=Path)
    p.add_argument("--output-csv", type=Path)
    args = p.parse_args()
    report = compare_mappings(args.base, args.compare)
    diff_rows = report.pop("diff_rows")
    write_json(args.output_json, report)
    if args.output_md:
        write_md(args.output_md, report)
    if args.output_csv:
        write_csv(args.output_csv, diff_rows, ["mapping", "source_name", "target_name", "diff_type", "field", "base_value", "other_value"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
