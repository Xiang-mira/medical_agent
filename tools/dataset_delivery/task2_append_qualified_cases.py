#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_qualified_cases(
    *,
    base_manifest: Path,
    candidate_report: Path,
    output_manifest: Path,
    audit_json: Path | None = None,
    target_group: str | None = None,
    max_cases: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    base_rows = _read_csv(base_manifest)
    candidate_rows = _read_csv(candidate_report)
    fields: list[str] = []
    for row in base_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for key in ("index", "case_id", "ct_path", "annotation_folder", "append_reason", "candidate_target_group", "fov_reason"):
        if key not in fields:
            fields.append(key)
    seen = {str(row.get("case_id") or row.get("id") or "").strip() for row in base_rows}
    final_rows = [dict(row, index=index) for index, row in enumerate(base_rows)]
    audit_rows: list[dict[str, Any]] = []
    appended: list[str] = []
    for row in sorted(candidate_rows, key=lambda item: str(item.get("case_id") or "")):
        if max_cases is not None and len(appended) >= max_cases:
            break
        case_id = str(row.get("case_id") or row.get("id") or "").strip()
        if not case_id:
            continue
        groups = {item.strip() for item in str(row.get("candidate_target_group") or "").split(",") if item.strip()}
        if target_group and target_group not in groups:
            continue
        if case_id in seen:
            audit_rows.append({"case_id": case_id, "action": "duplicate_skipped", "reason": row.get("reason", "")})
            continue
        out = {
            "index": len(final_rows),
            "case_id": case_id,
            "ct_path": row.get("ct_path", ""),
            "annotation_folder": row.get("annotation_folder", ""),
            "append_reason": "fov_candidate_search",
            "candidate_target_group": row.get("candidate_target_group", ""),
            "fov_reason": row.get("reason", ""),
        }
        final_rows.append(out)
        seen.add(case_id)
        appended.append(case_id)
        audit_rows.append({"case_id": case_id, "action": "appended", "reason": row.get("reason", "")})
    audit = {
        "status": "DRY_RUN" if dry_run else "READY",
        "generated_at": _utc_timestamp(),
        "base_manifest": str(base_manifest),
        "candidate_report": str(candidate_report),
        "output_manifest": str(output_manifest),
        "base_count": len(base_rows),
        "appended_count": len(appended),
        "final_count": len(final_rows),
        "appended_case_ids": appended,
        "audit_rows": audit_rows,
        "append_only": True,
        "target_group_filter": target_group or "",
    }
    if not dry_run:
        _write_csv(output_manifest, final_rows, fields)
    if audit_json:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Append FOV-qualified Task2 cases without changing existing manifest order.")
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--target-group", choices=["head", "thorax", "central_airway"])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit = append_qualified_cases(
        base_manifest=args.base_manifest,
        candidate_report=args.candidate_report,
        output_manifest=args.output_manifest,
        audit_json=args.audit_json,
        target_group=args.target_group,
        max_cases=args.max_cases,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "status": audit["status"],
        "base_count": audit["base_count"],
        "appended_count": audit["appended_count"],
        "final_count": audit["final_count"],
        "appended_case_ids": audit["appended_case_ids"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
