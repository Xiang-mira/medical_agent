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

from tools.dataset_delivery.delivery_lib import (  # noqa: E402
    ALLOWED_BOUNDARY_CLASSIFICATION,
    ALLOWED_BOUNDARY_STATUS,
    load_task2_targets,
    load_taxonomy_names,
    read_alias_groups,
    read_boundary_classification,
    read_rename_mapping,
    safe_label_name,
    validate_rename_mapping,
    write_csv,
    write_json,
)


FIELDS = [
    "source_name",
    "target_name",
    "same_anatomy",
    "same_laterality",
    "same_granularity",
    "requires_voxel_change",
    "classification",
    "status",
    "reason",
    "evidence",
    "alias_group",
]


def _bool_text(value: str) -> str:
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "false"


def build_boundary_audit(
    mapping: Path,
    taxonomy: Path,
    task2_targets: Path,
    alias_groups: Path,
    *,
    boundary: Path | None,
    non_rename_decisions: Path | None,
) -> dict[str, Any]:
    taxonomy_names = set(load_taxonomy_names(taxonomy))
    task2 = load_task2_targets(task2_targets)
    aliases = read_alias_groups(alias_groups)
    boundary_rows = read_boundary_classification(boundary) if boundary else {}
    mapping_rows = read_rename_mapping(mapping)
    errors: list[dict[str, Any]] = []
    output_rows: list[dict[str, str]] = []

    for row in mapping_rows:
        key = (row.source_name, row.target_name)
        boundary_row = boundary_rows.get(key)
        if not boundary_row:
            errors.append({"type": "missing_boundary_row", "source_name": row.source_name, "target_name": row.target_name})
            continue
        out = {field: boundary_row.get(field, "") for field in FIELDS}
        out["source_name"] = safe_label_name(out["source_name"])
        out["target_name"] = safe_label_name(out["target_name"])
        out["same_anatomy"] = _bool_text(out["same_anatomy"])
        out["same_laterality"] = _bool_text(out["same_laterality"])
        out["same_granularity"] = _bool_text(out["same_granularity"])
        out["requires_voxel_change"] = _bool_text(out["requires_voxel_change"])
        if out["classification"] not in ALLOWED_BOUNDARY_CLASSIFICATION:
            errors.append({"type": "invalid_classification", "source_name": row.source_name, "target_name": row.target_name, "classification": out["classification"]})
        if out["status"] not in ALLOWED_BOUNDARY_STATUS:
            errors.append({"type": "invalid_status", "source_name": row.source_name, "target_name": row.target_name, "status": out["status"]})
        if out["target_name"] not in taxonomy_names:
            errors.append({"type": "target_not_in_taxonomy", "target_name": out["target_name"]})
        if row.status == "confirmed" and (out["classification"] != "task1_rename" or out["status"] != "confirmed"):
            errors.append({"type": "mapping_boundary_status_mismatch", "source_name": row.source_name, "target_name": row.target_name})
        if row.status == "pending_review" and (out["classification"] != "boundary_review" or out["status"] != "pending_review"):
            errors.append({"type": "pending_boundary_status_mismatch", "source_name": row.source_name, "target_name": row.target_name})
        if out["classification"] == "task1_rename" and out["status"] == "confirmed":
            if out["target_name"] in task2:
                errors.append({"type": "task2_overlap", "target_name": out["target_name"]})
            if not out["evidence"]:
                errors.append({"type": "missing_evidence", "source_name": out["source_name"], "target_name": out["target_name"]})
            if not (out["same_anatomy"] == "true" and out["same_laterality"] == "true" and out["same_granularity"] == "true" and out["requires_voxel_change"] == "false"):
                errors.append({"type": "confirmed_boundary_condition_failed", "source_name": out["source_name"], "target_name": out["target_name"]})
        if out["alias_group"]:
            alias = aliases.get(out["target_name"])
            if not alias or out["source_name"] not in alias["sources"]:
                errors.append({"type": "alias_group_reference_missing", "source_name": out["source_name"], "target_name": out["target_name"], "alias_group": out["alias_group"]})
        output_rows.append(out)

    validation = {}
    try:
        validation = validate_rename_mapping(
            mapping,
            taxonomy,
            task2_targets=task2_targets,
            alias_groups=alias_groups,
            boundary_classification=boundary,
        )
    except Exception as exc:
        errors.append({"type": "mapping_validation_failed", "error": str(exc)})

    counts: dict[str, int] = {}
    for row in output_rows:
        key = f"{row['classification']}:{row['status']}"
        counts[key] = counts.get(key, 0) + 1
    high_risk = [
        row for row in output_rows
        if row["source_name"] in {"parotid_gland", "submandibular_gland", "celiac_aa", "celiac_artery", "celiac_truck"}
    ]
    non_rename_count = 0
    if non_rename_decisions and non_rename_decisions.exists():
        from tools.dataset_delivery.delivery_lib import read_csv_rows

        non_rename_count = len(read_csv_rows(non_rename_decisions))
    return {
        "status": "failed" if errors else "success",
        "mapping": str(mapping),
        "taxonomy": str(taxonomy),
        "task2_targets": str(task2_targets),
        "alias_groups": str(alias_groups),
        "rows": len(output_rows),
        "counts": counts,
        "confirmed_count": sum(1 for row in output_rows if row["classification"] == "task1_rename" and row["status"] == "confirmed"),
        "pending_review_count": sum(1 for row in output_rows if row["status"] == "pending_review"),
        "task2_generate_count": sum(1 for row in output_rows if row["classification"] == "task2_generate"),
        "exclude_count": sum(1 for row in output_rows if row["classification"] == "exclude"),
        "task2_overlap_count": len(set(row["target_name"] for row in output_rows if row["status"] == "confirmed") & task2),
        "many_to_one_alias_group_count": len([target for target, alias in aliases.items() if len(alias["sources"]) > 1]),
        "non_rename_decision_count": non_rename_count,
        "high_risk_decisions": high_risk,
        "mapping_validation": validation,
        "errors": errors,
        "rows_out": output_rows,
    }


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Boundary Audit",
        "",
        f"- status: `{report['status']}`",
        f"- mapping rows: `{report['rows']}`",
        f"- confirmed: `{report['confirmed_count']}`",
        f"- pending_review: `{report['pending_review_count']}`",
        f"- task2_generate: `{report['task2_generate_count']}`",
        f"- exclude: `{report['exclude_count']}`",
        f"- Task 2 overlap: `{report['task2_overlap_count']}`",
        f"- many-to-one alias groups: `{report['many_to_one_alias_group_count']}`",
        "",
        "## High Risk Decisions",
    ]
    for row in report["high_risk_decisions"]:
        lines.append(f"- `{row['source_name']} -> {row['target_name']}`: `{row['classification']}/{row['status']}`; {row['reason']}; evidence: {row['evidence']}")
    if report["errors"]:
        lines.extend(["", "## Errors"])
        for error in report["errors"][:200]:
            lines.append(f"- `{error.get('type', 'error')}`: {json.dumps(error, ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Audit Task 1 rename boundary decisions.")
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--task2-targets", required=True, type=Path)
    p.add_argument("--alias-groups", required=True, type=Path)
    p.add_argument("--boundary-classification", type=Path)
    p.add_argument("--non-rename-decisions", type=Path)
    p.add_argument("--output-csv", type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    args = p.parse_args()
    boundary = args.boundary_classification or args.mapping.parent / "task_boundary_classification.csv"
    report = build_boundary_audit(
        args.mapping,
        args.taxonomy,
        args.task2_targets,
        args.alias_groups,
        boundary=boundary,
        non_rename_decisions=args.non_rename_decisions,
    )
    rows_out = report.pop("rows_out")
    if args.output_csv:
        write_csv(args.output_csv, rows_out, FIELDS)
    write_json(args.output_json, report)
    write_md(args.output_md, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
