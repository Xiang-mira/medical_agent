#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import (  # noqa: E402
    NIFTI_SUFFIX,
    load_task2_targets,
    read_alias_groups,
    read_boundary_classification,
    read_csv_rows,
    read_rename_mapping,
    safe_label_name,
    validate_rename_mapping,
    write_csv,
    write_json,
)


ROW_FIELDS = [
    "case_id",
    "source_name",
    "target_name",
    "alias_group",
    "source_path",
    "target_path",
    "status",
    "reason",
    "mode",
]


def _case_mask_dir(data_root: Path, row: dict[str, str]) -> Path:
    ref = row.get("reference_mask_dir") or row.get("mask_dir") or ""
    if ref:
        path = Path(ref)
        return path if path.is_absolute() else data_root / path
    return data_root / row["case_id"] / "segmentations"


def _output_mask_dir(source_mask_dir: Path, data_root: Path, output_data_root: Path, case_id: str) -> Path:
    try:
        rel = source_mask_dir.resolve().relative_to(data_root.resolve())
    except Exception:
        rel = Path(case_id) / "segmentations"
    return output_data_root / rel


def _copy_once(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)


def verify_task1_hpc(
    *,
    data_root: Path,
    case_manifest: Path,
    mapping: Path,
    taxonomy: Path,
    task2_targets: Path,
    alias_groups: Path,
    output_root: Path,
    apply: bool = False,
    output_data_root: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if apply:
        if output_data_root is None:
            raise RuntimeError("--apply requires --output-data-root")
        if output_data_root.resolve() == data_root.resolve():
            raise RuntimeError("--output-data-root must not equal --data-root")
    boundary = mapping.parent / "task_boundary_classification.csv"
    validation_path = output_root / "task1_mapping_validation.json"
    validation = validate_rename_mapping(
        mapping,
        taxonomy,
        validation_path,
        task2_targets=task2_targets,
        alias_groups=alias_groups,
        boundary_classification=boundary if boundary.exists() else None,
    )
    task2 = load_task2_targets(task2_targets)
    aliases = read_alias_groups(alias_groups)
    boundary_rows = read_boundary_classification(boundary if boundary.exists() else None)
    alias_for_source_target = {
        (source, target): str(alias["alias_group"])
        for target, alias in aliases.items()
        for source in alias["sources"]
    }
    mapping_rows = read_rename_mapping(mapping)
    cases = read_csv_rows(case_manifest)
    rows: list[dict[str, Any]] = []
    coexistence_rows: list[dict[str, Any]] = []
    mode = "apply" if apply else "dry_run"
    counts: dict[str, int] = {
        "would_rename": 0,
        "renamed": 0,
        "already_normalized": 0,
        "source_missing": 0,
        "conflict": 0,
        "skipped_pending_review": 0,
        "skipped_task2_generate": 0,
        "skipped_excluded": 0,
        "task2_overlap": 0,
        "invalid_taxonomy": 0,
        "alias_group_coexistence": 0,
    }
    for case in cases:
        case_id = case.get("case_id", "")
        source_mask_dir = _case_mask_dir(data_root, case)
        exec_mask_dir = source_mask_dir
        if apply and output_data_root is not None and source_mask_dir.exists():
            exec_mask_dir = _output_mask_dir(source_mask_dir, data_root, output_data_root, case_id)
            _copy_once(source_mask_dir, exec_mask_dir)
        present = {safe_label_name(path.name): path for path in exec_mask_dir.glob(f"*{NIFTI_SUFFIX}")} if exec_mask_dir.exists() else {}
        coexistence_targets: set[str] = set()
        for target, alias in aliases.items():
            present_aliases = sorted(set(alias["sources"]) & set(present))
            if len(present_aliases) > 1:
                coexistence_targets.add(target)
                counts["alias_group_coexistence"] += 1
                coexistence_rows.append({
                    "case_id": case_id,
                    "target_name": target,
                    "alias_group": alias["alias_group"],
                    "present_sources": ";".join(present_aliases),
                    "mask_dir": str(exec_mask_dir),
                })
        for item in mapping_rows:
            boundary_row = boundary_rows.get((item.source_name, item.target_name), {})
            alias_group = alias_for_source_target.get((item.source_name, item.target_name), "")
            src = exec_mask_dir / f"{item.source_name}{NIFTI_SUFFIX}"
            dst = exec_mask_dir / f"{item.target_name}{NIFTI_SUFFIX}"
            row = {
                "case_id": case_id,
                "source_name": item.source_name,
                "target_name": item.target_name,
                "alias_group": alias_group,
                "source_path": str(src),
                "target_path": str(dst),
                "status": "",
                "reason": "",
                "mode": mode,
            }
            classification = boundary_row.get("classification", "")
            if item.target_name in task2:
                row.update({"status": "task2_overlap", "reason": "target_is_fixed_task2_generate"})
            elif item.target_name not in {r.target_name for r in mapping_rows} and not boundary_row:
                row.update({"status": "invalid_taxonomy", "reason": "invalid_target"})
            elif item.status == "pending_review":
                row.update({"status": "skipped_pending_review", "reason": "mapping_status_pending_review"})
            elif classification == "task2_generate":
                row.update({"status": "skipped_task2_generate", "reason": "boundary_classification_task2_generate"})
            elif classification == "exclude" or item.status == "rejected":
                row.update({"status": "skipped_excluded", "reason": "boundary_classification_exclude"})
            elif item.target_name in coexistence_targets:
                row.update({"status": "conflict", "reason": "alias_group_coexistence"})
            elif src.exists() and dst.exists():
                row.update({"status": "conflict", "reason": "target_exists_no_overwrite"})
            elif not src.exists() and dst.exists():
                row.update({"status": "already_normalized", "reason": "target_exists_source_missing"})
            elif not src.exists():
                row.update({"status": "source_missing", "reason": "source_missing"})
            else:
                if apply:
                    src.rename(dst)
                    row.update({"status": "renamed", "reason": "renamed_in_output_data_root"})
                else:
                    row.update({"status": "would_rename", "reason": "confirmed_task1_rename"})
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            rows.append(row)
    write_csv(output_root / "task1_hpc_dry_run_rows.csv", rows, ROW_FIELDS)
    write_csv(output_root / "task1_alias_coexistence.csv", coexistence_rows, ["case_id", "target_name", "alias_group", "present_sources", "mask_dir"])
    summary = {
        "status": "success",
        "mode": mode,
        "read_only": not apply,
        "data_root": str(data_root),
        "case_manifest": str(case_manifest),
        "case_count": len(cases),
        "mapping": str(mapping),
        "output_root": str(output_root),
        "output_data_root": str(output_data_root) if output_data_root else None,
        "status_counts": counts,
        "mapping_validation": validation,
        "rows": len(rows),
    }
    write_json(output_root / "task1_hpc_dry_run_summary.json", summary)
    write_report(output_root / "task1_hpc_dry_run_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 1 HPC Dry Run",
        "",
        f"- status: `{summary['status']}`",
        f"- mode: `{summary['mode']}`",
        f"- read_only: `{summary['read_only']}`",
        f"- cases: `{summary['case_count']}`",
        f"- rows: `{summary['rows']}`",
        "",
        "## Status Counts",
    ]
    for key, value in sorted(summary["status_counts"].items()):
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Read-only Task 1 HPC verification over a fixed case manifest.")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--case-manifest", required=True, type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--task2-targets", required=True, type=Path)
    p.add_argument("--alias-groups", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--apply", action="store_true", default=False)
    p.add_argument("--output-data-root", type=Path)
    args = p.parse_args()
    try:
        result = verify_task1_hpc(
            data_root=args.data_root,
            case_manifest=args.case_manifest,
            mapping=args.mapping,
            taxonomy=args.taxonomy,
            task2_targets=args.task2_targets,
            alias_groups=args.alias_groups,
            output_root=args.output_root,
            apply=args.apply,
            output_data_root=args.output_data_root,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "error": str(exc)}
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "task1_hpc_dry_run_summary.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
