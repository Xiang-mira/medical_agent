#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


NIFTI_SUFFIX = ".nii.gz"
MAPPING_FIELDS = ["current_name", "target_name", "issue_type", "action", "status", "notes"]
REPORT_FIELDS = [
    "case_id",
    "current_name",
    "target_name",
    "source_path",
    "target_path",
    "mode",
    "status",
    "reason",
    "action",
]
ALLOWED_ISSUE_TYPES = {
    "word_order",
    "synonym",
    "language_variant",
    "abbreviation",
    "spelling",
    "laterality",
    "granularity_mismatch",
    "missing_label",
}
ALLOWED_ACTIONS = {"rename", "generate", "manual_review"}
ALLOWED_STATUSES = {"confirmed", "blocked", "needs_review"}


class RenameError(RuntimeError):
    pass


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RenameError(f"mapping file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RenameError(f"mapping file has no header: {path}")
        missing = [field for field in MAPPING_FIELDS if field not in reader.fieldnames]
        if missing:
            raise RenameError(f"mapping file missing required columns: {missing}")
        return [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]


def validate_label_name(value: str, *, required: bool) -> str:
    name = value.strip()
    if not name:
        if required:
            raise ValueError("missing label name")
        return ""
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"unsafe label name: {value!r}")
    if name.endswith(".nii") or name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"label name must not include a file suffix: {value!r}")
    return name


def side_of(name: str) -> str:
    tokens = [token.lower() for token in name.split("_") if token]
    has_left = "left" in tokens
    has_right = "right" in tokens
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if has_left and has_right:
        return "mixed"
    return ""


def laterality_reason(current_name: str, target_name: str) -> str:
    current_side = side_of(current_name)
    target_side = side_of(target_name)
    if "mixed" in {current_side, target_side}:
        return "laterality_mismatch"
    if current_side and target_side and current_side != target_side:
        return "laterality_mismatch"
    if current_side and not target_side:
        return "laterality_mismatch"
    if target_side and not current_side:
        return "laterality_mismatch"
    return ""


def validate_mapping(path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows = read_csv_rows(path)
    errors: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, str]] = []
    seen_records: set[tuple[str, str, str, str, str, str]] = set()
    current_to_targets: dict[str, set[str]] = {}

    for row_number, row in enumerate(rows, start=2):
        cleaned = {field: row.get(field, "").strip() for field in MAPPING_FIELDS}
        normalized_rows.append(cleaned)
        record_key = tuple(cleaned[field] for field in MAPPING_FIELDS)
        if record_key in seen_records:
            errors.append({"row": row_number, "type": "duplicate_record"})
        seen_records.add(record_key)

        if cleaned["issue_type"] not in ALLOWED_ISSUE_TYPES:
            errors.append({"row": row_number, "type": "invalid_issue_type", "value": cleaned["issue_type"]})
        if cleaned["action"] not in ALLOWED_ACTIONS:
            errors.append({"row": row_number, "type": "invalid_action", "value": cleaned["action"]})
        if cleaned["status"] not in ALLOWED_STATUSES:
            errors.append({"row": row_number, "type": "invalid_status", "value": cleaned["status"]})

        try:
            validate_label_name(cleaned["current_name"], required=cleaned["action"] == "rename")
        except ValueError as exc:
            errors.append({"row": row_number, "type": "invalid_current_name", "message": str(exc)})
        try:
            validate_label_name(cleaned["target_name"], required=cleaned["action"] == "rename")
        except ValueError as exc:
            errors.append({"row": row_number, "type": "invalid_target_name", "message": str(exc)})

        if cleaned["action"] == "rename" and not cleaned["target_name"]:
            errors.append({"row": row_number, "type": "rename_missing_target_name"})
        if cleaned["action"] == "rename" and cleaned["current_name"] == cleaned["target_name"]:
            errors.append({"row": row_number, "type": "rename_source_equals_target", "current_name": cleaned["current_name"]})
        if cleaned["current_name"] and cleaned["target_name"]:
            current_to_targets.setdefault(cleaned["current_name"], set()).add(cleaned["target_name"])
        if cleaned["action"] == "rename" and cleaned["current_name"] and cleaned["target_name"]:
            reason = laterality_reason(cleaned["current_name"], cleaned["target_name"])
            if reason:
                errors.append(
                    {
                        "row": row_number,
                        "type": "laterality_mismatch",
                        "current_name": cleaned["current_name"],
                        "target_name": cleaned["target_name"],
                    }
                )

    for current_name, target_names in sorted(current_to_targets.items()):
        rename_targets = {
            row["target_name"]
            for row in normalized_rows
            if row["current_name"] == current_name and row["action"] == "rename"
        }
        if len(rename_targets) > 1:
            errors.append(
                {
                    "type": "one_current_name_to_multiple_rename_targets",
                    "current_name": current_name,
                    "target_names": sorted(rename_targets),
                }
            )
        elif len(target_names) > 1:
            errors.append(
                {
                    "type": "one_current_name_to_multiple_targets",
                    "current_name": current_name,
                    "target_names": sorted(target_names),
                }
            )

    return normalized_rows, errors


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise RenameError(f"case manifest not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if "," not in lines[0]:
        return list(dict.fromkeys(lines))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        preferred = "case_id" if "case_id" in reader.fieldnames else reader.fieldnames[0]
        return list(dict.fromkeys(str(row.get(preferred) or "").strip() for row in reader if str(row.get(preferred) or "").strip()))


def case_dir_for_id(data_root: Path, case_id: str, segmentations_subdir: str) -> Path:
    direct_seg = data_root / segmentations_subdir
    if data_root.name == case_id and direct_seg.is_dir():
        return data_root
    return data_root / case_id


def discover_cases(data_root: Path, segmentations_subdir: str, case_ids: list[str], manifest: Path | None) -> list[tuple[str, Path, Path]]:
    root = data_root.resolve()
    if not root.exists():
        raise RenameError(f"data-root does not exist: {root}")
    selected_ids = list(case_ids)
    if manifest is not None:
        selected_ids.extend(read_manifest(manifest))
    selected_ids = list(dict.fromkeys(selected_ids))

    candidates: list[Path]
    if selected_ids:
        candidates = [case_dir_for_id(root, case_id, segmentations_subdir) for case_id in selected_ids]
    elif (root / segmentations_subdir).is_dir():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / segmentations_subdir).is_dir())

    cases: list[tuple[str, Path, Path]] = []
    for case_dir in candidates:
        seg_dir = case_dir / segmentations_subdir
        if seg_dir.is_dir():
            cases.append((case_dir.name, case_dir, seg_dir))
    if not cases:
        raise RenameError(f"no case segmentation directories found under: {root}")
    return cases


def mask_path(mask_dir: Path, label_name: str) -> Path:
    return mask_dir / f"{label_name}{NIFTI_SUFFIX}"


def build_plan(
    mappings: list[dict[str, str]],
    validation_errors: list[dict[str, Any]],
    cases: list[tuple[str, Path, Path]],
    mode: str,
) -> list[dict[str, str]]:
    invalid_rename_keys: set[tuple[str, str]] = set()
    for error in validation_errors:
        if error.get("type") == "laterality_mismatch":
            invalid_rename_keys.add((str(error.get("current_name") or ""), str(error.get("target_name") or "")))

    report_rows: list[dict[str, str]] = []
    rename_mappings = [row for row in mappings if row["action"] == "rename" and row["status"] == "confirmed"]

    for case_id, _case_dir, mask_dir in cases:
        present_sources_by_target: dict[str, list[str]] = {}
        for row in rename_mappings:
            current_name = row["current_name"]
            target_name = row["target_name"]
            if current_name and mask_path(mask_dir, current_name).exists():
                present_sources_by_target.setdefault(target_name, []).append(current_name)
        blocked_targets = {
            target_name
            for target_name, current_names in present_sources_by_target.items()
            if len(set(current_names)) > 1
        }

        for row in mappings:
            current_name = row["current_name"]
            target_name = row["target_name"]
            source = mask_path(mask_dir, current_name) if current_name else None
            target = mask_path(mask_dir, target_name) if target_name else None
            report_row = {
                "case_id": case_id,
                "current_name": current_name,
                "target_name": target_name,
                "source_path": str(source) if source else "",
                "target_path": str(target) if target else "",
                "mode": mode,
                "status": "",
                "reason": "",
                "action": row["action"],
            }

            if row["action"] != "rename":
                report_row["status"] = "skipped"
                report_row["reason"] = f"{row['action']}_not_executed"
            elif row["status"] != "confirmed":
                report_row["status"] = "skipped"
                report_row["reason"] = f"status_{row['status']}_not_executed"
            elif (current_name, target_name) in invalid_rename_keys or laterality_reason(current_name, target_name):
                report_row["status"] = "invalid_mapping"
                report_row["reason"] = "laterality_mismatch"
            elif target_name in blocked_targets:
                report_row["status"] = "conflict"
                report_row["reason"] = "multiple_sources_to_same_target_in_case"
            elif source is not None and target is not None and source.exists() and target.exists():
                report_row["status"] = "conflict"
                report_row["reason"] = "source_and_target_exist"
            elif source is not None and target is not None and not source.exists() and target.exists():
                report_row["status"] = "already_normalized"
                report_row["reason"] = "source_missing_target_exists"
            elif source is not None and target is not None and not source.exists() and not target.exists():
                report_row["status"] = "source_missing"
                report_row["reason"] = "source_and_target_missing"
            else:
                report_row["status"] = "would_rename"
                report_row["reason"] = "confirmed_mapping"
            report_rows.append(report_row)
    return report_rows


def apply_plan(report_rows: list[dict[str, str]]) -> None:
    for row in report_rows:
        if row["status"] != "would_rename":
            continue
        source = Path(row["source_path"])
        target = Path(row["target_path"])
        if not source.exists():
            row["status"] = "source_missing"
            row["reason"] = "source_missing_before_apply"
            continue
        if target.exists():
            row["status"] = "conflict"
            row["reason"] = "target_exists_before_apply"
            continue
        source.rename(target)
        row["status"] = "renamed"


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely rename anatomical NIfTI label files using a standalone CSV mapping.")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root or a single case directory.")
    parser.add_argument("--mapping-file", required=True, type=Path, help="Path to taxonomy_373_mapping.csv.")
    parser.add_argument("--report", required=True, type=Path, help="Output audit report CSV.")
    parser.add_argument("--segmentations-subdir", default="segmentations", help="Mask subdirectory inside each case.")
    parser.add_argument("--case-id", action="append", default=[], help="Limit processing to one case ID; can be repeated.")
    parser.add_argument("--case-manifest", type=Path, help="Optional CSV or line-based case manifest.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero for invalid mapping rows or laterality errors.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Scan and report without modifying files.")
    mode.add_argument("--apply", action="store_true", help="Execute safe, conflict-free renames.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    mode = "apply" if args.apply else "dry_run"
    mappings, validation_errors = validate_mapping(args.mapping_file)
    cases = discover_cases(args.data_root, args.segmentations_subdir, args.case_id, args.case_manifest)
    rows = build_plan(mappings, validation_errors, cases, mode)
    if args.apply:
        apply_plan(rows)
    write_report(args.report, rows)

    counts = status_counts(rows)
    invalid_mapping_count = counts.get("invalid_mapping", 0)
    exit_code = 0
    if args.strict and (validation_errors or invalid_mapping_count):
        exit_code = 2
    summary = {
        "status": "failed" if exit_code else "success",
        "mode": mode,
        "cases_scanned": len(cases),
        "operations": len(rows),
        "status_counts": counts,
        "mapping_errors": validation_errors,
        "report": str(args.report),
    }
    return summary, exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        summary, exit_code = run(parse_args(sys.argv[1:] if argv is None else argv))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
