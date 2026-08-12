#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import DeliveryError, read_csv_rows, sha256_file, utc_now, write_csv, write_json  # noqa: E402
DEFAULT_BASE_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)
DEFAULT_APPEND_CASES = REPO_ROOT / "configs" / "dataset_delivery" / "task2_formal_append_cases.csv"
DEFAULT_OUTPUT_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_103cases_work/cases_103_manifest.csv"
)

FORMAL_CASE_COUNT = 103
FORMAL_APPEND_CASE_IDS = ("BDMAP_00000001", "BDMAP_00000002", "BDMAP_00000006")
FORMAL_MODEL_TARGETS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        (
            "cads",
            [
                "blood",
                "cerebrospinal_fluid",
                "common_iliac_artery_left",
                "common_iliac_artery_right",
                "common_iliac_vein_left",
                "common_iliac_vein_right",
                "compact_bone",
                "eyeball",
                "face",
                "gland_structure",
                "gray_matter",
                "muscle_of_head",
                "scalp",
                "spongy_bone",
                "white_matter",
            ],
        ),
        ("atm", ["airway_tree"]),
        (
            "airrc",
            [
                "airway_wall",
                "lung_pulmonary_arteries",
                "lung_pulmonary_veins",
            ],
        ),
        (
            "unest",
            [
                "kidney_cortex",
                "kidney_medulla",
                "kidney_pelvicalyceal_system",
            ],
        ),
    ]
)
FORMAL_GROUP_MODELS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("cads", ["cads553", "cads557", "cads559"]),
        ("atm", ["atm"]),
        ("airrc", ["airrc"]),
        ("unest", ["unest"]),
    ]
)
FORMAL_TARGET_TO_GROUP = {
    target: group
    for group, targets in FORMAL_MODEL_TARGETS.items()
    for target in targets
}
FORMAL_TARGETS = tuple(FORMAL_TARGET_TO_GROUP)
FORMAL_MODEL_KEYS = tuple(model for models in FORMAL_GROUP_MODELS.values() for model in models)
FORMAL_REQUIRED_COLUMNS = ("index", "case_id", "annotation_folder")
FORMAL_IMAGE_COLUMNS = ("ct_path", "image_path")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    write_csv(path, rows, fieldnames)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def _case_id(row: dict[str, str], index: int | None = None) -> str:
    value = row.get("case_id") or row.get("id") or ""
    if value:
        return str(value).strip()
    return f"case_{index:03d}" if index is not None else ""


def _ct_path(row: dict[str, str]) -> str:
    return str(row.get("ct_path") or row.get("image_path") or "").strip()


def _annotation_folder(row: dict[str, str]) -> str:
    return str(row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or "").strip()


def _ordered_fieldnames(base_rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for row in base_rows:
        for key in row:
            if key and key not in fields:
                fields.append(key)
    for required in ("index", "case_id", "ct_path", "annotation_folder", "append_reason"):
        if required not in fields:
            fields.append(required)
    return fields


def _materialize_formal_row(
    row: dict[str, str],
    *,
    index: int,
    image_root: Path | None = None,
    mask_root: Path | None = None,
    append_reason: str = "",
) -> dict[str, Any]:
    """Return one row with the canonical formal manifest columns present.

    Existing source paths win.  The root-based fallback is used only when a
    compatible legacy column is absent or empty.
    """
    case_id = _case_id(row, index)
    ct_path = _ct_path(row)
    annotation_folder = _annotation_folder(row)
    resolved: dict[str, Any] = dict(row)
    resolved["index"] = index
    resolved["case_id"] = case_id
    if not ct_path and image_root is not None and case_id:
        ct_path = str(image_root / case_id / "ct.nii.gz")
    if not annotation_folder and mask_root is not None and case_id:
        annotation_folder = str(mask_root / case_id / "segmentations")
    if not str(resolved.get("ct_path") or "").strip():
        resolved["ct_path"] = ct_path
    if not str(resolved.get("annotation_folder") or "").strip():
        resolved["annotation_folder"] = annotation_folder
    resolved.setdefault("append_reason", append_reason)
    return resolved


def _normalize_manifest_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "case_id": _case_id(row),
        "ct_path": _ct_path(row),
        "annotation_folder": _annotation_folder(row),
    }


def _append_row(case_id: str, reason: str, *, image_root: Path, mask_root: Path, source_row: dict[str, str] | None = None) -> dict[str, Any]:
    row = dict(source_row or {})
    row["case_id"] = case_id
    row.setdefault("append_reason", reason)
    return _materialize_formal_row(
        row,
        index=0,
        image_root=image_root,
        mask_root=mask_root,
        append_reason=reason,
    )


def _validate_rows(
    *,
    base_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    required_append_case_ids: tuple[str, ...],
    check_exists: bool,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    columns = {key for row in final_rows for key in row}
    missing_cols = sorted(set(FORMAL_REQUIRED_COLUMNS) - columns)
    if missing_cols:
        errors.append({"type": "missing_columns", "columns": missing_cols})
    if final_rows and not (set(FORMAL_IMAGE_COLUMNS) & columns):
        errors.append({"type": "missing_columns", "columns": ["ct_path_or_image_path"]})
    if len(final_rows) != FORMAL_CASE_COUNT:
        errors.append({"type": "row_count", "actual": len(final_rows), "expected": FORMAL_CASE_COUNT})
    case_ids = [_case_id(row, index) for index, row in enumerate(final_rows)]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        errors.append({"type": "duplicate_case_id", "case_ids": duplicates})
    unique_count = len(set(case_ids))
    if unique_count != FORMAL_CASE_COUNT:
        errors.append({"type": "unique_case_count", "actual": unique_count, "expected": FORMAL_CASE_COUNT})
    base_prefix = [_normalize_manifest_row(row) for row in base_rows]
    final_prefix = [_normalize_manifest_row(row) for row in final_rows[: len(base_rows)]]
    if base_prefix != final_prefix:
        errors.append({"type": "base_prefix_mismatch", "expected_count": len(base_rows)})
    seen = {}
    for index, row in enumerate(final_rows):
        case_id = _case_id(row, index)
        seen[case_id] = seen.get(case_id, 0) + 1
        index_value = row.get("index")
        if "index" not in row or index_value is None or not str(index_value).strip():
            errors.append({"row": index + 2, "type": "missing_required_value", "case_id": case_id, "column": "index"})
        if not case_id:
            errors.append({"row": index + 2, "type": "missing_required_value", "case_id": case_id, "column": "case_id"})
        if not _ct_path(row):
            errors.append({"row": index + 2, "type": "missing_required_value", "case_id": case_id, "column": "ct_path_or_image_path"})
        if not _annotation_folder(row):
            errors.append({"row": index + 2, "type": "missing_required_value", "case_id": case_id, "column": "annotation_folder"})
        if check_exists:
            ct_path = Path(_ct_path(row))
            ann_dir = Path(_annotation_folder(row))
            if not ct_path.is_file() or ct_path.stat().st_size <= 0:
                errors.append({
                    "row": index + 2,
                    "type": "ct_path_missing_or_empty",
                    "case_id": case_id,
                    "path": str(ct_path),
                })
            if not ann_dir.is_dir():
                errors.append({
                    "row": index + 2,
                    "type": "annotation_folder_missing",
                    "case_id": case_id,
                    "path": str(ann_dir),
                })
    for case_id in required_append_case_ids:
        if seen.get(case_id, 0) != 1:
            errors.append({"type": "required_append_case_missing_or_duplicate", "case_id": case_id, "count": seen.get(case_id, 0)})
    return errors


def build_formal_manifest(
    *,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    append_cases: Path = DEFAULT_APPEND_CASES,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    audit_json: Path | None = None,
    audit_csv: Path | None = None,
    image_root: Path | None = None,
    mask_root: Path | None = None,
    allow_missing: bool = False,
    required_append_case_ids: tuple[str, ...] = FORMAL_APPEND_CASE_IDS,
) -> dict[str, Any]:
    base_rows = _read_manifest(base_manifest)
    append_rows = _read_manifest(append_cases)
    image_root = image_root or Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
    mask_root = mask_root or Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")
    fieldnames = _ordered_fieldnames(base_rows + append_rows)
    seen: set[str] = set()
    final_rows: list[dict[str, Any]] = []
    duplicate_skipped: list[str] = []
    base_case_ids: list[str] = []

    for index, row in enumerate(base_rows):
        case_id = _case_id(row, index)
        if not case_id:
            raise DeliveryError(f"base manifest row {index} has no case_id")
        if case_id in seen:
            duplicate_skipped.append(case_id)
            continue
        seen.add(case_id)
        base_case_ids.append(case_id)
        resolved = _materialize_formal_row(row, index=index, image_root=image_root, mask_root=mask_root)
        final_rows.append(resolved)

    append_audit_rows: list[dict[str, Any]] = []
    appended_case_ids: list[str] = []
    required_append_set = set(required_append_case_ids)
    append_rows_by_case = {
        _case_id(row): dict(row)
        for row in append_rows
        if _case_id(row)
    }
    missing_required_appends = sorted(required_append_set - set(append_rows_by_case))
    if missing_required_appends:
        raise DeliveryError(f"required append cases missing from append manifest: {missing_required_appends}")

    for case_id in required_append_case_ids:
        row = append_rows_by_case[case_id]
        reason = str(row.get("reason") or row.get("append_reason") or "formal_append_case").strip()
        if case_id in seen:
            duplicate_skipped.append(case_id)
            append_audit_rows.append({
                "case_id": case_id,
                "action": "duplicate_in_base",
                "reason": reason,
                "ct_path": "",
                "annotation_folder": "",
                "ct_exists": "",
                "annotation_folder_exists": "",
            })
            continue
        resolved = _append_row(case_id, reason, image_root=image_root, mask_root=mask_root, source_row=row)
        ct_path = Path(resolved["ct_path"])
        ann_dir = Path(resolved["annotation_folder"])
        ct_exists = ct_path.is_file() and ct_path.stat().st_size > 0 if ct_path.exists() else False
        ann_exists = ann_dir.is_dir()
        action = "appended" if (allow_missing or (ct_exists and ann_exists)) else "blocked_missing_input"
        append_audit_rows.append({
            "case_id": case_id,
            "action": action,
            "reason": reason,
            "ct_path": resolved["ct_path"],
            "annotation_folder": resolved["annotation_folder"],
            "ct_exists": ct_exists,
            "annotation_folder_exists": ann_exists,
        })
        if action != "appended":
            continue
        seen.add(case_id)
        appended_case_ids.append(case_id)
        resolved["index"] = len(final_rows)
        final_rows.append(resolved)

    errors = _validate_rows(
        base_rows=base_rows,
        final_rows=final_rows,
        required_append_case_ids=required_append_case_ids,
        check_exists=not allow_missing,
    )
    manifest_sha256 = ""
    if not errors or allow_missing:
        _write_csv(output_manifest, final_rows, fieldnames)
        manifest_sha256 = sha256_file(output_manifest)
    status = "READY" if not errors else "HAS_ERRORS"
    audit = {
        "status": status,
        "generated_at": utc_now(),
        "base_manifest": str(base_manifest),
        "append_cases": str(append_cases),
        "output_manifest": str(output_manifest),
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "base_count": len(base_rows),
        "base_unique_count": len(base_case_ids),
        "append_count": len(appended_case_ids),
        "final_count": len(final_rows),
        "final_unique_count": len({row["case_id"] for row in final_rows}),
        "required_append_case_ids": list(required_append_case_ids),
        "base_case_ids": base_case_ids,
        "appended_case_ids": appended_case_ids,
        "duplicate_skipped": duplicate_skipped,
        "append_audit_rows": append_audit_rows,
        "errors": errors,
        "manifest_sha256": manifest_sha256,
        "provenance": {
            "append_only": True,
            "base_order_preserved": True,
            "indices_rebuilt_contiguously": True,
            "final_unique_case_count": len({row["case_id"] for row in final_rows}),
        },
    }
    if audit_json:
        _write_json(audit_json, audit)
    if audit_csv:
        _write_csv(
            audit_csv,
            append_audit_rows,
            ["case_id", "action", "reason", "ct_path", "annotation_folder", "ct_exists", "annotation_folder_exists"],
        )
    if errors and not allow_missing:
        actual_unique = audit["final_unique_count"]
        raise DeliveryError(
            f"formal manifest validation failed with {len(errors)} error(s); actual_unique_count={actual_unique}"
        )
    return audit


def validate_formal_manifest(
    *,
    manifest: Path,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    required_append_case_ids: tuple[str, ...] = FORMAL_APPEND_CASE_IDS,
    check_exists: bool = True,
    report: Path | None = None,
) -> dict[str, Any]:
    base_rows = _read_manifest(base_manifest)
    final_rows = _read_manifest(manifest)
    errors = _validate_rows(
        base_rows=base_rows,
        final_rows=final_rows,
        required_append_case_ids=required_append_case_ids,
        check_exists=check_exists,
    )
    summary = {
        "status": "success" if not errors else "failed",
        "manifest": str(manifest),
        "base_manifest": str(base_manifest),
        "rows": len(final_rows),
        "unique_case_count": len({row.get("case_id") or row.get("id") for row in final_rows}),
        "required_append_case_ids": list(required_append_case_ids),
        "errors": errors,
    }
    if report:
        _write_json(report, summary)
    if errors:
        raise DeliveryError(f"formal manifest validation failed with {len(errors)} error(s)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate the Task 2 formal 103-case manifest.")
    parser.add_argument("--base-manifest", default=DEFAULT_BASE_MANIFEST, type=Path)
    parser.add_argument("--append-cases", default=DEFAULT_APPEND_CASES, type=Path)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST, type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-exists", action="store_true", default=True)
    parser.add_argument("--no-exists-check", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_formal_manifest(
            manifest=args.output_manifest.resolve(),
            base_manifest=args.base_manifest.resolve(),
            check_exists=not args.no_exists_check,
            report=args.audit_json.resolve() if args.audit_json else None,
        )
    else:
        report = build_formal_manifest(
            base_manifest=args.base_manifest.resolve(),
            append_cases=args.append_cases.resolve(),
            output_manifest=args.output_manifest.resolve(),
            audit_json=args.audit_json.resolve() if args.audit_json else None,
            audit_csv=args.audit_csv.resolve() if args.audit_csv else None,
            image_root=args.image_root.resolve() if args.image_root else None,
            mask_root=args.mask_root.resolve() if args.mask_root else None,
            allow_missing=bool(args.allow_missing),
        )
    print(
        json.dumps(
            {
            "status": report["status"],
            "base_count": report.get("base_count", 0),
            "append_count": report.get("append_count", 0),
            "final_count": report.get("final_count", report.get("rows", 0)),
            "manifest_sha256": report.get("manifest_sha256", ""),
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"READY", "success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
