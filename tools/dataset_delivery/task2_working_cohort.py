#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_APPEND_CASES = Path(__file__).resolve().parents[2] / "configs" / "dataset_delivery" / "task2_append_cases.csv"
DEFAULT_IMAGE_ROOT = Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_MASK_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")


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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _case_id(row: dict[str, str]) -> str:
    return str(row.get("case_id") or row.get("id") or "").strip()


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


def _resolve_append_row(case_id: str, reason: str, *, image_root: Path, mask_root: Path) -> dict[str, str]:
    return {
        "case_id": case_id,
        "ct_path": str(image_root / case_id / "ct.nii.gz"),
        "annotation_folder": str(mask_root / case_id / "segmentations"),
        "append_reason": reason,
    }


def build_working_cohort(
    *,
    base_manifest: Path,
    append_cases: Path = DEFAULT_APPEND_CASES,
    output_manifest: Path,
    audit_json: Path | None = None,
    audit_csv: Path | None = None,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    mask_root: Path = DEFAULT_MASK_ROOT,
    allow_missing: bool = False,
) -> dict[str, Any]:
    base_rows = _read_csv(base_manifest)
    append_rows = _read_csv(append_cases)
    fieldnames = _ordered_fieldnames(base_rows)
    seen: set[str] = set()
    final_rows: list[dict[str, Any]] = []
    duplicate_skipped: list[str] = []
    base_case_ids: list[str] = []

    for index, row in enumerate(base_rows):
        case_id = _case_id(row)
        if not case_id:
            raise ValueError(f"base manifest row {index} has no case_id")
        if case_id in seen:
            duplicate_skipped.append(case_id)
            continue
        seen.add(case_id)
        base_case_ids.append(case_id)
        out = dict(row)
        out["index"] = index
        out.setdefault("append_reason", "")
        final_rows.append(out)

    audit_rows: list[dict[str, Any]] = []
    appended_case_ids: list[str] = []
    for row in append_rows:
        case_id = _case_id(row)
        if not case_id:
            continue
        reason = str(row.get("reason") or row.get("append_reason") or "append_case").strip()
        if case_id in seen:
            duplicate_skipped.append(case_id)
            audit_rows.append({
                "case_id": case_id,
                "action": "duplicate_skipped",
                "reason": reason,
                "ct_path": "",
                "annotation_folder": "",
                "ct_exists": "",
                "annotation_folder_exists": "",
            })
            continue
        resolved = _resolve_append_row(case_id, reason, image_root=image_root, mask_root=mask_root)
        ct_exists = Path(resolved["ct_path"]).exists()
        ref_exists = Path(resolved["annotation_folder"]).exists()
        action = "appended" if (allow_missing or (ct_exists and ref_exists)) else "blocked_missing_input"
        audit_rows.append({
            "case_id": case_id,
            "action": action,
            "reason": reason,
            "ct_path": resolved["ct_path"],
            "annotation_folder": resolved["annotation_folder"],
            "ct_exists": ct_exists,
            "annotation_folder_exists": ref_exists,
        })
        if action != "appended":
            continue
        seen.add(case_id)
        appended_case_ids.append(case_id)
        resolved["index"] = len(final_rows)
        final_rows.append(resolved)

    missing_ct = [row["case_id"] for row in audit_rows if row["action"] == "blocked_missing_input" and not row["ct_exists"]]
    missing_reference_dir = [
        row["case_id"]
        for row in audit_rows
        if row["action"] == "blocked_missing_input" and not row["annotation_folder_exists"]
    ]
    status = "READY" if not missing_ct and not missing_reference_dir else "HAS_MISSING_INPUTS"
    _write_csv(output_manifest, final_rows, fieldnames)
    audit = {
        "status": status,
        "generated_at": _utc_timestamp(),
        "base_manifest": str(base_manifest),
        "append_cases": str(append_cases),
        "output_manifest": str(output_manifest),
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "base_count": len(base_rows),
        "base_unique_count": len(base_case_ids),
        "appended_count": len(appended_case_ids),
        "final_count": len(final_rows),
        "base_case_ids": base_case_ids,
        "appended_case_ids": appended_case_ids,
        "duplicate_skipped": duplicate_skipped,
        "missing_ct": missing_ct,
        "missing_reference_dir": missing_reference_dir,
        "append_audit_rows": audit_rows,
        "provenance": {
            "append_only": True,
            "base_order_preserved": True,
            "indices_rebuilt_contiguously": True,
        },
    }
    if audit_json:
        _write_json(audit_json, audit)
    if audit_csv:
        _write_csv(
            audit_csv,
            audit_rows,
            ["case_id", "action", "reason", "ct_path", "annotation_folder", "ct_exists", "annotation_folder_exists"],
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Task2 working cohort as fixed base manifest plus configured append-only cases.")
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--append-cases", default=DEFAULT_APPEND_CASES, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT, type=Path)
    parser.add_argument("--allow-missing", action="store_true", help="Write resolved append rows even when CT/reference paths are absent on this machine.")
    args = parser.parse_args()
    audit = build_working_cohort(
        base_manifest=args.base_manifest,
        append_cases=args.append_cases,
        output_manifest=args.output_manifest,
        audit_json=args.audit_json,
        audit_csv=args.audit_csv,
        image_root=args.image_root,
        mask_root=args.mask_root,
        allow_missing=args.allow_missing,
    )
    print(json.dumps({
        "status": audit["status"],
        "base_count": audit["base_count"],
        "appended_count": audit["appended_count"],
        "final_count": audit["final_count"],
        "appended_case_ids": audit["appended_case_ids"],
    }, indent=2))
    return 0 if audit["status"] == "READY" or args.allow_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
