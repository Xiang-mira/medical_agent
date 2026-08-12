#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import DeliveryError, read_csv_rows, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import FORMAL_CASE_COUNT, validate_formal_manifest  # noqa: E402


def rewrite_manifest_mask_root(
    *,
    input_manifest: Path,
    output_manifest: Path,
    mask_root: Path,
    expected_case_count: int = FORMAL_CASE_COUNT,
    require_existing: bool = True,
    base_manifest: Path | None = None,
    audit_json: Path | None = None,
) -> dict[str, Any]:
    rows = read_csv_rows(input_manifest)
    errors: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    case_ids = [str(row.get("case_id") or row.get("id") or "").strip() for row in rows]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if len(rows) != expected_case_count:
        errors.append({"type": "row_count", "actual": len(rows), "expected": expected_case_count})
    if len(set(case_ids)) != expected_case_count:
        errors.append({"type": "unique_case_count", "actual": len(set(case_ids)), "expected": expected_case_count, "duplicates": duplicates})
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()
        if not case_id:
            errors.append({"type": "case_id_missing", "row_index": index})
        case_mask_dir = mask_root / case_id / "segmentations"
        if require_existing:
            if not case_mask_dir.is_dir():
                errors.append({"type": "mask_dir_missing", "case_id": case_id, "path": str(case_mask_dir)})
            elif not any(case_mask_dir.glob("*.nii.gz")):
                errors.append({"type": "mask_dir_has_no_nifti", "case_id": case_id, "path": str(case_mask_dir)})
        out = dict(row)
        out["index"] = index
        out["case_id"] = case_id
        out["annotation_folder"] = str(case_mask_dir)
        out["reference_mask_dir"] = str(case_mask_dir)
        out["mask_root_policy"] = "task1_renamed_personal_workspace"
        output_rows.append(out)
    status = "READY" if not errors else "BLOCKED"
    report = {
        "status": status,
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "mask_root": str(mask_root),
        "case_count": len(rows),
        "unique_case_count": len(set(case_ids)),
        "expected_case_count": expected_case_count,
        "require_existing": require_existing,
        "errors": errors,
    }
    if status == "READY":
        fieldnames: list[str] = []
        for row in output_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(output_manifest, output_rows, fieldnames)
        if base_manifest:
            validate_formal_manifest(manifest=output_manifest, base_manifest=base_manifest)
    if audit_json:
        write_json(audit_json, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite a formal 103-case manifest to use a selected mask root.")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--expected-case-count", default=FORMAL_CASE_COUNT, type=int)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    try:
        report = rewrite_manifest_mask_root(
            input_manifest=args.input_manifest.resolve(),
            output_manifest=args.output_manifest.resolve(),
            mask_root=args.mask_root.resolve(),
            expected_case_count=args.expected_case_count,
            require_existing=not bool(args.allow_missing),
            base_manifest=args.base_manifest.resolve() if args.base_manifest else None,
            audit_json=args.audit_json.resolve() if args.audit_json else None,
        )
    except DeliveryError as exc:
        report = {"status": "BLOCKED", "errors": [{"type": "delivery_error", "message": str(exc)}]}
    print(json.dumps({"status": report["status"], "case_count": report.get("case_count", 0), "output_manifest": report.get("output_manifest", "")}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
