#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import DeliveryError, read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import FORMAL_CASE_COUNT, validate_formal_manifest  # noqa: E402


WORKSPACE_DIRS = [
    "inputs/images",
    "inputs/masks_original",
    "task1",
    "teacher",
    "candidates",
    "labelcritic",
    "annotation_versions",
    "student",
    "qc",
    "round1",
    "delivery",
    "manifests",
    "logs",
]


def _case_id(row: dict[str, str], index: int) -> str:
    return str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()


def _ct_path(row: dict[str, str], image_root: Path | None, case_id: str) -> Path:
    raw = row.get("ct_path") or row.get("image_path")
    if raw:
        return Path(raw)
    if image_root:
        return image_root / case_id / "ct.nii.gz"
    raise DeliveryError(f"case {case_id} has no ct_path/image_path and --image-root was not provided")


def _mask_dir(row: dict[str, str], mask_root: Path | None, case_id: str) -> Path:
    raw = row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir")
    if raw:
        return Path(raw)
    if mask_root:
        return mask_root / case_id / "segmentations"
    raise DeliveryError(f"case {case_id} has no annotation_folder/reference_mask_dir and --mask-root was not provided")


def _copy_file(src: Path, dst: Path, *, resume: bool) -> dict[str, Any]:
    if not src.is_file():
        return {"ok": False, "reason": "source_file_missing", "source": str(src), "destination": str(dst)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    if dst.exists():
        dst_hash = sha256_file(dst)
        if src_hash == dst_hash:
            return {"ok": True, "action": "reused_existing", "sha256": src_hash, "source": str(src), "destination": str(dst)}
        if not resume:
            return {"ok": False, "reason": "destination_exists_with_different_content", "source": str(src), "destination": str(dst)}
    shutil.copy2(src, dst)
    return {"ok": True, "action": "copied", "sha256": src_hash, "source": str(src), "destination": str(dst)}


def _copy_mask_dir(src: Path, dst: Path, *, resume: bool) -> dict[str, Any]:
    if not src.is_dir():
        return {"ok": False, "reason": "source_mask_dir_missing", "source": str(src), "destination": str(dst), "files": []}
    copied = []
    errors = []
    for path in sorted(src.glob("*.nii.gz")):
        result = _copy_file(path, dst / path.name, resume=resume)
        copied.append(result)
        if not result.get("ok"):
            errors.append(result)
    if not copied:
        errors.append({"ok": False, "reason": "source_mask_dir_has_no_nifti", "source": str(src), "destination": str(dst)})
    return {"ok": not errors, "source": str(src), "destination": str(dst), "files": copied, "errors": errors}


def _assert_not_same_or_child(dst: Path, forbidden: Path, label: str) -> None:
    try:
        dst.relative_to(forbidden.resolve())
    except ValueError:
        return
    raise DeliveryError(f"workspace root must not be inside {label}: workspace={dst} {label}={forbidden.resolve()}")


def stage_workspace(
    *,
    cases_manifest: Path,
    workspace_root: Path,
    image_root: Path | None = None,
    mask_root: Path | None = None,
    expected_case_count: int = FORMAL_CASE_COUNT,
    resume: bool = False,
    dry_run: bool = False,
    base_manifest: Path | None = None,
) -> dict[str, Any]:
    rows = read_csv_rows(cases_manifest)
    case_ids = [_case_id(row, idx) for idx, row in enumerate(rows)]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    errors: list[dict[str, Any]] = []
    if len(rows) != expected_case_count:
        errors.append({"type": "row_count", "actual": len(rows), "expected": expected_case_count})
    if len(set(case_ids)) != expected_case_count:
        errors.append({"type": "unique_case_count", "actual": len(set(case_ids)), "expected": expected_case_count, "duplicates": duplicates})
    workspace_root = workspace_root.resolve()
    if image_root:
        _assert_not_same_or_child(workspace_root, image_root, "image_root")
    if mask_root:
        _assert_not_same_or_child(workspace_root, mask_root, "mask_root")
    if str(workspace_root).startswith("/projects/bodymaps/Data"):
        errors.append({"type": "public_workspace_forbidden", "workspace_root": str(workspace_root)})
    if not dry_run:
        for rel in WORKSPACE_DIRS:
            (workspace_root / rel).mkdir(parents=True, exist_ok=True)
        probe = workspace_root / ".write_probe"
        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            errors.append({"type": "workspace_not_writable", "workspace_root": str(workspace_root), "error": str(exc)})
    staged_rows: list[dict[str, Any]] = []
    copy_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        case_id = _case_id(row, idx)
        src_ct = _ct_path(row, image_root, case_id)
        src_mask = _mask_dir(row, mask_root, case_id)
        dst_ct = workspace_root / "inputs" / "images" / case_id / "ct.nii.gz"
        dst_mask = workspace_root / "inputs" / "masks_original" / case_id / "segmentations"
        if dry_run:
            ct_ok = src_ct.is_file()
            mask_ok = src_mask.is_dir()
            copy_rows.append({"case_id": case_id, "kind": "ct", "ok": ct_ok, "source": str(src_ct), "destination": str(dst_ct), "action": "dry_run"})
            copy_rows.append({"case_id": case_id, "kind": "mask_dir", "ok": mask_ok, "source": str(src_mask), "destination": str(dst_mask), "action": "dry_run"})
            if not ct_ok:
                errors.append({"type": "ct_missing", "case_id": case_id, "path": str(src_ct)})
            if not mask_ok:
                errors.append({"type": "mask_dir_missing", "case_id": case_id, "path": str(src_mask)})
        else:
            ct_result = _copy_file(src_ct, dst_ct, resume=resume)
            mask_result = _copy_mask_dir(src_mask, dst_mask, resume=resume)
            copy_rows.append({"case_id": case_id, "kind": "ct", **ct_result})
            copy_rows.append({"case_id": case_id, "kind": "mask_dir", "ok": mask_result["ok"], "source": mask_result["source"], "destination": mask_result["destination"], "action": "copied_or_reused"})
            if not ct_result.get("ok"):
                errors.append({"type": "ct_copy_failed", "case_id": case_id, **ct_result})
            if not mask_result.get("ok"):
                errors.append({"type": "mask_copy_failed", "case_id": case_id, **mask_result})
        staged = dict(row)
        staged["index"] = idx
        staged["case_id"] = case_id
        staged["ct_path"] = str(dst_ct)
        staged["image_path"] = str(dst_ct)
        staged["annotation_folder"] = str(dst_mask)
        staged["reference_mask_dir"] = str(dst_mask)
        staged_rows.append(staged)
    status = "READY" if not errors else "BLOCKED"
    manifest_path = workspace_root / "manifests" / "cases_103_manifest.csv"
    if not dry_run and not errors:
        fieldnames = []
        for row in staged_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(manifest_path, staged_rows, fieldnames)
        if base_manifest:
            validate_formal_manifest(manifest=manifest_path, base_manifest=base_manifest)
    report = {
        "status": status,
        "workspace_root": str(workspace_root),
        "cases_manifest": str(cases_manifest),
        "output_manifest": str(manifest_path),
        "case_count": len(rows),
        "unique_case_count": len(set(case_ids)),
        "expected_case_count": expected_case_count,
        "workspace_dirs": WORKSPACE_DIRS,
        "copy_mode": "copy2_no_hardlinks",
        "resume": resume,
        "dry_run": dry_run,
        "errors": errors,
        "copy_rows": copy_rows,
    }
    if not dry_run:
        write_json(workspace_root / "manifests" / "workspace_staging_report.json", report)
        write_csv(workspace_root / "manifests" / "workspace_staging_copy_audit.csv", copy_rows, ["case_id", "kind", "ok", "reason", "source", "destination", "action", "sha256"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the fixed 103-case Task2/Round1 workspace into a personal writable root.")
    parser.add_argument("--cases-manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--expected-case-count", default=FORMAL_CASE_COUNT, type=int)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = stage_workspace(
        cases_manifest=args.cases_manifest.resolve(),
        workspace_root=args.workspace_root,
        image_root=args.image_root.resolve() if args.image_root else None,
        mask_root=args.mask_root.resolve() if args.mask_root else None,
        expected_case_count=args.expected_case_count,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        base_manifest=args.base_manifest.resolve() if args.base_manifest else None,
    )
    print(json.dumps({"status": report["status"], "case_count": report["case_count"], "output_manifest": report["output_manifest"]}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
