from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import resolve_path
from .utils import CASE_ID_RE, SchedulerError, normalize_mask_stem, read_json, sha256_file


GT_FORBIDDEN_COLUMNS = {
    "gt_path",
    "label_path",
    "mask_path",
    "mask_root",
    "annotation_folder",
    "segmentation_folder",
    "segmentations",
    "eval_reference",
}


def read_case_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SchedulerError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if any((v or "").strip() for v in row.values())]
    if not rows:
        raise SchedulerError(f"Manifest has no rows: {path}")
    return rows


def case_id_from_row(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_id") or row.get("BDMAP_ID") or row.get("bdmap_id") or "").strip()
    if not case_id:
        ct = str(row.get("ct_path") or row.get("image") or "").strip()
        if ct:
            case_id = Path(ct).parent.name
    if not CASE_ID_RE.match(case_id):
        raise SchedulerError(f"Invalid or missing BDMAP case_id in row: {row}")
    return case_id


def assert_strict_no_gt_manifest(path: Path) -> dict[str, Any]:
    rows = read_case_csv(path)
    columns = {key for row in rows for key in row}
    lower_columns = {key.lower() for key in columns}
    blocked = sorted(lower_columns & GT_FORBIDDEN_COLUMNS)
    if blocked:
        raise SchedulerError(f"Strict input manifest contains GT-like columns {blocked}: {path}")
    for row in rows:
        for key, value in row.items():
            text = str(value or "").lower()
            if key.lower() in GT_FORBIDDEN_COLUMNS or "/segmentations/" in text or "mask_only" in text:
                raise SchedulerError(f"Strict input manifest leaks GT-like value in {key}: {path}")
    case_ids = [case_id_from_row(row) for row in rows]
    return {"path": str(path), "rows": len(rows), "case_ids": case_ids, "sha256": sha256_file(path)}


def validate_split(train: Path, test: Path, reserve: Path) -> dict[str, Any]:
    docs = {
        "train": assert_strict_no_gt_manifest(train),
        "test": assert_strict_no_gt_manifest(test),
        "reserve": {"path": str(reserve), "sha256": sha256_file(reserve), "case_ids": [case_id_from_row(r) for r in read_case_csv(reserve)]},
    }
    sets = {name: set(doc["case_ids"]) for name, doc in docs.items()}
    overlaps = {
        "train_test": sorted(sets["train"] & sets["test"]),
        "train_reserve": sorted(sets["train"] & sets["reserve"]),
        "test_reserve": sorted(sets["test"] & sets["reserve"]),
    }
    if any(overlaps.values()):
        raise SchedulerError(f"Pilot split overlap detected: {overlaps}")
    if len(sets["train"]) != 20 or len(sets["test"]) != 20 or len(sets["reserve"]) != 10:
        raise SchedulerError(
            f"Pilot split must be 20/20/10, got train={len(sets['train'])}, test={len(sets['test'])}, reserve={len(sets['reserve'])}"
        )
    return {"splits": docs, "overlaps": overlaps}


def validate_target_config(path: Path, *, expected_count: int) -> dict[str, Any]:
    doc = read_json(path)
    targets = [str(x) for x in doc.get("target_organs", [])]
    duplicates = sorted({x for x in targets if targets.count(x) > 1})
    if len(targets) != expected_count:
        raise SchedulerError(f"Target config {path} has {len(targets)} targets, expected {expected_count}")
    if duplicates:
        raise SchedulerError(f"Target config has duplicate targets: {duplicates[:10]}")
    return {"path": str(path), "target_count": len(targets), "sha256": sha256_file(path), "targets": targets}


def validate_target_mapping(path: Path, full_targets: list[str], *, pilot_target_count: int) -> dict[str, Any]:
    doc = read_json(path)
    items = doc.get("targets")
    if not isinstance(items, list):
        raise SchedulerError("Target mapping must contain a targets list")
    by_name = {str(item.get("target_name")): item for item in items if isinstance(item, dict)}
    missing = [target for target in full_targets if target not in by_name]
    extra = sorted(set(by_name) - set(full_targets))
    if missing or extra:
        raise SchedulerError(f"Target mapping mismatch missing={missing[:10]} extra={extra[:10]}")
    allowed = {"direct", "alias", "union", "composite", "dataset_absent", "unverified_skip"}
    pilot = []
    invalid_status = []
    unresolved_in_pilot = []
    for target in full_targets:
        item = by_name[target]
        status = str(item.get("mapping_status"))
        if status not in allowed:
            invalid_status.append({"target": target, "status": status})
        if bool(item.get("participates_in_pilot338")):
            pilot.append(target)
            if status in {"dataset_absent", "unverified_skip"}:
                unresolved_in_pilot.append(target)
    if invalid_status:
        raise SchedulerError(f"Invalid mapping statuses: {invalid_status[:10]}")
    if len(pilot) != pilot_target_count:
        raise SchedulerError(f"Mapping has {len(pilot)} pilot targets, expected {pilot_target_count}")
    if unresolved_in_pilot:
        raise SchedulerError(f"Unresolved targets cannot participate in pilot338: {unresolved_in_pilot[:10]}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "full_target_count": len(full_targets),
        "pilot_target_count": len(pilot),
        "unresolved_target_count": len(full_targets) - len(pilot),
    }


def audit_case_masks(image_root: Path, mask_root: Path, case_ids: list[str]) -> dict[str, Any]:
    rows = []
    failures = []
    for case_id in case_ids:
        ct = image_root / case_id / "ct.nii.gz"
        seg = mask_root / case_id / "segmentations"
        item: dict[str, Any] = {
            "case_id": case_id,
            "ct_exists": ct.is_file(),
            "segmentations_exists": seg.is_dir(),
            "mask_count": 0,
            "normalized_conflicts": [],
            "leading_underscore_count": 0,
        }
        if not item["ct_exists"]:
            failures.append({"case_id": case_id, "reason": "missing_ct", "path": str(ct)})
        if not item["segmentations_exists"]:
            failures.append({"case_id": case_id, "reason": "missing_segmentations", "path": str(seg)})
        if seg.is_dir():
            masks = sorted(p for p in seg.iterdir() if p.is_file() and p.name.endswith(".nii.gz"))
            item["mask_count"] = len(masks)
            normalized: dict[str, list[str]] = {}
            for mask in masks:
                if mask.name.startswith("_"):
                    item["leading_underscore_count"] += 1
                normalized.setdefault(normalize_mask_stem(mask.name), []).append(mask.name)
            item["normalized_conflicts"] = [
                {"name": name, "files": files}
                for name, files in sorted(normalized.items())
                if len(set(files)) > 1
            ]
            if not masks:
                failures.append({"case_id": case_id, "reason": "empty_segmentations", "path": str(seg)})
            if item["normalized_conflicts"]:
                failures.append({"case_id": case_id, "reason": "normalized_mask_name_conflict"})
        rows.append(item)
    return {"cases": rows, "failures": failures, "status": "failed" if failures else "success"}


def resolved_manifest_path(paths: dict[str, Any], key: str) -> Path:
    path = resolve_path(paths.get(key))
    if path is None:
        raise SchedulerError(f"Missing manifest path config: {key}")
    return path
