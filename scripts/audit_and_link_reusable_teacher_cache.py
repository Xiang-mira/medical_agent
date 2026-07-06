#!/usr/bin/env python3
"""Audit and link reusable hierarchical teacher cache before running new cases.

This is a no-inference helper.  It scans prior output trees for case-level
hierarchical teacher caches that are compatible with the current formal Round2
path, then symlinks whole case directories into the target Round1 E-step cache.

Compatibility is intentionally conservative:
  * the case must have a valid current hierarchical ROI manifest;
  * all required teachers must have non-empty segmentation directories;
  * existing target case directories are never overwritten;
  * selected/GT annotations are not copied into training labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_em_training as em


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_case_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            str(row.get("case_id") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or "").strip()
        ]


def valid_hierarchical_manifest(path: Path) -> bool:
    doc = read_json(path, {})
    return bool(
        doc.get("teacher_inference_mode") == "hierarchical_roi"
        and doc.get("hierarchical_plan_cache_key")
        and doc.get("hierarchical_plan_cache_key_sha256")
    )


def teacher_seg_dir(case_dir: Path, teacher: str) -> Path | None:
    candidates = [
        case_dir / "hierarchical_predictions" / teacher / "segmentations",
        case_dir / "raw_predictions" / "hierarchical_full" / teacher / "segmentations",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*.nii.gz")):
            return path
    return None


def cache_completeness(case_dir: Path, required_teachers: set[str]) -> dict[str, Any]:
    manifest = case_dir / "hierarchical_inference_plan.json"
    available = sorted(
        teacher for teacher in required_teachers
        if teacher_seg_dir(case_dir, teacher) is not None
    )
    missing = sorted(required_teachers - set(available))
    return {
        "case_dir": str(case_dir),
        "valid_hierarchical_manifest": valid_hierarchical_manifest(manifest),
        "available_teachers": available,
        "available_teacher_count": len(available),
        "missing_required_teachers": missing,
        "complete": valid_hierarchical_manifest(manifest) and not missing,
    }


def link_or_copy_tree(source: Path, destination: Path, *, copy: bool = False) -> str:
    if destination.exists() or destination.is_symlink():
        return "exists"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, destination)
        return "copied"
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
        return "symlinked"
    except OSError:
        shutil.copytree(source, destination)
        return "copied_after_symlink_failed"


def discover_case_roots(search_roots: list[Path], exclude_roots: list[Path]) -> list[Path]:
    exclude_resolved = [root.resolve() for root in exclude_roots if root.exists()]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for cases_root in root.glob("**/estep/cases"):
            resolved = cases_root.resolve()
            if resolved in seen:
                continue
            if any(resolved == ex or ex in resolved.parents for ex in exclude_resolved):
                continue
            seen.add(resolved)
            out.append(resolved)
    return sorted(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--target-cases-root", type=Path, required=True)
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[ROOT / "outputs"],
        help="Root(s) to scan for prior **/estep/cases trees. Can be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    parser.add_argument(
        "--required-teachers",
        default=",".join(em.ALL_TEACHERS),
        help="Comma-separated required teacher keys. Default: formal ALL_TEACHERS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = read_case_ids(args.case_list.resolve())
    required = {item.strip() for item in args.required_teachers.split(",") if item.strip()}
    target_cases_root = args.target_cases_root.resolve()
    cases_roots = discover_case_roots(
        [path.resolve() for path in args.search_root],
        exclude_roots=[target_cases_root],
    )
    rows: list[dict[str, Any]] = []
    linked: list[dict[str, Any]] = []
    for case_id in case_ids:
        target_case = target_cases_root / case_id
        if target_case.exists() or target_case.is_symlink():
            rows.append({
                "case_id": case_id,
                "decision": "skip",
                "reason": "target_case_already_exists",
                "target_case": str(target_case),
            })
            continue
        candidates = []
        for cases_root in cases_roots:
            source_case = cases_root / case_id
            if not source_case.is_dir():
                continue
            audit = cache_completeness(source_case, required)
            candidates.append(audit)
        complete = [row for row in candidates if row.get("complete")]
        if not complete:
            best = max(candidates, key=lambda row: int(row.get("available_teacher_count") or 0), default={})
            rows.append({
                "case_id": case_id,
                "decision": "miss",
                "reason": "no_complete_compatible_cache_found",
                "best_available_teacher_count": best.get("available_teacher_count", 0),
                "best_missing_required_teachers": best.get("missing_required_teachers", []),
                "best_case_dir": best.get("case_dir"),
                "candidate_count": len(candidates),
            })
            continue
        chosen = sorted(complete, key=lambda row: (row["case_dir"].count("/archived_"), row["case_dir"]))[0]
        source = Path(str(chosen["case_dir"]))
        action = link_or_copy_tree(source, target_case, copy=args.copy)
        record = {
            "case_id": case_id,
            "decision": "linked",
            "action": action,
            "source_case_dir": str(source),
            "target_case_dir": str(target_case),
            "available_teacher_count": chosen.get("available_teacher_count"),
            "available_teachers": chosen.get("available_teachers"),
        }
        rows.append(record)
        linked.append(record)

    summary = {
        "stage": "audit_and_link_reusable_teacher_cache",
        "status": "success",
        "case_list": str(args.case_list.resolve()),
        "target_cases_root": str(target_cases_root),
        "search_roots": [str(path.resolve()) for path in args.search_root],
        "required_teacher_count": len(required),
        "required_teachers": sorted(required),
        "case_count": len(case_ids),
        "linked_case_count": len(linked),
        "miss_case_count": sum(1 for row in rows if row.get("decision") == "miss"),
        "skip_existing_count": sum(1 for row in rows if row.get("decision") == "skip"),
        "copy_mode": bool(args.copy),
        "policy": "No inference; whole compatible case cache dirs are linked/copied only when every required teacher is present.",
        "rows": rows,
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
