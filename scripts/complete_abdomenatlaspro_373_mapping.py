#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.utils import normalize_mask_stem, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote AbdomenAtlasPro unresolved 373 targets to direct mappings "
            "when real mask files exist. This never creates all-zero masks."
        )
    )
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, action="append", default=[])
    parser.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    parser.add_argument("--target-mapping", type=Path, default=ROOT / "configs/abdomenatlaspro_target_mapping_373.json")
    parser.add_argument("--output-mapping", type=Path, default=ROOT / "configs/abdomenatlaspro_target_mapping_373.completed.json")
    parser.add_argument("--output-target-config", type=Path, default=ROOT / "configs/abdomenatlaspro_373_target_config.json")
    parser.add_argument("--output-allowlist", type=Path, default=ROOT / "configs/abdomenatlaspro_373_target_allowlist.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/abdomenatlaspro_373_mapping_completion_report.json")
    parser.add_argument(
        "--require-all-cases",
        action="store_true",
        help="Require every scanned case to contain the target mask before promoting it.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write completed mapping/config files. Without this, only write the report.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def case_ids_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            str(row.get("case_id") or row.get("BDMAP_ID") or row.get("bdmap_id") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or row.get("BDMAP_ID") or row.get("bdmap_id") or "").strip()
        ]


def discover_case_dirs(mask_root: Path, case_lists: list[Path]) -> list[Path]:
    if case_lists:
        case_ids: list[str] = []
        for case_list in case_lists:
            case_ids.extend(case_ids_from_csv(case_list))
        return [mask_root / case_id for case_id in dict.fromkeys(case_ids)]
    return sorted(path for path in mask_root.iterdir() if path.is_dir())


def segmentation_dir(case_dir: Path) -> Path | None:
    for candidate in (case_dir / "segmentations", case_dir):
        if candidate.is_dir():
            return candidate
    return None


def mask_names_for_case(case_dir: Path) -> set[str]:
    seg_dir = segmentation_dir(case_dir)
    if seg_dir is None:
        return set()
    names: set[str] = set()
    for path in seg_dir.glob("*.nii.gz"):
        try:
            names.add(normalize_mask_stem(path.name))
        except Exception:
            continue
    return names


def build_completed_target_config(source: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    doc = deepcopy(source)
    doc["version"] = "abdomenatlaspro_373_target_config_v1"
    doc["status"] = "abdomenatlaspro_completed_373_direct_match"
    doc["full_target_config"] = "configs/student_3d_prompt_target_organs.json"
    doc["target_count"] = len(targets)
    doc["target_organs"] = targets
    for key in ("organ_to_student_id", "organ_to_prompt", "organ_prompt_bank", "prompt_variants"):
        if isinstance(doc.get(key), dict):
            doc[key] = {target: doc[key][target] for target in targets if target in doc[key]}
    doc["policy"] = {
        "is_complete_373_experiment": True,
        "unresolved_targets_skipped_not_negative": False,
        "do_not_materialize_missing_targets_as_zero_masks": True,
    }
    return doc


def main() -> int:
    args = parse_args()
    if not args.mask_root.is_dir():
        raise SystemExit(f"mask root not found: {args.mask_root}")

    target_config = read_json(args.target_config)
    mapping = read_json(args.target_mapping)
    targets = [str(item) for item in target_config.get("target_organs", [])]
    if len(targets) != 373:
        raise SystemExit(f"expected 373 targets in {args.target_config}, got {len(targets)}")

    case_dirs = discover_case_dirs(args.mask_root, args.case_list)
    if not case_dirs:
        raise SystemExit(f"no case directories found under {args.mask_root}")

    masks_by_case = {case_dir.name: mask_names_for_case(case_dir) for case_dir in case_dirs}
    mapping_by_name = {str(item.get("target_name")): item for item in mapping.get("targets", []) if isinstance(item, dict)}
    unresolved = [
        target
        for target in targets
        if str(mapping_by_name.get(target, {}).get("mapping_status")) in {"dataset_absent", "unverified_skip"}
        or not mapping_by_name.get(target, {}).get("participates_in_pilot338")
    ]

    promoted: list[dict[str, Any]] = []
    still_unresolved: list[dict[str, Any]] = []
    for target in unresolved:
        present_cases = sorted(case_id for case_id, names in masks_by_case.items() if target in names)
        missing_cases = sorted(case_id for case_id, names in masks_by_case.items() if target not in names)
        can_promote = bool(present_cases) and (not args.require_all_cases or not missing_cases)
        row = {
            "target_name": target,
            "present_case_count": len(present_cases),
            "missing_case_count": len(missing_cases),
            "present_cases_sample": present_cases[:10],
            "missing_cases_sample": missing_cases[:10],
        }
        if can_promote:
            promoted.append(row)
            item = mapping_by_name[target]
            item["mapping_status"] = "direct"
            item["source_masks"] = [target]
            item["mapping_rule"] = "read_single_binary_mask_after_leading_underscore_normalization"
            item["participates_in_pilot338"] = True
            item["participates_in_abdomenatlaspro_373"] = True
            item["requires_manual_review"] = False
            item["notes"] = "promoted by complete_abdomenatlaspro_373_mapping.py after real mask-file audit"
        else:
            still_unresolved.append(row)

    completed_count = sum(1 for target in targets if mapping_by_name.get(target, {}).get("participates_in_pilot338"))
    status = "complete_373_ready" if completed_count == 373 and not still_unresolved else "incomplete"
    report = {
        "status": status,
        "mask_root": str(args.mask_root),
        "case_count": len(case_dirs),
        "full_target_count": len(targets),
        "completed_target_count": completed_count,
        "promoted_count": len(promoted),
        "still_unresolved_count": len(still_unresolved),
        "require_all_cases": args.require_all_cases,
        "promoted": promoted,
        "still_unresolved": still_unresolved,
    }
    write_json_atomic(args.report, report)

    if args.write:
        mapping["version"] = "abdomenatlaspro_target_mapping_373_completed_v1"
        mapping["pilot_target_count"] = completed_count
        mapping["completed_target_count"] = completed_count
        mapping["unresolved_target_count"] = len(targets) - completed_count
        mapping["generation_note"] = "Completed from real AbdomenAtlasPro mask filenames; no synthetic zero masks were created."
        for item in mapping.get("targets", []):
            if isinstance(item, dict) and item.get("participates_in_pilot338"):
                item["participates_in_abdomenatlaspro_373"] = True
        write_json_atomic(args.output_mapping, mapping)
        if status == "complete_373_ready":
            write_json_atomic(args.output_target_config, build_completed_target_config(target_config, targets))
            write_json_atomic(args.output_allowlist, {
                "version": "abdomenatlaspro_373_target_allowlist_v1",
                "target_count": len(targets),
                "targets": targets,
            })

    print(json.dumps({k: report[k] for k in ("status", "case_count", "completed_target_count", "promoted_count", "still_unresolved_count")}, indent=2))
    return 0 if status == "complete_373_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
