#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ, _load_case_presence_context  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_rows, write_csv, write_json  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)
DEFAULT_DATA_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_TAXONOMY = REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json"

MODEL_GROUP_TARGETS: dict[str, list[str]] = {
    "cads": [
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
    "atm": ["airway_tree"],
    "airrc": ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"],
    "unest": ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"],
}
BLOCKED_TARGETS = ["brain_ventricle"]
ROW_FIELDS = [
    "case_id",
    "case_index",
    "model_group",
    "target_organs",
    "coverage_status",
    "fully_visible",
    "partially_visible",
    "out_of_fov",
    "unknown",
    "override_required",
    "eligible_for_launch",
    "reason",
    "ct_path",
    "reference_mask_dir",
]


def _load_taxonomy(path: Path) -> set[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("target_organs"), list):
        return {str(item) for item in doc["target_organs"]}
    if isinstance(doc, dict) and isinstance(doc.get("organs"), dict):
        return {str(item) for item in doc["organs"].keys()}
    if isinstance(doc, dict) and isinstance(doc.get("organ_to_id"), dict):
        return {str(item) for item in doc["organ_to_id"].keys()}
    raise ValueError(f"Unsupported taxonomy format: {path}")


def _split_organs(value: str | None) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in (value or "").replace(";", ",").split(",") if item.strip()))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _reference_mask_dir(row: dict[str, str], data_root: Path, case_id: str) -> str:
    explicit = row.get("reference_mask_dir") or row.get("annotation_folder") or row.get("mask_dir")
    if explicit:
        return explicit
    case_root = data_root / case_id
    seg = case_root / "segmentations"
    return str(seg if seg.exists() else case_root)


def _group_coverage_status(by_status: dict[str, list[str]]) -> str:
    if by_status["out_of_fov"]:
        return "out_of_fov"
    if by_status["unknown"]:
        return "unknown"
    if by_status["partially_visible"]:
        return "partially_visible"
    return "fully_visible"


def _eligibility(
    *,
    model_group: str,
    by_status: dict[str, list[str]],
    override_organs: set[str],
) -> tuple[bool, bool, str]:
    coverage_status = _group_coverage_status(by_status)
    if model_group == "atm":
        if coverage_status == "fully_visible":
            return True, False, "fully_visible"
        if coverage_status == "partially_visible":
            if "airway_tree" in override_organs:
                return True, True, "partial_visibility_allowed_by_explicit_airway_tree_override"
            return False, True, "partial_visibility_requires_explicit_airway_tree_override"
        if coverage_status == "out_of_fov":
            return False, False, "visibility_out_of_fov"
        return False, False, "visibility_unknown"
    if coverage_status == "fully_visible":
        return True, False, "fully_visible"
    if coverage_status == "partially_visible":
        return False, False, "partial_visibility_no_override_supported_for_model_group"
    if coverage_status == "out_of_fov":
        return False, False, "one_or_more_targets_out_of_fov"
    return False, False, "one_or_more_targets_unknown"


def _status_buckets(targets: list[str], presence_context: dict[str, Any]) -> dict[str, list[str]]:
    by_status = {
        "fully_visible": [],
        "partially_visible": [],
        "out_of_fov": [],
        "unknown": [],
    }
    for organ in targets:
        status = _fov_status_for_organ(organ, presence_context)
        by_status.setdefault(status, []).append(organ)
    return by_status


def build_preflight(
    *,
    case_manifest: Path,
    data_root: Path,
    taxonomy: Path,
    output_root: Path,
    override_organs: list[str],
) -> dict[str, Any]:
    taxonomy_names = _load_taxonomy(taxonomy)
    configured_targets = sorted({organ for targets in MODEL_GROUP_TARGETS.values() for organ in targets})
    invalid_targets = sorted(set(configured_targets) - taxonomy_names)
    invalid_overrides = sorted(set(override_organs) - taxonomy_names)
    rows_in = read_csv_rows(case_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    case_context_root = output_root / "case_coverage_context"
    row_out: list[dict[str, Any]] = []
    eligible_cases_by_group: dict[str, list[dict[str, str]]] = {group: [] for group in MODEL_GROUP_TARGETS}

    for idx, source_row in enumerate(rows_in):
        case_id = _case_id(source_row, idx)
        ct_path = _ct_path(source_row)
        ref_dir = _reference_mask_dir(source_row, data_root, case_id)
        case_for_loop = dict(source_row)
        case_for_loop["case_id"] = case_id
        case_for_loop["ct_path"] = ct_path
        case_for_loop["annotation_folder"] = ref_dir
        presence_context = _load_case_presence_context(
            case_for_loop,
            Path(ct_path) if ct_path else Path(),
            case_id,
            case_context_root / case_id,
        )
        for model_group, targets in MODEL_GROUP_TARGETS.items():
            by_status = _status_buckets(targets, presence_context)
            coverage_status = _group_coverage_status(by_status)
            eligible, override_required, reason = _eligibility(
                model_group=model_group,
                by_status=by_status,
                override_organs=set(override_organs),
            )
            row = {
                "case_id": case_id,
                "case_index": idx,
                "model_group": model_group,
                "target_organs": ";".join(targets),
                "coverage_status": coverage_status,
                "fully_visible": ";".join(by_status["fully_visible"]),
                "partially_visible": ";".join(by_status["partially_visible"]),
                "out_of_fov": ";".join(by_status["out_of_fov"]),
                "unknown": ";".join(by_status["unknown"]),
                "override_required": str(bool(override_required)).lower(),
                "eligible_for_launch": str(bool(eligible)).lower(),
                "reason": reason,
                "ct_path": ct_path,
                "reference_mask_dir": ref_dir,
            }
            row_out.append(row)
            if eligible:
                eligible_cases_by_group[model_group].append({
                    "case_id": case_id,
                    "ct_path": ct_path,
                    "annotation_folder": ref_dir,
                    "reference_mask_dir": ref_dir,
                })

    write_csv(output_root / "task2_100case_preflight_rows.csv", row_out, ROW_FIELDS)
    eligible_dir = output_root / "eligible_case_manifests"
    for group, group_rows in eligible_cases_by_group.items():
        write_csv(
            eligible_dir / f"{group}_eligible_cases.csv",
            group_rows,
            ["case_id", "ct_path", "annotation_folder", "reference_mask_dir"],
        )

    group_summary: dict[str, Any] = {}
    for group in MODEL_GROUP_TARGETS:
        group_rows = [row for row in row_out if row["model_group"] == group]
        group_summary[group] = {
            "target_organs": MODEL_GROUP_TARGETS[group],
            "eligible_case_count": sum(1 for row in group_rows if row["eligible_for_launch"] == "true"),
            "ineligible_case_count": sum(1 for row in group_rows if row["eligible_for_launch"] != "true"),
            "coverage_status_counts": {
                status: sum(1 for row in group_rows if row["coverage_status"] == status)
                for status in ["fully_visible", "partially_visible", "out_of_fov", "unknown"]
            },
            "override_required_count": sum(1 for row in group_rows if row["override_required"] == "true"),
        }
    summary = {
        "status": "failed" if invalid_targets or invalid_overrides else "success",
        "read_only": True,
        "apply": False,
        "source_data_mutation": False,
        "case_manifest": str(case_manifest),
        "data_root": str(data_root),
        "taxonomy": str(taxonomy),
        "case_count": len(rows_in),
        "model_groups": group_summary,
        "task2_currently_launchable_target_count": sum(len(v) for v in MODEL_GROUP_TARGETS.values()),
        "task2_blocked_targets": BLOCKED_TARGETS,
        "strict_delivery_fov_override_organs": override_organs,
        "invalid_taxonomy_targets": invalid_targets,
        "invalid_override_organs": invalid_overrides,
        "rows_csv": str((output_root / "task2_100case_preflight_rows.csv").resolve()),
        "eligible_case_manifests": {
            group: str((eligible_dir / f"{group}_eligible_cases.csv").resolve())
            for group in MODEL_GROUP_TARGETS
        },
    }
    write_json(output_root / "task2_100case_preflight_summary.json", summary)
    _write_report(output_root / "task2_100case_preflight_report.md", summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 2 100-case Launch Preflight",
        "",
        f"- Status: `{summary['status']}`",
        f"- Read-only: `{summary['read_only']}`",
        f"- Apply: `{summary['apply']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Launchable targets: `{summary['task2_currently_launchable_target_count']}`",
        f"- Blocked targets: `{','.join(summary['task2_blocked_targets'])}`",
        f"- Strict-delivery FOV override organs: `{','.join(summary['strict_delivery_fov_override_organs'])}`",
        "",
        "| model_group | targets | eligible | ineligible | fully | partial | out_of_fov | unknown | override_required |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in summary["model_groups"].items():
        counts = item["coverage_status_counts"]
        lines.append(
            "| {group} | {targets} | {eligible} | {ineligible} | {fully} | {partial} | {out} | {unknown} | {override} |".format(
                group=group,
                targets=len(item["target_organs"]),
                eligible=item["eligible_case_count"],
                ineligible=item["ineligible_case_count"],
                fully=counts.get("fully_visible", 0),
                partial=counts.get("partially_visible", 0),
                out=counts.get("out_of_fov", 0),
                unknown=counts.get("unknown", 0),
                override=item["override_required_count"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Task 2 fixed 100-case launch preflight."
    )
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--strict-delivery-fov-override-organs",
        default="airway_tree",
        help="Comma-separated strict-delivery FOV override organs to model in the launch eligibility audit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_preflight(
            case_manifest=args.case_manifest,
            data_root=args.data_root,
            taxonomy=args.taxonomy,
            output_root=args.output_root,
            override_organs=_split_organs(args.strict_delivery_fov_override_organs),
        )
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "read_only": True, "apply": False}
        write_json(args.output_root / "task2_100case_preflight_summary.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
