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

from tools.dataset_delivery.delivery_lib import read_csv_rows, write_csv, write_json
from tools.dataset_delivery.task2_formal_manifest import (
    DEFAULT_BASE_MANIFEST,
    DEFAULT_OUTPUT_MANIFEST,
    FORMAL_GROUP_MODELS,
    FORMAL_MODEL_TARGETS,
    FORMAL_TARGETS,
    FORMAL_TARGET_TO_GROUP,
    _annotation_folder,
    _case_id,
    _ct_path,
    _read_manifest,
    validate_formal_manifest,
)
from tools.dataset_delivery.task2_smoke_validator import validate_mask  # noqa: E402


VALID_COMPLETION_STATUSES = {
    "generated_valid_mask",
    "confirmed_absent",
    "out_of_fov",
    "not_applicable",
}
FAILURE_STATUSES = {"expected_present_but_missing", "runtime_failed", "validation_failed"}
ROW_FIELDS = [
    "case_id",
    "target_name",
    "model_group",
    "fov_status",
    "inference_status",
    "mask_path",
    "mask_exists",
    "nonzero_voxels",
    "geometry_valid",
    "final_status",
    "reason",
]
SUMMARY_FIELDS = [
    "status",
    "case_count",
    "target_count",
    "row_count",
    "expected_row_count",
    "generated_valid_mask",
    "confirmed_absent",
    "out_of_fov",
    "not_applicable",
    "expected_present_but_missing",
    "runtime_failed",
    "validation_failed",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _task_state_path(output_root: Path, case_id: str, group: str) -> Path:
    return output_root / "tasks" / case_id / group / "task_state.json"


def _group_run_out(output_root: Path, case_id: str, group: str) -> Path:
    candidates = [
        output_root / "cases" / case_id / group / "run_loop",
        output_root / group / "cases" / case_id / "run_loop",
        output_root / "cases" / case_id / "run_loop",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _task_summary(output_root: Path, case_id: str, group: str) -> dict[str, Any]:
    run_out = _group_run_out(output_root, case_id, group)
    return {
        "task_state": _read_json(_task_state_path(output_root, case_id, group)),
        "run_summary": _read_json(run_out / "run_summary.json"),
        "final_delivery_status": _read_json(run_out / "final_delivery_status.json"),
        "case_execution_plan": _read_json(run_out / "annotation_versions" / case_id / "case_execution_plan.json"),
        "inference_results": _read_json(run_out / "inference_results.json"),
    }


def _terminal_success(summary: dict[str, Any]) -> bool:
    status = str(summary.get("status") or "").lower()
    return status in {"success", "passed", "completed"}


def _group_teacher_called(plan: dict[str, Any], run_summary: dict[str, Any], group: str) -> bool:
    teacher_models = {
        str(item)
        for item in [
            *(run_summary.get("teacher_inference_models") or []),
            *(plan.get("teacher_run_list") or []),
        ]
        if str(item)
    }
    required_models = set(FORMAL_GROUP_MODELS[group])
    teacher_count = int(run_summary.get("teacher_inference_count") or 0)
    inference_success_count = int(run_summary.get("inference_success_count") or 0)
    return bool(required_models & teacher_models) and teacher_count >= 1 and inference_success_count >= 1


def _final_row_index(final_doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = final_doc.get("rows") or []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "")
        target = str(row.get("organ") or row.get("target") or "")
        if case_id and target:
            indexed[(case_id, target)] = dict(row)
    return indexed


def _final_status_for_row(
    *,
    case_id: str,
    target: str,
    group: str,
    ct_path: Path,
    summary: dict[str, Any],
    final_row: dict[str, Any],
    final_run: dict[str, Any],
) -> dict[str, Any]:
    run_summary = final_run["run_summary"]
    task_state = final_run["task_state"]
    plan = final_run["case_execution_plan"]
    mask_text = str(final_row.get("mask_path") or "") if final_row else ""
    mask_path = Path(mask_text) if mask_text else (
        _group_run_out(Path(summary["output_root"]), case_id, group) / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"
    )
    validation = validate_mask(mask_path, ct_path if ct_path.exists() else None)
    final_row_status = str(final_row.get("final_status") or "")
    delivery_status = str(final_row.get("delivery_status") or "")
    fov_status = str(final_row.get("fov_status") or plan.get("per_organ", {}).get(target, {}).get("fov_status") or "")
    teacher_called = _group_teacher_called(plan, run_summary, group)
    task_completed = _terminal_success(run_summary) or str(task_state.get("status") or "") == "completed"
    route_applicable = target in set(plan.get("per_organ") or {})
    if final_row_status in {"confirmed_absent", "out_of_fov", "not_applicable"}:
        final_status = final_row_status
        reason = str(final_row.get("reason") or final_row.get("delivery_status") or final_row_status)
        inference_status = "INFERENCE_SUCCEEDED" if teacher_called else "INFERENCE_NOT_CALLED"
    elif validation["valid"] and final_row_status in {"delivered", "delivered_for_review"}:
        final_status = "generated_valid_mask"
        reason = "valid_mask_delivered"
        inference_status = "INFERENCE_SUCCEEDED"
    elif validation["exists"] and not validation["valid"]:
        final_status = "validation_failed"
        reason = validation["reason"]
        inference_status = "INFERENCE_SUCCEEDED" if teacher_called else "INFERENCE_EVIDENCE_INCOMPLETE"
    elif task_completed and not route_applicable:
        final_status = "not_applicable"
        reason = "target_not_routed_for_group"
        inference_status = "INFERENCE_NOT_CALLED"
    elif task_completed and teacher_called and (fov_status not in {"out_of_fov"}):
        final_status = "expected_present_but_missing"
        reason = "no_valid_mask_materialized"
        inference_status = "INFERENCE_SUCCEEDED"
    elif task_completed and fov_status == "out_of_fov":
        final_status = "out_of_fov"
        reason = "fov_out_of_scan"
        inference_status = "INFERENCE_SUCCEEDED"
    else:
        final_status = "runtime_failed"
        reason = str(
            task_state.get("reason")
            or run_summary.get("reason")
            or run_summary.get("status")
            or final_row_status
            or "task_not_completed"
        )
        inference_status = "INFERENCE_FAILED" if not teacher_called else "INFERENCE_EVIDENCE_INCOMPLETE"
    return {
        "case_id": case_id,
        "target_name": target,
        "model_group": group,
        "fov_status": fov_status or ("out_of_fov" if final_status == "out_of_fov" else ""),
        "inference_status": inference_status,
        "mask_path": str(mask_path),
        "mask_exists": bool(validation["exists"]),
        "nonzero_voxels": int(validation.get("foreground_voxels") or 0),
        "geometry_valid": bool(validation.get("valid")),
        "final_status": final_status,
        "reason": reason,
        "delivery_status": delivery_status,
        "task_state_status": str(task_state.get("status") or ""),
        "run_summary_status": str(run_summary.get("status") or ""),
        "mask_reason": validation.get("reason", ""),
        "teacher_called": str(bool(teacher_called)).lower(),
    }


def validate_case_group_run(
    *,
    output_root: Path,
    case_id: str,
    group: str,
    target: str,
    ct_path: Path,
) -> dict[str, Any]:
    final_run = _task_summary(output_root, case_id, group)
    final_doc = final_run["final_delivery_status"]
    final_row = _final_row_index(final_doc).get((case_id, target), {})
    return _final_status_for_row(
        case_id=case_id,
        target=target,
        group=group,
        ct_path=ct_path,
        summary={"output_root": str(output_root)},
        final_row=final_row,
        final_run=final_run,
    )


def build_case_target_table(
    *,
    output_root: Path,
    case_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    manifest_report: Path | None = None,
) -> dict[str, Any]:
    validate_formal_manifest(manifest=case_manifest, base_manifest=base_manifest, report=manifest_report)
    cases = _read_manifest(case_manifest)
    rows: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, int]] = {group: {status: 0 for status in SUMMARY_FIELDS if status not in {"status", "case_count", "target_count", "row_count", "expected_row_count"}} for group in FORMAL_MODEL_TARGETS}
    for index, case in enumerate(cases):
        case_id = _case_id(case, index)
        ct = Path(_ct_path(case))
        for target in FORMAL_TARGETS:
            group = FORMAL_TARGET_TO_GROUP[target]
            row = validate_case_group_run(
                output_root=output_root,
                case_id=case_id,
                group=group,
                target=target,
                ct_path=ct,
            )
            rows.append(row)
            if row["final_status"] in group_summary[group]:
                group_summary[group][row["final_status"]] += 1
    counts = {status: 0 for status in SUMMARY_FIELDS if status not in {"status", "case_count", "target_count", "row_count", "expected_row_count"}}
    for row in rows:
        counts[row["final_status"]] = counts.get(row["final_status"], 0) + 1
    summary = {
        "status": "PASSED" if not any(counts[status] for status in FAILURE_STATUSES) else "VALIDATION_FAILED",
        "case_count": len(cases),
        "target_count": len(FORMAL_TARGETS),
        "row_count": len(rows),
        "expected_row_count": len(cases) * len(FORMAL_TARGETS),
        "strict_delivery_failure_count": sum(counts[status] for status in FAILURE_STATUSES),
        **counts,
        "group_counts": group_summary,
        "rows_csv": str(output_root / "task2_formal_case_target_status.csv"),
        "summary_json": str(output_root / "task2_formal_summary.json"),
    }
    return {"summary": summary, "rows": rows}


def write_case_target_outputs(
    *,
    output_root: Path,
    case_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    manifest_report: Path | None = None,
) -> dict[str, Any]:
    report = build_case_target_table(output_root=output_root, case_manifest=case_manifest, base_manifest=base_manifest, manifest_report=manifest_report)
    summary = report["summary"]
    rows = report["rows"]
    write_csv(output_root / "task2_formal_case_target_status.csv", rows, ROW_FIELDS + ["delivery_status", "task_state_status", "run_summary_status", "mask_reason", "teacher_called"])
    counts_csv = [
        {"status": status, "count": summary.get(status, 0)}
        for status in ["generated_valid_mask", "confirmed_absent", "out_of_fov", "not_applicable", "expected_present_but_missing", "runtime_failed", "validation_failed"]
    ]
    write_csv(output_root / "task2_formal_summary_counts.csv", counts_csv, ["status", "count"])
    write_json(output_root / "task2_formal_summary.json", summary)
    write_json(output_root / "task2_formal_case_target_status.json", {"summary": summary, "rows": rows})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Task 2 formal 103-case x 22-target outputs.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-manifest", default=DEFAULT_OUTPUT_MANIFEST, type=Path)
    parser.add_argument("--base-manifest", default=DEFAULT_BASE_MANIFEST, type=Path)
    parser.add_argument("--manifest-report", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report = build_case_target_table(
        output_root=args.output_root.resolve(),
        case_manifest=args.case_manifest.resolve(),
        base_manifest=args.base_manifest.resolve(),
        manifest_report=args.manifest_report.resolve() if args.manifest_report else None,
    )
    if not args.no_write:
        summary = report["summary"]
        rows = report["rows"]
        write_csv(args.output_root.resolve() / "task2_formal_case_target_status.csv", rows, ROW_FIELDS + ["delivery_status", "task_state_status", "run_summary_status", "mask_reason", "teacher_called"])
        write_csv(
            args.output_root.resolve() / "task2_formal_summary_counts.csv",
            [{"status": status, "count": summary.get(status, 0)} for status in ["generated_valid_mask", "confirmed_absent", "out_of_fov", "not_applicable", "expected_present_but_missing", "runtime_failed", "validation_failed"]],
            ["status", "count"],
        )
        write_json(args.output_root.resolve() / "task2_formal_summary.json", summary)
        write_json(args.output_root.resolve() / "task2_formal_case_target_status.json", {"summary": summary, "rows": rows})
    print(json.dumps({"status": report["summary"]["status"], "row_count": report["summary"]["row_count"]}, indent=2))
    return 0 if report["summary"]["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
