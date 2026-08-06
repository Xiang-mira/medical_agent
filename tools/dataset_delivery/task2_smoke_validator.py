#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402


SMOKE_SPECS: dict[str, dict[str, Any]] = {
    "atm": {
        "models": ["atm"],
        "targets": ["airway_tree"],
    },
    "cads": {
        "models": ["cads553", "cads557", "cads559"],
        "targets": [
            "blood",
            "common_iliac_artery_left",
            "common_iliac_artery_right",
            "common_iliac_vein_left",
            "common_iliac_vein_right",
            "compact_bone",
            "gland_structure",
            "spongy_bone",
        ],
    },
    "unest": {
        "models": ["unest"],
        "targets": ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_id_from_run(run_out: Path) -> str | None:
    rows = _read_csv(run_out.parent / "selected_case_manifest.csv")
    if rows:
        return rows[0].get("case_id") or rows[0].get("id")
    ann = run_out / "annotation_versions"
    if ann.exists():
        cases = sorted(path.name for path in ann.iterdir() if path.is_dir())
        if len(cases) == 1:
            return cases[0]
    return None


def _slurm_status(slurm_rows: list[dict[str, str]], group: str) -> dict[str, Any]:
    row = next((item for item in slurm_rows if item.get("group") == group), None)
    if row is None:
        return {"known": False, "ok": True, "reason": "slurm_not_checked"}
    state = row.get("state") or row.get("State") or ""
    exit_code = row.get("exit_code") or row.get("ExitCode") or ""
    terminal_success = state == "COMPLETED" and exit_code == "0:0"
    nonterminal = state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
    return {
        "known": True,
        "ok": terminal_success,
        "state": state,
        "exit_code": exit_code,
        "job_id": row.get("job_id") or row.get("JobIDRaw") or "",
        "reason": "success" if terminal_success else ("slurm_job_not_terminal" if nonterminal else "slurm_job_failed"),
    }


def validate_mask(mask_path: Path, ct_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(mask_path),
        "exists": mask_path.exists(),
        "nonempty_file": False,
        "nifti_readable": False,
        "binary": False,
        "foreground_voxels": 0,
        "shape_matches_ct": None,
        "spacing_matches_ct": None,
        "affine_matches_ct": None,
        "valid": False,
        "reason": "",
    }
    if not mask_path.exists():
        result["reason"] = "missing_mask"
        return result
    try:
        stat = mask_path.stat()
        result["size_bytes"] = int(stat.st_size)
        result["nonempty_file"] = stat.st_size > 0
        if stat.st_size <= 0:
            result["reason"] = "empty_file"
            return result
    except Exception as exc:
        result["reason"] = f"stat_failed:{type(exc).__name__}:{exc}"
        return result
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(mask_path))
        arr = np.asanyarray(img.dataobj)
        labels = sorted({int(value) for value in np.unique(arr)})
        foreground = int((arr != 0).sum())
        result.update({
            "nifti_readable": True,
            "labels": labels,
            "binary": set(labels).issubset({0, 1}),
            "foreground_voxels": foreground,
            "shape": list(img.shape[:3]),
            "spacing": [float(x) for x in img.header.get_zooms()[:3]],
        })
        if ct_path and ct_path.exists():
            ct = nib.load(str(ct_path))
            result.update({
                "ct_path": str(ct_path),
                "ct_shape": list(ct.shape[:3]),
                "ct_spacing": [float(x) for x in ct.header.get_zooms()[:3]],
                "shape_matches_ct": tuple(img.shape[:3]) == tuple(ct.shape[:3]),
                "spacing_matches_ct": bool(np.allclose(img.header.get_zooms()[:3], ct.header.get_zooms()[:3], rtol=0, atol=1e-5)),
                "affine_matches_ct": bool(np.allclose(img.affine, ct.affine, rtol=0, atol=1e-5)),
            })
    except Exception as exc:
        result["reason"] = f"nifti_read_error:{type(exc).__name__}:{exc}"
        return result
    geometry_ok = (
        result["shape_matches_ct"] is not False
        and result["spacing_matches_ct"] is not False
        and result["affine_matches_ct"] is not False
    )
    if not result["binary"]:
        result["reason"] = "non_binary_mask"
    elif int(result["foreground_voxels"]) <= 0:
        result["reason"] = "empty_mask"
    elif not geometry_ok:
        result["reason"] = "geometry_mismatch"
    else:
        result["valid"] = True
        result["reason"] = "valid"
    return result


def _model_inference_ok(run_out: Path, case_id: str, model: str) -> dict[str, Any]:
    summaries = sorted((run_out / "cases" / case_id / "raw_predictions").glob(f"**/{model}/{case_id}/inference_summary.json"))
    if not summaries:
        summaries = sorted((run_out / "cases" / case_id / "raw_predictions").glob(f"**/{model}*/**/inference_summary.json"))
    rows = [_read_json(path) | {"path": str(path)} for path in summaries]
    ok = any(row.get("status") == "success" and int(row.get("return_code") or 0) == 0 for row in rows)
    return {"model": model, "ok": ok, "summaries": rows}


def validate_group(smoke_root: Path, group: str, slurm_rows: list[dict[str, str]]) -> dict[str, Any]:
    spec = SMOKE_SPECS[group]
    group_root = smoke_root / group
    run_out = group_root / "run_loop"
    case_id = _case_id_from_run(run_out)
    failures: list[str] = []
    target_rows: list[dict[str, Any]] = []
    slurm = _slurm_status(slurm_rows, group)
    if not slurm.get("ok"):
        failures.append(str(slurm.get("reason")))
    if not case_id:
        failures.append("case_id_not_resolved")
        case_id = ""

    summary = _read_json(run_out / "run_summary.json")
    plan = _read_json(run_out / "annotation_versions" / case_id / "case_execution_plan.json") if case_id else {}
    ct_path = Path(str(plan.get("ct_path") or ""))
    if summary.get("status") != "success":
        failures.append(f"run_summary_status:{summary.get('status')}")
    if int(summary.get("strict_delivery_failure_count") or 0) != 0:
        failures.append(f"strict_delivery_failure_count:{summary.get('strict_delivery_failure_count')}")
    teacher_run_list = set(plan.get("teacher_run_list") or [])
    for model in spec["models"]:
        if model not in teacher_run_list:
            failures.append(f"teacher_run_list_missing:{model}")
    inference_checks = [_model_inference_ok(run_out, case_id, model) for model in spec["models"] if case_id]
    for check in inference_checks:
        if not check["ok"]:
            failures.append(f"inference_not_success:{check['model']}")
    strict_failures = summary.get("strict_delivery_failures") or []
    for target in spec["targets"]:
        mask = run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"
        validation = validate_mask(mask, ct_path if ct_path.exists() else None)
        failure_reasons = [
            str(item.get("reason"))
            for item in strict_failures
            if item.get("organ") == target
        ]
        row = {
            "group": group,
            "case_id": case_id,
            "target": target,
            "mask_path": str(mask),
            "valid": validation["valid"],
            "reason": validation["reason"],
            "strict_failure_reasons": ";".join(failure_reasons),
            **{f"mask_{key}": value for key, value in validation.items() if key not in {"path", "valid", "reason"}},
        }
        target_rows.append(row)
        if not validation["valid"]:
            failures.append(f"{target}:{validation['reason']}")
        if failure_reasons:
            failures.append(f"{target}:strict_delivery_failure:{';'.join(failure_reasons)}")
    passed_targets = [row["target"] for row in target_rows if row["valid"]]
    failed_targets = [row["target"] for row in target_rows if not row["valid"]]
    return {
        "group": group,
        "status": "passed" if not failures else "failed",
        "case_id": case_id,
        "run_out": str(run_out),
        "slurm": slurm,
        "requested_models": spec["models"],
        "requested_targets": spec["targets"],
        "teacher_run_list": sorted(teacher_run_list),
        "inference_checks": inference_checks,
        "passed_targets": passed_targets,
        "failed_targets": failed_targets,
        "failures": failures,
        "target_rows": target_rows,
    }


def validate_smoke_root(
    *,
    smoke_root: Path,
    groups: list[str],
    slurm_status_csv: Path | None = None,
) -> dict[str, Any]:
    slurm_rows = _read_csv(slurm_status_csv) if slurm_status_csv else []
    group_results = [validate_group(smoke_root, group, slurm_rows) for group in groups]
    target_rows = [row for group in group_results for row in group["target_rows"]]
    status = "passed" if all(group["status"] == "passed" for group in group_results) else "failed"
    report = {
        "status": status,
        "smoke_root": str(smoke_root),
        "groups": group_results,
        "passed_groups": [group["group"] for group in group_results if group["status"] == "passed"],
        "failed_groups": [group["group"] for group in group_results if group["status"] != "passed"],
        "target_rows": target_rows,
    }
    write_json(smoke_root / "task2_smoke_verdict.json", report)
    write_csv(
        smoke_root / "task2_smoke_target_validation.csv",
        target_rows,
        ["group", "case_id", "target", "mask_path", "valid", "reason", "strict_failure_reasons"],
    )
    lines = ["# Task 2 Teacher Smoke Verdict", "", f"- Status: `{status}`", f"- Root: `{smoke_root}`", ""]
    for group in group_results:
        lines.append(f"- {group['group']}: `{group['status']}`; passed `{len(group['passed_targets'])}/{len(group['requested_targets'])}`")
        if group["failures"]:
            lines.append(f"  failures: `{'; '.join(group['failures'])}`")
    (smoke_root / "task2_smoke_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_groups(value: str) -> list[str]:
    groups = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    unknown = sorted(set(groups) - set(SMOKE_SPECS))
    if unknown:
        raise ValueError(f"unknown smoke groups: {unknown}")
    return sorted(set(groups), key=groups.index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Task 2 Teacher strict-delivery smoke outputs.")
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--groups", default="atm,cads,unest")
    parser.add_argument("--slurm-status-csv", default=None, type=Path)
    args = parser.parse_args()
    report = validate_smoke_root(
        smoke_root=args.smoke_root.resolve(),
        groups=parse_groups(args.groups),
        slurm_status_csv=args.slurm_status_csv.resolve() if args.slurm_status_csv else None,
    )
    print(json.dumps({"status": report["status"], "failed_groups": report["failed_groups"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
