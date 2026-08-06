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

from tools.dataset_delivery.cads15_contract_audit import CADS15_TARGETS, DEFAULT_CONTRACT, contract_targets  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402


CADS_MODELS = ["cads553", "cads557", "cads559"]

SMOKE_SPECS: dict[str, dict[str, Any]] = {
    "atm": {
        "models": ["atm"],
        "targets": ["airway_tree"],
    },
    "cads": {
        "models": CADS_MODELS,
        "targets": CADS15_TARGETS,
        "panel_capable": True,
    },
    "cads15": {
        "models": CADS_MODELS,
        "targets": CADS15_TARGETS,
        "panel_capable": True,
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


def _slurm_status(slurm_rows: list[dict[str, str]], group: str, case_id: str | None = None) -> dict[str, Any]:
    matches = [
        item for item in slurm_rows
        if item.get("group") == group and (case_id is None or not item.get("case_id") or item.get("case_id") == case_id)
    ]
    if not matches:
        return {"known": False, "ok": True, "reason": "slurm_not_checked"}
    rows: list[dict[str, Any]] = []
    ok = True
    reasons: list[str] = []
    for row in matches:
        state = row.get("state") or row.get("State") or ""
        exit_code = row.get("exit_code") or row.get("ExitCode") or ""
        terminal_success = state == "COMPLETED" and exit_code == "0:0"
        nonterminal = state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
        reason = "success" if terminal_success else ("slurm_job_not_terminal" if nonterminal else "slurm_job_failed")
        rows.append({
            "state": state,
            "exit_code": exit_code,
            "job_id": row.get("job_id") or row.get("JobIDRaw") or "",
            "case_id": row.get("case_id") or case_id or "",
            "reason": reason,
        })
        if not terminal_success:
            ok = False
            reasons.append(reason)
    return {
        "known": True,
        "ok": ok,
        "rows": rows,
        "state": rows[0]["state"] if len(rows) == 1 else "MULTI",
        "exit_code": rows[0]["exit_code"] if len(rows) == 1 else "",
        "job_id": rows[0]["job_id"] if len(rows) == 1 else "",
        "reason": "success" if ok else ";".join(sorted(set(reasons))),
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


def _contract_model_by_target(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, str]:
    return {
        str(row.get("canonical_id")): str(row.get("primary_model"))
        for row in contract_targets(contract_path)
        if row.get("canonical_id") and row.get("primary_model")
    }


def _required_models_for_targets(group: str, targets: list[str], contract_path: Path = DEFAULT_CONTRACT) -> list[str]:
    if group in {"cads", "cads15"}:
        model_by_target = _contract_model_by_target(contract_path)
        ordered = []
        for model in CADS_MODELS:
            if any(model_by_target.get(target) == model for target in targets):
                ordered.append(model)
        return ordered
    return list(SMOKE_SPECS[group]["models"])


def _final_delivery_row(run_out: Path, case_id: str, target: str) -> dict[str, Any]:
    doc = _read_json(run_out / "final_delivery_status.json")
    for row in doc.get("rows") or []:
        if row.get("case_id") == case_id and row.get("organ") == target:
            return dict(row)
    return {}


def _validate_case_targets(
    *,
    smoke_root: Path,
    group: str,
    run_out: Path,
    case_id: str,
    targets: list[str],
    slurm_rows: list[dict[str, str]],
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    failures: list[str] = []
    target_rows: list[dict[str, Any]] = []
    slurm = _slurm_status(slurm_rows, group, case_id)
    if not slurm.get("ok"):
        failures.append(str(slurm.get("reason")))

    summary = _read_json(run_out / "run_summary.json")
    plan = _read_json(run_out / "annotation_versions" / case_id / "case_execution_plan.json")
    ct_path = Path(str(plan.get("ct_path") or ""))
    if summary.get("status") != "success":
        failures.append(f"run_summary_status:{summary.get('status')}")
    if int(summary.get("strict_delivery_failure_count") or 0) != 0:
        failures.append(f"strict_delivery_failure_count:{summary.get('strict_delivery_failure_count')}")
    teacher_run_list = set(plan.get("teacher_run_list") or [])
    required_models = _required_models_for_targets(group, targets, contract_path)
    for model in required_models:
        if model not in teacher_run_list:
            failures.append(f"teacher_run_list_missing:{model}")
    inference_checks = [_model_inference_ok(run_out, case_id, model) for model in required_models]
    for check in inference_checks:
        if not check["ok"]:
            failures.append(f"inference_not_success:{check['model']}")
    strict_failures = summary.get("strict_delivery_failures") or []
    for target in targets:
        mask = run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"
        validation = validate_mask(mask, ct_path if ct_path.exists() else None)
        final_row = _final_delivery_row(run_out, case_id, target)
        final_status = str(final_row.get("final_status") or "")
        delivery_status = str(final_row.get("delivery_status") or "")
        acceptable_final = final_status in {"delivered", "delivered_for_review"} or (
            not final_row and validation["valid"]
        )
        failure_reasons = [
            str(item.get("reason"))
            for item in strict_failures
            if item.get("organ") == target
        ]
        valid = bool(validation["valid"] and acceptable_final and not failure_reasons)
        row = {
            "group": group,
            "case_id": case_id,
            "target": target,
            "mask_path": str(mask),
            "valid": valid,
            "reason": (
                "valid"
                if valid else
                ("final_status_not_delivered" if validation["valid"] and not acceptable_final else validation["reason"])
            ),
            "final_status": final_status,
            "delivery_status": delivery_status,
            "strict_failure_reasons": ";".join(failure_reasons),
            "run_out": str(run_out),
            **{f"mask_{key}": value for key, value in validation.items() if key not in {"path", "valid", "reason"}},
        }
        target_rows.append(row)
        if not validation["valid"]:
            failures.append(f"{target}:{validation['reason']}")
        elif not acceptable_final:
            failures.append(f"{target}:final_status:{final_status or 'missing'}")
        if failure_reasons:
            failures.append(f"{target}:strict_delivery_failure:{';'.join(failure_reasons)}")
    return {
        "case_id": case_id,
        "run_out": str(run_out),
        "slurm": slurm,
        "requested_models": required_models,
        "requested_targets": targets,
        "teacher_run_list": sorted(teacher_run_list),
        "inference_checks": inference_checks,
        "passed_targets": [row["target"] for row in target_rows if row["valid"]],
        "failed_targets": [row["target"] for row in target_rows if not row["valid"]],
        "failures": failures,
        "target_rows": target_rows,
    }


def _single_run_out(smoke_root: Path, group: str) -> Path:
    group_root = smoke_root / group
    if (group_root / "run_loop").exists():
        return group_root / "run_loop"
    if group == "cads15" and (smoke_root / "cads" / "run_loop").exists():
        return smoke_root / "cads" / "run_loop"
    return group_root / "run_loop"


def _panel_json_default(smoke_root: Path) -> Path | None:
    for path in [
        smoke_root / "preflight" / "cads15_smoke_case_panel.json",
        smoke_root / "cads15_smoke_case_panel.json",
        smoke_root / "cads15" / "cads15_smoke_case_panel.json",
    ]:
        if path.exists():
            return path
    return None


def validate_group(
    smoke_root: Path,
    group: str,
    slurm_rows: list[dict[str, str]],
    *,
    panel_json: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    spec = SMOKE_SPECS[group]
    if group in {"cads", "cads15"} and panel_json and panel_json.exists():
        return validate_cads15_panel(
            smoke_root=smoke_root,
            group=group,
            panel_json=panel_json,
            slurm_rows=slurm_rows,
            contract_path=contract_path,
        )
    run_out = _single_run_out(smoke_root, group)
    case_id = _case_id_from_run(run_out) or ""
    if not case_id:
        return {
            "group": group,
            "status": "failed",
            "case_id": "",
            "run_out": str(run_out),
            "slurm": _slurm_status(slurm_rows, group),
            "requested_models": spec["models"],
            "requested_targets": spec["targets"],
            "teacher_run_list": [],
            "inference_checks": [],
            "passed_targets": [],
            "failed_targets": spec["targets"],
            "failures": ["case_id_not_resolved"],
            "target_rows": [],
        }
    case_result = _validate_case_targets(
        smoke_root=smoke_root,
        group=group,
        run_out=run_out,
        case_id=case_id,
        targets=list(spec["targets"]),
        slurm_rows=slurm_rows,
        contract_path=contract_path,
    )
    return {
        "group": group,
        "status": "passed" if not case_result["failures"] else "failed",
        **case_result,
    }


def _run_out_for_panel_case(smoke_root: Path, group: str, case_id: str) -> Path:
    for path in [
        smoke_root / group / case_id / "run_loop",
        smoke_root / group / "cases" / case_id / "run_loop",
        smoke_root / "cads15" / case_id / "run_loop",
        smoke_root / "cads" / case_id / "run_loop",
    ]:
        if path.exists():
            return path
    return smoke_root / group / case_id / "run_loop"


def validate_cads15_panel(
    *,
    smoke_root: Path,
    group: str,
    panel_json: Path,
    slurm_rows: list[dict[str, str]],
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    panel = _read_json(panel_json)
    target_rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    failures: list[str] = []
    if panel.get("status") != "READY_FOR_HPC_SMOKE":
        failures.append(f"panel_status:{panel.get('status')}")
    for case in panel.get("cases") or []:
        case_id = str(case.get("case_id") or "")
        targets = [str(target) for target in (case.get("targets") or []) if str(target)]
        run_out = _run_out_for_panel_case(smoke_root, group, case_id)
        case_result = _validate_case_targets(
            smoke_root=smoke_root,
            group=group,
            run_out=run_out,
            case_id=case_id,
            targets=targets,
            slurm_rows=slurm_rows,
            contract_path=contract_path,
        )
        case_results.append(case_result)
        target_rows.extend(case_result["target_rows"])
        failures.extend(f"{case_id}:{reason}" for reason in case_result["failures"])
    expected_targets = list(CADS15_TARGETS)
    covered: dict[str, list[str]] = {target: [] for target in expected_targets}
    for row in target_rows:
        if row.get("valid") and row.get("target") in covered:
            covered[str(row["target"])].append(str(row["case_id"]))
    missing = sorted(target for target, cases in covered.items() if not cases)
    if missing:
        failures.extend(f"target_missing_positive_smoke:{target}" for target in missing)
    passed_targets = sorted(target for target, cases in covered.items() if cases)
    status = "passed" if not failures else "failed"
    return {
        "group": group,
        "status": status,
        "panel_json": str(panel_json),
        "case_results": case_results,
        "case_count": len(case_results),
        "requested_models": CADS_MODELS,
        "requested_targets": expected_targets,
        "teacher_run_list": sorted({model for case in case_results for model in case.get("teacher_run_list", [])}),
        "inference_checks": [check for case in case_results for check in case.get("inference_checks", [])],
        "passed_targets": passed_targets,
        "failed_targets": missing,
        "target_positive_smoke_cases": covered,
        "failures": failures,
        "target_rows": target_rows,
        "CADS15_SMOKE_STATUS": "PASSED" if status == "passed" else "FAILED",
        "TARGETS_REQUESTED": len(expected_targets),
        "TARGETS_STATICALLY_VERIFIED": len(expected_targets),
        "TARGETS_WITH_POSITIVE_SMOKE": len(passed_targets),
        "TARGETS_DELIVERED": sum(1 for row in target_rows if row.get("valid") and row.get("final_status") == "delivered"),
        "TARGETS_DELIVERED_FOR_REVIEW": sum(1 for row in target_rows if row.get("valid") and row.get("final_status") == "delivered_for_review"),
        "TARGETS_FAILED": len(missing),
        "TARGETS_REJECTED": sum(1 for row in target_rows if row.get("final_status") == "rejected"),
        "TARGETS_UNCOVERED": len(missing),
        "STRICT_DELIVERY_FAILURE_COUNT": sum(1 for row in target_rows if row.get("strict_failure_reasons")),
    }


def validate_smoke_root(
    *,
    smoke_root: Path,
    groups: list[str],
    slurm_status_csv: Path | None = None,
    panel_json: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    slurm_rows = _read_csv(slurm_status_csv) if slurm_status_csv else []
    effective_panel_json = panel_json or _panel_json_default(smoke_root)
    group_results = [
        validate_group(
            smoke_root,
            group,
            slurm_rows,
            panel_json=effective_panel_json,
            contract_path=contract_path,
        )
        for group in groups
    ]
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
    cads15 = next((group for group in group_results if group["group"] in {"cads", "cads15"}), None)
    if cads15 and "CADS15_SMOKE_STATUS" in cads15:
        report["cads15_summary"] = {
            key: cads15[key]
            for key in [
                "CADS15_SMOKE_STATUS", "TARGETS_REQUESTED", "TARGETS_STATICALLY_VERIFIED",
                "TARGETS_WITH_POSITIVE_SMOKE", "TARGETS_DELIVERED",
                "TARGETS_DELIVERED_FOR_REVIEW", "TARGETS_FAILED", "TARGETS_REJECTED",
                "TARGETS_UNCOVERED", "STRICT_DELIVERY_FAILURE_COUNT",
            ]
        }
    if write_outputs:
        write_json(smoke_root / "task2_smoke_verdict.json", report)
        write_csv(
            smoke_root / "task2_smoke_target_validation.csv",
            target_rows,
            [
                "group", "case_id", "target", "mask_path", "valid", "reason",
                "final_status", "delivery_status", "strict_failure_reasons", "run_out",
            ],
        )
        lines = ["# Task 2 Teacher Smoke Verdict", "", f"- Status: `{status}`", f"- Root: `{smoke_root}`", ""]
        if report.get("cads15_summary"):
            lines.append("## CADS15")
            lines.append("")
            for key, value in report["cads15_summary"].items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")
        for group in group_results:
            lines.append(f"- {group['group']}: `{group['status']}`; passed `{len(group['passed_targets'])}/{len(group['requested_targets'])}`")
            if group["failures"]:
                lines.append(f"  failures: `{'; '.join(group['failures'][:40])}`")
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
    parser.add_argument("--panel-json", default=None, type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--no-write", action="store_true", help="Validate in memory without writing verdict files.")
    args = parser.parse_args()
    report = validate_smoke_root(
        smoke_root=args.smoke_root.resolve(),
        groups=parse_groups(args.groups),
        slurm_status_csv=args.slurm_status_csv.resolve() if args.slurm_status_csv else None,
        panel_json=args.panel_json.resolve() if args.panel_json else None,
        contract_path=args.contract.resolve(),
        write_outputs=not bool(args.no_write),
    )
    print(json.dumps({"status": report["status"], "failed_groups": report["failed_groups"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
