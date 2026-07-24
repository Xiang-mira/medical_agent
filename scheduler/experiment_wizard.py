from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import SchedulerConfig, resolve_path
from .manifest import assert_strict_no_gt_manifest
from .planner import build_plan, submit_plan
from .resource_discovery import discover_resource_snapshot, snapshot_markdown
from .resource_recommender import recommend_resource_plans, validate_user_plan, workload_estimates
from .utils import SchedulerError, sha256_file, sha256_text, write_json_atomic


def _case_counts(cfg: SchedulerConfig) -> tuple[int, int]:
    train = resolve_path(cfg.paths.get("train_input_case_list") or cfg.paths.get("pilot_train_input_case_list"))
    test = resolve_path(cfg.paths.get("test_input_case_list") or cfg.paths.get("pilot_test_input_case_list"))
    if train is None or test is None:
        raise SchedulerError("Config must define train/test input manifests")
    return assert_strict_no_gt_manifest(train)["rows"], assert_strict_no_gt_manifest(test)["rows"]


def _array_count(array: str | None) -> int:
    if not array:
        return 1
    span = str(array).split("%", 1)[0]
    if "-" not in span:
        return 1
    start, end = span.split("-", 1)
    return max(0, int(end) - int(start) + 1)


def _array_concurrency(array: str | None) -> int:
    if not array:
        return 1
    if "%" in str(array):
        return int(str(array).split("%", 1)[1])
    return _array_count(array)


def _plan_artifacts(config: SchedulerConfig, scheduler_plan: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    tasks = scheduler_plan.get("tasks") or []
    run_dir = Path(str(scheduler_plan.get("run_dir")))
    t4_concurrency = [
        _array_concurrency(t.get("array"))
        for t in tasks
        if (t.get("requested_resources") or {}).get("partition") == "gpu" and int((t.get("requested_resources") or {}).get("gpu_count") or 0) == 1
    ]
    h100 = [
        int((t.get("requested_resources") or {}).get("gpu_count") or 0) * _array_concurrency(t.get("array"))
        for t in tasks
        if (t.get("requested_resources") or {}).get("partition") == "gpuh100"
    ]
    a100 = [
        int((t.get("requested_resources") or {}).get("gpu_count") or 0) * _array_concurrency(t.get("array"))
        for t in tasks
        if (t.get("requested_resources") or {}).get("partition") == "gpua100"
    ]
    path_keys = ("train_input_case_list", "test_input_case_list", "train_eval_case_list", "test_eval_case_list")
    manifest_sha = {}
    for key in path_keys:
        path = resolve_path(config.paths.get(key))
        if path and path.exists():
            manifest_sha[key] = sha256_file(path)
    mapping_path = resolve_path(config.paths.get("target_mapping"))
    config_text = yaml.safe_dump(config.data, sort_keys=True)
    dependency_graph = {
        "nodes": [t["task_name"] for t in tasks],
        "edges": [{"from": dep, "to": t["task_name"]} for t in tasks for dep in t.get("dependencies", [])],
    }
    resource_plan = {
        "selected_plan_id": selected.get("plan_id"),
        "resource_profile": selected,
        "max_t4_concurrent": max(t4_concurrency or [0]),
        "max_a100_training_gpus": max(a100 or [0]),
        "max_h100_gpus": max(h100 or [0]),
        "task_counts": {t["task_name"]: _array_count(t.get("array")) for t in tasks},
        "arrays": {t["task_name"]: t.get("array") for t in tasks if t.get("array")},
    }
    run_plan = {
        "run_id": scheduler_plan.get("run_id"),
        "run_dir": scheduler_plan.get("run_dir"),
        "pipeline": scheduler_plan.get("pipeline"),
        "backend": scheduler_plan.get("backend"),
        "git": scheduler_plan.get("git"),
        "config_sha": sha256_text(config_text),
        "manifest_sha": manifest_sha,
        "mapping_sha": sha256_file(mapping_path) if mapping_path and mapping_path.exists() else None,
        "fingerprint": sha256_text(json.dumps({"config_sha": sha256_text(config_text), "manifest_sha": manifest_sha, "mapping_sha": sha256_file(mapping_path) if mapping_path and mapping_path.exists() else None, "pipeline": scheduler_plan.get("pipeline")}, sort_keys=True)),
        "output_root": str(run_dir),
        "submitted": False,
        "slurm_job_ids": {},
    }
    return {
        "run_plan": run_plan,
        "resource_plan": resource_plan,
        "task_manifest": {"tasks": tasks},
        "dependency_graph": dependency_graph,
    }


def _write_launch_artifacts(run_dir: Path, config: SchedulerConfig, snapshot: dict[str, Any], workloads: dict[str, Any], plans: dict[str, Any], selected: dict[str, Any], scheduler_plan: dict[str, Any]) -> None:
    write_json_atomic(run_dir / "resource_snapshot.json", snapshot)
    (run_dir / "resource_snapshot.md").write_text(snapshot_markdown(snapshot), encoding="utf-8")
    write_json_atomic(run_dir / "workload_estimates.json", workloads)
    (run_dir / "candidate_resource_plans.yaml").write_text(yaml.safe_dump(plans, sort_keys=False), encoding="utf-8")
    (run_dir / "selected_resource_plan.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    artifacts = _plan_artifacts(config, scheduler_plan, selected)
    write_json_atomic(run_dir / "run_plan.json", artifacts["run_plan"])
    write_json_atomic(run_dir / "resource_plan.json", artifacts["resource_plan"])
    write_json_atomic(run_dir / "task_manifest.json", artifacts["task_manifest"])
    write_json_atomic(run_dir / "dependency_graph.json", artifacts["dependency_graph"])
    write_json_atomic(run_dir / "user_choices.json", {"plan_id": selected.get("plan_id"), "submitted": False})


def prepare_experiment(config: SchedulerConfig, pipeline: str, *, resource_plan: str = "balanced", run_id: str | None = None, dry_run: bool = True) -> dict[str, Any]:
    train_cases, test_cases = _case_counts(config)
    snapshot = discover_resource_snapshot(config.path, account=config.slurm_defaults.get("account"), qos=config.slurm_defaults.get("qos"))
    if not dry_run and config.data.get("require_slurm_account") and not snapshot.get("account"):
        raise SchedulerError("Formal submission requires a resolved Slurm account; set slurm_defaults.account or SLURM_ACCOUNT.")
    workloads = workload_estimates(train_cases, test_cases)
    candidates = recommend_resource_plans(snapshot, train_cases, test_cases, gpu_budget=(config.data.get("gpu_budget") or {}))
    selected = next((p for p in candidates["plans"] if p["plan_id"] == resource_plan), None)
    if selected is None:
        raise SchedulerError(f"Unknown resource plan {resource_plan!r}")
    validate_user_plan(selected, ddp_validated=bool(config.data.get("ddp_validated", False)), h100_ddp_allowed=bool(config.data.get("h100_ddp_allowed", False)))
    plan = build_plan(config, pipeline, backend="slurm", run_id=run_id)
    run_dir = Path(plan["run_dir"])
    _write_launch_artifacts(run_dir, config, snapshot, workloads, candidates, selected, plan)
    return {"status": "planned", "dry_run": dry_run, "run_id": plan["run_id"], "run_dir": str(run_dir), "resource_snapshot": snapshot, "resource_plan": selected, "scheduler_plan": plan}


def launch_experiment(config: SchedulerConfig, pipeline: str, *, resource_plan: str = "balanced", yes: bool = False, dry_run: bool = False, run_id: str | None = None) -> dict[str, Any]:
    prepared = prepare_experiment(config, pipeline, resource_plan=resource_plan, run_id=run_id, dry_run=dry_run or not yes)
    if dry_run or not yes:
        prepared["submission"] = {"status": "not_submitted", "reason": "--yes is required for non-interactive submission"}
        return prepared
    if config.data.get("require_slurm_account") and not prepared.get("resource_snapshot", {}).get("account"):
        raise SchedulerError("Formal submission requires a resolved Slurm account; set slurm_defaults.account or SLURM_ACCOUNT.")
    submission = submit_plan(prepared["scheduler_plan"], dry_run=False)
    write_json_atomic(Path(prepared["run_dir"]) / "submission_receipt.json", submission)
    prepared["submission"] = submission
    return prepared


def experiment_wizard(config: SchedulerConfig, pipeline: str, *, dry_run: bool = False) -> dict[str, Any]:
    train_cases, test_cases = _case_counts(config)
    snapshot = discover_resource_snapshot(config.path, account=config.slurm_defaults.get("account"), qos=config.slurm_defaults.get("qos"))
    print(snapshot_markdown(snapshot))
    candidates = recommend_resource_plans(snapshot, train_cases, test_cases, gpu_budget=(config.data.get("gpu_budget") or {}))
    for idx, plan in enumerate(candidates["plans"], start=1):
        marker = " (recommended)" if plan.get("recommended") else ""
        print(f"{idx}. {plan['plan_id']}{marker}")
    print(f"{len(candidates['plans']) + 1}. manual")
    print(f"{len(candidates['plans']) + 2}. exit without submitting")
    choice = input("Choose plan [balanced]: ").strip().lower() or "balanced"
    if choice.isdigit():
        number = int(choice)
        if number == len(candidates["plans"]) + 2:
            return {"status": "cancelled"}
        if 1 <= number <= len(candidates["plans"]):
            choice = candidates["plans"][number - 1]["plan_id"]
        elif number == len(candidates["plans"]) + 1:
            choice = "balanced"
    if choice == "manual":
        print("Manual mode currently records a balanced static-at-launch plan; unsupported multi-GPU choices are rejected before submission.")
        choice = "balanced"
    prepared = prepare_experiment(config, pipeline, resource_plan=choice, dry_run=True)
    print(json.dumps({"run_id": prepared["run_id"], "run_dir": prepared["run_dir"], "resource_plan": choice}, indent=2))
    if dry_run:
        prepared["submission"] = {"status": "dry_run"}
        return prepared
    confirm = input("Submit this experiment? [yes/no] ").strip().lower()
    if confirm != "yes":
        prepared["submission"] = {"status": "not_submitted", "reason": "default_no"}
        return prepared
    submission = submit_plan(prepared["scheduler_plan"], dry_run=False)
    write_json_atomic(Path(prepared["run_dir"]) / "submission_receipt.json", submission)
    prepared["submission"] = submission
    return prepared
