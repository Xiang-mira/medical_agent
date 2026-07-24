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
from .utils import SchedulerError, write_json_atomic


def _case_counts(cfg: SchedulerConfig) -> tuple[int, int]:
    train = resolve_path(cfg.paths.get("train_input_case_list") or cfg.paths.get("pilot_train_input_case_list"))
    test = resolve_path(cfg.paths.get("test_input_case_list") or cfg.paths.get("pilot_test_input_case_list"))
    if train is None or test is None:
        raise SchedulerError("Config must define train/test input manifests")
    return assert_strict_no_gt_manifest(train)["rows"], assert_strict_no_gt_manifest(test)["rows"]


def _write_launch_artifacts(run_dir: Path, snapshot: dict[str, Any], workloads: dict[str, Any], plans: dict[str, Any], selected: dict[str, Any]) -> None:
    write_json_atomic(run_dir / "resource_snapshot.json", snapshot)
    (run_dir / "resource_snapshot.md").write_text(snapshot_markdown(snapshot), encoding="utf-8")
    write_json_atomic(run_dir / "workload_estimates.json", workloads)
    (run_dir / "candidate_resource_plans.yaml").write_text(yaml.safe_dump(plans, sort_keys=False), encoding="utf-8")
    (run_dir / "selected_resource_plan.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    write_json_atomic(run_dir / "user_choices.json", {"plan_id": selected.get("plan_id"), "submitted": False})


def prepare_experiment(config: SchedulerConfig, pipeline: str, *, resource_plan: str = "balanced", run_id: str | None = None, dry_run: bool = True) -> dict[str, Any]:
    train_cases, test_cases = _case_counts(config)
    snapshot = discover_resource_snapshot(config.path, account=config.slurm_defaults.get("account"), qos=config.slurm_defaults.get("qos"))
    workloads = workload_estimates(train_cases, test_cases)
    candidates = recommend_resource_plans(snapshot, train_cases, test_cases, gpu_budget=(config.data.get("gpu_budget") or {}))
    selected = next((p for p in candidates["plans"] if p["plan_id"] == resource_plan), None)
    if selected is None:
        raise SchedulerError(f"Unknown resource plan {resource_plan!r}")
    validate_user_plan(selected, ddp_validated=bool(config.data.get("ddp_validated", False)), h100_ddp_allowed=bool(config.data.get("h100_ddp_allowed", False)))
    plan = build_plan(config, pipeline, backend="slurm", run_id=run_id)
    run_dir = Path(plan["run_dir"])
    _write_launch_artifacts(run_dir, snapshot, workloads, candidates, selected)
    return {"status": "planned", "dry_run": dry_run, "run_id": plan["run_id"], "run_dir": str(run_dir), "resource_plan": selected, "scheduler_plan": plan}


def launch_experiment(config: SchedulerConfig, pipeline: str, *, resource_plan: str = "balanced", yes: bool = False, dry_run: bool = False, run_id: str | None = None) -> dict[str, Any]:
    prepared = prepare_experiment(config, pipeline, resource_plan=resource_plan, run_id=run_id, dry_run=dry_run or not yes)
    if dry_run or not yes:
        prepared["submission"] = {"status": "not_submitted", "reason": "--yes is required for non-interactive submission"}
        return prepared
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
