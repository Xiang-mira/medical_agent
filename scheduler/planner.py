from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .config import SchedulerConfig, load_pipeline, load_resource_profiles, resolve_path
from .manifest import read_case_csv
from .slurm import render_labelcritic_script, render_task_script
from .state import init_run_dirs
from .utils import ROOT, SchedulerError, git_snapshot, utc_now, write_json_atomic


def pipeline_path(name: str) -> Path:
    if name.endswith(".yaml") or "/" in name:
        path = resolve_path(name)
        assert path is not None
        return path
    return ROOT / "configs" / "pipelines" / f"{name}.yaml"


def make_run_id(pipeline: str) -> str:
    safe = pipeline.replace("/", "_").replace(".", "_")
    return f"{safe}_{utc_now().replace(':', '').replace('+0000', 'Z')}"


def _expand_arg(arg: str, config: SchedulerConfig, run_dir: Path) -> str:
    paths = config.paths
    replacements = {
        "${TARGET_CONFIG}": str(resolve_path(paths.get("target_config")) or ""),
        "${TRAIN_INPUT_CASE_LIST}": str(resolve_path(paths.get("train_input_case_list") or paths.get("pilot_train_input_case_list")) or ""),
        "${TRAIN_EVAL_CASE_LIST}": str(resolve_path(paths.get("train_eval_case_list") or paths.get("pilot_train_eval_case_list")) or ""),
        "${TEST_INPUT_CASE_LIST}": str(resolve_path(paths.get("test_input_case_list") or paths.get("pilot_test_input_case_list")) or ""),
        "${TEST_EVAL_CASE_LIST}": str(resolve_path(paths.get("test_eval_case_list") or paths.get("pilot_test_eval_case_list")) or ""),
        "${SELECTION_RUN_DIR}": str(resolve_path(paths.get("selection_run_dir")) or ""),
        "${PILOT_TRAIN_INPUT_CASE_LIST}": str(resolve_path(paths.get("pilot_train_input_case_list")) or ""),
        "${PILOT_TEST_INPUT_CASE_LIST}": str(resolve_path(paths.get("pilot_test_input_case_list")) or ""),
        "${PILOT_338_TARGET_CONFIG}": str(resolve_path(paths.get("pilot_target_config") or paths.get("target_config")) or ""),
        "${RUN_OUTPUT_DIR}": str(run_dir / "outputs"),
        "${RUN_CHECKPOINT_DIR}": str(run_dir / "checkpoints"),
        "${PSEUDO_LABEL_STUDENT_MANIFEST}": str(run_dir / "manifests" / "voxtell_prompt_student_manifest.json"),
        "${STUDENT_MODEL_DIR}": str(run_dir / "checkpoints" / "student"),
        "${ARRAY_CT_PATH}": "$ARRAY_CT_PATH",
        "${ARRAY_OUTPUT_DIR}": "$ARRAY_OUTPUT_DIR",
    }
    value = str(arg)
    for key, replacement in replacements.items():
        value = value.replace(key, replacement)
    return value


def _manifest_count(config: SchedulerConfig, key: str) -> int:
    path = resolve_path(config.paths.get(key))
    if path is None:
        raise SchedulerError(f"Missing manifest for dynamic array: {key}")
    return len(read_case_csv(path))


def _resolve_dynamic_array(task: dict[str, Any], config: SchedulerConfig) -> str | None:
    raw_array = task.get("array")
    if raw_array != "auto":
        return raw_array
    if task.get("array_manifest") == "train":
        count = _manifest_count(config, "train_input_case_list" if config.paths.get("train_input_case_list") else "pilot_train_input_case_list")
    elif task.get("array_manifest") == "test":
        count = _manifest_count(config, "test_input_case_list" if config.paths.get("test_input_case_list") else "pilot_test_input_case_list")
    else:
        raise SchedulerError(f"Task {task.get('task_name')} uses array=auto without array_manifest=train/test")
    if count <= 0:
        raise SchedulerError(f"Cannot build empty Slurm array for {task.get('task_name')}")
    concurrency = int(task.get("array_max_concurrent") or count)
    concurrency = max(1, min(concurrency, count))
    return f"0-{count - 1}%{concurrency}"


def command_for_task(task: dict[str, Any], run_dir: Path, config: SchedulerConfig) -> list[str]:
    entry = task.get("entrypoint")
    args = [_expand_arg(str(x), config, run_dir) for x in task.get("arguments", [])]
    if entry == "python_module":
        module = task.get("module")
        return ["python", "-m", str(module), *args]
    if entry == "python_script":
        return ["python", str(task.get("script")), *args]
    if entry == "shell":
        return ["bash", str(task.get("script")), *args]
    if entry == "scheduler_builtin":
        return ["python", "-m", "scheduler.cli", "run-task", "--task", str(task.get("task_name")), "--run-dir", str(run_dir)]
    raise SchedulerError(f"Unknown task entrypoint {entry!r} for {task.get('task_name')}")


def build_plan(config: SchedulerConfig, pipeline_name: str, *, backend: str, run_id: str | None = None) -> dict[str, Any]:
    pipeline = load_pipeline(pipeline_path(pipeline_name))
    profiles_path = resolve_path(config.data.get("resource_profiles", "configs/resource_profiles.yaml"))
    assert profiles_path is not None
    profiles = load_resource_profiles(profiles_path)
    run_root = resolve_path(config.paths.get("run_root") or "runs")
    assert run_root is not None
    run_id = run_id or make_run_id(pipeline_name)
    run_dir = run_root / run_id
    init_run_dirs(run_dir)
    config_snapshot = run_dir / "config_snapshot.yaml"
    config_snapshot.write_text(yaml.safe_dump(config.data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    tasks = []
    jobs: dict[str, Any] = {}
    for task_name, raw in pipeline["tasks"].items():
        task = dict(raw)
        task["task_name"] = task_name
        task["array"] = _resolve_dynamic_array(task, config)
        profile_name = task.get("resource_profile")
        profile = profiles.get(profile_name)
        if not profile:
            raise SchedulerError(f"Missing resource profile {profile_name!r} for task {task_name}")
        command = command_for_task(task, run_dir, config)
        wrapper_command = ["python", "-m", "scheduler.cli", "run-task", "--task", task_name, "--run-dir", str(run_dir)]
        if task.get("array"):
            wrapper_command.extend(["--array-index", "${SLURM_ARRAY_TASK_ID:-0}"])
        script_name = f"{task_name}.sbatch"
        script_path = run_dir / "generated_slurm" / script_name
        if backend == "slurm":
            if task.get("launcher") == "vllm_same_job":
                text = render_labelcritic_script(task, profile, wrapper_command, run_dir=run_dir)
            else:
                text = render_task_script(task, profile, wrapper_command, run_dir=run_dir)
            script_path.write_text(text, encoding="utf-8")
        tasks.append({
            "task_name": task_name,
            "task_type": task.get("task_type"),
            "dependencies": task.get("dependencies", []),
            "resource_profile": profile_name,
            "requested_resources": {
                "partition": profile.get("partition"),
                "gpu_count": profile.get("gpu_count", 0),
                "nodes": profile.get("nodes", 1),
                "cpus_per_task": profile.get("cpus_per_task"),
                "memory_gb": profile.get("memory_gb"),
                "walltime": profile.get("walltime"),
            },
            "requires_gpu": bool(profile.get("gpu_count")),
            "gt_access_policy": task.get("gt_access_policy"),
            "array": task.get("array"),
            "launcher": task.get("launcher"),
            "slurm_script": str(script_path) if backend == "slurm" else None,
            "command": command,
            "execution": task,
        })
        jobs[task_name] = {"status": "planned", "dependencies": task.get("dependencies", []), "script": str(script_path)}
    plan = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pipeline": pipeline_name,
        "backend": backend,
        "created_at": utc_now(),
        "git": git_snapshot(),
        "tasks": tasks,
    }
    write_json_atomic(run_dir / "plan.json", plan)
    write_json_atomic(run_dir / "jobs.json", jobs)
    return plan


def submit_plan(plan: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    run_dir = Path(plan["run_dir"])
    jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))
    submitted: dict[str, Any] = {}
    for task in plan["tasks"]:
        script = task.get("slurm_script")
        if not script:
            submitted[task["task_name"]] = {"status": "local_or_unplanned"}
            continue
        cmd = ["sbatch"]
        dep_ids = [submitted[d]["job_id"] for d in task.get("dependencies", []) if submitted.get(d, {}).get("job_id")]
        if dep_ids:
            cmd.append("--dependency=afterok:" + ":".join(dep_ids))
        cmd.append(script)
        if dry_run:
            result = {"status": "dry_run", "command": cmd}
        else:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0:
                raise SchedulerError(f"sbatch failed for {task['task_name']}: {proc.stderr}")
            job_id = proc.stdout.strip().split()[-1]
            status = _slurm_job_status(job_id)
            result = {
                "status": "submitted",
                "command": cmd,
                "job_id": job_id,
                "stdout": proc.stdout.strip(),
                "task_name": task["task_name"],
                "partition": (task.get("requested_resources") or {}).get("partition"),
                "requested_resources": task.get("requested_resources") or {},
                "dependency": next((x for x in cmd if str(x).startswith("--dependency=")), None),
                "submission_time": utc_now(),
                **status,
            }
        submitted[task["task_name"]] = result
        jobs[task["task_name"]].update(result)
    write_json_atomic(run_dir / "jobs.json", jobs)
    return {"run_id": plan["run_id"], "run_dir": str(run_dir), "dry_run": dry_run, "jobs": submitted}


def _slurm_job_status(job_id: str) -> dict[str, Any]:
    status = {"initial_state": "unknown", "pending_reason": None, "slurm_estimated_start": None}
    proc = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T|%R"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        line = proc.stdout.strip().splitlines()[0]
        parts = line.split("|", 1)
        status["initial_state"] = parts[0]
        if len(parts) > 1 and parts[0].upper().startswith("PEND"):
            status["pending_reason"] = parts[1]
    proc = subprocess.run(["squeue", "--start", "-h", "-j", str(job_id)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0 and proc.stdout.strip() and "N/A" not in proc.stdout:
        status["slurm_estimated_start"] = proc.stdout.strip().splitlines()[0]
    return status
