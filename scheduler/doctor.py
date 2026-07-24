from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import SchedulerConfig, load_pipeline, load_resource_profiles, resolve_path
from .manifest import assert_strict_no_gt_manifest, validate_target_config, validate_target_mapping
from .resource_discovery import discover_resource_snapshot
from .utils import SchedulerError, git_snapshot, write_json_atomic
from .utils import read_json


def run_doctor(config: SchedulerConfig, pipeline: str, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    checks["git"] = git_snapshot()
    checks["python"] = {"executable": sys.executable, "version": sys.version.split()[0]}
    try:
        pipe = load_pipeline(pipeline)
        checks["pipeline"] = {"status": "success", "task_count": len(pipe.get("tasks", {}))}
    except Exception as exc:
        blockers.append(f"pipeline: {exc}")
        checks["pipeline"] = {"status": "failed", "error": str(exc)}
    try:
        target = validate_target_config(resolve_path(config.paths.get("target_config")) or Path(), expected_count=373)
        mapping_path = resolve_path(config.paths.get("target_mapping")) or Path()
        mapping_doc = read_json(mapping_path)
        expected_pilot = int(mapping_doc.get("pilot_target_count") or config.project.get("pilot_target_count", 338))
        mapping = validate_target_mapping(mapping_path, target["targets"], pilot_target_count=expected_pilot)
        checks["targets"] = {"status": "success", "target_count": target["target_count"], "mapping": mapping}
    except Exception as exc:
        blockers.append(f"targets: {exc}")
        checks["targets"] = {"status": "failed", "error": str(exc)}
    for key in ("train_input_case_list", "test_input_case_list", "pilot_train_input_case_list", "pilot_test_input_case_list"):
        if config.paths.get(key):
            try:
                checks[key] = assert_strict_no_gt_manifest(resolve_path(config.paths.get(key)) or Path())
            except Exception as exc:
                blockers.append(f"{key}: {exc}")
                checks[key] = {"status": "failed", "error": str(exc)}
    checks["scripts"] = {
        "case_selector_help": subprocess.run([sys.executable, "scripts/abdomenatlaspro_case_selector.py", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).returncode == 0
    }
    checks["slurm_commands"] = {cmd: bool(shutil.which(cmd)) for cmd in ("sbatch", "squeue", "sinfo", "scontrol", "sacct", "sacctmgr")}
    checks["resource_snapshot"] = discover_resource_snapshot(config.path, account=config.slurm_defaults.get("account"), qos=config.slurm_defaults.get("qos"))
    try:
        profiles = load_resource_profiles(resolve_path(config.data.get("resource_profiles", "configs/resource_profiles.yaml")) or Path())
        checks["resource_profiles"] = {"status": "success", "count": len(profiles)}
        label = profiles.get("labelcritic_72b_4h100_formal") or {}
        checks["labelcritic"] = {
            "container_exists": Path(str(label.get("container") or "")).exists(),
            "model_dir_exists": Path(str(label.get("model_dir") or "")).exists(),
            "gpu_count": label.get("gpu_count"),
            "tensor_parallel_size": label.get("tensor_parallel_size"),
        }
        checks["ddp"] = {
            "ddp_validated": bool(config.data.get("ddp_validated", False)),
            "h100_ddp_allowed": bool(config.data.get("h100_ddp_allowed", False)),
        }
    except Exception as exc:
        blockers.append(f"profiles: {exc}")
    run_root = resolve_path(config.paths.get("run_root") or "runs")
    checks["output_writable"] = {"path": str(run_root), "parent_writable": bool(run_root and run_root.parent.exists() and run_root.parent.stat())}
    status = "failed" if blockers else "success"
    report = {"status": status, "formal_blockers": blockers, "checks": checks}
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "doctor_report.json", report)
        (out / "doctor_report.md").write_text("# Scheduler Doctor\n\nStatus: " + status + "\n", encoding="utf-8")
    return report
