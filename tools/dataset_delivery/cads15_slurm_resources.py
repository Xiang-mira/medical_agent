from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PANEL_PARTITION = "cpu"
DEFAULT_PANEL_CPUS = 8
DEFAULT_PANEL_MEMORY = "32G"
DEFAULT_PANEL_TIME = "02:00:00"
DEFAULT_GPU_PARTITION = "gpu"
DEFAULT_GPU_GRES = "gpu:T4:1"
DEFAULT_GPU_CPUS = 8
DEFAULT_GPU_MEMORY = "64G"
DEFAULT_GPU_TIME = "06:00:00"


@dataclass(frozen=True)
class SlurmResourceProfile:
    stage: str
    partition: str
    partition_source: str
    cpus_per_task: int
    cpus_source: str
    memory: str
    memory_source: str
    time_limit: str
    time_source: str
    gres: str = ""
    gres_source: str = "none"


def _resolve_text(cli_value: str | None, env_name: str, default_value: str) -> tuple[str, str]:
    if cli_value not in {None, ""}:
        return str(cli_value), "cli"
    env_value = os.getenv(env_name)
    if env_value not in {None, ""}:
        return str(env_value), f"env:{env_name}"
    return default_value, "project_default"


def _resolve_int(cli_value: int | None, env_name: str, default_value: int) -> tuple[int, str]:
    if cli_value is not None:
        return int(cli_value), "cli"
    env_value = os.getenv(env_name)
    if env_value not in {None, ""}:
        return int(str(env_value)), f"env:{env_name}"
    return default_value, "project_default"


def resolve_panel_profile(
    *,
    partition: str | None = None,
    cpus_per_task: int | None = None,
    memory: str | None = None,
    time_limit: str | None = None,
) -> SlurmResourceProfile:
    part, part_source = _resolve_text(partition, "TASK2_CPU_PARTITION", DEFAULT_PANEL_PARTITION)
    cpus, cpus_source = _resolve_int(cpus_per_task, "TASK2_CADS_PANEL_CPUS", DEFAULT_PANEL_CPUS)
    mem, mem_source = _resolve_text(memory, "TASK2_CADS_PANEL_MEMORY", DEFAULT_PANEL_MEMORY)
    wall, wall_source = _resolve_text(time_limit, "TASK2_CADS_PANEL_TIME", DEFAULT_PANEL_TIME)
    return SlurmResourceProfile(
        stage="panel_preparation",
        partition=part,
        partition_source=part_source,
        cpus_per_task=cpus,
        cpus_source=cpus_source,
        memory=mem,
        memory_source=mem_source,
        time_limit=wall,
        time_source=wall_source,
        gres="",
        gres_source="none",
    )


def resolve_gpu_profile(
    *,
    partition: str | None = None,
    gres: str | None = None,
    cpus_per_task: int | None = None,
    memory: str | None = None,
    time_limit: str | None = None,
) -> SlurmResourceProfile:
    part, part_source = _resolve_text(partition, "TASK2_CADS_GPU_PARTITION", DEFAULT_GPU_PARTITION)
    resolved_gres, gres_source = _resolve_text(gres, "TASK2_CADS_GPU_GRES", DEFAULT_GPU_GRES)
    cpus, cpus_source = _resolve_int(cpus_per_task, "TASK2_CADS_GPU_CPUS", DEFAULT_GPU_CPUS)
    mem, mem_source = _resolve_text(memory, "TASK2_CADS_GPU_MEMORY", DEFAULT_GPU_MEMORY)
    wall, wall_source = _resolve_text(time_limit, "TASK2_CADS_GPU_TIME", DEFAULT_GPU_TIME)
    return SlurmResourceProfile(
        stage="gpu_smoke",
        partition=part,
        partition_source=part_source,
        cpus_per_task=cpus,
        cpus_source=cpus_source,
        memory=mem,
        memory_source=mem_source,
        time_limit=wall,
        time_source=wall_source,
        gres=resolved_gres,
        gres_source=gres_source,
    )


def available_partitions() -> tuple[list[str], dict[str, Any]]:
    command = ["sinfo", "-h", "-o", "%P"]
    if shutil.which("sinfo") is None:
        return [], {"command": command, "return_code": 127, "stdout": "", "stderr": "sinfo not found", "ok": False, "skipped": True}
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    partitions: list[str] = []
    for line in proc.stdout.splitlines():
        name = line.strip().rstrip("*")
        if name and name not in partitions:
            partitions.append(name)
    return partitions, {
        "command": command,
        "return_code": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        return {"command": command, "return_code": 127, "stdout": "", "stderr": str(exc), "ok": False}
    return {
        "command": command,
        "return_code": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def sbatch_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#SBATCH")]


def preflight_sbatch_script(
    *,
    profile: SlurmResourceProfile,
    sbatch_file: Path,
    available: list[str] | None = None,
    available_query: dict[str, Any] | None = None,
    run_sbatch_test_only: bool = True,
) -> dict[str, Any]:
    if available is None or available_query is None:
        available, available_query = available_partitions()
    shell_check = run_command(["bash", "-n", str(sbatch_file)])
    partition_exists = bool(profile.partition and profile.partition in set(available))
    sbatch_test_only: dict[str, Any]
    if not shell_check.get("ok"):
        status = "PANEL_SCRIPT_INVALID" if profile.stage == "panel_preparation" else "GPU_SCRIPT_INVALID"
        reason = "bash_syntax_failed"
        sbatch_test_only = {"skipped": True, "reason": reason, "command": ["sbatch", "--test-only", str(sbatch_file)], "return_code": None, "stdout": "", "stderr": ""}
    elif available_query and available_query.get("skipped"):
        status = "READY"
        reason = "partition_check_skipped_sinfo_not_found"
        sbatch_test_only = {"skipped": True, "reason": reason, "command": ["sbatch", "--test-only", str(sbatch_file)], "return_code": None, "stdout": "", "stderr": ""}
    elif not partition_exists:
        status = "PANEL_RESOURCE_INVALID" if profile.stage == "panel_preparation" else "GPU_RESOURCE_INVALID"
        reason = f"partition_not_found:{profile.partition}"
        sbatch_test_only = {"skipped": True, "reason": "partition_not_found", "command": ["sbatch", "--test-only", str(sbatch_file)], "return_code": None, "stdout": "", "stderr": ""}
    elif not run_sbatch_test_only:
        status = "READY"
        reason = "sbatch_test_only_not_requested"
        sbatch_test_only = {"skipped": True, "reason": reason, "command": ["sbatch", "--test-only", str(sbatch_file)], "return_code": None, "stdout": "", "stderr": ""}
    else:
        sbatch_test_only = run_command(["sbatch", "--test-only", str(sbatch_file)])
        status = "READY" if sbatch_test_only.get("ok") else ("PANEL_RESOURCE_INVALID" if profile.stage == "panel_preparation" else "GPU_RESOURCE_INVALID")
        reason = "ready" if sbatch_test_only.get("ok") else "sbatch_test_only_failed"
    return {
        "stage": profile.stage,
        "status": status,
        "reason": reason,
        "configured_partition": profile.partition,
        "resolved_partition": profile.partition,
        "partition_source": profile.partition_source,
        "available_partitions": list(available or []),
        "partition_exists": partition_exists,
        "cpus_per_task": profile.cpus_per_task,
        "memory": profile.memory,
        "time_limit": profile.time_limit,
        "gres": profile.gres,
        "sbatch_file": str(sbatch_file),
        "sbatch_resource_lines": sbatch_lines(sbatch_file),
        "partition_query": available_query,
        "shell_syntax_check": shell_check,
        "sbatch_test_only": sbatch_test_only,
        "sbatch_test_only_command": sbatch_test_only.get("command"),
        "sbatch_test_only_return_code": sbatch_test_only.get("return_code"),
        "sbatch_test_only_stdout": sbatch_test_only.get("stdout", ""),
        "sbatch_test_only_stderr": sbatch_test_only.get("stderr", ""),
    }


def profile_manifest(profile: SlurmResourceProfile) -> dict[str, Any]:
    return {
        "stage": profile.stage,
        "partition": profile.partition,
        "partition_source": profile.partition_source,
        "cpus_per_task": profile.cpus_per_task,
        "cpus_source": profile.cpus_source,
        "memory": profile.memory,
        "memory_source": profile.memory_source,
        "time_limit": profile.time_limit,
        "time_source": profile.time_source,
        "gres": profile.gres,
        "gres_source": profile.gres_source,
    }
