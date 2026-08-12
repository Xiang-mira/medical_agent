#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scheduler.resource_recommender import adaptive_overrequest_count  # noqa: E402
from scheduler.resource_discovery import discover_resource_snapshot  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_fieldnames, read_csv_rows, utc_now, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.slurm_reliability import (  # noqa: E402
    CANDIDATE_TASK_V1,
    DEFAULT_LOGICAL_TASK_NAMESPACE,
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    SUBMITTED_JOB_FIELDS,
    append_submission_attempt,
    classify_sbatch_failure,
    existing_active_logical_keys,
    parse_sbatch_job_id,
    persist_submitted_job,
    record_job_lifecycle,
    slurm_comment,
)


DEFAULT_GROUP_WEIGHTS = {
    "cads": 0.45,
    "atm": 0.15,
    "airrc": 0.20,
    "unest": 0.20,
}
DEFAULT_PROFILE_SPECS = "generic_gpu|gpu|gpu:1|8|64G|06:00:00"
DEFAULT_SLURM_MAX_ARRAY_SIZE_FALLBACK = 1000
GPU_VRAM_ESTIMATE_GB = {
    "T4": 16,
    "A10": 24,
    "A30": 24,
    "A40": 48,
    "A6000": 48,
    "L40": 48,
    "L40S": 48,
    "A100": 80,
    "H100": 80,
    "GPU": 16,
}


@dataclass(frozen=True)
class GpuSubmitProfile:
    name: str
    partition: str
    gres: str
    cpus_per_task: int
    mem: str
    time_limit: str

    @property
    def resource_class(self) -> str:
        upper = f"{self.partition} {self.gres} {self.name}".upper()
        if "H100" in upper:
            return "GPU_HIGH_MEMORY_H100"
        if "A100" in upper:
            return "GPU_HIGH_MEMORY_A100"
        if "T4" in upper:
            return "GPU_LIGHT_T4"
        if any(token in upper for token in ("L40", "A40", "A6000", "A30", "A10")):
            return "GPU_MEDIUM"
        return "GPU_INFERENCE_COMPATIBLE"


def parse_profile_specs(value: str) -> list[GpuSubmitProfile]:
    profiles: list[GpuSubmitProfile] = []
    for raw in str(value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 6:
            raise ValueError(
                "GPU profile specs must be comma-separated "
                "name|partition|gres|cpus|mem|time entries"
            )
        name, partition, gres, cpus, mem, time_limit = parts
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid GPU profile name: {name!r}")
        profiles.append(
            GpuSubmitProfile(
                name=name,
                partition=partition,
                gres=gres,
                cpus_per_task=int(cpus),
                mem=mem,
                time_limit=time_limit,
            )
        )
    if not profiles:
        raise ValueError("At least one GPU profile is required")
    return profiles


def _gpu_type_rank(gpu_type: str) -> int:
    upper = str(gpu_type or "").upper()
    if "T4" in upper:
        return 0
    if upper in {"GPU", "GENERIC"} or "GPU:1" == upper:
        return 1
    if any(token in upper for token in ("A10", "A30", "A40", "A6000", "L40")):
        return 2
    if "A100" in upper:
        return 3
    if "H100" in upper:
        return 4
    return 2


def build_auto_gpu_profiles(snapshot: dict[str, Any], *, min_vram_gb: int = 12, include_h100_overflow: bool = True) -> list[GpuSubmitProfile]:
    profiles: list[GpuSubmitProfile] = []
    for partition, row in sorted((snapshot.get("partitions") or {}).items()):
        if int(row.get("allocatable_configured_total") or row.get("gpus_configured_total") or 0) <= 0:
            continue
        gpu_type = str(row.get("gpu_type") or "GPU").upper()
        vram = next((gb for key, gb in GPU_VRAM_ESTIMATE_GB.items() if key in gpu_type), GPU_VRAM_ESTIMATE_GB["GPU"])
        if vram < min_vram_gb:
            continue
        if "H100" in gpu_type and not include_h100_overflow:
            continue
        name = re.sub(r"[^A-Za-z0-9_]+", "_", f"{partition}_{gpu_type.lower()}").strip("_").lower()
        gres_type = "" if gpu_type == "GPU" else f":{gpu_type}"
        mem = "96G" if vram >= 80 else ("80G" if vram >= 48 else "64G")
        profiles.append(
            GpuSubmitProfile(
                name=name or str(partition),
                partition=str(partition),
                gres=f"gpu{gres_type}:1",
                cpus_per_task=8,
                mem=mem,
                time_limit="06:00:00",
            )
        )
    profiles.sort(key=lambda profile: _gpu_type_rank(f"{profile.name} {profile.partition} {profile.gres}"))
    return profiles or parse_profile_specs(DEFAULT_PROFILE_SPECS)


def resolve_profiles(profile_specs: str, *, output_root: Path) -> tuple[list[GpuSubmitProfile], dict[str, Any]]:
    if str(profile_specs or "").strip().lower() not in {"", "auto", "cluster", "cluster_auto"}:
        return parse_profile_specs(profile_specs), {"mode": "explicit_profile_specs", "snapshot": None}
    snapshot = discover_resource_snapshot(output_dir=output_root / "slurm" / "resource_inventory", include_raw=False)
    profiles = build_auto_gpu_profiles(
        snapshot,
        min_vram_gb=int(os.getenv("TASK2_TEACHER_MIN_VRAM_GB", "12")),
        include_h100_overflow=os.getenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW", "1").strip().lower() not in {"0", "false", "no"},
    )
    return profiles, {"mode": "cluster_auto", "snapshot": snapshot}


def auto_worker_target(*, task_count: int, profiles: list[GpuSubmitProfile], resource_inventory: dict[str, Any], explicit_target: int, explicit_overrequest: int | None) -> tuple[int, int, dict[str, Any]]:
    safety_cap = int(os.getenv("TASK2_GPU_WORKER_SAFETY_CAP", "128"))
    partitions = (resource_inventory.get("snapshot") or {}).get("partitions") or {}
    compatible_capacity = 0
    for profile in profiles:
        part = partitions.get(profile.partition) or {}
        compatible_capacity += int(part.get("allocatable_configured_total") or part.get("gpus_configured_total") or 0)
    if compatible_capacity <= 0:
        compatible_capacity = len(profiles)
    target = int(explicit_target)
    source = "explicit"
    if target <= 0:
        target = min(max(1, int(task_count)), max(1, compatible_capacity), max(1, safety_cap))
        source = "cluster_inventory"
    else:
        target = min(target, max(1, int(task_count)), max(1, safety_cap))
    if explicit_overrequest is None or int(explicit_overrequest) <= 0:
        overrequest = min(max(1, int(task_count)), max(target, adaptive_overrequest_count(target)))
        over_source = "adaptive_overrequest_count"
    else:
        overrequest = min(max(1, int(task_count)), max(target, int(explicit_overrequest)), max(1, safety_cap))
        over_source = "explicit"
    return target, overrequest, {
        "target_source": source,
        "overrequest_source": over_source,
        "compatible_capacity": compatible_capacity,
        "safety_cap": safety_cap,
        "fixed_30_ceiling": False,
    }


def parse_group_weights(value: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_GROUP_WEIGHTS)
    if not value:
        return weights
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Group weights must use group=weight entries")
        group, weight = item.split("=", 1)
        weights[group.strip()] = float(weight)
    return weights


def _allocate_slots(keys: list[str], counts: dict[str, int], total_slots: int, weights: dict[str, float]) -> dict[str, int]:
    active = [key for key in keys if int(counts.get(key, 0)) > 0]
    if not active:
        return {key: 0 for key in keys}
    capped_total = min(max(1, int(total_slots)), sum(int(counts[key]) for key in active))
    allocation = {key: 0 for key in keys}
    active_weight_sum = sum(max(float(weights.get(key, 1.0)), 0.0) for key in active) or float(len(active))
    fractional: list[tuple[float, str]] = []
    for key in active:
        raw = capped_total * (max(float(weights.get(key, 1.0)), 0.0) / active_weight_sum)
        add = min(max(1, int(math.floor(raw))), int(counts[key]))
        allocation[key] = add
        fractional.append((raw - math.floor(raw), key))
    while sum(allocation.values()) > capped_total:
        reduced = False
        for _fraction, key in sorted(fractional):
            if allocation[key] > 1:
                allocation[key] -= 1
                reduced = True
                if sum(allocation.values()) <= capped_total:
                    break
        if not reduced:
            break
    while sum(allocation.values()) < capped_total:
        progressed = False
        for _fraction, key in sorted(fractional, reverse=True):
            if allocation[key] < int(counts[key]):
                allocation[key] += 1
                progressed = True
                if sum(allocation.values()) >= capped_total:
                    break
        if not progressed:
            break
    return allocation


def allocate_group_concurrency(
    groups: list[str],
    task_counts: dict[str, int],
    *,
    target_workers: int,
    overrequest_workers: int | None = None,
    group_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    planned = int(overrequest_workers or adaptive_overrequest_count(target_workers))
    return _allocate_slots(groups, task_counts, planned, group_weights or DEFAULT_GROUP_WEIGHTS)


def _replace_directive(lines: list[str], prefix: str, value: str) -> list[str]:
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            output.append(f"{prefix}{value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.insert(1, f"{prefix}{value}")
    return output


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed > 0 else None


def detect_max_array_size(explicit: int | None = None) -> dict[str, Any]:
    cli_value = _positive_int(explicit)
    if cli_value is not None:
        return {"value": cli_value, "source": "cli"}
    env_value = _positive_int(os.getenv("TASK2_SLURM_MAX_ARRAY_SIZE", ""))
    if env_value is not None:
        return {"value": env_value, "source": "env"}
    try:
        proc = subprocess.run(
            ["scontrol", "show", "config"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        return {
            "value": DEFAULT_SLURM_MAX_ARRAY_SIZE_FALLBACK,
            "source": "fallback",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    if proc.returncode == 0:
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", proc.stdout)
        if match:
            discovered = _positive_int(match.group(1))
            if discovered is not None:
                return {"value": discovered, "source": "scontrol", "stdout": proc.stdout.strip()}
    return {
        "value": DEFAULT_SLURM_MAX_ARRAY_SIZE_FALLBACK,
        "source": "fallback",
        "stderr": proc.stderr.strip(),
    }


def _logical_task_id(row: dict[str, Any]) -> str:
    existing = str(row.get("logical_task_id") or "").strip()
    if existing:
        return existing
    case_id = str(row.get("case_id") or "").strip()
    target = str(row.get("target") or "").strip()
    teacher = str(row.get("teacher") or "").strip()
    if case_id and target and teacher:
        return "|".join([case_id, target, teacher])
    return "|".join(
        [
            case_id,
            str(row.get("model_group") or row.get("group") or "").strip(),
            str(row.get("task_index") or row.get("source_manifest_row") or row.get("source_task_index") or "").strip(),
        ]
    )


def _chunk_rows(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size <= 0:
        raise ValueError(f"Invalid shard size: {size}")
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _rewrite_sbatch(
    source: Path,
    destination: Path,
    *,
    group: str,
    profile: GpuSubmitProfile,
    shard_id: str,
    task_manifest: Path,
    output_root: Path,
    state_root: Path,
    run_id: str,
    submission_id: str,
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    lines = _replace_directive(lines, "#SBATCH --job-name=", f"task2_{group}_{profile.name}_{shard_id}")
    lines = _replace_directive(lines, "#SBATCH --partition=", profile.partition)
    lines = _replace_directive(lines, "#SBATCH --gres=", profile.gres)
    lines = _replace_directive(lines, "#SBATCH --cpus-per-task=", str(profile.cpus_per_task))
    lines = _replace_directive(lines, "#SBATCH --mem=", profile.mem)
    lines = _replace_directive(lines, "#SBATCH --time=", profile.time_limit)
    lines = _replace_directive(lines, "#SBATCH --output=", str(output_root / "slurm" / f"{group}_{profile.name}_{shard_id}_%A_%a.out"))
    lines = _replace_directive(lines, "#SBATCH --error=", str(output_root / "slurm" / f"{group}_{profile.name}_{shard_id}_%A_%a.err"))
    lines = _replace_directive(lines, "#SBATCH --signal=", "B:USR1@900")
    rewritten: list[str] = []
    inserted_guard = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--task-manifest "):
            indent = line[: len(line) - len(line.lstrip())]
            suffix = " \\" if stripped.endswith("\\") else ""
            rewritten.append(f"{indent}--task-manifest {task_manifest}{suffix}")
        elif not inserted_guard and "task2_formal_launcher.py" in stripped:
            logical_task = f"{run_id}:{submission_id}:{CANDIDATE_TASK_V1}:{group}:{profile.name}:{shard_id}"
            rewritten.append(
                "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout "
                f"--state-root {state_root} "
                f"--logical-task-id {logical_task} "
                "--job-id \"${SLURM_JOB_ID:-}\" "
                f"--task-manifest {task_manifest} "
                "--task-index \"${SLURM_ARRAY_TASK_ID:-}\"' USR1"
            )
            rewritten.append(line)
            inserted_guard = True
        else:
            rewritten.append(line)
    if not inserted_guard:
        logical_task = f"{run_id}:{submission_id}:{CANDIDATE_TASK_V1}:{group}:{profile.name}:{shard_id}"
        rewritten.append(
            "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout "
            f"--state-root {state_root} "
            f"--logical-task-id {logical_task} "
            "--job-id \"${SLURM_JOB_ID:-}\" "
            f"--task-manifest {task_manifest} "
            "--task-index \"${SLURM_ARRAY_TASK_ID:-}\"' USR1"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    destination.chmod(0o755)


def _write_shard_manifest(source_manifest: Path, destination: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_fields = read_csv_fieldnames(source_manifest)
    extra_fields = [
        "local_array_index",
        "global_logical_index",
        "logical_task_id",
        "teacher_family",
        "source_manifest_row",
        "source_task_index",
        "execution_schema_version",
        "manifest_schema_version",
        "logical_task_namespace",
    ]
    fieldnames = ["task_index", *[field for field in source_fields if field != "task_index"], *[field for field in extra_fields if field not in source_fields]]
    shard_rows: list[dict[str, Any]] = []
    for new_index, row in enumerate(rows):
        updated = dict(row)
        source_manifest_row = str(
            row.get("source_manifest_row")
            or row.get("source_task_index")
            or row.get("global_logical_index")
            or row.get("task_index")
            or new_index
        )
        updated["task_index"] = new_index
        updated["local_array_index"] = str(new_index)
        updated["global_logical_index"] = str(row.get("global_logical_index") or source_manifest_row)
        updated["logical_task_id"] = _logical_task_id(row)
        updated["teacher_family"] = str(row.get("teacher_family") or row.get("teacher") or "")
        updated["source_manifest_row"] = source_manifest_row
        updated["source_task_index"] = source_manifest_row
        updated["execution_schema_version"] = CANDIDATE_TASK_V1
        updated["manifest_schema_version"] = DEFAULT_MANIFEST_SCHEMA_VERSION
        updated["logical_task_namespace"] = DEFAULT_LOGICAL_TASK_NAMESPACE
        shard_rows.append(updated)
    write_csv(destination, shard_rows, fieldnames)
    return shard_rows


def _validate_array_plan(row: dict[str, Any], *, max_array_size: int) -> dict[str, Any]:
    manifest_path = Path(str(row["task_manifest"]))
    sbatch_path = Path(str(row["sbatch_file"]))
    task_count = int(row.get("task_count") or 0)
    concurrency = int(row.get("array_concurrency") or 0)
    local_start = int(str(row.get("local_start", 0)))
    local_end = int(str(row.get("local_end", -1)))
    manifest_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    logical_ids = [str(item.get("logical_task_id") or "") for item in manifest_rows]
    duplicate_count = len(logical_ids) - len(set(logical_ids))
    errors: list[str] = []
    if task_count <= 0:
        errors.append("task_count_must_be_positive")
    if local_start != 0:
        errors.append("local_start_must_equal_zero")
    if task_count > 0 and local_end != task_count - 1:
        errors.append("local_end_must_equal_task_count_minus_one")
    if local_end >= max_array_size:
        errors.append("local_end_exceeds_max_array_size")
    if concurrency <= 0 or concurrency > max(1, task_count):
        errors.append("invalid_array_concurrency")
    if len(manifest_rows) != task_count:
        errors.append("manifest_row_count_mismatch")
    if duplicate_count > 0:
        errors.append("duplicate_logical_task_ids_within_shard")
    if not sbatch_path.exists():
        errors.append("sbatch_file_missing")
    if not manifest_path.exists():
        errors.append("manifest_missing")
    return {
        "status": "READY" if not errors else "INVALID_ARRAY_PLAN",
        "errors": errors,
        "profile": row.get("profile"),
        "shard_id": row.get("shard_id"),
        "task_count": task_count,
        "array_spec": row.get("array_spec"),
        "concurrency": concurrency,
        "max_array_size": max_array_size,
        "manifest": str(manifest_path),
        "duplicate_logical_task_ids": duplicate_count,
        "manifest_rows": len(manifest_rows),
    }


def _preflight_sbatch(row: dict[str, Any], *, run_sbatch_test_only: bool) -> dict[str, Any]:
    sbatch_file = str(row["sbatch_file"])
    shell_proc = subprocess.run(
        ["bash", "-n", sbatch_file],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report: dict[str, Any] = {
        "status": "READY" if shell_proc.returncode == 0 else "SHELL_INVALID",
        "bash_n_return_code": int(shell_proc.returncode),
        "bash_n_stdout": shell_proc.stdout.strip(),
        "bash_n_stderr": shell_proc.stderr.strip(),
        "sbatch_test_only_return_code": None,
        "sbatch_test_only_stdout": "",
        "sbatch_test_only_stderr": "",
        "sbatch_test_only_skipped": not run_sbatch_test_only,
    }
    if shell_proc.returncode != 0:
        return report
    if run_sbatch_test_only:
        array_spec = str(row["array_spec"])
        sbatch_proc = subprocess.run(
            ["sbatch", "--test-only", f"--array={array_spec}", sbatch_file],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        report.update(
            {
                "status": "READY" if sbatch_proc.returncode == 0 else "SBATCH_INVALID",
                "sbatch_test_only_return_code": int(sbatch_proc.returncode),
                "sbatch_test_only_stdout": sbatch_proc.stdout.strip(),
                "sbatch_test_only_stderr": sbatch_proc.stderr.strip(),
                "sbatch_test_only_skipped": False,
                "sbatch_test_only_array_spec": array_spec,
            }
        )
    return report


def build_dynamic_submission_plan(
    *,
    summary_path: Path,
    output_root: Path,
    state_root: Path,
    target_workers: int,
    overrequest_workers: int | None,
    profile_specs: str,
    groups: list[str] | None = None,
    group_weights: dict[str, float] | None = None,
    dry_run: bool = False,
    run_sbatch_test_only: bool = True,
    submission_id: str = "",
    append_submitted_jobs: bool = False,
    run_id: str = "",
    git_commit: str = "",
    max_array_size: int | None = None,
    max_new_shards_per_round: int | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "READY":
        raise RuntimeError(f"Formal Task2 preflight is not READY: {summary.get('status')}")
    profiles, resource_inventory = resolve_profiles(profile_specs, output_root=output_root)
    summary_groups = summary.get("groups") or {}
    requested_groups = groups or [group for group in ("cads", "atm", "airrc", "unest") if group in summary_groups]
    task_counts = {
        group: int((summary_groups.get(group) or {}).get("task_count") or 0)
        for group in requested_groups
    }
    ready_task_total = sum(int(value) for value in task_counts.values())
    target_workers, planned_overrequest, worker_sizing = auto_worker_target(
        task_count=ready_task_total,
        profiles=profiles,
        resource_inventory=resource_inventory,
        explicit_target=target_workers,
        explicit_overrequest=overrequest_workers,
    )
    group_slots = allocate_group_concurrency(
        requested_groups,
        task_counts,
        target_workers=target_workers,
        overrequest_workers=planned_overrequest,
        group_weights=group_weights,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    slurm_root = output_root / "slurm"
    safe_submission_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(submission_id or "").strip()).strip("_")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(run_id or os.getenv("ROUND1_RUN_ID") or state_root.name or "round1").strip()).strip("_")
    dynamic_root = slurm_root / "dynamic" / safe_submission_id if safe_submission_id else slurm_root / "dynamic"
    dynamic_root.mkdir(parents=True, exist_ok=True)
    active_logical_keys = existing_active_logical_keys(slurm_root)
    max_array_size_info = detect_max_array_size(max_array_size)
    detected_max_array_size = int(max_array_size_info["value"])
    per_round_limit = _positive_int(max_new_shards_per_round)
    if per_round_limit is None:
        per_round_limit = _positive_int(os.getenv("TASK2_MAX_NEW_SHARDS_PER_ROUND", ""))
    if per_round_limit is None:
        per_round_limit = max(1, len(requested_groups) * max(1, len(profiles)))

    shard_rows: list[dict[str, Any]] = []
    sharding_reports: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    for group in requested_groups:
        group_summary = summary_groups.get(group) or {}
        task_manifest = Path(str(group_summary.get("task_manifest") or ""))
        source_sbatch = Path(str(group_summary.get("sbatch_file") or ""))
        task_count = int(task_counts.get(group, 0))
        if task_count <= 0:
            continue
        if not task_manifest.exists():
            raise FileNotFoundError(f"Task manifest not found for {group}: {task_manifest}")
        if not source_sbatch.exists():
            raise FileNotFoundError(f"Sbatch file not found for {group}: {source_sbatch}")
        rows = read_csv_rows(task_manifest)
        for index, row in enumerate(rows):
            logical_id = _logical_task_id(row)
            updated = dict(row)
            updated["global_logical_index"] = str(index)
            updated["logical_task_id"] = logical_id
            updated["source_manifest_row"] = str(row.get("task_index") or index)
            updated["source_task_index"] = str(row.get("task_index") or index)
            updated["teacher_family"] = str(row.get("teacher_family") or row.get("teacher") or "")
            all_source_rows.append(updated)
        rows = all_source_rows[-len(rows):]
        active_profiles = profiles[: min(len(profiles), len(rows))]
        profile_counts = {profile.name: len(rows[index:: len(active_profiles)]) for index, profile in enumerate(active_profiles)}
        profile_weights = {profile.name: 1.0 for profile in active_profiles}
        profile_slots = _allocate_slots(
            [profile.name for profile in active_profiles],
            profile_counts,
            group_slots[group],
            profile_weights,
        )
        for profile_index, profile in enumerate(active_profiles):
            rows_for_profile = rows[profile_index:: len(active_profiles)]
            if not rows_for_profile:
                continue
            shard_sets = _chunk_rows(rows_for_profile, detected_max_array_size)
            sharding_reports.append(
                {
                    "group": group,
                    "profile": profile.name,
                    "task_count": len(rows_for_profile),
                    "shard_count": len(shard_sets),
                    "first_shard_task_count": len(shard_sets[0]) if shard_sets else 0,
                    "last_shard_task_count": len(shard_sets[-1]) if shard_sets else 0,
                }
            )
            profile_concurrency = max(1, min(int(profile_slots.get(profile.name) or 1), len(rows_for_profile)))
            for shard_index, shard in enumerate(shard_sets):
                shard_id = f"shard_{shard_index:03d}"
                manifest_for_shard = dynamic_root / f"{group}_{profile.name}_{shard_id}_task_manifest.csv"
                written_rows = _write_shard_manifest(task_manifest, manifest_for_shard, shard)
                sbatch_for_shard = dynamic_root / f"{group}_{profile.name}_{shard_id}_task2_array.sbatch"
                _rewrite_sbatch(
                    source_sbatch,
                    sbatch_for_shard,
                    group=group,
                    profile=profile,
                    shard_id=shard_id,
                    task_manifest=manifest_for_shard,
                    output_root=output_root,
                    state_root=state_root,
                    run_id=safe_run_id,
                    submission_id=safe_submission_id or "default",
                )
                shard_task_count = len(written_rows)
                local_end = shard_task_count - 1
                concurrency = max(1, min(profile_concurrency, shard_task_count))
                array_spec = f"0-{local_end}%{concurrency}"
                logical_task_id = f"{safe_run_id}:{safe_submission_id}:{CANDIDATE_TASK_V1}:{group}:{profile.name}:{shard_id}"
                row = {
                    "run_id": safe_run_id,
                    "submission_id": safe_submission_id,
                    "logical_task_id": logical_task_id,
                    "execution_schema_version": CANDIDATE_TASK_V1,
                    "manifest_schema_version": DEFAULT_MANIFEST_SCHEMA_VERSION,
                    "logical_task_namespace": DEFAULT_LOGICAL_TASK_NAMESPACE,
                    "model_group": group,
                    "group": group,
                    "profile": profile.name,
                    "shard_id": shard_id,
                    "shard_index": str(shard_index),
                    "partition": profile.partition,
                    "gres": profile.gres,
                    "cpus_per_task": profile.cpus_per_task,
                    "mem": profile.mem,
                    "time_limit": profile.time_limit,
                    "task_count": shard_task_count,
                    "array_concurrency": concurrency,
                    "task_manifest": str(manifest_for_shard),
                    "sbatch_file": str(sbatch_for_shard),
                    "job_id": "",
                    "array_job_id": "",
                    "array_task_id": "",
                    "display_id": "",
                    "submission_status": "dry_run" if dry_run else "pending",
                    "preflight_status": "",
                    "preflight_stderr": "",
                    "stderr": "",
                    "array_range": f"0-{local_end}",
                    "array_spec": array_spec,
                    "local_start": 0,
                    "local_end": local_end,
                    "comment": slurm_comment(
                        run_id=safe_run_id,
                        submission_id=safe_submission_id or "default",
                        group=group,
                        profile=profile.name,
                        execution_schema_version=CANDIDATE_TASK_V1,
                        shard_id=shard_id,
                    ),
                    "formal_root": str(output_root),
                    "state_root": str(state_root),
                    "git_commit": git_commit,
                    "scheduler_status": "PLANNED",
                }
                row["array_plan_validation"] = _validate_array_plan(row, max_array_size=detected_max_array_size)
                shard_rows.append(row)

    all_logical_ids = [_logical_task_id(row) for row in all_source_rows]
    unique_logical_ids = sorted(set(all_logical_ids))
    sharded_logical_ids: list[str] = []
    for row in shard_rows:
        for manifest_row in read_csv_rows(Path(str(row["task_manifest"]))):
            sharded_logical_ids.append(str(manifest_row.get("logical_task_id") or ""))
    sharded_unique_ids = sorted(set(sharded_logical_ids))
    duplicate_count = len(sharded_logical_ids) - len(sharded_unique_ids)
    missing_ids = sorted(set(unique_logical_ids) - set(sharded_unique_ids))
    all_array_specs_valid = all(
        str(row.get("array_plan_validation", {}).get("status") or "") == "READY"
        for row in shard_rows
    )
    sharding_audit = {
        "status": "READY" if duplicate_count == 0 and not missing_ids and all_array_specs_valid else "INVALID_ARRAY_PLAN",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "max_array_size": detected_max_array_size,
        "max_array_size_source": max_array_size_info.get("source"),
        "profile_count": len(profiles),
        "shard_count": len(shard_rows),
        "logical_task_count": len(all_logical_ids),
        "sharded_task_count": len(sharded_logical_ids),
        "unique_logical_task_ids": len(sharded_unique_ids),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing_ids),
        "missing_logical_task_ids": missing_ids[:50],
        "first_shard": shard_rows[0] if shard_rows else {},
        "last_shard": shard_rows[-1] if shard_rows else {},
        "all_array_specs_valid": all_array_specs_valid,
        "profiles": sharding_reports,
        "jobs": shard_rows,
        "legacy_active_jobs_adopted": 0,
        "compressed_job_ids_as_queryable": 0,
    }
    write_json(slurm_root / "sharding_audit.json", sharding_audit)
    if sharding_audit["status"] != "READY":
        plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
        write_json(
            plan_path,
            {
                "status": "INVALID_ARRAY_PLAN",
                "scheduler_status": "FATAL",
                "scheduler_mode": "dynamic_gpu_overrequest",
                "submission_id": safe_submission_id,
                "run_id": safe_run_id,
                "sharding_audit": sharding_audit,
                "jobs": shard_rows,
            },
        )
        raise RuntimeError("INVALID_ARRAY_PLAN: shard plan failed validation; no jobs were submitted")

    if not dry_run:
        preflight_failures = []
        for row in shard_rows:
            array_validation = row.get("array_plan_validation") or {}
            if array_validation.get("status") != "READY":
                row["preflight_status"] = "INVALID_ARRAY_PLAN"
                row["preflight_stderr"] = ",".join(array_validation.get("errors") or [])
                preflight_failures.append({"job": row, "preflight": array_validation})
                continue
            report = _preflight_sbatch(row, run_sbatch_test_only=run_sbatch_test_only)
            row["preflight_status"] = report["status"]
            row["preflight_stderr"] = report.get("sbatch_test_only_stderr") or report.get("bash_n_stderr") or ""
            row["preflight"] = report
            if report["status"] != "READY":
                preflight_failures.append({"job": row, "preflight": report})
        if preflight_failures:
            plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
            write_json(
                plan_path,
                {
                    "status": "PREFLIGHT_FAILED",
                    "scheduler_mode": "dynamic_gpu_overrequest",
                    "planned_target_workers": int(target_workers),
                    "planned_overrequest_workers": planned_overrequest,
                    "submission_id": safe_submission_id,
                    "resource_inventory": resource_inventory,
                    "worker_sizing": worker_sizing,
                    "sharding_audit": sharding_audit,
                    "failures": preflight_failures,
                    "jobs": shard_rows,
                },
            )
            raise RuntimeError(f"Dynamic GPU sbatch preflight failed for {len(preflight_failures)} shard(s); no jobs were submitted")

    csv_fields = SUBMITTED_JOB_FIELDS
    backpressured: list[dict[str, Any]] = []
    fatal_failures: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    submit_budget = max(1, per_round_limit)
    for row in shard_rows:
        if dry_run:
            continue
        logical_key = (
            str(row.get("execution_schema_version") or ""),
            str(row.get("submission_id") or ""),
            str(row["model_group"]),
            str(row["profile"]),
            str(row.get("shard_id") or ""),
        )
        if logical_key in active_logical_keys:
            row["submission_status"] = "reused"
            row["scheduler_status"] = "ADOPTED_ACTIVE_JOB"
            row["status"] = "ADOPTED_ACTIVE_JOB"
            row["submitted_at"] = utc_now()
            persist_submitted_job(slurm_root, row)
            record_job_lifecycle(
                state_root,
                {
                    "status": "ADOPTED_ACTIVE_JOB",
                    "logical_task_id": row["logical_task_id"],
                    "submission_id": safe_submission_id,
                    "execution_schema_version": CANDIDATE_TASK_V1,
                    "group": row["model_group"],
                    "profile": row["profile"],
                    "shard_id": row["shard_id"],
                    "partition": row["partition"],
                    "gres": row["gres"],
                    "task_manifest": row["task_manifest"],
                    "attempt": "reuse_existing_active",
                },
            )
            reused.append(dict(row))
            jobs.append(dict(row))
            continue
        if submit_budget <= 0:
            row["submission_status"] = "deferred"
            row["scheduler_status"] = "READY_SHARD_QUEUE"
            continue
        task_count = int(row["task_count"])
        concurrency = int(row["array_concurrency"])
        comment = str(row["comment"])
        array_spec = str(row["array_spec"])
        proc = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--comment", comment,
                f"--array={array_spec}",
                str(row["sbatch_file"]),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        parsed_job = parse_sbatch_job_id(proc.stdout)
        row["job_id"] = parsed_job["job_id"]
        row["array_job_id"] = parsed_job["array_job_id"]
        row["array_task_id"] = parsed_job["array_task_id"]
        row["display_id"] = parsed_job["display_id"]
        row["submission_status"] = "submitted" if proc.returncode == 0 else "failed"
        row["stderr"] = proc.stderr.strip()
        row["stdout"] = proc.stdout.strip()
        row["submitted_at"] = utc_now()
        attempt = {
            **row,
            "return_code": int(proc.returncode),
            "command": [
                "sbatch",
                "--parsable",
                "--comment", comment,
                f"--array={array_spec}",
                str(row["sbatch_file"]),
            ],
        }
        append_submission_attempt(slurm_root, attempt)
        if proc.returncode != 0:
            classification = classify_sbatch_failure(proc.stderr or proc.stdout)
            row["failure_reason"] = classification["reason"]
            row["scheduler_status"] = "BACKPRESSURED" if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE" else "FATAL"
            if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
                backpressured.append(dict(row))
                continue
            fatal_failures.append(dict(row))
            continue
        row["status"] = "SUBMITTED"
        row["scheduler_status"] = "ACTIVE"
        row["slurm_state"] = "PENDING"
        persist_submitted_job(slurm_root, row)
        submit_budget -= 1
        record_job_lifecycle(
            state_root,
            {
                "status": "SUBMITTED",
                "scheduler_state": "ACTIVE",
                "job_id": row["job_id"],
                "array_job_id": row["array_job_id"],
                "array_task_id": row["array_task_id"],
                "display_id": row["display_id"],
                "logical_task_id": row["logical_task_id"],
                "submission_id": safe_submission_id,
                "execution_schema_version": CANDIDATE_TASK_V1,
                "group": row["model_group"],
                "profile": row["profile"],
                "shard_id": row["shard_id"],
                "partition": row["partition"],
                "gres": row["gres"],
                "task_manifest": row["task_manifest"],
                "time_limit": row["time_limit"],
                "formal_root": str(output_root),
                "git_commit": git_commit,
            },
        )
        jobs.append(dict(row))
    if fatal_failures:
        plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
        write_json(
            plan_path,
            {
                "status": "SUBMISSION_FAILED",
                "scheduler_status": "FATAL",
                "submission_id": safe_submission_id,
                "sharding_audit": sharding_audit,
                "jobs": jobs,
                "backpressured_jobs": backpressured,
                "failed_jobs": fatal_failures,
            },
        )
        raise RuntimeError(f"sbatch fatal failure for {len(fatal_failures)} shard(s): {fatal_failures[0].get('stderr')}")
    if not dry_run and not jobs and not (slurm_root / "submitted_jobs.csv").exists():
        write_csv(slurm_root / "submitted_jobs.csv", [], csv_fields)
        write_json(slurm_root / "submitted_jobs.json", {"updated_at": utc_now(), "jobs": []})
    jobs_csv = slurm_root / "submitted_jobs.csv"
    if dry_run:
        csv_rows = shard_rows
        if append_submitted_jobs and jobs_csv.exists():
            csv_rows = read_csv_rows(jobs_csv) + shard_rows
        write_csv(jobs_csv, csv_rows, csv_fields)

    scheduler_status = "ACTIVE"
    status = "DRY_RUN" if dry_run else "SUBMITTED"
    if not dry_run and backpressured:
        scheduler_status = "BACKPRESSURED"
        status = "PARTIALLY_SUBMITTED" if jobs else "WAITING_FOR_SUBMISSION_CAPACITY"
    elif not dry_run and any(str(row.get("submission_status") or "") == "deferred" for row in shard_rows):
        scheduler_status = "ACTIVE" if jobs else "READY_SHARD_QUEUE"
        status = "PARTIALLY_SUBMITTED" if jobs else "READY_SHARD_QUEUE"

    plan = {
        "status": status,
        "scheduler_status": scheduler_status,
        "scheduler_mode": "dynamic_gpu_overrequest",
        "created_at": utc_now(),
        "submission_id": safe_submission_id,
        "run_id": safe_run_id,
        "summary_path": str(summary_path),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "planned_target_workers": int(target_workers),
        "planned_overrequest_workers": planned_overrequest,
        "execution_schema_version": CANDIDATE_TASK_V1,
        "manifest_schema_version": DEFAULT_MANIFEST_SCHEMA_VERSION,
        "logical_task_namespace": DEFAULT_LOGICAL_TASK_NAMESPACE,
        "max_array_size": detected_max_array_size,
        "max_array_size_source": max_array_size_info.get("source"),
        "max_new_shards_per_round": per_round_limit,
        "total_task_count": sum(int(row["task_count"]) for row in shard_rows),
        "logical_task_count": len(all_logical_ids),
        "sharded_task_count": len(sharded_logical_ids),
        "unique_logical_task_count": len(sharded_unique_ids),
        "duplicate_logical_task_count": duplicate_count,
        "missing_logical_task_count": len(missing_ids),
        "total_array_concurrency": sum(int(row["array_concurrency"]) for row in shard_rows),
        "group_concurrency": group_slots,
        "profile_specs": [{**profile.__dict__, "resource_class": profile.resource_class} for profile in profiles],
        "resource_policy": {
            "teacher_resource_requirement": "GPU_INFERENCE_COMPATIBLE",
            "profile_priority_order": "T4/generic/immediately-compatible GPUs first; A100/H100 are opportunistic overflow, not scientific ownership",
            "labelcritic_h100_reservation": os.getenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW", "1").strip().lower() in {"0", "false", "no"},
            "h100_overflow_policy": "DEFERRED_LOW_UTILITY_PROFILE when LabelCritic service is active unless TASK2_ALLOW_H100_TEACHER_OVERFLOW=1",
        },
        "resource_inventory": resource_inventory,
        "worker_sizing": worker_sizing,
        "sharding_audit": sharding_audit,
        "jobs": shard_rows,
        "submitted_jobs": jobs,
        "reused_jobs": reused,
        "backpressured_jobs": backpressured,
        "deferred_jobs": [row for row in shard_rows if str(row.get("submission_status") or "") == "deferred"],
        "submitted_job_count": len([job for job in jobs if str(job.get("submission_status")) == "submitted"]),
        "reused_job_count": len(reused),
        "backpressured_count": len(backpressured),
        "dependency_policy": {
            "parallel": [
                "cads/atm/airrc/unest teacher arrays are independent after manifest/preflight",
                "ShapeKit postprocessing runs inside each case/group worker after teacher inference",
                "LabelCritic candidate selection can run per case/target once candidate masks exist",
            ],
            "serial": [
                "per-case workspace staging and Task1 rename/materialization precede that case's teacher task",
                "global manifest/config/model preflight precedes Slurm submissions",
                "within one case/group: teacher inference -> recovery -> ShapeKit -> LabelCritic selection -> delivery validation",
                "M-step student training waits for E-step selected pseudo labels to be READY",
            ],
        },
        "cpu_gpu_policy": {
            "cpu": [
                "workspace staging and rename/materialization",
                "manifest/preflight/validation/recovery inventory",
                "Slurm orchestration and status aggregation",
                "ShapeKit geometry cleanup inside allocated worker CPU cores",
            ],
            "gpu": [
                "teacher model inference arrays",
                "formal LabelCritic 72B vLLM service",
                "student M-step training and later student inference",
            ],
        },
    }
    plan_name = f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json"
    write_json(slurm_root / plan_name, plan)
    if safe_submission_id:
        write_json(slurm_root / "dynamic_gpu_submission_plan.json", plan)
    if not dry_run:
        state_root.mkdir(parents=True, exist_ok=True)
        tmp_last = state_root / f".last_task2_formal.tmp.{os.getpid()}"
        tmp_last.write_text(str(output_root) + "\n", encoding="utf-8")
        tmp_last.replace(state_root / ".last_task2_formal")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamically submit Task2 formal GPU arrays with over-requested concurrency.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--target-workers", default=0, type=int)
    parser.add_argument("--overrequest-workers", default=None, type=int)
    parser.add_argument("--profile-specs", default="auto")
    parser.add_argument("--groups", default="")
    parser.add_argument("--group-weights", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sbatch-test-only", action="store_true")
    parser.add_argument("--submission-id", default="")
    parser.add_argument("--append-submitted-jobs", action="store_true")
    parser.add_argument("--run-id", default=os.getenv("ROUND1_RUN_ID", ""))
    parser.add_argument("--git-commit", default=os.getenv("EXPECTED_GIT_COMMIT", ""))
    args = parser.parse_args()
    groups = [item.strip() for item in args.groups.replace(";", ",").split(",") if item.strip()] or None
    plan = build_dynamic_submission_plan(
        summary_path=args.summary.resolve(),
        output_root=args.output_root.resolve(),
        state_root=args.state_root.resolve(),
        target_workers=args.target_workers,
        overrequest_workers=args.overrequest_workers,
        profile_specs=args.profile_specs,
        groups=groups,
        group_weights=parse_group_weights(args.group_weights),
        dry_run=bool(args.dry_run),
        run_sbatch_test_only=not bool(args.skip_sbatch_test_only),
        submission_id=args.submission_id,
        append_submitted_jobs=bool(args.append_submitted_jobs),
        run_id=args.run_id,
        git_commit=args.git_commit,
    )
    print(json.dumps({k: plan[k] for k in ("status", "scheduler_status", "planned_target_workers", "planned_overrequest_workers", "total_array_concurrency")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
