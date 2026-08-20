#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
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
    ACTIVE_STATES,
    CANDIDATE_TASK_V1,
    DEFAULT_LOGICAL_TASK_NAMESPACE,
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    IMPOSSIBLE_PENDING_REASONS,
    SUBMITTED_JOB_FIELDS,
    append_submission_attempt,
    classify_sbatch_failure,
    existing_active_logical_keys,
    load_submitted_jobs,
    parse_sbatch_job_id,
    persist_submitted_job,
    query_slurm_worker_units,
    record_job_lifecycle,
    current_worker_accounting_from_rows,
    reconcile_submitted_worker_accounting,
    slurm_comment,
)
from tools.dataset_delivery.task2_h100_policy import resolve_teacher_h100_policy  # noqa: E402


DEFAULT_GROUP_WEIGHTS = {
    "cads": 0.45,
    "atm": 0.15,
    "airrc": 0.20,
    "unest": 0.20,
}
DEFAULT_PROFILE_SPECS = "generic_gpu|gpu|gpu:1|8|64G|06:00:00"
DEFAULT_SLURM_MAX_ARRAY_SIZE_FALLBACK = 1000
DEFAULT_QOS_CONSERVATIVE_FALLBACK = 20
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
QOS_CACHE_FILENAME = "qos_capacity_cache.json"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _slurm_mutations_disabled() -> bool:
    return str(os.getenv("TASK2_DISABLE_SLURM_MUTATIONS") or "").strip().lower() in {"1", "true", "yes", "on"}


def _run_sbatch_submit(command: list[str]) -> subprocess.CompletedProcess[str]:
    if _slurm_mutations_disabled():
        return subprocess.CompletedProcess(
            command,
            125,
            "",
            "SLURM_MUTATION_DISABLED: refusing real scheduler mutation while TASK2_DISABLE_SLURM_MUTATIONS=1",
        )
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


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


def _profile_from_plan(value: dict[str, Any]) -> GpuSubmitProfile | None:
    try:
        return GpuSubmitProfile(
            name=str(value["name"]),
            partition=str(value["partition"]),
            gres=str(value["gres"]),
            cpus_per_task=int(value["cpus_per_task"]),
            mem=str(value["mem"]),
            time_limit=str(value["time_limit"]),
        )
    except Exception:
        return None


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


def _time_to_seconds(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw or raw.upper() in {"UNLIMITED", "NOT_SET", "N/A"}:
        return None
    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        try:
            days = int(day_text)
        except ValueError:
            return None
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        else:
            hours, minutes, seconds = int(parts[0]), 0, 0
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _seconds_to_time(seconds: int) -> str:
    seconds = max(60, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _memory_to_gb(value: Any) -> int:
    raw = str(value or "").strip().upper()
    if not raw:
        return 0
    try:
        if raw.endswith("G"):
            return int(float(raw[:-1]))
        if raw.endswith("M"):
            return max(1, int(float(raw[:-1]) / 1024))
        if raw.endswith("T"):
            return int(float(raw[:-1]) * 1024)
        return int(float(raw) / 1024) if float(raw) > 512 else int(float(raw))
    except Exception:
        return 0


def _int_first(mapping: dict[str, Any], keys: list[str], default: int = 0) -> int:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            try:
                return int(mapping.get(key) or 0)
            except Exception:
                return default
    return default


def _effective_profile_time_limit(partition_row: dict[str, Any], *, requested: str, interactive_short: bool) -> str:
    requested_sec = _time_to_seconds(requested) or 6 * 3600
    max_sec = _time_to_seconds(partition_row.get("time_limit") or partition_row.get("time"))
    if not interactive_short or max_sec is None:
        return requested
    configured_short = _time_to_seconds(os.getenv("TASK2_INTERACTIVE_T4_SHORT_TIME", "03:50:00")) or 13800
    safety = int(os.getenv("TASK2_INTERACTIVE_T4_TIME_SAFETY_SEC", "600"))
    return _seconds_to_time(min(requested_sec, configured_short, max(60, max_sec - safety)))


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
        interactive_short = str(partition).lower() == "interactive" and "T4" in gpu_type
        name = "interactive_t4_short" if interactive_short else re.sub(r"[^A-Za-z0-9_]+", "_", f"{partition}_{gpu_type.lower()}").strip("_").lower()
        gres_type = "" if gpu_type == "GPU" else f":{gpu_type}"
        mem = "96G" if vram >= 80 else ("80G" if vram >= 48 else "64G")
        profiles.append(
            GpuSubmitProfile(
                name=name or str(partition),
                partition=str(partition),
                gres=f"gpu{gres_type}:1",
                cpus_per_task=8,
                mem=mem,
                time_limit=_effective_profile_time_limit(row, requested="06:00:00", interactive_short=interactive_short),
            )
        )
    profiles.sort(key=lambda profile: _gpu_type_rank(f"{profile.name} {profile.partition} {profile.gres}"))
    return profiles or parse_profile_specs(DEFAULT_PROFILE_SPECS)


def resolve_profiles(profile_specs: str, *, output_root: Path, include_h100_overflow: bool = True) -> tuple[list[GpuSubmitProfile], dict[str, Any]]:
    if str(profile_specs or "").strip().lower() not in {"", "auto", "cluster", "cluster_auto"}:
        return parse_profile_specs(profile_specs), {"mode": "explicit_profile_specs", "snapshot": None}
    snapshot = discover_resource_snapshot(output_dir=output_root / "slurm" / "resource_inventory", include_raw=False)
    profiles = build_auto_gpu_profiles(
        snapshot,
        min_vram_gb=int(os.getenv("TASK2_TEACHER_MIN_VRAM_GB", "12")),
        include_h100_overflow=include_h100_overflow,
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


def _qos_cache_path(slurm_root: Path) -> Path:
    return slurm_root / QOS_CACHE_FILENAME


def _load_qos_cache(slurm_root: Path) -> dict[str, Any]:
    path = _qos_cache_path(slurm_root)
    if not path.exists():
        return {"profiles": {}, "updated_at": ""}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            doc.setdefault("profiles", {})
            return doc
    except Exception:
        pass
    return {"profiles": {}, "updated_at": ""}


def _save_qos_cache(slurm_root: Path, cache: dict[str, Any]) -> None:
    cache = dict(cache)
    cache["updated_at"] = utc_now()
    write_json(_qos_cache_path(slurm_root), cache)


def _profile_env_var_candidates(profile: GpuSubmitProfile) -> list[str]:
    raw = f"{profile.name}_{profile.partition}_{profile.gres}".upper()
    normalized = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    parts = [part for part in normalized.split("_") if part]
    candidates = [f"TASK2_{normalized}_MAX_ARRAY_TASKS"]
    upper = f"{profile.name} {profile.partition} {profile.gres}".upper()
    if "T4" in parts or profile.partition.lower() == "gpu" or profile.resource_class in {"GPU_LIGHT_T4", "GPU_INFERENCE_COMPATIBLE"}:
        candidates.extend(["TASK2_GPU_T4_MAX_ARRAY_TASKS", "TASK2_INTERACTIVE_T4_MAX_ARRAY_TASKS" if "INTERACTIVE" in parts else "TASK2_T4_MAX_ARRAY_TASKS"])
    if "A100" in parts:
        candidates.append("TASK2_A100_MAX_ARRAY_TASKS")
    if "H100" in parts:
        candidates.append("TASK2_H100_MAX_ARRAY_TASKS")
    if profile.partition:
        candidates.append(f"TASK2_{re.sub(r'[^A-Z0-9]+', '_', profile.partition.upper()).strip('_')}_MAX_ARRAY_TASKS")
    deduped: list[str] = []
    for key in candidates:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _profile_seed_capacity(profile: GpuSubmitProfile, *, max_array_size: int, h100_enabled: bool) -> dict[str, Any]:
    upper = f"{profile.name} {profile.partition} {profile.gres}".upper()
    if "H100" in upper and not h100_enabled:
        return {"known_good_size": 0, "known_bad_size": 1, "source": "labelcritic_reserved"}
    if "INTERACTIVE" in upper and "T4" in upper:
        return {"known_good_size": 1, "known_bad_size": 25, "source": "seed_observed_qos"}
    if "A100" in upper:
        return {"known_good_size": min(max_array_size, 500), "known_bad_size": 1000, "source": "seed_observed_qos"}
    if "H100" in upper:
        return {"known_good_size": min(max_array_size, 250), "known_bad_size": 500, "source": "seed_observed_qos"}
    if "T4" in upper or profile.resource_class in {"GPU_LIGHT_T4", "GPU_INFERENCE_COMPATIBLE"}:
        return {"known_good_size": max_array_size, "known_bad_size": None, "source": "seed_t4_primary"}
    return {"known_good_size": min(max_array_size, DEFAULT_QOS_CONSERVATIVE_FALLBACK), "known_bad_size": None, "source": "seed_conservative"}


def _profile_qos_state(profile: GpuSubmitProfile, *, slurm_root: Path, max_array_size: int, h100_enabled: bool) -> dict[str, Any]:
    cache = _load_qos_cache(slurm_root)
    cached = dict((cache.get("profiles") or {}).get(profile.name) or {})
    for env_name in _profile_env_var_candidates(profile):
        env_value = _positive_int(os.getenv(env_name, ""))
        if env_value is not None:
            return {
                "profile": profile.name,
                "known_good_size": min(max_array_size, env_value),
                "known_bad_size": cached.get("known_bad_size"),
                "effective_limit": min(max_array_size, env_value),
                "source": f"env:{env_name}",
                "last_backpressure": cached.get("last_backpressure") or {},
            }
    seed = _profile_seed_capacity(profile, max_array_size=max_array_size, h100_enabled=h100_enabled)
    known_good = _positive_int(cached.get("known_good_size"))
    if known_good is None:
        known_good = _positive_int(seed.get("known_good_size")) or 0
    known_bad = _positive_int(cached.get("known_bad_size"))
    if known_bad is None:
        known_bad = _positive_int(seed.get("known_bad_size"))
    effective = min(max_array_size, known_good) if known_good > 0 else 0
    return {
        "profile": profile.name,
        "known_good_size": known_good,
        "known_bad_size": known_bad,
        "effective_limit": effective,
        "source": cached.get("source") or seed.get("source") or "seed",
        "last_backpressure": cached.get("last_backpressure") or {},
    }


def _update_qos_cache(
    slurm_root: Path,
    *,
    profile: GpuSubmitProfile,
    attempted_size: int,
    success: bool = False,
    backpressured: bool = False,
    source: str = "",
    reason: str = "",
) -> dict[str, Any]:
    cache = _load_qos_cache(slurm_root)
    profiles = cache.setdefault("profiles", {})
    current = dict(profiles.get(profile.name) or {})
    if success and attempted_size > 0:
        current["known_good_size"] = max(int(current.get("known_good_size") or 0), int(attempted_size))
    if backpressured and attempted_size > 0:
        bad = int(current.get("known_bad_size") or 0)
        current["known_bad_size"] = attempted_size if bad <= 0 else min(bad, attempted_size)
        current["last_backpressure"] = {
            "time": utc_now(),
            "attempted_array_size": int(attempted_size),
            "reason": reason,
            "source": source,
        }
    if source:
        current["source"] = source
    profiles[profile.name] = current
    _save_qos_cache(slurm_root, cache)
    return current


def _profile_capacity_hint(profile: GpuSubmitProfile, resource_inventory: dict[str, Any]) -> int:
    partitions = (resource_inventory.get("snapshot") or {}).get("partitions") or {}
    part = partitions.get(profile.partition) or {}
    capacity = int(part.get("resource_feasible_gpu_slots_by_profile", {}).get(profile.name) or 0)
    if capacity <= 0:
        capacity = int(part.get("allocatable_configured_total") or part.get("gpus_configured_total") or 0)
    return capacity if capacity > 0 else 0


def _profile_feasible_slots(profile: GpuSubmitProfile, resource_inventory: dict[str, Any]) -> dict[str, Any]:
    partitions = (resource_inventory.get("snapshot") or {}).get("partitions") or {}
    part = partitions.get(profile.partition) or {}
    gpu_capacity_slots = _int_first(part, ["allocatable_configured_total", "gpus_configured_total"], 0)
    gpu_slots = _int_first(part, ["idle_estimate", "gpus_idle_estimate"], gpu_capacity_slots)
    cpu_total = _int_first(part, ["cpus_total", "cpus_idle_estimate"], 0)
    cpu_idle = _int_first(part, ["cpus_idle_estimate"], cpu_total)
    mem_total_gb = _int_first(part, ["memory_total_gb", "memory_idle_gb_estimate", "memory_free_gb_estimate"], 0)
    mem_idle_gb = _int_first(part, ["memory_idle_gb_estimate", "memory_free_gb_estimate"], mem_total_gb)
    mem_margin = int(os.getenv("NODE_MEMORY_SAFETY_MARGIN_GB", "8"))
    mem_per_worker = max(1, _memory_to_gb(profile.mem))
    cpu_slots = cpu_idle // max(1, int(profile.cpus_per_task)) if cpu_idle > 0 else gpu_slots
    memory_slots = max(0, mem_idle_gb - mem_margin) // mem_per_worker if mem_idle_gb > 0 else gpu_slots
    feasible = max(0, min(gpu_slots, cpu_slots, memory_slots))
    cpu_capacity_slots = cpu_total // max(1, int(profile.cpus_per_task)) if cpu_total > 0 else gpu_capacity_slots
    memory_capacity_slots = max(0, mem_total_gb - mem_margin) // mem_per_worker if mem_total_gb > 0 else gpu_capacity_slots
    capacity = max(0, min(gpu_capacity_slots, cpu_capacity_slots, memory_capacity_slots))
    return {
        "profile": profile.name,
        "partition_resource_known": bool(part),
        "gpu_slots": gpu_slots,
        "cpu_slots": cpu_slots,
        "memory_slots": memory_slots,
        "feasible_slots": feasible,
        "gpu_capacity_slots": gpu_capacity_slots,
        "cpu_capacity_slots": cpu_capacity_slots,
        "memory_capacity_slots": memory_capacity_slots,
        "capacity_slots": capacity,
        "memory_per_worker_gb": mem_per_worker,
        "memory_safety_margin_gb": mem_margin,
        "cpu_per_worker": int(profile.cpus_per_task),
    }


def _profile_desired_weight(profile: GpuSubmitProfile, *, h100_enabled: bool) -> float:
    upper = f"{profile.name} {profile.partition} {profile.gres}".upper()
    if "H100" in upper:
        return 0.05 if h100_enabled else 0.0
    if "A100" in upper:
        return 0.35
    if "INTERACTIVE" in upper and "T4" in upper:
        return 0.10
    if "T4" in upper or profile.resource_class in {"GPU_LIGHT_T4", "GPU_INFERENCE_COMPATIBLE"}:
        return 1.0
    return 0.25


def _profile_desired_ceiling(profile: GpuSubmitProfile, *, planned_workers: int, resource_inventory: dict[str, Any], h100_enabled: bool) -> int:
    upper = f"{profile.name} {profile.partition} {profile.gres}".upper()
    feasible = _profile_feasible_slots(profile, resource_inventory)
    physical = _profile_capacity_hint(profile, resource_inventory)
    if feasible.get("partition_resource_known"):
        ceiling = int(feasible.get("capacity_slots") or 0)
    else:
        ceiling = physical if physical > 0 else int(planned_workers)
    if "H100" in upper:
        return 0 if not h100_enabled else min(ceiling, int(os.getenv("TASK2_H100_DESIRED_MAX", "2")))
    if "A100" in upper:
        return min(ceiling, int(os.getenv("TASK2_A100_DESIRED_MAX", "4")))
    if "INTERACTIVE" in upper and "T4" in upper:
        if os.getenv("RESOURCE_ROUTING_MODE", "shadow").strip().lower() != "enforce":
            return 0
        partitions = (resource_inventory.get("snapshot") or {}).get("partitions") or {}
        part = partitions.get(profile.partition) or {}
        idle = _int_first(part, ["idle_estimate", "gpus_idle_estimate"], int(feasible.get("capacity_slots") or 0))
        physical = _int_first(part, ["allocatable_configured_total", "gpus_configured_total"], 0)
        available = min(int(feasible.get("capacity_slots") or 0), idle)
        cap = int(os.getenv("TASK2_INTERACTIVE_T4_DESIRED_MAX", os.getenv("TASK2_INTERACTIVE_DESIRED_MAX", str(max(1, physical or ceiling)))))
        return min(ceiling, max(0, available), max(1, cap))
    return ceiling


def _desired_workers_by_profile(
    *,
    profiles: list[GpuSubmitProfile],
    planned_workers: int,
    logical_task_count: int,
    resource_inventory: dict[str, Any],
    qos_states: dict[str, dict[str, Any]],
    h100_enabled: bool,
) -> dict[str, int]:
    total_slots = min(max(0, int(planned_workers)), max(0, int(logical_task_count)))
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for profile in profiles:
        qos_limit = int((qos_states.get(profile.name) or {}).get("effective_limit") or 0)
        if qos_limit <= 0:
            counts[profile.name] = 0
            weights[profile.name] = 0.0
            continue
        counts[profile.name] = max(
            0,
            _profile_desired_ceiling(
                profile,
                planned_workers=total_slots,
                resource_inventory=resource_inventory,
                h100_enabled=h100_enabled,
            ),
        )
        weights[profile.name] = _profile_desired_weight(profile, h100_enabled=h100_enabled)
    return _allocate_slots([profile.name for profile in profiles], counts, total_slots, weights)


def _slurm_job_state(job_id: str) -> dict[str, Any]:
    query_id = str(job_id or "").split("_", 1)[0].strip()
    if not query_id:
        return {"state": "UNKNOWN", "job_id": str(job_id or "")}
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", query_id, "-o", "%T|%R"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        return {"state": "UNKNOWN", "job_id": str(job_id or ""), "source": "squeue_error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode == 0 and proc.stdout.strip():
        parts = proc.stdout.splitlines()[0].split("|", 1)
        return {"state": parts[0].strip(), "reason": parts[1].strip() if len(parts) > 1 else "", "job_id": str(job_id or ""), "source": "squeue"}
    try:
        proc = subprocess.run(
            ["sacct", "-n", "-j", query_id, "--format=State,Reason", "-P"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        return {"state": "UNKNOWN", "job_id": str(job_id or ""), "source": "sacct_error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode == 0 and proc.stdout.strip():
        parts = proc.stdout.splitlines()[0].split("|")
        return {"state": parts[0].strip(), "reason": parts[1].strip() if len(parts) > 1 else "", "job_id": str(job_id or ""), "source": "sacct"}
    return {"state": "UNKNOWN", "job_id": str(job_id or ""), "source": "unknown"}


def _slurm_worker_units(row: dict[str, Any]) -> dict[str, Any]:
    return query_slurm_worker_units(row)


def _worker_pool_counts(slurm_root: Path, profiles: list[GpuSubmitProfile], *, execution_attempt_id: str = "") -> dict[str, dict[str, int]]:
    profile_names = {profile.name for profile in profiles}
    reconciled = reconcile_submitted_worker_accounting(
        slurm_root,
        profiles=profile_names,
        execution_attempt_id=execution_attempt_id,
        job_state_fn=_slurm_worker_units,
    )
    return reconciled["counts"]


def _render_queue_worker_sbatch(
    destination: Path,
    *,
    profile: GpuSubmitProfile,
    task_manifest: Path,
    output_root: Path,
    state_root: Path,
    run_id: str,
    submission_id: str,
    worker_shard_id: str,
    python_executable: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=task2_full373_{profile.name}_{worker_shard_id}",
        f"#SBATCH --partition={profile.partition}",
        f"#SBATCH --gres={profile.gres}",
        f"#SBATCH --cpus-per-task={profile.cpus_per_task}",
        f"#SBATCH --mem={profile.mem}",
        f"#SBATCH --time={profile.time_limit}",
        f"#SBATCH --output={output_root / 'slurm' / f'full373_{profile.name}_{worker_shard_id}_%A_%a.out'}",
        f"#SBATCH --error={output_root / 'slurm' / f'full373_{profile.name}_{worker_shard_id}_%A_%a.err'}",
        "#SBATCH --signal=B:USR1@900",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        "export RUNTIME_NO_GIT=1",
        "export SKIP_GIT_SYNC=1",
        "export GIT_TERMINAL_PROMPT=0",
        f"export RESOURCE_ROUTING_MODE={shlex.quote(os.getenv('RESOURCE_ROUTING_MODE', 'shadow'))}",
        f"export TASK2_WORKER_WALLTIME_SEC={_time_to_seconds(profile.time_limit) or 0}",
        f"export TASK2_WORKER_PROFILE={shlex.quote(profile.name)}",
        f"export TASK2_WORKER_REQUESTED_MEM={shlex.quote(profile.mem)}",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        (
            "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout "
            f"--state-root {shlex.quote(str(state_root))} "
            f"--logical-task-id {shlex.quote(f'{run_id}:{submission_id}:{CANDIDATE_TASK_V1}:full373:{profile.name}:{worker_shard_id}')} "
            "--job-id \"${SLURM_JOB_ID:-}\" "
            f"--task-manifest {shlex.quote(str(task_manifest))} "
            "--task-index \"${SLURM_ARRAY_TASK_ID:-}\"' USR1"
        ),
        (
            f"{shlex.quote(str(python_executable or sys.executable))} tools/dataset_delivery/task2_full373_round1_launcher.py "
            f"--output-root {shlex.quote(str(output_root))} "
            f"--task-manifest {shlex.quote(str(task_manifest))} "
            f"--state-root {shlex.quote(str(state_root))} "
            "--queue-worker "
            f"--worker-profile {shlex.quote(profile.name)} "
            f"--worker-resource-class {shlex.quote(profile.resource_class)} "
            "--worker-id \"${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}\""
        ),
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    destination.chmod(0o755)


def _preflight_worker_array(row: dict[str, Any], *, run_sbatch_test_only: bool) -> dict[str, Any]:
    report = _preflight_sbatch(row, run_sbatch_test_only=run_sbatch_test_only)
    if report["status"] == "SBATCH_INVALID":
        classification = classify_sbatch_failure(report.get("sbatch_test_only_stderr") or report.get("bash_n_stderr") or "")
        if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
            report["status"] = "BACKPRESSURED"
            report["failure_reason"] = classification["reason"]
    return report


def _write_shared_manifest(source_manifest: Path, destination: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_fields = read_csv_fieldnames(source_manifest)
    extra_fields = [
        "global_logical_index",
        "logical_task_id",
        "teacher_family",
        "source_manifest_row",
        "source_task_index",
        "execution_schema_version",
        "manifest_schema_version",
        "logical_task_namespace",
    ]
    fieldnames = [*source_fields, *[field for field in extra_fields if field not in source_fields]]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        source_manifest_row = str(
            row.get("source_manifest_row")
            or row.get("source_task_index")
            or row.get("task_index")
            or index
        )
        updated["task_index"] = str(row.get("task_index") or index)
        updated["global_logical_index"] = str(row.get("global_logical_index") or index)
        updated["logical_task_id"] = _logical_task_id(row)
        updated["teacher_family"] = str(row.get("teacher_family") or row.get("teacher") or "")
        updated["source_manifest_row"] = source_manifest_row
        updated["source_task_index"] = source_manifest_row
        updated["execution_schema_version"] = CANDIDATE_TASK_V1
        updated["manifest_schema_version"] = DEFAULT_MANIFEST_SCHEMA_VERSION
        updated["logical_task_namespace"] = DEFAULT_LOGICAL_TASK_NAMESPACE
        normalized.append(updated)
    write_csv(destination, normalized, fieldnames)
    return normalized


def _validate_worker_array_plan(row: dict[str, Any], *, max_array_size: int, logical_task_count: int) -> dict[str, Any]:
    manifest_path = Path(str(row["task_manifest"]))
    sbatch_path = Path(str(row["sbatch_file"]))
    worker_count = int(row.get("task_count") or 0)
    local_start = int(str(row.get("local_start", 0)))
    local_end = int(str(row.get("local_end", -1)))
    manifest_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    logical_ids = [str(item.get("logical_task_id") or "") for item in manifest_rows]
    duplicate_count = len(logical_ids) - len(set(logical_ids))
    errors: list[str] = []
    if worker_count <= 0:
        errors.append("worker_count_must_be_positive")
    if local_start != 0:
        errors.append("local_start_must_equal_zero")
    if worker_count > 0 and local_end != worker_count - 1:
        errors.append("local_end_must_equal_task_count_minus_one")
    if local_end >= max_array_size:
        errors.append("local_end_exceeds_max_array_size")
    if len(manifest_rows) != logical_task_count:
        errors.append("shared_manifest_row_count_mismatch")
    if duplicate_count > 0:
        errors.append("duplicate_logical_task_ids_in_shared_manifest")
    if not sbatch_path.exists():
        errors.append("sbatch_file_missing")
    if not manifest_path.exists():
        errors.append("manifest_missing")
    return {
        "status": "READY" if not errors else "INVALID_ARRAY_PLAN",
        "errors": errors,
        "profile": row.get("profile"),
        "shard_id": row.get("shard_id"),
        "task_count": worker_count,
        "array_spec": row.get("array_spec"),
        "max_array_size": max_array_size,
        "manifest": str(manifest_path),
        "duplicate_logical_task_ids": duplicate_count,
        "manifest_rows": len(manifest_rows),
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
    execution_attempt_id: str = "",
    append_submitted_jobs: bool = False,
    run_id: str = "",
    git_commit: str = "",
    max_array_size: int | None = None,
    max_new_shards_per_round: int | None = None,
    allow_h100_teacher_overflow: Any | None = None,
    labelcritic_required: Any | None = None,
    labelcritic_job_id: str | None = None,
    labelcritic_job_state: str | None = None,
    labelcritic_h100_reserved: Any | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "READY":
        raise RuntimeError(f"Formal Task2 preflight is not READY: {summary.get('status')}")
    h100_policy = resolve_teacher_h100_policy(
        state_root=state_root,
        allow_h100_teacher_overflow=allow_h100_teacher_overflow,
        labelcritic_required=labelcritic_required,
        labelcritic_job_id=labelcritic_job_id,
        labelcritic_job_state=labelcritic_job_state,
        labelcritic_h100_reserved=labelcritic_h100_reserved,
    )
    h100_enabled = bool(h100_policy.get("effective_teacher_h100_enabled"))
    profiles, resource_inventory = resolve_profiles(profile_specs, output_root=output_root, include_h100_overflow=h100_enabled)
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

    output_root.mkdir(parents=True, exist_ok=True)
    slurm_root = output_root / "slurm"
    safe_submission_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(submission_id or "").strip()).strip("_")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(run_id or os.getenv("ROUND1_RUN_ID") or state_root.name or "round1").strip()).strip("_")
    safe_execution_attempt_id = re.sub(
        r"[^A-Za-z0-9_.:-]+",
        "_",
        str(execution_attempt_id or os.getenv("TASK2_EXECUTION_ATTEMPT_ID") or f"{safe_run_id}_{safe_submission_id or 'default'}").strip(),
    ).strip("_")
    dynamic_root = slurm_root / "dynamic" / safe_submission_id if safe_submission_id else slurm_root / "dynamic"
    dynamic_root.mkdir(parents=True, exist_ok=True)
    max_array_size_info = detect_max_array_size(max_array_size)
    detected_max_array_size = int(max_array_size_info["value"])
    per_round_limit = _positive_int(max_new_shards_per_round)
    if per_round_limit is None:
        per_round_limit = _positive_int(os.getenv("TASK2_MAX_NEW_SHARDS_PER_ROUND", ""))
    if per_round_limit is None:
        per_round_limit = max(1, planned_overrequest, len(profiles))

    all_source_rows: list[dict[str, Any]] = []
    manifest_reference: Path | None = None
    python_executable = sys.executable
    for group in requested_groups:
        group_summary = summary_groups.get(group) or {}
        task_manifest = Path(str(group_summary.get("task_manifest") or ""))
        task_count = int(task_counts.get(group, 0))
        if task_count <= 0:
            continue
        if not task_manifest.exists():
            raise FileNotFoundError(f"Task manifest not found for {group}: {task_manifest}")
        manifest_reference = manifest_reference or task_manifest
        rows = read_csv_rows(task_manifest)
        for index, row in enumerate(rows):
            logical_id = _logical_task_id(row)
            updated = dict(row)
            updated["group"] = str(row.get("group") or group)
            updated["model_group"] = str(row.get("model_group") or group)
            updated["global_logical_index"] = str(index)
            updated["logical_task_id"] = logical_id
            updated["source_manifest_row"] = str(row.get("task_index") or index)
            updated["source_task_index"] = str(row.get("task_index") or index)
            updated["teacher_family"] = str(row.get("teacher_family") or row.get("teacher") or "")
            if row.get("python"):
                python_executable = str(row.get("python"))
            all_source_rows.append(updated)
    if manifest_reference is None:
        manifest_reference = summary_path

    shared_manifest = dynamic_root / "shared_ready_candidate_manifest.csv"
    shared_rows = _write_shared_manifest(manifest_reference, shared_manifest, all_source_rows)
    from cli_anything.medai.core.totalseg_runner import preflight_totalseg_executable
    from tools.dataset_delivery.task2_full373_round1_launcher import recover_candidate_seed_marker, recover_stale_candidate_claims, repair_recoverable_failed_candidates, seed_candidate_states

    seed_recovery = recover_candidate_seed_marker(output_root, task_manifest=shared_manifest)
    candidate_seed = seed_recovery if seed_recovery.get("status") == "READY" else seed_candidate_states(output_root, task_manifest=shared_manifest)
    stale_claim_recovery = recover_stale_candidate_claims(output_root, task_manifest=shared_manifest)
    target_config_value = next((str(row.get("target_config") or "").strip() for row in shared_rows if str(row.get("target_config") or "").strip()), "")
    target_config_for_repair = Path(target_config_value) if target_config_value else None
    configured_totalseg = next((str(row.get("totalsegmentator_executable") or "") for row in shared_rows if str(row.get("totalsegmentator_executable") or "")), "")
    previous_totalseg = os.environ.get("TOTAL_SEGMENTATOR_EXECUTABLE")
    if configured_totalseg:
        os.environ["TOTAL_SEGMENTATOR_EXECUTABLE"] = configured_totalseg
    try:
        totalseg_preflight = preflight_totalseg_executable()
    finally:
        if configured_totalseg:
            if previous_totalseg is None:
                os.environ.pop("TOTAL_SEGMENTATOR_EXECUTABLE", None)
            else:
                os.environ["TOTAL_SEGMENTATOR_EXECUTABLE"] = previous_totalseg
    failed_candidate_repair = repair_recoverable_failed_candidates(
        output_root,
        task_manifest=shared_manifest,
        target_config=target_config_for_repair if target_config_for_repair and target_config_for_repair.exists() else None,
        totalseg_executable_ready=totalseg_preflight.get("status") == "ok",
        execution_attempt_id=safe_execution_attempt_id,
    )
    all_logical_ids = [_logical_task_id(row) for row in all_source_rows]
    sharded_logical_ids = [str(row.get("logical_task_id") or "") for row in shared_rows]
    unique_logical_ids = sorted(set(all_logical_ids))
    sharded_unique_ids = sorted(set(sharded_logical_ids))
    duplicate_count = len(sharded_logical_ids) - len(sharded_unique_ids)
    missing_ids = sorted(set(unique_logical_ids) - set(sharded_unique_ids))
    qos_states = {
        profile.name: _profile_qos_state(
            profile,
            slurm_root=slurm_root,
            max_array_size=detected_max_array_size,
            h100_enabled=h100_enabled,
        )
        for profile in profiles
    }
    desired_workers = _desired_workers_by_profile(
        profiles=profiles,
        planned_workers=planned_overrequest,
        logical_task_count=len(shared_rows),
        resource_inventory=resource_inventory,
        qos_states=qos_states,
        h100_enabled=h100_enabled,
    )
    existing_workers = _worker_pool_counts(slurm_root, profiles, execution_attempt_id="")
    accounting_invariants: list[dict[str, Any]] = []
    for profile in profiles:
        physical = _profile_capacity_hint(profile, resource_inventory)
        running = int((existing_workers.get(profile.name) or {}).get("running") or 0)
        if physical > 0 and running > physical:
            accounting_invariants.append(
                {
                    "status": "FAILED",
                    "profile": profile.name,
                    "reported_running": running,
                    "physical_capacity": physical,
                    "failure_reason": "reported_running_exceeds_physical_capacity",
                }
            )
            desired_workers[profile.name] = max(int(desired_workers.get(profile.name) or 0), int((existing_workers.get(profile.name) or {}).get("active") or 0))
    group_slots = allocate_group_concurrency(
        requested_groups,
        task_counts,
        target_workers=target_workers,
        overrequest_workers=planned_overrequest,
        group_weights=group_weights,
    )

    worker_rows: list[dict[str, Any]] = []
    profile_reports: list[dict[str, Any]] = []
    primary_teacher_profile = next((profile.name for profile in profiles if "T4" in f"{profile.name} {profile.gres} {profile.partition}".upper()), profiles[0].name if profiles else "")
    desired_teacher_h100_workers = sum(
        int(desired_workers.get(profile.name) or 0)
        for profile in profiles
        if "H100" in f"{profile.name} {profile.partition} {profile.gres}".upper()
    )
    for profile in profiles:
        qos_state = qos_states.get(profile.name) or {}
        feasible = _profile_feasible_slots(profile, resource_inventory)
        effective_limit = int(qos_state.get("effective_limit") or 0)
        active_workers = int((existing_workers.get(profile.name) or {}).get("active") or 0)
        pending_workers = int((existing_workers.get(profile.name) or {}).get("pending") or 0)
        running_workers = int((existing_workers.get(profile.name) or {}).get("running") or 0)
        desired = int(desired_workers.get(profile.name) or 0)
        deficit = max(0, desired - active_workers)
        profile_report = {
            "profile": profile.name,
            "partition": profile.partition,
            "resource_class": profile.resource_class,
            "known_good_size": qos_state.get("known_good_size"),
            "known_bad_size": qos_state.get("known_bad_size"),
            "effective_shard_size": effective_limit,
            "requested_mem": profile.mem,
            "requested_cpus": profile.cpus_per_task,
            "time_limit": profile.time_limit,
            "feasible_slots": feasible,
            "desired_workers": desired,
            "running_workers": running_workers,
            "pending_workers": pending_workers,
            "active_workers": active_workers,
            "invalid_pending_workers": int((existing_workers.get(profile.name) or {}).get("invalid_pending") or 0),
            "new_worker_deficit": deficit,
            "last_backpressure": qos_state.get("last_backpressure") or {},
            "source": qos_state.get("source") or "",
        }
        profile_reports.append(profile_report)
        if effective_limit <= 0 or deficit <= 0:
            continue
        shard_index = 0
        while deficit > 0:
            worker_count = min(deficit, effective_limit)
            shard_id = f"worker_shard_{shard_index:03d}"
            sbatch_for_shard = dynamic_root / f"full373_{profile.name}_{shard_id}_queue_worker.sbatch"
            _render_queue_worker_sbatch(
                sbatch_for_shard,
                profile=profile,
                task_manifest=shared_manifest,
                output_root=output_root,
                state_root=state_root,
                run_id=safe_run_id,
                submission_id=safe_submission_id or "default",
                worker_shard_id=shard_id,
                python_executable=python_executable,
            )
            local_end = worker_count - 1
            row = {
                "run_id": safe_run_id,
                "execution_attempt_id": safe_execution_attempt_id,
                "worker_generation": safe_execution_attempt_id,
                "submission_id": safe_submission_id,
                "logical_task_id": f"{safe_run_id}:{safe_submission_id}:{CANDIDATE_TASK_V1}:full373:{profile.name}:{shard_id}",
                "execution_schema_version": CANDIDATE_TASK_V1,
                "manifest_schema_version": DEFAULT_MANIFEST_SCHEMA_VERSION,
                "logical_task_namespace": DEFAULT_LOGICAL_TASK_NAMESPACE,
                "model_group": "full373",
                "group": "full373",
                "profile": profile.name,
                "shard_id": shard_id,
                "shard_index": str(shard_index),
                "partition": profile.partition,
                "gres": profile.gres,
                "cpus_per_task": profile.cpus_per_task,
                "mem": profile.mem,
                "time_limit": profile.time_limit,
                "task_count": worker_count,
                "array_concurrency": worker_count,
                "task_manifest": str(shared_manifest),
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
                "array_spec": f"0-{local_end}%{worker_count}",
                "local_start": 0,
                "local_end": local_end,
                "comment": slurm_comment(
                    run_id=safe_run_id,
                    submission_id=safe_submission_id or "default",
                    group="full373",
                    profile=profile.name,
                    execution_schema_version=CANDIDATE_TASK_V1,
                    shard_id=shard_id,
                ),
                "formal_root": str(output_root),
                "state_root": str(state_root),
                "git_commit": git_commit,
                "scheduler_status": "PLANNED",
                "worker_mode": "shared_ready_queue_consumer",
                "resource_class": profile.resource_class,
            }
            row["array_plan_validation"] = _validate_worker_array_plan(
                row,
                max_array_size=detected_max_array_size,
                logical_task_count=len(shared_rows),
            )
            worker_rows.append(row)
            deficit -= worker_count
            shard_index += 1

    all_array_specs_valid = all(
        str(row.get("array_plan_validation", {}).get("status") or "") == "READY"
        for row in worker_rows
    )
    sharding_audit = {
        "status": "READY" if duplicate_count == 0 and not missing_ids and all_array_specs_valid else "INVALID_ARRAY_PLAN",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "max_array_size": detected_max_array_size,
        "max_array_size_source": max_array_size_info.get("source"),
        "profile_count": len(profiles),
        "shard_count": len(worker_rows),
        "logical_task_count": len(all_logical_ids),
        "sharded_task_count": len(sharded_logical_ids),
        "unique_logical_task_ids": len(sharded_unique_ids),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing_ids),
        "missing_logical_task_ids": missing_ids[:50],
        "first_shard": worker_rows[0] if worker_rows else {},
        "last_shard": worker_rows[-1] if worker_rows else {},
        "all_array_specs_valid": all_array_specs_valid,
        "profiles": profile_reports,
        "accounting_invariants": accounting_invariants,
        "jobs": worker_rows,
        "legacy_active_jobs_adopted": 0,
        "compressed_job_ids_as_queryable": 0,
        "task_ownership": "shared_queue",
        "profile_binding": False,
        "primary_teacher_profile": primary_teacher_profile,
        "allow_h100_teacher_overflow": bool(h100_policy.get("allow_h100_teacher_overflow")),
        "labelcritic_required": bool(h100_policy.get("labelcritic_required")),
        "labelcritic_job_found": bool(h100_policy.get("labelcritic_job_found")),
        "labelcritic_job_id": h100_policy.get("labelcritic_job_id", ""),
        "labelcritic_job_state": h100_policy.get("labelcritic_job_state", "UNKNOWN"),
        "labelcritic_h100_reserved": bool(h100_policy.get("labelcritic_h100_reserved")),
        "effective_teacher_h100_enabled": h100_enabled,
        "teacher_h100_deferred_for_labelcritic": bool(h100_policy.get("teacher_h100_deferred_for_labelcritic")),
        "desired_teacher_h100_workers": desired_teacher_h100_workers,
        "teacher_h100_policy": h100_policy,
    }
    write_json(slurm_root / "sharding_audit.json", sharding_audit)
    if sharding_audit["status"] != "READY":
        plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
        write_json(
            plan_path,
            {
                "status": "INVALID_ARRAY_PLAN",
                "scheduler_status": "FATAL",
                "scheduler_mode": "shared_queue_worker_pool",
                "submission_id": safe_submission_id,
                "execution_attempt_id": safe_execution_attempt_id,
                "run_id": safe_run_id,
                "sharding_audit": sharding_audit,
                "jobs": worker_rows,
            },
        )
        raise RuntimeError("INVALID_ARRAY_PLAN: shard plan failed validation; no jobs were submitted")

    jobs: list[dict[str, Any]] = []
    backpressured: list[dict[str, Any]] = []
    fatal_failures: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    active_logical_keys = existing_active_logical_keys(
        slurm_root,
        execution_attempt_id="",
        job_state_fn=_slurm_worker_units,
    )

    if not dry_run:
        preflight_failures = []
        for row in worker_rows:
            array_validation = row.get("array_plan_validation") or {}
            if array_validation.get("status") != "READY":
                row["preflight_status"] = "INVALID_ARRAY_PLAN"
                row["preflight_stderr"] = ",".join(array_validation.get("errors") or [])
                preflight_failures.append({"job": row, "preflight": array_validation})
                continue
            report = _preflight_worker_array(row, run_sbatch_test_only=run_sbatch_test_only)
            row["preflight_status"] = report["status"]
            row["preflight_stderr"] = report.get("sbatch_test_only_stderr") or report.get("bash_n_stderr") or ""
            row["preflight"] = report
            if report["status"] == "BACKPRESSURED":
                row["scheduler_status"] = "BACKPRESSURED"
                row["failure_reason"] = report.get("failure_reason") or "QOSMaxSubmitJobPerUserLimit"
                backpressured.append(dict(row))
                _update_qos_cache(
                    slurm_root,
                    profile=next(profile for profile in profiles if profile.name == row["profile"]),
                    attempted_size=int(row["task_count"]),
                    backpressured=True,
                    source="sbatch_test_only",
                    reason=str(row.get("failure_reason") or ""),
                )
                continue
            if report["status"] != "READY":
                preflight_failures.append({"job": row, "preflight": report})
        if preflight_failures:
            plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
            write_json(
                plan_path,
                {
                    "status": "PREFLIGHT_FAILED",
                    "scheduler_mode": "shared_queue_worker_pool",
                    "planned_target_workers": int(target_workers),
                    "planned_overrequest_workers": planned_overrequest,
                    "submission_id": safe_submission_id,
                    "execution_attempt_id": safe_execution_attempt_id,
                    "resource_inventory": resource_inventory,
                    "worker_sizing": worker_sizing,
                    "sharding_audit": sharding_audit,
                    "failures": preflight_failures,
                    "jobs": worker_rows,
                    "backpressured_jobs": backpressured,
                },
            )
            raise RuntimeError(f"Dynamic GPU sbatch preflight failed for {len(preflight_failures)} shard(s); no jobs were submitted")

    csv_fields = SUBMITTED_JOB_FIELDS
    submit_budget = max(1, per_round_limit)
    for row in worker_rows:
        if dry_run:
            continue
        if str(row.get("preflight_status") or "") == "BACKPRESSURED":
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
                    "execution_attempt_id": safe_execution_attempt_id,
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
            row["scheduler_status"] = "READY_WORKER_QUEUE"
            continue
        task_count = int(row["task_count"])
        comment = str(row["comment"])
        array_spec = str(row["array_spec"])
        command = [
            "sbatch",
            "--parsable",
            "--comment", comment,
            f"--array={array_spec}",
            str(row["sbatch_file"]),
        ]
        proc = _run_sbatch_submit(command)
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
            "command": command,
        }
        append_submission_attempt(slurm_root, attempt)
        if proc.returncode != 0:
            classification = classify_sbatch_failure(proc.stderr or proc.stdout)
            row["failure_reason"] = classification["reason"]
            row["scheduler_status"] = "BACKPRESSURED" if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE" else "FATAL"
            if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
                backpressured.append(dict(row))
                _update_qos_cache(
                    slurm_root,
                    profile=next(profile for profile in profiles if profile.name == row["profile"]),
                    attempted_size=task_count,
                    backpressured=True,
                    source="sbatch_submit",
                    reason=str(classification["reason"] or ""),
                )
                continue
            fatal_failures.append(dict(row))
            continue
        row["status"] = "SUBMITTED"
        row["scheduler_status"] = "ACTIVE"
        row["slurm_state"] = "PENDING"
        persist_submitted_job(slurm_root, row)
        _update_qos_cache(
            slurm_root,
            profile=next(profile for profile in profiles if profile.name == row["profile"]),
            attempted_size=task_count,
            success=True,
            source="sbatch_submit",
        )
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
                "execution_attempt_id": safe_execution_attempt_id,
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
                "worker_mode": "shared_ready_queue_consumer",
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
                "execution_attempt_id": safe_execution_attempt_id,
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
        csv_rows = worker_rows
        if append_submitted_jobs and jobs_csv.exists():
            csv_rows = read_csv_rows(jobs_csv) + worker_rows
        write_csv(jobs_csv, csv_rows, csv_fields)

    scheduler_status = "ACTIVE"
    status = "DRY_RUN" if dry_run else "SUBMITTED"
    if not dry_run and backpressured:
        scheduler_status = "BACKPRESSURED"
        status = "PARTIALLY_SUBMITTED" if jobs else "WAITING_FOR_SUBMISSION_CAPACITY"
    elif not dry_run and any(str(row.get("submission_status") or "") == "deferred" for row in worker_rows):
        scheduler_status = "ACTIVE" if jobs else "READY_WORKER_QUEUE"
        status = "PARTIALLY_SUBMITTED" if jobs else "READY_WORKER_QUEUE"

    plan = {
        "status": status,
        "scheduler_status": scheduler_status,
        "scheduler_mode": "shared_queue_worker_pool",
        "created_at": utc_now(),
        "submission_id": safe_submission_id,
        "execution_attempt_id": safe_execution_attempt_id,
        "worker_generation": safe_execution_attempt_id,
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
        "total_task_count": len(shared_rows),
        "candidate_seed": candidate_seed,
        "candidate_seed_recovery": seed_recovery,
        "stale_claim_recovery": stale_claim_recovery,
        "totalsegmentator_executable_preflight": totalseg_preflight,
        "failed_candidate_repair": failed_candidate_repair,
        "logical_task_count": len(all_logical_ids),
        "sharded_task_count": len(sharded_logical_ids),
        "unique_logical_task_count": len(sharded_unique_ids),
        "duplicate_logical_task_count": duplicate_count,
        "missing_logical_task_count": len(missing_ids),
        "total_array_concurrency": sum(int(row["array_concurrency"]) for row in worker_rows),
        "group_concurrency": group_slots,
        "profile_specs": [{**profile.__dict__, "resource_class": profile.resource_class} for profile in profiles],
        "task_ownership": "shared_queue",
        "profile_binding": False,
        "primary_teacher_profile": primary_teacher_profile,
        "allow_h100_teacher_overflow": bool(h100_policy.get("allow_h100_teacher_overflow")),
        "labelcritic_required": bool(h100_policy.get("labelcritic_required")),
        "labelcritic_job_found": bool(h100_policy.get("labelcritic_job_found")),
        "labelcritic_job_id": h100_policy.get("labelcritic_job_id", ""),
        "labelcritic_job_state": h100_policy.get("labelcritic_job_state", "UNKNOWN"),
        "labelcritic_h100_reserved": bool(h100_policy.get("labelcritic_h100_reserved")),
        "effective_teacher_h100_enabled": h100_enabled,
        "teacher_h100_deferred_for_labelcritic": bool(h100_policy.get("teacher_h100_deferred_for_labelcritic")),
        "desired_teacher_h100_workers": desired_teacher_h100_workers,
        "t4_only_reachability_logical_task_count": len(all_logical_ids),
        "resource_policy": {
            "teacher_resource_requirement": "GPU_INFERENCE_COMPATIBLE",
            "profile_priority_order": "T4/generic/immediately-compatible GPUs first; A100/H100 are opportunistic overflow, not scientific ownership",
            "allow_h100_teacher_overflow": bool(h100_policy.get("allow_h100_teacher_overflow")),
            "labelcritic_required": bool(h100_policy.get("labelcritic_required")),
            "labelcritic_job_found": bool(h100_policy.get("labelcritic_job_found")),
            "labelcritic_job_id": h100_policy.get("labelcritic_job_id", ""),
            "labelcritic_job_state": h100_policy.get("labelcritic_job_state", "UNKNOWN"),
            "labelcritic_h100_reservation": bool(h100_policy.get("labelcritic_h100_reserved")),
            "effective_teacher_h100_enabled": h100_enabled,
            "teacher_h100_deferred_for_labelcritic": bool(h100_policy.get("teacher_h100_deferred_for_labelcritic")),
            "desired_teacher_h100_workers": desired_teacher_h100_workers,
            "reservation_reason": h100_policy.get("reservation_reason", ""),
            "source": h100_policy.get("source", ""),
            "h100_overflow_policy": "TASK2_ALLOW_H100_TEACHER_OVERFLOW only grants admin permission; effective Teacher H100 use also requires LabelCritic H100 not reserved",
            "qos_backpressure_classification": "QOSMaxSubmitJobPerUserLimit classified as BACKPRESSURE",
        },
        "teacher_h100_policy": h100_policy,
        "resource_inventory": resource_inventory,
        "worker_sizing": worker_sizing,
        "worker_pool": {
            "desired_workers_by_profile": desired_workers,
            "existing_workers_by_profile": existing_workers,
            "profile_reports": profile_reports,
            "accounting_invariants": accounting_invariants,
        },
        "qos": {
            "per_profile": qos_states,
        },
        "sharding_audit": sharding_audit,
        "jobs": worker_rows,
        "submitted_jobs": jobs,
        "reused_jobs": reused,
        "backpressured_jobs": backpressured,
        "deferred_jobs": [row for row in worker_rows if str(row.get("submission_status") or "") == "deferred"],
        "submitted_job_count": len([job for job in jobs if str(job.get("submission_status")) == "submitted"]),
        "reused_job_count": len(reused),
        "backpressured_count": len(backpressured),
        "dependency_policy": {
            "parallel": [
                "generic GPU workers consume any compatible READY candidate from the shared queue",
                "ShapeKit postprocessing runs inside each worker after teacher inference",
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
                "teacher shared-queue worker arrays",
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


def _compact_candidate_summary(output_root: Path) -> dict[str, Any]:
    telemetry = _read_json(output_root / "queues" / "telemetry.json", {})
    marker = _read_json(output_root / "queues" / "candidate_seed_complete.json", {})
    teacher = telemetry.get("teacher_candidate") if isinstance(telemetry, dict) else {}
    teacher = teacher if isinstance(teacher, dict) else {}
    updated_at = str(telemetry.get("updated_at") or "") if isinstance(telemetry, dict) else ""
    status = "AVAILABLE" if teacher else "UNAVAILABLE"
    max_age = int(os.getenv("TASK2_COMPACT_TELEMETRY_MAX_AGE_SEC", "900") or 900)
    if teacher and updated_at:
        try:
            from datetime import datetime, timezone

            parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc).timestamp() - parsed.timestamp() > max_age:
                status = "STALE"
        except Exception:
            status = "UNKNOWN"
    def _maybe_int(value: Any) -> int | None:
        if status != "AVAILABLE":
            return None
        return int(value or 0)

    return {
        "status": status,
        "ready": _maybe_int(teacher.get("ready_candidates")),
        "running": _maybe_int(teacher.get("running")),
        "retry": _maybe_int(teacher.get("retry")),
        "terminal": _maybe_int(teacher.get("terminal")),
        "success": _maybe_int(teacher.get("success")),
        "total": int(teacher.get("total") or marker.get("logical_task_count") or 0),
        "source": str(output_root / "queues" / "telemetry.json") if telemetry else "",
        "updated_at": updated_at,
        "seed_marker": marker,
    }


def _validate_existing_seed_marker(output_root: Path, *, expected_scientific_run_id: str = "", expected_count: int | None = None) -> dict[str, Any]:
    marker_path = output_root / "queues" / "candidate_seed_complete.json"
    marker = _read_json(marker_path, {})
    if not marker:
        return {"status": "MISSING", "path": str(marker_path)}
    failures = []
    if marker.get("schema_version") != "candidate_seed_complete_v1":
        failures.append("schema_version")
    if marker.get("execution_schema_version") != CANDIDATE_TASK_V1:
        failures.append("execution_schema_version")
    if expected_scientific_run_id and marker.get("scientific_run_id") != expected_scientific_run_id:
        failures.append("scientific_run_id")
    if expected_count is not None and int(marker.get("logical_task_count") or 0) != int(expected_count):
        failures.append("logical_task_count")
    if not marker.get("manifest_sha256"):
        failures.append("manifest_sha256_missing")
    manifest_path = Path(str(marker.get("manifest_path") or output_root / "queues" / "shared_ready_candidate_manifest.csv"))
    if not manifest_path.exists():
        failures.append("manifest_missing")
    else:
        stat = manifest_path.stat()
        if marker.get("manifest_size") is not None and int(marker.get("manifest_size") or 0) != int(stat.st_size):
            failures.append("manifest_size")
        if marker.get("manifest_mtime_ns") is not None and int(marker.get("manifest_mtime_ns") or 0) != int(stat.st_mtime_ns):
            failures.append("manifest_mtime_ns")
    return {
        "status": "READY" if not failures else "MISMATCH",
        "path": str(marker_path),
        "failures": failures,
        "marker": marker,
        "manifest_path": str(manifest_path),
    }


def _profiles_from_existing_plan(plan: dict[str, Any]) -> list[GpuSubmitProfile]:
    profiles = []
    for row in plan.get("profile_specs") or []:
        if isinstance(row, dict):
            profile = _profile_from_plan(row)
            if profile is not None:
                profiles.append(profile)
    return profiles


def _desired_workers_from_existing_plan(plan: dict[str, Any]) -> dict[str, int]:
    desired = {
        str(name): int(value or 0)
        for name, value in (((plan.get("worker_pool") or {}).get("desired_workers_by_profile") or {}) if isinstance(plan.get("worker_pool"), dict) else {}).items()
    }
    for row in ((plan.get("worker_pool") or {}).get("profile_reports") or []) if isinstance(plan.get("worker_pool"), dict) else []:
        if isinstance(row, dict) and row.get("profile"):
            desired.setdefault(str(row["profile"]), int(row.get("desired_workers") or 0))
    return desired


def fast_replenish_existing_worker_pool(
    *,
    output_root: Path,
    state_root: Path,
    submission_id: str,
    execution_attempt_id: str,
    run_id: str,
    git_commit: str = "",
    expected_scientific_run_id: str = "",
    expected_candidate_count: int | None = None,
    run_sbatch_test_only: bool = False,
    labelcritic_required: Any | None = None,
    labelcritic_job_id: str | None = None,
    labelcritic_job_state: str | None = None,
    labelcritic_h100_reserved: Any | None = None,
    extra_worker_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    state_root = state_root.resolve()
    slurm_root = output_root / "slurm"
    plan_path = slurm_root / "dynamic_gpu_submission_plan.json"
    existing_plan = _read_json(plan_path, {})
    if not existing_plan:
        return {"status": "SKIPPED", "reason": "dynamic_plan_missing", "plan_path": str(plan_path)}
    seed = _validate_existing_seed_marker(
        output_root,
        expected_scientific_run_id=expected_scientific_run_id,
        expected_count=expected_candidate_count,
    )
    if seed["status"] != "READY":
        return {"status": "SKIPPED", "reason": "candidate_seed_marker_not_ready", "seed": seed}
    candidate_summary = _compact_candidate_summary(output_root)
    if candidate_summary.get("status") == "AVAILABLE" and int(candidate_summary.get("ready") or 0) <= 0:
        return {"status": "NO_READY_CANDIDATES", "candidate_summary": candidate_summary, "seed": seed}
    profiles = _profiles_from_existing_plan(existing_plan)
    desired = _desired_workers_from_existing_plan(existing_plan)
    if not profiles or not desired:
        return {"status": "SKIPPED", "reason": "worker_plan_missing", "candidate_summary": candidate_summary, "seed": seed}
    h100_policy = resolve_teacher_h100_policy(
        state_root=state_root,
        labelcritic_required=labelcritic_required,
        labelcritic_job_id=labelcritic_job_id,
        labelcritic_job_state=labelcritic_job_state,
        labelcritic_h100_reserved=labelcritic_h100_reserved,
    )
    if not h100_policy.get("effective_teacher_h100_enabled"):
        for profile in profiles:
            if "H100" in f"{profile.name} {profile.partition} {profile.gres}".upper():
                desired[profile.name] = 0
    profile_names = {profile.name for profile in profiles}
    if extra_worker_rows:
        accounting = current_worker_accounting_from_rows(
            [*load_submitted_jobs(slurm_root), *extra_worker_rows],
            profiles=profile_names,
            execution_attempt_id="",
            job_state_fn=_slurm_worker_units,
        )
    else:
        accounting = reconcile_submitted_worker_accounting(
            slurm_root,
            profiles=profile_names,
            execution_attempt_id="",
            job_state_fn=_slurm_worker_units,
        )
    existing_workers = accounting.get("counts") or {}
    profile_reports_by_name = {
        str(row.get("profile")): row
        for row in ((existing_plan.get("worker_pool") or {}).get("profile_reports") or [])
        if isinstance(row, dict) and row.get("profile")
    }
    deficits = {}
    for profile in profiles:
        active = int((existing_workers.get(profile.name) or {}).get("active") or 0)
        deficits[profile.name] = max(0, int(desired.get(profile.name) or 0) - active)
    if not any(deficits.values()):
        return {
            "status": "NO_DEFICIT",
            "candidate_summary": candidate_summary,
            "seed": seed,
            "desired_workers_by_profile": desired,
            "existing_workers_by_profile": existing_workers,
            "deficit_by_profile": deficits,
            "teacher_h100_policy": h100_policy,
        }
    task_manifest = output_root / "queues" / "shared_ready_candidate_manifest.csv"
    if not task_manifest.exists():
        return {"status": "SKIPPED", "reason": "shared_ready_candidate_manifest_missing", "task_manifest": str(task_manifest)}
    safe_submission_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(submission_id or "fast_replenish").strip()).strip("_")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(run_id or os.getenv("ROUND1_RUN_ID") or "round1").strip()).strip("_")
    safe_execution_attempt_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(execution_attempt_id or os.getenv("TASK2_EXECUTION_ATTEMPT_ID") or safe_submission_id).strip()).strip("_")
    dynamic_root = slurm_root / "dynamic" / safe_submission_id
    dynamic_root.mkdir(parents=True, exist_ok=True)
    max_array_size = int(existing_plan.get("max_array_size") or DEFAULT_SLURM_MAX_ARRAY_SIZE_FALLBACK)
    submitted = []
    backpressured = []
    failed = []
    planned_rows = []
    for profile in profiles:
        deficit = int(deficits.get(profile.name) or 0)
        if deficit <= 0:
            continue
        report = profile_reports_by_name.get(profile.name) or {}
        effective_limit = int(report.get("effective_shard_size") or max_array_size or deficit)
        effective_limit = max(1, min(effective_limit, max_array_size, deficit))
        remaining = deficit
        shard_index = 0
        while remaining > 0:
            worker_count = min(remaining, effective_limit)
            shard_id = f"fast_replenish_{int(time.time())}_{os.getpid()}_{profile.name}_{shard_index:03d}"
            sbatch_for_shard = dynamic_root / f"full373_{profile.name}_{shard_id}_queue_worker.sbatch"
            _render_queue_worker_sbatch(
                sbatch_for_shard,
                profile=profile,
                task_manifest=task_manifest,
                output_root=output_root,
                state_root=state_root,
                run_id=safe_run_id,
                submission_id=safe_submission_id,
                worker_shard_id=shard_id,
                python_executable=sys.executable,
            )
            local_end = worker_count - 1
            row = {
                "run_id": safe_run_id,
                "execution_attempt_id": safe_execution_attempt_id,
                "worker_generation": safe_execution_attempt_id,
                "submission_id": safe_submission_id,
                "logical_task_id": f"{safe_run_id}:{safe_submission_id}:{CANDIDATE_TASK_V1}:full373:{profile.name}:{shard_id}",
                "execution_schema_version": CANDIDATE_TASK_V1,
                "manifest_schema_version": DEFAULT_MANIFEST_SCHEMA_VERSION,
                "logical_task_namespace": DEFAULT_LOGICAL_TASK_NAMESPACE,
                "model_group": "full373",
                "group": "full373",
                "profile": profile.name,
                "shard_id": shard_id,
                "shard_index": str(shard_index),
                "partition": profile.partition,
                "gres": profile.gres,
                "cpus_per_task": profile.cpus_per_task,
                "mem": profile.mem,
                "time_limit": profile.time_limit,
                "task_count": worker_count,
                "array_concurrency": worker_count,
                "task_manifest": str(task_manifest),
                "sbatch_file": str(sbatch_for_shard),
                "job_id": "",
                "array_job_id": "",
                "array_task_id": "",
                "display_id": "",
                "submission_status": "pending",
                "preflight_status": "",
                "preflight_stderr": "",
                "stderr": "",
                "stdout": "",
                "array_range": f"0-{local_end}",
                "array_spec": f"0-{local_end}%{worker_count}",
                "local_start": 0,
                "local_end": local_end,
                "comment": slurm_comment(
                    run_id=safe_run_id,
                    submission_id=safe_submission_id,
                    group="full373",
                    profile=profile.name,
                    execution_schema_version=CANDIDATE_TASK_V1,
                    shard_id=shard_id,
                ),
                "formal_root": str(output_root),
                "state_root": str(state_root),
                "git_commit": git_commit,
                "scheduler_status": "PLANNED",
                "worker_mode": "shared_ready_queue_consumer",
                "resource_class": profile.resource_class,
            }
            planned_rows.append(dict(row))
            preflight = _preflight_worker_array(row, run_sbatch_test_only=run_sbatch_test_only)
            row["preflight_status"] = preflight["status"]
            row["preflight_stderr"] = preflight.get("sbatch_test_only_stderr") or preflight.get("bash_n_stderr") or ""
            if preflight["status"] != "READY":
                row["submission_status"] = "failed"
                row["scheduler_status"] = "BACKPRESSURED" if preflight["status"] == "BACKPRESSURED" else "FATAL"
                row["failure_reason"] = preflight.get("failure_reason") or row["preflight_stderr"]
                (backpressured if preflight["status"] == "BACKPRESSURED" else failed).append(dict(row))
                remaining -= worker_count
                shard_index += 1
                continue
            command = ["sbatch", "--parsable", "--comment", row["comment"], f"--array={row['array_spec']}", str(sbatch_for_shard)]
            proc = _run_sbatch_submit(command)
            parsed_job = parse_sbatch_job_id(proc.stdout)
            row["job_id"] = parsed_job["job_id"]
            row["array_job_id"] = parsed_job["array_job_id"]
            row["array_task_id"] = parsed_job["array_task_id"]
            row["display_id"] = parsed_job["display_id"]
            row["stderr"] = proc.stderr.strip()
            row["stdout"] = proc.stdout.strip()
            row["submitted_at"] = utc_now()
            append_submission_attempt(slurm_root, {**row, "return_code": int(proc.returncode), "command": command})
            if proc.returncode != 0:
                classification = classify_sbatch_failure(proc.stderr or proc.stdout)
                row["submission_status"] = "failed"
                row["scheduler_status"] = "BACKPRESSURED" if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE" else "FATAL"
                row["failure_reason"] = classification["reason"]
                (backpressured if row["scheduler_status"] == "BACKPRESSURED" else failed).append(dict(row))
                remaining -= worker_count
                shard_index += 1
                continue
            row["status"] = "SUBMITTED"
            row["submission_status"] = "submitted"
            row["scheduler_status"] = "ACTIVE"
            row["slurm_state"] = "PENDING"
            persist_submitted_job(slurm_root, row)
            record_job_lifecycle(
                state_root,
                {
                    "status": "SUBMITTED",
                    "scheduler_state": "ACTIVE",
                    "job_id": row["job_id"],
                    "logical_task_id": row["logical_task_id"],
                    "submission_id": safe_submission_id,
                    "execution_attempt_id": safe_execution_attempt_id,
                    "execution_schema_version": CANDIDATE_TASK_V1,
                    "group": row["model_group"],
                    "profile": row["profile"],
                    "shard_id": row["shard_id"],
                    "partition": row["partition"],
                    "gres": row["gres"],
                    "task_manifest": row["task_manifest"],
                    "formal_root": str(output_root),
                    "git_commit": git_commit,
                    "worker_mode": "shared_ready_queue_consumer",
                    "attempt": "fast_teacher_replenishment",
                },
            )
            submitted.append(dict(row))
            remaining -= worker_count
            shard_index += 1
    status = "SUBMITTED" if submitted else ("WAITING_FOR_SUBMISSION_CAPACITY" if backpressured else "FAILED" if failed else "NO_DEFICIT")
    result = {
        "status": status,
        "scheduler_status": "ACTIVE" if submitted else ("BACKPRESSURED" if backpressured else "FATAL" if failed else "IDLE"),
        "scheduler_mode": "fast_teacher_replenishment",
        "created_at": utc_now(),
        "submission_id": safe_submission_id,
        "execution_attempt_id": safe_execution_attempt_id,
        "run_id": safe_run_id,
        "candidate_summary": candidate_summary,
        "seed": seed,
        "desired_workers_by_profile": desired,
        "existing_workers_by_profile": existing_workers,
        "deficit_by_profile": deficits,
        "teacher_h100_policy": h100_policy,
        "planned_jobs": planned_rows,
        "submitted_jobs": submitted,
        "backpressured_jobs": backpressured,
        "failed_jobs": failed,
        "submitted_job_count": len(submitted),
    }
    write_json(slurm_root / f"fast_teacher_replenishment_{safe_submission_id}.json", result)
    return result


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
    parser.add_argument("--execution-attempt-id", default=os.getenv("TASK2_EXECUTION_ATTEMPT_ID", ""))
    parser.add_argument("--append-submitted-jobs", action="store_true")
    parser.add_argument("--run-id", default=os.getenv("ROUND1_RUN_ID", ""))
    parser.add_argument("--git-commit", default=os.getenv("EXPECTED_GIT_COMMIT", ""))
    parser.add_argument("--allow-h100-teacher-overflow", default=None)
    parser.add_argument("--labelcritic-required", default=None)
    parser.add_argument("--labelcritic-job-id", default="")
    parser.add_argument("--labelcritic-state", default="")
    parser.add_argument("--labelcritic-h100-reserved", default=None)
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
        execution_attempt_id=args.execution_attempt_id,
        append_submitted_jobs=bool(args.append_submitted_jobs),
        run_id=args.run_id,
        git_commit=args.git_commit,
        allow_h100_teacher_overflow=args.allow_h100_teacher_overflow,
        labelcritic_required=args.labelcritic_required,
        labelcritic_job_id=args.labelcritic_job_id,
        labelcritic_job_state=args.labelcritic_state,
        labelcritic_h100_reserved=args.labelcritic_h100_reserved,
    )
    print(
        json.dumps(
            {
                k: plan[k]
                for k in (
                    "status",
                    "scheduler_status",
                    "planned_target_workers",
                    "planned_overrequest_workers",
                    "total_array_concurrency",
                    "allow_h100_teacher_overflow",
                    "labelcritic_job_state",
                    "labelcritic_h100_reserved",
                    "effective_teacher_h100_enabled",
                    "teacher_h100_deferred_for_labelcritic",
                    "desired_teacher_h100_workers",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
