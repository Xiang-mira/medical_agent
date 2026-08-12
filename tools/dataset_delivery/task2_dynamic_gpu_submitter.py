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


DEFAULT_GROUP_WEIGHTS = {
    "cads": 0.45,
    "atm": 0.15,
    "airrc": 0.20,
    "unest": 0.20,
}
DEFAULT_PROFILE_SPECS = "generic_gpu|gpu|gpu:1|8|64G|06:00:00"
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
    if upper in {"GPU", "GENERIC"}:
        return 1
    if "A100" in upper:
        return 2
    if "H100" in upper:
        return 3
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
    profiles.sort(key=lambda profile: _gpu_type_rank(profile.gres))
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


def _rewrite_sbatch(
    source: Path,
    destination: Path,
    *,
    group: str,
    profile: GpuSubmitProfile,
    task_manifest: Path,
    output_root: Path,
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    lines = _replace_directive(lines, "#SBATCH --job-name=", f"task2_{group}_{profile.name}")
    lines = _replace_directive(lines, "#SBATCH --partition=", profile.partition)
    lines = _replace_directive(lines, "#SBATCH --gres=", profile.gres)
    lines = _replace_directive(lines, "#SBATCH --cpus-per-task=", str(profile.cpus_per_task))
    lines = _replace_directive(lines, "#SBATCH --mem=", profile.mem)
    lines = _replace_directive(lines, "#SBATCH --time=", profile.time_limit)
    lines = _replace_directive(lines, "#SBATCH --output=", str(output_root / "slurm" / f"{group}_{profile.name}_%A_%a.out"))
    lines = _replace_directive(lines, "#SBATCH --error=", str(output_root / "slurm" / f"{group}_{profile.name}_%A_%a.err"))
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--task-manifest "):
            indent = line[: len(line) - len(line.lstrip())]
            suffix = " \\" if stripped.endswith("\\") else ""
            rewritten.append(f"{indent}--task-manifest {task_manifest}{suffix}")
        else:
            rewritten.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    destination.chmod(0o755)


def _write_shard_manifest(source_manifest: Path, destination: Path, rows: list[dict[str, str]]) -> None:
    source_fields = read_csv_fieldnames(source_manifest)
    fieldnames = ["task_index", *[field for field in source_fields if field != "task_index"], "source_task_index"]
    shard_rows: list[dict[str, Any]] = []
    for new_index, row in enumerate(rows):
        updated = dict(row)
        updated["source_task_index"] = str(row.get("task_index") or new_index)
        updated["task_index"] = new_index
        shard_rows.append(updated)
    write_csv(destination, shard_rows, fieldnames)


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
        sbatch_proc = subprocess.run(
            ["sbatch", "--test-only", sbatch_file],
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
    dynamic_root = slurm_root / "dynamic" / safe_submission_id if safe_submission_id else slurm_root / "dynamic"
    dynamic_root.mkdir(parents=True, exist_ok=True)

    shard_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
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
            manifest_for_profile = task_manifest
            if len(active_profiles) > 1:
                manifest_for_profile = dynamic_root / f"{group}_{profile.name}_task_manifest.csv"
                _write_shard_manifest(task_manifest, manifest_for_profile, rows_for_profile)
            sbatch_for_profile = dynamic_root / f"{group}_{profile.name}_task2_array.sbatch"
            _rewrite_sbatch(
                source_sbatch,
                sbatch_for_profile,
                group=group,
                profile=profile,
                task_manifest=manifest_for_profile,
                output_root=output_root,
            )
            concurrency = max(1, min(int(profile_slots.get(profile.name) or 1), len(rows_for_profile)))
            row = {
                "model_group": group,
                "profile": profile.name,
                "partition": profile.partition,
                "gres": profile.gres,
                "cpus_per_task": profile.cpus_per_task,
                "mem": profile.mem,
                "time_limit": profile.time_limit,
                "task_count": len(rows_for_profile),
                "array_concurrency": concurrency,
                "task_manifest": str(manifest_for_profile),
                "sbatch_file": str(sbatch_for_profile),
                "job_id": "",
                "submission_status": "dry_run" if dry_run else "pending",
                "preflight_status": "",
                "preflight_stderr": "",
                "stderr": "",
            }
            shard_rows.append(row)

    if not dry_run:
        preflight_failures = []
        for row in shard_rows:
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
                    "failures": preflight_failures,
                    "jobs": shard_rows,
                },
            )
            raise RuntimeError(f"Dynamic GPU sbatch preflight failed for {len(preflight_failures)} shard(s); no jobs were submitted")

    for row in shard_rows:
        if dry_run:
            continue
        task_count = int(row["task_count"])
        concurrency = int(row["array_concurrency"])
        proc = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--array=0-{task_count - 1}%{concurrency}",
                str(row["sbatch_file"]),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        row["job_id"] = proc.stdout.strip()
        row["submission_status"] = "submitted" if proc.returncode == 0 else "failed"
        row["stderr"] = proc.stderr.strip()
        if proc.returncode != 0:
            plan_path = output_root / "slurm" / (f"dynamic_gpu_submission_plan_{safe_submission_id}.json" if safe_submission_id else "dynamic_gpu_submission_plan.json")
            write_json(plan_path, {"status": "SUBMISSION_FAILED", "submission_id": safe_submission_id, "jobs": jobs, "failed_job": row})
            raise RuntimeError(f"sbatch failed for {row['model_group']}:{row['profile']}: {proc.stderr.strip()}")
        jobs.append(dict(row))

    csv_fields = [
        "model_group", "profile", "partition", "gres", "cpus_per_task", "mem", "time_limit",
        "job_id", "task_count", "array_concurrency", "task_manifest", "sbatch_file",
        "submission_status", "preflight_status", "preflight_stderr", "stderr",
    ]
    jobs_csv = slurm_root / "submitted_jobs.csv"
    csv_rows = shard_rows
    if append_submitted_jobs and jobs_csv.exists():
        csv_rows = read_csv_rows(jobs_csv) + shard_rows
    write_csv(jobs_csv, csv_rows, csv_fields)

    plan = {
        "status": "DRY_RUN" if dry_run else "SUBMITTED",
        "scheduler_mode": "dynamic_gpu_overrequest",
        "created_at": utc_now(),
        "submission_id": safe_submission_id,
        "summary_path": str(summary_path),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "planned_target_workers": int(target_workers),
        "planned_overrequest_workers": planned_overrequest,
        "total_task_count": sum(int(row["task_count"]) for row in shard_rows),
        "total_array_concurrency": sum(int(row["array_concurrency"]) for row in shard_rows),
        "group_concurrency": group_slots,
        "profile_specs": [profile.__dict__ for profile in profiles],
        "resource_inventory": resource_inventory,
        "worker_sizing": worker_sizing,
        "jobs": shard_rows,
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
    )
    print(json.dumps({k: plan[k] for k in ("status", "planned_target_workers", "planned_overrequest_workers", "total_array_concurrency")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
