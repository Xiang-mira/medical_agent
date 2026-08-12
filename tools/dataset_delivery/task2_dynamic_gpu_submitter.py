#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scheduler.resource_recommender import adaptive_overrequest_count  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_fieldnames, read_csv_rows, utc_now, write_csv, write_json  # noqa: E402


DEFAULT_GROUP_WEIGHTS = {
    "cads": 0.45,
    "atm": 0.15,
    "airrc": 0.20,
    "unest": 0.20,
}
DEFAULT_PROFILE_SPECS = "generic_gpu|gpu|gpu:1|8|64G|06:00:00"


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
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "READY":
        raise RuntimeError(f"Formal Task2 preflight is not READY: {summary.get('status')}")
    profiles = parse_profile_specs(profile_specs)
    summary_groups = summary.get("groups") or {}
    requested_groups = groups or [group for group in ("cads", "atm", "airrc", "unest") if group in summary_groups]
    task_counts = {
        group: int((summary_groups.get(group) or {}).get("task_count") or 0)
        for group in requested_groups
    }
    planned_overrequest = int(overrequest_workers or adaptive_overrequest_count(target_workers))
    group_slots = allocate_group_concurrency(
        requested_groups,
        task_counts,
        target_workers=target_workers,
        overrequest_workers=planned_overrequest,
        group_weights=group_weights,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    slurm_root = output_root / "slurm"
    dynamic_root = slurm_root / "dynamic"
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
                "stderr": "",
            }
            shard_rows.append(row)

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
            write_json(output_root / "slurm" / "dynamic_gpu_submission_plan.json", {"status": "SUBMISSION_FAILED", "jobs": jobs, "failed_job": row})
            raise RuntimeError(f"sbatch failed for {row['model_group']}:{row['profile']}: {proc.stderr.strip()}")
        jobs.append(dict(row))

    csv_fields = [
        "model_group", "profile", "partition", "gres", "cpus_per_task", "mem", "time_limit",
        "job_id", "task_count", "array_concurrency", "task_manifest", "sbatch_file",
        "submission_status", "stderr",
    ]
    write_csv(slurm_root / "submitted_jobs.csv", shard_rows, csv_fields)

    plan = {
        "status": "DRY_RUN" if dry_run else "SUBMITTED",
        "scheduler_mode": "dynamic_gpu_overrequest",
        "created_at": utc_now(),
        "summary_path": str(summary_path),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "planned_target_workers": int(target_workers),
        "planned_overrequest_workers": planned_overrequest,
        "total_task_count": sum(int(row["task_count"]) for row in shard_rows),
        "total_array_concurrency": sum(int(row["array_concurrency"]) for row in shard_rows),
        "group_concurrency": group_slots,
        "profile_specs": [profile.__dict__ for profile in profiles],
        "jobs": shard_rows,
        "dependency_policy": {
            "parallel": [
                "cads/atm/airrc/unest teacher arrays are independent after manifest/preflight",
                "ShapeKit postprocessing runs inside each case/group worker after teacher inference",
                "LabelCritic candidate selection can run per case/target once candidate masks exist",
            ],
            "serial": [
                "workspace staging, Task1 rename, manifest build, and formal preflight precede GPU arrays",
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
    parser.add_argument("--target-workers", default=30, type=int)
    parser.add_argument("--overrequest-workers", default=None, type=int)
    parser.add_argument("--profile-specs", default=DEFAULT_PROFILE_SPECS)
    parser.add_argument("--groups", default="")
    parser.add_argument("--group-weights", default="")
    parser.add_argument("--dry-run", action="store_true")
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
    )
    print(json.dumps({k: plan[k] for k in ("status", "planned_target_workers", "planned_overrequest_workers", "total_array_concurrency")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
