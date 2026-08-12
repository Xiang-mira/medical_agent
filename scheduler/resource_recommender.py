from __future__ import annotations

import math
from typing import Any

from .queue_estimator import heuristic_wait
from .resource_discovery import validate_snapshot_invariants
from .utils import SchedulerError


PLAN_IDS = {"fastest-start", "fastest-completion", "balanced", "conservative", "manual"}
LABELCRITIC_72B_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct-AWQ"


def workload_estimates(train_cases: int, test_cases: int, *, epochs: int = 1, steps_per_epoch: int | None = None, batch_size: int = 2, target_count: int = 373) -> dict[str, Any]:
    train_steps = int(steps_per_epoch or max(1, math.ceil(train_cases / max(batch_size, 1)))) * max(epochs, 1)
    return {
        "teacher_train_inference": {"work_units": train_cases, "workload_source": "train_case_count", "workload_confidence": "high"},
        "teacher_test_inference": {"work_units": test_cases, "workload_source": "test_case_count", "workload_confidence": "high"},
        "student_training": {
            "work_units": train_steps,
            "workload_source": "train_cases, epochs, steps_per_epoch, batch_size, target_count",
            "workload_confidence": "medium",
            "target_count": target_count,
        },
        "student_inference": {"work_units": test_cases, "workload_source": "test_case_count", "workload_confidence": "high"},
        "labelcritic": {"work_units": train_cases * target_count, "workload_source": "case_count * target_count conservative candidate estimate", "workload_confidence": "low"},
    }


def _stage(stage: str, profile: str, partition: str, gpu_type: str, gpu_count: int, jobs: int, concurrent: int, snapshot: dict[str, Any], reason: str, risk: str) -> dict[str, Any]:
    part = snapshot.get("partitions", {}).get(partition, {})
    wait = heuristic_wait(part, gpu_count=gpu_count)
    shape_key = f"nodes_with_at_least_{gpu_count}_free_gpus" if gpu_count in {1, 2, 4} else "nodes_with_at_least_1_free_gpu"
    alloc_now = int(part.get(shape_key) or 0) > 0
    return {
        "task_stage": stage,
        "resource_profile": profile,
        "partition": partition,
        "gpu_type": gpu_type,
        "gpu_count_per_job": gpu_count,
        "number_of_jobs": jobs,
        "max_concurrent": concurrent,
        "ddp_world_size": gpu_count if stage == "student_training" and gpu_count > 1 else 1,
        "labelcritic_replicas": jobs if stage == "labelcritic" else 0,
        "currently_allocatable_shape": part.get(shape_key, 0),
        "likely_immediate": alloc_now,
        "queue_expected": not alloc_now,
        "wait_p50": wait.get("estimated_wait_p50", "unknown"),
        "wait_p90": wait.get("estimated_wait_p90", "unknown"),
        "execution_time": "estimate_pending_history",
        "total_time_p50": "unknown",
        "total_time_p90": "unknown",
        "confidence": "low",
        "recommendation_reason": reason,
        "risk": risk,
        "fallback": "ask_before_fallback",
    }


def adaptive_overrequest_count(target_available: int, *, reserve_ratio: float = 1.0 / 3.0, minimum_extra: int = 1) -> int:
    """Plan a larger schedulable pool than the desired live concurrency.

    Slurm cannot reserve "spare" GPUs for a running array, but the workflow can
    keep enough queued work to refill slots when nodes free up.  The default
    gives the operational rule used in the 103-case launch: target 30 live GPU
    workers by planning 40 queued/eligible workers.
    """
    target_available = max(1, int(target_available))
    extra = max(int(minimum_extra), math.ceil(target_available * float(reserve_ratio)))
    return target_available + extra


def select_labelcritic_72b_profile(snapshot: dict[str, Any], profiles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Select a formal Qwen2-VL-72B profile without model downgrade."""
    profiles = profiles or [
        {"profile": "labelcritic_72b_2h100_formal", "partition": "gpuh100", "gpu_type": "H100", "gpu_count": 2, "tensor_parallel_size": 2, "min_total_vram_gb": 144},
        {"profile": "labelcritic_72b_1h100_smoke", "partition": "gpuh100", "gpu_type": "H100", "gpu_count": 1, "tensor_parallel_size": 1, "min_total_vram_gb": 80, "smoke_only": True},
        {"profile": "labelcritic_72b_2a100_formal", "partition": "gpua100", "gpu_type": "A100", "gpu_count": 2, "tensor_parallel_size": 2, "min_total_vram_gb": 144},
        {"profile": "labelcritic_72b_4x48gb_formal", "partition": "gpu", "gpu_type": "48GB", "gpu_count": 4, "tensor_parallel_size": 4, "min_total_vram_gb": 192},
    ]
    parts = snapshot.get("partitions", {}) or {}
    for profile in profiles:
        part = parts.get(str(profile["partition"]), {}) or {}
        gpu_count = int(profile["gpu_count"])
        shape_key = f"nodes_with_at_least_{gpu_count}_free_gpus" if gpu_count in {1, 2, 4} else "nodes_with_at_least_1_free_gpu"
        if int(part.get(shape_key) or 0) > 0:
            return {
                "status": "READY",
                "model_id": LABELCRITIC_72B_MODEL_ID,
                "selected_profile": profile,
                "tensor_parallel_size": int(profile["tensor_parallel_size"]),
                "gpu_memory_utilization": 0.88 if gpu_count >= 2 else 0.82,
                "max_model_len": 8192 if gpu_count >= 2 else 4096,
                "visual_resolution": "formal" if gpu_count >= 2 else "smoke_low_resolution",
                "silent_downgrade": False,
            }
    return {
        "status": "insufficient_resource",
        "model_id": LABELCRITIC_72B_MODEL_ID,
        "required_profiles": profiles,
        "silent_downgrade": False,
        "reason": "No configured node shape is currently allocatable for Qwen2-VL-72B-AWQ.",
    }


def recommend_resource_plans(snapshot: dict[str, Any], train_cases: int, test_cases: int, *, gpu_budget: dict[str, int] | None = None) -> dict[str, Any]:
    invariants = validate_snapshot_invariants(snapshot)
    if invariants["status"] != "success":
        raise SchedulerError("Resource snapshot invariant check failed: " + "; ".join(invariants["errors"]))
    budget = {"max_t4_gpus": 8, "max_a100_gpus": 2, "max_h100_gpus": 8, "labelcritic_replicas": 1, "max_labelcritic_replicas": 2}
    if gpu_budget:
        budget.update({k: int(v) for k, v in gpu_budget.items() if v is not None})
    t4_conc = max(1, min(budget["max_t4_gpus"], train_cases, 8))
    plans = []
    templates = {
        "fastest-start": ("t4_single_gpu", "gpu", "T4", t4_conc, "uses the least scarce currently allocatable inference tier", "T4 throughput may be lower than A100/H100"),
        "fastest-completion": ("a100_single_gpu", "gpua100", "A100", max(1, min(budget["max_a100_gpus"], train_cases)), "minimizes estimated runtime when A100 is allocatable", "A100 availability may require queueing"),
        "balanced": ("t4_single_gpu", "gpu", "T4", t4_conc, "balances completion time, scarcity and validated single-GPU behavior", "future-stage availability may change before dependencies complete"),
        "conservative": ("t4_single_gpu", "gpu", "T4", max(1, min(4, t4_conc)), "keeps to validated single-GPU profiles and modest concurrency", "longer execution time"),
    }
    for plan_id, (profile, partition, gpu_type, conc, reason, risk) in templates.items():
        label_replicas = max(1, min(2, budget["max_labelcritic_replicas"], budget["labelcritic_replicas"], max(1, budget["max_h100_gpus"] // 2)))
        labelcritic_selection = select_labelcritic_72b_profile(snapshot)
        selected_lc = labelcritic_selection.get("selected_profile") or {"profile": "labelcritic_72b_2h100_formal", "partition": "gpuh100", "gpu_type": "H100", "gpu_count": 2}
        teacher_overrequest = adaptive_overrequest_count(conc)
        stages = [
            _stage("teacher_train_inference", profile, partition, gpu_type, 1, train_cases, conc, snapshot, reason, risk) | {"planned_overrequest_workers": teacher_overrequest},
            _stage("teacher_test_inference", profile, partition, gpu_type, 1, test_cases, conc, snapshot, reason, risk) | {"planned_overrequest_workers": teacher_overrequest},
            _stage("student_training", "a100_single_gpu", "gpua100", "A100", 1, 1, 1, snapshot, "single-GPU training remains the default until DDP smoke passes", "slower than validated DDP would be"),
            _stage("student_inference", profile, partition, gpu_type, 1, test_cases, conc, snapshot, reason, risk),
            _stage("labelcritic", selected_lc["profile"], selected_lc["partition"], selected_lc["gpu_type"], int(selected_lc["gpu_count"]), label_replicas, label_replicas, snapshot, "formal LabelCritic keeps Qwen2-VL-72B-AWQ and adapts tensor parallelism to available VRAM", "H100/A100 availability may require queueing") | {"labelcritic_resource_preflight": labelcritic_selection},
        ]
        plans.append(
            {
                "plan_id": plan_id,
                "recommended": plan_id == "balanced",
                "resource_binding_mode": "static_at_launch",
                "requested_gpu_budget": budget,
                "stages": stages,
                "potential_peak_t4_use": max([s["max_concurrent"] for s in stages if s["gpu_type"] == "T4"] or [0]),
                "potential_peak_a100_use": max([s["gpu_count_per_job"] * s["max_concurrent"] for s in stages if s["gpu_type"] == "A100"] or [0]),
                "potential_peak_h100_use": max([s["gpu_count_per_job"] * s["max_concurrent"] for s in stages if s["gpu_type"] == "H100"] or [0]),
            }
        )
    return {"snapshot_time": snapshot.get("snapshot_time"), "plans": plans, "snapshot_invariants": invariants, "limitations": ["Estimates do not reserve resources; all jobs must still enter Slurm.", "Runtime estimates remain low confidence until throughput history is available."]}


def validate_user_plan(plan: dict[str, Any], *, ddp_validated: bool = False, h100_ddp_allowed: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    for stage in plan.get("stages", []):
        name = str(stage.get("task_stage"))
        gpu_count = int(stage.get("gpu_count_per_job") or 0)
        gpu_type = str(stage.get("gpu_type") or "")
        if name == "student_training" and gpu_count > 1 and not ddp_validated:
            errors.append("Student multi-GPU training requires validated single-node DDP smoke before submission.")
        if name == "student_training" and gpu_type == "H100" and gpu_count == 4 and not h100_ddp_allowed:
            errors.append("4xH100 student DDP requires explicit user allowance and dedicated smoke.")
        if name == "labelcritic":
            model_id = str(stage.get("model_id") or LABELCRITIC_72B_MODEL_ID)
            allowed = (
                (gpu_type == "H100" and gpu_count in {1, 2})
                or (gpu_type == "A100" and gpu_count == 2)
                or (gpu_count == 4 and str(gpu_type).upper() in {"48GB", "L40S", "A6000", "A40"})
            )
            if model_id != LABELCRITIC_72B_MODEL_ID:
                errors.append("Formal LabelCritic must use Qwen/Qwen2-VL-72B-Instruct-AWQ; silent downgrade is forbidden.")
            if not allowed:
                errors.append("Formal LabelCritic resource profile must be one of: 1xH100 smoke, 2xH100, 2xA100, or 4x48GB.")
    if errors:
        raise SchedulerError("; ".join(errors))
    return {"status": "success", "plan_id": plan.get("plan_id")}


def labelcritic_replica_shards(task_count: int, replicas: int, output_root: str) -> list[dict[str, Any]]:
    replicas = max(1, min(int(replicas), 2))
    shards = []
    for replica in range(replicas):
        shards.append(
            {
                "replica_index": replica,
                "gpu_type": "H100",
                "gpu_count": 2,
                "tensor_parallel_size": 2,
                "model_id": LABELCRITIC_72B_MODEL_ID,
                "task_start": (task_count * replica) // replicas,
                "task_end": (task_count * (replica + 1)) // replicas,
                "output_dir": f"{output_root}/replica_{replica:02d}",
                "formal": True,
            }
        )
    return shards
