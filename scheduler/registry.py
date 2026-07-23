from __future__ import annotations

from typing import Any

from .utils import SchedulerError


def select_resource_tier(policy: dict[str, Any], task_name: str, smoke_results: dict[str, Any] | None = None) -> str:
    """Select the first usable tier according to smoke results.

    Missing smoke data intentionally returns the first tier, but callers should
    keep the result marked as "requires_smoke" when smoke_required is true.
    """
    task_policy = policy.get(task_name) or {}
    order = list(task_policy.get("preferred_order") or [])
    if not order:
        raise SchedulerError(f"No preferred_order configured for resource policy {task_name}")
    smoke_results = smoke_results or {}
    for tier in order:
        result = smoke_results.get(tier)
        if result is None:
            return tier
        if isinstance(result, dict) and result.get("status") == "success":
            return tier
    return order[-1]


def enforce_student_single_gpu(profile: dict[str, Any]) -> None:
    if int(profile.get("gpu_count") or 0) > 1:
        raise SchedulerError("Student training profile must remain single GPU unless DDP is explicitly validated")
