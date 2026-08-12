from __future__ import annotations

import pytest

from scheduler.resource_recommender import adaptive_overrequest_count, recommend_resource_plans, validate_user_plan
from scheduler.utils import SchedulerError


def _snapshot():
    return {
        "snapshot_time": "now",
        "partitions": {
            "gpu": {"nodes_with_at_least_1_free_gpu": 2, "queued_jobs": 0},
            "gpua100": {"nodes_with_at_least_1_free_gpu": 1, "nodes_with_at_least_2_free_gpus": 0, "queued_jobs": 1},
            "gpuh100": {"nodes_with_at_least_1_free_gpu": 1, "nodes_with_at_least_2_free_gpus": 1, "nodes_with_at_least_4_free_gpus": 0, "queued_jobs": 2},
        },
    }


def test_recommender_emits_balanced_default_and_labelcritic_shape():
    plans = recommend_resource_plans(_snapshot(), 5, 3)
    balanced = next(p for p in plans["plans"] if p["plan_id"] == "balanced")
    assert balanced["recommended"] is True
    labelcritic = next(s for s in balanced["stages"] if s["task_stage"] == "labelcritic")
    assert labelcritic["gpu_count_per_job"] == 2
    assert labelcritic["currently_allocatable_shape"] == 1
    assert labelcritic["labelcritic_resource_preflight"]["model_id"] == "Qwen/Qwen2-VL-72B-Instruct-AWQ"


def test_adaptive_overrequest_30_to_40():
    assert adaptive_overrequest_count(30) == 40


def test_user_plan_rejects_unvalidated_ddp():
    plan = {"plan_id": "manual", "stages": [{"task_stage": "student_training", "gpu_type": "A100", "gpu_count_per_job": 2}]}
    with pytest.raises(SchedulerError):
        validate_user_plan(plan, ddp_validated=False)
