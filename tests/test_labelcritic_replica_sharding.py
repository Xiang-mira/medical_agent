from __future__ import annotations

import pytest

from scheduler.resource_recommender import labelcritic_replica_shards, validate_user_plan
from scheduler.slurm import render_labelcritic_script
from scheduler.utils import SchedulerError


def test_labelcritic_replicas_are_4xh100_and_separate_outputs():
    shards = labelcritic_replica_shards(101, 2, "/runs/labelcritic")
    assert len(shards) == 2
    assert all(s["gpu_count"] == 4 and s["tensor_parallel_size"] == 4 for s in shards)
    assert shards[0]["task_end"] == shards[1]["task_start"]
    assert shards[0]["output_dir"] != shards[1]["output_dir"]


def test_labelcritic_formal_rejects_1xh100():
    with pytest.raises(SchedulerError):
        validate_user_plan({"stages": [{"task_stage": "labelcritic", "gpu_type": "H100", "gpu_count_per_job": 1}]})


def test_labelcritic_server_and_client_same_job(tmp_path):
    script = render_labelcritic_script(
        {"task_name": "labelcritic_batch", "task_type": "labelcritic"},
        {"partition": "gpuh100", "gpu_count": 4, "memory_gb": 256, "container": "/c.sif", "model_dir": "/m", "model_id": "Qwen/Qwen2-VL-72B-Instruct-AWQ", "tensor_parallel_size": 4},
        ["python", "client.py"],
        run_dir=tmp_path,
    )
    assert "--gres=gpu:4" in script
    assert "--tensor-parallel-size 4" in script
    assert "127.0.0.1" in script
    assert "python client.py" in script
