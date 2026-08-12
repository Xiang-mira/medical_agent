from __future__ import annotations

import pytest

from scheduler.resource_recommender import LABELCRITIC_72B_MODEL_ID, labelcritic_replica_shards, select_labelcritic_72b_profile, validate_user_plan
from scheduler.slurm import render_labelcritic_script
from scheduler.utils import SchedulerError


def test_labelcritic_replicas_are_2xh100_and_separate_outputs():
    shards = labelcritic_replica_shards(101, 2, "/runs/labelcritic")
    assert len(shards) == 2
    assert all(s["gpu_count"] == 2 and s["tensor_parallel_size"] == 2 for s in shards)
    assert all(s["model_id"] == LABELCRITIC_72B_MODEL_ID for s in shards)
    assert shards[0]["task_end"] == shards[1]["task_start"]
    assert shards[0]["output_dir"] != shards[1]["output_dir"]


def test_labelcritic_formal_rejects_model_downgrade():
    with pytest.raises(SchedulerError):
        validate_user_plan({"stages": [{"task_stage": "labelcritic", "gpu_type": "H100", "gpu_count_per_job": 2, "model_id": "Qwen/Qwen2-VL-7B-Instruct"}]})


def test_labelcritic_resource_selection_reports_insufficient_without_downgrade():
    result = select_labelcritic_72b_profile({"partitions": {"gpuh100": {}, "gpua100": {}, "gpu": {}}})
    assert result["status"] == "insufficient_resource"
    assert result["model_id"] == LABELCRITIC_72B_MODEL_ID
    assert result["silent_downgrade"] is False


def test_labelcritic_server_and_client_same_job(tmp_path):
    script = render_labelcritic_script(
        {"task_name": "labelcritic_batch", "task_type": "labelcritic"},
        {"partition": "gpuh100", "gpu_count": 2, "memory_gb": 192, "container": "/c.sif", "model_dir": "/m", "model_id": "Qwen/Qwen2-VL-72B-Instruct-AWQ", "tensor_parallel_size": 2, "gpu_memory_utilization": 0.88, "max_model_len": 8192},
        ["python", "client.py"],
        run_dir=tmp_path,
    )
    assert "--gres=gpu:2" in script
    assert "--tensor-parallel-size 2" in script
    assert "--gpu-memory-utilization 0.88" in script
    assert "--max-model-len 8192" in script
    assert "127.0.0.1" in script
    assert "python client.py" in script
