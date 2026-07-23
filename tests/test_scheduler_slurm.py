from __future__ import annotations

from pathlib import Path

import pytest

from scheduler.slurm import profile_to_directives, render_labelcritic_script, render_task_script
from scheduler.utils import SchedulerError


def test_cpu_task_has_no_gpu_directive(tmp_path):
    task = {"task_name": "cpu", "task_type": "audit"}
    profile = {"partition": "interactive", "gpu_count": 0, "cpus_per_task": 2, "memory_gb": 8}
    text = "\n".join(profile_to_directives(task, profile, run_dir=tmp_path))
    assert "--gres=gpu" not in text


def test_student_train_rejects_multi_gpu(tmp_path):
    task = {"task_name": "student_train", "task_type": "student_train"}
    profile = {"partition": "gpua100", "gpu_count": 2, "cpus_per_task": 8, "memory_gb": 64}
    with pytest.raises(SchedulerError):
        profile_to_directives(task, profile, run_dir=tmp_path)


def test_labelcritic_script_keeps_vllm_and_client_same_job(tmp_path):
    task = {"task_name": "labelcritic_batch", "task_type": "labelcritic", "resource_profile": "labelcritic_72b_4h100_formal"}
    profile = {
        "partition": "gpuh100",
        "gpu_count": 4,
        "cpus_per_task": 24,
        "memory_gb": 256,
        "container": "/containers/vllm.sif",
        "model_dir": "/models/qwen72b",
        "model_id": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
        "tensor_parallel_size": 4,
        "port": 8000,
    }
    script = render_labelcritic_script(task, profile, ["python", "-m", "scheduler.cli", "run-task", "--task", "labelcritic_batch", "--run-dir", str(tmp_path)], run_dir=tmp_path)
    assert "--gres=gpu:4" in script
    assert "--tensor-parallel-size 4" in script
    assert "NO_PROXY=127.0.0.1,localhost,::1" in script
    assert "curl --noproxy '*'" in script
    assert "export CUDA_VISIBLE_DEVICES" not in script
    assert "\nCUDA_VISIBLE_DEVICES=" not in script


def test_unsafe_sbatch_value_rejected(tmp_path):
    task = {"task_name": "bad", "task_type": "audit"}
    profile = {"partition": "gpu;rm -rf /", "gpu_count": 0}
    with pytest.raises(SchedulerError):
        profile_to_directives(task, profile, run_dir=tmp_path)
