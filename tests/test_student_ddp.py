from __future__ import annotations

from pathlib import Path

import pytest

from scheduler.registry import enforce_student_single_gpu
from scheduler.slurm import render_task_script
from scheduler.utils import SchedulerError


def test_ddp_validated_false_rejects_multi_gpu_profiles(tmp_path):
    with pytest.raises(SchedulerError):
        enforce_student_single_gpu({"gpu_count": 2, "ddp_validated": False})
    with pytest.raises(SchedulerError):
        enforce_student_single_gpu({"gpu_count": 4, "ddp_validated": False})


def test_single_card_uses_single_process(tmp_path):
    script = render_task_script({"task_name": "student_train", "task_type": "student_train"}, {"partition": "gpua100", "gpu_count": 1, "memory_gb": 16}, ["python", "train.py"], run_dir=tmp_path)
    assert "torchrun" not in script


def test_ddp_profiles_use_torchrun_when_validated(tmp_path):
    task = {"task_name": "student_train", "task_type": "student_train", "launcher": "torchrun"}
    script2 = render_task_script(task, {"partition": "gpua100", "gpu_count": 2, "memory_gb": 16, "launcher": "torchrun", "ddp_validated": True}, ["python", "train.py"], run_dir=tmp_path)
    script4 = render_task_script(task, {"partition": "gpuh100", "gpu_count": 4, "memory_gb": 16, "launcher": "torchrun", "ddp_validated": True}, ["python", "train.py"], run_dir=tmp_path)
    assert "--nproc_per_node=2" in script2
    assert "--nproc_per_node=4" in script4


def test_train_script_contains_rank0_and_local_rank_contract():
    text = Path("scripts/train_voxtell_prompt_student.py").read_text(encoding="utf-8")
    assert "LOCAL_RANK" in text
    assert "is_rank0" in text
    assert "rank0_write" in text
    assert "DistributedDataParallel" in text
