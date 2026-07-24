from __future__ import annotations

from pathlib import Path

from scheduler.config import load_config
from scheduler.planner import build_plan, submit_plan


def test_plan_generates_dag_and_dry_run_jobs(tmp_path, monkeypatch):
    cfg = load_config("configs/scheduler.local.example.yaml")
    data = dict(cfg.data)
    data["paths"] = dict(cfg.paths)
    data["paths"]["run_root"] = str(tmp_path)
    cfg = type(cfg)(path=cfg.path, data=data)
    plan = build_plan(cfg, "abdomenatlaspro_pilot338", backend="slurm", run_id="test_run")
    assert (tmp_path / "test_run" / "plan.json").exists()
    task_names = [t["task_name"] for t in plan["tasks"]]
    assert "teacher_test_array" in task_names
    metrics = next(t for t in plan["tasks"] if t["task_name"] == "metrics_cpu")
    assert metrics["dependencies"] == ["test_prediction_audit"]
    result = submit_plan(plan, dry_run=True)
    assert result["dry_run"] is True
    assert all(job["status"] == "dry_run" for job in result["jobs"].values())


def test_full_373_pipeline_uses_full_target_config(tmp_path):
    cfg = load_config("configs/scheduler.local.373.example.yaml")
    data = dict(cfg.data)
    data["paths"] = dict(cfg.paths)
    data["paths"]["run_root"] = str(tmp_path)
    cfg = type(cfg)(path=cfg.path, data=data)

    plan = build_plan(cfg, "abdomenatlaspro_373", backend="local", run_id="full_373")
    teacher = next(t for t in plan["tasks"] if t["task_name"] == "teacher_train_array")
    student = next(t for t in plan["tasks"] if t["task_name"] == "student_test_array")

    assert any("configs/student_3d_prompt_target_organs.json" in arg for arg in teacher["command"])
    assert any("configs/student_3d_prompt_target_organs.json" in arg for arg in student["command"])
    assert not any("abdomenatlaspro_pilot_338_target_config.json" in arg for arg in teacher["command"])


def test_adaptive_pipeline_uses_dynamic_manifest_arrays(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("case_id,ct_path\nBDMAP_0001,/tmp/a.nii.gz\nBDMAP_0002,/tmp/b.nii.gz\nBDMAP_0003,/tmp/c.nii.gz\n", encoding="utf-8")
    test.write_text("case_id,ct_path\nBDMAP_0004,/tmp/d.nii.gz\nBDMAP_0005,/tmp/e.nii.gz\n", encoding="utf-8")
    cfg = load_config("configs/scheduler.local.373.example.yaml")
    data = dict(cfg.data)
    data["paths"] = dict(cfg.paths)
    data["paths"]["run_root"] = str(tmp_path / "runs")
    data["paths"]["train_input_case_list"] = str(train)
    data["paths"]["test_input_case_list"] = str(test)
    cfg = type(cfg)(path=cfg.path, data=data)

    plan = build_plan(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", backend="slurm", run_id="adaptive")
    teacher = next(t for t in plan["tasks"] if t["task_name"] == "teacher_train_array")
    student = next(t for t in plan["tasks"] if t["task_name"] == "student_test_array")

    assert teacher["array"] == "0-2%3"
    assert student["array"] == "0-1%2"
