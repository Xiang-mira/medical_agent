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
