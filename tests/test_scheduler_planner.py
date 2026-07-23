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
