from __future__ import annotations

import csv
from pathlib import Path

from scheduler.config import load_config
from scheduler.executor import execute_task
from scheduler.planner import build_plan
from scheduler.utils import read_json


def _write_manifest(path: Path, start: int, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        for i in range(start, start + count):
            case_id = f"BDMAP_{i:08d}"
            writer.writerow({"case_id": case_id, "ct_path": f"/images/{case_id}/ct.nii.gz"})


def _config_with_manifests(tmp_path: Path):
    cfg = load_config("configs/scheduler.local.example.yaml")
    data = dict(cfg.data)
    data["paths"] = dict(cfg.paths)
    data["paths"]["run_root"] = str(tmp_path / "runs")
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    reserve = tmp_path / "reserve.csv"
    _write_manifest(train, 1, 20)
    _write_manifest(test, 21, 20)
    _write_manifest(reserve, 41, 10)
    data["paths"]["pilot_train_input_case_list"] = str(train)
    data["paths"]["pilot_test_input_case_list"] = str(test)
    data["paths"]["pilot_reserve_case_list"] = str(reserve)
    return type(cfg)(path=cfg.path, data=data)


def test_array_task_materializes_one_row_manifest_and_scrubs_gt_env(tmp_path, monkeypatch):
    cfg = _config_with_manifests(tmp_path)
    plan = build_plan(cfg, "abdomenatlaspro_pilot338", backend="local", run_id="exec")
    run_dir = Path(plan["run_dir"])
    captured = {}

    def fake_run(command, *, cwd, env=None, log_path=None):
        captured["command"] = command
        captured["env"] = env or {}
        return {"return_code": 0, "elapsed_seconds": 0}

    monkeypatch.setenv("MASK_ROOT", "/forbidden/mask_only")
    monkeypatch.setattr("scheduler.executor._run_subprocess", fake_run)
    result = execute_task(run_dir, "teacher_train_array", array_index=3)
    assert result["status"] == "success"
    assert "--case-list" in captured["command"]
    shard = Path(captured["command"][captured["command"].index("--case-list") + 1])
    rows = list(csv.DictReader(shard.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "BDMAP_00000004"
    assert "MASK_ROOT" not in captured["env"]
    status = read_json(run_dir / "status" / "teacher_train_array_3.json")
    assert status["status"] == "success"


def test_metrics_task_keeps_eval_policy_env(tmp_path, monkeypatch):
    cfg = _config_with_manifests(tmp_path)
    plan = build_plan(cfg, "abdomenatlaspro_pilot338", backend="local", run_id="exec_metrics")
    run_dir = Path(plan["run_dir"])
    task = next(t for t in plan["tasks"] if t["task_name"] == "metrics_cpu")
    from scheduler.executor import _env_for_task

    monkeypatch.setenv("MASK_ROOT", "/allowed/for/metrics")
    env = _env_for_task(task)
    assert env["MASK_ROOT"] == "/allowed/for/metrics"
