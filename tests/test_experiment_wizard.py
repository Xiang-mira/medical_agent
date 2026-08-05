from __future__ import annotations

import builtins
from pathlib import Path

import yaml

from scheduler.config import load_config
from scheduler.experiment_wizard import experiment_wizard, prepare_experiment


def _cfg(tmp_path: Path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("case_id,ct_path\nBDMAP_0001,/tmp/a.nii.gz\nBDMAP_0002,/tmp/b.nii.gz\n", encoding="utf-8")
    test.write_text("case_id,ct_path\nBDMAP_0003,/tmp/c.nii.gz\n", encoding="utf-8")
    cfg = {
        "project": {"name": "abdomenatlaspro_373", "full_target_count": 373, "pilot_target_count": 373, "target_terminology": "target_anatomical_structures"},
        "paths": {"run_root": str(tmp_path / "runs"), "target_config": "configs/student_3d_prompt_target_organs.json", "target_mapping": "configs/abdomenatlaspro_target_mapping_373.json", "train_input_case_list": str(train), "test_input_case_list": str(test)},
        "resource_profiles": "configs/resource_profiles.yaml",
        "resource_policy": {"teacher_train_resource_selection": {"preferred_order": ["t4_single_gpu"]}},
        "gpu_budget": {"max_t4_gpus": 2, "max_a100_gpus": 1, "max_h100_gpus": 4, "max_labelcritic_replicas": 1},
        "slurm_defaults": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(path)


def test_wizard_default_no_does_not_submit(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {"gpu": {}, "gpua100": {}, "gpuh100": {}}})
    answers = iter(["balanced", "no"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    called = {"submit": False}
    monkeypatch.setattr("scheduler.experiment_wizard.submit_plan", lambda *a, **k: called.update(submit=True))
    result = experiment_wizard(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml")
    assert result["submission"]["status"] == "not_submitted"
    assert called["submit"] is False


def test_wizard_yes_calls_submit(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {"gpu": {}, "gpua100": {}, "gpuh100": {}}})
    answers = iter(["balanced", "yes"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    monkeypatch.setattr("scheduler.experiment_wizard.submit_plan", lambda plan, dry_run=False: {"status": "submitted", "dry_run": dry_run})
    result = experiment_wizard(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml")
    assert result["submission"]["status"] == "submitted"


def test_prepare_saves_user_plan_and_budget(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {"gpu": {"nodes_with_at_least_1_free_gpu": 1}, "gpua100": {}, "gpuh100": {}}})
    result = prepare_experiment(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", dry_run=True)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "user_choices.json").exists()
    assert result["resource_plan"]["requested_gpu_budget"]["max_t4_gpus"] == 2
    t4_peak = result["resource_plan"]["potential_peak_t4_use"]
    assert t4_peak <= 4
