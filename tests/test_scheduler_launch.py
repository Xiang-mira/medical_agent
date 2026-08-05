from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scheduler.config import load_config
from scheduler.experiment_wizard import launch_experiment
from scheduler.utils import SchedulerError


def _cfg(tmp_path: Path, require_account: bool = False):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("case_id,ct_path\nBDMAP_0001,/tmp/a.nii.gz\n", encoding="utf-8")
    test.write_text("case_id,ct_path\nBDMAP_0002,/tmp/b.nii.gz\n", encoding="utf-8")
    data = {
        "project": {"name": "abdomenatlaspro_373", "full_target_count": 373, "pilot_target_count": 373, "target_terminology": "target_anatomical_structures"},
        "paths": {"run_root": str(tmp_path / "runs"), "target_config": "configs/student_3d_prompt_target_organs.json", "target_mapping": "configs/abdomenatlaspro_target_mapping_373.json", "train_input_case_list": str(train), "test_input_case_list": str(test)},
        "resource_profiles": "configs/resource_profiles.yaml",
        "resource_policy": {"teacher_train_resource_selection": {"preferred_order": ["t4_single_gpu"]}},
        "require_slurm_account": require_account,
        "slurm_defaults": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def test_launch_without_yes_and_dry_run_do_not_submit(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {"gpu": {}, "gpua100": {}, "gpuh100": {}}})
    called = {"submit": False}
    monkeypatch.setattr("scheduler.experiment_wizard.submit_plan", lambda *a, **k: called.update(submit=True))
    assert launch_experiment(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml")["submission"]["status"] == "not_submitted"
    assert launch_experiment(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", dry_run=True)["submission"]["status"] == "not_submitted"
    assert called["submit"] is False


def test_launch_requires_account_when_configured(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, require_account=True)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "account": None, "partitions": {"gpu": {}, "gpua100": {}, "gpuh100": {}}})
    with pytest.raises(SchedulerError):
        launch_experiment(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", yes=True)
