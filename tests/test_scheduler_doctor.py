from __future__ import annotations

from pathlib import Path

import yaml

from scheduler.cli import main


def test_doctor_writes_reports_and_does_not_submit(monkeypatch, tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("case_id,ct_path\nBDMAP_0001,/tmp/a.nii.gz\n", encoding="utf-8")
    test.write_text("case_id,ct_path\nBDMAP_0002,/tmp/b.nii.gz\n", encoding="utf-8")
    cfg = {
        "project": {"name": "abdomenatlaspro_373", "full_target_count": 373, "pilot_target_count": 373, "target_terminology": "target_anatomical_structures"},
        "paths": {"run_root": str(tmp_path / "runs"), "target_config": "configs/student_3d_prompt_target_organs.json", "target_mapping": "configs/abdomenatlaspro_target_mapping_373.json", "train_input_case_list": str(train), "test_input_case_list": str(test)},
        "resource_profiles": "configs/resource_profiles.yaml",
        "resource_policy": {"teacher_train_resource_selection": {"preferred_order": ["t4_single_gpu"]}},
        "slurm_defaults": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "doctor"
    monkeypatch.setattr("scheduler.doctor.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {}})
    assert main(["doctor", "--config", str(path), "--pipeline", "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", "--output-dir", str(out)]) == 0
    assert (out / "doctor_report.json").exists()
    assert (out / "doctor_report.md").exists()
