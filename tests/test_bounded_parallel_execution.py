from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scheduler.case_selection import (
    build_case_selection_slurm_plan,
    case_selection_fingerprint,
    deterministic_shard,
    load_case_selection_config,
    merge_metric_shards,
)
from scheduler.config import load_config
from scheduler.experiment_wizard import prepare_experiment
from scheduler.utils import SchedulerError, write_json_atomic


def _case_selection_cfg(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    mapping = tmp_path / "mapping.json"
    image_root.mkdir()
    mask_root.mkdir()
    mapping.write_text(json.dumps({"targets": []}), encoding="utf-8")
    cfg = tmp_path / "case_selection.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"image_root": str(image_root), "mask_root": str(mask_root), "target_mapping": str(mapping)},
                "candidate_pool": {"multiplier": 2, "minimum": 4, "maximum": 10},
                "scoring": {},
            }
        ),
        encoding="utf-8",
    )
    return cfg


def test_case_selection_dry_run_generates_bounded_cpu_arrays(tmp_path):
    cfg = _case_selection_cfg(tmp_path)
    out = tmp_path / "selection"
    plan = build_case_selection_slurm_plan(50, 50, 20260724, out, config_path=cfg, dry_run=True, max_inventory_cases=10000)
    assert plan["submitted"] is False
    assert (out / "run_plan.json").exists()
    assert (out / "resource_plan.json").exists()
    assert (out / "task_manifest.json").exists()
    assert (out / "dependency_graph.json").exists()
    header = out / "generated_slurm" / "header_scan_array.sbatch"
    deep = out / "generated_slurm" / "deep_mask_audit_array.sbatch"
    assert "#SBATCH --array=0-19%5" in header.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-9%4" in deep.read_text(encoding="utf-8")
    all_scripts = "\n".join(p.read_text(encoding="utf-8") for p in (out / "generated_slurm").glob("*.sbatch"))
    assert "#SBATCH --partition=cpu" in all_scripts
    assert "--gres=gpu" not in all_scripts
    assert "--max-inventory-cases 10000" in all_scripts
    assert "--shard-index ${SLURM_ARRAY_TASK_ID}" in all_scripts


def test_deterministic_shards_cover_without_overlap():
    rows = [{"case_id": case_id} for case_id in ["BDMAP_0003", "BDMAP_0001", "BDMAP_0002", "BDMAP_0004", "BDMAP_0005"]]
    shards = [deterministic_shard(rows, idx, 3) for idx in range(3)]
    flattened = [row["case_id"] for shard in shards for row in shard]
    assert sorted(flattened) == ["BDMAP_0001", "BDMAP_0002", "BDMAP_0003", "BDMAP_0004", "BDMAP_0005"]
    assert len(flattened) == len(set(flattened))
    assert [row["case_id"] for row in deterministic_shard(list(reversed(rows)), 0, 3)] == [row["case_id"] for row in shards[0]]


def test_merge_metric_shards_fails_missing_and_duplicate(tmp_path):
    cfg = _case_selection_cfg(tmp_path)
    fingerprint = case_selection_fingerprint(load_case_selection_config(cfg), 1, 1, 7, 10000)
    shard_dir = tmp_path / "out" / "shards" / "header"
    write_json_atomic(shard_dir / "header_0.json", {"fingerprint": fingerprint, "shard_index": 0, "num_shards": 2, "rows": [{"case_id": "BDMAP_0001"}]})
    with pytest.raises(SchedulerError):
        merge_metric_shards("header", tmp_path / "out", expected_shards=2, expected_fingerprint=fingerprint)
    write_json_atomic(shard_dir / "header_1.json", {"fingerprint": fingerprint, "shard_index": 1, "num_shards": 2, "rows": [{"case_id": "BDMAP_0001"}]})
    with pytest.raises(SchedulerError):
        merge_metric_shards("header", tmp_path / "out", expected_shards=2, expected_fingerprint=fingerprint)


def _experiment_cfg(tmp_path: Path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("case_id,ct_path\n" + "\n".join(f"BDMAP_{i:04d},/tmp/train_{i}.nii.gz" for i in range(50)) + "\n", encoding="utf-8")
    test.write_text("case_id,ct_path\n" + "\n".join(f"BDMAP_{i + 100:04d},/tmp/test_{i}.nii.gz" for i in range(50)) + "\n", encoding="utf-8")
    cfg = {
        "project": {"name": "abdomenatlaspro_373", "full_target_count": 373, "pilot_target_count": 373, "target_terminology": "target_anatomical_structures"},
        "paths": {
            "run_root": str(tmp_path / "runs"),
            "target_config": "configs/student_3d_prompt_target_organs.json",
            "target_mapping": "configs/abdomenatlaspro_target_mapping_373.json",
            "train_input_case_list": str(train),
            "test_input_case_list": str(test),
        },
        "resource_profiles": "configs/resource_profiles.yaml",
        "resource_policy": {"teacher_train_resource_selection": {"preferred_order": ["t4_single_gpu"]}},
        "gpu_budget": {"max_t4_gpus": 8, "max_a100_gpus": 2, "max_h100_gpus": 8, "labelcritic_replicas": 1, "max_labelcritic_replicas": 2},
        "slurm_defaults": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(path)


def test_formal_experiment_generates_50_case_t4_arrays_and_artifacts(monkeypatch, tmp_path):
    cfg = _experiment_cfg(tmp_path)
    monkeypatch.setattr("scheduler.experiment_wizard.discover_resource_snapshot", lambda *a, **k: {"snapshot_time": "now", "partitions": {"gpu": {}, "gpua100": {}, "gpuh100": {}}})
    result = prepare_experiment(cfg, "configs/pipelines/abdomenatlaspro_373_adaptive.yaml", run_id="bounded", dry_run=True)
    run_dir = Path(result["run_dir"])
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    arrays = {task["task_name"]: task["array"] for task in plan["tasks"] if task.get("array")}
    assert arrays["teacher_train_array"] == "0-49%8"
    assert arrays["teacher_test_array"] == "0-49%8"
    assert arrays["student_test_array"] == "0-49%8"
    teacher_script = (run_dir / "generated_slurm" / "teacher_train_array.sbatch").read_text(encoding="utf-8")
    student_script = (run_dir / "generated_slurm" / "student_test_array.sbatch").read_text(encoding="utf-8")
    assert "--gres=gpu:1" in teacher_script
    assert "--gres=gpu:1" in student_script
    assert "gpuh100" not in teacher_script
    assert "gpuh100" not in student_script
    student_train = (run_dir / "generated_slurm" / "student_train.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpua100" in student_train
    assert "--gres=gpu:1" in student_train
    assert (run_dir / "run_plan.json").exists()
    resource_plan = json.loads((run_dir / "resource_plan.json").read_text(encoding="utf-8"))
    assert resource_plan["max_t4_concurrent"] == 8
    assert resource_plan["max_a100_training_gpus"] == 1
    assert resource_plan["max_h100_gpus"] == 4
