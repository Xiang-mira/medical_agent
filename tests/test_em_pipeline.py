from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

from scheduler.cli import main
from scheduler.em_pipeline import build_em_plan, em_run, load_em_config
from scheduler.utils import SchedulerError, read_json, write_json_atomic


def _write_cases(path: Path, prefix: str, count: int, *, gt: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "ct_path"] + (["mask_dir"] if gt else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(count):
            case_id = f"{prefix}_{idx:02d}"
            row = {"case_id": case_id, "ct_path": f"/images/{case_id}/ct.nii.gz"}
            if gt:
                row["mask_dir"] = f"/masks/{case_id}/segmentations"
            writer.writerow(row)


def _cfg(tmp_path: Path) -> Path:
    image = tmp_path / "raw" / "images"
    mask = tmp_path / "raw" / "masks"
    image.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)
    mapping = tmp_path / "mapping_373.json"
    mapping.write_text(json.dumps({"targets": [{"target_name": "liver"}]}), encoding="utf-8")
    case_cfg = tmp_path / "case_selection.yaml"
    case_cfg.write_text(
        yaml.safe_dump({"paths": {"image_root": str(image), "mask_root": str(mask), "target_mapping": str(mapping)}, "candidate_pool": {}, "scoring": {}}),
        encoding="utf-8",
    )
    cfg = {
        "experiment": {"name": "em_train30_test20_round1_round2", "seed": 20260724, "train_cases": 30, "test_cases": 20, "max_inventory_cases": 10000, "ordering": "sorted_case_id"},
        "paths": {"work_root": str(tmp_path), "case_selection_config": str(case_cfg), "image_root": str(image), "mask_root": str(mask), "target_mapping": str(mapping), "target_config": "configs/student_3d_prompt_target_organs.json"},
        "cpu_parallel": {"header_scan": {"num_shards": 20, "max_concurrent": 5}, "deep_audit": {"num_shards": 10, "max_concurrent": 4}},
        "gpu_parallel": {
            "teacher_train": {"gpu_type": "t4", "max_concurrent": 10},
            "teacher_test": {"gpu_type": "t4", "max_concurrent": 6},
            "student_inference_train": {"gpu_type": "t4", "max_concurrent": 10},
            "student_inference_test": {"gpu_type": "t4", "max_concurrent": 6},
            "student_training": {"gpu_type": "a100", "gpus": 1},
            "labelcritic": {"replicas": 2, "gpus_per_replica": 4, "tensor_parallel_size": 4, "gpu_type": "h100"},
        },
        "resource_limits": {"max_t4_gpus": 16, "max_a100_gpus": 1, "max_h100_gpus": 8},
    }
    path = tmp_path / "em.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _plan(tmp_path: Path) -> tuple[Path, dict]:
    cfg = _cfg(tmp_path)
    out = tmp_path / "runs" / "em"
    plan = build_em_plan(cfg, out, dry_run=True)
    return out, plan


def _train10_cfg(tmp_path: Path) -> Path:
    image = tmp_path / "raw" / "images"
    mask = tmp_path / "raw" / "masks"
    image.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)
    mapping = tmp_path / "mapping_373.json"
    mapping.write_text(json.dumps({"targets": [{"target_name": "liver"}]}), encoding="utf-8")
    target_config = tmp_path / "student_3d_prompt_target_organs.json"
    target_config.write_text(json.dumps({"target_organs": [f"organ_{idx:03d}" for idx in range(373)]}), encoding="utf-8")
    case_list = tmp_path / "train10.csv"
    case_list.parent.mkdir(parents=True, exist_ok=True)
    with case_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        for idx in range(10):
            case_id = f"BDMAP_{idx:08d}"
            writer.writerow({"case_id": case_id, "ct_path": f"/images/{case_id}/ct.nii.gz"})
    cfg = {
        "experiment": {"name": "em_train10_round1_round2", "seed": 20260726, "train_cases": 10, "test_cases": 0, "max_inventory_cases": 10, "ordering": "sorted_case_id"},
        "paths": {
            "work_root": str(tmp_path),
            "code_root": str(tmp_path),
            "case_list": str(case_list),
            "target_config": str(target_config),
            "target_mapping": str(mapping),
            "python_bin": sys.executable,
            "vllm_container": "/containers/vllm.sif",
            "labelcritic_model_dir": "/models/qwen72b",
            "labelcritic_served_model_name": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            "labelcritic_port": 8000,
            "voxtell_model_dir": "/models/voxtell",
            "qwen_embedding_model": "/models/qwen3-embedding",
            "image_root": str(image),
            "mask_root": str(mask),
        },
        "teachers": {"models": ["epai_20250421", "vsmtrans"]},
        "runtime": {"student_backend": "voxtell_style_3d_prompt", "mstep_backend": "project_voxtell_prompt_distillation_student", "candidate_mode_round2": "em_student_vs_previous", "amp_mode": "off", "batch_size": 1, "max_steps": 2000},
        "gpu_parallel": {
            "teacher_train": {"partition": "gpu", "gpu_type": "t4", "gpus": 1, "cpus_per_task": 4, "memory": "32G", "time": "08:00:00", "smoke_time": "00:45:00", "max_concurrent": 8},
            "student_inference_train": {"partition": "gpu", "gpu_type": "t4", "gpus": 1, "cpus_per_task": 4, "memory": "32G", "time": "08:00:00", "max_concurrent": 8},
        },
        "resource_limits": {"max_t4_gpus": 8, "max_a100_gpus": 1, "max_h100_gpus": 4},
    }
    path = tmp_path / "em_train10.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_em_fixed_split_and_no_gt_manifests(tmp_path):
    out, plan = _plan(tmp_path)
    _write_cases(out / "manifests" / "train30_input_no_gt.csv", "TRAIN", 30)
    _write_cases(out / "manifests" / "test20_input_no_gt.csv", "TEST", 20)
    cfg = load_em_config(_cfg(tmp_path))
    from scheduler.em_pipeline import write_split_metadata

    metadata = write_split_metadata(out, cfg, plan["fingerprint"])
    assert len(metadata["train_case_ids"]) == 30
    assert len(metadata["test_case_ids"]) == 20
    assert not set(metadata["train_case_ids"]) & set(metadata["test_case_ids"])
    assert metadata["seed"] == 20260724
    assert metadata["rounds_use_same_split"] == [1, 2]
    assert "mask_path" not in (out / "manifests" / "train30_input_no_gt.csv").read_text(encoding="utf-8")
    assert "mask_path" not in (out / "manifests" / "test20_input_no_gt.csv").read_text(encoding="utf-8")


def test_em_test20_never_enters_training_or_labelcritic_and_round2_reuses_teacher(tmp_path):
    _, plan = _plan(tmp_path)
    stages = {s["stage"]: s for s in plan["stages"]}
    train_only = [s for s in stages.values() if s["kind"] in {"student_train", "student_train_smoke", "labelcritic_shard", "labelcritic_smoke", "build_manifest"}]
    assert all(s.get("split") == "train" for s in train_only)
    assert "teacher_train30_array" not in stages["labelcritic_r2_shard_0"]["dependencies"]
    assert stages["labelcritic_r2_shard_0"]["dependencies"] == ["student_r1_train30_inference"]
    teacher_stages = [name for name in stages if name.startswith("teacher_") and name.endswith("_array")]
    assert teacher_stages == ["teacher_train30_array", "teacher_test20_array"]


def test_em_case_selection_dag_and_report_stage_are_real(tmp_path):
    out, plan = _plan(tmp_path)
    case_tasks = {task["stage"]: task for task in plan["case_selection_plan"]["tasks"]}
    assert case_tasks["header_scan_array"]["array"] == "0-19%5"
    assert case_tasks["deep_mask_audit_array"]["array"] == "0-9%4"
    assert case_tasks["merge_header_metrics"]["command"][4] == "merge-header-metrics"
    assert case_tasks["candidate_prefilter"]["command"][4] == "prefilter"
    assert case_tasks["merge_deep_metrics"]["command"][4] == "merge-deep-metrics"
    assert case_tasks["select_and_split"]["command"][4] == "select"
    assert case_tasks["validate_manifests"]["command"][4] == "validate"
    assert case_tasks["final_selection_report"]["command"][4] == "report"
    assert (out / "case_selection_30_20" / "generated_slurm" / "header_scan_array.sbatch").exists()


def test_em_gpu_tasks_resources_and_dependencies(tmp_path):
    _, plan = _plan(tmp_path)
    stages = {s["stage"]: s for s in plan["stages"]}
    assert stages["teacher_train30_array"]["array"] == "0-29%10"
    assert stages["teacher_test20_array"]["array"] == "0-19%6"
    assert stages["student_r1_train30_inference"]["array"] == "0-29%10"
    assert stages["student_r1_test20_inference"]["array"] == "0-19%6"
    assert stages["student_r2_test20_inference"]["array"] == "0-19%8"
    assert stages["student_r1_train"]["gpu_type"] == "a100"
    assert stages["student_r1_train"]["gpu_count"] == 1
    assert stages["labelcritic_r1_shard_0"]["gpu_count"] == 4
    assert stages["labelcritic_r1_shard_1"]["gpu_count"] == 4
    assert len(stages["labelcritic_r1_shard_0"]["labelcritic_cases"]) == 15
    assert len(stages["labelcritic_r1_shard_1"]["labelcritic_cases"]) == 15
    assert stages["student_r1_train30_inference"]["dependencies"] == ["student_r1_train"]
    assert stages["student_r1_test20_inference"]["dependencies"] == ["student_r1_train"]
    assert "round1_evaluation" not in stages["labelcritic_r2_shard_0"]["dependencies"]


def test_em_dry_run_writes_artifacts_and_does_not_submit(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    called = {"sbatch": False}

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "sbatch":
            called["sbatch"] = True
            raise AssertionError("dry-run must not call sbatch")
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("scheduler.em_pipeline.subprocess.run", fake_run)
    out = tmp_path / "runs" / "em"
    result = em_run(cfg, out, dry_run=True)
    assert result["submitted"] is False
    for name in ("run_plan.json", "resource_plan.json", "dependency_graph.json", "submission_receipt.json", "state.json"):
        assert (out / name).exists()
    assert (out / "generated_slurm").is_dir()
    assert (out / "logs").is_dir()
    assert (out / "provenance").is_dir()
    assert called["sbatch"] is False


def test_em_output_dir_must_be_inside_work_root(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(SchedulerError):
        build_em_plan(cfg, Path("/tmp/outside-em-work-root"), dry_run=True)


def test_em_retry_failed_and_resume_filters_stages(tmp_path):
    cfg = _cfg(tmp_path)
    out = tmp_path / "runs" / "em"
    build_em_plan(cfg, out, dry_run=True)
    state = read_json(out / "state.json")
    state["stages"]["teacher_train30_smoke"]["status"] = "completed"
    state["stages"]["teacher_test20_smoke"]["status"] = "failed"
    write_json_atomic(out / "state.json", state)
    retry = build_em_plan(cfg, out, dry_run=True, retry_failed=True)
    assert [s["stage"] for s in retry["stages"]] == ["teacher_test20_smoke"]
    build_em_plan(cfg, out, dry_run=True)
    state = read_json(out / "state.json")
    state["stages"]["teacher_train30_smoke"]["status"] = "completed"
    state["stages"]["teacher_test20_smoke"]["status"] = "failed"
    write_json_atomic(out / "state.json", state)
    resume = build_em_plan(cfg, out, dry_run=True, resume=True)
    assert "teacher_train30_smoke" not in [s["stage"] for s in resume["stages"]]
    state["stages"]["teacher_test20_smoke"]["status"] = "partial"
    write_json_atomic(out / "state.json", state)
    with pytest.raises(SchedulerError):
        build_em_plan(cfg, out, dry_run=True, resume=True)


def test_em_cli_and_mapping_file_not_modified(tmp_path):
    cfg = _cfg(tmp_path)
    mapping = Path(yaml.safe_load(cfg.read_text(encoding="utf-8"))["paths"]["target_mapping"])
    before = mapping.read_bytes()
    out = tmp_path / "runs" / "cli-em"
    assert main(["em-run", "--config", str(cfg), "--output-dir", str(out), "--dry-run"]) == 0
    assert main(["em-status", "--output-dir", str(out)]) == 0
    assert mapping.read_bytes() == before


def test_train10_dag_matches_required_formal_sequence(tmp_path):
    cfg = _train10_cfg(tmp_path)
    out = tmp_path / "runs" / "em_train10"
    plan = build_em_plan(cfg, out, dry_run=True)
    stages = [stage["stage"] for stage in plan["stages"]]
    assert stages == [
        "teacher_train10_smoke",
        "teacher_train10_array",
        "teacher_train10_merge_and_audit",
        "labelcritic_r1",
        "build_round1_manifest",
        "student_r1_train_smoke",
        "student_r1_train",
        "student_r1_inference",
        "prepare_round2_em_student_vs_previous",
        "labelcritic_r2",
        "round2_material_update_gate",
        "student_r2_train",
        "student_r2_inference",
        "final_train10_audit",
    ]


def test_train10_resources_arrays_and_round2_dependency(tmp_path):
    cfg = _train10_cfg(tmp_path)
    out = tmp_path / "runs" / "em_train10"
    plan = build_em_plan(cfg, out, dry_run=True)
    stages = {s["stage"]: s for s in plan["stages"]}
    assert stages["teacher_train10_array"]["array"] == "0-9%8"
    assert stages["student_r1_inference"]["array"] == "0-9%8"
    assert stages["student_r2_inference"]["array"] == "0-9%8"
    assert stages["labelcritic_r1"]["partition"] == "gpuh100"
    assert stages["labelcritic_r1"]["gpu_count"] == 4
    assert stages["labelcritic_r2"]["partition"] == "gpuh100"
    assert stages["labelcritic_r2"]["gpu_count"] == 4
    assert stages["labelcritic_r2"]["dependencies"] == ["prepare_round2_em_student_vs_previous"]
    assert stages["prepare_round2_em_student_vs_previous"]["dependencies"] == ["student_r1_inference"]
    assert not any(stage.get("eval_reference") for stage in plan["stages"])
    assert plan["split_metadata"]["train_case_ids"] == [f"BDMAP_{idx:08d}" for idx in range(10)]


def test_train10_dry_run_writes_formal_slurm_scripts(tmp_path):
    cfg = _train10_cfg(tmp_path)
    out = tmp_path / "runs" / "em_train10"
    plan = build_em_plan(cfg, out, dry_run=True)
    labelcritic_script = (out / "generated_slurm" / "labelcritic_r1.sbatch").read_text(encoding="utf-8")
    assert "curl --noproxy '*'" in labelcritic_script
    assert "/v1/models" in labelcritic_script
    assert "--dependency=afterok" not in labelcritic_script
    assert (out / "generated_slurm" / "teacher_train10_array.sbatch").exists()
    assert plan["policies"]["no_test_or_eval_reference"] is True
