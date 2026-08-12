from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


def _save(array: np.ndarray, path: Path, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return path


def _write_executable(path: Path, body: str = "#!/usr/bin/env bash\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_checkpoint_registry_paths_resolve_from_checkpoint_root_not_registry_copy(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    checkpoint_root = tmp_path / "hpc_checkpoints"
    copied_registry = tmp_path / "output_copy" / "registry.yaml"
    fake_predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    copied_registry.parent.mkdir(parents=True)
    copied_registry.write_text(
        """
checkpoint_root: checkpoints
models:
  cads557:
    name: CADS557
    enabled: true
    runner: command_template
    covered_organs: [blood]
    checkpoint_path: checkpoints/CADS_series/CADS_series
    dataset_json_path: checkpoints/CADS_series/CADS_series/Dataset557_Brain257/trainer/dataset.json
    dataset_id: 557
    trainer: nnUNetTrainer
    plans: nnUNetPlans
    command_template: python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --predict-executable {predict_executable} --organs blood
""",
        encoding="utf-8",
    )
    image = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")

    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "cads557",
        registry_path=copied_registry,
        case_id="case",
        dry_run=True,
        extra_context={
            "requested_organs": ["blood"],
            "checkpoint_root": str(checkpoint_root),
            "predict_executable": str(fake_predictor),
        },
    )

    assert result["status"] == "dry_run"
    assert str(checkpoint_root / "CADS_series" / "CADS_series") in result["command"]
    assert str(copied_registry.parent / "checkpoints") not in result["command"]
    assert result["resolved_predict_executable"] == str(fake_predictor.resolve())


def test_nnunet_predictor_resolver_returns_absolute_path_from_path(monkeypatch, tmp_path: Path):
    from cli_anything.medai.core.runtime_resolver import resolve_nnunet_predictor

    fake = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    monkeypatch.delenv("NNUNETV2_PREDICT_EXECUTABLE", raising=False)
    monkeypatch.delenv("MEDAI_NNUNETV2_PREDICT", raising=False)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    resolved = resolve_nnunet_predictor(require_exists=True)

    assert resolved == fake.resolve()
    assert resolved.is_absolute()


def test_nnunet_wrapper_dry_run_uses_absolute_predictor_from_path(monkeypatch, tmp_path: Path):
    fake = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    monkeypatch.delenv("NNUNETV2_PREDICT_EXECUTABLE", raising=False)
    monkeypatch.delenv("MEDAI_NNUNETV2_PREDICT", raising=False)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/nnunetv2_predict_and_split.py",
            "--image", str(tmp_path / "case" / "ct.nii.gz"),
            "--output", str(tmp_path / "out"),
            "--dataset-id", "557",
            "--nnunet-results", str(tmp_path / "checkpoints"),
            "--dataset-json", str(tmp_path / "dataset.json"),
            "--trainer", "nnUNetTrainer",
            "--plans", "nnUNetPlans",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["command"][0] == str(fake.resolve())


def test_unest_does_not_inherit_nnunet_diagnostic_args(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    monkeypatch.setenv("MEDAI_NNUNET_DIAGNOSTIC", "1")
    monkeypatch.setenv("MEDAI_NNUNET_KEEP_WORKDIR", "1")
    image = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "unest",
        registry_path=REPO_ROOT / "configs" / "model_registry.yaml",
        case_id="case",
        dry_run=True,
        extra_context={"requested_organs": ["kidney_cortex"]},
    )

    assert result["status"] == "dry_run"
    assert "--diagnostic" not in result["command"]
    assert "--keep-workdir" not in result["command"]
    assert "--python-executable" in result["command"]
    assert "/home/xhan74/envs/medical_agent_train_py311/bin/python" in result["command"]


def test_preflight_blocks_formal_existing_output_root(tmp_path: Path):
    from tools.dataset_delivery.task2_preflight import build_preflight

    case_root = tmp_path / "case"
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), case_root / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")
    output_root = tmp_path / "existing_out"
    output_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    report = build_preflight(
        models=[],
        case_list=case_csv,
        output_root=output_root,
        registry_path=REPO_ROOT / "configs" / "model_registry.yaml",
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root_arg=checkpoint_root,
        json_output=tmp_path / "preflight.json",
        canonical_code_root=REPO_ROOT,
        formal_mode=True,
        run_predictor_help=False,
    )

    assert report["status"] == "BLOCKED"
    assert any(check["name"] == "output_root_new" for check in report["blocked_checks"])


def test_run_loop_cli_passes_explicit_runtime_paths(tmp_path: Path, monkeypatch):
    from cli_anything.medai import medai_cli

    captured: dict[str, object] = {}

    def fake_loop(*args, **kwargs):
        captured.update(kwargs)
        return {"stage": "run_loop", "status": "success", "strict_delivery_failure_count": 0, "strict_delivery_failures": []}

    monkeypatch.setattr(medai_cli, "run_multimodel_annotation_loop", fake_loop)
    checkpoint_root = tmp_path / "checkpoints"
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    unest_python = _write_executable(tmp_path / "py311" / "python")
    result = CliRunner().invoke(
        medai_cli.cli,
        [
            "--json", "run-loop",
            "--case-list", str(tmp_path / "cases.csv"),
            "--models", "atm",
            "--organs", "airway_tree",
            "--output", str(tmp_path / "out"),
            "--checkpoint-root", str(checkpoint_root),
            "--nnunet-predict-executable", str(predictor),
            "--unest-python-executable", str(unest_python),
            "--dry-run",
            "--strict-delivery-targets",
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(captured["checkpoint_root"]) == str(checkpoint_root.resolve())
    assert str(captured["predict_executable"]) == str(predictor.resolve())
    assert str(captured["unest_python_executable"]) == str(unest_python.resolve())


def test_model_key_resolver_airrc_aliases_and_regressions():
    from cli_anything.medai.core.model_key_resolver import resolve_model_key
    from cli_anything.medai.core.model_registry import load_registry

    registry = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")

    for raw in ["AirRC", "AIRRC", "airrc", "air_rc", "Dataset1380_AirRC"]:
        assert resolve_model_key(raw, registry).resolved == "airrc"
    for raw, expected in {
        "ATM": "atm",
        "atm": "atm",
        "UNEST": "unest",
        "unest": "unest",
        "CADS553": "cads553",
        "CADS557": "cads557",
        "CADS559": "cads559",
    }.items():
        assert resolve_model_key(raw, registry).resolved == expected


def test_airrc_full_volume_execution_plan_uses_single_canonical_teacher():
    from cli_anything.medai.core.model_registry import load_registry
    from cli_anything.medai.core.multimodel_loop import _build_case_execution_plan

    registry = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")
    organs = ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"]
    plan = _build_case_execution_plan(
        registry=registry,
        project_root=REPO_ROOT,
        organs=organs,
        requested_models=["Dataset1380_AirRC"],
        preseeded_model_dirs=None,
        candidate_mode="route_pruned_with_competition",
    )

    assert plan["teacher_run_list"] == ["airrc"]
    assert list(plan["per_organ"]) == organs
    assert "hard_palate" not in plan["per_organ"]
    assert all(plan["per_organ"][organ]["primary_teacher"] == "airrc" for organ in organs)


def test_airrc_route_preflight_ready_with_canonical_run_list():
    from tools.dataset_delivery.task2_smoke_launcher import build_route_preflight

    report = build_route_preflight(group="airrc", registry_path=REPO_ROOT / "configs" / "model_registry.yaml")

    assert report["status"] == "READY"
    assert report["teacher_run_list"] == ["airrc"]
    for target in ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"]:
        row = report["targets"][target]
        assert row["primary_teacher_raw"] == "AirRC"
        assert row["primary_teacher_resolved"] == "airrc"
        assert row["registry_enabled"] is True
        assert row["route_eligible"] is True


def test_cads_route_preflight_uses_contract_source_labels():
    from tools.dataset_delivery.task2_smoke_launcher import build_route_preflight

    report = build_route_preflight(group="cads15", registry_path=REPO_ROOT / "configs" / "model_registry.yaml")

    assert report["status"] == "READY"
    assert report["teacher_run_list"] == ["cads557", "cads553", "cads559"]
    assert report["targets"]["cerebrospinal_fluid"]["source_label"] == "csf"
    assert report["targets"]["common_iliac_artery_left"]["source_label"] == "iliac_artery_left"
    assert report["targets"]["gland_structure"]["source_label"] == "glands"


def test_airrc_route_failure_validator_does_not_report_model_failed(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    case_id = "case_001"
    root = tmp_path / "smoke"
    run_out = root / "airrc" / "run_loop"
    plan_dir = run_out / "annotation_versions" / case_id
    plan_dir.mkdir(parents=True)
    (root / "airrc" / "selected_case_manifest.csv").write_text(
        f"case_id,ct_path,annotation_folder\n{case_id},{tmp_path / 'ct.nii.gz'},{tmp_path / 'ref'}\n",
        encoding="utf-8",
    )
    (run_out / "run_summary.json").write_text(
        json.dumps({
            "status": "ROUTE_RESOLUTION_FAILED",
            "strict_delivery_failure_count": 1,
            "strict_delivery_failures": [{"status": "failed", "reason": "route_resolution_failed"}],
        }),
        encoding="utf-8",
    )
    (plan_dir / "route_resolution_failure.json").write_text(json.dumps({"inference_called": False}), encoding="utf-8")
    slurm_csv = tmp_path / "slurm.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\nairrc,4440395,FAILED,2:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=root, groups=["airrc"], slurm_status_csv=slurm_csv)

    assert report["status"] == "ROUTE_RESOLUTION_FAILED"
    assert report["groups"][0]["status"] == "ROUTE_RESOLUTION_FAILED"
    assert report["groups"][0]["validation_started"] is False


def _write_smoke_run(root: Path, group: str, *, final_masks: bool = True) -> None:
    from tools.dataset_delivery.task2_smoke_validator import SMOKE_SPECS

    spec = SMOKE_SPECS[group]
    case_id = "case_001"
    group_root = root / group
    run_out = group_root / "run_loop"
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), group_root / "case" / "ct.nii.gz")
    (group_root / "selected_case_manifest.csv").write_text(
        f"case_id,ct_path,annotation_folder\n{case_id},{ct},{group_root / 'ref'}\n",
        encoding="utf-8",
    )
    plan = {
        "ct_path": str(ct),
        "teacher_run_list": spec["models"],
        "per_organ": {target: {"primary_teacher": spec["models"][0], "eligible_teachers": spec["models"]} for target in spec["targets"]},
    }
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    (run_out / "run_summary.json").write_text(
        json.dumps({"status": "success", "strict_delivery_failure_count": 0, "strict_delivery_failures": []}),
        encoding="utf-8",
    )
    for model in spec["models"]:
        summary = {"status": "success", "return_code": 0, "model_key": model}
        path = run_out / "cases" / case_id / "raw_predictions" / model / case_id / "inference_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    for target in spec["targets"]:
        destination = (
            run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"
            if final_masks
            else run_out / "cases" / case_id / "raw_predictions" / spec["models"][0] / case_id / "segmentations" / f"{target}.nii.gz"
        )
        _save(arr, destination)


def _write_unest_hpc_success_shape(root: Path, *, include_call_evidence: bool = True, include_artifact: bool = True) -> None:
    case_id = "case_001"
    group_root = root / "unest"
    run_out = group_root / "run_loop"
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), group_root / "case" / "ct.nii.gz")
    ref = group_root / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    (group_root / "selected_case_manifest.csv").write_text(
        f"case_id,ct_path,annotation_folder\n{case_id},{ct},{ref}\n",
        encoding="utf-8",
    )
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({
            "ct_path": str(ct),
            "teacher_run_list": ["unest"] if include_call_evidence else [],
            "per_organ": {
                target: {"primary_teacher": "unest", "eligible_teachers": ["unest"]}
                for target in ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"]
            },
        }),
        encoding="utf-8",
    )
    teacher_models = ["unest"] if include_call_evidence else []
    run_summary = {
        "status": "success",
        "teacher_inference_models": teacher_models,
        "teacher_inference_count": 1 if include_call_evidence else 0,
        "inference_success_count": 1 if include_call_evidence else 0,
        "candidate_generated_count": 3,
        "valid_candidate_count": 3,
        "strict_delivery_failure_count": 0,
        "strict_delivery_failures": [],
        "case_timing_breakdown": [{
            "case_id": case_id,
            "teacher_inference_models": teacher_models,
            "teacher_inference_count": 1 if include_call_evidence else 0,
        }],
        "final_delivery_counts": {
            "inference_success_count": 1 if include_call_evidence else 0,
            "candidate_generated_count": 3,
            "valid_candidate_count": 3,
        },
    }
    (run_out / "run_summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
    (run_out / "final_delivery_status.json").write_text(
        json.dumps({
            "status": "success",
            "rows": [
                {
                    "case_id": case_id,
                    "organ": target,
                    "final_status": "delivered_for_review",
                    "delivery_status": "delivered_for_review",
                }
                for target in ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"]
            ],
        }),
        encoding="utf-8",
    )
    if include_artifact:
        (run_out / "inference_results.json").write_text(
            json.dumps([{"case_id": case_id, "model_key": "unest", "status": "success"}]),
            encoding="utf-8",
        )
        manifest_path = run_out / "cases" / case_id / "hierarchical_inference_plan.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({
                "case_id": case_id,
                "roi_tasks": [{"inference": {"model_key": "unest", "status": "success"}}],
            }),
            encoding="utf-8",
        )
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    for target in ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"]:
        _save(arr, run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz")


def test_smoke_validator_requires_formal_updated_layer(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=False)
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"])

    assert report["status"] == "VALIDATION_FAILED"
    assert all(row["reason"] == "missing_mask" for row in report["target_rows"])


def test_smoke_validator_running_slurm_is_not_failed(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\ncads,123,RUNNING,0:0\n", encoding="utf-8")
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"], slurm_status_csv=slurm_csv)

    assert report["status"] == "RUNNING"
    assert report["failed_groups"] == []
    assert report["groups"][0]["validation_started"] is False
    assert report["target_rows"][0]["reason"] == "still_running"


def test_smoke_validator_passes_all_formal_cads_masks(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\ncads,123,COMPLETED,0:0\n", encoding="utf-8")
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"], slurm_status_csv=slurm_csv)

    assert report["status"] == "PASSED"
    assert len(report["groups"][0]["passed_targets"]) == 15


def test_unest_validator_accepts_hierarchical_run_level_success_without_summary(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_unest_hpc_success_shape(tmp_path)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\nunest,4440396,COMPLETED,0:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=tmp_path, groups=["unest"], slurm_status_csv=slurm_csv)

    assert report["status"] == "PASSED"
    check = report["groups"][0]["inference_checks"][0]
    assert check["ok"] is True
    assert check["summaries"] == []
    assert report["groups"][0]["passed_targets"] == [
        "kidney_cortex",
        "kidney_medulla",
        "kidney_pelvicalyceal_system",
    ]


def test_unest_validator_fails_without_inference_success_evidence_even_if_masks_valid(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_unest_hpc_success_shape(tmp_path, include_call_evidence=False, include_artifact=False)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\nunest,4440396,COMPLETED,0:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=tmp_path, groups=["unest"], slurm_status_csv=slurm_csv)

    assert report["status"] == "VALIDATION_FAILED"
    assert "INFERENCE_EVIDENCE_INCOMPLETE:unest" in report["groups"][0]["failures"]


def test_unest_validator_fails_when_model_never_called_with_valid_masks(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_unest_hpc_success_shape(tmp_path, include_call_evidence=False, include_artifact=True)

    report = validate_smoke_root(smoke_root=tmp_path, groups=["unest"])

    assert report["status"] == "VALIDATION_FAILED"
    assert "teacher_run_list_missing:unest" in report["groups"][0]["failures"]
    assert "INFERENCE_EVIDENCE_INCOMPLETE:unest" in report["groups"][0]["failures"]


def test_unest_validator_does_not_accept_stale_summary_without_run_call(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_unest_hpc_success_shape(tmp_path, include_call_evidence=False, include_artifact=False)
    summary_path = (
        tmp_path
        / "unest"
        / "run_loop"
        / "cases"
        / "case_001"
        / "raw_predictions"
        / "unest"
        / "case_001"
        / "inference_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"model_key": "unest", "status": "success", "return_code": 0}),
        encoding="utf-8",
    )

    report = validate_smoke_root(smoke_root=tmp_path, groups=["unest"])

    assert report["status"] == "VALIDATION_FAILED"
    assert "teacher_run_list_missing:unest" in report["groups"][0]["failures"]
    assert "INFERENCE_EVIDENCE_INCOMPLETE:unest" in report["groups"][0]["failures"]


def test_atm_and_airrc_inference_summary_checkers_still_pass(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "atm", final_masks=True)
    _write_smoke_run(tmp_path, "airrc", final_masks=True)

    report = validate_smoke_root(smoke_root=tmp_path, groups=["atm", "airrc"])

    assert report["status"] == "PASSED"
    assert all(check["ok"] for group in report["groups"] for check in group["inference_checks"])


def test_unest_python_resolver_preserves_venv_symlink(monkeypatch, tmp_path: Path):
    from cli_anything.medai.core.runtime_resolver import resolve_unest_python_details

    real_python = _write_executable(tmp_path / "system" / "python3.11")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    monkeypatch.setenv("UNEST_PYTHON_EXECUTABLE", str(venv_python))
    monkeypatch.delenv("MEDAI_UNEST_PYTHON", raising=False)

    details = resolve_unest_python_details()

    assert details.source == "env:UNEST_PYTHON_EXECUTABLE"
    assert details.execution_path == str(venv_python.absolute())
    assert details.real_path == str(real_python)


def test_unest_python_cli_and_env_priority(monkeypatch, tmp_path: Path):
    from cli_anything.medai.core.runtime_resolver import resolve_unest_python_details

    cli_python = _write_executable(tmp_path / "cli" / "python")
    env_python = _write_executable(tmp_path / "env" / "python")
    medai_python = _write_executable(tmp_path / "medai" / "python")
    monkeypatch.setenv("UNEST_PYTHON_EXECUTABLE", str(env_python))
    monkeypatch.setenv("MEDAI_UNEST_PYTHON", str(medai_python))

    cli = resolve_unest_python_details(explicit=cli_python)
    env = resolve_unest_python_details(explicit=None)
    monkeypatch.delenv("UNEST_PYTHON_EXECUTABLE")
    medai = resolve_unest_python_details(explicit=None)

    assert cli.source == "cli"
    assert cli.execution_path == str(cli_python.absolute())
    assert env.source == "env:UNEST_PYTHON_EXECUTABLE"
    assert env.execution_path == str(env_python.absolute())
    assert medai.source == "env:MEDAI_UNEST_PYTHON"
    assert medai.execution_path == str(medai_python.absolute())


def test_preflight_manifest_records_unest_execution_and_realpath(tmp_path: Path):
    from tools.dataset_delivery.task2_preflight import build_preflight

    real_python = _write_executable(tmp_path / "system" / "python3.11")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")

    report = build_preflight(
        models=[],
        case_list=case_csv,
        output_root=tmp_path / "out",
        registry_path=REPO_ROOT / "configs" / "model_registry.yaml",
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root_arg=tmp_path / "checkpoints",
        json_output=tmp_path / "preflight.json",
        canonical_code_root=REPO_ROOT,
        formal_mode=False,
        unest_python_executable=str(venv_python),
        run_predictor_help=False,
    )

    resolution = report["runtime_manifest"]["unest_python_resolution"]
    assert resolution["execution_path"] == str(venv_python.absolute())
    assert resolution["real_path"] == str(real_python)


def test_preflight_unest_smoke_uses_venv_execution_path_not_binary_realpath(tmp_path: Path, monkeypatch):
    from tools.dataset_delivery.task2_preflight import build_preflight

    log = tmp_path / "python_invocations.log"
    base_python = _write_executable(
        tmp_path / "base" / "python3.11",
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$0|$*\" >> {log}\n"
        "if [ \"$0\" != \"$EXPECTED_UNEST_PYTHON\" ]; then echo wrong-python >&2; exit 9; fi\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  echo '{\"sys_executable\":\"venv-python\",\"sys_prefix\":\"venv\",\"sys_base_prefix\":\"base\",\"monai_version\":\"1.4.0\",\"torch_version\":\"2.2.0+cu121\",\"cuda_available\":false,\"cuda_version\":\"12.1\"}'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"monai.bundle\" ] && [ \"$3\" = \"--help\" ]; then echo monai-help; exit 0; fi\n"
        "exit 0\n",
    )
    link_middle = tmp_path / "venv" / "bin" / "python3.11"
    venv_python = tmp_path / "venv" / "bin" / "python"
    link_middle.parent.mkdir(parents=True)
    link_middle.symlink_to(base_python)
    venv_python.symlink_to("python3.11")
    monkeypatch.setenv("EXPECTED_UNEST_PYTHON", str(venv_python.absolute()))
    checkpoint_root = tmp_path / "checkpoints"
    unest_root = checkpoint_root / "UNEST" / "renalStructures_UNEST_segmentation"
    for rel in ["models/model.pt", "configs/metadata.json", "configs/inference.json", "configs/logging.conf"]:
        path = unest_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
checkpoint_root: {checkpoint_root}
models:
  unest:
    enabled: true
    runner: command_template
    checkpoint_path: UNEST/renalStructures_UNEST_segmentation
    source_code_path: UNEST/renalStructures_UNEST_segmentation
    unest_python_executable: {base_python}
""",
        encoding="utf-8",
    )
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")

    report = build_preflight(
        models=["unest"],
        case_list=case_csv,
        output_root=tmp_path / "out",
        registry_path=registry,
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root_arg=checkpoint_root,
        json_output=tmp_path / "preflight.json",
        canonical_code_root=REPO_ROOT,
        formal_mode=True,
        outer_python=Path(sys.executable),
        unest_python_executable=str(venv_python),
        run_predictor_help=False,
        allow_dirty_tracked=True,
    )

    assert report["status"] == "READY"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["unest_import_monai_torch"]["command"][0] == str(venv_python.absolute())
    assert checks["unest_monai_bundle_help"]["command"][0] == str(venv_python.absolute())
    assert checks["unest_import_monai_torch"]["parsed"]["monai_version"] == "1.4.0"
    assert checks["unest_import_monai_torch"]["parsed"]["torch_version"] == "2.2.0+cu121"
    resolution = report["runtime_manifest"]["unest_python_resolution"]
    assert resolution["execution_path"] == str(venv_python.absolute())
    assert resolution["real_path"] == str(base_python)
    assert all(line.startswith(str(venv_python.absolute()) + "|") for line in log.read_text(encoding="utf-8").splitlines())


def test_outer_python_manifest_preserves_venv_execution_path(tmp_path: Path):
    from tools.dataset_delivery.task2_preflight import build_preflight

    real_python = _write_executable(tmp_path / "base" / "python3.11")
    outer_python = tmp_path / "outer" / "bin" / "python"
    outer_python.parent.mkdir(parents=True)
    outer_python.symlink_to(real_python)
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")

    report = build_preflight(
        models=[],
        case_list=case_csv,
        output_root=tmp_path / "out",
        registry_path=REPO_ROOT / "configs" / "model_registry.yaml",
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root_arg=checkpoint_root,
        json_output=tmp_path / "preflight.json",
        canonical_code_root=REPO_ROOT,
        formal_mode=False,
        outer_python=outer_python,
        run_predictor_help=False,
    )

    manifest = report["runtime_manifest"]
    assert manifest["outer_python"] == str(outer_python.absolute())
    assert manifest["outer_python_execution_path"] == str(outer_python.absolute())
    assert manifest["outer_python_binary_realpath"] == str(real_python)


def test_airrc_group_schema_and_parse_groups():
    from tools.dataset_delivery.task2_smoke_validator import SMOKE_SPECS, parse_groups

    assert parse_groups("atm,airrc,unest") == ["atm", "airrc", "unest"]
    assert parse_groups("totalsegmentator") == ["totalsegmentator"]
    assert SMOKE_SPECS["airrc"]["models"] == ["airrc"]
    assert SMOKE_SPECS["airrc"]["targets"] == [
        "airway_wall",
        "lung_pulmonary_arteries",
        "lung_pulmonary_veins",
    ]
    assert SMOKE_SPECS["totalsegmentator"]["models"] == ["totalsegmentator"]
    assert SMOKE_SPECS["totalsegmentator"]["targets"] == ["brain_ventricle"]


def test_airrc_smoke_validation_requires_three_valid_masks(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "airrc", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\nairrc,123,COMPLETED,0:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=tmp_path, groups=["airrc"], slurm_status_csv=slurm_csv)

    assert report["status"] == "PASSED"
    assert report["groups"][0]["passed_targets"] == [
        "airway_wall",
        "lung_pulmonary_arteries",
        "lung_pulmonary_veins",
    ]


def test_totalsegmentator_smoke_validation_requires_brain_ventricle_mask(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "totalsegmentator", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\ntotalsegmentator,123,COMPLETED,0:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=tmp_path, groups=["totalsegmentator"], slurm_status_csv=slurm_csv)

    assert report["status"] == "PASSED"
    assert report["groups"][0]["passed_targets"] == ["brain_ventricle"]


def test_airrc_missing_or_zero_mask_fails_after_completed_job(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "airrc", final_masks=True)
    zero = np.zeros((3, 3, 3), dtype=np.uint8)
    _save(zero, tmp_path / "airrc" / "run_loop" / "annotation_versions" / "case_001" / "updated" / "airway_wall.nii.gz")
    (tmp_path / "airrc" / "run_loop" / "annotation_versions" / "case_001" / "updated" / "lung_pulmonary_veins.nii.gz").unlink()
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\nairrc,123,COMPLETED,0:0\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=tmp_path, groups=["airrc"], slurm_status_csv=slurm_csv)

    reasons = {row["target"]: row["reason"] for row in report["target_rows"]}
    assert report["status"] == "VALIDATION_FAILED"
    assert reasons["airway_wall"] == "empty_mask"
    assert reasons["lung_pulmonary_veins"] == "missing_mask"


def test_task2_smoke_launcher_preserves_t4_gres_and_records_test_only(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import task2_smoke_launcher as launcher

    calls: list[list[str]] = []

    def fake_preflight(**kwargs):
        return {"status": "READY", "blocked_checks": []}

    def fake_run(command):
        calls.append(command)
        return {"command": command, "return_code": 0, "stdout": "test ok", "stderr": "", "ok": True}

    monkeypatch.setattr(launcher, "build_preflight", fake_preflight)
    monkeypatch.setattr(launcher, "_run_command", fake_run)
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")

    summary = launcher.prepare_smokes(
        smoke_root=tmp_path / "smoke",
        groups=["atm"],
        case_manifest=case_csv,
        case_id="case",
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        registry=REPO_ROOT / "configs" / "model_registry.yaml",
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root=tmp_path / "checkpoints",
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        unest_python_executable=_write_executable(tmp_path / "venv" / "python"),
        canonical_code_root=REPO_ROOT,
        formal_mode=False,
        timeout_sec=10,
        partition="gpu",
        gres="gpu:T4:1",
        cpus_per_task=8,
        mem="64G",
        time_limit="06:00:00",
        account="",
        run_slurm_test_only=True,
        skip_predictor_help=True,
    )

    sbatch_text = Path(summary["groups"][0]["sbatch_file"]).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:T4:1" in sbatch_text
    assert "gpu:t4:1" not in sbatch_text
    assert "MEDAI_TOTALSEG_HOME" in sbatch_text
    assert "TOTALSEG_HOME_DIR" in sbatch_text
    assert "MEDAI_TOTALSEG_OFFLINE=${MEDAI_TOTALSEG_OFFLINE:-1}" in sbatch_text
    assert "totalsegmentator_brain_ventricle_offline_manifest.json" in sbatch_text
    assert summary["groups"][0]["status"] == "READY_TO_SUBMIT"
    assert summary["groups"][0]["resources"]["configured_gres"] == "gpu:T4:1"
    assert any(command[:2] == ["sbatch", "--test-only"] for command in calls)


def test_task2_smoke_launcher_blocks_test_only_failure_before_submit(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import task2_smoke_launcher as launcher

    def fake_run(command):
        if command[:2] == ["sbatch", "--test-only"]:
            return {"command": command, "return_code": 1, "stdout": "", "stderr": "Requested node configuration is not available", "ok": False}
        return {"command": command, "return_code": 0, "stdout": "", "stderr": "", "ok": True}

    monkeypatch.setattr(launcher, "build_preflight", lambda **kwargs: {"status": "READY", "blocked_checks": []})
    monkeypatch.setattr(launcher, "_run_command", fake_run)
    ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    ref = tmp_path / "ref"
    ref.mkdir()
    case_csv = tmp_path / "cases.csv"
    case_csv.write_text(f"case_id,ct_path,annotation_folder\ncase,{ct},{ref}\n", encoding="utf-8")

    summary = launcher.prepare_smokes(
        smoke_root=tmp_path / "smoke",
        groups=["atm"],
        case_manifest=case_csv,
        case_id="case",
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        registry=REPO_ROOT / "configs" / "model_registry.yaml",
        target_config=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
        checkpoint_root=tmp_path / "checkpoints",
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        unest_python_executable=_write_executable(tmp_path / "venv" / "python"),
        canonical_code_root=REPO_ROOT,
        formal_mode=False,
        timeout_sec=10,
        partition="gpu",
        gres="gpu:t4:1",
        cpus_per_task=8,
        mem="64G",
        time_limit="06:00:00",
        account="",
        run_slurm_test_only=True,
        skip_predictor_help=True,
    )
    submitted = launcher.submit_prepared_smokes(summary=summary, submit_ready_groups=True, runtime_state_root=tmp_path / "state")

    assert summary["groups"][0]["status"] == "RESOURCE_REQUEST_INVALID"
    assert summary["groups"][0]["resources"]["sbatch_test_only_return_code"] == 1
    assert submitted["submitted_groups"] == []
    assert not (tmp_path / "state" / ".last_task2_teacher_smoke").exists()


def test_partial_ready_submit_submits_ready_group_and_keeps_blocked_unsubmitted(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import task2_smoke_launcher as launcher

    sbatch = tmp_path / "atm.sbatch"
    sbatch.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    summary = {
        "status": "BLOCKED_BY_PREFLIGHT",
        "smoke_root": str(tmp_path / "smoke"),
        "expected_commit": "abc123",
        "submitted_groups": [],
        "blocked_groups": ["unest"],
        "skipped_groups": [],
        "submission_rejected_groups": [],
        "groups": [
            {"group": "atm", "status": "READY_TO_SUBMIT", "sbatch_file": str(sbatch), "submission_status": "NOT_SUBMITTED"},
            {"group": "unest", "status": "BLOCKED_BY_PREFLIGHT", "sbatch_file": "", "submission_status": "NOT_SUBMITTED"},
        ],
    }
    monkeypatch.setattr(launcher, "_run_command", lambda command: {"command": command, "return_code": 0, "stdout": "999", "stderr": "", "ok": True})

    result = launcher.submit_prepared_smokes(summary=summary, submit_ready_groups=True, runtime_state_root=tmp_path / "state")

    assert result["status"] == "SUBMITTED"
    assert result["submitted_groups"] == ["atm"]
    assert result["groups"][0]["job_id"] == "999"
    assert result["groups"][1]["submission_status"] == "NOT_SUBMITTED"
    assert (tmp_path / "state" / ".last_task2_teacher_smoke").exists()


def test_all_or_nothing_blocked_group_skips_ready_without_pointer(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import task2_smoke_launcher as launcher

    summary = {
        "status": "BLOCKED_BY_PREFLIGHT",
        "smoke_root": str(tmp_path / "smoke"),
        "expected_commit": "abc123",
        "submitted_groups": [],
        "blocked_groups": ["unest"],
        "skipped_groups": [],
        "submission_rejected_groups": [],
        "groups": [
            {"group": "atm", "status": "READY_TO_SUBMIT", "sbatch_file": str(tmp_path / "atm.sbatch"), "submission_status": "NOT_SUBMITTED"},
            {"group": "unest", "status": "BLOCKED_BY_PREFLIGHT", "sbatch_file": "", "submission_status": "NOT_SUBMITTED"},
        ],
    }
    calls = []
    monkeypatch.setattr(launcher, "_run_command", lambda command: calls.append(command) or {"command": command, "return_code": 0, "stdout": "999", "stderr": "", "ok": True})

    result = launcher.submit_prepared_smokes(summary=summary, submit_ready_groups=False, runtime_state_root=tmp_path / "state")

    assert result["submitted_groups"] == []
    assert result["skipped_groups"] == ["atm"]
    assert calls == []
    assert not (tmp_path / "state" / ".last_task2_teacher_smoke").exists()


def test_submission_rejected_does_not_create_pointer(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import task2_smoke_launcher as launcher

    summary = {
        "status": "READY_TO_SUBMIT",
        "smoke_root": str(tmp_path / "smoke"),
        "expected_commit": "abc123",
        "submitted_groups": [],
        "blocked_groups": [],
        "skipped_groups": [],
        "submission_rejected_groups": [],
        "groups": [{"group": "atm", "status": "READY_TO_SUBMIT", "sbatch_file": str(tmp_path / "atm.sbatch"), "submission_status": "NOT_SUBMITTED"}],
    }
    monkeypatch.setattr(launcher, "_run_command", lambda command: {"command": command, "return_code": 1, "stdout": "", "stderr": "reject", "ok": False})

    result = launcher.submit_prepared_smokes(summary=summary, submit_ready_groups=True, runtime_state_root=tmp_path / "state")

    assert result["status"] == "SUBMISSION_REJECTED"
    assert result["submission_rejected_groups"] == ["atm"]
    assert not (tmp_path / "state" / ".last_task2_teacher_smoke").exists()


def test_submission_manifest_blocked_group_does_not_trigger_mask_validation(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"groups": [{"group": "unest", "status": "BLOCKED_BY_PREFLIGHT", "case_id": "case_001"}]}),
        encoding="utf-8",
    )

    report = validate_smoke_root(smoke_root=root, groups=["unest"])

    assert report["status"] == "BLOCKED_BY_PREFLIGHT"
    assert report["failed_groups"] == []
    assert report["target_rows"][0]["reason"] == "blocked_by_preflight"


def test_submitted_without_terminal_slurm_state_does_not_trigger_mask_validation(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"groups": [{"group": "atm", "status": "SUBMITTED", "case_id": "case_001", "job_id": "123"}]}),
        encoding="utf-8",
    )
    slurm_csv = root / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\natm,123,SUBMITTED,\n", encoding="utf-8")

    report = validate_smoke_root(smoke_root=root, groups=["atm"], slurm_status_csv=slurm_csv)

    assert report["status"] == "PENDING"
    assert report["failed_groups"] == []
    assert report["target_rows"][0]["reason"] == "submitted_not_terminal"


def test_submit_and_check_scripts_use_task2_smoke_groups_not_bash_groups():
    submit_script = Path("scripts/task2/submit_teacher_smokes.sh").read_text(encoding="utf-8")
    check_script = Path("scripts/task2/check_teacher_smokes.sh").read_text(encoding="utf-8")

    assert "TASK2_SMOKE_GROUPS" in submit_script
    assert "TASK2_SMOKE_GROUPS" in check_script
    assert "GROUPS=${GROUPS" not in submit_script
    assert "GROUPS=${GROUPS" not in check_script
    assert "--groups \"$TASK2_SMOKE_GROUPS\"" in submit_script
    assert "--groups \"$TASK2_SMOKE_GROUPS\"" in check_script
