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


def test_smoke_validator_requires_formal_updated_layer(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=False)
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"])

    assert report["status"] == "failed"
    assert all(row["reason"] == "missing_mask" for row in report["target_rows"])


def test_smoke_validator_running_slurm_is_not_success(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\ncads,123,RUNNING,0:0\n", encoding="utf-8")
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"], slurm_status_csv=slurm_csv)

    assert report["status"] == "failed"
    assert report["groups"][0]["slurm"]["reason"] == "slurm_job_not_terminal"


def test_smoke_validator_passes_all_formal_cads_masks(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    _write_smoke_run(tmp_path, "cads", final_masks=True)
    slurm_csv = tmp_path / "slurm_status.csv"
    slurm_csv.write_text("group,job_id,state,exit_code\ncads,123,COMPLETED,0:0\n", encoding="utf-8")
    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads"], slurm_status_csv=slurm_csv)

    assert report["status"] == "passed"
    assert len(report["groups"][0]["passed_targets"]) == 15
