from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


CADS15_TARGETS = [
    "blood",
    "cerebrospinal_fluid",
    "common_iliac_artery_left",
    "common_iliac_artery_right",
    "common_iliac_vein_left",
    "common_iliac_vein_right",
    "compact_bone",
    "eyeball",
    "face",
    "gland_structure",
    "gray_matter",
    "muscle_of_head",
    "scalp",
    "spongy_bone",
    "white_matter",
]


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_manifest(tmp_path: Path, count: int = 2) -> Path:
    lines = ["case_id,ct_path,annotation_folder"]
    for idx in range(count):
        case_id = f"case_{idx:03d}"
        ct = _save(np.zeros((2, 2, 2), dtype=np.int16), tmp_path / case_id / "ct.nii.gz")
        ref = tmp_path / case_id / "ref"
        ref.mkdir()
        lines.append(f"{case_id},{ct},{ref}")
    manifest = tmp_path / "cases.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _write_smoke_pass(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task2_smoke_verdict.json").write_text(
        json.dumps({
            "status": "passed",
            "cads15_summary": {
                "CADS15_SMOKE_STATUS": "PASSED",
                "TARGETS_WITH_POSITIVE_SMOKE": 15,
                "TARGETS_FAILED": 0,
                "STRICT_DELIVERY_FAILURE_COUNT": 0,
            },
        }),
        encoding="utf-8",
    )


def test_submit_script_does_not_run_heavy_panel_on_login_node():
    script = Path("scripts/task2/submit_cads15_smoke.sh").read_text(encoding="utf-8")

    assert "--prepare-orchestration" in script
    assert "sbatch --parsable \"$PANEL_SBATCH\"" in script
    assert "cads15_smoke_panel.py" not in script
    assert "--dependency=afterok:" in script


def test_orchestration_writes_cpu_panel_and_dependent_gpu_sbatch(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_launcher import prepare_cads15_orchestration

    smoke_root = tmp_path / "smoke"
    manifest = _write_manifest(tmp_path)
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    summary = prepare_cads15_orchestration(
        smoke_root=smoke_root,
        case_manifest=manifest,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        timeout_sec=10,
        panel_partition="shared",
        panel_cpus_per_task=2,
        panel_mem="4G",
        panel_time_limit="00:30:00",
        gpu_partition="gpu",
        gpu_gres="gpu:T4:1",
        gpu_cpus_per_task=4,
        gpu_mem="8G",
        gpu_time_limit="01:00:00",
        allow_heavy_ct_fov=True,
        require_clean_tracked=False,
    )

    panel = smoke_root / "slurm" / "cads15_panel_prepare.sbatch"
    gpu = smoke_root / "slurm" / "cads15_gpu_smoke.sbatch"
    assert summary["status"] == "NOT_SUBMITTED"
    assert panel.exists()
    assert gpu.exists()
    assert "cads15_smoke_panel.py" in panel.read_text(encoding="utf-8")
    assert "--allow-heavy-ct-fov" in panel.read_text(encoding="utf-8")
    assert "--execute-panel" in gpu.read_text(encoding="utf-8")


def test_default_cads15_panel_partition_is_cpu_and_has_no_gres(monkeypatch):
    from tools.dataset_delivery.cads15_slurm_resources import resolve_gpu_profile, resolve_panel_profile

    monkeypatch.delenv("TASK2_CPU_PARTITION", raising=False)
    panel = resolve_panel_profile()
    gpu = resolve_gpu_profile()

    assert panel.partition == "cpu"
    assert panel.gres == ""
    assert gpu.partition == "gpu"
    assert gpu.gres == "gpu:T4:1"


def test_task2_cpu_partition_env_overrides_default(monkeypatch):
    from tools.dataset_delivery.cads15_slurm_resources import resolve_panel_profile

    monkeypatch.setenv("TASK2_CPU_PARTITION", "interactive")
    panel = resolve_panel_profile()

    assert panel.partition == "interactive"
    assert panel.partition_source == "env:TASK2_CPU_PARTITION"


def test_cads15_orchestration_blocks_missing_panel_partition(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_launcher as launcher

    monkeypatch.setattr(
        launcher,
        "available_partitions",
        lambda: (["cpu", "gpu", "gpua100", "gpuh100", "interactive"], {"ok": True, "return_code": 0}),
    )
    smoke_root = tmp_path / "smoke"
    manifest = _write_manifest(tmp_path)
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")

    summary = launcher.prepare_cads15_orchestration(
        smoke_root=smoke_root,
        case_manifest=manifest,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        timeout_sec=10,
        panel_partition="shared",
        panel_cpus_per_task=2,
        panel_mem="4G",
        panel_time_limit="00:30:00",
        gpu_partition="gpu",
        gpu_gres="gpu:T4:1",
        gpu_cpus_per_task=4,
        gpu_mem="8G",
        gpu_time_limit="01:00:00",
        allow_heavy_ct_fov=True,
        require_clean_tracked=False,
    )

    assert summary["status"] == "PANEL_RESOURCE_INVALID"
    preflight = summary["resource_preflight"]["panel"]
    assert preflight["reason"] == "partition_not_found:shared"
    assert preflight["available_partitions"] == ["cpu", "gpu", "gpua100", "gpuh100", "interactive"]
    assert summary["gpu"]["job_ids"] == []
    assert "SBATCH --gres" not in (smoke_root / "slurm" / "cads15_panel_prepare.sbatch").read_text(encoding="utf-8")


def test_smoke_status_pending_and_running_are_not_failed(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"panel": {"job_id": "1"}, "gpu": {"job_ids": ["2"]}}),
        encoding="utf-8",
    )
    (root / "panel_state.json").write_text(json.dumps({"status": "PANEL_RUNNING"}), encoding="utf-8")

    status = smoke_status(smoke_root=root, use_sacct=False)

    assert status["status"] == "PANEL_RUNNING"
    assert status["gpu_smoke_started"] is True


def test_panel_failed_blocks_gpu_success_state(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"panel": {"job_id": "1"}, "gpu": {"job_ids": ["2"]}}),
        encoding="utf-8",
    )
    (root / "PANEL_FAILED").touch()

    assert smoke_status(smoke_root=root, use_sacct=False)["status"] == "PANEL_FAILED"


def test_panel_completed_then_gpu_pending(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"panel": {"job_id": "1"}, "gpu": {"job_ids": ["2"]}}),
        encoding="utf-8",
    )
    (root / "PANEL_COMPLETED").touch()

    assert smoke_status(smoke_root=root, use_sacct=False)["status"] == "GPU_SMOKE_PENDING"


def test_manifest_without_real_job_id_stays_not_submitted(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({
            "status": "NOT_SUBMITTED",
            "panel": {"job_id": "", "state": "NOT_SUBMITTED"},
            "gpu": {"job_ids": [], "state": "NOT_SUBMITTED"},
        }),
        encoding="utf-8",
    )
    (root / "panel_state.json").write_text(json.dumps({"status": "NOT_SUBMITTED"}), encoding="utf-8")

    status = smoke_status(smoke_root=root, use_sacct=False)

    assert status["status"] == "NOT_SUBMITTED"
    assert status["gpu_smoke_started"] is False


def test_gpu_submission_rejected_is_specific_status(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({
            "status": "GPU_SUBMISSION_REJECTED",
            "panel": {"job_id": "123", "state": "PANEL_PENDING"},
            "gpu": {"job_ids": [], "state": "GPU_SUBMISSION_REJECTED"},
        }),
        encoding="utf-8",
    )

    assert smoke_status(smoke_root=root, use_sacct=False)["status"] == "GPU_SUBMISSION_REJECTED"


def test_empty_sbatch_script_is_rejected(tmp_path: Path):
    from tools.dataset_delivery.cads15_slurm_resources import SlurmResourceProfile, preflight_sbatch_script

    sbatch = tmp_path / "empty.sbatch"
    sbatch.write_text("", encoding="utf-8")
    profile = SlurmResourceProfile(
        stage="panel_preparation",
        partition="cpu",
        partition_source="test",
        cpus_per_task=1,
        cpus_source="test",
        memory="1G",
        memory_source="test",
        time_limit="00:01:00",
        time_source="test",
    )

    result = preflight_sbatch_script(
        profile=profile,
        sbatch_file=sbatch,
        available=["cpu"],
        available_query={"ok": True},
    )

    assert result["status"] == "PANEL_SCRIPT_INVALID"
    assert result["reason"] == "empty_sbatch_script"
    assert result["sbatch_test_only"]["skipped"] is True


def test_malformed_panel_job_id_is_invalid_status(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_status import smoke_status

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"status": "SUBMITTED", "panel": {"job_id": "abc"}, "gpu": {"job_ids": []}}),
        encoding="utf-8",
    )

    status = smoke_status(smoke_root=root, use_sacct=False)

    assert status["status"] == "PANEL_JOB_ID_INVALID"
    assert status["reason"] == "malformed_job_id"


def test_dependency_never_satisfied_is_specific_gpu_status(tmp_path: Path, monkeypatch):
    from tools.dataset_delivery import cads15_smoke_status as status_mod

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "submission_manifest.json").write_text(
        json.dumps({"panel": {"job_id": "1"}, "gpu": {"job_ids": ["2"]}}),
        encoding="utf-8",
    )
    (root / "PANEL_COMPLETED").touch()

    def fake_job_state(job_id: str, *, use_sacct: bool):
        if job_id == "2":
            return {"known": "true", "state": "DependencyNeverSatisfied", "exit_code": "0:0"}
        return {"known": "true", "state": "COMPLETED", "exit_code": "0:0"}

    monkeypatch.setattr(status_mod, "_job_state", fake_job_state)

    assert status_mod.smoke_status(smoke_root=root, use_sacct=True)["status"] == "GPU_DEPENDENCY_UNSATISFIED"


def test_panel_cache_reuses_completed_cases_without_reloading_reference(monkeypatch, tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_panel as panel

    manifest = _write_manifest(tmp_path, count=1)
    case_ref = tmp_path / "case_000" / "ref"
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 1
    for target in CADS15_TARGETS:
        _save(arr, case_ref / f"{target}.nii.gz")

    first = panel.build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")
    assert first["status"] == "READY_FOR_HPC_SMOKE"

    def fail_mask_positive(*args, **kwargs):
        raise AssertionError("cache was not reused")

    monkeypatch.setattr(panel, "_mask_positive", fail_mask_positive)
    second = panel.build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")

    assert second["status"] == "READY_FOR_HPC_SMOKE"
    assert json.loads((tmp_path / "panel" / "panel_progress.json").read_text())["panel_cases_reused_from_cache"] == 1


def test_ct_fingerprint_change_invalidates_only_case_cache(tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_panel as panel

    manifest = _write_manifest(tmp_path, count=1)
    case_ref = tmp_path / "case_000" / "ref"
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 1
    _save(arr, case_ref / "blood.nii.gz")
    panel.build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")
    os.utime(tmp_path / "case_000" / "ct.nii.gz", None)
    panel.build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")

    assert json.loads((tmp_path / "panel" / "panel_progress.json").read_text())["panel_cases_reused_from_cache"] == 0


def test_panel_missing_reference_with_fov_compatible_case_is_eligible(tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_panel as panel

    manifest = _write_manifest(tmp_path, count=1)
    (tmp_path / "case_000" / "scan_coverage.json").write_text(
        json.dumps({"scan_coverage": ["multi_region"], "coverage_regions": ["multi_region"]}),
        encoding="utf-8",
    )

    result = panel.build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")

    assert result["status"] == "READY_FOR_HPC_SMOKE"
    assert all(row["eligible"] for row in result["audit_rows"])
    assert {row["reason"] for row in result["audit_rows"]} == {"FOV_COMPATIBLE_NO_REFERENCE"}


def test_panel_uses_historical_teacher_positive_as_selection_evidence(tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_panel as panel

    manifest = _write_manifest(tmp_path, count=1)
    (tmp_path / "case_000" / "scan_coverage.json").write_text(
        json.dumps({"scan_coverage": ["multi_region"], "coverage_regions": ["multi_region"]}),
        encoding="utf-8",
    )
    ct = tmp_path / "case_000" / "ct.nii.gz"
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 1
    _save(
        arr,
        tmp_path / "old_run" / "annotation_versions" / "case_000" / "updated" / "blood.nii.gz",
    )

    result = panel.build_smoke_panel(
        case_manifest=manifest,
        output_root=tmp_path / "panel",
        historical_teacher_roots=[tmp_path / "old_run"],
    )

    blood = [row for row in result["audit_rows"] if row["target"] == "blood"][0]
    assert ct.exists()
    assert blood["eligible"] is True
    assert blood["selection_evidence"] == "historical_teacher_positive"
    assert blood["reason"] == "historical_teacher_positive"


def test_panel_rejects_historical_teacher_positive_when_current_fov_is_out(tmp_path: Path):
    from tools.dataset_delivery import cads15_smoke_panel as panel

    manifest = _write_manifest(tmp_path, count=1)
    (tmp_path / "case_000" / "scan_coverage.json").write_text(
        json.dumps({"scan_coverage": ["abdomen"], "coverage_regions": ["abdomen"]}),
        encoding="utf-8",
    )
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 1
    _save(
        arr,
        tmp_path / "old_run" / "annotation_versions" / "case_000" / "updated" / "gray_matter.nii.gz",
    )

    result = panel.build_smoke_panel(
        case_manifest=manifest,
        output_root=tmp_path / "panel",
        historical_teacher_roots=[tmp_path / "old_run"],
    )

    gray = [row for row in result["audit_rows"] if row["target"] == "gray_matter"][0]
    assert gray["eligible"] is False
    assert gray["fov_status"] == "out_of_fov"
    assert gray["selection_evidence"] == "historical_teacher_positive_rejected_by_current_fov"


def test_formal_launcher_uses_case_by_model_not_case_by_target(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_launcher import build_formal_plan

    manifest = _write_manifest(tmp_path, count=2)
    smoke = tmp_path / "smoke"
    _write_smoke_pass(smoke)
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")

    summary = build_formal_plan(
        output_root=tmp_path / "formal",
        case_manifest=manifest,
        smoke_root=smoke,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        expected_case_count=2,
        dry_run=True,
        allow_dirty_tracked=True,
    )

    assert summary["status"] == "READY"
    assert summary["task_count"] == 6
    rows = list(__import__("csv").DictReader((tmp_path / "formal" / "cads15_model_task_manifest.csv").open()))
    assert {(row["case_id"], row["model"]) for row in rows} == {
        ("case_000", "cads553"), ("case_000", "cads557"), ("case_000", "cads559"),
        ("case_001", "cads553"), ("case_001", "cads557"), ("case_001", "cads559"),
    }


def test_formal_launcher_blocks_without_passed_smoke(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_launcher import build_formal_plan

    manifest = _write_manifest(tmp_path, count=1)
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    summary = build_formal_plan(
        output_root=tmp_path / "formal",
        case_manifest=manifest,
        smoke_root=tmp_path / "missing_smoke",
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        expected_case_count=1,
        dry_run=True,
        allow_dirty_tracked=True,
    )

    assert summary["status"] == "BLOCKED"
    assert any(check["name"] == "smoke_passed" for check in summary["blocked_checks"])


def test_formal_resume_skips_completed_and_retry_failed_selects_failures(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_launcher import build_formal_plan

    manifest = _write_manifest(tmp_path, count=1)
    smoke = tmp_path / "smoke"
    _write_smoke_pass(smoke)
    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    root = tmp_path / "formal"
    completed = root / "tasks" / "case_000" / "cads553"
    completed.mkdir(parents=True)
    contract_hash = __import__("hashlib").sha256(Path("configs/cads15_target_contract.json").read_bytes()).hexdigest()
    (completed / "task_state.json").write_text(json.dumps({"status": "completed", "contract_sha256": contract_hash}), encoding="utf-8")
    failed = root / "tasks" / "case_000" / "cads557"
    failed.mkdir(parents=True)
    (failed / "task_state.json").write_text(json.dumps({"status": "failed", "contract_sha256": contract_hash}), encoding="utf-8")

    resume = build_formal_plan(
        output_root=root,
        case_manifest=manifest,
        smoke_root=smoke,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        expected_case_count=1,
        resume=True,
        dry_run=True,
        allow_dirty_tracked=True,
    )
    retry = build_formal_plan(
        output_root=root,
        case_manifest=manifest,
        smoke_root=smoke,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=predictor,
        expected_case_count=1,
        resume=True,
        retry_failed=True,
        dry_run=True,
        allow_dirty_tracked=True,
    )

    assert resume["task_count"] == 2
    assert retry["task_count"] == 1


def test_formal_blocks_existing_output_root_without_resume(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_launcher import build_formal_plan

    root = tmp_path / "formal"
    root.mkdir()
    summary = build_formal_plan(
        output_root=root,
        case_manifest=_write_manifest(tmp_path, count=1),
        smoke_root=tmp_path / "smoke",
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=REPO_ROOT / "checkpoints",
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        expected_case_count=1,
        allow_dirty_tracked=True,
    )

    assert summary["status"] == "BLOCKED"
    assert summary["task_manifest"] == ""


def test_formal_blocks_missing_checkpoint_and_predictor(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_launcher import build_formal_plan

    smoke = tmp_path / "smoke"
    _write_smoke_pass(smoke)
    summary = build_formal_plan(
        output_root=tmp_path / "formal",
        case_manifest=_write_manifest(tmp_path, count=1),
        smoke_root=smoke,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=tmp_path / "missing_checkpoints",
        nnunet_predict_executable=tmp_path / "missing_predictor",
        expected_case_count=1,
        dry_run=True,
        allow_dirty_tracked=True,
    )

    assert summary["status"] == "BLOCKED"
    names = {check["name"] for check in summary["blocked_checks"]}
    assert {"checkpoint_root_exists", "predictor_executable"} <= names


def test_formal_status_matrix_has_100_by_15_rows_and_zero_mask_fails(tmp_path: Path):
    from tools.dataset_delivery.cads15_formal_validator import build_status_matrix

    manifest = _write_manifest(tmp_path, count=100)
    root = tmp_path / "formal"
    case_id = "case_000"
    run_out = root / "cases" / case_id / "cads557" / "run_loop"
    zero = np.zeros((2, 2, 2), dtype=np.uint8)
    _save(zero, run_out / "annotation_versions" / case_id / "updated" / "blood.nii.gz")
    (run_out / "final_delivery_status.json").write_text(
        json.dumps({
            "rows": [{
                "case_id": case_id,
                "organ": "blood",
                "final_status": "delivered",
                "delivery_status": "delivered",
                "fov_status": "fully_visible",
            }]
        }),
        encoding="utf-8",
    )

    report = build_status_matrix(output_root=root, case_manifest=manifest)

    assert report["row_count"] == 1500
    blood = next(row for row in report["rows"] if row["case_id"] == case_id and row["canonical_target"] == "blood")
    assert blood["delivery_status"] == "failed"
    assert blood["failure_reason"] == "empty_mask"
