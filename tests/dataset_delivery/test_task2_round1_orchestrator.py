from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.dataset_delivery import task2_round1_orchestrator as orch


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state_root=tmp_path / "state",
        workspace_root=tmp_path / "workspace",
        case_manifest=tmp_path / "cases.csv",
        base_manifest=tmp_path / "base.csv",
        python=Path("python"),
        checkpoint_root=tmp_path / "checkpoints",
        nnunet_predict_executable=tmp_path / "nnUNetv2_predict",
        unest_python_executable=tmp_path / "unest_python",
        registry=tmp_path / "model_registry.yaml",
        target_config=tmp_path / "targets.json",
        gpu_target_workers=30,
        gpu_overrequest_workers=40,
        gpu_profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        poll_sec=1,
        controller_partition="cpu",
        controller_cpus=2,
        controller_mem="8G",
        controller_time="48:00:00",
        labelcritic_partition="gpuh100",
        labelcritic_gres="gpu:H100:2",
        labelcritic_tensor_parallel_size=2,
        labelcritic_port=8000,
        skip_sbatch_test_only=True,
        run_static_tests=False,
        static_tests_timeout_sec=1,
        retry_failed=True,
        new_attempt=False,
    )


def test_existing_pending_labelcritic_not_resubmitted(tmp_path, monkeypatch):
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("4527571\n", encoding="utf-8")
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "PENDING", "job_id": job_id})
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")

    def fail_submit(*args, **kwargs):
        raise AssertionError("submit_labelcritic_72b_service.sh must not be called")

    monkeypatch.setattr(orch, "_run", fail_submit)
    result = orch.ensure_labelcritic_service(tmp_path)
    assert result == {"status": "REUSED_ACTIVE_JOB", "job_id": "4527571"}


def test_pending_labelcritic_waits_until_runtime_ready(tmp_path, monkeypatch):
    services = iter([
        {"status": "REUSED_ACTIVE_JOB", "job_id": "4527571"},
        {"status": "REUSED_HEALTHY", "base_url": "http://node1", "port": 8000},
    ])
    sleeps: list[int] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: next(services))
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "PENDING", "job_id": job_id})
    monkeypatch.setattr(orch.time, "sleep", lambda sec: sleeps.append(sec))
    monkeypatch.setattr(orch, "labelcritic_runtime_preflight", lambda base, port: {"status": "PASSED"})

    result = orch.wait_for_labelcritic_runtime(tmp_path, poll_sec=1)
    assert result["status"] == "PASSED"
    assert sleeps


def test_pending_labelcritic_submits_estep_before_runtime_ready(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: calls.append("ensure_labelcritic") or {"status": "REUSED_ACTIVE_JOB", "job_id": "4527698"})

    poll_results = iter([
        {"status": "WAITING", "job": {"state": "PENDING"}, "service": {"status": "REUSED_ACTIVE_JOB", "job_id": "4527698"}},
        {"status": "PASSED", "base_url": "http://node", "port": 8000, "runtime": {"status": "PASSED"}},
    ])
    estep_results = iter([
        {"status": "RUNNING"},
        {"status": "PASSED"},
    ])
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or next(poll_results))
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic=None: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "check_estep", lambda args: calls.append("check_estep") or next(estep_results))
    monkeypatch.setattr(orch, "submit_mstep", lambda args: calls.append("submit_mstep") or {"status": "SUBMITTED", "job_id": "99"})
    monkeypatch.setattr(orch, "check_mstep", lambda args: calls.append("check_mstep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "run_round1_final_validator", lambda args: calls.append("final_validator") or {"status": "PASSED", "terminal_state": "ROUND1_PASSED"})
    monkeypatch.setattr(orch.time, "sleep", lambda sec: calls.append("sleep"))

    assert orch.controller(args) == 0
    assert calls.index("submit_estep") < calls.index("poll_labelcritic")
    assert calls.index("submit_mstep") > calls.index("check_estep")


def test_service_runtime_validation_failure_blocks_mstep_not_estep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "4527571"})
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic=None: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: {"status": "FAILED", "failure_reason": "bad_runtime"})

    def fail_mstep(*args, **kwargs):
        raise AssertionError("M-step must not launch after LabelCritic runtime failure")

    monkeypatch.setattr(orch, "submit_mstep", fail_mstep)
    assert orch.controller(args) == 2
    state = orch._load_state(args.state_root)
    assert state["terminal_state"] == "ROUND1_FAILED"
    assert state["stage"] == "labelcritic"
    assert calls == ["submit_estep"]
    paths = orch._state_paths(args.state_root)
    assert paths["last_failure"].is_file()
    assert paths["failures"].is_file()
    status = orch.status(SimpleNamespace(state_root=args.state_root))
    assert status["failure_reason"] == "bad_runtime"
    assert status["last_failure"]["stage"] == "labelcritic"
    assert status["failure_log"] == str(paths["failures"])


def test_estep_passed_releases_mstep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_HEALTHY", "base_url": "http://node", "port": 8000})
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or {"status": "PASSED", "base_url": "http://node", "port": 8000})
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "check_estep", lambda args: calls.append("check_estep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "submit_mstep", lambda args: calls.append("submit_mstep") or {"status": "SUBMITTED", "job_id": "99"})
    monkeypatch.setattr(orch, "check_mstep", lambda args: calls.append("check_mstep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "run_round1_final_validator", lambda args: calls.append("final_validator") or {"status": "PASSED", "terminal_state": "ROUND1_PASSED"})

    assert orch.controller(args) == 0
    assert calls == ["submit_estep", "poll_labelcritic", "check_estep", "submit_mstep", "check_mstep", "final_validator"]
    assert orch._load_state(args.state_root)["terminal_state"] == "ROUND1_PASSED"


def test_estep_passed_waits_for_labelcritic_gate_before_mstep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "4527698"})
    poll_results = iter([
        {"status": "WAITING", "job": {"state": "RUNNING"}},
        {"status": "PASSED", "base_url": "http://node", "port": 8000},
    ])
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or next(poll_results))
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "check_estep", lambda args: calls.append("check_estep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "submit_mstep", lambda args: calls.append("submit_mstep") or {"status": "SUBMITTED", "job_id": "99"})
    monkeypatch.setattr(orch, "check_mstep", lambda args: calls.append("check_mstep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "run_round1_final_validator", lambda args: calls.append("final_validator") or {"status": "PASSED", "terminal_state": "ROUND1_PASSED"})
    monkeypatch.setattr(orch.time, "sleep", lambda sec: calls.append("sleep"))

    assert orch.controller(args) == 0
    assert calls == [
        "submit_estep",
        "poll_labelcritic",
        "check_estep",
        "sleep",
        "poll_labelcritic",
        "check_estep",
        "submit_mstep",
        "check_mstep",
        "final_validator",
    ]


def test_submit_controller_exits_after_sbatch_submission(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    paths["root"].mkdir(parents=True)
    paths["controller_sbatch"].write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    def fake_preflight(args):
        new_paths = orch._state_paths(args.state_root)
        new_paths["root"].mkdir(parents=True, exist_ok=True)
        new_paths["controller_sbatch"].write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        return {"status": "PASSED"}

    monkeypatch.setattr(orch, "run_static_preflight", fake_preflight)
    submitted: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["sbatch", "--parsable"]:
            submitted.append(command)
            return {"ok": True, "stdout": "777\n", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_controller(args)
    assert result["status"] == "CONTROLLER_SUBMITTED"
    assert result["controller_job_id"] == "777"
    assert len(submitted) == 1


def test_resume_does_not_resubmit_estep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    (formal_root / "slurm").mkdir(parents=True)
    (formal_root / "slurm" / "submitted_jobs.csv").write_text("job_id\n123\n", encoding="utf-8")
    orch._save_state(args.state_root, e_step_status="SUBMITTED", formal_root=str(formal_root))

    def fail_run(*args, **kwargs):
        raise AssertionError("formal E-step submitter must not be called on resume")

    monkeypatch.setattr(orch, "_run", fail_run)
    result = orch.submit_estep(args, {"base_url": "http://node", "port": 8000})
    assert result["status"] == "REUSED"


def test_terminal_labelcritic_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "4527571"})
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "FAILED", "job_id": job_id})
    monkeypatch.setattr(orch.time, "sleep", lambda sec: pytest.fail("terminal failure must not keep waiting"))
    result = orch.wait_for_labelcritic_runtime(tmp_path, poll_sec=1)
    assert result["status"] == "FAILED"
    assert result["failure_reason"] == "labelcritic_job_terminal:FAILED"


def test_retry_failed_archives_existing_attempt_and_submits_new_controller(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    orch._save_state(args.state_root, terminal_state="ROUND1_FAILED", failure_reason="old_failure")
    paths["root"].mkdir(parents=True, exist_ok=True)
    def fake_preflight(args):
        new_paths = orch._state_paths(args.state_root)
        new_paths["root"].mkdir(parents=True, exist_ok=True)
        new_paths["controller_sbatch"].write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        return {"status": "PASSED"}

    monkeypatch.setattr(orch, "run_static_preflight", fake_preflight)
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "COMPLETED", "job_id": job_id})

    def fake_run(command, **kwargs):
        if command[:2] == ["sbatch", "--parsable"]:
            return {"ok": True, "stdout": "888\n", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_controller(args)
    assert result["status"] == "CONTROLLER_SUBMITTED"
    archived = list((args.state_root / "round1_orchestrated_attempts").glob("attempt_001/attempt_archive.json"))
    assert archived
    assert orch._load_state(args.state_root)["controller_job_id"] == "888"
