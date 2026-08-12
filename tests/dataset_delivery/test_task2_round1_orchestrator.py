from __future__ import annotations

import os
import subprocess
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
        expected_git_commit="",
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


def test_submit_estep_sets_no_git_runtime_env_without_display_or_github_credentials(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.expected_git_commit = "41f8fad"
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    captured: dict[str, str] = {}

    def fake_run(command, **kwargs):
        assert command == ["bash", "scripts/task2/submit_task2_formal_103cases.sh"]
        captured.update(kwargs["env"])
        return {"ok": True, "stdout": "TASK2_FORMAL_ROOT=/tmp/formal", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_estep(args, {"status": "REUSED_ACTIVE_JOB", "job_id": "4527698"})
    assert result["status"] == "SUBMITTED"
    assert captured["RUNTIME_NO_GIT"] == "1"
    assert captured["SKIP_GIT_SYNC"] == "1"
    assert captured["EXPECTED_GIT_COMMIT"] == "41f8fad"
    assert "DISPLAY" not in captured
    assert "GITHUB_TOKEN" not in captured


def test_formal_submitter_runtime_no_git_skips_remote_sync_without_display_or_credentials(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git_calls.log"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {git_log}\n"
        "case \"$1\" in\n"
        "  fetch|pull|switch|clone) echo forbidden git network/sync >&2; exit 17 ;;\n"
        "  branch) echo main ;;\n"
        "  rev-parse) echo \"$EXPECTED_GIT_COMMIT\" ;;\n"
        "  status) exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_git.chmod(0o755)
    fake_python.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case_manifest = workspace / "cases_103_manifest.csv"
    case_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "HOME": str(tmp_path / "home"),
            "CODE_ROOT": str(repo),
            "PYTHON": str(fake_python),
            "WORKSPACE_ROOT": str(workspace),
            "CASE_MANIFEST": str(case_manifest),
            "BASE_MANIFEST": str(tmp_path / "base.csv"),
            "FORMAL_OUT_ROOT": str(tmp_path / "formal"),
            "STATE_ROOT": str(tmp_path / "state"),
            "DRY_RUN": "1",
            "RUNTIME_NO_GIT": "1",
            "SKIP_GIT_SYNC": "1",
            "EXPECTED_GIT_COMMIT": "abc123",
        }
    )
    env.pop("DISPLAY", None)
    env.pop("GITHUB_TOKEN", None)

    proc = subprocess.run(
        ["bash", "scripts/task2/submit_task2_formal_103cases.sh"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    calls = git_log.read_text(encoding="utf-8").splitlines()
    assert not any(call.startswith(("fetch", "pull", "switch", "clone")) for call in calls)


def test_expected_commit_pin_matches_and_mismatch_are_local(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command == ["git", "rev-parse", "HEAD"]
        return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    assert orch.verify_expected_git_commit(tmp_path, "abc123")["status"] == "PASSED"
    mismatch = orch.verify_expected_git_commit(tmp_path, "def456")
    assert mismatch["status"] == "FAILED"
    assert mismatch["failure_reason"] == "expected_git_commit_mismatch"
    assert calls == [["git", "rev-parse", "HEAD"], ["git", "rev-parse", "HEAD"], ["git", "rev-parse", "HEAD"]]


def test_controller_expected_commit_mismatch_fails_before_labelcritic_or_estep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.expected_git_commit = "expected"
    monkeypatch.setattr(orch, "_git_commit", lambda: "actual")

    def fail_labelcritic(*args, **kwargs):
        raise AssertionError("LabelCritic must not be touched on commit mismatch")

    monkeypatch.setattr(orch, "ensure_labelcritic_service", fail_labelcritic)
    monkeypatch.setattr(orch, "submit_estep", fail_labelcritic)

    assert orch.controller(args) == 2
    state = orch._load_state(args.state_root)
    assert state["terminal_state"] == "ROUND1_FAILED"
    assert state["stage"] == "git_commit_pin"
    assert state["failure_reason"] == "expected_git_commit_mismatch"


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
