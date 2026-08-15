from __future__ import annotations

import os
import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.dataset_delivery import task2_round1_orchestrator as orch
from tools.dataset_delivery import task2_full373_round1_launcher as full373
from tools.dataset_delivery import slurm_reliability as reliability
from tools.dataset_delivery import task2_workspace_staging as staging
from tools.dataset_delivery.slurm_reliability import CANDIDATE_TASK_V1


@pytest.fixture(autouse=True)
def _clean_runtime_env(monkeypatch):
    for key in {
        *orch.STATIC_TEST_ENV_DROP,
        "LABELCRITIC_BASE_URL",
        "LABELCRITIC_PORT",
        "LABELCRITIC_SERVICE_ROOT",
        "GITHUB_TOKEN",
    }:
        monkeypatch.delenv(key, raising=False)


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
        totalsegmentator_executable="",
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


def _labelcritic_record(job_id: str, *, state: str = "PENDING", user: str | None = None, name: str | None = None, comment: str | None = None) -> dict[str, str]:
    spec_hash = os.getenv("LABELCRITIC_SERVICE_SPEC_HASH", "")
    return {
        "status": "FOUND",
        "state": state,
        "job_id": str(job_id),
        "user": user if user is not None else (os.getenv("USER") or os.getenv("LOGNAME") or ""),
        "name": name if name is not None else orch.LABELCRITIC_JOB_NAME,
        "comment": comment if comment is not None else (f"medical_agent:labelcritic_72b:{spec_hash}" if spec_hash else ""),
    }


def _write_source_manifest(path: Path, case_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["index,case_id,ct_path,annotation_folder"]
    for index, case_id in enumerate(case_ids):
        lines.append(f"{index},{case_id},/source/{case_id}/ct.nii.gz,/source/{case_id}/segmentations")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stage_ready_case(workspace_root: Path, case_id: str) -> None:
    ct = workspace_root / "inputs" / "images" / case_id / "ct.nii.gz"
    mask = workspace_root / "inputs" / "masks_original" / case_id / "segmentations" / "liver.nii.gz"
    ct.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    ct.write_bytes(b"ct")
    mask.write_bytes(b"mask")


def _fake_teacher_submitter(monkeypatch, *, submitted_commands: list[list[str]] | None = None, captured_ready_ids: list[list[str]] | None = None):
    def fake_run(command, **kwargs):
        if submitted_commands is not None:
            submitted_commands.append(command)
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            formal_root = Path(command[command.index("--output-root") + 1])
            ready_manifest = Path(command[command.index("--case-manifest") + 1])
            ready_ids = [row["case_id"] for row in csv.DictReader(ready_manifest.open("r", encoding="utf-8"))]
            if captured_ready_ids is not None:
                captured_ready_ids.append(ready_ids)
            formal_root.mkdir(parents=True, exist_ok=True)
            groups = {
                "full373": {
                    "task_count": len([item for item in ready_ids if item]),
                    "task_manifest": str(formal_root / "slurm" / "full373_task_manifest.csv"),
                    "sbatch_file": str(formal_root / "slurm" / "full373_multiteacher_array.sbatch"),
                }
            }
            (formal_root / "formal_task2_submission_manifest.json").write_text(
                '{"status":"READY","task_count":4,"groups":' + json.dumps(groups) + "}\n",
                encoding="utf-8",
            )
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            return {"ok": True, "stdout": '{"status":"SUBMITTED"}', "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)


def test_existing_pending_labelcritic_not_resubmitted(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("111111\n", encoding="utf-8")
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: _labelcritic_record(str(job_id), state="PENDING"),
    )
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")

    def fail_submit(*args, **kwargs):
        raise AssertionError("submit_labelcritic_72b_service.sh must not be called")

    monkeypatch.setattr(orch, "_run", fail_submit)
    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "111111"
    assert result["source"] == "labelcritic_service_state"


def test_active_state_recorded_labelcritic_reused_before_service_file(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("222222\n", encoding="utf-8")
    orch._save_state(tmp_path, labelcritic={"job_id": "111111"})
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: _labelcritic_record(str(job_id), state="RUNNING"),
    )
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("active state-recorded job should be reused"))

    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "111111"
    assert result["source"] == "current_attempt_state"


def test_stale_recorded_labelcritic_ignored_and_slurm_discovered_job_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("111111\n", encoding="utf-8")

    def fake_record(job_id):
        state = "COMPLETED" if str(job_id) == "111111" else "PENDING"
        return _labelcritic_record(str(job_id), state=state)

    monkeypatch.setattr(orch, "slurm_job_record", fake_record)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "222222")
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("stale recorded job should not force new submit when discovery finds active job"))

    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "222222"
    assert result["source"] == "slurm_name_discovery"
    assert result["ignored_jobs"][0]["validation"]["failure_reason"] == "non_active_state:COMPLETED"


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED"])
def test_failed_or_cancelled_recorded_labelcritic_ignored_and_new_service_submitted(tmp_path, monkeypatch, state):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("111111\n", encoding="utf-8")
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: _labelcritic_record(str(job_id), state=state),
    )
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")
    submits: list[list[str]] = []

    def fake_run(command, **kwargs):
        submits.append(command)
        assert command == ["bash", "scripts/task2/submit_labelcritic_72b_service.sh"]
        return {"ok": True, "stdout": "LABELCRITIC_JOB_ID=222222\n", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "222222"
    assert len(submits) == 1


def test_no_labelcritic_job_submits_exactly_one_new_service(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: {"status": "NOT_FOUND", "state": "UNKNOWN", "job_id": str(job_id), "user": "", "name": ""},
    )
    submits: list[list[str]] = []

    def fake_run(command, **kwargs):
        submits.append(command)
        assert command == ["bash", "scripts/task2/submit_labelcritic_72b_service.sh"]
        return {"ok": True, "stdout": "LABELCRITIC_JOB_ID=333333\n", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "333333"
    assert submits == [["bash", "scripts/task2/submit_labelcritic_72b_service.sh"]]


def test_active_slurm_discovered_labelcritic_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "222222")
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: _labelcritic_record(str(job_id), state="RUNNING"),
    )
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("discovered active service should be reused"))

    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "222222"
    assert result["source"] == "slurm_name_discovery"


def test_labelcritic_spec_hash_mismatch_is_not_reused_and_pending_job_is_controlled_cancelled(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("111111\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_record(job_id):
        return _labelcritic_record(str(job_id), state="PENDING", comment="medical_agent:labelcritic_72b:oldhash")

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if command[:1] == ["scancel"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command == ["bash", "scripts/task2/submit_labelcritic_72b_service.sh"]:
            return {"ok": True, "stdout": "LABELCRITIC_JOB_ID=222222\n", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "slurm_job_record", fake_record)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda *args: "")
    monkeypatch.setattr(orch, "_run", fake_run)

    result = orch.ensure_labelcritic_service(tmp_path)

    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "222222"
    ignored = result["ignored_jobs"][0]
    assert "JOB_SPEC_MISMATCH" in ignored["validation"]["failure_reason"]
    assert ignored["controlled_cancellation"]["status"] == "CANCELLED_OBSOLETE"
    assert ["scancel", "111111"] in calls


def test_labelcritic_spec_hash_match_reuses_active_job_without_duplicate_submit(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    service = orch._service_paths(tmp_path)
    service["root"].mkdir(parents=True)
    service["job"].write_text("111111\n", encoding="utf-8")

    def fake_record(job_id):
        expected = os.environ["LABELCRITIC_SERVICE_SPEC_HASH"]
        return _labelcritic_record(str(job_id), state="RUNNING", comment=f"medical_agent:labelcritic_72b:{expected}")

    monkeypatch.setattr(orch, "slurm_job_record", fake_record)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda *args: "")
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("matching spec should be reused"))

    result = orch.ensure_labelcritic_service(tmp_path)

    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "111111"
    assert result["validation"]["running_spec_hash"] == result["validation"]["expected_spec_hash"]


def test_labelcritic_override_is_validated_and_wrong_user_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELCRITIC_JOB_ID", "111111")
    monkeypatch.setenv("USER", "current_user")

    def fake_record(job_id):
        expected = os.environ.get("LABELCRITIC_SERVICE_SPEC_HASH", "")
        if str(job_id) == "111111":
            return {"status": "FOUND", "state": "RUNNING", "job_id": "111111", "user": "other_user", "name": orch.LABELCRITIC_JOB_NAME, "comment": f"medical_agent:labelcritic_72b:{expected}"}
        return {"status": "FOUND", "state": "RUNNING", "job_id": "222222", "user": "current_user", "name": orch.LABELCRITIC_JOB_NAME, "comment": f"medical_agent:labelcritic_72b:{expected}"}

    monkeypatch.setattr(orch, "slurm_job_record", fake_record)
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda *args: "222222")
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("valid Slurm-discovered service should be reused"))
    result = orch.ensure_labelcritic_service(tmp_path)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "222222"
    assert result["ignored_jobs"][0]["validation"]["failure_reason"] == "user_mismatch:other_user!=current_user"


def test_multiple_matching_labelcritic_jobs_select_running_lowest_id(monkeypatch):
    records = {
        "333333": _labelcritic_record("333333", state="PENDING"),
        "222222": _labelcritic_record("222222", state="RUNNING"),
        "111111": _labelcritic_record("111111", state="RUNNING"),
    }
    monkeypatch.setattr(orch, "slurm_job_record", lambda job_id: records[str(job_id)])
    selected = orch.select_labelcritic_job(list(records.values()))
    assert selected["status"] == "SELECTED"
    assert selected["job_id"] == "111111"


def test_pending_labelcritic_waits_until_runtime_ready(tmp_path, monkeypatch):
    services = iter([
        {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"},
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
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: calls.append("ensure_labelcritic") or {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})

    poll_results = iter([
        {"status": "WAITING", "job": {"state": "PENDING"}, "service": {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"}},
        {"status": "PASSED", "base_url": "http://node", "port": 8000, "runtime": {"status": "PASSED"}},
    ])
    estep_results = iter([
        {"status": "RUNNING"},
        {"status": "PASSED"},
    ])
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or next(poll_results))
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic=None: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "advance_estep", lambda args, labelcritic=None: calls.append("advance_estep") or {"status": "SUBMITTED"})
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
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic=None: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "advance_estep", lambda args, labelcritic=None: calls.append("advance_estep") or {"status": "SUBMITTED"})
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


def test_labelcritic_ready_submits_selection_workers_while_teacher_continues(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    orch._save_state(args.state_root, formal_root=str(formal_root))
    (formal_root / "queues").mkdir(parents=True)
    monkeypatch.setattr(orch, "build_estep_telemetry", lambda formal: {"labelcritic": {"queue_depth": 3}, "case_target": {"total": 4, "terminal": 1}})
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "COMPLETED", "job_id": job_id})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(part) for part in command])
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command[:2] == ["sbatch", "--parsable"]:
            return {"ok": True, "stdout": "555\n", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_labelcritic_selection_workers(args, {"status": "PASSED", "base_url": "http://node", "port": 8000})
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "555"
    assert any("--array=0-3%4" in call for command in calls for call in command)


def test_status_reports_round1_streaming_telemetry(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    formal_root.mkdir(parents=True)
    orch._save_state(args.state_root, formal_root=str(formal_root), labelcritic={"job_id": "111111"}, mstep_job_id="222222")
    monkeypatch.setattr(
        orch,
        "build_estep_telemetry",
        lambda formal: {
            "teacher_candidate": {"total": 10, "success": 4, "running": 2, "pending": 4, "retry": 0, "terminal": 4},
            "case_target": {"total": 6, "waiting_candidates": 4, "candidates_ready": 1, "labelcritic_running": 1, "selected": 0},
            "labelcritic": {"queue_depth": 1, "running_requests": 1, "completed": 0, "retry": 0},
        },
    )
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "RUNNING", "job_id": job_id})
    status = orch.status(SimpleNamespace(state_root=args.state_root))
    assert status["teacher"]["total"] == 10
    assert status["case_target"]["total"] == 6
    assert status["labelcritic_queue"]["queue_depth"] == 1
    assert status["resources"]["labelcritic_h100_state"] == "RUNNING"


def test_estep_passed_releases_mstep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_HEALTHY", "base_url": "http://node", "port": 8000})
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or {"status": "PASSED", "base_url": "http://node", "port": 8000})
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "advance_estep", lambda args, labelcritic=None: calls.append("advance_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "check_estep", lambda args: calls.append("check_estep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "submit_mstep", lambda args: calls.append("submit_mstep") or {"status": "SUBMITTED", "job_id": "99"})
    monkeypatch.setattr(orch, "check_mstep", lambda args: calls.append("check_mstep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "run_round1_final_validator", lambda args: calls.append("final_validator") or {"status": "PASSED", "terminal_state": "ROUND1_PASSED"})

    assert orch.controller(args) == 0
    assert calls == ["submit_estep", "poll_labelcritic", "advance_estep", "check_estep", "submit_mstep", "check_mstep", "final_validator"]
    assert orch._load_state(args.state_root)["terminal_state"] == "ROUND1_PASSED"


def test_mstep_uses_a100_preferred_h100_fallback_on_backpressure(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    mask = tmp_path / "mask.nii.gz"
    mask.write_bytes(b"mask")
    orch._save_state(args.state_root, formal_root=str(formal_root), e_step_status="PASSED")
    (formal_root / "training_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (formal_root / "training_manifest.json").write_text(json.dumps({"status": "success", "items": [{"case_id": "CASE001", "organ": "organ_a", "mask_path": str(mask), "training_weight": 1.0}]}), encoding="utf-8")
    (formal_root / "full_373_estep_status.json").write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
    monkeypatch.delenv("MSTEP_PARTITION", raising=False)
    monkeypatch.delenv("MSTEP_GRES", raising=False)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(part) for part in command])
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command[:2] == ["sbatch", "--parsable"] and "student_a100" in " ".join(map(str, command)):
            return {"ok": False, "stdout": "", "stderr": "Batch job submission failed: QOSMaxJobsPerUserLimit", "return_code": 1}
        if command[:2] == ["sbatch", "--parsable"] and "student_h100" in " ".join(map(str, command)):
            return {"ok": True, "stdout": "666\n", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_mstep(args)
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "666"
    assert result["profile"]["partition"] == "gpuh100"


def test_estep_passed_waits_for_labelcritic_gate_before_mstep(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})
    poll_results = iter([
        {"status": "WAITING", "job": {"state": "RUNNING"}},
        {"status": "PASSED", "base_url": "http://node", "port": 8000},
    ])
    monkeypatch.setattr(orch, "poll_labelcritic_runtime", lambda state_root: calls.append("poll_labelcritic") or next(poll_results))
    monkeypatch.setattr(orch, "submit_estep", lambda args, labelcritic: calls.append("submit_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "advance_estep", lambda args, labelcritic=None: calls.append("advance_estep") or {"status": "SUBMITTED"})
    monkeypatch.setattr(orch, "check_estep", lambda args: calls.append("check_estep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "submit_mstep", lambda args: calls.append("submit_mstep") or {"status": "SUBMITTED", "job_id": "99"})
    monkeypatch.setattr(orch, "check_mstep", lambda args: calls.append("check_mstep") or {"status": "PASSED"})
    monkeypatch.setattr(orch, "run_round1_final_validator", lambda args: calls.append("final_validator") or {"status": "PASSED", "terminal_state": "ROUND1_PASSED"})
    monkeypatch.setattr(orch.time, "sleep", lambda sec: calls.append("sleep"))

    assert orch.controller(args) == 0
    assert calls == [
        "submit_estep",
        "poll_labelcritic",
        "advance_estep",
        "check_estep",
        "sleep",
        "poll_labelcritic",
        "advance_estep",
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


def test_ready_teacher_submit_sets_no_git_runtime_env_without_display_or_github_credentials(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.expected_git_commit = "41f8fad"
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("GIT_ASKPASS", "askpass")
    monkeypatch.setenv("SSH_ASKPASS", "askpass")
    source_manifest = tmp_path / "source.csv"
    source_manifest.write_text("index,case_id,ct_path,annotation_folder\n0,CASE001,/src/ct.nii.gz,/src/segmentations\n", encoding="utf-8")
    ct = args.workspace_root / "inputs" / "images" / "CASE001" / "ct.nii.gz"
    mask = args.workspace_root / "inputs" / "masks_original" / "CASE001" / "segmentations" / "liver.nii.gz"
    ct.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    ct.write_bytes(b"ct")
    mask.write_bytes(b"mask")
    orch._save_state(args.state_root, formal_root=str(tmp_path / "formal"))
    captured_envs: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "41f8fad", "stderr": "", "return_code": 0}
        if "env" in kwargs:
            captured_envs.append(kwargs["env"])
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            formal_root = Path(command[command.index("--output-root") + 1])
            formal_root.mkdir(parents=True, exist_ok=True)
            (formal_root / "formal_task2_submission_manifest.json").write_text('{"status":"READY","task_count":4,"groups":{}}\n', encoding="utf-8")
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            return {"ok": True, "stdout": '{"status":"SUBMITTED"}', "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "TASK2_FORMAL_ROOT=/tmp/formal", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})
    assert result["status"] == "SUBMITTED"
    assert captured_envs
    for captured in captured_envs:
        assert captured["RUNTIME_NO_GIT"] == "1"
        assert captured["SKIP_GIT_SYNC"] == "1"
        assert captured["EXPECTED_GIT_COMMIT"] == "41f8fad"
        assert captured["GIT_TERMINAL_PROMPT"] == "0"
        assert "DISPLAY" not in captured
        assert "GITHUB_TOKEN" not in captured
        assert "GH_TOKEN" not in captured
        assert "GIT_ASKPASS" not in captured
        assert "SSH_ASKPASS" not in captured


def test_ready_teacher_submit_passes_labelcritic_h100_policy_without_disabling_admin_allow(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW", "1")
    source_manifest = tmp_path / "source.csv"
    source_manifest.write_text("index,case_id,ct_path,annotation_folder\n0,CASE001,/src/ct.nii.gz,/src/segmentations\n", encoding="utf-8")
    ct = args.workspace_root / "inputs" / "images" / "CASE001" / "ct.nii.gz"
    mask = args.workspace_root / "inputs" / "masks_original" / "CASE001" / "segmentations" / "liver.nii.gz"
    ct.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    ct.write_bytes(b"ct")
    mask.write_bytes(b"mask")
    orch._save_state(args.state_root, formal_root=str(tmp_path / "formal"))
    captured_dynamic_env: dict[str, str] = {}
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "PENDING", "job_id": str(job_id), "source": "test"})

    def fake_run(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            formal_root = Path(command[command.index("--output-root") + 1])
            formal_root.mkdir(parents=True, exist_ok=True)
            (formal_root / "formal_task2_submission_manifest.json").write_text('{"status":"READY","task_count":4,"groups":{}}\n', encoding="utf-8")
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            captured_dynamic_env.update(kwargs.get("env") or {})
            formal_root = Path(command[command.index("--output-root") + 1])
            (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
            (formal_root / "slurm" / "dynamic_gpu_submission_plan_ready_batch_001.json").write_text(
                json.dumps(
                    {
                        "status": "SUBMITTED",
                        "scheduler_status": "ACTIVE",
                        "teacher_h100_policy": {
                            "allow_h100_teacher_overflow": True,
                            "labelcritic_required": True,
                            "labelcritic_job_found": True,
                            "labelcritic_job_id": "111111",
                            "labelcritic_job_state": "PENDING",
                            "labelcritic_h100_reserved": True,
                            "effective_teacher_h100_enabled": False,
                            "teacher_h100_deferred_for_labelcritic": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": '{"status":"SUBMITTED"}', "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert captured_dynamic_env["TASK2_ALLOW_H100_TEACHER_OVERFLOW"] == "1"
    policy = json.loads(captured_dynamic_env["TASK2_TEACHER_H100_POLICY_JSON"])
    assert policy["allow_h100_teacher_overflow"] is True
    assert policy["labelcritic_job_state"] == "PENDING"
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["effective_teacher_h100_enabled"] is False
    assert policy["teacher_h100_deferred_for_labelcritic"] is True
    assert orch._load_state(args.state_root)["teacher_h100_policy"]["labelcritic_job_state"] == "PENDING"


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


def test_static_preflight_pytest_uses_sanitized_environment(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.run_static_tests = True
    args.skip_sbatch_test_only = True
    args.workspace_root.mkdir(parents=True)
    args.case_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.base_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.checkpoint_root.mkdir()
    args.nnunet_predict_executable.write_text("", encoding="utf-8")
    args.unest_python_executable.write_text("", encoding="utf-8")
    args.registry.write_text("models: {}\n", encoding="utf-8")
    args.target_config.write_text('{"target_organs":[]}\n', encoding="utf-8")
    for key in orch.STATIC_TEST_ENV_DROP:
        monkeypatch.setenv(key, f"runtime_{key}")
    captured_env: dict[str, str] = {}
    monkeypatch.setattr(
        orch,
        "ensure_case_level_manifests",
        lambda args: {
            "source_manifest": str(args.case_manifest),
            "staged_manifest": str(args.case_manifest),
            "case_count": orch.FORMAL_CASE_COUNT,
            "unique_case_count": orch.FORMAL_CASE_COUNT,
        },
    )
    monkeypatch.setattr(orch, "validate_formal_manifest", lambda **kwargs: {"rows": orch.FORMAL_CASE_COUNT, "unique_case_count": orch.FORMAL_CASE_COUNT})
    monkeypatch.setattr(
        orch,
        "build_full_round1_scope",
        lambda **kwargs: {
            "status": "READY",
            "case_count": orch.FORMAL_CASE_COUNT,
            "canonical_target_count": 373,
            "enabled_teacher_count": 21,
            "target_teacher_pairs": 599,
            "total_logical_candidate_tasks": orch.FORMAL_CASE_COUNT * 599,
            "unroutable_target_count": 0,
        },
    )

    def fake_run(command, **kwargs):
        if "-m" in command and "pytest" in command:
            captured_env.update(kwargs.get("env") or {})
        if command[:2] == ["git", "status"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command[:2] == ["git", "branch"]:
            return {"ok": True, "stdout": "main", "stderr": "", "return_code": 0}
        if command[:2] == ["git", "rev-parse"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    report = orch.run_static_preflight(args)
    assert report["status"] == "PASSED"
    assert captured_env
    assert not any(key in captured_env for key in orch.STATIC_TEST_ENV_DROP)


def test_static_preflight_requires_totalsegmentator_executable_when_routed(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.skip_sbatch_test_only = True
    args.workspace_root.mkdir(parents=True)
    args.case_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.base_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.checkpoint_root.mkdir()
    args.nnunet_predict_executable.write_text("", encoding="utf-8")
    args.unest_python_executable.write_text("", encoding="utf-8")
    args.registry.write_text("models: {}\n", encoding="utf-8")
    args.target_config.write_text('{"target_organs":["liver"]}\n', encoding="utf-8")
    monkeypatch.setenv("TOTAL_SEGMENTATOR_EXECUTABLE", str(tmp_path / "missing" / "TotalSegmentator"))
    monkeypatch.setattr(orch, "ensure_case_level_manifests", lambda args: {"source_manifest": str(args.case_manifest), "staged_manifest": str(args.case_manifest), "case_count": orch.FORMAL_CASE_COUNT, "unique_case_count": orch.FORMAL_CASE_COUNT})
    monkeypatch.setattr(orch, "validate_formal_manifest", lambda **kwargs: {"rows": orch.FORMAL_CASE_COUNT, "unique_case_count": orch.FORMAL_CASE_COUNT})
    monkeypatch.setattr(
        orch,
        "build_full_round1_scope",
        lambda **kwargs: {
            "status": "READY",
            "case_count": orch.FORMAL_CASE_COUNT,
            "canonical_target_count": 373,
            "enabled_teacher_count": 1,
            "target_teacher_pairs": 1,
            "total_logical_candidate_tasks": orch.FORMAL_CASE_COUNT,
            "unroutable_target_count": 0,
            "routes": {"liver": ["totalsegmentator"]},
        },
    )

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "status"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command[:2] == ["git", "branch"]:
            return {"ok": True, "stdout": "main", "stderr": "", "return_code": 0}
        if command[:2] == ["git", "rev-parse"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)

    report = orch.run_static_preflight(args)

    assert report["status"] == "FAILED"
    check = next(item for item in report["checks"] if item["name"] == "totalsegmentator_executable")
    assert check["detail"]["status"] == "failed"


def test_static_preflight_accepts_absolute_totalsegmentator_executable_when_routed(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.skip_sbatch_test_only = True
    exe = tmp_path / "bin" / "TotalSegmentator"
    exe.parent.mkdir()
    exe.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    exe.chmod(0o755)
    args.totalsegmentator_executable = str(exe)
    args.workspace_root.mkdir(parents=True)
    args.case_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.base_manifest.write_text("case_id,ct_path,annotation_folder\n", encoding="utf-8")
    args.checkpoint_root.mkdir()
    args.nnunet_predict_executable.write_text("", encoding="utf-8")
    args.unest_python_executable.write_text("", encoding="utf-8")
    args.registry.write_text("models: {}\n", encoding="utf-8")
    args.target_config.write_text('{"target_organs":["liver"]}\n', encoding="utf-8")
    monkeypatch.setattr(orch, "ensure_case_level_manifests", lambda args: {"source_manifest": str(args.case_manifest), "staged_manifest": str(args.case_manifest), "case_count": orch.FORMAL_CASE_COUNT, "unique_case_count": orch.FORMAL_CASE_COUNT})
    monkeypatch.setattr(orch, "validate_formal_manifest", lambda **kwargs: {"rows": orch.FORMAL_CASE_COUNT, "unique_case_count": orch.FORMAL_CASE_COUNT})
    monkeypatch.setattr(
        orch,
        "build_full_round1_scope",
        lambda **kwargs: {
            "status": "READY",
            "case_count": orch.FORMAL_CASE_COUNT,
            "canonical_target_count": 373,
            "enabled_teacher_count": 1,
            "target_teacher_pairs": 1,
            "total_logical_candidate_tasks": orch.FORMAL_CASE_COUNT,
            "unroutable_target_count": 0,
            "routes": {"liver": ["totalsegmentator"]},
        },
    )
    monkeypatch.setattr(orch, "_run", lambda command, **kwargs: {"ok": True, "stdout": "main" if command[:2] == ["git", "branch"] else ("abc123" if command[:2] == ["git", "rev-parse"] else ""), "stderr": "", "return_code": 0})

    report = orch.run_static_preflight(args)

    check = next(item for item in report["checks"] if item["name"] == "totalsegmentator_executable")
    assert report["status"] == "PASSED"
    assert check["detail"]["executable"] == str(exe)


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
    monkeypatch.setattr(orch, "ensure_labelcritic_service", lambda state_root: {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})
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


def test_archive_current_attempt_never_renames_round1_orchestrated_root(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    paths["root"].mkdir(parents=True)
    paths["state"].write_text('{"run_id":"round1_test"}\n', encoding="utf-8")
    durable = paths["root"] / orch.FULL373_ROOT_NAME
    durable.mkdir()
    (durable / "scientific_sentinel.txt").write_text("keep me\n", encoding="utf-8")
    original_rename = Path.rename

    def spy_rename(self, target):
        if self == paths["root"]:
            raise AssertionError("archive_current_attempt must not rename the durable round1_orchestrated root")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", spy_rename)

    archive = orch.archive_current_attempt(args.state_root, reason="safety")

    assert archive["status"] == "ARCHIVED"
    assert durable.is_dir()
    assert (durable / "scientific_sentinel.txt").read_text(encoding="utf-8") == "keep me\n"
    assert "full_373_multiteacher_round1" in archive["retained_artifacts"]


def test_retry_failed_fault_injection_preserves_scientific_state_and_replenishes_workers(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    formal_root = paths["root"] / orch.FULL373_ROOT_NAME
    queue_manifest = formal_root / "queues" / "shared_ready_candidate_manifest.csv"
    queue_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_success", "ct_path": "/ct", "annotation_folder": "/ann"},
        {"task_index": "1", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher2", "candidate_id": "cand_empty", "ct_path": "/ct", "annotation_folder": "/ann"},
        {"task_index": "2", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher3", "candidate_id": "cand_ready", "ct_path": "/ct", "annotation_folder": "/ann"},
        {"task_index": "3", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher4", "candidate_id": "cand_running", "ct_path": "/ct", "annotation_folder": "/ann"},
    ]
    queue_manifest.write_text(
        "task_index,case_id,target,teacher,candidate_id,ct_path,annotation_folder\n"
        + "\n".join(",".join(row[field] for field in ["task_index", "case_id", "target", "teacher", "candidate_id", "ct_path", "annotation_folder"]) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    full373.write_json(
        formal_root / "full_round1_scope.json",
        {
            "run_id": "round1_test_scientific",
            "case_count": 1,
            "canonical_target_count": 1,
            "routes": {"organ_a": ["teacher1", "teacher2", "teacher3", "teacher4"]},
            "task_rows": rows,
            "total_logical_candidate_tasks": 4,
        },
    )
    (formal_root / "scientific_sentinel.txt").write_text("durable scientific output\n", encoding="utf-8")
    for status, teacher, candidate_id in (
        ("SUCCESS", "teacher1", "cand_success"),
        ("COMPLETED_NO_NONZERO", "teacher2", "cand_empty"),
        ("READY", "teacher3", "cand_ready"),
        ("RUNNING", "teacher4", "cand_running"),
    ):
        full373.publish_candidate_state(
            formal_root,
            {
                "status": status,
                "case_id": "CASE001",
                "target": "organ_a",
                "teacher": teacher,
                "candidate_id": candidate_id,
                "candidate_exists": status == "SUCCESS",
                "prediction": "/mask" if status == "SUCCESS" else "",
            },
            recompute_target=False,
        )
    (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
    (formal_root / "slurm" / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"round1_test_scientific,attempt_A,ready_batch_001,4584475,full373,gpu_t4,4,{CANDIDATE_TASK_V1},submitted,ACTIVE,CANCELLED\n"
        f"round1_test_scientific,attempt_A,ready_batch_001,4584476,full373,gpu_t4,4,{CANDIDATE_TASK_V1},submitted,ACTIVE,TIMEOUT\n"
        f"round1_test_scientific,attempt_A,ready_batch_001,4584477,full373,gpu_t4,4,{CANDIDATE_TASK_V1},submitted,ACTIVE,FAILED\n",
        encoding="utf-8",
    )
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    source_manifest = _write_source_manifest(orch._state_paths(args.state_root)["source_manifest"], case_ids)
    for case_id in case_ids:
        _stage_ready_case(args.workspace_root, case_id)
    label_service = orch._service_paths(args.state_root)
    label_service["root"].mkdir(parents=True, exist_ok=True)
    label_service["job"].write_text("111111\n", encoding="utf-8")
    orch._save_state(
        args.state_root,
        run_id="round1_test_scientific",
        terminal_state="ROUND1_FAILED",
        scheduler_status="FATAL",
        e_step_status="FAILED",
        failure_reason="e_step_jobs_terminal_failed",
        execution_attempt_id="attempt_A",
        worker_generation="attempt_A",
        controller_job_id="old_controller",
        formal_root=str(formal_root),
        source_manifest=str(source_manifest),
        teacher_submitted_case_ids=case_ids,
        labelcritic={"job_id": "111111", "status": "SUBMITTED"},
    )
    commands: list[list[str]] = []

    def fake_preflight(call_args):
        new_paths = orch._state_paths(call_args.state_root)
        new_paths["root"].mkdir(parents=True, exist_ok=True)
        new_paths["controller_sbatch"].write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        return {"status": "PASSED"}

    def fake_run(command, **kwargs):
        commands.append([str(item) for item in command])
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        if command[:2] == ["sbatch", "--parsable"]:
            return {"ok": True, "stdout": "999001\n", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            (formal_root / "formal_task2_submission_manifest.json").write_text(
                json.dumps({"status": "READY", "task_count": 4, "groups": {"full373": {"task_count": 4, "task_manifest": str(queue_manifest), "sbatch_file": str(formal_root / "slurm" / "worker.sbatch")}}}),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            safe_id = "ready_batch_001"
            (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
            full373.recover_candidate_seed_marker(formal_root, task_manifest=queue_manifest)
            full373.recover_stale_candidate_claims(formal_root, task_manifest=queue_manifest)
            (formal_root / "slurm" / f"dynamic_gpu_submission_plan_{safe_id}.json").write_text(
                json.dumps({"status": "SUBMITTED", "scheduler_status": "ACTIVE", "task_ownership": "shared_queue", "profile_binding": False}),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": '{"status":"SUBMITTED"}', "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "run_static_preflight", fake_preflight)
    monkeypatch.setattr(orch, "_run", fake_run)
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "PENDING" if str(job_id) == "111111" else "CANCELLED", "job_id": str(job_id)})
    monkeypatch.setattr(orch, "slurm_job_timing", lambda job_id: {"state": "CANCELLED", "job_id": str(job_id)})
    monkeypatch.setattr(orch, "slurm_job_record", lambda job_id: _labelcritic_record(str(job_id), state="PENDING"))
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")

    submit = orch.submit_controller(args)
    labelcritic = orch.ensure_labelcritic_service(args.state_root)
    estep = orch.submit_ready_teacher_batch(args, source_manifest, labelcritic)
    check = orch.check_estep(args)
    claim_a = full373.claim_next_ready_candidate(formal_root, task_manifest=queue_manifest, worker_id="w1", profile="gpu_t4")
    claim_b = full373.claim_next_ready_candidate(formal_root, task_manifest=queue_manifest, worker_id="w2", profile="gpu_t4")

    archive_records = list((args.state_root / "round1_orchestrated_attempts").glob("attempt_001/attempt_archive.json"))
    state = orch._load_state(args.state_root)
    marker = json.loads((formal_root / "queues" / "candidate_seed_complete.json").read_text(encoding="utf-8"))
    dynamic_commands = [command for command in commands if any(part.endswith("task2_dynamic_gpu_submitter.py") for part in command)]

    assert submit["status"] == "CONTROLLER_SUBMITTED"
    assert archive_records
    assert (formal_root / "scientific_sentinel.txt").read_text(encoding="utf-8") == "durable scientific output\n"
    assert full373.load_candidate_state(formal_root, case_id="CASE001", target="organ_a", teacher="teacher1")["status"] == "SUCCESS"
    assert full373.load_candidate_state(formal_root, case_id="CASE001", target="organ_a", teacher="teacher2")["status"] == "COMPLETED_NO_NONZERO"
    assert state["run_id"] == "round1_test_scientific"
    assert state["execution_attempt_id"] != "attempt_A"
    assert state["worker_generation"] != "attempt_A"
    assert state["controller_job_id"] == "999001"
    assert labelcritic["status"] == "REUSED_ACTIVE_JOB"
    assert labelcritic["job_id"] == "111111"
    assert estep["status"] == "SUBMITTED"
    assert dynamic_commands
    assert check["status"] == "RUNNING"
    assert check["lifecycle"]["historical"]
    assert marker["logical_task_count"] == 4
    assert marker["validated_existing_count"] == 4
    assert marker["scientific_run_id"] == "round1_test_scientific"
    assert claim_a["status"] == "CLAIMED"
    assert claim_b["status"] == "CLAIMED"
    assert claim_a["candidate"]["candidate_id"] != claim_b["candidate"]["candidate_id"]


def test_retry_attempt_dynamically_reuses_active_labelcritic_service(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELCRITIC_JOB_ID", raising=False)
    args = _args(tmp_path)
    orch._save_state(args.state_root, terminal_state="ROUND1_FAILED", failure_reason="old_failure")
    service = orch._service_paths(args.state_root)
    service["root"].mkdir(parents=True, exist_ok=True)
    service["job"].write_text("111111\n", encoding="utf-8")
    orch.archive_current_attempt(args.state_root, reason="retry")
    monkeypatch.setattr(
        orch,
        "slurm_job_record",
        lambda job_id: _labelcritic_record(str(job_id), state="PENDING"),
    )
    monkeypatch.setattr(orch, "find_labelcritic_job_by_name", lambda: "")
    monkeypatch.setattr(orch, "_run", lambda *args, **kwargs: pytest.fail("active service should be dynamically reused after retry archive"))

    result = orch.ensure_labelcritic_service(args.state_root)
    assert result["status"] == "REUSED_ACTIVE_JOB"
    assert result["job_id"] == "111111"
    assert result["source"] == "labelcritic_service_state"


def test_case_input_ready_submits_teacher_while_other_case_staging_running(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001", "CASE002"])
    _stage_ready_case(args.workspace_root, "CASE001")
    state_dir = args.workspace_root / "manifests" / "staging_cases"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "CASE002.json").write_text('{"status":"STAGING_RUNNING","case_id":"CASE002"}\n', encoding="utf-8")
    captured_ready: list[list[str]] = []
    _fake_teacher_submitter(monkeypatch, captured_ready_ids=captured_ready)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert result["case_ids"] == ["CASE001"]
    assert captured_ready == [["CASE001"]]
    assert orch._load_state(args.state_root)["teacher_submitted_case_ids"] == ["CASE001"]


def test_seventy_of_103_staged_allows_gpu_teacher_workers(tmp_path, monkeypatch):
    args = _args(tmp_path)
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    source_manifest = _write_source_manifest(tmp_path / "source.csv", case_ids)
    for case_id in case_ids[:70]:
        _stage_ready_case(args.workspace_root, case_id)
    commands: list[list[str]] = []
    captured_ready: list[list[str]] = []
    _fake_teacher_submitter(monkeypatch, submitted_commands=commands, captured_ready_ids=captured_ready)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert result["case_count"] == 70
    assert captured_ready == [case_ids[:70]]
    assert any(any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command) for command in commands)


def test_shared_queue_restart_ignores_historical_cancelled_worker_as_scientific_failure(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    (formal_root / "slurm").mkdir(parents=True)
    (formal_root / "slurm" / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"round1_run,attempt_a,ready_batch_001,4584475,full373,gpu_t4,6,{CANDIDATE_TASK_V1},submitted,ACTIVE,CANCELLED\n",
        encoding="utf-8",
    )
    orch._save_state(args.state_root, formal_root=str(formal_root), run_id="round1_run", execution_attempt_id="attempt_b")
    monkeypatch.setattr(orch, "aggregate_full373_estep", lambda *args, **kwargs: {"status": "RUNNING"})
    monkeypatch.setattr(orch, "slurm_job_timing", lambda job_id: {"state": "CANCELLED", "job_id": job_id})
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "CANCELLED", "job_id": job_id})

    result = orch.check_estep(args)

    assert result["status"] == "RUNNING"
    assert result["active_jobs"] == []
    assert result["lifecycle"]["historical"][0]["job_id"] == "4584475"
    assert result["lifecycle"]["fatal"] == []
    assert (orch._state_paths(args.state_root)["job_lifecycle"]).is_file()


def test_shared_queue_restart_replenishes_workers_even_when_all_cases_previously_submitted(tmp_path, monkeypatch):
    args = _args(tmp_path)
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    source_manifest = _write_source_manifest(tmp_path / "source.csv", case_ids)
    for case_id in case_ids:
        _stage_ready_case(args.workspace_root, case_id)
    formal_root = tmp_path / "formal"
    orch._save_state(
        args.state_root,
        formal_root=str(formal_root),
        run_id="round1_run",
        execution_attempt_id="attempt_b",
        teacher_submitted_case_ids=case_ids,
    )
    commands: list[list[str]] = []
    captured_ready: list[list[str]] = []
    _fake_teacher_submitter(monkeypatch, submitted_commands=commands, captured_ready_ids=captured_ready)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert result["case_count"] == 103
    assert result["previously_submitted_case_count"] == 103
    assert captured_ready == [case_ids]
    dynamic_commands = [command for command in commands if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command)]
    assert dynamic_commands
    assert "--execution-attempt-id" in dynamic_commands[0]
    assert dynamic_commands[0][dynamic_commands[0].index("--execution-attempt-id") + 1] == "attempt_b"


def test_existing_staged_case_resume_does_not_recopied(tmp_path, monkeypatch):
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001"])
    _stage_ready_case(tmp_path / "workspace", "CASE001")
    monkeypatch.setattr(staging, "_copy_file", lambda *args, **kwargs: pytest.fail("ready case must be reused without copying"))
    monkeypatch.setattr(staging, "_copy_mask_dir", lambda *args, **kwargs: pytest.fail("ready case must be reused without copying"))

    result = staging.stage_workspace_case(
        cases_manifest=source_manifest,
        case_index=0,
        workspace_root=tmp_path / "workspace",
        image_root=tmp_path / "public_images",
        mask_root=tmp_path / "public_masks",
        resume=True,
    )

    assert result["status"] == "INPUT_READY"
    assert result["action"] == "reused_existing"


def test_one_staging_failure_does_not_block_other_ready_case_teacher_submit(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001", "CASE002"])
    _stage_ready_case(args.workspace_root, "CASE001")
    state_dir = args.workspace_root / "manifests" / "staging_cases"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "CASE002.json").write_text(
        '{"status":"STAGING_FAILED","case_id":"CASE002","errors":[{"type":"source_file_missing"}]}\n',
        encoding="utf-8",
    )
    captured_ready: list[list[str]] = []
    _fake_teacher_submitter(monkeypatch, captured_ready_ids=captured_ready)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert result["case_ids"] == ["CASE001"]
    assert captured_ready == [["CASE001"]]


def test_cpu_staging_gpu_teacher_and_labelcritic_tracks_are_started_together(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001", "CASE002"])
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "ensure_case_level_manifests",
        lambda args: calls.append("manifests") or {
            "source_manifest": str(source_manifest),
            "staged_manifest": str(args.case_manifest),
            "case_count": 103,
            "unique_case_count": 103,
        },
    )
    monkeypatch.setattr(orch, "submit_staging_workers", lambda args, source: calls.append("cpu_staging") or {"status": "SUBMITTED", "job_id": "222222", "ready_count": 70, "case_count": 103})
    monkeypatch.setattr(orch, "submit_ready_teacher_batch", lambda args, source, labelcritic=None: calls.append("gpu_teacher") or {"status": "SUBMITTED", "case_count": 70})

    result = orch.submit_estep(args, {"status": "REUSED_ACTIVE_JOB", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert calls == ["manifests", "cpu_staging", "gpu_teacher"]


def test_submit_estep_no_longer_waits_on_whole_batch_staging_shell(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "advance_estep",
        lambda args, labelcritic=None: calls.append("advance_estep") or {
            "status": "SUBMITTED",
            "staging": {"status": "SUBMITTED", "job_id": "222222"},
            "teachers": {"status": "SUBMITTED", "case_count": 70},
        },
    )

    result = orch.submit_estep(args, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "SUBMITTED"
    assert calls == ["advance_estep"]


def test_case_level_source_manifest_preserves_valid_source_paths(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    source_manifest = _write_source_manifest(paths["source_manifest"], case_ids)
    calls: list[str] = []
    monkeypatch.setattr(orch, "build_formal_manifest", lambda **kwargs: calls.append("build"))

    result = orch.ensure_case_level_manifests(args)

    assert result["source_manifest"] == str(source_manifest)
    assert calls == []
    rows = source_manifest.read_text(encoding="utf-8")
    assert "/source/CASE000/ct.nii.gz" in rows


def test_case_level_staged_manifest_distinguishes_original_and_canonical_mask_dirs(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    _write_source_manifest(paths["source_manifest"], case_ids)
    monkeypatch.setattr(orch, "build_formal_manifest", lambda **kwargs: pytest.fail("source manifest should be reused"))

    result = orch.ensure_case_level_manifests(args)
    rows = orch.read_csv_rows(Path(result["staged_manifest"]))
    first = rows[0]

    assert first["annotation_folder"].endswith("/inputs/masks_original/CASE000/segmentations")
    assert first["reference_mask_dir"] == first["annotation_folder"]
    assert first["original_annotation_folder"] == first["annotation_folder"]
    assert first["canonical_annotation_folder"].endswith("/inputs/masks_373_canonical/CASE000/segmentations")


def test_case_level_source_manifest_rebuilds_stale_staged_paths(tmp_path, monkeypatch):
    args = _args(tmp_path)
    paths = orch._state_paths(args.state_root)
    case_ids = [f"CASE{i:03d}" for i in range(103)]
    paths["source_manifest"].parent.mkdir(parents=True, exist_ok=True)
    lines = ["index,case_id,ct_path,annotation_folder"]
    for index, case_id in enumerate(case_ids):
        lines.append(
            f"{index},{case_id},{args.workspace_root}/inputs/images/{case_id}/ct.nii.gz,"
            f"{args.workspace_root}/inputs/masks_original/{case_id}/segmentations"
        )
    paths["source_manifest"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    def fake_build_formal_manifest(*, output_manifest, **kwargs):
        _write_source_manifest(Path(output_manifest), case_ids)
        return {"status": "READY"}

    monkeypatch.setattr(orch, "build_formal_manifest", fake_build_formal_manifest)

    result = orch.ensure_case_level_manifests(args)

    assert result["case_count"] == 103
    rebuilt = paths["source_manifest"].read_text(encoding="utf-8")
    assert "/source/CASE000/ct.nii.gz" in rebuilt
    assert str(args.workspace_root) not in rebuilt


def test_no_production_code_contains_concrete_historical_slurm_job_ids():
    repo = Path(__file__).resolve().parents[2]
    historical_ids = {
        "452" + suffix
        for suffix in ("7698", "7571", "7761", "7697")
    }
    production_roots = [repo / "scripts", repo / "tools", repo / "scheduler", repo / "configs"]
    offenders: list[str] = []
    for root in production_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml", ".json"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for job_id in historical_ids:
                    if job_id in text:
                        offenders.append(f"{path.relative_to(repo)}:{job_id}")
    assert offenders == []


def test_teacher_submit_backpressure_is_not_round1_failure_and_keeps_submission_id(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001"])
    _stage_ready_case(args.workspace_root, "CASE001")
    formal_root = tmp_path / "formal"
    orch._save_state(args.state_root, formal_root=str(formal_root), run_id="run_bp")

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            formal_root.mkdir(parents=True, exist_ok=True)
            (formal_root / "formal_task2_submission_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "READY",
                        "task_count": 1,
                        "groups": {
                            "cads": {
                                "task_count": 1,
                                "task_manifest": str(formal_root / "slurm" / "cads_task_manifest.csv"),
                                "sbatch_file": str(formal_root / "slurm" / "cads_task2_array.sbatch"),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
            (formal_root / "slurm" / "dynamic_gpu_submission_plan_ready_batch_001.json").write_text(
                '{"status":"WAITING_FOR_SUBMISSION_CAPACITY","scheduler_status":"BACKPRESSURED"}\n',
                encoding="utf-8",
            )
            return {"ok": True, "stdout": '{"status":"WAITING_FOR_SUBMISSION_CAPACITY"}', "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)

    result = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})

    assert result["status"] == "WAITING_FOR_SUBMISSION_CAPACITY"
    state = orch._load_state(args.state_root)
    assert state["scheduler_status"] == "BACKPRESSURED"
    assert state["teacher_active_submission_id"] == "ready_batch_001"
    assert state.get("teacher_submitted_case_ids") in (None, [])


def test_dynamic_submit_timeout_is_retryable_and_reconciles_partial_jobs_without_duplicate(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_manifest = _write_source_manifest(tmp_path / "source.csv", ["CASE001"])
    _stage_ready_case(args.workspace_root, "CASE001")
    formal_root = tmp_path / "formal"
    orch._save_state(args.state_root, formal_root=str(formal_root), run_id="round1_science_stable", e_step_status="RUNNING")
    dynamic_calls: list[dict[str, object]] = []
    reconcile_calls: list[str] = []

    def fake_reconcile(call_args, root=None):
        resolved_root = Path(root or formal_root)
        reconcile_calls.append(str(resolved_root))
        reliability.persist_submitted_job(
            resolved_root / "slurm",
            {
                "run_id": "round1_science_stable",
                "execution_attempt_id": orch._execution_attempt_id(call_args.state_root),
                "worker_generation": orch._execution_attempt_id(call_args.state_root),
                "submission_id": "ready_batch_001",
                "job_id": "777777",
                "group": "full373",
                "model_group": "full373",
                "profile": "generic_gpu",
                "shard_id": "shard_000",
                "execution_schema_version": CANDIDATE_TASK_V1,
                "submission_status": "adopted",
                "scheduler_status": "ADOPTED_ACTIVE_JOB",
                "slurm_state": "RUNNING",
            },
        )
        return {"status": "RECONCILED", "adopted_count": 1, "jobs": [{"job_id": "777777"}]}

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_full373_round1_launcher.py") for part in command):
            formal_root.mkdir(parents=True, exist_ok=True)
            (formal_root / "formal_task2_submission_manifest.json").write_text(
                json.dumps({"status": "READY", "task_count": 1, "groups": {"full373": {"task_count": 1}}}),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if any(str(part).endswith("task2_dynamic_gpu_submitter.py") for part in command):
            dynamic_calls.append({"command": [str(part) for part in command], "timeout": kwargs.get("timeout")})
            if len(dynamic_calls) == 1:
                return {
                    "ok": False,
                    "stdout": "submitted job 777777",
                    "stderr": "TimeoutExpired",
                    "return_code": 124,
                    "timed_out": True,
                    "timeout_sec": kwargs.get("timeout"),
                }
            (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
            (formal_root / "slurm" / "dynamic_gpu_submission_plan_ready_batch_001.json").write_text(
                json.dumps({"status": "SUBMITTED", "scheduler_status": "ACTIVE"}),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": '{"status":"SUBMITTED"}', "stderr": "", "return_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setenv("TASK2_DYNAMIC_GPU_SUBMIT_TIMEOUT_SEC", "60")
    monkeypatch.setattr(orch, "reconcile_active_teacher_jobs", fake_reconcile)
    monkeypatch.setattr(orch, "_run", fake_run)

    first = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})
    state_after_timeout = orch._load_state(args.state_root)
    second = orch.submit_ready_teacher_batch(args, source_manifest, {"status": "WAITING", "job_id": "111111"})
    jobs = reliability.load_submitted_jobs(formal_root / "slurm")

    assert first["status"] == "RETRY_READY"
    assert state_after_timeout["scheduler_status"] == "RETRY_READY"
    assert state_after_timeout["e_step_status"] == "RUNNING"
    assert state_after_timeout.get("terminal_state") != "ROUND1_FAILED"
    assert state_after_timeout["dynamic_submit_timeout"]["reconcile"]["adopted_count"] == 1
    assert second["status"] == "SUBMITTED"
    assert second["submission_id"] == "ready_batch_001"
    assert orch._load_state(args.state_root)["run_id"] == "round1_science_stable"
    assert orch._load_state(args.state_root)["formal_root"] == str(formal_root)
    assert len([row for row in jobs if row["job_id"] == "777777"]) == 1
    assert len(dynamic_calls) == 2
    assert all("ready_batch_001" in call["command"] for call in dynamic_calls)
    assert dynamic_calls[0]["timeout"] == 60
    assert reconcile_calls


def test_reconcile_active_teacher_jobs_adopts_matching_slurm_job_without_hardcoded_id(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    orch._save_state(args.state_root, formal_root=str(formal_root), run_id="run_reconcile")

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return {
                "ok": True,
                "stdout": f"91919|RUNNING|{os.getenv('USER') or ''}|task2_cads_gpu_t4|/repo|medical_agent:run_reconcile:{CANDIDATE_TASK_V1}:ready_batch_001:gpu_t4:shard_000|{formal_root}/slurm/dynamic/ready_batch_001/cads_gpu_t4_shard_000_task2_array.sbatch\n",
                "stderr": "",
                "return_code": 0,
            }
        if command == ["git", "rev-parse", "HEAD"]:
            return {"ok": True, "stdout": "abc123", "stderr": "", "return_code": 0}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(orch, "_run", fake_run)
    result = orch.reconcile_active_teacher_jobs(args, formal_root)

    assert result["adopted_count"] == 1
    rows = list(csv.DictReader((formal_root / "slurm" / "submitted_jobs.csv").open("r", encoding="utf-8")))
    assert rows[0]["job_id"] == "91919"
    assert rows[0]["submission_status"] == "adopted"


def test_running_job_disappears_and_sacct_timeout_records_retry_pending_not_failed(tmp_path, monkeypatch):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    (formal_root / "slurm").mkdir(parents=True)
    (formal_root / "slurm" / "submitted_jobs.csv").write_text(
        "job_id,model_group,profile,submission_status,scheduler_status\n91920,cads,gpu_t4,submitted,ACTIVE\n",
        encoding="utf-8",
    )
    orch._save_state(args.state_root, formal_root=str(formal_root))

    def fake_run(command, **kwargs):
        if command[:2] == ["bash", "scripts/task2/check_task2_formal_103cases.sh"]:
            return {"ok": False, "stdout": "", "stderr": "not ready", "return_code": 2}
        return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}

    monkeypatch.setattr(orch, "_run", fake_run)
    monkeypatch.setattr(orch, "slurm_job_state", lambda job_id: {"state": "TIMEOUT", "job_id": job_id, "source": "sacct"})
    monkeypatch.setattr(orch, "slurm_job_timing", lambda job_id: {"state": "TIMEOUT", "job_id": job_id, "elapsed": "06:00:00", "time_limit": "06:00:00", "time_left": "00:00:00"})

    result = orch.check_estep(args)

    assert result["status"] == "RUNNING"
    assert result["scheduler_status"] == "RETRY_PENDING"
    assert orch._load_state(args.state_root)["walltime_recovery_status"] == "RETRY_PENDING"
    assert (orch._state_paths(args.state_root)["job_lifecycle"]).is_file()


def test_worker_pretimeout_marks_retry_pending_without_partial_publish(tmp_path):
    event = reliability.worker_pretimeout(
        tmp_path / "state",
        logical_task_id="run:batch:cads:gpu",
        job_id="91921",
        task_manifest="/tmp/tasks.csv",
        task_index="7",
    )

    assert event["status"] == "RETRY_PENDING"
    assert event["partial_publish_allowed"] is False
    assert (tmp_path / "state" / "round1_orchestrated" / "walltime_guard_last.json").is_file()


def test_student_pretimeout_writes_checkpoint_request_for_resume(tmp_path):
    event = reliability.student_pretimeout(tmp_path / "state", tmp_path / "ckpt", job_id="91922")

    assert event["status"] == "CHECKPOINTED_FOR_WALLTIME"
    assert Path(event["checkpoint_path"]).is_file()
    assert json.loads(Path(event["checkpoint_path"]).read_text(encoding="utf-8"))["resume_required"] is True


def test_time_limit_extension_denied_is_recorded_not_fatal(tmp_path):
    event = reliability.time_extension_denied(tmp_path / "state", job_id="91923", reason="scontrol_denied")

    assert event["status"] == "TIME_EXTENSION_UNAVAILABLE"
    assert event["scheduler_state"] == "RETRY_PENDING"


def test_status_reports_shared_queue_worker_and_qos_telemetry(tmp_path):
    args = _args(tmp_path)
    formal_root = tmp_path / "formal"
    (formal_root / "slurm").mkdir(parents=True, exist_ok=True)
    (formal_root / "slurm" / "dynamic_gpu_submission_plan.json").write_text(
        json.dumps(
            {
                "task_ownership": "shared_queue",
                "profile_binding": False,
                "primary_teacher_profile": "gpu_t4",
                "allow_h100_teacher_overflow": True,
                "labelcritic_required": True,
                "labelcritic_job_found": True,
                "labelcritic_job_id": "777777",
                "labelcritic_job_state": "RUNNING",
                "labelcritic_h100_reserved": True,
                "effective_teacher_h100_enabled": False,
                "teacher_h100_deferred_for_labelcritic": True,
                "desired_teacher_h100_workers": 0,
                "teacher_h100_policy": {
                    "allow_h100_teacher_overflow": True,
                    "labelcritic_required": True,
                    "labelcritic_job_found": True,
                    "labelcritic_job_id": "777777",
                    "labelcritic_job_state": "RUNNING",
                    "labelcritic_h100_reserved": True,
                    "effective_teacher_h100_enabled": False,
                    "teacher_h100_deferred_for_labelcritic": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (formal_root / "slurm" / "qos_capacity_cache.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "gpu_t4": {"known_good_size": 4000},
                    "gpu_a100": {"known_good_size": 500, "known_bad_size": 1000, "last_backpressure": {"reason": "QOSMaxSubmitJobPerUserLimit"}},
                }
            }
        ),
        encoding="utf-8",
    )
    (formal_root / "slurm" / "submitted_jobs.csv").write_text(
        "job_id,profile,gres,submission_status,scheduler_status\n"
        "91001,gpu_t4,gpu:T4:1,submitted,ACTIVE\n"
        "91002,gpu_a100,gpu:A100:1,submitted,ACTIVE\n",
        encoding="utf-8",
    )
    orch._save_state(args.state_root, formal_root=str(formal_root), labelcritic={"job_id": "777777"})

    state_map = {
        "91001": {"state": "RUNNING", "job_id": "91001"},
        "91002": {"state": "PENDING", "job_id": "91002"},
        "777777": {"state": "RUNNING", "job_id": "777777"},
    }
    orch.slurm_job_state = lambda job_id: state_map.get(str(job_id), {"state": "UNKNOWN", "job_id": str(job_id)})  # type: ignore[assignment]
    payload = orch.status(args)

    assert payload["resources"]["task_ownership"] == "shared_queue"
    assert payload["resources"]["profile_binding"] is False
    assert payload["resources"]["primary_teacher_profile"] == "gpu_t4"
    assert payload["resources"]["allow_h100_teacher_overflow"] is True
    assert payload["resources"]["labelcritic_job_state"] == "RUNNING"
    assert payload["resources"]["labelcritic_h100_reserved"] is True
    assert payload["resources"]["effective_teacher_h100_enabled"] is False
    assert payload["resources"]["teacher_h100_deferred_for_labelcritic"] is True
    assert payload["resources"]["desired_teacher_h100_workers"] == 0
    assert payload["resources"]["teacher_workers"]["t4_running"] == 1
    assert payload["resources"]["teacher_workers"]["a100_pending"] == 1
    assert payload["resources"]["qos"]["per_profile_known_good"]["gpu_t4"] == 4000
    assert payload["resources"]["qos"]["per_profile_known_bad"]["gpu_a100"] == 1000
