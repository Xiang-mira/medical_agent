from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dataset_delivery import task2_h100_policy as h100_policy
from tools.dataset_delivery.task2_h100_policy import resolve_teacher_h100_policy


@pytest.fixture(autouse=True)
def _clean_policy_env(monkeypatch):
    for key in {
        "TASK2_ALLOW_H100_TEACHER_OVERFLOW",
        "TASK2_LABELCRITIC_REQUIRED",
        "TASK2_LABELCRITIC_H100_RESERVED",
        "TASK2_LABELCRITIC_JOB_ID",
        "TASK2_LABELCRITIC_JOB_STATE",
        "TASK2_TEACHER_H100_POLICY_JSON",
        "LABELCRITIC_JOB_ID",
        "LABELCRITIC_SERVICE_ROOT",
    }:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "STARTING"])
def test_labelcritic_active_states_reserve_h100_when_teacher_overflow_allowed(tmp_path: Path, state: str):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="111111",
        labelcritic_job_state=state,
        slurm_job_state_fn=lambda job_id: {"state": state, "job_id": str(job_id), "name": "labelcritic_72b_service"},
    )

    assert policy["allow_h100_teacher_overflow"] is True
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["effective_teacher_h100_enabled"] is False
    assert policy["teacher_h100_deferred_for_labelcritic"] is True
    assert policy["labelcritic_job_state"] == state


def test_labelcritic_required_without_active_job_keeps_h100_reserved(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        find_labelcritic_job_fn=lambda: "",
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["effective_teacher_h100_enabled"] is False
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job"


def test_labelcritic_not_required_allows_opportunistic_teacher_h100(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=False,
        find_labelcritic_job_fn=lambda: "",
    )

    assert policy["labelcritic_h100_reserved"] is False
    assert policy["effective_teacher_h100_enabled"] is True
    assert policy["teacher_h100_deferred_for_labelcritic"] is False


def test_labelcritic_not_required_overrides_discovered_pending_job(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=False,
        find_labelcritic_job_fn=lambda: {"status": "SELECTED", "job_id": "987654", "state": "PENDING", "source": "test_discovery"},
        slurm_job_state_fn=lambda job_id: {"state": "PENDING", "job_id": str(job_id), "name": "labelcritic_72b_service"},
    )

    assert policy["labelcritic_job_found"] is True
    assert policy["labelcritic_job_state"] == "PENDING"
    assert policy["labelcritic_h100_reserved"] is False
    assert policy["effective_teacher_h100_enabled"] is True
    assert policy["reservation_reason"] == "labelcritic_not_required"


def test_overflow_env_zero_disables_h100_even_without_labelcritic_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW", "0")

    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        labelcritic_required=False,
        find_labelcritic_job_fn=lambda: "",
    )

    assert policy["allow_h100_teacher_overflow"] is False
    assert policy["labelcritic_h100_reserved"] is False
    assert policy["effective_teacher_h100_enabled"] is False
    assert policy["teacher_h100_deferred_for_labelcritic"] is False


def test_controller_restart_restores_reservation_from_service_state(tmp_path: Path):
    state_root = tmp_path / "state"
    service_root = state_root / "labelcritic_72b_service"
    service_root.mkdir(parents=True)
    (service_root / "service_state.json").write_text(
        json.dumps({"status": "REUSED_ACTIVE_JOB", "job_id": "222222", "validation": {"record": {"state": "PENDING"}}}),
        encoding="utf-8",
    )

    policy = resolve_teacher_h100_policy(
        state_root=state_root,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        slurm_job_state_fn=lambda job_id: {"state": "PENDING", "job_id": str(job_id), "name": "labelcritic_72b_service"},
    )

    assert policy["labelcritic_job_id"] == "222222"
    assert policy["labelcritic_job_state"] == "PENDING"
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["effective_teacher_h100_enabled"] is False


def test_labelcritic_discovery_does_not_hardcode_runtime_job_id(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        find_labelcritic_job_fn=lambda: {"status": "SELECTED", "job_id": "987654", "state": "RUNNING", "source": "test_discovery"},
        slurm_job_state_fn=lambda job_id: {"state": "RUNNING", "job_id": str(job_id), "name": "labelcritic_72b_service"},
    )

    assert policy["labelcritic_job_id"] == "987654"
    assert policy["labelcritic_job_id"] != "111111"
    assert policy["labelcritic_job_state"] == "RUNNING"
    assert policy["labelcritic_h100_reserved"] is True


def test_stale_persisted_labelcritic_pending_is_reconciled_with_slurm_terminal(tmp_path: Path):
    state_root = tmp_path / "state"
    service_root = state_root / "labelcritic_72b_service"
    service_root.mkdir(parents=True)
    (service_root / "service_state.json").write_text(
        json.dumps({"status": "REUSED_ACTIVE_JOB", "job_id": "111111", "validation": {"record": {"state": "PENDING"}}}),
        encoding="utf-8",
    )

    policy = resolve_teacher_h100_policy(
        state_root=state_root,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        slurm_job_state_fn=lambda job_id: {"state": "FAILED", "job_id": str(job_id), "name": "labelcritic_72b_service", "source": "sacct"},
    )

    assert policy["labelcritic_job_id"] == "111111"
    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_state"] == "FAILED"
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job"


def test_explicit_labelcritic_job_missing_in_live_slurm_is_not_active(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state="RUNNING",
        slurm_job_state_fn=lambda job_id: {"state": "UNKNOWN", "job_id": str(job_id), "source": "unknown"},
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_state"] == "UNKNOWN"
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job"


def test_sacct_terminal_labelcritic_job_is_not_active(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state="PENDING",
        slurm_job_state_fn=lambda job_id: {"state": "COMPLETED", "job_id": str(job_id), "name": "labelcritic_72b_service", "source": "sacct"},
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_state"] == "COMPLETED"
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job"


def test_labelcritic_job_name_mismatch_is_not_formal_service(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state="RUNNING",
        slurm_job_state_fn=lambda job_id: {"state": "RUNNING", "job_id": str(job_id), "name": "unrelated_h100_job", "source": "squeue"},
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_state"] == "RUNNING"
    assert policy["labelcritic_h100_reserved"] is True
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job:name_mismatch"


def test_acceptance_job_is_not_counted_as_formal_labelcritic_reservation(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        find_labelcritic_job_fn=lambda: {"status": "SELECTED", "job_id": "222222", "state": "PENDING", "name": "labelcritic_72b_acceptance", "source": "test_discovery"},
        slurm_job_state_fn=lambda job_id: {"state": "PENDING", "job_id": str(job_id), "name": "labelcritic_72b_acceptance"},
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_id"] == ""
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job"


def test_explicit_acceptance_job_id_is_not_counted_as_formal_labelcritic_reservation(tmp_path: Path):
    policy = resolve_teacher_h100_policy(
        state_root=tmp_path / "state",
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="4946921",
        labelcritic_job_state="PENDING",
        slurm_job_state_fn=lambda job_id: {"state": "PENDING", "job_id": str(job_id), "name": "labelcritic_72b_acceptance", "source": "squeue"},
    )

    assert policy["labelcritic_job_found"] is False
    assert policy["labelcritic_job_state"] == "PENDING"
    assert policy["reservation_reason"] == "labelcritic_required_without_active_job:name_mismatch"


def test_default_slurm_job_state_uses_sacct_terminal_when_squeue_missing(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(command):
        calls.append(command)
        if command[:3] == ["squeue", "-h", "-j"]:
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0}
        if command[:3] == ["sacct", "-X", "-n"]:
            return {"ok": True, "stdout": "777777|FAILED|labelcritic_72b_service\n", "stderr": "", "return_code": 0}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(h100_policy, "_run", fake_run)

    result = h100_policy.default_slurm_job_state("777777")

    assert result["state"] == "FAILED"
    assert result["name"] == "labelcritic_72b_service"
    assert result["source"] == "sacct"
    assert calls[1] == ["sacct", "-X", "-n", "-j", "777777", "--format=JobID,State,JobName", "-P"]
