from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from tools.dataset_delivery.slurm_reliability import CANDIDATE_TASK_V1


def _write_group_plan(root: Path, group: str, count: int) -> dict[str, str | int]:
    manifest = root / group / f"{group}_task_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_index", "case_id", "model_group", "status"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"task_index": index, "case_id": f"BDMAP_{index:08d}", "model_group": group, "status": "planned"})
    sbatch = root / "slurm" / f"{group}_task2_array.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name=task2_{group}\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --gres=gpu:T4:1\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --mem=64G\n"
        "#SBATCH --time=06:00:00\n"
        f"#SBATCH --output={root / 'slurm' / (group + '_%A_%a.out')}\n"
        f"#SBATCH --error={root / 'slurm' / (group + '_%A_%a.err')}\n"
        "cd /repo\n"
        "python tools/dataset_delivery/task2_formal_launcher.py \\\n"
        '  --execute-task-index "$SLURM_ARRAY_TASK_ID" \\\n'
        f"  --task-manifest {manifest} \\\n"
        f"  --output-root {root}\n",
        encoding="utf-8",
    )
    return {"task_count": count, "task_manifest": str(manifest), "sbatch_file": str(sbatch)}


def _summary(root: Path, counts: dict[str, int]) -> Path:
    groups = {group: _write_group_plan(root, group, count) for group, count in counts.items()}
    path = root / "formal_task2_submission_manifest.json"
    path.write_text(json.dumps({"status": "READY", "groups": groups}), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mock_slurm_units(monkeypatch: pytest.MonkeyPatch, states: dict[str, tuple[str, str, int] | list[dict[str, str | int]]]) -> None:
    def fake_units(row):
        job_id = str(row.get("job_id") or row.get("display_id") or "")
        value = states.get(job_id)
        if isinstance(value, list):
            return {"state": str(value[0].get("state") if value else "UNKNOWN"), "job_id": job_id, "units": value}
        state, reason, count = value or ("UNKNOWN", "", 1)
        return {"state": state, "reason": reason, "job_id": job_id, "units": [{"job_id": job_id, "state": state, "reason": reason, "unit_count": count}]}

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter._slurm_worker_units", fake_units)


def test_dynamic_submitter_overrequests_target_30_to_40_and_uses_generic_gpu(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 103, "atm": 103, "airrc": 103, "unest": 103})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=30,
        overrequest_workers=None,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads", "atm", "airrc", "unest"],
        dry_run=True,
    )

    assert plan["status"] == "DRY_RUN"
    assert plan["planned_target_workers"] == 30
    assert plan["planned_overrequest_workers"] == 40
    assert plan["total_array_concurrency"] == 40
    assert plan["task_ownership"] == "shared_queue"
    assert plan["profile_binding"] is False
    assert plan["group_concurrency"] == {"cads": 18, "atm": 6, "airrc": 8, "unest": 8}
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert {row["gres"] for row in rows} == {"gpu:1"}
    assert "gpu:T4:1" not in (tmp_path / "slurm" / "dynamic" / "full373_generic_gpu_worker_shard_000_queue_worker.sbatch").read_text(encoding="utf-8")


def test_dynamic_submitter_profile_shards_do_not_duplicate_source_task_indices(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 8})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4,
        overrequest_workers=6,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00,a100|gpu|gpu:A100:1|8|80G|06:00:00",
        groups=["cads"],
        dry_run=True,
    )

    assert plan["total_array_concurrency"] == 6
    assert plan["task_ownership"] == "shared_queue"
    assert plan["profile_binding"] is False
    assert plan["t4_only_reachability_logical_task_count"] == 8
    manifest_paths = {Path(row["task_manifest"]) for row in plan["jobs"]}
    assert len(manifest_paths) == 1
    rows = _rows(next(iter(manifest_paths)))
    assert sorted((row["source_task_index"] for row in rows), key=int) == [str(index) for index in range(8)]


def test_dynamic_submitter_replenishes_current_attempt_when_only_historical_workers_remain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 4})
    (tmp_path / "slurm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "slurm" / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,4584475,cads,generic_gpu,4,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n",
        encoding="utf-8",
    )
    _mock_slurm_units(monkeypatch, {"4584475": ("CANCELLED", "", 4)})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=2,
        overrequest_workers=2,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=True,
        run_id="run_a",
        submission_id="batch_b",
        execution_attempt_id="attempt_b",
    )

    assert plan["status"] == "DRY_RUN"
    assert plan["execution_attempt_id"] == "attempt_b"
    assert plan["worker_pool"]["existing_workers_by_profile"]["generic_gpu"]["active"] == 0
    assert plan["total_array_concurrency"] == 2
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert any(row["execution_attempt_id"] == "attempt_b" for row in rows)


def test_dynamic_submitter_preflight_failure_submits_no_partial_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2, "atm": 2})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if command[:2] == ["bash", "-n"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 1, "", "invalid gres")
        if "--parsable" in command:
            return subprocess.CompletedProcess(command, 0, "999", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="no jobs were submitted"):
        build_dynamic_submission_plan(
            summary_path=summary,
            output_root=tmp_path,
            state_root=tmp_path / "state",
            target_workers=2,
            overrequest_workers=3,
            profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
            groups=["cads", "atm"],
            dry_run=False,
        )

    assert not any("--parsable" in call for call in calls)
    plan = json.loads((tmp_path / "slurm" / "dynamic_gpu_submission_plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "PREFLIGHT_FAILED"


def test_dynamic_submitter_auto_profiles_use_cluster_inventory_without_fixed_30_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 12, "gpus_configured_total": 12},
            "gpua100": {"partition": "gpua100", "gpu_type": "A100", "allocatable_configured_total": 24, "gpus_configured_total": 24},
            "spare_l40": {"partition": "spare_l40", "gpu_type": "L40S", "allocatable_configured_total": 20, "gpus_configured_total": 20},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"cads": 80})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["cads"],
        dry_run=True,
        labelcritic_required=False,
    )

    assert plan["planned_target_workers"] == 64
    assert plan["planned_overrequest_workers"] == 80
    assert plan["worker_sizing"]["fixed_30_ceiling"] is False
    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu", "gpua100", "spare_l40", "gpuh100"}
    assert [profile["resource_class"] for profile in plan["profile_specs"]][0] == "GPU_LIGHT_T4"
    assert plan["task_ownership"] == "shared_queue"
    assert plan["profile_binding"] is False
    manifest_paths = {Path(row["task_manifest"]) for row in plan["jobs"]}
    assert len(manifest_paths) == 1
    source_indices = [row["source_task_index"] for row in _rows(next(iter(manifest_paths)))]
    assert sorted(source_indices, key=int) == [str(index) for index in range(80)]


def test_interactive_t4_auto_profile_uses_short_walltime_and_idle_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    monkeypatch.setenv("RESOURCE_ROUTING_MODE", "enforce")
    snapshot = {
        "partitions": {
            "interactive": {
                "partition": "interactive",
                "gpu_type": "T4",
                "allocatable_configured_total": 10,
                "gpus_configured_total": 10,
                "idle_estimate": 8,
                "gpus_idle_estimate": 8,
                "time_limit": "04:00:00",
            }
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 20})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        labelcritic_required=True,
        labelcritic_job_id="111111",
        labelcritic_job_state="PENDING",
    )

    profile = plan["profile_specs"][0]
    report = plan["worker_pool"]["profile_reports"][0]
    assert profile["name"] == "interactive_t4_short"
    assert profile["partition"] == "interactive"
    assert profile["time_limit"] == "03:50:00"
    assert report["desired_workers"] > 1
    assert report["desired_workers"] <= 8
    assert plan["total_array_concurrency"] == report["desired_workers"]


def test_dynamic_submitter_subtracts_existing_interactive_workers_without_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    monkeypatch.setenv("RESOURCE_ROUTING_MODE", "enforce")
    snapshot = {
        "partitions": {
            "interactive": {
                "partition": "interactive",
                "gpu_type": "T4",
                "allocatable_configured_total": 10,
                "gpus_configured_total": 10,
                "idle_estimate": 8,
                "gpus_idle_estimate": 8,
                "time_limit": "04:00:00",
            }
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    _mock_slurm_units(monkeypatch, {"700001": ("RUNNING", "", 3), "700002": ("PENDING", "Resources", 2)})
    (tmp_path / "slurm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "slurm" / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,700001,700001,,full373,interactive_t4_short,3,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n"
        f"run_a,attempt_a,batch_a,700002,700002,,full373,interactive_t4_short,2,{CANDIDATE_TASK_V1},submitted,ACTIVE,PENDING\n",
        encoding="utf-8",
    )
    summary = _summary(tmp_path, {"full373": 20})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        run_id="run_a",
        submission_id="batch_b",
        execution_attempt_id="attempt_a",
        labelcritic_required=True,
        labelcritic_job_id="111111",
        labelcritic_job_state="PENDING",
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["running_workers"] == 3
    assert report["pending_workers"] == 2
    assert report["active_workers"] == 5
    assert report["new_worker_deficit"] == report["desired_workers"] - 5
    assert plan["total_array_concurrency"] == report["new_worker_deficit"]


def test_worker_pool_counts_child_array_tasks_without_double_counting_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import GpuSubmitProfile, _worker_pool_counts

    slurm_root = tmp_path / "slurm"
    slurm_root.mkdir()
    (slurm_root / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,800000,800000,,full373,gpu_t4,10,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n"
        f"run_a,attempt_a,batch_a,800000_0,800000,0,full373,gpu_t4,1,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n"
        f"run_a,attempt_a,batch_a,800000_1,800000,1,full373,gpu_t4,1,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n",
        encoding="utf-8",
    )
    _mock_slurm_units(monkeypatch, {"800000_0": ("RUNNING", "", 1), "800000_1": ("RUNNING", "", 1), "800000": ("RUNNING", "", 10)})

    counts = _worker_pool_counts(
        slurm_root,
        [GpuSubmitProfile("gpu_t4", "gpu", "gpu:T4:1", 8, "64G", "06:00:00")],
        execution_attempt_id="attempt_a",
    )

    assert counts["gpu_t4"]["running"] == 2
    assert counts["gpu_t4"]["active"] == 2


def test_impossible_pending_partition_time_limit_does_not_count_as_active_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import GpuSubmitProfile, _worker_pool_counts

    slurm_root = tmp_path / "slurm"
    slurm_root.mkdir()
    (slurm_root / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,900000,900000,,full373,interactive_t4_short,6,{CANDIDATE_TASK_V1},submitted,ACTIVE,PENDING\n",
        encoding="utf-8",
    )
    _mock_slurm_units(monkeypatch, {"900000": ("PENDING", "PartitionTimeLimit", 6)})

    counts = _worker_pool_counts(
        slurm_root,
        [GpuSubmitProfile("interactive_t4_short", "interactive", "gpu:T4:1", 8, "64G", "03:50:00")],
        execution_attempt_id="attempt_a",
    )

    assert counts["interactive_t4_short"]["active"] == 0
    assert counts["interactive_t4_short"]["invalid_pending"] == 6


def test_stale_adopted_completed_workers_do_not_suppress_replenishment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"full373": 200})
    slurm_root = tmp_path / "slurm"
    slurm_root.mkdir(parents=True, exist_ok=True)
    lines = ["run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state"]
    states = {}
    for index in range(75):
        job_id = f"880000_{index}"
        state = "RUNNING" if index < 5 else "COMPLETED"
        states[job_id] = (state, "", 1)
        lines.append(f"run_a,attempt_a,batch_old,{job_id},880000,{index},full373,gpu_t4,1,{CANDIDATE_TASK_V1},adopted,ADOPTED_ACTIVE_JOB,{state}")
    (slurm_root / "submitted_jobs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _mock_slurm_units(monkeypatch, states)

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=75,
        overrequest_workers=75,
        profile_specs="gpu_t4|gpu|gpu:T4:1|8|64G|06:00:00",
        groups=["full373"],
        dry_run=True,
        run_id="run_a",
        submission_id="batch_new",
        execution_attempt_id="attempt_a",
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["running_workers"] == 5
    assert report["pending_workers"] == 0
    assert report["active_workers"] == 5
    assert report["new_worker_deficit"] == 70
    assert plan["total_array_concurrency"] == 70


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"])
def test_terminal_slurm_states_never_count_as_active(tmp_path: Path, state: str):
    from tools.dataset_delivery.slurm_reliability import current_worker_accounting_from_rows

    row = {
        "run_id": "run_a",
        "execution_attempt_id": "attempt_a",
        "submission_id": "batch_a",
        "job_id": "990001",
        "model_group": "full373",
        "profile": "gpu_t4",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "submission_status": "adopted",
        "scheduler_status": "ADOPTED_ACTIVE_JOB",
        "slurm_state": "RUNNING",
    }
    result = current_worker_accounting_from_rows(
        [row],
        profiles={"gpu_t4"},
        execution_attempt_id="attempt_a",
        job_state_fn=lambda _row: {"state": state, "job_id": "990001", "units": [{"job_id": "990001", "state": state, "unit_count": 1}]},
    )

    counts = result["counts"]["gpu_t4"]
    assert counts["running"] == 0
    assert counts["pending"] == 0
    assert counts["active"] == 0
    assert counts["terminal"] == 1
    assert result["rows"][0]["scheduler_current_active"] == 0


def test_adopted_active_job_counts_only_when_slurm_running(tmp_path: Path):
    from tools.dataset_delivery.slurm_reliability import current_worker_accounting_from_rows

    row = {
        "run_id": "run_a",
        "execution_attempt_id": "attempt_a",
        "submission_id": "batch_a",
        "job_id": "990002",
        "model_group": "full373",
        "profile": "gpu_t4",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "submission_status": "adopted",
        "scheduler_status": "ADOPTED_ACTIVE_JOB",
    }
    result = current_worker_accounting_from_rows(
        [row],
        profiles={"gpu_t4"},
        execution_attempt_id="attempt_a",
        job_state_fn=lambda _row: {"state": "RUNNING", "job_id": "990002", "units": [{"job_id": "990002", "state": "RUNNING", "unit_count": 1}]},
    )

    assert result["counts"]["gpu_t4"]["running"] == 1
    assert result["counts"]["gpu_t4"]["active"] == 1


def test_valid_pending_counts_as_bounded_active_demand(tmp_path: Path):
    from tools.dataset_delivery.slurm_reliability import current_worker_accounting_from_rows

    row = {
        "run_id": "run_a",
        "execution_attempt_id": "attempt_a",
        "submission_id": "batch_a",
        "job_id": "990003",
        "model_group": "full373",
        "profile": "gpu_t4",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "submission_status": "submitted",
        "scheduler_status": "ACTIVE",
    }
    result = current_worker_accounting_from_rows(
        [row],
        profiles={"gpu_t4"},
        execution_attempt_id="attempt_a",
        job_state_fn=lambda _row: {"state": "PENDING", "reason": "Priority", "job_id": "990003", "units": [{"job_id": "990003", "state": "PENDING", "reason": "Priority", "unit_count": 4}]},
    )

    assert result["counts"]["gpu_t4"]["pending"] == 4
    assert result["counts"]["gpu_t4"]["valid_pending"] == 4
    assert result["counts"]["gpu_t4"]["active"] == 4


def test_squeue_missing_sacct_terminal_adopted_job_is_not_reused(tmp_path: Path):
    from tools.dataset_delivery.slurm_reliability import existing_active_logical_keys, persist_submitted_job

    persist_submitted_job(
        tmp_path / "slurm",
        {
            "run_id": "run_a",
            "execution_attempt_id": "attempt_a",
            "submission_id": "batch_a",
            "job_id": "990004",
            "model_group": "full373",
            "group": "full373",
            "profile": "gpu_t4",
            "shard_id": "worker_shard_000",
            "execution_schema_version": CANDIDATE_TASK_V1,
            "submission_status": "adopted",
            "scheduler_status": "ADOPTED_ACTIVE_JOB",
            "slurm_state": "RUNNING",
        },
    )

    keys = existing_active_logical_keys(
        tmp_path / "slurm",
        execution_attempt_id="attempt_a",
        job_state_fn=lambda _row: {"state": "COMPLETED", "source": "sacct", "units": [{"job_id": "990004", "state": "COMPLETED", "unit_count": 1}]},
    )

    assert keys == set()


def test_slurm_worker_units_parses_compressed_pending_array(monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import _slurm_worker_units

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return subprocess.CompletedProcess(command, 0, "990100_[0-9%4]|PENDING|Resources\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    result = _slurm_worker_units({"job_id": "990100", "array_job_id": "990100"})

    assert result["source"] == "squeue"
    assert result["units"][0]["state"] == "PENDING"
    assert result["units"][0]["unit_count"] == 10


def test_slurm_worker_units_uses_sacct_terminal_when_squeue_missing(monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import _slurm_worker_units

    def fake_run(command, **kwargs):
        if command[:2] == ["squeue", "-h"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["sacct", "-n"]:
            return subprocess.CompletedProcess(command, 0, "990101|COMPLETED|\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    result = _slurm_worker_units({"job_id": "990101"})

    assert result["source"] == "sacct"
    assert result["units"][0]["state"] == "COMPLETED"


def test_valid_pending_window_prevents_duplicate_replenishment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"full373": 200})
    slurm_root = tmp_path / "slurm"
    slurm_root.mkdir(parents=True, exist_ok=True)
    (slurm_root / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,991000,991000,,full373,gpu_t4,32,{CANDIDATE_TASK_V1},submitted,ACTIVE,PENDING\n",
        encoding="utf-8",
    )
    _mock_slurm_units(monkeypatch, {"991000": ("PENDING", "Resources", 32)})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=32,
        overrequest_workers=32,
        profile_specs="gpu_t4|gpu|gpu:T4:1|8|64G|06:00:00",
        groups=["full373"],
        dry_run=True,
        run_id="run_a",
        submission_id="batch_b",
        execution_attempt_id="attempt_a",
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["active_workers"] == 32
    assert report["new_worker_deficit"] == 0
    assert plan["total_array_concurrency"] == 0


def test_completed_workers_release_capacity_on_next_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import GpuSubmitProfile, _worker_pool_counts

    slurm_root = tmp_path / "slurm"
    slurm_root.mkdir()
    (slurm_root / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,992000,992000,,full373,gpu_t4,25,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n",
        encoding="utf-8",
    )
    profile = GpuSubmitProfile("gpu_t4", "gpu", "gpu:T4:1", 8, "64G", "06:00:00")
    _mock_slurm_units(monkeypatch, {"992000": ("RUNNING", "", 25)})
    first = _worker_pool_counts(slurm_root, [profile], execution_attempt_id="attempt_a")
    _mock_slurm_units(monkeypatch, {"992000": ("COMPLETED", "", 25)})
    second = _worker_pool_counts(slurm_root, [profile], execution_attempt_id="attempt_a")

    assert first["gpu_t4"]["active"] == 25
    assert second["gpu_t4"]["active"] == 0
    assert second["gpu_t4"]["terminal"] == 25


def test_no_backlog_submits_no_replenishment(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"full373": 0})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=32,
        overrequest_workers=32,
        profile_specs="gpu_t4|gpu|gpu:T4:1|8|64G|06:00:00",
        groups=["full373"],
        dry_run=True,
    )

    assert plan["logical_task_count"] == 0
    assert plan["total_array_concurrency"] == 0


def test_no_idle_gpu_still_maintains_bounded_pending_demand_when_partition_capable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {
                "partition": "gpu",
                "gpu_type": "T4",
                "allocatable_configured_total": 32,
                "gpus_configured_total": 32,
                "idle_estimate": 0,
                "gpus_idle_estimate": 0,
                "cpus_total": 256,
                "cpus_idle_estimate": 0,
                "memory_total_gb": 4096,
                "memory_idle_gb_estimate": 0,
            }
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 200})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=32,
        overrequest_workers=32,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["feasible_slots"]["feasible_slots"] == 0
    assert report["feasible_slots"]["capacity_slots"] == 32
    assert report["desired_workers"] == 32
    assert report["new_worker_deficit"] == 32


def test_t4_running_72_desired_114_deficit_42(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"full373": 200})
    (tmp_path / "slurm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "slurm" / "submitted_jobs.csv").write_text(
        "run_id,execution_attempt_id,submission_id,job_id,array_job_id,array_task_id,model_group,profile,task_count,execution_schema_version,submission_status,scheduler_status,slurm_state\n"
        f"run_a,attempt_a,batch_a,710000,710000,,full373,gpu_t4,72,{CANDIDATE_TASK_V1},submitted,ACTIVE,RUNNING\n",
        encoding="utf-8",
    )
    _mock_slurm_units(monkeypatch, {"710000": ("RUNNING", "", 72)})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=114,
        overrequest_workers=114,
        profile_specs="gpu_t4|gpu|gpu:T4:1|8|64G|06:00:00",
        groups=["full373"],
        dry_run=True,
        run_id="run_a",
        submission_id="batch_b",
        execution_attempt_id="attempt_a",
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["running_workers"] == 72
    assert report["desired_workers"] == 114
    assert report["new_worker_deficit"] == 42
    assert plan["total_array_concurrency"] == 42


def test_dynamic_submitter_can_defer_teacher_h100_when_labelcritic_reserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 4, "gpus_configured_total": 4},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 8})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state="PENDING",
    )

    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu"}
    assert plan["allow_h100_teacher_overflow"] is True
    assert plan["labelcritic_job_state"] == "PENDING"
    assert plan["labelcritic_h100_reserved"] is True
    assert plan["effective_teacher_h100_enabled"] is False
    assert plan["teacher_h100_deferred_for_labelcritic"] is True
    assert plan["desired_teacher_h100_workers"] == 0
    assert plan["resource_policy"]["teacher_resource_requirement"] == "GPU_INFERENCE_COMPATIBLE"
    assert plan["resource_policy"]["labelcritic_h100_reservation"] is True


@pytest.mark.parametrize("state", ["RUNNING", "STARTING"])
def test_dynamic_submitter_blocks_teacher_h100_for_labelcritic_runtime_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 1, "gpus_configured_total": 1},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 8})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state=state,
    )

    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu"}
    assert plan["labelcritic_job_state"] == state
    assert plan["labelcritic_h100_reserved"] is True
    assert plan["effective_teacher_h100_enabled"] is False
    assert plan["desired_teacher_h100_workers"] == 0
    assert not any("H100" in str(row.get("gres") or "").upper() for row in plan["jobs"])


def test_dynamic_submitter_keeps_h100_reserved_when_labelcritic_required_but_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 2, "gpus_configured_total": 2},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    monkeypatch.setattr(
        "tools.dataset_delivery.task2_h100_policy.default_find_labelcritic_job_by_name",
        lambda: {"status": "NONE"},
    )
    summary = _summary(tmp_path, {"full373": 8})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
    )

    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu"}
    assert plan["labelcritic_job_found"] is False
    assert plan["labelcritic_h100_reserved"] is True
    assert plan["effective_teacher_h100_enabled"] is False
    assert plan["desired_teacher_h100_workers"] == 0


def test_dynamic_submitter_allows_teacher_h100_when_labelcritic_not_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 1, "gpus_configured_total": 1},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 24})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        allow_h100_teacher_overflow=True,
        labelcritic_required=False,
    )

    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu", "gpuh100"}
    assert plan["labelcritic_h100_reserved"] is False
    assert plan["effective_teacher_h100_enabled"] is True
    assert plan["desired_teacher_h100_workers"] > 0
    assert any("H100" in str(row.get("gres") or "").upper() for row in plan["jobs"])


def test_dynamic_submitter_env_zero_blocks_teacher_h100_even_when_labelcritic_not_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 2, "gpus_configured_total": 2},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    monkeypatch.setenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW", "0")
    summary = _summary(tmp_path, {"full373": 8})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        labelcritic_required=False,
    )

    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu"}
    assert plan["allow_h100_teacher_overflow"] is False
    assert plan["labelcritic_h100_reserved"] is False
    assert plan["effective_teacher_h100_enabled"] is False
    assert plan["desired_teacher_h100_workers"] == 0


def test_dynamic_submitter_partial_sbatch_success_persists_before_qos_backpressure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2, "atm": 2})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command and "worker_shard_000" in str(command[-1]):
            return subprocess.CompletedProcess(command, 0, "91001\n", "")
        if "--parsable" in command and "worker_shard_001" in str(command[-1]):
            return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: Job violates accounting/QOS policy")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    monkeypatch.setenv("TASK2_GPU_T4_MAX_ARRAY_TASKS", "1")

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=2,
        overrequest_workers=2,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads", "atm"],
        dry_run=False,
        run_id="run_a",
        submission_id="batch_a",
    )

    assert plan["status"] == "PARTIALLY_SUBMITTED"
    assert plan["scheduler_status"] == "BACKPRESSURED"
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert [row["job_id"] for row in rows] == ["91001"]
    assert (tmp_path / "slurm" / "submitted_jobs.json").is_file()
    attempts = (tmp_path / "slurm" / "submission_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 2
    assert any("--comment" in call for call in calls if "--parsable" in call)


def test_dynamic_submitter_all_profiles_backpressured_waits_without_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2})

    def fake_run(command, **kwargs):
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command:
            return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: QOSMaxSubmitJobPerUserLimit")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=2,
        overrequest_workers=2,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=False,
        run_id="run_b",
        submission_id="batch_b",
    )

    assert plan["status"] == "WAITING_FOR_SUBMISSION_CAPACITY"
    assert plan["backpressured_count"] == 1
    assert _rows(tmp_path / "slurm" / "submitted_jobs.csv") == []
    assert json.loads((tmp_path / "slurm" / "submitted_jobs.json").read_text(encoding="utf-8"))["jobs"] == []


def test_dynamic_submitter_continues_other_profile_after_one_backpressured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 4})
    submitted: list[str] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command:
            sbatch_file = str(command[-1])
            submitted.append(sbatch_file)
            if "gpu_t4" in sbatch_file:
                return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: QOSMaxJobsPerUserLimit")
            return subprocess.CompletedProcess(command, 0, "91002\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    monkeypatch.setenv("TASK2_GPU_T4_MAX_ARRAY_TASKS", "1")

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=2,
        overrequest_workers=2,
        profile_specs="gpu_t4|gpu|gpu:T4:1|8|64G|06:00:00,gpu_a100|gpua100|gpu:A100:1|8|80G|06:00:00",
        groups=["cads"],
        dry_run=False,
        run_id="run_c",
        submission_id="batch_c",
    )

    assert plan["status"] == "PARTIALLY_SUBMITTED"
    assert len(submitted) == 2
    assert [row["profile"] for row in _rows(tmp_path / "slurm" / "submitted_jobs.csv")] == ["gpu_a100"]


def test_dynamic_submitter_restart_reuses_active_logical_job_without_duplicate_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.slurm_reliability import persist_submitted_job
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2, "atm": 2})
    persist_submitted_job(
        tmp_path / "slurm",
        {
            "run_id": "run_d",
            "submission_id": "batch_d",
            "execution_schema_version": CANDIDATE_TASK_V1,
            "job_id": "91003",
            "model_group": "full373",
            "group": "full373",
            "profile": "generic_gpu",
            "shard_id": "worker_shard_existing",
            "logical_task_id": "run_d:batch_d:candidate_task_v1:full373:generic_gpu:worker_shard_existing",
            "task_count": "1",
            "submission_status": "submitted",
            "scheduler_status": "ACTIVE",
            "slurm_state": "RUNNING",
        },
    )
    submitted: list[str] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command:
            submitted.append(str(command[-1]))
            return subprocess.CompletedProcess(command, 0, "91004\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    _mock_slurm_units(monkeypatch, {"91003": ("RUNNING", "", 1)})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=2,
        overrequest_workers=2,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads", "atm"],
        dry_run=False,
        run_id="run_d",
        submission_id="batch_d",
    )

    assert plan["status"] == "SUBMITTED"
    assert len(submitted) == 1
    assert "worker_shard_000" in submitted[0]
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert {row["model_group"] for row in rows} == {"full373"}


def test_dynamic_submitter_qos_slot_later_frees_and_remaining_workers_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2})
    attempt = {"count": 0}

    def fake_run(command, **kwargs):
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command:
            attempt["count"] += 1
            if attempt["count"] == 1:
                return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: JobArrayTaskLimit")
            return subprocess.CompletedProcess(command, 0, "91005\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    first = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=1,
        overrequest_workers=1,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=False,
        run_id="run_e",
        submission_id="batch_e",
    )
    second = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=1,
        overrequest_workers=1,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=False,
        run_id="run_e",
        submission_id="batch_e",
    )

    assert first["status"] == "WAITING_FOR_SUBMISSION_CAPACITY"
    assert second["status"] == "SUBMITTED"
    assert _rows(tmp_path / "slurm" / "submitted_jobs.csv")[0]["job_id"] == "91005"


def test_dynamic_submitter_shards_by_detected_max_array_size_and_conserves_logical_tasks(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 61697})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4,
        overrequest_workers=4,
        profile_specs="p0|gpu|gpu:1|8|64G|06:00:00,p1|gpu|gpu:1|8|64G|06:00:00,p2|gpu|gpu:1|8|64G|06:00:00,p3|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=True,
        max_array_size=4000,
    )

    assert plan["execution_schema_version"] == CANDIDATE_TASK_V1
    assert plan["max_array_size"] == 4000
    assert plan["logical_task_count"] == 61697
    assert plan["sharded_task_count"] == 61697
    assert plan["unique_logical_task_count"] == 61697
    assert plan["duplicate_logical_task_count"] == 0
    assert plan["missing_logical_task_count"] == 0
    assert plan["sharding_audit"]["all_array_specs_valid"] is True
    assert all(int(row["local_end"]) <= 3999 for row in plan["jobs"])


def test_dynamic_submitter_preserves_full_61697_t4_reachability_when_h100_reserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    snapshot = {
        "partitions": {
            "gpu": {"partition": "gpu", "gpu_type": "T4", "allocatable_configured_total": 4000, "gpus_configured_total": 4000},
            "gpuh100": {"partition": "gpuh100", "gpu_type": "H100", "allocatable_configured_total": 8, "gpus_configured_total": 8},
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    monkeypatch.setenv("TASK2_GPU_WORKER_SAFETY_CAP", "5000")
    summary = _summary(tmp_path, {"full373": 61697})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4000,
        overrequest_workers=4000,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        max_array_size=4000,
        allow_h100_teacher_overflow=True,
        labelcritic_required=True,
        labelcritic_job_id="777777",
        labelcritic_job_state="PENDING",
    )

    assert plan["logical_task_count"] == 61697
    assert plan["unique_logical_task_count"] == 61697
    assert plan["t4_only_reachability_logical_task_count"] == 61697
    assert plan["task_ownership"] == "shared_queue"
    assert plan["profile_binding"] is False
    assert plan["candidate_seed"]["logical_task_count"] == 61697
    assert plan["primary_teacher_profile"] == "gpu_t4"
    assert plan["desired_teacher_h100_workers"] == 0


def test_dynamic_submitter_worker_arrays_split_at_max_array_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 4001})
    monkeypatch.setenv("TASK2_GPU_WORKER_SAFETY_CAP", "5000")
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4001,
        overrequest_workers=4001,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=True,
        max_array_size=4000,
    )

    assert [row["array_spec"] for row in plan["jobs"]] == ["0-3999%4000", "0-0%1"]


def test_dynamic_submitter_test_only_uses_real_array_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 4001})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command:
            return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: QOSMaxSubmitJobPerUserLimit")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)
    monkeypatch.setenv("TASK2_GPU_WORKER_SAFETY_CAP", "5000")
    build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4001,
        overrequest_workers=4001,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads"],
        dry_run=False,
        max_array_size=4000,
    )

    test_only_calls = [call for call in calls if call[:2] == ["sbatch", "--test-only"]]
    assert any("--array=0-3999%4000" in call for call in test_only_calls)
    assert any("--array=0-0%1" in call for call in test_only_calls)


def test_profile_feasible_slots_respects_gpu_cpu_and_memory_limits(monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import GpuSubmitProfile, _profile_feasible_slots

    monkeypatch.setenv("NODE_MEMORY_SAFETY_MARGIN_GB", "8")
    inventory = {
        "snapshot": {
            "partitions": {
                "gpu": {
                    "idle_estimate": 5,
                    "cpus_idle_estimate": 44,
                    "memory_idle_gb_estimate": 72,
                }
            }
        }
    }

    standard64 = GpuSubmitProfile("gpu_t4", "gpu", "gpu:T4:1", 8, "64G", "06:00:00")
    lowmem16 = GpuSubmitProfile("gpu_t4_lowmem", "gpu", "gpu:T4:1", 8, "16G", "06:00:00")
    cpu_heavy = GpuSubmitProfile("gpu_t4_cpuheavy", "gpu", "gpu:T4:1", 24, "16G", "06:00:00")

    assert _profile_feasible_slots(standard64, inventory)["feasible_slots"] == 1
    assert _profile_feasible_slots(lowmem16, inventory)["feasible_slots"] == 4
    assert _profile_feasible_slots(cpu_heavy, inventory)["feasible_slots"] == 1


def test_dynamic_submitter_does_not_fallback_to_gpu_count_when_memory_blocks_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    monkeypatch.setenv("NODE_MEMORY_SAFETY_MARGIN_GB", "8")
    snapshot = {
        "partitions": {
            "gpu": {
                "partition": "gpu",
                "gpu_type": "T4",
                "allocatable_configured_total": 5,
                "gpus_configured_total": 5,
                "idle_estimate": 5,
                "cpus_idle_estimate": 44,
                "memory_idle_gb_estimate": 40,
            }
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 20})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=5,
        overrequest_workers=5,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["feasible_slots"]["memory_slots"] == 0
    assert report["desired_workers"] == 0
    assert plan["total_array_concurrency"] == 0


def test_interactive_t4_short_profile_is_shadowed_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    monkeypatch.delenv("RESOURCE_ROUTING_MODE", raising=False)
    snapshot = {
        "partitions": {
            "interactive": {
                "partition": "interactive",
                "gpu_type": "T4",
                "allocatable_configured_total": 10,
                "gpus_configured_total": 10,
                "idle_estimate": 8,
                "gpus_idle_estimate": 8,
                "cpus_idle_estimate": 80,
                "memory_idle_gb_estimate": 640,
                "time_limit": "04:00:00",
            }
        }
    }
    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.discover_resource_snapshot", lambda **kwargs: snapshot)
    summary = _summary(tmp_path, {"full373": 20})

    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=0,
        overrequest_workers=0,
        profile_specs="auto",
        groups=["full373"],
        dry_run=True,
        labelcritic_required=True,
        labelcritic_job_id="111111",
        labelcritic_job_state="PENDING",
    )

    report = plan["worker_pool"]["profile_reports"][0]
    assert report["profile"] == "interactive_t4_short"
    assert report["desired_workers"] == 0
    assert report["new_worker_deficit"] == 0
    assert plan["total_array_concurrency"] == 0
