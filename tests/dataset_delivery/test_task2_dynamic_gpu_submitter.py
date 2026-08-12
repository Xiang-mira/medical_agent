from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest


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
    assert plan["group_concurrency"] == {"cads": 18, "atm": 6, "airrc": 8, "unest": 8}
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert {row["gres"] for row in rows} == {"gpu:1"}
    assert "gpu:T4:1" not in (tmp_path / "slurm" / "dynamic" / "cads_generic_gpu_task2_array.sbatch").read_text(encoding="utf-8")


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
    shard_paths = [Path(row["task_manifest"]) for row in plan["jobs"]]
    source_indices: list[str] = []
    for path in shard_paths:
        rows = _rows(path)
        assert [row["task_index"] for row in rows] == [str(index) for index in range(len(rows))]
        source_indices.extend(row["source_task_index"] for row in rows)
    assert sorted(source_indices, key=int) == [str(index) for index in range(8)]


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
    )

    assert plan["planned_target_workers"] == 64
    assert plan["planned_overrequest_workers"] == 80
    assert plan["worker_sizing"]["fixed_30_ceiling"] is False
    assert {profile["partition"] for profile in plan["profile_specs"]} == {"gpu", "gpua100", "spare_l40", "gpuh100"}
    shard_paths = [Path(row["task_manifest"]) for row in plan["jobs"]]
    source_indices: list[str] = []
    for path in shard_paths:
        source_indices.extend(row["source_task_index"] for row in _rows(path))
    assert sorted(source_indices, key=int) == [str(index) for index in range(80)]


def test_dynamic_submitter_partial_sbatch_success_persists_before_qos_backpressure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 2, "atm": 2})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if command[:2] == ["bash", "-n"] or command[:2] == ["sbatch", "--test-only"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--parsable" in command and "cads" in str(command[-1]):
            return subprocess.CompletedProcess(command, 0, "91001\n", "")
        if "--parsable" in command and "atm" in str(command[-1]):
            return subprocess.CompletedProcess(command, 1, "", "Batch job submission failed: Job violates accounting/QOS policy")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tools.dataset_delivery.task2_dynamic_gpu_submitter.subprocess.run", fake_run)

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
            "job_id": "91003",
            "model_group": "cads",
            "group": "cads",
            "profile": "generic_gpu",
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
    assert "atm" in submitted[0]
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert {row["model_group"] for row in rows} == {"cads", "atm"}


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
