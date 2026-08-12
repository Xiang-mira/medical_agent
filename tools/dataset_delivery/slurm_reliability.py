from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from tools.dataset_delivery.delivery_lib import read_csv_rows, utc_now


ACTIVE_STATES = {"PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
SUCCESS_STATES = {"COMPLETED"}
RETRYABLE_TERMINAL_STATES = {"TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY", "OOM"}
FATAL_TERMINAL_STATES = {"FAILED", "CANCELLED", "BOOT_FAIL", "DEADLINE"}
TERMINAL_STATES = SUCCESS_STATES | RETRYABLE_TERMINAL_STATES | FATAL_TERMINAL_STATES

BACKPRESSURE_PATTERNS = (
    "QOSMaxSubmitJobPerUserLimit",
    "QOSMaxJobsPerUserLimit",
    "AssocMaxSubmitJobLimit",
    "AssocMaxJobsLimit",
    "MaxSubmitJobPerUserLimit",
    "Job violates accounting/QOS policy",
    "JobArrayTaskLimit",
    "Resources",
    "Priority",
    "ReqNodeNotAvail",
    "PartitionConfig",
    "temporarily unavailable",
    "Socket timed out",
)
FATAL_SUBMIT_PATTERNS = (
    "Invalid account",
    "Invalid partition",
    "Invalid qos",
    "Invalid generic resource",
    "invalid gres",
    "Invalid node name",
    "Unable to open file",
    "No such file or directory",
)

SUBMITTED_JOB_FIELDS = [
    "run_id",
    "submission_id",
    "job_id",
    "array_job_id",
    "group",
    "model_group",
    "profile",
    "partition",
    "gres",
    "task_manifest",
    "sbatch_file",
    "array_range",
    "array_concurrency",
    "task_count",
    "status",
    "submission_status",
    "scheduler_status",
    "slurm_state",
    "failure_reason",
    "stdout",
    "stderr",
    "comment",
    "formal_root",
    "state_root",
    "git_commit",
    "submitted_at",
    "updated_at",
]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def classify_sbatch_failure(text: str) -> dict[str, str]:
    message = str(text or "")
    for pattern in BACKPRESSURE_PATTERNS:
        if pattern.lower() in message.lower():
            return {"class": "TRANSIENT_RESOURCE_BACKPRESSURE", "reason": pattern}
    for pattern in FATAL_SUBMIT_PATTERNS:
        if pattern.lower() in message.lower():
            return {"class": "FATAL_SUBMISSION_ERROR", "reason": pattern}
    return {"class": "FATAL_SUBMISSION_ERROR", "reason": "unclassified_sbatch_failure"}


def parse_sbatch_job_id(stdout: str) -> dict[str, str]:
    raw = str(stdout or "").splitlines()[-1].strip() if str(stdout or "").strip() else ""
    raw = raw.split(";", 1)[0].strip()
    array_job_id = raw.split("_", 1)[0].strip()
    return {"job_id": raw, "array_job_id": array_job_id}


def slurm_comment(*, run_id: str, submission_id: str, group: str, profile: str) -> str:
    safe = []
    for value in (run_id, submission_id, group, profile):
        safe.append(re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "na")).strip("_") or "na")
    return "medical_agent:" + ":".join(safe)


def load_submitted_jobs(slurm_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    csv_path = slurm_root / "submitted_jobs.csv"
    if csv_path.exists():
        rows.extend(read_csv_rows(csv_path))
    json_path = slurm_root / "submitted_jobs.json"
    if json_path.exists():
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(doc, list):
                rows.extend({str(k): str(v) for k, v in item.items()} for item in doc if isinstance(item, dict))
            elif isinstance(doc, dict) and isinstance(doc.get("jobs"), list):
                rows.extend({str(k): str(v) for k, v in item.items()} for item in doc["jobs"] if isinstance(item, dict))
        except Exception:
            pass
    deduped: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("job_id") or ""),
            str(row.get("submission_id") or ""),
            str(row.get("model_group") or row.get("group") or ""),
            str(row.get("profile") or ""),
        )
        deduped[key] = row
    return list(deduped.values())


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    job_id = str(row.get("job_id") or "")
    if job_id:
        return ("job", job_id, "", "", "")
    return (
        "logical",
        str(row.get("submission_id") or ""),
        str(row.get("model_group") or row.get("group") or ""),
        str(row.get("profile") or ""),
        str(row.get("task_manifest") or ""),
    )


def persist_submitted_job(slurm_root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    existing = load_submitted_jobs(slurm_root)
    normalized = {field: str(row.get(field, "")) for field in SUBMITTED_JOB_FIELDS}
    normalized["updated_at"] = utc_now()
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {_row_key(item): dict(item) for item in existing}
    by_key[_row_key(normalized)] = normalized
    rows = list(by_key.values())
    slurm_root.mkdir(parents=True, exist_ok=True)
    tmp_csv = slurm_root / f".submitted_jobs.csv.tmp.{os.getpid()}"
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMITTED_JOB_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(slurm_root / "submitted_jobs.csv")
    atomic_write_json(slurm_root / "submitted_jobs.json", {"updated_at": utc_now(), "jobs": rows})
    return rows


def append_submission_attempt(slurm_root: Path, row: dict[str, Any]) -> None:
    append_jsonl(slurm_root / "submission_attempts.jsonl", {"time": utc_now(), **row})


def existing_active_logical_keys(slurm_root: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in load_submitted_jobs(slurm_root):
        status = str(row.get("submission_status") or row.get("status") or "").lower()
        scheduler_status = str(row.get("scheduler_status") or "")
        slurm_state = str(row.get("slurm_state") or "")
        if status in {"submitted", "adopted", "reused"} or scheduler_status in {"ACTIVE", "ADOPTED_ACTIVE_JOB"} or slurm_state in ACTIVE_STATES:
            keys.add((
                str(row.get("submission_id") or ""),
                str(row.get("model_group") or row.get("group") or ""),
                str(row.get("profile") or ""),
            ))
            keys.add(("", str(row.get("model_group") or row.get("group") or ""), str(row.get("profile") or "")))
    return keys


def record_job_lifecycle(state_root: Path, payload: dict[str, Any]) -> None:
    root = state_root / "round1_orchestrated"
    event = {"time": utc_now(), **payload}
    append_jsonl(root / "job_lifecycle.jsonl", event)
    current_path = root / "job_lifecycle_current.json"
    current = {}
    if current_path.exists():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    jobs = current.get("jobs") if isinstance(current.get("jobs"), dict) else {}
    key = str(payload.get("logical_task_id") or payload.get("job_id") or payload.get("submission_id") or len(jobs))
    jobs[key] = {**jobs.get(key, {}), **event}
    current.update({"updated_at": utc_now(), "jobs": jobs})
    atomic_write_json(current_path, current)


def write_walltime_guard_event(state_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    event = {"time": utc_now(), **payload}
    root = state_root / "round1_orchestrated"
    append_jsonl(root / "walltime_guard.jsonl", event)
    atomic_write_json(root / "walltime_guard_last.json", event)
    return event


def worker_pretimeout(state_root: Path, *, logical_task_id: str = "", job_id: str = "", task_manifest: str = "", task_index: str = "") -> dict[str, Any]:
    payload = {
        "status": "RETRY_PENDING",
        "scheduler_state": "DRAINING_FOR_WALLTIME",
        "logical_task_id": logical_task_id,
        "job_id": job_id,
        "task_manifest": task_manifest,
        "task_index": task_index,
        "partial_publish_allowed": False,
    }
    record_job_lifecycle(state_root, payload)
    return write_walltime_guard_event(state_root, payload)


def student_pretimeout(state_root: Path, checkpoint_dir: Path, *, job_id: str = "") -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker = checkpoint_dir / "walltime_checkpoint_request.json"
    payload = {
        "status": "CHECKPOINTED_FOR_WALLTIME",
        "scheduler_state": "DRAINING_FOR_WALLTIME",
        "job_id": job_id,
        "checkpoint_path": str(marker),
        "resume_required": True,
    }
    atomic_write_json(marker, payload)
    record_job_lifecycle(state_root, payload)
    return write_walltime_guard_event(state_root, payload)


def time_extension_denied(state_root: Path, *, job_id: str, reason: str = "time_extension_unavailable") -> dict[str, Any]:
    return write_walltime_guard_event(
        state_root,
        {
            "status": "TIME_EXTENSION_UNAVAILABLE",
            "scheduler_state": "RETRY_PENDING",
            "job_id": str(job_id),
            "failure_reason": reason,
        },
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Task2 Slurm reliability helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    worker = sub.add_parser("worker-pretimeout")
    worker.add_argument("--state-root", required=True, type=Path)
    worker.add_argument("--logical-task-id", default="")
    worker.add_argument("--job-id", default="")
    worker.add_argument("--task-manifest", default="")
    worker.add_argument("--task-index", default="")
    student = sub.add_parser("student-pretimeout")
    student.add_argument("--state-root", required=True, type=Path)
    student.add_argument("--checkpoint-dir", required=True, type=Path)
    student.add_argument("--job-id", default="")
    args = parser.parse_args()
    if args.cmd == "worker-pretimeout":
        print(json.dumps(worker_pretimeout(args.state_root, logical_task_id=args.logical_task_id, job_id=args.job_id, task_manifest=args.task_manifest, task_index=args.task_index), indent=2))
    elif args.cmd == "student-pretimeout":
        print(json.dumps(student_pretimeout(args.state_root, args.checkpoint_dir, job_id=args.job_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
