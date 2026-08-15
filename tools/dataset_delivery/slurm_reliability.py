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
IMPOSSIBLE_PENDING_REASONS = {
    "PartitionTimeLimit",
    "DependencyNeverSatisfied",
    "BadConstraints",
    "InvalidAccount",
    "InvalidQOS",
    "InvalidPartition",
    "InvalidQoS",
    "QOSGrpGRES",
    "QOSMaxGRESPerUser",
}

LEGACY_CASE_FULL373_V1 = "legacy_case_full373_v1"
CANDIDATE_TASK_V1 = "candidate_task_v1"
DEFAULT_MANIFEST_SCHEMA_VERSION = "candidate_task_manifest_v1"
DEFAULT_LOGICAL_TASK_NAMESPACE = "case_id|canonical_target|eligible_teacher"

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
    "execution_attempt_id",
    "worker_generation",
    "submission_id",
    "job_id",
    "array_job_id",
    "array_task_id",
    "display_id",
    "group",
    "model_group",
    "profile",
    "shard_id",
    "shard_index",
    "partition",
    "gres",
    "task_manifest",
    "sbatch_file",
    "array_range",
    "array_spec",
    "array_concurrency",
    "task_count",
    "execution_schema_version",
    "manifest_schema_version",
    "logical_task_namespace",
    "logical_task_id",
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


def normalize_slurm_job_identifier(value: str) -> dict[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return {"job_id": "", "array_job_id": "", "array_task_id": "", "display_id": "", "query_id": ""}
    concrete = re.match(r"^(?P<array>\d+)_(?P<task>\d+)$", raw)
    if concrete:
        array_job_id = concrete.group("array")
        array_task_id = concrete.group("task")
        return {
            "job_id": f"{array_job_id}_{array_task_id}",
            "array_job_id": array_job_id,
            "array_task_id": array_task_id,
            "display_id": raw,
            "query_id": array_job_id,
        }
    compressed = re.match(r"^(?P<array>\d+)_\[(?P<display>.+)\]$", raw)
    if compressed:
        array_job_id = compressed.group("array")
        return {
            "job_id": "",
            "array_job_id": array_job_id,
            "array_task_id": "",
            "display_id": raw,
            "query_id": array_job_id,
        }
    if raw.isdigit():
        return {
            "job_id": raw,
            "array_job_id": raw,
            "array_task_id": "",
            "display_id": raw,
            "query_id": raw,
        }
    return {
        "job_id": "",
        "array_job_id": "",
        "array_task_id": "",
        "display_id": raw,
        "query_id": "",
    }


def parse_sbatch_job_id(stdout: str) -> dict[str, str]:
    raw = str(stdout or "").splitlines()[-1].strip() if str(stdout or "").strip() else ""
    raw = raw.split(";", 1)[0].strip()
    parsed = normalize_slurm_job_identifier(raw)
    if raw and not parsed["display_id"]:
        parsed["display_id"] = raw
    return parsed


def slurm_comment(
    *,
    run_id: str,
    submission_id: str,
    group: str,
    profile: str,
    execution_schema_version: str = "",
    shard_id: str = "",
) -> str:
    safe = []
    if execution_schema_version:
        values = (run_id, execution_schema_version, submission_id, profile, shard_id or "na")
    else:
        values = (run_id, submission_id, group, profile)
    for value in values:
        safe.append(re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "na")).strip("_") or "na")
    return "medical_agent:" + ":".join(safe)


def parse_slurm_comment(comment: str) -> dict[str, str]:
    raw = str(comment or "").strip()
    prefix = "medical_agent:"
    if not raw.startswith(prefix):
        return {
            "run_id": "",
            "execution_schema_version": "",
            "submission_id": "",
            "group": "",
            "profile": "",
            "shard_id": "",
            "comment": raw,
        }
    parts = raw[len(prefix):].split(":")
    if len(parts) >= 5 and parts[1] == CANDIDATE_TASK_V1:
        return {
            "run_id": parts[0],
            "execution_schema_version": parts[1],
            "submission_id": parts[2],
            "group": "",
            "profile": parts[3],
            "shard_id": parts[4],
            "comment": raw,
        }
    return {
        "run_id": parts[0] if len(parts) > 0 else "",
        "execution_schema_version": "",
        "submission_id": parts[1] if len(parts) > 1 else "",
        "group": parts[2] if len(parts) > 2 else "",
        "profile": parts[3] if len(parts) > 3 else "",
        "shard_id": "",
        "comment": raw,
    }


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
        parsed = normalize_slurm_job_identifier(str(row.get("job_id") or row.get("display_id") or ""))
        if parsed["job_id"]:
            row["job_id"] = parsed["job_id"]
        if parsed["array_job_id"] and not row.get("array_job_id"):
            row["array_job_id"] = parsed["array_job_id"]
        if parsed["array_task_id"] and not row.get("array_task_id"):
            row["array_task_id"] = parsed["array_task_id"]
        if parsed["display_id"] and not row.get("display_id"):
            row["display_id"] = parsed["display_id"]
        key = (
            str(row.get("job_id") or row.get("display_id") or ""),
            str(row.get("submission_id") or ""),
            str(row.get("model_group") or row.get("group") or ""),
            str(row.get("profile") or row.get("shard_id") or ""),
        )
        deduped[key] = row
    return list(deduped.values())


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    job_id = str(row.get("job_id") or "")
    if job_id:
        return ("job", job_id, "", "", "")
    return (
        "logical",
        str(row.get("execution_schema_version") or ""),
        str(row.get("submission_id") or ""),
        str(row.get("model_group") or row.get("group") or ""),
        str(row.get("profile") or row.get("shard_id") or ""),
        str(row.get("logical_task_id") or row.get("task_manifest") or ""),
    )


def persist_submitted_job(slurm_root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    existing = load_submitted_jobs(slurm_root)
    normalized = {field: str(row.get(field, "")) for field in SUBMITTED_JOB_FIELDS}
    parsed = normalize_slurm_job_identifier(normalized.get("job_id") or normalized.get("display_id") or "")
    if parsed["job_id"]:
        normalized["job_id"] = parsed["job_id"]
    if parsed["array_job_id"]:
        normalized["array_job_id"] = normalized.get("array_job_id") or parsed["array_job_id"]
    if parsed["array_task_id"]:
        normalized["array_task_id"] = normalized.get("array_task_id") or parsed["array_task_id"]
    if parsed["display_id"]:
        normalized["display_id"] = normalized.get("display_id") or parsed["display_id"]
    normalized["execution_schema_version"] = normalized.get("execution_schema_version") or LEGACY_CASE_FULL373_V1
    normalized["manifest_schema_version"] = normalized.get("manifest_schema_version") or DEFAULT_MANIFEST_SCHEMA_VERSION
    normalized["logical_task_namespace"] = normalized.get("logical_task_namespace") or DEFAULT_LOGICAL_TASK_NAMESPACE
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


def job_query_id(row: dict[str, Any]) -> str:
    parsed = normalize_slurm_job_identifier(str(row.get("job_id") or row.get("display_id") or ""))
    return parsed["query_id"]


def is_terminal_slurm_state(state: str) -> bool:
    return str(state or "").strip().upper() in TERMINAL_STATES


def is_impossible_pending_reason(reason: str) -> bool:
    raw = str(reason or "").strip()
    lower = raw.lower()
    return any(str(item).lower() == lower or str(item).lower() in lower for item in IMPOSSIBLE_PENDING_REASONS)


def current_worker_accounting_from_rows(
    rows: list[dict[str, Any]],
    *,
    profiles: set[str] | None = None,
    execution_attempt_id: str = "",
    job_state_fn: Any | None = None,
) -> dict[str, Any]:
    profile_names = set(profiles or [])
    counts: dict[str, dict[str, int]] = {}

    def ensure_profile(name: str) -> dict[str, int]:
        if name not in counts:
            counts[name] = {
                "running": 0,
                "pending": 0,
                "active": 0,
                "valid_pending": 0,
                "transitional": 0,
                "terminal": 0,
                "invalid_pending": 0,
                "unknown": 0,
                "lifecycle_adopted": 0,
                "lifecycle_terminal": 0,
            }
        return counts[name]

    for name in profile_names:
        ensure_profile(name)

    filtered = []
    for row in rows:
        schema = str(row.get("execution_schema_version") or "")
        profile = str(row.get("profile") or "")
        if schema != CANDIDATE_TASK_V1:
            continue
        if execution_attempt_id and str(row.get("execution_attempt_id") or "") != str(execution_attempt_id):
            continue
        if profile_names and profile not in profile_names:
            continue
        filtered.append(row)

    child_arrays = {str(row.get("array_job_id") or "").strip() for row in filtered if str(row.get("array_task_id") or "").strip()}
    seen_units: set[tuple[str, str, str]] = set()
    reconciled_rows: list[dict[str, Any]] = []
    for row in filtered:
        profile = str(row.get("profile") or "")
        profile_counts = ensure_profile(profile)
        scheduler_status = str(row.get("scheduler_status") or "")
        if scheduler_status == "ADOPTED_ACTIVE_JOB":
            profile_counts["lifecycle_adopted"] += 1
        job_id = str(row.get("job_id") or row.get("display_id") or "")
        array_job_id = str(row.get("array_job_id") or job_id.split("_", 1)[0]).strip()
        array_task_id = str(row.get("array_task_id") or "").strip()
        if not array_task_id and array_job_id in child_arrays:
            continue
        slurm = job_state_fn(row) if job_state_fn else {}
        units = slurm.get("units") if isinstance(slurm, dict) and isinstance(slurm.get("units"), list) else []
        if not units:
            units = [
                {
                    "job_id": job_id,
                    "array_job_id": array_job_id,
                    "array_task_id": array_task_id,
                    "state": (slurm or {}).get("state") or row.get("slurm_state") or "UNKNOWN",
                    "reason": (slurm or {}).get("reason") or row.get("failure_reason") or "",
                    "unit_count": 1,
                }
            ]
        row_running = row_pending = row_active = row_terminal = row_invalid_pending = row_unknown = row_transitional = 0
        last_state = ""
        last_reason = ""
        for unit in units:
            state = str(unit.get("state") or "UNKNOWN").split("+", 1)[0].strip().upper()
            reason = str(unit.get("reason") or "")
            parsed = normalize_slurm_job_identifier(str(unit.get("job_id") or job_id))
            unit_array = str(unit.get("array_job_id") or parsed.get("array_job_id") or array_job_id or parsed.get("job_id") or "").strip()
            unit_task = str(unit.get("array_task_id") or parsed.get("array_task_id") or array_task_id or "").strip()
            unit_key = (profile, unit_array or str(unit.get("job_id") or job_id), unit_task or str(unit.get("display_id") or unit.get("job_id") or job_id or "parent"))
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            unit_count = max(1, int(unit.get("unit_count") or 1))
            last_state = state
            last_reason = reason
            if state in TERMINAL_STATES:
                profile_counts["terminal"] += unit_count
                row_terminal += unit_count
                if scheduler_status == "ADOPTED_ACTIVE_JOB":
                    profile_counts["lifecycle_terminal"] += unit_count
                continue
            if state == "PENDING" and is_impossible_pending_reason(reason):
                profile_counts["invalid_pending"] += unit_count
                row_invalid_pending += unit_count
                continue
            if state == "RUNNING":
                profile_counts["running"] += unit_count
                profile_counts["active"] += unit_count
                row_running += unit_count
                row_active += unit_count
                continue
            if state == "PENDING":
                profile_counts["pending"] += unit_count
                profile_counts["valid_pending"] += unit_count
                profile_counts["active"] += unit_count
                row_pending += unit_count
                row_active += unit_count
                continue
            if state in ACTIVE_STATES:
                profile_counts["transitional"] += unit_count
                profile_counts["active"] += unit_count
                row_transitional += unit_count
                row_active += unit_count
                continue
            profile_counts["unknown"] += unit_count
            row_unknown += unit_count
        reconciled = dict(row)
        reconciled.update(
            {
                "slurm_state": last_state or str(row.get("slurm_state") or "UNKNOWN"),
                "slurm_reason": last_reason,
                "scheduler_current_running": row_running,
                "scheduler_current_pending": row_pending,
                "scheduler_current_active": row_active,
                "scheduler_current_terminal": row_terminal,
                "scheduler_current_invalid_pending": row_invalid_pending,
                "scheduler_current_unknown": row_unknown,
                "scheduler_current_transitional": row_transitional,
            }
        )
        reconciled_rows.append(reconciled)
    return {"counts": counts, "rows": reconciled_rows}


def reconcile_submitted_worker_accounting(
    slurm_root: Path,
    *,
    profiles: set[str] | None = None,
    execution_attempt_id: str = "",
    job_state_fn: Any | None = None,
) -> dict[str, Any]:
    return current_worker_accounting_from_rows(
        load_submitted_jobs(slurm_root),
        profiles=profiles,
        execution_attempt_id=execution_attempt_id,
        job_state_fn=job_state_fn,
    )


def existing_active_logical_keys(slurm_root: Path, *, execution_attempt_id: str = "", job_state_fn: Any | None = None) -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    reconciled = reconcile_submitted_worker_accounting(slurm_root, execution_attempt_id=execution_attempt_id, job_state_fn=job_state_fn)
    for row in reconciled.get("rows") or []:
        status = str(row.get("submission_status") or row.get("status") or "").lower()
        scheduler_status = str(row.get("scheduler_status") or "")
        schema = str(row.get("execution_schema_version") or "")
        if schema != CANDIDATE_TASK_V1:
            continue
        if int(row.get("scheduler_current_active") or 0) <= 0:
            continue
        if status not in {"submitted", "adopted", "reused"} and scheduler_status not in {"ACTIVE", "ADOPTED_ACTIVE_JOB"}:
            continue
        key = (
            schema,
            str(row.get("submission_id") or ""),
            str(row.get("model_group") or row.get("group") or ""),
            str(row.get("profile") or ""),
            str(row.get("shard_id") or ""),
        )
        keys.add(key)
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
