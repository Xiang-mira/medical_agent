#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from tools.dataset_delivery.delivery_lib import read_csv_rows, utc_now, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.slurm_reliability import (  # noqa: E402
    CANDIDATE_TASK_V1,
    LEGACY_CASE_FULL373_V1,
    job_query_id,
    normalize_slurm_job_identifier,
    parse_slurm_comment,
    RETRYABLE_TERMINAL_STATES,
    classify_sbatch_failure,
    persist_submitted_job,
    record_job_lifecycle,
    slurm_comment,
    student_pretimeout,
)
from tools.dataset_delivery.task2_h100_policy import resolve_teacher_h100_policy  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import FORMAL_CASE_COUNT, build_formal_manifest, validate_formal_manifest  # noqa: E402
from tools.dataset_delivery.task2_full373_round1_launcher import (  # noqa: E402
    FULL373_GROUP,
    FULL373_ROOT_NAME,
    aggregate_full373_estep,
    build_estep_telemetry,
    build_full_round1_scope,
)
from tools.dataset_delivery.task2_workspace_staging import staged_case_status  # noqa: E402


LABELCRITIC_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "OOM", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
ACTIVE_STATES = {"PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
SUCCESS_STATES = {"COMPLETED"}
BACKPRESSURE_STATUSES = {"BACKPRESSURED", "PARTIALLY_SUBMITTED", "WAITING_FOR_SUBMISSION_CAPACITY", "READY_WORKER_QUEUE"}
TEACHER_JOB_GROUPS = {"cads", "atm", "airrc", "unest", "full373"}
LABELCRITIC_JOB_NAME = "labelcritic_72b_service"
STATIC_TEST_ENV_DROP = {
    "LABELCRITIC_JOB_ID",
    "RETRY_FAILED",
    "EXPECTED_GIT_COMMIT",
    "RUNTIME_NO_GIT",
    "SKIP_GIT_SYNC",
    "GPU_TARGET_WORKERS",
    "GPU_OVERREQUEST_WORKERS",
    "GPU_PROFILE_SPECS",
}
RUNTIME_GIT_ENV_DROP = {
    "DISPLAY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
}
ONE_BY_ONE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": command,
            "return_code": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"command": command, "return_code": 127, "stdout": "", "stderr": str(exc), "ok": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "return_code": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "timeout", "ok": False}


def sanitized_static_test_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in STATIC_TEST_ENV_DROP:
        env.pop(key, None)
    return env


def runtime_no_git_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in RUNTIME_GIT_ENV_DROP:
        env.pop(key, None)
    env["RUNTIME_NO_GIT"] = "1"
    env["SKIP_GIT_SYNC"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def normalize_labelcritic_endpoint(base_url: str, port: int) -> tuple[str, int]:
    base = re.sub(r"/v1/?$", "", (base_url or "http://localhost").rstrip("/"))
    match = re.match(r"^(https?://[^/:]+):(\d+)$", base)
    if match:
        return match.group(1), int(match.group(2))
    return base, int(port)


def _urlopen_json(url: str, *, timeout: int = 20, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def labelcritic_health(base_url: str, port: int, *, expected_model: str = LABELCRITIC_MODEL_ID) -> dict[str, Any]:
    base, resolved_port = normalize_labelcritic_endpoint(base_url, port)
    try:
        urllib.request.urlopen(f"{base}:{resolved_port}/health", timeout=10).read()
        models_doc = _urlopen_json(f"{base}:{resolved_port}/v1/models", timeout=20)
        models = [str(item.get("id") or "") for item in models_doc.get("data", []) if isinstance(item, dict)]
        return {
            "status": "READY" if expected_model in models else "MODEL_MISMATCH",
            "base_url": base,
            "port": resolved_port,
            "expected_model": expected_model,
            "served_models": models,
            "failure_reason": "" if expected_model in models else f"expected {expected_model}, served={models}",
        }
    except Exception as exc:
        return {
            "status": "OFFLINE",
            "base_url": base,
            "port": resolved_port,
            "expected_model": expected_model,
            "served_models": [],
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def labelcritic_runtime_preflight(base_url: str, port: int, *, expected_model: str = LABELCRITIC_MODEL_ID) -> dict[str, Any]:
    health = labelcritic_health(base_url, port, expected_model=expected_model)
    if health["status"] != "READY":
        return {"status": "FAILED", "stage": "health", "health": health, "failure_reason": health.get("failure_reason", "")}
    base, resolved_port = normalize_labelcritic_endpoint(base_url, port)
    image_url = f"data:image/png;base64,{ONE_BY_ONE_PNG}"
    payload = {
        "model": expected_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are LabelCritic. Compare mask1 and mask2. This is a runtime parse smoke. Reply exactly: MASK1"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        doc = _urlopen_json(f"{base}:{resolved_port}/v1/chat/completions", timeout=90, payload=payload)
        text = str((((doc.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
        parsed = "MASK1" in text.upper() or "1" == text.strip()
        return {
            "status": "PASSED" if parsed else "FAILED",
            "stage": "multi_image_parse_request",
            "health": health,
            "response_text": text[:500],
            "failure_reason": "" if parsed else "runtime response did not parse as MASK1",
        }
    except Exception as exc:
        return {"status": "FAILED", "stage": "multi_image_request", "health": health, "failure_reason": f"{type(exc).__name__}: {exc}"}


def slurm_job_state(job_id: str) -> dict[str, Any]:
    parsed = normalize_slurm_job_identifier(job_id)
    query_id = parsed.get("query_id") or ""
    if not query_id:
        return {"state": "UNKNOWN", "job_id": "", "array_job_id": "", "array_task_id": "", "display_id": str(job_id or ""), "source": "none"}
    squeue = _run(["squeue", "-h", "-j", str(query_id), "-o", "%i|%T"])
    if squeue["ok"] and squeue["stdout"].strip():
        first = squeue["stdout"].splitlines()[0].split("|")
        job_display = first[0].strip() if first else query_id
        return {
            "state": first[1].strip() if len(first) > 1 else "UNKNOWN",
            "job_id": parsed.get("job_id") or query_id,
            "array_job_id": parsed.get("array_job_id") or query_id,
            "array_task_id": parsed.get("array_task_id") or "",
            "display_id": job_display,
            "source": "squeue",
        }
    sacct = _run(["sacct", "-n", "-j", str(query_id), "--format=JobIDRaw,State", "-P"])
    if sacct["ok"] and sacct["stdout"].strip():
        preferred = parsed.get("job_id") or query_id
        chosen = None
        for line in sacct["stdout"].splitlines():
            parts = line.split("|")
            raw_id = parts[0].strip() if parts else ""
            if raw_id == preferred:
                chosen = parts
                break
            if chosen is None and raw_id == query_id:
                chosen = parts
        if chosen:
            return {
                "state": chosen[1].strip().split()[0] if len(chosen) > 1 else "UNKNOWN",
                "job_id": preferred,
                "array_job_id": parsed.get("array_job_id") or query_id,
                "array_task_id": parsed.get("array_task_id") or "",
                "display_id": parsed.get("display_id") or preferred,
                "source": "sacct",
            }
    return {
        "state": "UNKNOWN",
        "job_id": parsed.get("job_id") or query_id,
        "array_job_id": parsed.get("array_job_id") or query_id,
        "array_task_id": parsed.get("array_task_id") or "",
        "display_id": parsed.get("display_id") or str(job_id or ""),
        "source": "unknown",
        "squeue": squeue,
        "sacct": sacct,
    }


def _current_user() -> str:
    return os.getenv("USER") or os.getenv("LOGNAME") or ""


def slurm_job_record(job_id: str) -> dict[str, Any]:
    parsed = normalize_slurm_job_identifier(job_id)
    query_id = parsed.get("query_id") or ""
    if not query_id:
        return {"status": "INVALID", "state": "UNKNOWN", "job_id": "", "display_id": str(job_id or ""), "source": "none"}
    squeue = _run(["squeue", "-h", "-j", str(query_id), "-o", "%i|%T|%u|%j"])
    if squeue["ok"] and squeue["stdout"].strip():
        parts = squeue["stdout"].splitlines()[0].split("|")
        return {
            "status": "FOUND",
            "job_id": parsed.get("job_id") or query_id,
            "array_job_id": parsed.get("array_job_id") or query_id,
            "array_task_id": parsed.get("array_task_id") or "",
            "display_id": parts[0].strip() if len(parts) > 0 else (parsed.get("display_id") or query_id),
            "state": parts[1].strip() if len(parts) > 1 else "UNKNOWN",
            "user": parts[2].strip() if len(parts) > 2 else "",
            "name": parts[3].strip() if len(parts) > 3 else "",
            "source": "squeue",
        }
    sacct = _run(["sacct", "-n", "-j", str(query_id), "--format=JobIDRaw,State,User,JobName", "-P"])
    if sacct["ok"] and sacct["stdout"].strip():
        preferred = parsed.get("job_id") or query_id
        for line in sacct["stdout"].splitlines():
            parts = line.split("|")
            if parts and parts[0].strip() in {preferred, query_id}:
                return {
                    "status": "FOUND",
                    "job_id": preferred,
                    "array_job_id": parsed.get("array_job_id") or query_id,
                    "array_task_id": parsed.get("array_task_id") or "",
                    "display_id": parsed.get("display_id") or parts[0].strip(),
                    "state": parts[1].strip().split()[0] if len(parts) > 1 else "UNKNOWN",
                    "user": parts[2].strip() if len(parts) > 2 else "",
                    "name": parts[3].strip() if len(parts) > 3 else "",
                    "source": "sacct",
                }
    state = slurm_job_state(job_id)
    return {"status": "FOUND" if state.get("state") != "UNKNOWN" else "NOT_FOUND", **state, "user": "", "name": ""}


def slurm_job_timing(job_id: str) -> dict[str, Any]:
    parsed = normalize_slurm_job_identifier(job_id)
    query_id = parsed.get("query_id") or ""
    if not query_id:
        return {"job_id": "", "display_id": str(job_id or ""), "state": "UNKNOWN"}
    squeue = _run(["squeue", "-h", "-j", str(query_id), "-o", "%i|%T|%M|%l|%L|%e|%N"])
    if squeue["ok"] and squeue["stdout"].strip():
        parts = squeue["stdout"].splitlines()[0].split("|")
        return {
            "job_id": parsed.get("job_id") or query_id,
            "array_job_id": parsed.get("array_job_id") or query_id,
            "array_task_id": parsed.get("array_task_id") or "",
            "display_id": parts[0].strip() if len(parts) > 0 else (parsed.get("display_id") or query_id),
            "state": parts[1].strip() if len(parts) > 1 else "UNKNOWN",
            "elapsed": parts[2].strip() if len(parts) > 2 else "",
            "time_limit": parts[3].strip() if len(parts) > 3 else "",
            "time_left": parts[4].strip() if len(parts) > 4 else "",
            "end_time": parts[5].strip() if len(parts) > 5 else "",
            "node": parts[6].strip() if len(parts) > 6 else "",
            "source": "squeue",
        }
    state = slurm_job_state(job_id)
    return {"job_id": str(job_id), **state}


def _parse_teacher_job_name(name: str) -> tuple[str, str]:
    value = str(name or "")
    if not value.startswith("task2_"):
        return "", ""
    body = value[len("task2_") :]
    for group in TEACHER_JOB_GROUPS:
        prefix = f"{group}_"
        if body == group:
            return group, "default"
        if body.startswith(prefix):
            return group, body[len(prefix) :]
    return "", ""


def discover_active_medical_agent_teacher_jobs(*, formal_root: Path, state_root: Path, run_id: str = "") -> list[dict[str, Any]]:
    user = _current_user()
    command = ["squeue", "-h"]
    if user:
        command.extend(["-u", user])
    command.extend(["-o", "%i|%T|%u|%j|%Z|%k|%o"])
    result = _run(command)
    if not result["ok"] or not result["stdout"].strip():
        return []
    matches: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        parts = line.split("|")
        job_id = parts[0].strip() if len(parts) > 0 else ""
        state = parts[1].strip() if len(parts) > 1 else "UNKNOWN"
        job_user = parts[2].strip() if len(parts) > 2 else ""
        name = parts[3].strip() if len(parts) > 3 else ""
        workdir = parts[4].strip() if len(parts) > 4 else ""
        comment = parts[5].strip() if len(parts) > 5 else ""
        command_text = parts[6].strip() if len(parts) > 6 else ""
        group, profile = _parse_teacher_job_name(name)
        comment_meta = parse_slurm_comment(comment)
        combined = "\n".join([workdir, comment, command_text])
        comment_match = bool(run_id and str(comment_meta.get("run_id") or "") == str(run_id))
        root_match = str(formal_root) in combined or str(state_root) in combined
        if state not in ACTIVE_STATES or not group or (not comment_match and not root_match):
            continue
        execution_schema_version = str(comment_meta.get("execution_schema_version") or "")
        if execution_schema_version and execution_schema_version != CANDIDATE_TASK_V1:
            record_job_lifecycle(
                state_root,
                {
                    "status": "HISTORICAL_INCOMPATIBLE_EXECUTION",
                    "job_id": job_id,
                    "display_id": job_id,
                    "execution_schema_version": execution_schema_version,
                    "comment": comment,
                    "group": group,
                    "profile": profile or "default",
                },
            )
            continue
        if not execution_schema_version:
            continue
        parsed_job = normalize_slurm_job_identifier(job_id)
        matches.append(
            {
                "job_id": parsed_job.get("job_id") or parsed_job.get("array_job_id") or "",
                "array_job_id": parsed_job.get("array_job_id") or "",
                "array_task_id": parsed_job.get("array_task_id") or "",
                "display_id": parsed_job.get("display_id") or job_id,
                "slurm_state": state,
                "user": job_user,
                "job_name": name,
                "model_group": group,
                "group": group,
                "profile": str(comment_meta.get("profile") or profile or "default"),
                "shard_id": str(comment_meta.get("shard_id") or ""),
                "execution_schema_version": execution_schema_version,
                "workdir": workdir,
                "comment": comment,
                "command": command_text,
                "scheduler_status": "ADOPTED_ACTIVE_JOB",
                "submission_status": "adopted",
                "status": "ADOPTED_ACTIVE_JOB",
                "formal_root": str(formal_root),
                "state_root": str(state_root),
                "updated_at": utc_now(),
            }
        )
    return sorted(matches, key=lambda row: (str(row["model_group"]), str(row["profile"]), str(row["job_id"])))


def reconcile_active_teacher_jobs(args: argparse.Namespace, formal_root: Path | None = None) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    state = _load_state(state_root)
    run_id = state.get("run_id") or _run_id(state_root)
    resolved_formal_root = Path(str(formal_root or state.get("formal_root") or (_state_paths(state_root)["root"] / "formal_task2_round1"))).resolve()
    rows = discover_active_medical_agent_teacher_jobs(formal_root=resolved_formal_root, state_root=state_root, run_id=str(run_id))
    slurm_root = resolved_formal_root / "slurm"
    for row in rows:
        persist_submitted_job(slurm_root, {**row, "run_id": run_id, "git_commit": _git_commit()})
        record_job_lifecycle(
            state_root,
            {
                **row,
                "logical_task_id": f"{run_id}:adopted:{row['model_group']}:{row['profile']}:{row.get('shard_id') or 'na'}",
                "scheduler_state": "ADOPTED_ACTIVE_JOB",
                "attempt": "slurm_reconcile",
            },
        )
    if rows:
        _save_state(state_root, formal_root=str(resolved_formal_root), reconciled_teacher_jobs=rows, scheduler_status="ACTIVE")
    return {"status": "RECONCILED", "adopted_count": len(rows), "jobs": rows}


def refresh_lifecycle_for_submitted_jobs(args: argparse.Namespace, formal_root: Path) -> dict[str, Any]:
    jobs_csv = formal_root / "slurm" / "submitted_jobs.csv"
    rows = read_csv_rows(jobs_csv) if jobs_csv.exists() else []
    refreshed = []
    retryable = []
    fatal = []
    active = []
    for row in rows:
        job_id = str(row.get("job_id") or "")
        if not job_id:
            continue
        timing = slurm_job_timing(job_id)
        state = str(timing.get("state") or "UNKNOWN")
        payload = {
            **row,
            "job_id": job_id,
            "slurm_state": state,
            "elapsed": timing.get("elapsed", ""),
            "time_limit": timing.get("time_limit", ""),
            "time_left": timing.get("time_left", ""),
            "end_time": timing.get("end_time", ""),
            "node": timing.get("node", ""),
            "status": "SLURM_STATE_REFRESH",
        }
        record_job_lifecycle(args.state_root.resolve(), payload)
        refreshed.append(payload)
        if state in ACTIVE_STATES:
            active.append(payload)
        elif state in RETRYABLE_TERMINAL_STATES:
            retryable.append(payload)
        elif state in TERMINAL_FAILURE_STATES:
            fatal.append(payload)
    return {"status": "REFRESHED", "jobs": refreshed, "active": active, "retryable": retryable, "fatal": fatal}


def validate_labelcritic_job(job_id: str, *, require_identity: bool = True) -> dict[str, Any]:
    record = slurm_job_record(job_id)
    user = _current_user()
    active = record.get("state") in ACTIVE_STATES
    user_ok = not user or record.get("user") == user
    name_ok = not require_identity or record.get("name") == LABELCRITIC_JOB_NAME
    ok = bool(record.get("status") == "FOUND" and active and user_ok and name_ok)
    reasons = []
    if record.get("status") != "FOUND":
        reasons.append("not_found")
    if not active:
        reasons.append(f"non_active_state:{record.get('state')}")
    if not user_ok:
        reasons.append(f"user_mismatch:{record.get('user')}!={user}")
    if not name_ok:
        reasons.append(f"name_mismatch:{record.get('name')}!={LABELCRITIC_JOB_NAME}")
    return {
        "status": "VALID" if ok else "INVALID",
        "job_id": str(job_id or ""),
        "record": record,
        "failure_reason": ",".join(reasons),
    }


def _job_id_sort_key(job: dict[str, Any]) -> tuple[int, str]:
    value = str(job.get("job_id") or "")
    return (int(value) if value.isdigit() else sys.maxsize, value)


def select_labelcritic_job(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [job for job in candidates if validate_labelcritic_job(str(job.get("job_id") or ""))["status"] == "VALID"]
    if not valid:
        return {"status": "NONE"}
    running = [job for job in valid if str(job.get("state") or "").upper() == "RUNNING"]
    pool = running or valid
    selected = sorted(pool, key=_job_id_sort_key)[0]
    return {"status": "SELECTED", "job_id": str(selected.get("job_id")), "candidates": valid}


def find_labelcritic_job_by_name() -> str:
    command = ["squeue", "-h"]
    user = _current_user()
    if user:
        command.extend(["-u", user])
    command.extend(["-n", LABELCRITIC_JOB_NAME, "-o", "%i|%T|%u|%j"])
    result = _run(command)
    if not result["ok"]:
        return ""
    candidates = []
    for line in result["stdout"].splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            candidates.append({
                "job_id": parts[0].strip(),
                "state": parts[1].strip(),
                "user": parts[2].strip() if len(parts) > 2 else "",
                "name": parts[3].strip() if len(parts) > 3 else LABELCRITIC_JOB_NAME,
            })
    selected = select_labelcritic_job(candidates)
    return str(selected.get("job_id") or "") if selected["status"] == "SELECTED" else ""


def _job_id_from_labelcritic_state(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("job_id", "labelcritic_job_id"):
            if str(value.get(key) or "").strip():
                return str(value[key]).strip()
        for key in ("service", "labelcritic"):
            nested = _job_id_from_labelcritic_state(value.get(key))
            if nested:
                return nested
    return ""


def _write_labelcritic_service_state(state_root: Path, payload: dict[str, Any]) -> None:
    service = _service_paths(state_root)
    service["root"].mkdir(parents=True, exist_ok=True)
    if payload.get("job_id"):
        service["job"].write_text(str(payload["job_id"]) + "\n", encoding="utf-8")
    _write_json(service["root"] / "service_state.json", payload)


def _case_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()


def _staged_row(row: dict[str, Any], *, case_id: str, index: int, workspace_root: Path) -> dict[str, Any]:
    updated = dict(row)
    updated["index"] = index
    updated["case_id"] = case_id
    updated["ct_path"] = str(workspace_root.resolve() / "inputs" / "images" / case_id / "ct.nii.gz")
    updated["image_path"] = updated["ct_path"]
    updated["annotation_folder"] = str(workspace_root.resolve() / "inputs" / "masks_original" / case_id / "segmentations")
    updated["reference_mask_dir"] = updated["annotation_folder"]
    return updated


def _path_is_within(path_text: str, root: Path) -> bool:
    if not path_text:
        return False
    try:
        Path(path_text).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _source_manifest_needs_rebuild(path: Path, *, workspace_root: Path) -> bool:
    if not path.exists():
        return True
    rows = read_csv_rows(path)
    if len(rows) != FORMAL_CASE_COUNT or len({_case_id(row, idx) for idx, row in enumerate(rows)}) != FORMAL_CASE_COUNT:
        return True
    for row in rows:
        for key in ("ct_path", "image_path", "annotation_folder", "reference_mask_dir", "mask_dir"):
            if _path_is_within(str(row.get(key) or ""), workspace_root):
                return True
    return False


def ensure_case_level_manifests(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    source_manifest = paths["source_manifest"]
    if _source_manifest_needs_rebuild(source_manifest, workspace_root=args.workspace_root):
        source_manifest.parent.mkdir(parents=True, exist_ok=True)
        build_formal_manifest(
            base_manifest=args.base_manifest.resolve(),
            output_manifest=source_manifest,
            audit_json=paths["root"] / "cases_103_source_manifest_build_audit.json",
        )
    rows = read_csv_rows(source_manifest)
    staged_rows = [
        _staged_row(row, case_id=_case_id(row, index), index=index, workspace_root=args.workspace_root)
        for index, row in enumerate(rows)
    ]
    fieldnames: list[str] = []
    for row in staged_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    paths["staged_manifest"].parent.mkdir(parents=True, exist_ok=True)
    write_csv(paths["staged_manifest"], staged_rows, fieldnames)
    if args.case_manifest != paths["staged_manifest"]:
        args.case_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.case_manifest, staged_rows, fieldnames)
    return {
        "source_manifest": str(source_manifest),
        "staged_manifest": str(paths["staged_manifest"]),
        "case_count": len(rows),
        "unique_case_count": len({_case_id(row, idx) for idx, row in enumerate(rows)}),
    }


def _state_paths(state_root: Path) -> dict[str, Path]:
    root = state_root / "round1_orchestrated"
    return {
        "root": root,
        "state": root / "state.json",
        "events": root / "events.jsonl",
        "failures": root / "failures.jsonl",
        "last_failure": root / "last_failure.json",
        "controller_sbatch": root / "controller.sbatch",
        "controller_job": root / "controller_job_id.txt",
        "staging_sbatch": root / "staging_array.sbatch",
        "staging_job": root / "staging_job_id.txt",
        "source_manifest": state_root / "case_level_manifests" / "cases_103_source_manifest.csv",
        "staged_manifest": root / "cases_103_staged_manifest.csv",
        "teacher_batches": root / "teacher_batches.jsonl",
        "labelcritic_selection_sbatch": root / "labelcritic_selection_workers.sbatch",
        "labelcritic_selection_job": root / "labelcritic_selection_job_id.txt",
        "mstep_sbatch": root / "round1_mstep_student.sbatch",
        "mstep_job": root / "round1_mstep_job_id.txt",
        "mstep_manifest": root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json",
        "mstep_output": root / "round1" / "mstep",
        "final": root / "round1_final_status.json",
        "job_lifecycle": root / "job_lifecycle.jsonl",
        "job_lifecycle_current": root / "job_lifecycle_current.json",
        "walltime_guard": root / "walltime_guard.jsonl",
    }


def _attempts_root(state_root: Path) -> Path:
    return state_root / "round1_orchestrated_attempts"


def archive_current_attempt(state_root: Path, *, reason: str) -> dict[str, Any]:
    paths = _state_paths(state_root)
    root = paths["root"]
    if not root.exists():
        return {"status": "NO_CURRENT_ATTEMPT", "reason": reason}
    attempts = _attempts_root(state_root)
    attempts.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in attempts.glob("attempt_*"):
        match = re.search(r"attempt_(\d+)$", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    destination = attempts / f"attempt_{(max(numbers) if numbers else 0) + 1:03d}"
    root.rename(destination)
    record = {
        "status": "ARCHIVED",
        "archived_attempt": str(destination),
        "reason": reason,
        "archived_at": utc_now(),
    }
    _write_json(destination / "attempt_archive.json", record)
    return record


def _load_state(state_root: Path) -> dict[str, Any]:
    paths = _state_paths(state_root)
    state = _read_json(paths["state"], {})
    return state if isinstance(state, dict) else {}


def _save_state(state_root: Path, **updates: Any) -> dict[str, Any]:
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    if updates.get("terminal_state") == "ROUND1_FAILED":
        updates.setdefault("terminal", True)
        updates.setdefault("status", "FAILED")
        updates.setdefault("controller_status", "FAILED")
        updates.setdefault("scheduler_status", "FATAL")
        if updates.get("e_step_status") not in {"PASSED", "FAILED"}:
            updates["e_step_status"] = "FAILED"
    elif updates.get("terminal_state") == "ROUND1_PASSED":
        updates.setdefault("terminal", True)
        updates.setdefault("status", "PASSED")
    state.update(updates)
    state["updated_at"] = utc_now()
    _write_json(paths["state"], state)
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    with paths["events"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": state["updated_at"], **updates}, ensure_ascii=False) + "\n")
    return state


def _tail_text(value: Any, limit: int = 4000) -> str:
    text = str(value or "")
    return text[-limit:]


def _compact_details(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = str(key)
            if key in {"stdout", "stderr", "stdout_tail", "stderr_tail"}:
                compact[safe_key] = _tail_text(item)
            else:
                compact[safe_key] = _compact_details(item)
        return compact
    if isinstance(value, (list, tuple)):
        return [_compact_details(item) for item in value[:100]]
    if isinstance(value, set):
        return sorted(_compact_details(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def log_failure(state_root: Path, *, stage: str, failure_reason: str, details: Any | None = None) -> dict[str, Any]:
    paths = _state_paths(state_root)
    record = {
        "time": utc_now(),
        "stage": stage,
        "failure_reason": str(failure_reason or "unknown_failure"),
        "git_commit": _git_commit(),
        "details": _compact_details(details or {}),
    }
    paths["failures"].parent.mkdir(parents=True, exist_ok=True)
    with paths["failures"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(paths["last_failure"], record)
    return record


def _git_commit() -> str:
    result = _run(["git", "rev-parse", "HEAD"])
    return result["stdout"] if result["ok"] else ""


def _run_id(state_root: Path) -> str:
    state = _load_state(state_root)
    existing = str(state.get("run_id") or os.getenv("ROUND1_RUN_ID") or "").strip()
    if existing:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", existing)
    seed = f"round1_{utc_now()}_{state_root}"
    run_id = "round1_" + str(abs(hash(seed)))
    _save_state(state_root, run_id=run_id)
    return run_id


def verify_expected_git_commit(state_root: Path, expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit or "").strip()
    current = _git_commit()
    status = "PASSED" if not expected or current == expected else "FAILED"
    report = {
        "status": status,
        "expected_git_commit": expected,
        "actual_git_commit": current,
        "failure_reason": "" if status == "PASSED" else "expected_git_commit_mismatch",
    }
    if status != "PASSED":
        log_failure(state_root, stage="git_commit_pin", failure_reason=report["failure_reason"], details=report)
    return report


def run_static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    branch = _run(["git", "branch", "--show-current"])
    add("git_main_clean", status["ok"] and not status["stdout"] and branch["stdout"] == "main", {"status": status, "branch": branch["stdout"]})
    commit_pin = verify_expected_git_commit(state_root, str(getattr(args, "expected_git_commit", "") or ""))
    add("expected_git_commit", commit_pin["status"] == "PASSED", commit_pin)
    try:
        manifest_setup = ensure_case_level_manifests(args)
        add("case_level_manifests_ready", manifest_setup.get("case_count") == FORMAL_CASE_COUNT and manifest_setup.get("unique_case_count") == FORMAL_CASE_COUNT, manifest_setup)
        manifest_for_preflight = Path(str(manifest_setup["source_manifest"]))
    except Exception as exc:
        add("case_level_manifests_ready", False, f"{type(exc).__name__}: {exc}")
        manifest_for_preflight = args.case_manifest.resolve()
    try:
        manifest = validate_formal_manifest(manifest=manifest_for_preflight.resolve(), base_manifest=args.base_manifest.resolve())
        add("cases_103_manifest", manifest.get("rows") == FORMAL_CASE_COUNT and manifest.get("unique_case_count") == FORMAL_CASE_COUNT, manifest)
    except Exception as exc:
        add("cases_103_manifest", False, f"{type(exc).__name__}: {exc}")
    try:
        full_scope = build_full_round1_scope(
            case_manifest=manifest_for_preflight.resolve(),
            registry_path=args.registry.resolve(),
            target_config=args.target_config.resolve(),
            output_root=paths["root"] / FULL373_ROOT_NAME,
            cache_roots=[paths["root"] / "formal_task2_round1"],
            expected_case_count=FORMAL_CASE_COUNT,
        )
        add(
            "full_373_multiteacher_scope",
            full_scope.get("status") == "READY"
            and full_scope.get("case_count") == FORMAL_CASE_COUNT
            and full_scope.get("canonical_target_count") == 373
            and int(full_scope.get("unroutable_target_count") or 0) == 0,
            {
                "scope_json": str(paths["root"] / FULL373_ROOT_NAME / "full_round1_scope.json"),
                "case_count": full_scope.get("case_count"),
                "canonical_target_count": full_scope.get("canonical_target_count"),
                "enabled_teacher_count": full_scope.get("enabled_teacher_count"),
                "target_teacher_pairs": full_scope.get("target_teacher_pairs"),
                "total_logical_candidate_tasks": full_scope.get("total_logical_candidate_tasks"),
                "unroutable_target_count": full_scope.get("unroutable_target_count"),
            },
        )
    except Exception as exc:
        add("full_373_multiteacher_scope", False, f"{type(exc).__name__}: {exc}")
    add("personal_workspace_writable", os.access(args.workspace_root.resolve(), os.W_OK) if args.workspace_root.exists() else os.access(args.workspace_root.parent.resolve(), os.W_OK), str(args.workspace_root))
    public_root = Path("/projects/bodymaps/Data")
    add("public_data_not_output_root", public_root not in args.state_root.resolve().parents and args.state_root.resolve() != public_root, str(args.state_root))
    if public_root.exists():
        add("public_data_not_writable_by_user", not os.access(public_root, os.W_OK), str(public_root))
    for path_name, path in {
        "checkpoint_root": args.checkpoint_root,
        "nnunet_predict": args.nnunet_predict_executable,
        "unest_python": args.unest_python_executable,
        "resource_profiles": REPO_ROOT / "configs" / "resource_profiles.yaml",
        "target_config": args.target_config,
        "registry": args.registry,
    }.items():
        add(path_name, path.exists(), str(path))
    compile_result = _run([
        str(args.python), "-m", "py_compile",
        "tools/dataset_delivery/task2_round1_orchestrator.py",
        "tools/dataset_delivery/task2_formal_launcher.py",
        "tools/dataset_delivery/task2_dynamic_gpu_submitter.py",
        "tools/dataset_delivery/task2_full373_round1_launcher.py",
        "tools/dataset_delivery/formal_round1_preflight.py",
    ])
    add("py_compile_orchestration", compile_result["ok"], compile_result)
    for script in [
        "scripts/task2/submit_task2_formal_103cases.sh",
        "scripts/task2/submit_labelcritic_72b_service.sh",
        "scripts/task2/check_task2_formal_103cases.sh",
        "scripts/task2/submit_task2_round1_orchestrated.sh",
        "scripts/task2/check_task2_round1_orchestrated.sh",
    ]:
        result = _run(["bash", "-n", script])
        add(f"bash_n:{script}", result["ok"], result)
    if args.run_static_tests:
        tests = [
            "tests/dataset_delivery/test_task2_round1_orchestrator.py",
            "tests/dataset_delivery/test_round1_production_hardening.py",
            "tests/dataset_delivery/test_task2_runtime_repair.py",
            "tests/dataset_delivery/test_task2_formal_production.py",
            "tests/dataset_delivery/test_task2_dynamic_gpu_submitter.py",
            "tests/dataset_delivery/test_task2_full373_round1_launcher.py",
        ]
        result = _run([str(args.python), "-m", "pytest", "-q", *tests], env=sanitized_static_test_env(), timeout=int(args.static_tests_timeout_sec))
        add("task1_task2_scheduler_regression_tests", result["ok"], result)
    controller = render_controller_sbatch(args, paths["controller_sbatch"])
    add("controller_sbatch_rendered", paths["controller_sbatch"].exists(), controller)
    result = _run(["bash", "-n", str(paths["controller_sbatch"])])
    add("controller_bash_n", result["ok"], result)
    if not args.skip_sbatch_test_only:
        result = _run(["sbatch", "--test-only", str(paths["controller_sbatch"])])
        add("controller_sbatch_test_only", result["ok"], result)
        labelcritic_preview = render_labelcritic_preview_sbatch(args, paths["root"] / "labelcritic_72b_service.preview.sbatch")
        add("labelcritic_sbatch_preview_rendered", Path(labelcritic_preview["path"]).exists(), labelcritic_preview)
        result = _run(["bash", "-n", labelcritic_preview["path"]])
        add("labelcritic_sbatch_preview_bash_n", result["ok"], result)
        result = _run(["sbatch", "--test-only", labelcritic_preview["path"]])
        add("labelcritic_sbatch_preview_test_only", result["ok"], result)
    report = {
        "status": "PASSED" if all(check["ok"] for check in checks) else "FAILED",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "checks": checks,
    }
    _write_json(paths["root"] / "static_preflight.json", report)
    return report


def render_labelcritic_preview_sbatch(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=labelcritic_72b_service",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={args.labelcritic_partition}",
        f"#SBATCH --gres={args.labelcritic_gres}",
        f"#SBATCH --cpus-per-task={os.getenv('LABELCRITIC_CPUS', '16')}",
        f"#SBATCH --mem={os.getenv('LABELCRITIC_MEM', '192G')}",
        f"#SBATCH --time={os.getenv('LABELCRITIC_TIME', '08:00:00')}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export LABELCRITIC_MODEL_ID={shlex.quote(LABELCRITIC_MODEL_ID)}",
        f"export LABELCRITIC_TENSOR_PARALLEL_SIZE={int(args.labelcritic_tensor_parallel_size)}",
        "true",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path), "profile": {"partition": args.labelcritic_partition, "gres": args.labelcritic_gres, "tp": int(args.labelcritic_tensor_parallel_size)}}


def render_controller_sbatch(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    run_id = _run_id(args.state_root.resolve())
    command = [
        str(args.python),
        "tools/dataset_delivery/task2_round1_orchestrator.py",
        "controller",
        "--state-root", str(args.state_root),
        "--workspace-root", str(args.workspace_root),
        "--case-manifest", str(args.case_manifest),
        "--base-manifest", str(args.base_manifest),
        "--python", str(args.python),
        "--checkpoint-root", str(args.checkpoint_root),
        "--nnunet-predict-executable", str(args.nnunet_predict_executable),
        "--unest-python-executable", str(args.unest_python_executable),
        "--registry", str(args.registry),
        "--target-config", str(args.target_config),
        "--gpu-target-workers", str(args.gpu_target_workers),
        "--gpu-overrequest-workers", str(args.gpu_overrequest_workers),
        "--gpu-profile-specs", args.gpu_profile_specs,
        "--poll-sec", str(args.poll_sec),
        "--expected-git-commit", str(args.expected_git_commit),
    ]
    env_exports = {
        "CODE_ROOT": str(REPO_ROOT),
        "STATE_ROOT": str(args.state_root),
        "WORKSPACE_ROOT": str(args.workspace_root),
        "ROUND1_RUN_ID": run_id,
        "LABELCRITIC_PARTITION": args.labelcritic_partition,
        "LABELCRITIC_GRES": args.labelcritic_gres,
        "LABELCRITIC_TENSOR_PARALLEL_SIZE": str(args.labelcritic_tensor_parallel_size),
        "LABELCRITIC_PORT": str(args.labelcritic_port),
        "LABELCRITIC_MODEL_ID": LABELCRITIC_MODEL_ID,
        "EXPECTED_GIT_COMMIT": str(args.expected_git_commit),
        "RUNTIME_NO_GIT": "1",
        "SKIP_GIT_SYNC": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_round1_controller",
        f"#SBATCH --partition={args.controller_partition}",
        f"#SBATCH --cpus-per-task={args.controller_cpus}",
        f"#SBATCH --mem={args.controller_mem}",
        f"#SBATCH --time={args.controller_time}",
        "#SBATCH --signal=B:USR1@900",
        f"#SBATCH --output={_state_paths(args.state_root)['root'] / 'controller_%j.out'}",
        f"#SBATCH --error={_state_paths(args.state_root)['root'] / 'controller_%j.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        *[f"export {key}={shlex.quote(value)}" for key, value in env_exports.items()],
        "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout --state-root \"$STATE_ROOT\" --logical-task-id controller --job-id \"${SLURM_JOB_ID:-}\"' USR1",
        " ".join(shlex.quote(part) for part in command),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path), "command": command, "env": env_exports}


def submit_controller(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    state = _load_state(args.state_root.resolve())
    if state.get("terminal_state") == "ROUND1_FAILED" and (getattr(args, "retry_failed", False) or getattr(args, "new_attempt", False)):
        archive_current_attempt(args.state_root.resolve(), reason="retry_failed_or_new_attempt")
    elif state.get("terminal_state") in {"ROUND1_PASSED", "ROUND1_FAILED"}:
        return {"status": "ROUND1_ALREADY_TERMINAL", "terminal_state": state.get("terminal_state"), "state_root": str(paths["root"])}
    elif getattr(args, "new_attempt", False) and paths["root"].exists():
        existing = str(state.get("controller_job_id") or "").strip()
        if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
            return {"status": "CONTROLLER_ALREADY_ACTIVE", "controller_job_id": existing, "state_root": str(paths["root"])}
        archive_current_attempt(args.state_root.resolve(), reason="explicit_new_attempt")
    preflight = run_static_preflight(args)
    paths = _state_paths(args.state_root.resolve())
    if preflight["status"] != "PASSED":
        failure = log_failure(args.state_root.resolve(), stage="static_preflight", failure_reason="static_preflight_failed", details=preflight)
        _save_state(args.state_root.resolve(), terminal_state="ROUND1_FAILED", stage="static_preflight", failure_reason="static_preflight_failed", static_preflight=preflight)
        raise SystemExit(json.dumps({"status": "STATIC_PREFLIGHT_FAILED", "report": str(paths["root"] / "static_preflight.json"), "last_failure": failure}, indent=2))
    state = _load_state(args.state_root.resolve())
    existing = str(state.get("controller_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
        return {"status": "CONTROLLER_ALREADY_ACTIVE", "controller_job_id": existing, "state_root": str(paths["root"])}
    run_id = _run_id(args.state_root.resolve())
    result = _run(["sbatch", "--parsable", "--comment", slurm_comment(run_id=run_id, submission_id="controller", group="controller", profile="cpu"), str(paths["controller_sbatch"])])
    if not result["ok"]:
        log_failure(args.state_root.resolve(), stage="controller_submit", failure_reason=result["stderr"] or "controller_submit_failed", details=result)
        _save_state(args.state_root.resolve(), terminal_state="ROUND1_FAILED", stage="controller_submit", failure_reason=result["stderr"], controller_submit=result)
        raise SystemExit(result["stderr"])
    job_id = result["stdout"].splitlines()[-1].strip()
    paths["controller_job"].write_text(job_id + "\n", encoding="utf-8")
    _save_state(args.state_root.resolve(), status="CONTROLLER_SUBMITTED", controller_job_id=job_id, git_commit=_git_commit(), static_preflight_path=str(paths["root"] / "static_preflight.json"))
    record_job_lifecycle(args.state_root.resolve(), {"status": "SUBMITTED", "job_id": job_id, "logical_task_id": f"{run_id}:controller", "stage": "controller", "profile": "cpu"})
    return {"status": "CONTROLLER_SUBMITTED", "controller_job_id": job_id, "state_root": str(paths["root"])}


def _service_paths(state_root: Path) -> dict[str, Path]:
    root = Path(os.getenv("LABELCRITIC_SERVICE_ROOT", str(state_root / "labelcritic_72b_service"))).expanduser()
    return {
        "root": root,
        "job": root / "job_id.txt",
        "base": root / "base_url.txt",
        "port": root / "port.txt",
        "endpoint": root / "endpoint.url",
    }


def ensure_labelcritic_service(state_root: Path) -> dict[str, Any]:
    service = _service_paths(state_root)
    env_base = os.getenv("LABELCRITIC_BASE_URL", "").strip()
    if env_base:
        env_port = int(os.getenv("LABELCRITIC_PORT", "8000"))
        health = labelcritic_health(env_base, env_port)
        if health["status"] == "READY":
            service["root"].mkdir(parents=True, exist_ok=True)
            service["base"].write_text(health["base_url"] + "\n", encoding="utf-8")
            service["port"].write_text(str(health["port"]) + "\n", encoding="utf-8")
            service["endpoint"].write_text(f"{health['base_url']}:{health['port']}\n", encoding="utf-8")
            return {"status": "REUSED_HEALTHY", "base_url": health["base_url"], "port": health["port"], "health": health, "source": "LABELCRITIC_BASE_URL"}
    base = service["base"].read_text(encoding="utf-8").strip() if service["base"].exists() else ""
    port = int(service["port"].read_text(encoding="utf-8").strip()) if service["port"].exists() else int(os.getenv("LABELCRITIC_PORT", "8000"))
    if base:
        health = labelcritic_health(base, port)
        if health["status"] == "READY":
            return {"status": "REUSED_HEALTHY", "base_url": health["base_url"], "port": health["port"], "health": health}
    ignored: list[dict[str, Any]] = []
    state_job_id = _job_id_from_labelcritic_state(_load_state(state_root).get("labelcritic"))
    service_job_id = service["job"].read_text(encoding="utf-8").strip() if service["job"].exists() else ""
    env_job_id = os.getenv("LABELCRITIC_JOB_ID", "").strip()
    for source, job_id in (
        ("current_attempt_state", state_job_id),
        ("labelcritic_service_state", service_job_id),
        ("LABELCRITIC_JOB_ID", env_job_id),
    ):
        if not job_id:
            continue
        validation = validate_labelcritic_job(job_id)
        if validation["status"] == "VALID":
            reused = {"status": "REUSED_ACTIVE_JOB", "job_id": job_id, "source": source, "validation": validation}
            _write_labelcritic_service_state(state_root, reused)
            return reused
        ignored.append({"source": source, "job_id": job_id, "validation": validation})
    named = find_labelcritic_job_by_name()
    if named:
        validation = validate_labelcritic_job(named)
        if validation["status"] == "VALID":
            reused = {"status": "REUSED_ACTIVE_JOB", "job_id": named, "source": "slurm_name_discovery", "validation": validation, "ignored_jobs": ignored}
            _write_labelcritic_service_state(state_root, reused)
            return reused
        ignored.append({"source": "slurm_name_discovery", "job_id": named, "validation": validation})
    result = _run(["bash", "scripts/task2/submit_labelcritic_72b_service.sh"], env=runtime_no_git_env())
    if not result["ok"]:
        log_failure(state_root, stage="labelcritic_submit", failure_reason=result["stderr"] or result["stdout"] or "labelcritic_submit_failed", details=result)
        return {"status": "SUBMIT_FAILED", "failure_reason": result["stderr"], "submit": result}
    job_id_match = re.search(r"LABELCRITIC_JOB_ID=([^\s]+)", result["stdout"])
    job_id = job_id_match.group(1) if job_id_match else (service["job"].read_text(encoding="utf-8").strip() if service["job"].exists() else "")
    submitted = {"status": "SUBMITTED", "job_id": job_id, "submit": result, "ignored_jobs": ignored}
    _write_labelcritic_service_state(state_root, submitted)
    return submitted


def wait_for_labelcritic_runtime(state_root: Path, *, poll_sec: int) -> dict[str, Any]:
    while True:
        service = ensure_labelcritic_service(state_root)
        _save_state(state_root, stage="labelcritic_wait", labelcritic=service)
        if service["status"] in {"REUSED_HEALTHY"}:
            runtime = labelcritic_runtime_preflight(service["base_url"], int(service["port"]))
            _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
            if runtime["status"] == "PASSED":
                return {"status": "PASSED", "base_url": service["base_url"], "port": int(service["port"]), "runtime": runtime}
            log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
            return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime}
        job_id = str(service.get("job_id") or "")
        state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
        if state.get("state") in TERMINAL_FAILURE_STATES:
            log_failure(state_root, stage="labelcritic_job", failure_reason=f"labelcritic_job_terminal:{state.get('state')}", details=state)
            return {"status": "FAILED", "failure_reason": f"labelcritic_job_terminal:{state.get('state')}", "job": state}
        base_file = _service_paths(state_root)["base"]
        port_file = _service_paths(state_root)["port"]
        if base_file.exists() and port_file.exists():
            base = base_file.read_text(encoding="utf-8").strip()
            port = int(port_file.read_text(encoding="utf-8").strip())
            health = labelcritic_health(base, port)
            if health["status"] == "READY":
                runtime = labelcritic_runtime_preflight(base, port)
                _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
                if runtime["status"] == "PASSED":
                    return {"status": "PASSED", "base_url": base, "port": port, "runtime": runtime}
                log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
                return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime}
        time.sleep(max(5, poll_sec))


def poll_labelcritic_runtime(state_root: Path) -> dict[str, Any]:
    service = ensure_labelcritic_service(state_root)
    _save_state(state_root, stage="labelcritic_poll", labelcritic=service)
    if service["status"] == "SUBMIT_FAILED":
        return {"status": "FAILED", "failure_reason": service.get("failure_reason", "labelcritic_submit_failed"), "service": service}
    if service["status"] == "REUSED_HEALTHY":
        runtime = labelcritic_runtime_preflight(service["base_url"], int(service["port"]))
        _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
        if runtime["status"] == "PASSED":
            return {"status": "PASSED", "base_url": service["base_url"], "port": int(service["port"]), "runtime": runtime, "service": service}
        log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
        return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime, "service": service}
    job_id = str(service.get("job_id") or "")
    state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
    if state.get("state") in TERMINAL_FAILURE_STATES:
        log_failure(state_root, stage="labelcritic_job", failure_reason=f"labelcritic_job_terminal:{state.get('state')}", details=state)
        return {"status": "FAILED", "failure_reason": f"labelcritic_job_terminal:{state.get('state')}", "job": state, "service": service}
    base_file = _service_paths(state_root)["base"]
    port_file = _service_paths(state_root)["port"]
    if base_file.exists() and port_file.exists():
        base = base_file.read_text(encoding="utf-8").strip()
        port = int(port_file.read_text(encoding="utf-8").strip())
        health = labelcritic_health(base, port)
        if health["status"] == "READY":
            runtime = labelcritic_runtime_preflight(base, port)
            _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
            if runtime["status"] == "PASSED":
                return {"status": "PASSED", "base_url": base, "port": port, "runtime": runtime, "service": service}
            log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
            return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime, "service": service}
    return {"status": "WAITING", "job": state, "service": service}


def _labelcritic_endpoint_hint(state_root: Path, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    if labelcritic and labelcritic.get("base_url") and labelcritic.get("port"):
        return {"base_url": str(labelcritic["base_url"]), "port": int(labelcritic["port"]), "source": "runtime"}
    service = _service_paths(state_root)
    if service["base"].exists() and service["port"].exists():
        return {
            "base_url": service["base"].read_text(encoding="utf-8").strip(),
            "port": int(service["port"].read_text(encoding="utf-8").strip()),
            "source": "service_files",
        }
    return {
        "base_url": os.getenv("LABELCRITIC_BASE_URL", "http://localhost"),
        "port": int(os.getenv("LABELCRITIC_PORT", "8000")),
        "source": "env_or_default",
    }


def _staging_status_rows(args: argparse.Namespace, source_manifest: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(source_manifest)
    statuses = []
    for index, row in enumerate(rows):
        case_id = _case_id(row, index)
        state_path = args.workspace_root.resolve() / "manifests" / "staging_cases" / f"{case_id}.json"
        state = _read_json(state_path, {})
        status = staged_case_status(workspace_root=args.workspace_root.resolve(), case_id=case_id)
        state_status = str(state.get("status") or "")
        if status["status"] != "INPUT_READY" and state_status in {"STAGING_PENDING", "STAGING_RUNNING", "STAGING_FAILED"}:
            status["status"] = state_status
            status["errors"] = state.get("errors") or status.get("errors") or []
        statuses.append({"case_index": index, **status, "state_path": str(state_path)})
    return statuses


def render_staging_sbatch(args: argparse.Namespace, source_manifest: Path, path: Path) -> dict[str, Any]:
    image_root = Path(os.getenv("IMAGE_ROOT", "/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro"))
    mask_root = Path(os.getenv("MASK_ROOT", "/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro"))
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_case_staging",
        f"#SBATCH --partition={os.getenv('STAGING_PARTITION', 'cpu')}",
        f"#SBATCH --cpus-per-task={os.getenv('STAGING_CPUS', '2')}",
        f"#SBATCH --mem={os.getenv('STAGING_MEM', '12G')}",
        f"#SBATCH --time={os.getenv('STAGING_TIME', '04:00:00')}",
        "#SBATCH --signal=B:USR1@900",
        f"#SBATCH --output={_state_paths(args.state_root)['root'] / 'staging_%A_%a.out'}",
        f"#SBATCH --error={_state_paths(args.state_root)['root'] / 'staging_%A_%a.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export STATE_ROOT={shlex.quote(str(args.state_root))}",
        "export RUNTIME_NO_GIT=1",
        "export SKIP_GIT_SYNC=1",
        "export GIT_TERMINAL_PROMPT=0",
        "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout --state-root \"$STATE_ROOT\" --logical-task-id staging --job-id \"${SLURM_JOB_ID:-}\" --task-index \"${SLURM_ARRAY_TASK_ID:-}\"' USR1",
        f"{shlex.quote(str(args.python))} tools/dataset_delivery/task2_workspace_staging.py \\",
        f"  --cases-manifest {shlex.quote(str(source_manifest))} \\",
        f"  --workspace-root {shlex.quote(str(args.workspace_root))} \\",
        f"  --image-root {shlex.quote(str(image_root))} \\",
        f"  --mask-root {shlex.quote(str(mask_root))} \\",
        f"  --case-index \"${{SLURM_ARRAY_TASK_ID}}\" \\",
        "  --resume",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path)}


def submit_staging_workers(args: argparse.Namespace, source_manifest: Path) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    statuses = _staging_status_rows(args, source_manifest)
    ready_count = sum(1 for row in statuses if row["status"] == "INPUT_READY")
    if ready_count == len(statuses):
        return {"status": "ALL_INPUT_READY", "ready_count": ready_count, "case_count": len(statuses)}
    state = _load_state(state_root)
    existing = str(state.get("staging_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
        return {"status": "REUSED_ACTIVE_STAGING", "job_id": existing, "ready_count": ready_count, "case_count": len(statuses)}
    rendered = render_staging_sbatch(args, source_manifest, paths["staging_sbatch"])
    for command in (["bash", "-n", str(paths["staging_sbatch"])], ["sbatch", "--test-only", str(paths["staging_sbatch"])]):
        result = _run(command)
        if not result["ok"]:
            log_failure(state_root, stage="staging_sbatch_preflight", failure_reason="staging_sbatch_preflight_failed", details=result)
            return {"status": "FAILED", "failure_reason": "staging_sbatch_preflight_failed", "preflight": result}
    concurrency = max(1, int(os.getenv("STAGING_CONCURRENCY", "12")))
    run_id = _run_id(state_root)
    result = _run([
        "sbatch",
        "--parsable",
        "--comment", slurm_comment(run_id=run_id, submission_id="staging", group="staging", profile="cpu"),
        f"--array=0-{len(statuses) - 1}%{concurrency}",
        str(paths["staging_sbatch"]),
    ])
    if not result["ok"]:
        classification = classify_sbatch_failure(result["stderr"] or result["stdout"])
        if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
            _save_state(state_root, staging_status="WAITING_FOR_SUBMISSION_CAPACITY", scheduler_status="BACKPRESSURED", staging_submit=result)
            return {
                "status": "WAITING_FOR_SUBMISSION_CAPACITY",
                "scheduler_status": "BACKPRESSURED",
                "failure_reason": classification["reason"],
                "ready_count": ready_count,
                "case_count": len(statuses),
                "submit": result,
            }
        log_failure(state_root, stage="staging_submit", failure_reason=result["stderr"] or "staging_submit_failed", details=result)
        return {"status": "FAILED", "failure_reason": result["stderr"] or "staging_submit_failed", "submit": result}
    job_id = result["stdout"].splitlines()[-1].strip()
    paths["staging_job"].write_text(job_id + "\n", encoding="utf-8")
    _save_state(state_root, staging_job_id=job_id, staging_status="SUBMITTED", staging_sbatch=rendered)
    record_job_lifecycle(state_root, {"status": "SUBMITTED", "job_id": job_id, "logical_task_id": f"{run_id}:staging", "stage": "staging", "profile": "cpu", "task_count": len(statuses), "array_concurrency": concurrency})
    return {"status": "SUBMITTED", "job_id": job_id, "ready_count": ready_count, "case_count": len(statuses), "concurrency": concurrency}


def _append_teacher_batch(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    paths["teacher_batches"].parent.mkdir(parents=True, exist_ok=True)
    with paths["teacher_batches"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def submit_ready_teacher_batch(args: argparse.Namespace, source_manifest: Path, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    run_id = _run_id(state_root)
    submitted_cases = set(str(case_id) for case_id in state.get("teacher_submitted_case_ids") or [])
    statuses = _staging_status_rows(args, source_manifest)
    ready_status_rows = [row for row in statuses if row["status"] == "INPUT_READY" and row["case_id"] not in submitted_cases]
    ready_cases = [row["case_id"] for row in ready_status_rows]
    if not ready_cases:
        return {
            "status": "NO_NEW_READY_CASES",
            "ready_count": sum(1 for row in statuses if row["status"] == "INPUT_READY"),
            "submitted_case_count": len(submitted_cases),
            "staging_failed_count": sum(1 for row in statuses if row["status"] == "STAGING_FAILED"),
        }
    active_submission_id = str(state.get("teacher_active_submission_id") or "").strip()
    if active_submission_id and state.get("scheduler_status") in {"BACKPRESSURED", "WAITING_FOR_SUBMISSION_CAPACITY", "PARTIALLY_SUBMITTED"}:
        submission_id = active_submission_id
        match = re.search(r"ready_batch_(\d+)", submission_id)
        batch_index = int(match.group(1)) if match else int(state.get("teacher_batch_index") or 0)
    else:
        batch_index = int(state.get("teacher_batch_index") or 0) + 1
        submission_id = f"ready_batch_{batch_index:03d}"
    formal_root = Path(str(state.get("formal_root") or (paths["root"] / FULL373_ROOT_NAME))).resolve()
    reconcile_active_teacher_jobs(args, formal_root)
    ready_manifest = paths["root"] / f"{submission_id}_full373_ready_cases.csv"
    write_csv(
        ready_manifest,
        [
            {
                "case_id": row["case_id"],
                "ct_path": row["ct_path"],
                "annotation_folder": row["annotation_folder"],
            }
            for row in ready_status_rows
        ],
        ["case_id", "ct_path", "annotation_folder"],
    )
    endpoint = _labelcritic_endpoint_hint(state_root, labelcritic)
    h100_policy = resolve_teacher_h100_policy(
        state_root=state_root,
        labelcritic=labelcritic,
        slurm_job_state_fn=slurm_job_state,
        find_labelcritic_job_fn=find_labelcritic_job_by_name,
        source="orchestrator_submit_ready_teacher_batch",
    )
    env = runtime_no_git_env()
    env.update({
        "LABELCRITIC_SERVICE_ROOT": str(_service_paths(state_root)["root"]),
        "LABELCRITIC_BASE_URL": str(endpoint["base_url"]),
        "LABELCRITIC_PORT": str(endpoint["port"]),
        "LABELCRITIC_MODEL_ID": LABELCRITIC_MODEL_ID,
        "MEDAI_FORMAL_LABELCRITIC_72B_SELECTION_READY": "1",
        "MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC": str(os.getenv("MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC", os.getenv("WAIT_LABELCRITIC_SEC", "14400"))),
        "WAIT_LABELCRITIC_SEC": str(os.getenv("WAIT_LABELCRITIC_SEC", "14400")),
        "EXPECTED_GIT_COMMIT": str(args.expected_git_commit),
        "TASK2_TEACHER_H100_POLICY_JSON": json.dumps(h100_policy, ensure_ascii=False),
        "TASK2_LABELCRITIC_REQUIRED": "1" if h100_policy.get("labelcritic_required") else "0",
        "TASK2_LABELCRITIC_H100_RESERVED": "1" if h100_policy.get("labelcritic_h100_reserved") else "0",
        "TASK2_LABELCRITIC_JOB_ID": str(h100_policy.get("labelcritic_job_id") or ""),
        "TASK2_LABELCRITIC_JOB_STATE": str(h100_policy.get("labelcritic_job_state") or ""),
    })
    plan_cmd = [
        str(args.python),
        "tools/dataset_delivery/task2_full373_round1_launcher.py",
        "--output-root", str(formal_root),
        "--case-manifest", str(ready_manifest),
        "--python", str(args.python),
        "--registry", str(args.registry),
        "--target-config", str(args.target_config),
        "--checkpoint-root", str(args.checkpoint_root),
        "--nnunet-predict-executable", str(args.nnunet_predict_executable),
        "--unest-python-executable", str(args.unest_python_executable),
        "--state-root", str(state_root),
        "--cache-root", str(paths["root"] / "formal_task2_round1"),
    ]
    plan_result = _run(plan_cmd, env=env, timeout=600)
    if not plan_result["ok"]:
        log_failure(state_root, stage="teacher_batch_plan", failure_reason=plan_result["stderr"] or "teacher_batch_plan_failed", details=plan_result)
        return {"status": "FAILED", "failure_reason": plan_result["stderr"] or "teacher_batch_plan_failed", "plan": plan_result}
    summary = _read_json(formal_root / "formal_task2_submission_manifest.json", {})
    if int(summary.get("task_count") or 0) <= 0:
        updated = sorted(submitted_cases | set(ready_cases))
        _save_state(state_root, teacher_submitted_case_ids=updated, teacher_batch_index=batch_index, formal_root=str(formal_root))
        return {"status": "NO_TASKS_AFTER_RESUME", "submission_id": submission_id, "ready_cases": ready_cases}
    dynamic_cmd = [
        str(args.python),
        "tools/dataset_delivery/task2_dynamic_gpu_submitter.py",
        "--summary", str(formal_root / "formal_task2_submission_manifest.json"),
        "--output-root", str(formal_root),
        "--state-root", str(state_root),
        "--target-workers", str(args.gpu_target_workers),
        "--overrequest-workers", str(args.gpu_overrequest_workers),
        "--profile-specs", args.gpu_profile_specs,
        "--groups", FULL373_GROUP,
        "--group-weights", os.getenv("GPU_GROUP_WEIGHTS", "full373=1.0"),
        "--submission-id", submission_id,
        "--append-submitted-jobs",
        "--run-id", run_id,
        "--git-commit", _git_commit(),
    ]
    if os.getenv("DYNAMIC_SBATCH_TEST_ONLY", "1") != "1":
        dynamic_cmd.append("--skip-sbatch-test-only")
    dynamic_result = _run(dynamic_cmd, env=env, timeout=600)
    if not dynamic_result["ok"]:
        log_failure(state_root, stage="teacher_batch_submit", failure_reason=dynamic_result["stderr"] or "teacher_batch_submit_failed", details=dynamic_result)
        return {"status": "FAILED", "failure_reason": dynamic_result["stderr"] or "teacher_batch_submit_failed", "dynamic": dynamic_result}
    safe_submission_id = re.sub(r"[^A-Za-z0-9_-]+", "_", submission_id).strip("_")
    dynamic_plan = _read_json(formal_root / "slurm" / f"dynamic_gpu_submission_plan_{safe_submission_id}.json", {})
    if not dynamic_plan:
        try:
            dynamic_plan = json.loads(str(dynamic_result.get("stdout") or "{}"))
        except Exception:
            dynamic_plan = {}
    dynamic_status = str(dynamic_plan.get("status") or "SUBMITTED")
    scheduler_status = str(dynamic_plan.get("scheduler_status") or ("BACKPRESSURED" if dynamic_status in BACKPRESSURE_STATUSES else "ACTIVE"))
    batch = {
        "status": dynamic_status if dynamic_status in BACKPRESSURE_STATUSES else "SUBMITTED",
        "scheduler_status": scheduler_status,
        "submission_id": submission_id,
        "case_ids": ready_cases,
        "case_count": len(ready_cases),
        "planned_task_count": int(summary.get("task_count") or 0),
        "formal_root": str(formal_root),
        "plan": plan_result,
        "dynamic": dynamic_result,
        "dynamic_plan": dynamic_plan,
        "teacher_h100_policy": dynamic_plan.get("teacher_h100_policy") or h100_policy,
        "created_at": utc_now(),
    }
    _append_teacher_batch(paths, batch)
    if batch["status"] in BACKPRESSURE_STATUSES:
        _save_state(
            state_root,
            teacher_batch_index=batch_index,
            teacher_active_submission_id=submission_id,
            formal_root=str(formal_root),
            last_teacher_batch=batch,
            scheduler_status=scheduler_status,
            teacher_h100_policy=batch["teacher_h100_policy"],
        )
        return batch
    updated = sorted(submitted_cases | set(ready_cases))
    _save_state(
        state_root,
        teacher_submitted_case_ids=updated,
        teacher_batch_index=batch_index,
        teacher_active_submission_id="",
        formal_root=str(formal_root),
        last_teacher_batch=batch,
        scheduler_status=scheduler_status,
        teacher_h100_policy=batch["teacher_h100_policy"],
    )
    return batch


def render_labelcritic_selection_sbatch(args: argparse.Namespace, formal_root: Path, path: Path, labelcritic: dict[str, Any]) -> dict[str, Any]:
    worker_count = max(1, int(os.getenv("LABELCRITIC_SELECTION_WORKERS", "4")))
    endpoint = _labelcritic_endpoint_hint(args.state_root.resolve(), labelcritic)
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_labelcritic_select",
        f"#SBATCH --partition={os.getenv('LABELCRITIC_SELECTION_PARTITION', 'cpu')}",
        f"#SBATCH --cpus-per-task={os.getenv('LABELCRITIC_SELECTION_CPUS', '2')}",
        f"#SBATCH --mem={os.getenv('LABELCRITIC_SELECTION_MEM', '12G')}",
        f"#SBATCH --time={os.getenv('LABELCRITIC_SELECTION_TIME', '12:00:00')}",
        "#SBATCH --signal=B:USR1@900",
        f"#SBATCH --output={_state_paths(args.state_root)['root'] / 'labelcritic_select_%A_%a.out'}",
        f"#SBATCH --error={_state_paths(args.state_root)['root'] / 'labelcritic_select_%A_%a.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export STATE_ROOT={shlex.quote(str(args.state_root))}",
        f"export LABELCRITIC_BASE_URL={shlex.quote(str(endpoint['base_url']))}",
        f"export LABELCRITIC_PORT={shlex.quote(str(endpoint['port']))}",
        f"export LABELCRITIC_MODEL_ID={shlex.quote(LABELCRITIC_MODEL_ID)}",
        "export RUNTIME_NO_GIT=1",
        "export SKIP_GIT_SYNC=1",
        "export GIT_TERMINAL_PROMPT=0",
        "trap 'python tools/dataset_delivery/slurm_reliability.py worker-pretimeout --state-root \"$STATE_ROOT\" --logical-task-id labelcritic_selection --job-id \"${SLURM_JOB_ID:-}\" --task-index \"${SLURM_ARRAY_TASK_ID:-}\"' USR1",
        f"{shlex.quote(str(args.python))} tools/dataset_delivery/task2_full373_round1_launcher.py \\",
        f"  --output-root {shlex.quote(str(formal_root))} \\",
        f"  --state-root {shlex.quote(str(args.state_root))} \\",
        f"  --target-config {shlex.quote(str(args.target_config))} \\",
        "  --selection-worker \\",
        "  --worker-id \"${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}\" \\",
        "  --critic-base-url \"$LABELCRITIC_BASE_URL\" \\",
        "  --critic-port \"$LABELCRITIC_PORT\"",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path), "worker_count": worker_count, "endpoint": endpoint}


def submit_labelcritic_selection_workers(args: argparse.Namespace, labelcritic: dict[str, Any]) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    formal_root = Path(str(state.get("formal_root") or (paths["root"] / FULL373_ROOT_NAME))).resolve()
    if not formal_root.exists():
        return {"status": "WAITING_FOR_ESTEP_ROOT", "formal_root": str(formal_root)}
    telemetry = build_estep_telemetry(formal_root)
    if int((telemetry.get("labelcritic") or {}).get("queue_depth") or 0) <= 0 and int((telemetry.get("case_target") or {}).get("terminal") or 0) >= int((telemetry.get("case_target") or {}).get("total") or 1):
        return {"status": "NO_SELECTION_WORKERS_NEEDED", "telemetry": telemetry}
    existing = str(state.get("labelcritic_selection_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
        return {"status": "REUSED_ACTIVE_SELECTION_WORKERS", "job_id": existing, "telemetry": telemetry}
    rendered = render_labelcritic_selection_sbatch(args, formal_root, paths["labelcritic_selection_sbatch"], labelcritic)
    for command in (["bash", "-n", str(paths["labelcritic_selection_sbatch"])], ["sbatch", "--test-only", str(paths["labelcritic_selection_sbatch"])]):
        result = _run(command)
        if not result["ok"]:
            log_failure(state_root, stage="labelcritic_selection_sbatch_preflight", failure_reason="labelcritic_selection_sbatch_preflight_failed", details=result)
            return {"status": "FAILED", "failure_reason": "labelcritic_selection_sbatch_preflight_failed", "preflight": result}
    worker_count = int(rendered["worker_count"])
    run_id = _run_id(state_root)
    result = _run([
        "sbatch",
        "--parsable",
        "--comment", slurm_comment(run_id=run_id, submission_id="labelcritic_selection", group="labelcritic", profile="cpu"),
        f"--array=0-{worker_count - 1}%{worker_count}",
        str(paths["labelcritic_selection_sbatch"]),
    ])
    if not result["ok"]:
        classification = classify_sbatch_failure(result["stderr"] or result["stdout"])
        if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
            _save_state(state_root, labelcritic_selection_status="WAITING_FOR_SUBMISSION_CAPACITY", scheduler_status="BACKPRESSURED", labelcritic_selection_submit=result)
            return {"status": "WAITING_FOR_SUBMISSION_CAPACITY", "scheduler_status": "BACKPRESSURED", "failure_reason": classification["reason"], "telemetry": telemetry}
        log_failure(state_root, stage="labelcritic_selection_submit", failure_reason=result["stderr"] or "labelcritic_selection_submit_failed", details=result)
        return {"status": "FAILED", "failure_reason": result["stderr"] or "labelcritic_selection_submit_failed", "submit": result}
    job_id = result["stdout"].splitlines()[-1].strip()
    paths["labelcritic_selection_job"].write_text(job_id + "\n", encoding="utf-8")
    _save_state(state_root, labelcritic_selection_status="SUBMITTED", labelcritic_selection_job_id=job_id, labelcritic_selection_sbatch=rendered)
    record_job_lifecycle(state_root, {"status": "SUBMITTED", "job_id": job_id, "logical_task_id": f"{run_id}:labelcritic_selection", "stage": "labelcritic_selection", "profile": "cpu", "array_concurrency": worker_count})
    return {"status": "SUBMITTED", "job_id": job_id, "worker_count": worker_count, "telemetry": telemetry}


def advance_estep(args: argparse.Namespace, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    manifests = ensure_case_level_manifests(args)
    source_manifest = Path(str(manifests["source_manifest"]))
    staging = submit_staging_workers(args, source_manifest)
    if staging["status"] == "FAILED":
        return {"status": "FAILED", "failure_reason": staging.get("failure_reason", "staging_failed"), "staging": staging, "manifests": manifests}
    teachers = submit_ready_teacher_batch(args, source_manifest, labelcritic)
    if teachers["status"] == "FAILED":
        return {"status": "FAILED", "failure_reason": teachers.get("failure_reason", "teacher_submit_failed"), "staging": staging, "teachers": teachers, "manifests": manifests}
    statuses = _staging_status_rows(args, source_manifest)
    return {
        "status": "SUBMITTED",
        "manifests": manifests,
        "staging": staging,
        "teachers": teachers,
        "ready_count": sum(1 for row in statuses if row["status"] == "INPUT_READY"),
        "staging_pending_count": sum(1 for row in statuses if row["status"] == "STAGING_PENDING"),
        "staging_running_count": sum(1 for row in statuses if row["status"] == "STAGING_RUNNING"),
        "staging_failed_count": sum(1 for row in statuses if row["status"] == "STAGING_FAILED"),
        "case_count": len(statuses),
    }


def submit_estep(args: argparse.Namespace, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    state = _load_state(state_root)
    formal_root = Path(str(state.get("formal_root") or (_state_paths(state_root)["root"] / FULL373_ROOT_NAME))).resolve()
    full_summary = _read_json(formal_root / "full_373_estep_status.json", {})
    if state.get("e_step_status") in {"SUBMITTED", "PASSED"} and (
        (formal_root / "slurm" / "submitted_jobs.csv").exists()
        or full_summary.get("status") == "PASSED"
    ):
        return {"status": "REUSED", "formal_root": str(formal_root), "e_step_status": state.get("e_step_status")}
    result = advance_estep(args, labelcritic)
    if result["status"] == "FAILED":
        log_failure(state_root, stage="e_step_submit", failure_reason=result.get("failure_reason", "e_step_submit_failed"), details=result)
        return {"status": "FAILED", "failure_reason": result.get("failure_reason"), "submit": result, "formal_root": str(formal_root)}
    _save_state(state_root, e_step_status="SUBMITTED", formal_root=str(formal_root), e_step_submit=result)
    return {"status": "SUBMITTED", "formal_root": str(formal_root), "submit": result}


def check_estep(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_state(args.state_root.resolve())
    formal_root = Path(str(state.get("formal_root") or ""))
    if not formal_root.exists():
        return {"status": "PENDING", "reason": "formal_root_missing"}
    aggregate = aggregate_full373_estep(formal_root, expected_cases=FORMAL_CASE_COUNT, expected_targets=373)
    if aggregate["status"] == "PASSED":
        return {"status": "PASSED", "formal_root": str(formal_root), "check": aggregate}
    result = {"ok": aggregate["status"] != "FAILED", "stdout": json.dumps(aggregate), "stderr": "", "return_code": 0 if aggregate["status"] != "FAILED" else 2}
    jobs_csv = formal_root / "slurm" / "submitted_jobs.csv"
    active = []
    failed = []
    retryable = []
    if jobs_csv.exists():
        lifecycle = refresh_lifecycle_for_submitted_jobs(args, formal_root)
        for row in read_csv_rows(jobs_csv):
            job_id = str(row.get("job_id") or "")
            state = slurm_job_state(job_id)
            if state.get("state") in ACTIVE_STATES:
                active.append(state)
            if state.get("state") in RETRYABLE_TERMINAL_STATES:
                retryable.append(state)
            elif state.get("state") in TERMINAL_FAILURE_STATES:
                failed.append(state)
    else:
        lifecycle = {"status": "NO_SUBMITTED_JOBS"}
    if failed and not active:
        log_failure(args.state_root.resolve(), stage="e_step", failure_reason="e_step_jobs_terminal_failed", details={"failed_jobs": failed, "check": result})
        return {"status": "FAILED", "failure_reason": "e_step_jobs_terminal_failed", "failed_jobs": failed, "check": result}
    if retryable and not active:
        _save_state(
            args.state_root.resolve(),
            scheduler_status="RETRY_PENDING",
            walltime_recovery_status="RETRY_PENDING",
            e_step_retryable_terminal_jobs=retryable,
        )
        return {
            "status": "RUNNING",
            "scheduler_status": "RETRY_PENDING",
            "retryable_terminal_jobs": retryable,
            "lifecycle": lifecycle,
            "check": result,
        }
    source_manifest = _state_paths(args.state_root.resolve())["source_manifest"]
    if source_manifest.exists():
        staging_rows = _staging_status_rows(args, source_manifest)
        ready_count = sum(1 for row in staging_rows if row["status"] == "INPUT_READY")
        staging_failed_count = sum(1 for row in staging_rows if row["status"] == "STAGING_FAILED")
        staging_job_id = str(_load_state(args.state_root.resolve()).get("staging_job_id") or "")
        staging_job = slurm_job_state(staging_job_id) if staging_job_id else {"state": "UNKNOWN"}
        if staging_failed_count and ready_count + staging_failed_count >= len(staging_rows) and not active:
            log_failure(args.state_root.resolve(), stage="staging", failure_reason="one_or_more_case_staging_failed", details={"staging_failed_count": staging_failed_count, "ready_count": ready_count})
            return {"status": "FAILED", "failure_reason": "one_or_more_case_staging_failed", "staging_failed_count": staging_failed_count, "ready_count": ready_count, "check": result}
        if staging_job.get("state") in ACTIVE_STATES:
            active.append({"stage": "staging", **staging_job})
    return {"status": "RUNNING", "active_jobs": active, "lifecycle": lifecycle, "check": result}


def build_mstep_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from cli_anything.medai.core.continual_learning import TRAINING_CONTRACT_VERSION, canonicalize_training_record

    paths = _state_paths(args.state_root.resolve())
    if paths["mstep_manifest"].exists():
        doc = _read_json(paths["mstep_manifest"], {})
        if doc.get("items"):
            return {"status": "REUSED", "manifest": str(paths["mstep_manifest"]), "num_items": len(doc.get("items") or [])}
    state = _load_state(args.state_root.resolve())
    formal_root = Path(str(state.get("formal_root") or ""))
    full_manifest = _read_json(formal_root / "training_manifest.json", {})
    full_gate = _read_json(formal_root / "full_373_estep_status.json", {})
    if full_manifest.get("items"):
        if full_gate.get("status") != "PASSED":
            return {
                "status": "FAILED",
                "failure_reason": "full_373_estep_gate_not_passed",
                "gate": full_gate,
                "manifest": str(formal_root / "training_manifest.json"),
            }
        manifest = dict(full_manifest)
        manifest.update(
            {
                "version": "full_373_multiteacher_round1_voxtell_manifest_v1",
                "source_formal_root": str(formal_root),
                "source_stage": "full_373_multiteacher_round1_estep",
                "training_contract_version": manifest.get("training_contract_version") or TRAINING_CONTRACT_VERSION,
            }
        )
        _write_json(paths["mstep_manifest"], manifest)
        return {"status": "REUSED", "manifest": str(paths["mstep_manifest"]), "num_items": len(manifest.get("items") or [])}
    rows_doc = _read_json(formal_root / "task2_formal_case_target_status.json", {})
    if formal_root.name == FULL373_ROOT_NAME or not rows_doc.get("rows"):
        return {
            "status": "FAILED",
            "failure_reason": "full_373_training_manifest_missing",
            "formal_root": str(formal_root),
            "expected_manifest": str(formal_root / "training_manifest.json"),
        }
    case_rows = {str(row.get("case_id") or ""): row for row in read_csv_rows(args.case_manifest)}
    target_doc = _read_json(args.target_config, {})
    targets = list(target_doc.get("target_organs") or [])
    organ_to_prompt = target_doc.get("organ_to_prompt") or {}
    items = []
    for row in rows_doc.get("rows") or []:
        if row.get("final_status") != "generated_valid_mask":
            continue
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("target_name") or "")
        mask = str(row.get("mask_path") or "")
        ct = str((case_rows.get(case_id) or {}).get("ct_path") or "")
        if not case_id or not organ or not mask or not ct or organ not in targets:
            continue
        provider = str(row.get("selected_model") or row.get("source_model") or row.get("group") or row.get("target_group") or "task2_teacher")
        item = {
            "case_id": case_id,
            "image": ct,
            "ct_path": ct,
            "organ": organ,
            "canonical_organ": organ,
            "requested_canonical_id": organ,
            "resolved_canonical_id": organ,
            "prompt": organ_to_prompt.get(organ, organ.replace("_", " ")),
            "mask": mask,
            "mask_path": mask,
            "supervision_type": "positive",
            "target_type": "positive_hard",
            "label_role": "selected_pseudo_label",
            "supervision_role": "selected_pseudo_label",
            "distillation_role": "positive",
            "dataset_role": "pseudo_label",
            "source_model": provider,
            "selected_model": provider,
            "origin_provider": provider,
            "ground_truth_status": "selected_pseudo_label_not_expert_gt",
            "scoring_schema_version": "autolabel_core_v3",
            "grade": "A",
            "training_weight": 1.0,
            "distillation_eligible": True,
            "training_eligible": True,
            "student_target_id": targets.index(organ),
            "source_stage": "task2_formal_103case_round1_estep",
        }
        items.append(canonicalize_training_record(item, round_index=1, project_root=REPO_ROOT, strict_soft=True))
    manifest = {
        "version": "task2_formal_103_round1_voxtell_manifest_v1",
        "stage": "round1_mstep_manifest",
        "status": "success" if items else "failed",
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "source_formal_root": str(formal_root),
        "target_config": str(args.target_config),
        "num_items": len(items),
        "num_cases": len({item.get("case_id") for item in items}),
        "num_distillation_eligible_items": sum(1 for item in items if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0),
        "num_trainable_positive_items": sum(1 for item in items if item.get("training_eligible") is True and item.get("supervision_type") == "positive"),
        "novelty_audit": {
            "stage": "continual_learning_novelty",
            "status": "success",
            "decision": "full_update",
            "max_steps": int(os.getenv("MEDAI_MAX_STEPS", "2000")),
            "weighted_change_ratio": 1.0,
            "added_keys": [],
            "changed_keys": [],
            "removed_keys": [],
        },
        "items": items,
    }
    _write_json(paths["mstep_manifest"], manifest)
    return {"status": manifest["status"].upper(), "manifest": str(paths["mstep_manifest"]), "num_items": len(items)}


def _mstep_profiles() -> list[dict[str, str]]:
    if os.getenv("MSTEP_PARTITION") or os.getenv("MSTEP_GRES"):
        return [{
            "name": os.getenv("MSTEP_PROFILE", os.getenv("MSTEP_PARTITION", "custom")),
            "partition": os.getenv("MSTEP_PARTITION", "gpua100"),
            "gres": os.getenv("MSTEP_GRES", "gpu:A100:1"),
            "cpus": os.getenv("MSTEP_CPUS", "12"),
            "mem": os.getenv("MSTEP_MEM", "96G"),
            "time": os.getenv("MSTEP_TIME", "10:00:00"),
        }]
    specs = os.getenv(
        "MSTEP_PROFILE_SPECS",
        "student_a100|gpua100|gpu:A100:1|12|96G|10:00:00,student_h100|gpuh100|gpu:H100:1|12|96G|10:00:00",
    )
    profiles = []
    for raw in specs.split(","):
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 6:
            continue
        name, partition, gres, cpus, mem, time_limit = parts
        profiles.append({"name": name, "partition": partition, "gres": gres, "cpus": cpus, "mem": mem, "time": time_limit})
    return profiles or [{"name": "student_a100", "partition": "gpua100", "gres": "gpu:A100:1", "cpus": "12", "mem": "96G", "time": "10:00:00"}]


def _write_mstep_sbatch(args: argparse.Namespace, profile: dict[str, str], path: Path, manifest: dict[str, Any]) -> None:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_round1_mstep_student",
        f"#SBATCH --partition={profile['partition']}",
        f"#SBATCH --gres={profile['gres']}",
        f"#SBATCH --cpus-per-task={profile['cpus']}",
        f"#SBATCH --mem={profile['mem']}",
        f"#SBATCH --time={profile['time']}",
        "#SBATCH --signal=B:USR1@900",
        f"#SBATCH --output={paths['root'] / ('mstep_' + profile['name'] + '_%j.out')}",
        f"#SBATCH --error={paths['root'] / ('mstep_' + profile['name'] + '_%j.err')}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export STATE_ROOT={shlex.quote(str(state_root))}",
        f"export MEDAI_OUTPUT_ROOT={shlex.quote(str(paths['root']))}",
        f"export MEDAI_VOXTELL_MODEL_DIR={shlex.quote(str(os.getenv('MEDAI_VOXTELL_MODEL_DIR', '/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/VoxTell/voxtell_v1.1')))}",
        "export MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt",
        "export MEDAI_VOXTELL_MSTEP_MODE=project_voxtell_prompt_distillation_student",
        "export MEDAI_VOXTELL_TRAINING_PROFILE=quality_weighted_ablation",
        "export MEDAI_MSTEP_BATCH_SIZE=1",
        "export MEDAI_FORMAL_STATE_MACHINE=1",
        "export RUNTIME_NO_GIT=1",
        "export SKIP_GIT_SYNC=1",
        "export GIT_TERMINAL_PROMPT=0",
        f"trap 'python tools/dataset_delivery/slurm_reliability.py student-pretimeout --state-root {shlex.quote(str(state_root))} --checkpoint-dir {shlex.quote(str(paths['mstep_output'] / 'walltime_checkpoints'))} --job-id \"${{SLURM_JOB_ID:-}}\"' USR1",
        f"{shlex.quote(str(args.python))} - <<'PY'",
        "from pathlib import Path",
        "import json",
        "import torch",
        f"manifest = Path({str(paths['mstep_manifest'])!r})",
        "assert manifest.is_file(), f'missing training manifest: {manifest}'",
        "assert torch.cuda.is_available(), 'CUDA is not available for Student M-step'",
        "print(json.dumps({'stage':'student_runtime_preflight','status':'PASSED','cuda':True,'device':torch.cuda.get_device_name(0)}), flush=True)",
        "from scripts import run_em_training as em",
        f"result = em.run_prompt_student_mstep(1, Path({str(paths['mstep_manifest'])!r}))",
        "raise SystemExit(0 if result.get('status') == 'success' else 2)",
        "PY",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def submit_mstep(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    completed = _read_json(paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json", {})
    if state.get("mstep_status") == "PASSED" and completed.get("status") == "success":
        return {"status": "REUSED", "result": str(paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json")}
    existing = str(state.get("mstep_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in (ACTIVE_STATES | SUCCESS_STATES):
        return {"status": "REUSED", "job_id": existing}
    manifest = build_mstep_manifest(args)
    if manifest["status"] not in {"SUCCESS", "REUSED"}:
        log_failure(state_root, stage="m_step_manifest", failure_reason="mstep_manifest_failed", details=manifest)
        return {"status": "FAILED", "failure_reason": "mstep_manifest_failed", "manifest": manifest}
    run_id = _run_id(state_root)
    backpressured = []
    preflight_failures = []
    for profile in _mstep_profiles():
        sbatch_path = paths["mstep_sbatch"].with_name(f"round1_mstep_student_{profile['name']}.sbatch")
        _write_mstep_sbatch(args, profile, sbatch_path, manifest)
        failed_preflight = None
        for command in (["bash", "-n", str(sbatch_path)], ["sbatch", "--test-only", str(sbatch_path)]):
            result = _run(command)
            if not result["ok"]:
                failed_preflight = result
                break
        if failed_preflight is not None:
            preflight_failures.append({"profile": profile, "preflight": failed_preflight})
            continue
        result = _run(["sbatch", "--parsable", "--comment", slurm_comment(run_id=run_id, submission_id="mstep", group="student", profile=profile["name"]), str(sbatch_path)])
        if result["ok"]:
            job_id = result["stdout"].splitlines()[-1].strip()
            paths["mstep_job"].write_text(job_id + "\n", encoding="utf-8")
            _save_state(state_root, mstep_status="SUBMITTED", mstep_job_id=job_id, mstep_manifest=manifest, mstep_profile=profile, mstep_sbatch=str(sbatch_path))
            record_job_lifecycle(state_root, {"status": "SUBMITTED", "job_id": job_id, "logical_task_id": f"{run_id}:mstep", "stage": "mstep", "profile": profile["name"], "partition": profile["partition"], "gres": profile["gres"]})
            return {"status": "SUBMITTED", "job_id": job_id, "manifest": manifest, "profile": profile}
        classification = classify_sbatch_failure(result["stderr"] or result["stdout"])
        if classification["class"] == "TRANSIENT_RESOURCE_BACKPRESSURE":
            backpressured.append({"profile": profile, "failure_reason": classification["reason"], "submit": result})
            continue
        log_failure(state_root, stage="m_step_submit", failure_reason=result["stderr"] or "mstep_submit_failed", details=result)
        return {"status": "FAILED", "failure_reason": result["stderr"], "submit": result}
    if backpressured:
        _save_state(state_root, mstep_status="WAITING_FOR_SUBMISSION_CAPACITY", scheduler_status="BACKPRESSURED", mstep_submit={"backpressured": backpressured, "preflight_failures": preflight_failures})
        return {"status": "WAITING_FOR_SUBMISSION_CAPACITY", "scheduler_status": "BACKPRESSURED", "failure_reason": "all_mstep_profiles_backpressured", "backpressured": backpressured, "preflight_failures": preflight_failures}
    log_failure(state_root, stage="m_step_sbatch_preflight", failure_reason="no_mstep_profile_passed_preflight", details={"preflight_failures": preflight_failures})
    return {"status": "FAILED", "failure_reason": "no_mstep_profile_passed_preflight", "preflight_failures": preflight_failures}


def check_mstep(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    result_json = paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    result = _read_json(result_json, {})
    if result.get("status") == "success":
        checkpoint = paths["mstep_output"] / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth"
        if checkpoint.exists():
            return {"status": "PASSED", "result": str(result_json), "checkpoint": str(checkpoint), "failure_reason": ""}
        payload = {"status": "FAILED", "result": str(result_json), "checkpoint": str(checkpoint), "failure_reason": "checkpoint_missing"}
        log_failure(args.state_root.resolve(), stage="m_step", failure_reason="checkpoint_missing", details=payload)
        return payload
    state = _load_state(args.state_root.resolve())
    job_id = str(state.get("mstep_job_id") or "")
    job_state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
    if job_state.get("state") in TERMINAL_FAILURE_STATES:
        if job_state.get("state") in RETRYABLE_TERMINAL_STATES:
            checkpoint = paths["mstep_output"] / "walltime_checkpoints"
            marker = student_pretimeout(args.state_root.resolve(), checkpoint, job_id=job_id)
            _save_state(args.state_root.resolve(), mstep_status="RETRY_PENDING", walltime_recovery_status="RETRY_PENDING", mstep_walltime_guard=marker)
            return {"status": "RUNNING", "scheduler_status": "RETRY_PENDING", "job": job_state, "walltime_guard": marker}
        log_failure(args.state_root.resolve(), stage="m_step", failure_reason=f"mstep_job_terminal:{job_state.get('state')}", details=job_state)
        return {"status": "FAILED", "failure_reason": f"mstep_job_terminal:{job_state.get('state')}", "job": job_state}
    return {"status": "RUNNING", "job": job_state}


def run_round1_final_validator(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    state = _load_state(args.state_root.resolve())
    mstep_result_path = paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    mstep_result = _read_json(mstep_result_path, {})
    checkpoint = Path(str(mstep_result.get("inference_checkpoint") or paths["root"] / "round1" / "mstep" / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth"))
    formal_root = Path(str(state.get("formal_root") or ""))
    full_gate = _read_json(formal_root / "full_373_estep_status.json", {})
    formal_status = _read_json(formal_root / "task2_formal_summary.json", {})
    formal_validator_status = full_gate.get("status") or formal_status.get("status") or formal_status.get("validation_status")
    checks = [
        {"name": "e_step_status_passed", "ok": state.get("e_step_status") == "PASSED", "detail": state.get("e_step_status")},
        {
            "name": "full_373_estep_gate_passed",
            "ok": bool(
                full_gate.get("status") == "PASSED"
                and full_gate.get("case_count") == FORMAL_CASE_COUNT
                and full_gate.get("canonical_target_count") == 373
                and full_gate.get("manifest_targets") == FORMAL_CASE_COUNT * 373
            ),
            "detail": formal_validator_status or "missing_full_373_estep_status",
        },
        {"name": "mstep_result_success", "ok": mstep_result.get("status") == "success", "detail": str(mstep_result_path)},
        {"name": "mstep_checkpoint_exists", "ok": checkpoint.is_file(), "detail": str(checkpoint)},
        {
            "name": "student_checkpoint_eligible",
            "ok": bool(mstep_result.get("eligible_for_next_round_prompt_student") or mstep_result.get("checkpoint_eligible_for_next_round")),
            "detail": mstep_result.get("training_status"),
        },
    ]
    report = {
        "status": "PASSED" if all(check["ok"] for check in checks) else "FAILED",
        "terminal_state": "ROUND1_PASSED" if all(check["ok"] for check in checks) else "ROUND1_FAILED",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "state_root": str(paths["root"]),
        "formal_root": str(formal_root),
        "mstep_result": str(mstep_result_path),
        "checks": checks,
        "failure_reason": "" if all(check["ok"] for check in checks) else ",".join(check["name"] for check in checks if not check["ok"]),
    }
    _write_json(paths["final"], report)
    if report["status"] != "PASSED":
        log_failure(args.state_root.resolve(), stage="final_validator", failure_reason=report["failure_reason"], details=report)
    return report


def _controller_main(args: argparse.Namespace) -> int:
    state_root = args.state_root.resolve()
    current = _load_state(state_root)
    if current.get("terminal_state") == "ROUND1_PASSED":
        return 0
    if current.get("terminal_state") == "ROUND1_FAILED":
        return 2
    run_id = _run_id(state_root)
    _save_state(state_root, status="CONTROLLER_RUNNING", terminal_state="", stage="start", git_commit=_git_commit(), run_id=run_id, started_at=utc_now())
    commit_pin = verify_expected_git_commit(state_root, str(getattr(args, "expected_git_commit", "") or ""))
    _save_state(state_root, stage="git_commit_pin", expected_git_commit=commit_pin)
    if commit_pin["status"] != "PASSED":
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="git_commit_pin", failure_reason=commit_pin["failure_reason"], expected_git_commit=commit_pin)
        return 2
    labelcritic = ensure_labelcritic_service(state_root)
    if labelcritic["status"] == "SUBMIT_FAILED":
        log_failure(state_root, stage="labelcritic_submit", failure_reason=labelcritic.get("failure_reason", "labelcritic_submit_failed"), details=labelcritic)
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="labelcritic_submit", failure_reason=labelcritic.get("failure_reason"), labelcritic=labelcritic)
        return 2
    _save_state(state_root, stage="labelcritic_submitted_or_reused", labelcritic=labelcritic)
    reconcile_active_teacher_jobs(args)
    estep_submit = submit_estep(args, labelcritic)
    if estep_submit["status"] == "FAILED":
        log_failure(state_root, stage="e_step_submit", failure_reason=estep_submit.get("failure_reason", "e_step_submit_failed"), details=estep_submit)
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="e_step_submit", failure_reason=estep_submit.get("failure_reason"), e_step=estep_submit)
        return 2
    labelcritic_gate: dict[str, Any] = {"status": "WAITING"}
    while True:
        if labelcritic_gate.get("status") != "PASSED":
            labelcritic_gate = poll_labelcritic_runtime(state_root)
            if labelcritic_gate["status"] == "FAILED":
                log_failure(state_root, stage="labelcritic", failure_reason=labelcritic_gate.get("failure_reason", "labelcritic_failed"), details=labelcritic_gate)
                _save_state(state_root, terminal_state="ROUND1_FAILED", stage="labelcritic", failure_reason=labelcritic_gate.get("failure_reason"), labelcritic=labelcritic_gate)
                return 2
            if labelcritic_gate["status"] == "PASSED":
                _save_state(state_root, stage="labelcritic_ready", labelcritic=labelcritic_gate)
        reconcile_active_teacher_jobs(args)
        progress = advance_estep(args, labelcritic_gate)
        if progress["status"] == "FAILED":
            log_failure(state_root, stage="e_step_submit", failure_reason=progress.get("failure_reason", "e_step_submit_failed"), details=progress)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="e_step_submit", failure_reason=progress.get("failure_reason"), e_step=progress)
            return 2
        selection = {"status": "WAITING_FOR_LABELCRITIC"}
        if labelcritic_gate.get("status") == "PASSED":
            selection = submit_labelcritic_selection_workers(args, labelcritic_gate)
            if selection["status"] == "FAILED":
                log_failure(state_root, stage="labelcritic_selection_submit", failure_reason=selection.get("failure_reason", "labelcritic_selection_submit_failed"), details=selection)
                _save_state(state_root, terminal_state="ROUND1_FAILED", stage="labelcritic_selection_submit", failure_reason=selection.get("failure_reason"), labelcritic_selection=selection)
                return 2
        estep = check_estep(args)
        wait_stage = "e_step_wait" if labelcritic_gate.get("status") == "PASSED" else "e_step_and_labelcritic_wait"
        _save_state(state_root, stage=wait_stage, e_step=estep, e_step_status=estep["status"], labelcritic=labelcritic_gate, labelcritic_selection=selection, e_step_progress=progress)
        if estep["status"] == "PASSED":
            if labelcritic_gate.get("status") == "PASSED":
                break
            time.sleep(max(10, args.poll_sec))
            continue
        if estep["status"] == "FAILED":
            log_failure(state_root, stage="e_step", failure_reason=estep.get("failure_reason", "e_step_failed"), details=estep)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="e_step", failure_reason=estep.get("failure_reason"), e_step=estep)
            return 2
        time.sleep(max(10, args.poll_sec))
    while True:
        mstep_submit = submit_mstep(args)
        if mstep_submit["status"] == "FAILED":
            log_failure(state_root, stage="m_step_submit", failure_reason=mstep_submit.get("failure_reason", "m_step_submit_failed"), details=mstep_submit)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="m_step_submit", failure_reason=mstep_submit.get("failure_reason"), m_step=mstep_submit)
            return 2
        if mstep_submit["status"] in BACKPRESSURE_STATUSES:
            _save_state(state_root, stage="m_step_submit_backpressured", m_step=mstep_submit, scheduler_status="BACKPRESSURED")
            time.sleep(max(10, args.poll_sec))
            continue
        break
    while True:
        mstep = check_mstep(args)
        _save_state(state_root, stage="m_step_wait", m_step=mstep, mstep_status=mstep["status"])
        if mstep["status"] == "PASSED":
            final = run_round1_final_validator(args)
            if final["status"] == "PASSED":
                _save_state(state_root, terminal_state="ROUND1_PASSED", stage="complete", final=final, finished_at=utc_now())
                return 0
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="final_validator", failure_reason=final.get("failure_reason"), final=final, finished_at=utc_now())
            return 2
        if mstep["status"] == "FAILED":
            log_failure(state_root, stage="m_step", failure_reason=mstep.get("failure_reason", "m_step_failed"), details=mstep)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="m_step", failure_reason=mstep.get("failure_reason"), m_step=mstep)
            return 2
        time.sleep(max(10, args.poll_sec))


def controller(args: argparse.Namespace) -> int:
    try:
        return _controller_main(args)
    except Exception as exc:
        state_root = args.state_root.resolve()
        failure = log_failure(
            state_root,
            stage="controller_unhandled_exception",
            failure_reason=f"{type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
        _save_state(
            state_root,
            terminal_state="ROUND1_FAILED",
            stage="controller_unhandled_exception",
            failure_reason=failure["failure_reason"],
            last_failure=str(_state_paths(state_root)["last_failure"]),
        )
        return 2


def _resource_telemetry(state_root: Path) -> dict[str, Any]:
    state = _load_state(state_root)
    formal_root = Path(str(state.get("formal_root") or ""))
    jobs_csv = formal_root / "slurm" / "submitted_jobs.csv"
    worker_counts = {
        "t4_running": 0,
        "t4_pending": 0,
        "interactive_running": 0,
        "interactive_pending": 0,
        "a100_running": 0,
        "a100_pending": 0,
        "h100_running": 0,
        "h100_pending": 0,
    }
    if jobs_csv.exists():
        for row in read_csv_rows(jobs_csv):
            job_state = slurm_job_state(str(row.get("job_id") or ""))
            if job_state.get("state") not in ACTIVE_STATES:
                continue
            gres = str(row.get("gres") or row.get("profile") or "").upper()
            profile = str(row.get("profile") or "").lower()
            is_running = str(job_state.get("state") or "") == "RUNNING"
            suffix = "running" if is_running else "pending"
            if "INTERACTIVE" in gres or "interactive" in profile:
                worker_counts[f"interactive_{suffix}"] += 1
            elif "A100" in gres or "a100" in profile:
                worker_counts[f"a100_{suffix}"] += 1
            elif "H100" in gres or "h100" in profile:
                worker_counts[f"h100_{suffix}"] += 1
            else:
                worker_counts[f"t4_{suffix}"] += 1
    labelcritic = state.get("labelcritic") or {}
    label_job = slurm_job_state(str(labelcritic.get("job_id") or "")) if isinstance(labelcritic, dict) else {"state": "UNKNOWN"}
    mstep_job = slurm_job_state(str(state.get("mstep_job_id") or "")) if state.get("mstep_job_id") else {"state": "NOT_SUBMITTED"}
    qos_cache = _read_json(formal_root / "slurm" / "qos_capacity_cache.json", {}) if formal_root.exists() else {}
    dynamic_plan = _read_json(formal_root / "slurm" / "dynamic_gpu_submission_plan.json", {}) if formal_root.exists() else {}
    stored_policy = dynamic_plan.get("teacher_h100_policy") or state.get("teacher_h100_policy") or {}
    h100_policy = resolve_teacher_h100_policy(
        state_root=state_root,
        labelcritic=labelcritic if isinstance(labelcritic, dict) else None,
        labelcritic_job_id=str(stored_policy.get("labelcritic_job_id") or "") if isinstance(stored_policy, dict) else None,
        slurm_job_state_fn=slurm_job_state,
        find_labelcritic_job_fn=find_labelcritic_job_by_name,
        source="orchestrator_status",
    )
    return {
        "teacher_workers": worker_counts,
        "task_ownership": dynamic_plan.get("task_ownership", ""),
        "profile_binding": dynamic_plan.get("profile_binding"),
        "primary_teacher_profile": dynamic_plan.get("primary_teacher_profile", ""),
        "allow_h100_teacher_overflow": h100_policy.get("allow_h100_teacher_overflow"),
        "labelcritic_required": h100_policy.get("labelcritic_required"),
        "labelcritic_job_found": h100_policy.get("labelcritic_job_found"),
        "labelcritic_job_id": h100_policy.get("labelcritic_job_id"),
        "labelcritic_job_state": h100_policy.get("labelcritic_job_state"),
        "labelcritic_h100_reserved": h100_policy.get("labelcritic_h100_reserved"),
        "effective_teacher_h100_enabled": h100_policy.get("effective_teacher_h100_enabled"),
        "teacher_h100_deferred_for_labelcritic": h100_policy.get("teacher_h100_deferred_for_labelcritic"),
        "desired_teacher_h100_workers": dynamic_plan.get("desired_teacher_h100_workers"),
        "teacher_h100_policy": h100_policy,
        "qos": {
            "per_profile_known_good": {
                str(name): int((doc or {}).get("known_good_size") or 0)
                for name, doc in ((qos_cache.get("profiles") or {}) if isinstance(qos_cache, dict) else {}).items()
            },
            "per_profile_known_bad": {
                str(name): int((doc or {}).get("known_bad_size") or 0)
                for name, doc in ((qos_cache.get("profiles") or {}) if isinstance(qos_cache, dict) else {}).items()
                if (doc or {}).get("known_bad_size") is not None
            },
            "last_backpressure": {
                str(name): (doc or {}).get("last_backpressure") or {}
                for name, doc in ((qos_cache.get("profiles") or {}) if isinstance(qos_cache, dict) else {}).items()
                if (doc or {}).get("last_backpressure")
            },
        },
        "labelcritic_h100_state": label_job.get("state"),
        "student_state": mstep_job.get("state"),
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_state(args.state_root.resolve())
    paths = _state_paths(args.state_root.resolve())
    last_failure = _read_json(paths["last_failure"], {})
    formal_root = Path(str(state.get("formal_root") or ""))
    estep_telemetry = build_estep_telemetry(formal_root) if formal_root.exists() else {}
    labelcritic = state.get("labelcritic") or {}
    if isinstance(labelcritic, dict) and labelcritic.get("job_id"):
        labelcritic = {**labelcritic, "service_state": slurm_job_state(str(labelcritic.get("job_id"))).get("state")}
    return {
        "status": state.get("terminal_state") or state.get("stage") or state.get("status") or "NOT_STARTED",
        "state_root": str(paths["root"]),
        "state_json": str(paths["state"]),
        "events_log": str(paths["events"]),
        "failure_log": str(paths["failures"]),
        "last_failure_json": str(paths["last_failure"]),
        "controller_job_id": state.get("controller_job_id"),
        "labelcritic": labelcritic,
        "teacher": (estep_telemetry.get("teacher_candidate") or {}),
        "case_target": (estep_telemetry.get("case_target") or {}),
        "labelcritic_queue": (estep_telemetry.get("labelcritic") or {}),
        "resources": _resource_telemetry(args.state_root.resolve()),
        "e_step_status": state.get("e_step_status"),
        "mstep_status": state.get("mstep_status"),
        "failure_reason": state.get("failure_reason", "") or last_failure.get("failure_reason", ""),
        "last_failure": last_failure,
        "formal_root": state.get("formal_root", ""),
        "scheduler_status": state.get("scheduler_status", ""),
        "walltime_recovery_status": state.get("walltime_recovery_status", ""),
        "job_lifecycle": str(paths["job_lifecycle"]),
        "job_lifecycle_current": str(paths["job_lifecycle_current"]),
        "walltime_guard": str(paths["walltime_guard"]),
        "updated_at": state.get("updated_at", ""),
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", default=os.getenv("STATE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state"), type=Path)
    parser.add_argument("--workspace-root", default=os.getenv("WORKSPACE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812"), type=Path)
    parser.add_argument("--case-manifest", default=os.getenv("CASE_MANIFEST", "/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812/manifests/cases_103_manifest.csv"), type=Path)
    parser.add_argument("--base-manifest", default=os.getenv("BASE_MANIFEST", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"), type=Path)
    parser.add_argument("--python", default=os.getenv("PYTHON", "/home/xhan74/envs/medical_agent/bin/python"), type=Path)
    parser.add_argument("--checkpoint-root", default=os.getenv("CHECKPOINT_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints"), type=Path)
    parser.add_argument("--nnunet-predict-executable", default=os.getenv("NNUNETV2_PREDICT_EXECUTABLE", "/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict"), type=Path)
    parser.add_argument("--unest-python-executable", default=os.getenv("UNEST_PYTHON_EXECUTABLE", "/home/xhan74/envs/medical_agent_train_py311/bin/python"), type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--gpu-target-workers", default=int(os.getenv("GPU_TARGET_WORKERS", "0")), type=int)
    parser.add_argument("--gpu-overrequest-workers", default=int(os.getenv("GPU_OVERREQUEST_WORKERS", "0")), type=int)
    parser.add_argument("--gpu-profile-specs", default=os.getenv("GPU_PROFILE_SPECS", "auto"))
    parser.add_argument("--poll-sec", default=int(os.getenv("ROUND1_ORCH_POLL_SEC", "60")), type=int)
    parser.add_argument("--expected-git-commit", default=os.getenv("EXPECTED_GIT_COMMIT", _git_commit()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended Task2 103-case Round1 orchestrator.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    submit_p = sub.add_parser("submit-controller")
    add_common(submit_p)
    submit_p.add_argument("--controller-partition", default=os.getenv("CONTROLLER_PARTITION", "cpu"))
    submit_p.add_argument("--controller-cpus", default=int(os.getenv("CONTROLLER_CPUS", "2")), type=int)
    submit_p.add_argument("--controller-mem", default=os.getenv("CONTROLLER_MEM", "8G"))
    submit_p.add_argument("--controller-time", default=os.getenv("CONTROLLER_TIME", "48:00:00"))
    submit_p.add_argument("--labelcritic-partition", default=os.getenv("LABELCRITIC_PARTITION", "gpuh100"))
    submit_p.add_argument("--labelcritic-gres", default=os.getenv("LABELCRITIC_GRES", "gpu:H100:2"))
    submit_p.add_argument("--labelcritic-tensor-parallel-size", default=int(os.getenv("LABELCRITIC_TENSOR_PARALLEL_SIZE", "2")), type=int)
    submit_p.add_argument("--labelcritic-port", default=int(os.getenv("LABELCRITIC_PORT", "8000")), type=int)
    submit_p.add_argument("--skip-sbatch-test-only", action="store_true")
    submit_p.add_argument("--run-static-tests", default=os.getenv("RUN_STATIC_PREFLIGHT_TESTS", "1").lower() not in {"0", "false", "no"}, action=argparse.BooleanOptionalAction)
    submit_p.add_argument("--static-tests-timeout-sec", default=int(os.getenv("STATIC_PREFLIGHT_TESTS_TIMEOUT_SEC", "900")), type=int)
    submit_p.add_argument("--retry-failed", default=os.getenv("RETRY_FAILED", "1").lower() not in {"0", "false", "no"}, action=argparse.BooleanOptionalAction)
    submit_p.add_argument("--new-attempt", default=os.getenv("NEW_ATTEMPT", "0").lower() in {"1", "true", "yes"}, action=argparse.BooleanOptionalAction)
    controller_p = sub.add_parser("controller")
    add_common(controller_p)
    status_p = sub.add_parser("status")
    status_p.add_argument("--state-root", default=os.getenv("STATE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state"), type=Path)
    args = ap.parse_args()
    if args.cmd == "submit-controller":
        print(json.dumps(submit_controller(args), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "controller":
        return controller(args)
    if args.cmd == "status":
        print(json.dumps(status(args), indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
