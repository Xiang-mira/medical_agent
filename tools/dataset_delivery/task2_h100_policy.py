from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


LABELCRITIC_JOB_NAME = "labelcritic_72b_service"
LABELCRITIC_H100_RESERVING_STATES = {
    "SUBMITTED",
    "PENDING",
    "CONFIGURING",
    "COMPLETING",
    "STARTING",
    "RUNNING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}
LABELCRITIC_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "OOM",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
}
FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def parse_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def job_id_from_labelcritic_state(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("job_id", "labelcritic_job_id"):
            if str(value.get(key) or "").strip():
                return str(value[key]).strip()
        for key in ("service", "labelcritic", "validation", "record"):
            nested = job_id_from_labelcritic_state(value.get(key))
            if nested:
                return nested
    return ""


def labelcritic_state_from_payload(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("state", "labelcritic_job_state", "service_state"):
        if str(value.get(key) or "").strip():
            return str(value[key]).strip().upper()
    validation = value.get("validation")
    if isinstance(validation, dict):
        record = validation.get("record")
        if isinstance(record, dict) and str(record.get("state") or "").strip():
            return str(record["state"]).strip().upper()
    for key in ("record", "job", "service", "labelcritic"):
        nested = labelcritic_state_from_payload(value.get(key))
        if nested:
            return nested
    status = str(value.get("status") or "").strip().upper()
    if status in {"SUBMITTED", "PENDING", "STARTING", "RUNNING"}:
        return status
    if status in {"REUSED_ACTIVE_JOB", "REUSED_HEALTHY"}:
        return "RUNNING" if status == "REUSED_HEALTHY" else ""
    return ""


def _service_root(state_root: Path | None) -> Path | None:
    env_root = str(os.getenv("LABELCRITIC_SERVICE_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser()
    if state_root is None:
        return None
    return state_root / "labelcritic_72b_service"


def _load_runtime_labelcritic_payloads(state_root: Path | None) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    if state_root is not None:
        state_path = state_root / "round1_orchestrated" / "state.json"
        state = _read_json(state_path, {})
        if isinstance(state, dict) and isinstance(state.get("labelcritic"), dict):
            payloads.append(("round1_state", state["labelcritic"]))
    service_root = _service_root(state_root)
    if service_root is not None:
        service_state = _read_json(service_root / "service_state.json", {})
        if isinstance(service_state, dict) and service_state:
            payloads.append(("labelcritic_service_state", service_state))
        job_file = service_root / "job_id.txt"
        if job_file.exists():
            job_id = job_file.read_text(encoding="utf-8").strip()
            if job_id:
                payloads.append(("labelcritic_service_job_file", {"job_id": job_id}))
    env_job_id = str(os.getenv("LABELCRITIC_JOB_ID") or "").strip()
    if env_job_id:
        payloads.append(("LABELCRITIC_JOB_ID", {"job_id": env_job_id}))
    return payloads


def _current_user() -> str:
    return os.getenv("USER") or os.getenv("LOGNAME") or ""


def _run(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "return_code": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "command": command,
        }
    except Exception as exc:
        return {
            "ok": False,
            "return_code": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "command": command,
        }


def _normalize_query_id(job_id: str) -> str:
    raw = str(job_id or "").strip()
    if not raw:
        return ""
    concrete = re.match(r"^(?P<array>\d+)_(?P<task>\d+)$", raw)
    if concrete:
        return concrete.group("array")
    compressed = re.match(r"^(?P<array>\d+)_\[.+\]$", raw)
    if compressed:
        return compressed.group("array")
    return raw if raw.isdigit() else ""


def default_slurm_job_state(job_id: str) -> dict[str, Any]:
    query_id = _normalize_query_id(job_id)
    if not query_id:
        return {"state": "UNKNOWN", "job_id": str(job_id or ""), "source": "none"}
    result = _run(["squeue", "-h", "-j", query_id, "-o", "%i|%T|%u|%j"])
    if result["ok"] and result["stdout"].strip():
        parts = result["stdout"].splitlines()[0].split("|")
        return {
            "state": parts[1].strip().upper() if len(parts) > 1 else "UNKNOWN",
            "job_id": str(job_id or query_id),
            "display_id": parts[0].strip() if parts else query_id,
            "user": parts[2].strip() if len(parts) > 2 else "",
            "name": parts[3].strip() if len(parts) > 3 else "",
            "source": "squeue",
        }
    return {"state": "UNKNOWN", "job_id": str(job_id or query_id), "source": "unknown", "squeue": result}


def default_find_labelcritic_job_by_name() -> dict[str, Any]:
    command = ["squeue", "-h"]
    user = _current_user()
    if user:
        command.extend(["-u", user])
    command.extend(["-n", LABELCRITIC_JOB_NAME, "-o", "%i|%T|%u|%j"])
    result = _run(command)
    if not result["ok"] or not result["stdout"].strip():
        return {"status": "NONE", "source": "slurm_name_discovery", "query": result}
    candidates: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        state = parts[1].strip().upper()
        user_value = parts[2].strip() if len(parts) > 2 else ""
        name = parts[3].strip() if len(parts) > 3 else LABELCRITIC_JOB_NAME
        if state not in LABELCRITIC_H100_RESERVING_STATES:
            continue
        if user and user_value and user_value != user:
            continue
        if name != LABELCRITIC_JOB_NAME:
            continue
        candidates.append(
            {
                "job_id": parts[0].strip(),
                "state": state,
                "user": user_value,
                "name": name,
                "source": "slurm_name_discovery",
            }
        )
    if not candidates:
        return {"status": "NONE", "source": "slurm_name_discovery", "query": result}
    running = [job for job in candidates if job.get("state") == "RUNNING"]
    pool = running or candidates
    selected = sorted(pool, key=lambda job: (int(str(job["job_id"])) if str(job["job_id"]).isdigit() else 10**18, str(job["job_id"])))[0]
    return {"status": "SELECTED", **selected, "candidates": candidates}


def _policy_from_json_env() -> dict[str, Any]:
    raw = str(os.getenv("TASK2_TEACHER_H100_POLICY_JSON") or "").strip()
    if not raw:
        return {}
    try:
        doc = json.loads(raw)
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _coalesce_explicit_bool(*values: Any) -> bool | None:
    for value in values:
        parsed = parse_bool(value, None)
        if parsed is not None:
            return parsed
    return None


def resolve_teacher_h100_policy(
    *,
    state_root: Path | None = None,
    labelcritic: dict[str, Any] | None = None,
    allow_h100_teacher_overflow: bool | None = None,
    labelcritic_required: bool | None = None,
    labelcritic_job_id: str | None = None,
    labelcritic_job_state: str | None = None,
    labelcritic_h100_reserved: bool | None = None,
    source: str = "",
    slurm_job_state_fn: Callable[[str], dict[str, Any]] | None = None,
    find_labelcritic_job_fn: Callable[[], str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    env_policy = _policy_from_json_env()
    allow = _coalesce_explicit_bool(
        allow_h100_teacher_overflow,
        env_policy.get("allow_h100_teacher_overflow"),
        os.getenv("TASK2_ALLOW_H100_TEACHER_OVERFLOW"),
    )
    if allow is None:
        allow = True
    required = _coalesce_explicit_bool(
        labelcritic_required,
        env_policy.get("labelcritic_required"),
        os.getenv("TASK2_LABELCRITIC_REQUIRED"),
    )
    if required is None:
        required = True

    explicit_reserved = _coalesce_explicit_bool(
        labelcritic_h100_reserved,
        env_policy.get("labelcritic_h100_reserved"),
        os.getenv("TASK2_LABELCRITIC_H100_RESERVED"),
    )
    job_id = str(labelcritic_job_id or env_policy.get("labelcritic_job_id") or os.getenv("TASK2_LABELCRITIC_JOB_ID") or "").strip()
    job_state = str(labelcritic_job_state or env_policy.get("labelcritic_job_state") or os.getenv("TASK2_LABELCRITIC_JOB_STATE") or "").strip().upper()
    sources: list[str] = [source] if source else []
    job_found = False
    discovery_details: dict[str, Any] = {}

    if job_id or job_state or explicit_reserved is not None:
        sources.append("explicit_policy")
    if not job_id and env_policy:
        job_id = str(env_policy.get("labelcritic_job_id") or "").strip()
    if not job_state and env_policy:
        job_state = str(env_policy.get("labelcritic_job_state") or "").strip().upper()

    if labelcritic:
        job_id = job_id or job_id_from_labelcritic_state(labelcritic)
        job_state = job_state or labelcritic_state_from_payload(labelcritic)
        sources.append("labelcritic_payload")

    for payload_source, payload in _load_runtime_labelcritic_payloads(state_root):
        if not job_id:
            job_id = job_id_from_labelcritic_state(payload)
        if not job_state:
            job_state = labelcritic_state_from_payload(payload)
        if job_id or job_state:
            sources.append(payload_source)

    slurm_fn = slurm_job_state_fn or default_slurm_job_state
    if job_id and (not job_state or job_state == "UNKNOWN"):
        state_doc = slurm_fn(job_id)
        discovery_details["job_state_lookup"] = state_doc
        discovered = str(state_doc.get("state") or "").strip().upper()
        if discovered:
            job_state = discovered
            sources.append(str(state_doc.get("source") or "slurm_job_state"))
    if job_id and job_state and job_state != "UNKNOWN":
        job_found = True

    if not job_id and not job_state:
        finder = find_labelcritic_job_fn or default_find_labelcritic_job_by_name
        found = finder()
        if isinstance(found, str):
            found = {"status": "SELECTED", "job_id": found} if found else {"status": "NONE"}
        discovery_details["name_discovery"] = found
        if isinstance(found, dict) and str(found.get("status") or "") == "SELECTED":
            job_id = str(found.get("job_id") or "").strip()
            job_state = str(found.get("state") or "").strip().upper()
            job_found = bool(job_id or job_state)
            sources.append(str(found.get("source") or "slurm_name_discovery"))

    if job_id and (not job_state or job_state == "UNKNOWN"):
        state_doc = slurm_fn(job_id)
        discovery_details["post_discovery_job_state_lookup"] = state_doc
        discovered = str(state_doc.get("state") or "").strip().upper()
        if discovered:
            job_state = discovered
            sources.append(str(state_doc.get("source") or "slurm_job_state"))
    if job_id and job_state and job_state != "UNKNOWN":
        job_found = True

    if explicit_reserved is not None:
        reserved = bool(explicit_reserved)
        reason = "explicit_labelcritic_h100_reserved" if reserved else "explicit_labelcritic_h100_not_reserved"
    elif not bool(required):
        reserved = False
        reason = "labelcritic_not_required"
    elif job_state in LABELCRITIC_H100_RESERVING_STATES:
        reserved = True
        reason = f"labelcritic_job_state:{job_state}"
    elif bool(required):
        reserved = True
        if job_state in LABELCRITIC_TERMINAL_STATES:
            reason = f"labelcritic_required_job_terminal:{job_state}"
        else:
            reason = "labelcritic_required_without_active_job"

    effective = bool(allow) and not bool(reserved)
    return {
        "allow_h100_teacher_overflow": bool(allow),
        "labelcritic_required": bool(required),
        "labelcritic_job_found": bool(job_found),
        "labelcritic_job_id": job_id,
        "labelcritic_job_state": job_state or "UNKNOWN",
        "labelcritic_h100_reserved": bool(reserved),
        "effective_teacher_h100_enabled": bool(effective),
        "teacher_h100_deferred_for_labelcritic": bool(allow) and bool(reserved),
        "reservation_reason": reason,
        "source": "+".join(dict.fromkeys(filter(None, sources))) or "default_policy",
        "discovery": discovery_details,
    }
