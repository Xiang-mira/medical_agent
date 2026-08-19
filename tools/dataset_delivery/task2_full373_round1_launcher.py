#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "agent-harness"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry  # noqa: E402
from cli_anything.medai.core.continual_learning import TRAINING_CONTRACT_VERSION, canonicalize_training_record  # noqa: E402
from cli_anything.medai.core.multimodel_loop import _select_candidate  # noqa: E402
from cli_anything.medai.core.target_space import validate_formal_373_target_space  # noqa: E402
from tools.dataset_delivery.delivery_lib import NIFTI_SUFFIX, read_csv_rows, sha256_file_if_exists, utc_now, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_binary_mask_nifti_from_source  # noqa: E402
from tools.dataset_delivery.slurm_reliability import CANDIDATE_TASK_V1  # noqa: E402


FULL373_GROUP = "full373"
FULL373_ROOT_NAME = "full_373_multiteacher_round1"
FORMAL_TOTALSEGMENTATOR_EXECUTABLE = "/home/xhan74/envs/totalsegmentator_py310/bin/TotalSegmentator"
RESOURCE_POLICY_VERSION = "teacher_resource_policy_v2_shadow"
STANDARD_HOST_MEMORY_TIER = "STANDARD64"
LOWMEM_HOST_MEMORY_TIER = "LOWMEM"
HIGHMEM_96_TIER = "HOSTMEM96"
HIGHMEM_128_TIER = "HOSTMEM128"
TERMINAL_CANDIDATE_STATES = {
    "SUCCESS",
    "ABSENT",
    "OUT_OF_FOV",
    "COMPLETED_NO_NONZERO",
    "FAILED_FINAL",
}
NON_TERMINAL_CANDIDATE_STATES = {
    "READY",
    "CLAIMED",
    "QUEUED",
    "RUNNING",
    "RETRY_PENDING",
    "BACKPRESSURED",
}
VALID_TERMINAL_TARGET_STATES = {
    "SELECTED",
    "ABSENT",
    "VALID_SINGLE_TEACHER_ACCEPTED",
    "SELECTED_PSEUDO_LABEL",
    "ABSENT_NEGATIVE",
    "NEGATIVE_ABSENT",
    "WITHHELD_UNCERTAIN",
    "UNRESOLVED_REVIEW",
    "PARTIAL_FOV",
}
NON_TERMINAL_TARGET_STATES = {
    "READY",
    "CLAIMED",
    "RUNNING",
    "WAITING_FOR_CANDIDATES",
    "CANDIDATES_READY",
    "WAITING_FOR_LABELCRITIC",
    "LABELCRITIC_RUNNING",
    "RETRY_PENDING",
}


def _norm(text: str) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _scientific_candidate_identity(case_id: str, target: str, teacher: str) -> str:
    return "|".join([str(case_id), _norm(str(target)), str(teacher)])


def _resource_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "queues" / "resource_policy"
    return {
        "root": root,
        "telemetry": root / "resource_telemetry.jsonl",
        "lowmem": root / "lowmem_qualifications.json",
        "short": root / "short_qualifications.json",
        "revocations": root / "qualification_revocations.jsonl",
    }


def _resource_policy_mode() -> str:
    return os.getenv("RESOURCE_ROUTING_MODE", "shadow").strip().lower() or "shadow"


def _resource_evidence_key(row_or_state: dict[str, Any]) -> str:
    teacher = str(row_or_state.get("teacher") or row_or_state.get("model") or "")
    checkpoint = str(row_or_state.get("checkpoint_path") or row_or_state.get("checkpoint") or "")
    mode = str(row_or_state.get("inference_mode") or os.getenv("MEDAI_TEACHER_INFERENCE_MODE", "hierarchical_roi"))
    return "|".join([teacher, _sha(checkpoint or teacher, 12), mode, RESOURCE_POLICY_VERSION])


def _default_resource_demand(row_or_state: dict[str, Any]) -> dict[str, Any]:
    existing = dict(row_or_state.get("resource_demand") or {})
    host_tier = str(existing.get("host_memory_tier") or STANDARD_HOST_MEMORY_TIER)
    runtime_tier = str(existing.get("runtime_tier") or "STANDARD")
    return {
        "resource_policy_version": RESOURCE_POLICY_VERSION,
        "gpu_class_requirement": str(existing.get("gpu_class_requirement") or "GPU_INFERENCE_COMPATIBLE"),
        "host_memory_tier": host_tier,
        "runtime_tier": runtime_tier,
        "cpu_requirement": int(existing.get("cpu_requirement") or 8),
        "confidence": str(existing.get("confidence") or "UNKNOWN_DEFAULT_STANDARD"),
        "evidence_key": str(existing.get("evidence_key") or _resource_evidence_key(row_or_state)),
        "memory_escalation_level": int(existing.get("memory_escalation_level") or 0),
        "runtime_escalation_level": int(existing.get("runtime_escalation_level") or 0),
        "short_eligible": bool(existing.get("short_eligible", False)),
        "resource_blocked": bool(existing.get("resource_blocked", False)),
        "last_resource_retry_reason": str(existing.get("last_resource_retry_reason") or ""),
    }


def _read_resource_qualification(path: Path) -> dict[str, Any]:
    doc = _read_json(path, {})
    return doc if isinstance(doc, dict) else {}


def _qualified(output_root: Path, kind: str, evidence_key: str) -> bool:
    paths = _resource_paths(output_root)
    path = paths["lowmem"] if kind == "lowmem" else paths["short"]
    doc = _read_resource_qualification(path)
    item = doc.get(evidence_key)
    return bool(isinstance(item, dict) and item.get("status") == "QUALIFIED")


def _revoke_qualification(output_root: Path, *, kind: str, evidence_key: str, reason: str, candidate_id: str) -> None:
    paths = _resource_paths(output_root)
    path = paths["lowmem"] if kind == "lowmem" else paths["short"]
    doc = _read_resource_qualification(path)
    item = dict(doc.get(evidence_key) or {})
    item.update({"status": "REVOKED", "revoked_at": utc_now(), "qualification_revoked_reason": reason, "candidate_id": candidate_id})
    doc[evidence_key] = item
    atomic_write_json(path, doc)
    _append_event(output_root, {"event": "resource_qualification_revoked", "kind": kind, "evidence_key": evidence_key, "reason": reason, "candidate_id": candidate_id})
    paths["revocations"].parent.mkdir(parents=True, exist_ok=True)
    with paths["revocations"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), "kind": kind, "evidence_key": evidence_key, "reason": reason, "candidate_id": candidate_id}, ensure_ascii=False) + "\n")


def _worker_capability(profile: str, resource_class: str = "") -> dict[str, Any]:
    raw = f"{profile} {resource_class}".lower()
    short = "interactive" in raw or "short" in raw
    if "lowmem" in raw:
        memory = [LOWMEM_HOST_MEMORY_TIER]
    elif "highmem_128" in raw or "128" in raw:
        memory = [HIGHMEM_128_TIER]
    elif "highmem" in raw or "96" in raw:
        memory = [HIGHMEM_96_TIER]
    else:
        memory = [STANDARD_HOST_MEMORY_TIER, LOWMEM_HOST_MEMORY_TIER]
    return {
        "profile": profile,
        "resource_class": resource_class,
        "short_worker": short,
        "allowed_memory_classes": memory,
        "allowed_runtime_classes": ["SHORT"] if short else ["STANDARD", "SHORT", "LONG"],
        "walltime_sec": int(os.getenv("TASK2_WORKER_WALLTIME_SEC", "0") or 0),
    }


def _candidate_matches_worker(output_root: Path, state: dict[str, Any], row: dict[str, Any], *, profile: str, resource_class: str) -> tuple[bool, str, dict[str, Any]]:
    demand = _default_resource_demand({**row, **state})
    capability = _worker_capability(profile, resource_class)
    if demand.get("resource_blocked"):
        return False, "RESOURCE_BLOCKED", {"resource_demand": demand, "worker_capability": capability}
    memory_tier = str(demand.get("host_memory_tier") or STANDARD_HOST_MEMORY_TIER)
    if memory_tier == LOWMEM_HOST_MEMORY_TIER and not _qualified(output_root, "lowmem", str(demand["evidence_key"])):
        memory_tier = STANDARD_HOST_MEMORY_TIER
    if memory_tier not in capability["allowed_memory_classes"]:
        return False, f"memory_tier_not_allowed:{memory_tier}", {"resource_demand": demand, "worker_capability": capability}
    if capability["short_worker"]:
        if not bool(demand.get("short_eligible")) or not _qualified(output_root, "short", str(demand["evidence_key"])):
            return False, "short_worker_requires_short_qualified_candidate", {"resource_demand": demand, "worker_capability": capability}
    return True, "MATCH", {"resource_demand": demand, "worker_capability": capability}


def classify_resource_failure(*, return_code: int, stdout: str = "", stderr: str = "", slurm_state: str = "", profile: str = "") -> str:
    text = "\n".join([str(stdout or ""), str(stderr or ""), str(slurm_state or ""), str(profile or "")]).lower()
    if "out_of_memory" in text or "oom" in text or "killed" in text or int(return_code) in {125, 137}:
        return "CPU_OOM"
    if "timeout" in text or "time limit" in text or "timed out" in text or str(slurm_state).upper() == "TIMEOUT":
        return "WALLTIME_EXCEEDED"
    if "node_fail" in text or "node fail" in text:
        return "NODE_FAIL"
    if "preempt" in text:
        return "PREEMPTED"
    return ""


def _apply_resource_failure(output_root: Path, state: dict[str, Any], *, failure_class: str, profile: str) -> dict[str, Any]:
    demand = _default_resource_demand(state)
    original_tier = str(demand.get("host_memory_tier") or STANDARD_HOST_MEMORY_TIER)
    updates = {"status": "RETRY_PENDING", "resource_retry_reason": failure_class, "failure_reason": failure_class}
    candidate_id = str(state.get("candidate_id") or "")
    if failure_class == "CPU_OOM":
        if "lowmem" in str(profile).lower() or original_tier == LOWMEM_HOST_MEMORY_TIER:
            _revoke_qualification(output_root, kind="lowmem", evidence_key=str(demand["evidence_key"]), reason="LOWMEM_OOM", candidate_id=candidate_id)
            demand.update({"host_memory_tier": STANDARD_HOST_MEMORY_TIER, "memory_escalation_level": 0, "confidence": "LOWMEM_REVOKED_AFTER_OOM"})
        elif original_tier == HIGHMEM_96_TIER:
            demand.update({"host_memory_tier": HIGHMEM_128_TIER, "memory_escalation_level": 2, "confidence": "OOM_AT_96G"})
        elif original_tier == HIGHMEM_128_TIER:
            demand.update({"host_memory_tier": HIGHMEM_128_TIER, "resource_blocked": True, "memory_escalation_level": 3, "confidence": "RESOURCE_BLOCKED_HIGHMEM"})
            updates["resource_blocked_reason"] = "RESOURCE_BLOCKED_HIGHMEM"
        else:
            demand.update({"host_memory_tier": HIGHMEM_96_TIER, "memory_escalation_level": 1, "confidence": "OOM_AT_64G"})
    elif failure_class == "WALLTIME_EXCEEDED":
        if _worker_capability(profile).get("short_worker"):
            _revoke_qualification(output_root, kind="short", evidence_key=str(demand["evidence_key"]), reason="SHORT_TIMEOUT", candidate_id=candidate_id)
            demand.update({"short_eligible": False, "runtime_tier": "STANDARD", "runtime_escalation_level": int(demand.get("runtime_escalation_level") or 0) + 1, "confidence": "SHORT_REVOKED_AFTER_TIMEOUT"})
        else:
            demand.update({"runtime_tier": "LONG", "runtime_escalation_level": int(demand.get("runtime_escalation_level") or 0) + 1, "confidence": "NORMAL_WALLTIME_TIMEOUT"})
    else:
        demand.update({"confidence": f"RETRYABLE_RESOURCE_FAILURE:{failure_class}"})
    updates["resource_demand"] = demand
    return {**state, **updates, "updated_at": utc_now()}


def _rss_kb() -> int:
    try:
        return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except Exception:
        return 0


def _ct_geometry(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    try:
        import nibabel as nib

        img = nib.load(path_text)
        shape = tuple(int(v) for v in img.shape[:3])
        zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
        return {"input_shape": list(shape), "input_voxel_count": int(shape[0] * shape[1] * shape[2]), "spacing": list(zooms)}
    except Exception as exc:
        return {"geometry_status": "UNAVAILABLE", "geometry_failure_reason": f"{type(exc).__name__}: {exc}"}


def append_resource_telemetry(output_root: Path, payload: dict[str, Any]) -> None:
    paths = _resource_paths(output_root)
    paths["telemetry"].parent.mkdir(parents=True, exist_ok=True)
    with paths["telemetry"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **payload}, ensure_ascii=False, default=str) + "\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_targets(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    targets = [str(item).strip() for item in doc.get("target_organs", []) if str(item).strip()]
    if not targets:
        raise ValueError(f"No target_organs found in {path}")
    return targets


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _queue_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "queues"
    return {
        "root": root,
        "candidate_states": root / "candidate_states",
        "candidate_claims": root / "candidate_claims",
        "case_target_states": root / "case_target_states",
        "case_target_claims": root / "case_target_claims",
        "selection_results": root / "selection_results",
        "worker_active": root / "worker_active",
        "worker_crashes": root / "worker_crashes",
        "events": root / "events.jsonl",
        "telemetry": root / "telemetry.json",
        "candidate_seed_complete": root / "candidate_seed_complete.json",
    }


def _state_path(root: Path, case_id: str, target: str, teacher: str | None = None, *, kind: str) -> Path:
    safe_case = re_safe(case_id)
    safe_target = re_safe(target)
    if kind == "candidate":
        return _queue_paths(root)["candidate_states"] / safe_case / safe_target / f"{re_safe(teacher or '')}.json"
    if kind == "target":
        return _queue_paths(root)["case_target_states"] / safe_case / f"{safe_target}.json"
    if kind == "selection":
        return _queue_paths(root)["selection_results"] / safe_case / f"{safe_target}.json"
    raise ValueError(f"unknown state kind: {kind}")


def re_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_") or "na"


def _append_event(output_root: Path, payload: dict[str, Any]) -> None:
    path = _queue_paths(output_root)["events"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **payload}, ensure_ascii=False) + "\n")


def _tail_file(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _worker_runtime_context(worker_id: str) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "node": os.getenv("SLURMD_NODENAME", os.getenv("HOSTNAME", "")),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID", ""),
        "slurm_job_partition": os.getenv("SLURM_JOB_PARTITION", ""),
        "slurm_job_gpus": os.getenv("SLURM_JOB_GPUS", ""),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE", ""),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK", ""),
    }


def _worker_active_path(output_root: Path, worker_id: str) -> Path:
    return _queue_paths(output_root)["worker_active"] / f"{re_safe(worker_id)}.json"


def _set_worker_active(output_root: Path, *, worker_id: str, payload: dict[str, Any]) -> None:
    atomic_write_json(_worker_active_path(output_root, worker_id), {"updated_at": utc_now(), **payload})


def _clear_worker_active(output_root: Path, worker_id: str) -> None:
    try:
        _worker_active_path(output_root, worker_id).unlink()
    except FileNotFoundError:
        pass


def record_worker_crash(
    output_root: Path,
    *,
    worker_id: str,
    exc: BaseException,
    active_candidate: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = utc_now()
    record = {
        "schema_version": "task2_worker_crash_v1",
        "timestamp": timestamp,
        "scientific_run_id": _scientific_run_id(output_root),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "active_candidate": active_candidate or _read_json(_worker_active_path(output_root, worker_id), {}),
        **_worker_runtime_context(worker_id),
        **(extra or {}),
    }
    path = _queue_paths(output_root)["worker_crashes"] / f"{re_safe(worker_id)}_{int(time.time())}.json"
    atomic_write_json(path, record)
    _append_event(output_root, {"event": "worker_crash", "worker_id": worker_id, "crash_path": str(path), "exception_type": type(exc).__name__})
    return {"status": "RECORDED", "path": str(path), "record": record}


def _claim_path(output_root: Path, *, claim_kind: str, claim_key: str) -> Path:
    if claim_kind == "candidate":
        return _queue_paths(output_root)["candidate_claims"] / f"{re_safe(claim_key)}.json"
    if claim_kind == "case_target":
        return _queue_paths(output_root)["case_target_claims"] / f"{re_safe(claim_key)}.json"
    raise ValueError(f"unknown claim kind: {claim_kind}")


def _claim_expired(existing: dict[str, Any], *, now: float, lease_sec: int) -> bool:
    try:
        heartbeat_time = float(existing.get("heartbeat_time") or existing.get("created_time") or 0.0)
    except Exception:
        heartbeat_time = 0.0
    ttl = int(existing.get("lease_sec") or lease_sec or 0)
    if heartbeat_time <= 0 or ttl <= 0:
        return True
    return now - heartbeat_time >= ttl


def heartbeat_claim(output_root: Path, *, claim_kind: str, claim_key: str, worker_id: str) -> dict[str, Any]:
    path = _claim_path(output_root, claim_kind=claim_kind, claim_key=claim_key)
    if not path.exists():
        return {"status": "MISSING", "claim_path": str(path)}
    existing = _read_json(path, {})
    if str(existing.get("worker_id") or "") != str(worker_id or ""):
        return {"status": "NOT_OWNER", "claim_path": str(path), "claim": existing}
    now = time.time()
    existing["heartbeat_at"] = utc_now()
    existing["heartbeat_time"] = now
    existing["lease_expiry"] = now + int(existing.get("lease_sec") or 0)
    atomic_write_json(path, existing)
    return {"status": "HEARTBEAT", "claim_path": str(path), "claim": existing}


def claim_work(
    output_root: Path,
    *,
    claim_kind: str,
    claim_key: str,
    worker_id: str,
    lease_sec: int = 7200,
    execution_id: str = "",
    profile: str = "",
    resource_class: str = "",
) -> dict[str, Any]:
    path = _claim_path(output_root, claim_kind=claim_kind, claim_key=claim_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if path.exists():
        existing = _read_json(path, {})
        if not _claim_expired(existing, now=now, lease_sec=lease_sec):
            return {"status": "BUSY", "claim_path": str(path), "claim": existing}
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    execution = str(execution_id or f"{re_safe(worker_id)}_{_sha(str(now), 12)}")
    payload = {
        "status": "CLAIMED",
        "claim_kind": claim_kind,
        "claim_key": claim_key,
        "worker_id": worker_id,
        "claim_id": execution,
        "execution_id": execution,
        "profile": str(profile or ""),
        "resource_class": str(resource_class or ""),
        "claimed_at": utc_now(),
        "created_time": now,
        "heartbeat_at": utc_now(),
        "heartbeat_time": now,
        "lease_sec": int(lease_sec),
        "lease_expiry": now + int(lease_sec),
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {"status": "BUSY", "claim_path": str(path), "claim": _read_json(path, {})}
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {"status": "CLAIMED", "claim_path": str(path), "claim": payload}


def release_claim(output_root: Path, *, claim_kind: str, claim_key: str) -> None:
    try:
        _claim_path(output_root, claim_kind=claim_kind, claim_key=claim_key).unlink()
    except FileNotFoundError:
        pass


def _candidate_id_from_row(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_id") or "")
    target = _norm(str(row.get("target") or ""))
    teacher = str(row.get("teacher") or "")
    return str(row.get("candidate_id") or f"cand_{_sha(case_id + '|' + target + '|' + teacher)}")


def _cache_rows_from_formal_status(root: Path) -> list[dict[str, Any]]:
    path = root / "task2_formal_case_target_status.json"
    doc = _read_json(path, {})
    rows = doc.get("rows") if isinstance(doc, dict) else None
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        mask = str(row.get("mask_path") or row.get("final_mask") or "")
        if not mask or not Path(mask).is_file():
            continue
        out.append(
            {
                "source": "task2_formal_case_target_status",
                "source_manifest": str(path),
                "case_id": str(row.get("case_id") or ""),
                "target": _norm(str(row.get("target_name") or row.get("organ") or row.get("target") or "")),
                "teacher": str(row.get("selected_model") or row.get("source_model") or row.get("model") or row.get("teacher") or ""),
                "mask_path": mask,
                "validation_status": str(row.get("final_status") or row.get("validation_status") or ""),
                "provenance": row,
            }
        )
    return out


def _cache_rows_from_selection_metadata(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "annotation_versions").glob("*/selection_metadata.json")):
        doc = _read_json(path, {})
        case_id = str(doc.get("case_id") or path.parent.name)
        for row in doc.get("selected_organs") or []:
            if not isinstance(row, dict):
                continue
            mask = str(row.get("final_mask") or row.get("mask_path") or "")
            if not mask or not Path(mask).is_file():
                continue
            out.append(
                {
                    "source": "run_loop_selection_metadata",
                    "source_manifest": str(path),
                    "case_id": case_id,
                    "target": _norm(str(row.get("organ") or row.get("target") or "")),
                    "teacher": str(row.get("selected_model") or row.get("source_model") or ""),
                    "mask_path": mask,
                    "validation_status": str(row.get("selection_status") or ""),
                    "provenance": row,
                }
            )
    return out


def audit_candidate_cache(
    *,
    cache_roots: list[Path],
    case_ids: set[str],
    routes: dict[str, list[str]],
) -> dict[str, Any]:
    usable = []
    rejected = []
    route_pairs = {(case_id, target, teacher) for case_id in case_ids for target, teachers in routes.items() for teacher in teachers}
    for root in cache_roots:
        if not root or not root.exists():
            continue
        rows = [*_cache_rows_from_formal_status(root), *_cache_rows_from_selection_metadata(root)]
        for row in rows:
            key = (str(row["case_id"]), str(row["target"]), str(row["teacher"]))
            if key in route_pairs:
                usable.append({**row, "cache_status": "REUSED_VALID_CANDIDATE"})
            else:
                rejected.append({**row, "cache_status": "REJECTED_CACHE_MISMATCH"})
    return {
        "status": "success",
        "cache_roots": [str(path) for path in cache_roots],
        "reusable_candidate_count": len(usable),
        "rejected_candidate_count": len(rejected),
        "reusable_candidates": usable[:1000],
        "rejected_candidates_sample": rejected[:100],
    }


def build_full_round1_scope(
    *,
    case_manifest: Path,
    registry_path: Path,
    target_config: Path,
    output_root: Path,
    cache_roots: list[Path] | None = None,
    expected_case_count: int = 103,
    file_stem: str = "full_round1_scope",
) -> dict[str, Any]:
    cases = read_csv_rows(case_manifest)
    case_ids = [str(row.get("case_id") or row.get("id") or "").strip() for row in cases]
    targets = _load_targets(target_config)
    registry = load_registry(registry_path)
    raw_routes = candidate_models_for_organs(registry, targets)
    routes = {_norm(target): list(raw_routes.get(_norm(target), [])) for target in targets}
    disabled_routes = []
    not_ready_routes = []
    for target, teachers in routes.items():
        retained = []
        for teacher in teachers:
            entry = (registry.get("models") or {}).get(teacher) or {}
            if entry.get("enabled") is False:
                disabled_routes.append({"target": target, "teacher": teacher, "reason": "registry_enabled_false"})
                continue
            status = str(entry.get("status") or "")
            if status in {"disabled", "not_ready", "template"}:
                not_ready_routes.append({"target": target, "teacher": teacher, "status": status})
                continue
            retained.append(teacher)
        routes[target] = retained
    unroutable = [target for target in [_norm(item) for item in targets] if not routes.get(target)]
    task_rows = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("id") or "").strip()
        for target, teachers in routes.items():
            for teacher in teachers:
                entry = (registry.get("models") or {}).get(teacher) or {}
                task_rows.append(
                    {
                        "candidate_id": f"cand_{_sha(case_id + '|' + target + '|' + teacher)}",
                        "case_id": case_id,
                        "target": target,
                        "teacher": teacher,
                        "teacher_family": str(entry.get("evidence_family") or entry.get("architecture_lineage") or teacher),
                        "checkpoint": str(entry.get("checkpoint_path") or entry.get("source_code_path") or ""),
                    }
                )
    cache = audit_candidate_cache(cache_roots=cache_roots or [], case_ids=set(case_ids), routes=routes)
    pair_count = sum(len(value) for value in routes.values())
    scope = {
        "stage": "full_373_multiteacher_round1_scope",
        "status": "READY" if len(cases) == expected_case_count and len(set(case_ids)) == expected_case_count and len(targets) == 373 and not unroutable else "BLOCKED",
        "authoritative_routing_source": "cli_anything.medai.core.model_registry.candidate_models_for_organs; precedence: organ_router.route_organs(configs/organ_routing_from_xlsx.json + configs/routing_token_to_model.json), fallback registry.organ_to_models, fallback registry covered_organs/aliases",
        "case_manifest": str(case_manifest),
        "target_config": str(target_config),
        "registry_path": str(registry_path),
        "case_count": len(cases),
        "unique_case_count": len(set(case_ids)),
        "canonical_target_count": len(targets),
        "enabled_teacher_count": len({teacher for teachers in routes.values() for teacher in teachers}),
        "target_teacher_pairs": pair_count,
        "total_logical_candidate_tasks": len(cases) * pair_count,
        "expected_logical_candidate_tasks_for_103_cases": expected_case_count * pair_count,
        "single_teacher_target_count": sum(1 for teachers in routes.values() if len(teachers) == 1),
        "multi_teacher_target_count": sum(1 for teachers in routes.values() if len(teachers) > 1),
        "unroutable_target_count": len(unroutable),
        "unroutable_targets": unroutable,
        "disabled_teacher_routes": disabled_routes,
        "not_ready_teacher_routes": not_ready_routes,
        "routes": routes,
        "task_rows": task_rows,
        "cached_reusable_candidate_count": cache["reusable_candidate_count"],
        "new_inference_candidate_count": max(0, len(cases) * pair_count - int(cache["reusable_candidate_count"])),
        "cache_audit": cache,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / f"{file_stem}.json", scope)
    write_csv(
        output_root / f"{file_stem}.csv",
        task_rows,
        ["candidate_id", "case_id", "target", "teacher", "teacher_family", "checkpoint"],
    )
    return scope


def _write_case_manifest(path: Path, row: dict[str, str]) -> None:
    fieldnames = list(row.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [row], fieldnames)


def _candidate_queue_manifest(output_root: Path) -> Path:
    return _queue_paths(output_root)["root"] / "shared_ready_candidate_manifest.csv"


def resolve_totalsegmentator_executable_arg(value: str | Path | None = None) -> str:
    configured = str(value or "").strip()
    if configured:
        return configured
    for key in ("TOTAL_SEGMENTATOR_EXECUTABLE", "TOTALSEGMENTATOR_EXECUTABLE", "MEDAI_TOTALSEG_EXECUTABLE"):
        configured = str(os.getenv(key) or "").strip()
        if configured:
            return configured
    return FORMAL_TOTALSEGMENTATOR_EXECUTABLE


def _candidate_queue_rows(output_root: Path, *, task_manifest: Path | None = None) -> list[dict[str, Any]]:
    manifest = task_manifest or _candidate_queue_manifest(output_root)
    if manifest.exists():
        return read_csv_rows(manifest)
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    rows = []
    for row in scope.get("task_rows") or []:
        if isinstance(row, dict):
            rows.append(dict(row))
    return rows


def _manifest_cursor_offset(output_root: Path, *, worker_id: str, manifest_rows: list[dict[str, Any]]) -> int:
    if not manifest_rows:
        return 0
    seed = f"{worker_id}|{len(manifest_rows)}|{manifest_rows[0].get('case_id','')}|{manifest_rows[-1].get('case_id','')}"
    try:
        return int(_sha(seed, 8), 16) % len(manifest_rows)
    except Exception:
        return 0


def _manifest_identity(path: Path | None, rows: list[dict[str, Any]], *, include_hash: bool = True) -> dict[str, Any]:
    manifest_path = Path(str(path or ""))
    if manifest_path.exists():
        stat = manifest_path.stat()
        identity = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_size": int(stat.st_size),
            "manifest_mtime_ns": int(stat.st_mtime_ns),
        }
        if include_hash:
            identity["manifest_sha256"] = _sha(manifest_path.read_text(encoding="utf-8"), 24)
        return identity
    row_keys = [
        f"{row.get('case_id','')}|{_norm(str(row.get('target') or ''))}|{row.get('teacher','')}|{_candidate_id_from_row(row)}"
        for row in rows
    ]
    return {
        "manifest_path": str(manifest_path) if str(manifest_path) else "",
        "manifest_sha256": _sha("\n".join(row_keys), 24),
        "manifest_size": 0,
        "manifest_mtime_ns": 0,
    }


def _scientific_run_id(output_root: Path) -> str:
    scope = _read_json(output_root / "full_round1_submission_scope.json", {}) or _read_json(output_root / "full_round1_scope.json", {})
    return str(scope.get("run_id") or os.getenv("ROUND1_RUN_ID") or os.getenv("TASK2_SCIENTIFIC_RUN_ID") or "")


def _candidate_seed_marker_matches(output_root: Path, *, task_manifest: Path | None, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    loaded_rows = rows if rows is not None else _candidate_queue_rows(output_root, task_manifest=task_manifest)
    identity = _manifest_identity(task_manifest or _candidate_queue_manifest(output_root), loaded_rows, include_hash=False)
    marker = _read_json(_queue_paths(output_root)["candidate_seed_complete"], {})
    matches = (
        bool(marker)
        and marker.get("schema_version") == "candidate_seed_complete_v1"
        and str(marker.get("execution_schema_version") or CANDIDATE_TASK_V1) == CANDIDATE_TASK_V1
        and int(marker.get("logical_task_count") or -1) == len(loaded_rows)
        and str(marker.get("manifest_path") or "") == str(identity.get("manifest_path") or "")
        and int(marker.get("manifest_size") or -1) == int(identity.get("manifest_size") or -2)
        and int(marker.get("manifest_mtime_ns") or -1) == int(identity.get("manifest_mtime_ns") or -2)
    )
    return {"status": "MATCH" if matches else "MISSING_OR_STALE", "marker": marker, "identity": identity, "logical_task_count": len(loaded_rows)}


def _write_array_sbatch(path: Path, *, python: Path, output_root: Path, task_manifest: Path, state_root: Path | None = None) -> None:
    state_arg = f"  --state-root {shlex.quote(str(state_root))} \\\n" if state_root else ""
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=task2_full373_multiteacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output={output_root / 'slurm' / 'full373_%A_%a.out'}
#SBATCH --error={output_root / 'slurm' / 'full373_%A_%a.err'}

set -euo pipefail
unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS
export RUNTIME_NO_GIT=1
export SKIP_GIT_SYNC=1
export GIT_TERMINAL_PROMPT=0
cd {shlex.quote(str(REPO_ROOT))}
{shlex.quote(str(python))} tools/dataset_delivery/task2_full373_round1_launcher.py \\
  --execute-task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --task-manifest {shlex.quote(str(task_manifest))} \\
{state_arg}  --worker-id "${{SLURM_JOB_ID:-local}}_${{SLURM_ARRAY_TASK_ID:-0}}" \\
  --output-root {shlex.quote(str(output_root))}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def build_submission_manifest(
    *,
    case_manifest: Path,
    output_root: Path,
    registry_path: Path,
    target_config: Path,
    python: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    unest_python_executable: Path,
    totalsegmentator_executable: Path | str | None = None,
    state_root: Path | None = None,
    cache_roots: list[Path] | None = None,
    expected_case_count: int = 103,
) -> dict[str, Any]:
    input_rows = read_csv_rows(case_manifest)
    scope = build_full_round1_scope(
        case_manifest=case_manifest,
        registry_path=registry_path,
        target_config=target_config,
        output_root=output_root,
        cache_roots=cache_roots or [],
        expected_case_count=len(input_rows) if input_rows else expected_case_count,
        file_stem="full_round1_submission_scope",
    )
    if scope["status"] != "READY":
        raise RuntimeError(f"Full 373 Round1 scope blocked: {scope.get('unroutable_targets')}")
    case_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(input_rows):
        case_id = str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()
        case_by_id[case_id] = row
    rows = []
    for index, candidate in enumerate(scope.get("task_rows") or []):
        case_id = str(candidate.get("case_id") or "")
        row = case_by_id.get(case_id) or {}
        rows.append(
            {
                "task_index": index,
                "case_id": case_id,
                "target": str(candidate.get("target") or ""),
                "teacher": str(candidate.get("teacher") or ""),
                "candidate_id": str(candidate.get("candidate_id") or f"cand_{_sha(case_id + '|' + str(candidate.get('target')) + '|' + str(candidate.get('teacher')))}"),
                "ct_path": row.get("ct_path") or row.get("image_path") or "",
                "annotation_folder": row.get("annotation_folder") or row.get("original_annotation_folder") or row.get("reference_mask_dir") or "",
                "original_annotation_folder": row.get("original_annotation_folder") or row.get("annotation_folder") or row.get("reference_mask_dir") or "",
                "canonical_annotation_folder": row.get("canonical_annotation_folder") or "",
                "registry_path": str(registry_path),
                "target_config": str(target_config),
                "checkpoint_root": str(checkpoint_root),
                "nnunet_predict_executable": str(nnunet_predict_executable),
                "unest_python_executable": str(unest_python_executable),
                "totalsegmentator_executable": resolve_totalsegmentator_executable_arg(totalsegmentator_executable),
                "python": str(python),
            }
        )
    slurm_root = output_root / "slurm"
    task_manifest = _candidate_queue_manifest(output_root)
    write_csv(task_manifest, rows, list(rows[0].keys()) if rows else ["task_index", "case_id", "target", "teacher", "candidate_id"])
    sbatch = slurm_root / "full373_multiteacher_array.sbatch"
    _write_array_sbatch(sbatch, python=python, output_root=output_root, task_manifest=task_manifest, state_root=state_root)
    summary = {
        "status": "READY",
        "stage": "full_373_multiteacher_round1_submission_manifest",
        "created_at": utc_now(),
        "scope": str(output_root / "full_round1_submission_scope.json"),
        "scope_status": scope,
        "task_count": len(rows),
        "scientific_task_unit": "case_id x canonical_target x eligible_teacher",
        "scheduler_array_unit": "generic GPU worker array consuming shared READY candidate queue",
        "groups": {
            FULL373_GROUP: {
                "task_count": len(rows),
                "task_manifest": str(task_manifest),
                "sbatch_file": str(sbatch),
            }
        },
    }
    write_json(output_root / "formal_task2_submission_manifest.json", summary)
    return summary


def _selection_metadata_path(case_output: Path, case_id: str) -> Path:
    return case_output / "annotation_versions" / case_id / "selection_metadata.json"


def _candidate_from_single_teacher_run(case_output: Path, *, case_id: str, target: str, teacher: str) -> dict[str, Any]:
    meta = _read_json(_selection_metadata_path(case_output, case_id), {})
    selection_rows = list(meta.get("selection_rows") or [])
    selected_rows = list(meta.get("selected_organs") or [])
    target_rows = [row for row in selection_rows if _norm(str(row.get("organ") or row.get("target") or "")) == target]
    selected_target_rows = [row for row in selected_rows if _norm(str(row.get("organ") or row.get("target") or "")) == target]
    candidate: dict[str, Any] = {}
    for row in target_rows:
        for pred in row.get("candidate_predictions") or []:
            if str(pred.get("model") or "") == teacher:
                candidate = dict(pred)
                break
        if candidate:
            break
    selected = selected_target_rows[0] if selected_target_rows else {}
    cleaned_prediction = str(candidate.get("candidate_cleaned_prediction") or selected.get("final_mask") or "")
    raw_prediction = str(candidate.get("candidate_raw_prediction") or candidate.get("prediction") or selected.get("pre_shapekit_mask") or "")
    selected_prediction = str(selected.get("mask_path") or "")
    prediction = cleaned_prediction or raw_prediction or selected_prediction
    path = Path(str(prediction or ""))
    exists = bool(prediction and path.is_file())
    qc_status = str(candidate.get("candidate_qc_status") or selected.get("selected_candidate_qc_status") or selected.get("candidate_qc_status") or "")
    qc_flags = candidate.get("candidate_qc_flags") or selected.get("selected_candidate_qc_flags") or []
    if isinstance(qc_flags, str):
        qc_flags = [qc_flags]
    qc_failed = qc_status.lower() in {"fail", "failed", "unusable"}
    shapekit_status = str(candidate.get("candidate_shapekit_status") or selected.get("shapekit_status") or "")
    shapekit_reason = str(
        candidate.get("candidate_shapekit_reason")
        or selected.get("shapekit_reason")
        or candidate.get("postprocess_failure_reason")
        or selected.get("postprocess_failure_reason")
        or ""
    )
    shapekit_failed = (
        shapekit_status.lower() in {"postprocess_failed", "fallback_original", "failed", "error"}
        or shapekit_status.lower().startswith("warning_")
        or any(str(flag).lower() == "postprocess_failed" for flag in qc_flags)
    )
    if shapekit_failed:
        shapekit_status = "postprocess_failed"
        shapekit_reason = shapekit_reason or "ShapeKit postprocess failed; raw candidate preserved"
    target_type = str(selected.get("target_type") or "")
    if exists:
        state = "SUCCESS"
    elif target_type in {"absent_negative", "negative_absent"}:
        state = "ABSENT"
    elif target_type in {"out_of_fov", "partial_fov"}:
        state = "OUT_OF_FOV"
    else:
        state = "COMPLETED_NO_NONZERO"
    explicit_eligible = candidate.get("eligible_for_labelcritic")
    if isinstance(explicit_eligible, str):
        eligible_for_labelcritic = explicit_eligible.strip().lower() in {"1", "true", "yes"}
    else:
        eligible_for_labelcritic = bool(explicit_eligible) if explicit_eligible is not None else bool(exists and not qc_failed)
    return {
        "status": state,
        "candidate_exists": exists,
        "case_id": case_id,
        "target": target,
        "teacher": teacher,
        "model": teacher,
        "prediction": str(path) if exists else "",
        "candidate_cleaned_prediction": cleaned_prediction,
        "candidate_raw_prediction": raw_prediction,
        "candidate_id": candidate.get("candidate_id") or f"cand_{_sha(case_id + '|' + target + '|' + teacher)}",
        "candidate_qc_status": qc_status,
        "candidate_qc_score": candidate.get("candidate_qc_score") or selected.get("selected_candidate_qc_score"),
        "candidate_qc_flags": qc_flags,
        "candidate_qc": candidate.get("candidate_qc") or selected.get("selected_candidate_qc_checks") or {},
        "candidate_shapekit_status": shapekit_status,
        "candidate_shapekit_reason": shapekit_reason,
        "raw_candidate_survived_shapekit_failure": bool(exists and shapekit_failed and not Path(cleaned_prediction).is_file()),
        "eligible_for_labelcritic": eligible_for_labelcritic,
        "source_run_loop": str(case_output),
        "selection_metadata": str(_selection_metadata_path(case_output, case_id)),
        "published_at": utc_now(),
    }


def load_candidate_state(output_root: Path, *, case_id: str, target: str, teacher: str) -> dict[str, Any]:
    return _read_json(_state_path(output_root, case_id, target, teacher, kind="candidate"), {})


def publish_candidate_state(output_root: Path, state: dict[str, Any], *, recompute_target: bool = True) -> dict[str, Any]:
    case_id = str(state["case_id"])
    target = _norm(str(state["target"]))
    teacher = str(state["teacher"])
    path = _state_path(output_root, case_id, target, teacher, kind="candidate")
    atomic_write_json(path, state)
    _append_event(output_root, {"event": "candidate_state", "case_id": case_id, "target": target, "teacher": teacher, "status": state.get("status")})
    if recompute_target:
        recompute_case_target_readiness(output_root, case_id=case_id, target=target)
    return state


def _candidate_terminal(state: dict[str, Any]) -> bool:
    return str(state.get("status") or "") in TERMINAL_CANDIDATE_STATES


def _candidate_pending_like(state: dict[str, Any]) -> bool:
    return str(state.get("status") or "") in {"", "PENDING", "READY", "RETRY_PENDING", "BACKPRESSURED"}


def seed_candidate_states(output_root: Path, *, task_manifest: Path | None = None) -> dict[str, Any]:
    rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    seeded = 0
    retained = 0
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        target = _norm(str(row.get("target") or ""))
        teacher = str(row.get("teacher") or "").strip()
        if not case_id or not target or not teacher:
            continue
        existing = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
        if existing:
            retained += 1
            continue
        seeded += 1
        publish_candidate_state(
            output_root,
            {
                "status": "READY",
                "case_id": case_id,
                "target": target,
                "teacher": teacher,
                "model": teacher,
                "teacher_family": str(row.get("teacher_family") or teacher),
                "candidate_id": _candidate_id_from_row(row),
                "execution_schema_version": CANDIDATE_TASK_V1,
                "logical_task_id": f"{case_id}|{target}|{teacher}",
                "resource_policy_version": RESOURCE_POLICY_VERSION,
                "resource_demand": _default_resource_demand(row),
                "updated_at": utc_now(),
                "published_at": utc_now(),
            },
            recompute_target=False,
        )
    identity = _manifest_identity(task_manifest or _candidate_queue_manifest(output_root), rows)
    marker = {
        "schema_version": "candidate_seed_complete_v1",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "scientific_run_id": _scientific_run_id(output_root),
        "logical_task_count": len(rows),
        **identity,
        "seeded": seeded,
        "retained": retained,
        "seeded_count": seeded,
        "newly_seeded_count": seeded,
        "validated_existing_count": retained,
        "retained_terminal_count": 0,
        "completed_at": utc_now(),
    }
    atomic_write_json(_queue_paths(output_root)["candidate_seed_complete"], marker)
    return {"status": "READY", **marker}


def recover_candidate_seed_marker(output_root: Path, *, task_manifest: Path | None = None) -> dict[str, Any]:
    rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    existing = 0
    retained_terminal = 0
    missing: list[dict[str, str]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        target = _norm(str(row.get("target") or ""))
        teacher = str(row.get("teacher") or "").strip()
        if not case_id or not target or not teacher:
            missing.append({"case_id": case_id, "target": target, "teacher": teacher, "reason": "invalid_manifest_row"})
            continue
        state = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
        if not state:
            missing.append({"case_id": case_id, "target": target, "teacher": teacher, "reason": "candidate_state_missing"})
            continue
        existing += 1
        if _candidate_terminal(state):
            retained_terminal += 1
    identity = _manifest_identity(task_manifest or _candidate_queue_manifest(output_root), rows)
    if missing:
        return {
            "status": "INCOMPLETE",
            "schema_version": "candidate_seed_complete_v1",
            "execution_schema_version": CANDIDATE_TASK_V1,
            "scientific_run_id": _scientific_run_id(output_root),
            "logical_task_count": len(rows),
            **identity,
            "validated_existing_count": existing,
            "seeded_count": 0,
            "newly_seeded_count": 0,
            "retained_terminal_count": retained_terminal,
            "missing_count": len(missing),
            "missing_sample": missing[:50],
        }
    marker = {
        "schema_version": "candidate_seed_complete_v1",
        "execution_schema_version": CANDIDATE_TASK_V1,
        "scientific_run_id": _scientific_run_id(output_root),
        "logical_task_count": len(rows),
        **identity,
        "validated_existing_count": existing,
        "seeded_count": 0,
        "newly_seeded_count": 0,
        "retained_terminal_count": retained_terminal,
        "completed_at": utc_now(),
    }
    atomic_write_json(_queue_paths(output_root)["candidate_seed_complete"], marker)
    return {"status": "READY", **marker}


def _owner_job_id_from_state(state: dict[str, Any], claim_doc: dict[str, Any]) -> str:
    for source in (state, claim_doc):
        for key in ("worker_job_id", "slurm_job_id", "job_id"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
        worker_id = str(source.get("worker_id") or "").strip()
        if re.match(r"^\d+(_\d+)?$", worker_id):
            return worker_id
    return ""


def recover_stale_candidate_claims(
    output_root: Path,
    *,
    task_manifest: Path | None = None,
    lease_sec: int = 7200,
    worker_state_fn: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    recovered = []
    retained_terminal = 0
    now = time.time()
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        target = _norm(str(row.get("target") or ""))
        teacher = str(row.get("teacher") or "").strip()
        if not case_id or not target or not teacher:
            continue
        state = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
        status = str(state.get("status") or "")
        if status in TERMINAL_CANDIDATE_STATES:
            retained_terminal += 1
            continue
        if status not in {"CLAIMED", "RUNNING"}:
            continue
        candidate_id = _candidate_id_from_row(row)
        claim_path = _claim_path(output_root, claim_kind="candidate", claim_key=candidate_id)
        claim_doc = _read_json(claim_path, {}) if claim_path.exists() else {}
        recovery_reason = ""
        worker_state: dict[str, Any] = {}
        if not claim_path.exists():
            recovery_reason = "claim_missing_requeued"
        elif _claim_expired(claim_doc, now=now, lease_sec=int(claim_doc.get("lease_sec") or lease_sec)):
            recovery_reason = "lease_expired_requeued"
        elif worker_state_fn is not None:
            owner_job_id = _owner_job_id_from_state(state, claim_doc)
            if owner_job_id:
                try:
                    worker_state = worker_state_fn(owner_job_id) or {}
                except Exception as exc:
                    worker_state = {"state": "UNKNOWN", "failure_reason": f"{type(exc).__name__}: {exc}"}
                slurm_state = str(worker_state.get("state") or "").upper()
                if slurm_state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "OOM", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "COMPLETED"}:
                    recovery_reason = f"worker_slurm_terminal:{slurm_state}"
        if not recovery_reason:
            continue
        updated = {
            **state,
            "status": "RETRY_PENDING",
            "failure_reason": str(state.get("failure_reason") or recovery_reason),
            "recovery_reason": recovery_reason,
            "previous_status": status,
            "previous_worker_id": str(state.get("worker_id") or claim_doc.get("worker_id") or ""),
            "previous_execution_id": str(state.get("execution_id") or claim_doc.get("execution_id") or claim_doc.get("claim_id") or ""),
            "previous_claim": claim_doc,
            "worker_slurm_state": worker_state,
            "recovered_at": utc_now(),
            "updated_at": utc_now(),
        }
        publish_candidate_state(output_root, updated)
        recovered.append({"case_id": case_id, "target": target, "teacher": teacher, "candidate_id": candidate_id, "previous_status": status, "recovery_reason": recovery_reason})
    return {
        "status": "READY",
        "recovered_count": len(recovered),
        "recovered": recovered[:100],
        "logical_task_count": len(rows),
        "retained_terminal_count": retained_terminal,
    }


FORMAL_TARGET_CONTRACT_FAILURE = "Requested organs include non-target organs for the formal 373-organ mainline"
TOTALSEG_EXECUTABLE_FAILURE_TOKENS = ("FileNotFoundError", "No such file or directory: 'TotalSegmentator'")


def repair_recoverable_failed_candidates(
    output_root: Path,
    *,
    task_manifest: Path | None = None,
    target_config: Path | None = None,
    totalseg_executable_ready: bool = False,
    execution_attempt_id: str = "",
) -> dict[str, Any]:
    rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    repaired = []
    retained_terminal = 0
    target_config = target_config or (REPO_ROOT / "configs/student_3d_prompt_target_organs.json")
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        target = _norm(str(row.get("target") or ""))
        teacher = str(row.get("teacher") or "").strip()
        if not case_id or not target or not teacher:
            continue
        state = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
        status = str(state.get("status") or "")
        if status in {"SUCCESS", "COMPLETED_NO_NONZERO"}:
            retained_terminal += 1
        if status != "FAILED_FINAL":
            continue
        failure_text = "\n".join(str(state.get(key) or "") for key in ("failure_reason", "stderr_tail", "stdout_tail"))
        recovery_reason = ""
        if FORMAL_TARGET_CONTRACT_FAILURE in failure_text:
            validation = validate_formal_373_target_space(target_config, requested_organs=[target])
            if validation.get("status") == "success":
                recovery_reason = "formal_target_contract_fixed"
        elif all(token in failure_text for token in TOTALSEG_EXECUTABLE_FAILURE_TOKENS) and totalseg_executable_ready:
            recovery_reason = "totalsegmentator_executable_fixed"
        if not recovery_reason:
            continue
        updated = {
            **state,
            "status": "RETRY_PENDING",
            "previous_status": status,
            "recovery_reason": recovery_reason,
            "previous_failure_reason": str(state.get("failure_reason") or ""),
            "recovered_at": utc_now(),
            "execution_attempt_id": execution_attempt_id,
            "updated_at": utc_now(),
        }
        publish_candidate_state(output_root, updated, recompute_target=False)
        recompute_case_target_readiness(output_root, case_id=case_id, target=target, force=True)
        repaired.append({"case_id": case_id, "target": target, "teacher": teacher, "candidate_id": _candidate_id_from_row(row), "recovery_reason": recovery_reason})
    return {
        "status": "READY",
        "logical_task_count": len(rows),
        "repaired_count": len(repaired),
        "repaired": repaired[:100],
        "retained_success_or_completed_no_nonzero_count": retained_terminal,
    }


def claim_next_ready_candidate(
    output_root: Path,
    *,
    task_manifest: Path | None = None,
    worker_id: str,
    profile: str,
    resource_class: str = "GPU_INFERENCE_COMPATIBLE",
    lease_sec: int = 7200,
) -> dict[str, Any]:
    manifest_rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    return claim_next_ready_candidate_from_rows(
        output_root,
        manifest_rows=manifest_rows,
        worker_id=worker_id,
        profile=profile,
        resource_class=resource_class,
        lease_sec=lease_sec,
        task_manifest=task_manifest,
    )


def claim_next_ready_candidate_from_rows(
    output_root: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    worker_id: str,
    profile: str,
    resource_class: str = "GPU_INFERENCE_COMPATIBLE",
    lease_sec: int = 7200,
    task_manifest: Path | None = None,
    start_index: int = 0,
    seed_marker_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    marker_check = seed_marker_check or _candidate_seed_marker_matches(output_root, task_manifest=task_manifest, rows=manifest_rows)
    if marker_check["status"] != "MATCH":
        return {
            "status": "WAITING_FOR_CANDIDATE_SEED",
            "queue_depth": 0,
            "seed_marker": marker_check,
        }
    state_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    queue_depth = 0
    now = time.time()
    if not manifest_rows:
        return {"status": "NO_READY_CANDIDATES", "next_cursor": 0, "seed_marker": marker_check}
    start = max(0, int(start_index or 0)) % len(manifest_rows)
    indexed_rows = list(enumerate(manifest_rows))
    ordered_rows = indexed_rows[start:] + indexed_rows[:start]
    for row_index, row in ordered_rows:
        case_id = str(row.get("case_id") or "").strip()
        target = _norm(str(row.get("target") or ""))
        teacher = str(row.get("teacher") or "").strip()
        if not case_id or not target or not teacher:
            continue
        candidate_id = _candidate_id_from_row(row)
        state_key = (case_id, target, teacher)
        state = state_cache.get(state_key)
        if state is None:
            state = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
            state_cache[state_key] = state
        if _candidate_terminal(state):
            continue
        claim_path = _claim_path(output_root, claim_kind="candidate", claim_key=candidate_id)
        claim_doc = _read_json(claim_path, {}) if claim_path.exists() else {}
        state_status = str(state.get("status") or "READY")
        if state_status in {"CLAIMED", "RUNNING"} and _claim_expired(claim_doc, now=now, lease_sec=int(claim_doc.get("lease_sec") or lease_sec)):
            state = {
                **state,
                "status": "RETRY_PENDING",
                "failure_reason": str(state.get("failure_reason") or "lease_expired_requeued"),
                "updated_at": utc_now(),
            }
            publish_candidate_state(output_root, state)
        if not _candidate_pending_like(state):
            continue
        matches, reason, resource_decision = _candidate_matches_worker(
            output_root,
            state,
            row,
            profile=profile,
            resource_class=resource_class,
        )
        if not matches:
            _append_event(
                output_root,
                {
                    "event": "candidate_resource_claim_skipped",
                    "case_id": case_id,
                    "target": target,
                    "teacher": teacher,
                    "candidate_id": candidate_id,
                    "profile": profile,
                    "resource_class": resource_class,
                    "reason": reason,
                    "resource_policy_version": RESOURCE_POLICY_VERSION,
                },
            )
            continue
        queue_depth += 1
        claim = claim_work(
            output_root,
            claim_kind="candidate",
            claim_key=str(candidate_id),
            worker_id=worker_id,
            lease_sec=lease_sec,
            execution_id=f"{re_safe(worker_id)}_{_sha(str(time.time()), 12)}",
            profile=profile,
            resource_class=resource_class,
        )
        if claim["status"] != "CLAIMED":
            continue
        claimed_state = {
            **state,
            "status": "CLAIMED",
            "claim": claim["claim"],
            "worker_id": worker_id,
            "profile": profile,
            "resource_class": resource_class,
            "resource_policy_version": RESOURCE_POLICY_VERSION,
            "resource_decision": resource_decision,
            "updated_at": utc_now(),
        }
        state_cache[state_key] = claimed_state
        publish_candidate_state(output_root, claimed_state)
        return {
            "status": "CLAIMED",
            "candidate": claimed_state,
            "claim": claim,
            "row": row,
            "row_index": int(row_index),
            "next_cursor": (int(row_index) + 1) % len(manifest_rows),
        }
    if queue_depth <= 0:
        return {"status": "NO_READY_CANDIDATES", "next_cursor": start, "seed_marker": marker_check}
    return {"status": "NO_CLAIMABLE_READY_CANDIDATES", "queue_depth": queue_depth, "next_cursor": start, "seed_marker": marker_check}


def recompute_case_target_readiness(output_root: Path, *, case_id: str, target: str, force: bool = False) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_submission_scope.json", {}) or _read_json(output_root / "full_round1_scope.json", {})
    routes = scope.get("routes") or {}
    teachers = [str(item) for item in routes.get(_norm(target), [])]
    candidate_states = [load_candidate_state(output_root, case_id=case_id, target=_norm(target), teacher=teacher) for teacher in teachers]
    observed = [state for state in candidate_states if state]
    statuses = [str(state.get("status") or "READY") for state in candidate_states]
    all_terminal = bool(teachers) and len(observed) == len(teachers) and all(status in TERMINAL_CANDIDATE_STATES for status in statuses)
    if all_terminal:
        valid = [
            state for state in observed
            if state.get("status") == "SUCCESS"
            and state.get("candidate_exists")
            and state.get("prediction")
            and Path(str(state.get("prediction"))).is_file()
        ]
        status = "CANDIDATES_READY" if valid else "ABSENT"
    else:
        status = "WAITING_FOR_CANDIDATES"
    target_state = {
        "status": status,
        "case_id": case_id,
        "target": _norm(target),
        "eligible_teachers": teachers,
        "candidate_statuses": {teacher: str(state.get("status") or "READY") for teacher, state in zip(teachers, candidate_states)},
        "candidate_state_paths": [
            str(_state_path(output_root, case_id, _norm(target), teacher, kind="candidate")) for teacher in teachers
        ],
        "candidate_count": len(valid) if all_terminal else 0,
        "updated_at": utc_now(),
    }
    existing = _read_json(_state_path(output_root, case_id, _norm(target), kind="target"), {})
    if not force and str(existing.get("status") or "") in VALID_TERMINAL_TARGET_STATES:
        return existing
    atomic_write_json(_state_path(output_root, case_id, _norm(target), kind="target"), target_state)
    _append_event(output_root, {"event": "case_target_readiness", "case_id": case_id, "target": _norm(target), "status": status})
    return target_state


def _execute_candidate_row(
    row: dict[str, Any],
    output_root: Path,
    *,
    worker_id: str = "",
    state_root: Path | None = None,
    already_claimed: bool = False,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    target = _norm(str(row.get("target") or ""))
    teacher = str(row.get("teacher") or "")
    task_index = int(row.get("task_index") or 0)
    candidate_id = _candidate_id_from_row(row)
    existing = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
    if existing.get("status") in TERMINAL_CANDIDATE_STATES:
        return {"status": "REUSED_TERMINAL_CANDIDATE", "candidate": existing}
    if not already_claimed:
        claim = claim_work(output_root, claim_kind="candidate", claim_key=candidate_id, worker_id=worker_id or f"pid_{os.getpid()}")
        if claim["status"] != "CLAIMED":
            return {"status": "CLAIM_BUSY", "candidate_id": candidate_id, "claim": claim}
    else:
        claim = {"status": "CLAIMED", "claim": _read_json(_claim_path(output_root, claim_kind="candidate", claim_key=candidate_id), {})}
    case_csv = output_root / "case_manifests" / f"{case_id}.csv"
    try:
        heartbeat_claim(output_root, claim_kind="candidate", claim_key=candidate_id, worker_id=worker_id or f"pid_{os.getpid()}")
        _write_case_manifest(
            case_csv,
            {
                "case_id": case_id,
                "ct_path": str(row.get("ct_path") or ""),
                "annotation_folder": str(row.get("original_annotation_folder") or row.get("annotation_folder") or ""),
                "original_annotation_folder": str(row.get("original_annotation_folder") or row.get("annotation_folder") or ""),
                "canonical_annotation_folder": str(row.get("canonical_annotation_folder") or ""),
            },
        )
        case_output = output_root / "candidate_runs" / case_id / target / teacher
        case_output.mkdir(parents=True, exist_ok=True)
        child_stdout_path = case_output / "worker_child.stdout.log"
        child_stderr_path = case_output / "worker_child.stderr.log"
        command = [
            str(row.get("python") or sys.executable),
            "run_medai_cli.py",
            "--json",
            "run-loop",
            "--case-list", str(case_csv),
            "--models", teacher,
            "--organs", target,
            "--target-config", str(row.get("target_config") or REPO_ROOT / "configs/student_3d_prompt_target_organs.json"),
            "--registry", str(row.get("registry_path") or REPO_ROOT / "configs/model_registry.yaml"),
            "--output", str(case_output),
            "--checkpoint-root", str(row.get("checkpoint_root") or ""),
            "--nnunet-predict-executable", str(row.get("nnunet_predict_executable") or ""),
            "--unest-python-executable", str(row.get("unest_python_executable") or ""),
            "--enable-shapekit",
            "--no-enable-critic",
            "--teacher-inference-mode", os.getenv("MEDAI_TEACHER_INFERENCE_MODE", "hierarchical_roi"),
            "--roi-margin-mm", os.getenv("MEDAI_ROI_MARGIN_MM", "20"),
            "--timeout-sec", os.getenv("MEDAI_INFER_TIMEOUT_SEC", "3600"),
            "--no-use-annotation-folder-reference",
            "--log-file", str(case_output / "run_loop.log"),
        ]
        env = os.environ.copy()
        env.update({"RUNTIME_NO_GIT": "1", "SKIP_GIT_SYNC": "1", "GIT_TERMINAL_PROMPT": "0"})
        if str(row.get("totalsegmentator_executable") or "").strip():
            env["TOTAL_SEGMENTATOR_EXECUTABLE"] = str(row.get("totalsegmentator_executable"))
        if state_root:
            env["STATE_ROOT"] = str(state_root)
        for key in ("DISPLAY", "GITHUB_TOKEN", "GH_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"):
            env.pop(key, None)
        task_state_path = output_root / "task_states" / f"{candidate_id}.json"
        state = {
            "status": "RUNNING",
            "case_id": case_id,
            "target": target,
            "teacher": teacher,
            "candidate_id": candidate_id,
            "task_index": task_index,
            "worker_id": worker_id or f"pid_{os.getpid()}",
            "profile": str((claim.get("claim") or {}).get("profile") or row.get("profile") or ""),
            "resource_class": str((claim.get("claim") or {}).get("resource_class") or row.get("resource_class") or ""),
            "resource_policy_version": RESOURCE_POLICY_VERSION,
            "resource_demand": _default_resource_demand({**row, **existing}),
            "claim": claim.get("claim") or {},
            "command": command,
            "execution_id": str((claim.get("claim") or {}).get("execution_id") or (claim.get("claim") or {}).get("claim_id") or ""),
            "child_stdout_path": str(child_stdout_path),
            "child_stderr_path": str(child_stderr_path),
            "started_at": utc_now(),
            "last_heartbeat": utc_now(),
            **_worker_runtime_context(worker_id or f"pid_{os.getpid()}"),
        }
        _set_worker_active(output_root, worker_id=state["worker_id"], payload=state)
        publish_candidate_state(output_root, state)
        atomic_write_json(task_state_path, state)
        candidate_start_wall = time.time()
        rss_before_kb = _rss_kb()
        try:
            with child_stdout_path.open("a", encoding="utf-8") as stdout_handle, child_stderr_path.open("a", encoding="utf-8") as stderr_handle:
                proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=stdout_handle, stderr=stderr_handle, check=False)
        except (subprocess.TimeoutExpired, OSError, BrokenPipeError, MemoryError, Exception) as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            candidate_end_wall = time.time()
            failure = {
                **state,
                "status": "RETRY_PENDING",
                "failure_reason": f"parent_exception_after_running:{type(exc).__name__}",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "return_code": None,
                "stdout_tail": _tail_file(child_stdout_path),
                "stderr_tail": _tail_file(child_stderr_path),
                "finished_at": utc_now(),
                "elapsed_sec": round(candidate_end_wall - candidate_start_wall, 3),
                "last_heartbeat": utc_now(),
            }
            publish_candidate_state(output_root, failure)
            atomic_write_json(task_state_path, failure)
            try:
                append_resource_telemetry(
                    output_root,
                    {
                        "schema_version": "candidate_resource_telemetry_v1",
                        "scientific_run_id": _scientific_run_id(output_root),
                        "candidate_id": candidate_id,
                        "case_id": case_id,
                        "canonical_target": target,
                        "teacher_id": teacher,
                        "worker_id": state["worker_id"],
                        "exception_type": type(exc).__name__,
                        "failure_reason": failure["failure_reason"],
                        "elapsed_sec": failure["elapsed_sec"],
                        "child_stdout_path": str(child_stdout_path),
                        "child_stderr_path": str(child_stderr_path),
                        **_worker_runtime_context(state["worker_id"]),
                    },
                )
            except Exception as telemetry_exc:
                failure["resource_telemetry_warning"] = f"{type(telemetry_exc).__name__}: {telemetry_exc}"
                atomic_write_json(task_state_path, failure)
            return failure
        candidate_end_wall = time.time()
        rss_after_kb = _rss_kb()
        candidate_state = _candidate_from_single_teacher_run(case_output, case_id=case_id, target=target, teacher=teacher)
        candidate_state["ct_path"] = str(row.get("ct_path") or "")
        profile = str(state.get("profile") or "")
        stdout_tail = _tail_file(child_stdout_path)
        stderr_tail = _tail_file(child_stderr_path)
        resource_failure = classify_resource_failure(
            return_code=int(proc.returncode),
            stdout=stdout_tail,
            stderr=stderr_tail,
            profile=profile,
        )
        if proc.returncode != 0 and resource_failure and candidate_state["status"] not in {"SUCCESS", "ABSENT", "OUT_OF_FOV"}:
            candidate_state = _apply_resource_failure(
                output_root,
                {
                    **state,
                    **candidate_state,
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "target": target,
                    "teacher": teacher,
                },
                failure_class=resource_failure,
                profile=profile,
            )
        elif proc.returncode != 0 and candidate_state["status"] not in {"SUCCESS", "ABSENT", "OUT_OF_FOV"}:
            candidate_state["status"] = "FAILED_FINAL"
            candidate_state["failure_reason"] = stderr_tail[-1000:] or stdout_tail[-1000:] or "candidate_worker_failed"
        telemetry = {
            "schema_version": "candidate_resource_telemetry_v1",
            "scientific_run_id": _scientific_run_id(output_root),
            "execution_attempt_id": os.getenv("TASK2_EXECUTION_ATTEMPT_ID", ""),
            "candidate_id": candidate_id,
            "case_id": case_id,
            "canonical_target": target,
            "teacher_id": teacher,
            "model_id": teacher,
            "checkpoint_model_fingerprint": str(row.get("checkpoint_path") or row.get("checkpoint_root") or ""),
            "git_commit": os.getenv("EXPECTED_GIT_COMMIT", ""),
            "inference_mode": os.getenv("MEDAI_TEACHER_INFERENCE_MODE", "hierarchical_roi"),
            "resource_policy_version": RESOURCE_POLICY_VERSION,
            "gpu_type": os.getenv("SLURM_JOB_GPUS", ""),
            "partition": os.getenv("SLURM_JOB_PARTITION", ""),
            "node": os.getenv("SLURMD_NODENAME", os.getenv("HOSTNAME", "")),
            "worker_job_id": os.getenv("SLURM_JOB_ID", ""),
            "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID", ""),
            "worker_profile": profile,
            "requested_memory": os.getenv("SLURM_MEM_PER_NODE", os.getenv("SLURM_MEM_PER_CPU", "")),
            "requested_cpu": os.getenv("SLURM_CPUS_PER_TASK", ""),
            "requested_gpu": os.getenv("SLURM_GPUS", ""),
            "candidate_start_time": state["started_at"],
            "candidate_end_time": utc_now(),
            "elapsed_sec": round(candidate_end_wall - candidate_start_wall, 3),
            "candidate_peak_rss_kb": max(rss_before_kb, rss_after_kb),
            "telemetry_confidence": "PROCESS_TREE_RUSAGE" if max(rss_before_kb, rss_after_kb) > 0 else "WORKER_ONLY",
            "exit_code": int(proc.returncode),
            "child_stdout_path": str(child_stdout_path),
            "child_stderr_path": str(child_stderr_path),
            "slurm_state": os.getenv("SLURM_JOB_STATE", ""),
            "oom": resource_failure == "CPU_OOM",
            "timeout": resource_failure == "WALLTIME_EXCEEDED",
            "node_fail": resource_failure == "NODE_FAIL",
            "preempted": resource_failure == "PREEMPTED",
            "resource_failure_class": resource_failure,
            "resource_demand": candidate_state.get("resource_demand") or state.get("resource_demand"),
            **_ct_geometry(str(row.get("ct_path") or "")),
        }
        try:
            append_resource_telemetry(output_root, telemetry)
        except Exception as exc:
            candidate_state["resource_telemetry_warning"] = f"{type(exc).__name__}: {exc}"
        candidate_state.update({
            "return_code": int(proc.returncode),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "child_stdout_path": str(child_stdout_path),
            "child_stderr_path": str(child_stderr_path),
            "finished_at": utc_now(),
            "resource_policy_version": RESOURCE_POLICY_VERSION,
        })
        publish_candidate_state(output_root, candidate_state)
        final = {
            **state,
            "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
            "candidate_status": candidate_state["status"],
            "return_code": int(proc.returncode),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "child_stdout_path": str(child_stdout_path),
            "child_stderr_path": str(child_stderr_path),
            "finished_at": utc_now(),
            "case_output": str(case_output),
            "resource_failure_class": resource_failure,
            "resource_telemetry": telemetry,
        }
        atomic_write_json(task_state_path, final)
        return final
    finally:
        _clear_worker_active(output_root, worker_id or f"pid_{os.getpid()}")
        release_claim(output_root, claim_kind="candidate", claim_key=candidate_id)


def execute_task_index(task_index: int, task_manifest: Path, output_root: Path, *, worker_id: str = "", state_root: Path | None = None) -> dict[str, Any]:
    rows = read_csv_rows(task_manifest)
    row = next((item for item in rows if int(item.get("task_index") or -1) == int(task_index)), None)
    if row is None:
        raise IndexError(f"task index not found: {task_index}")
    return _execute_candidate_row(row, output_root, worker_id=worker_id, state_root=state_root, already_claimed=False)


def run_candidate_queue_worker(
    output_root: Path,
    *,
    task_manifest: Path,
    worker_id: str,
    profile: str,
    resource_class: str,
    state_root: Path | None = None,
    poll_sec: int = 15,
    max_idle_sec: int = 180,
    max_tasks: int = 8,
    lease_sec: int = 7200,
) -> dict[str, Any]:
    started = time.time()
    completed = 0
    idle_since: float | None = None
    manifest_rows = _candidate_queue_rows(output_root, task_manifest=task_manifest)
    seed_marker_check = _candidate_seed_marker_matches(output_root, task_manifest=task_manifest, rows=manifest_rows)
    cursor = _manifest_cursor_offset(output_root, worker_id=worker_id, manifest_rows=manifest_rows)
    while True:
        capability = _worker_capability(profile, resource_class)
        walltime_sec = int(capability.get("walltime_sec") or 0)
        if capability.get("short_worker") and walltime_sec > 0:
            elapsed = time.time() - started
            guard = int(os.getenv("TASK2_SHORT_CLAIM_GUARD_SEC", "1800")) + int(os.getenv("TASK2_SHORT_CLEANUP_MARGIN_SEC", "300"))
            if walltime_sec - elapsed < guard:
                telemetry = build_estep_telemetry(output_root)
                return {
                    "status": "DRAINING",
                    "reason": "remaining_walltime_below_claim_guard",
                    "worker_id": worker_id,
                    "profile": profile,
                    "resource_class": resource_class,
                    "completed": completed,
                    "runtime_sec": round(elapsed, 3),
                    "remaining_walltime_sec": round(walltime_sec - elapsed, 3),
                    "claim_guard_sec": guard,
                    "telemetry": telemetry,
                }
        if seed_marker_check["status"] != "MATCH":
            seed_marker_check = _candidate_seed_marker_matches(output_root, task_manifest=task_manifest, rows=manifest_rows)
        claim = claim_next_ready_candidate_from_rows(
            output_root,
            manifest_rows=manifest_rows,
            task_manifest=task_manifest,
            worker_id=worker_id,
            profile=profile,
            resource_class=resource_class,
            lease_sec=lease_sec,
            start_index=cursor,
            seed_marker_check=seed_marker_check,
        )
        if "next_cursor" in claim:
            cursor = int(claim.get("next_cursor") or cursor)
        if claim["status"] == "CLAIMED":
            idle_since = None
            result = _execute_candidate_row(
                claim["row"],
                output_root,
                worker_id=worker_id,
                state_root=state_root,
                already_claimed=True,
            )
            completed += 1
            if max_tasks > 0 and completed >= max_tasks:
                telemetry = build_estep_telemetry(output_root)
                return {
                    "status": "MAX_TASKS_REACHED",
                    "worker_id": worker_id,
                    "profile": profile,
                    "resource_class": resource_class,
                    "completed": completed,
                    "runtime_sec": round(time.time() - started, 3),
                    "last_result": result,
                    "telemetry": telemetry,
                }
            continue
        telemetry = build_estep_telemetry(output_root)
        if idle_since is None:
            idle_since = time.time()
        if time.time() - idle_since >= max_idle_sec:
            return {
                "status": "IDLE_EXIT",
                "worker_id": worker_id,
                "profile": profile,
                "resource_class": resource_class,
                "completed": completed,
                "runtime_sec": round(time.time() - started, 3),
                "last_claim_status": claim["status"],
                "telemetry": telemetry,
            }
        time.sleep(max(1, int(poll_sec)))


def _case_rows_from_scope(scope: dict[str, Any]) -> list[str]:
    rows = scope.get("task_rows") or []
    return sorted({str(row.get("case_id") or "") for row in rows if isinstance(row, dict) and row.get("case_id")})


def _target_rows_from_scope(scope: dict[str, Any]) -> list[str]:
    routes = scope.get("routes") or {}
    if routes:
        return sorted(str(target) for target in routes)
    return sorted({str(row.get("target") or "") for row in scope.get("task_rows") or [] if isinstance(row, dict) and row.get("target")})


def _load_target_config_prompts(path: Path) -> tuple[list[str], dict[str, str]]:
    doc = _read_json(path, {})
    targets = [str(item) for item in doc.get("target_organs") or []]
    prompts = {str(k): str(v) for k, v in (doc.get("organ_to_prompt") or {}).items()}
    return targets, prompts


def _candidate_for_selection(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(state.get("teacher") or state.get("model") or ""),
        "prediction": str(state.get("prediction") or ""),
        "candidate_id": str(state.get("candidate_id") or ""),
        "candidate_exists": bool(state.get("candidate_exists")),
        "eligible_for_labelcritic": bool(state.get("eligible_for_labelcritic", True)),
        "candidate_qc_status": state.get("candidate_qc_status") or "pass",
        "candidate_qc_score": state.get("candidate_qc_score"),
        "candidate_qc_flags": state.get("candidate_qc_flags") or [],
        "candidate_qc": state.get("candidate_qc") or {},
        "candidate_shapekit_status": state.get("candidate_shapekit_status"),
        "candidate_shapekit_reason": state.get("candidate_shapekit_reason"),
        "candidate_raw_prediction": state.get("candidate_raw_prediction"),
        "candidate_cleaned_prediction": state.get("candidate_cleaned_prediction"),
        "em_candidate_role": "teacher_candidate",
        "candidate_source_role": "teacher_pseudo_candidate",
    }


def _terminal_from_selection(selection: dict[str, Any], selected: dict[str, Any] | None, candidate_count: int) -> str:
    method = str(selection.get("selection_method") or "")
    status = str(selection.get("selection_status") or "")
    if selected and status == "selected" and method == "label_critic":
        return "SELECTED"
    if selected and method in {"single_teacher_provisional", "single_teacher_default"}:
        return "VALID_SINGLE_TEACHER_ACCEPTED"
    if selected and status in {"selected", "provisional"}:
        return "SELECTED"
    if candidate_count <= 0:
        return "ABSENT"
    return "UNRESOLVED_REVIEW"


def claim_next_case_target(output_root: Path, *, worker_id: str, labelcritic_ready: bool, lease_sec: int = 1800) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    ready_states: list[dict[str, Any]] = []
    for case_id in _case_rows_from_scope(scope):
        for target in _target_rows_from_scope(scope):
            state = recompute_case_target_readiness(output_root, case_id=case_id, target=target)
            if state.get("status") == "CANDIDATES_READY":
                ready_states.append(state)
    if not ready_states:
        return {"status": "NO_READY_CASE_TARGETS"}
    if not labelcritic_ready:
        for state in ready_states:
            state["status"] = "WAITING_FOR_LABELCRITIC"
            atomic_write_json(_state_path(output_root, state["case_id"], state["target"], kind="target"), state)
        return {"status": "WAITING_FOR_LABELCRITIC", "queue_depth": len(ready_states)}
    ready_states.sort(key=lambda row: (str(row.get("case_id") or ""), str(row.get("target") or "")))
    for state in ready_states:
        claim_key = f"{state['case_id']}|{state['target']}"
        claim = claim_work(output_root, claim_kind="case_target", claim_key=claim_key, worker_id=worker_id, lease_sec=lease_sec)
        if claim["status"] == "CLAIMED":
            state["status"] = "LABELCRITIC_RUNNING"
            state["claim"] = claim["claim"]
            atomic_write_json(_state_path(output_root, state["case_id"], state["target"], kind="target"), state)
            return {"status": "CLAIMED", "case_target": state, "claim": claim}
    return {"status": "NO_CLAIMABLE_CASE_TARGETS", "queue_depth": len(ready_states)}


def select_case_target(
    output_root: Path,
    *,
    case_id: str,
    target: str,
    critic_base_url: str,
    critic_port: int,
    worker_id: str = "",
    target_config: Path | None = None,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    target = _norm(target)
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    teachers = [str(item) for item in (scope.get("routes") or {}).get(target, [])]
    candidate_states = [load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher) for teacher in teachers]
    if not teachers or any(str(state.get("status") or "") not in TERMINAL_CANDIDATE_STATES for state in candidate_states):
        return {"status": "WAITING_FOR_CANDIDATES", "case_id": case_id, "target": target}
    candidates = [
        _candidate_for_selection(state)
        for state in candidate_states
        if state.get("status") == "SUCCESS" and state.get("candidate_exists") and state.get("prediction") and Path(str(state.get("prediction"))).is_file()
    ]
    ct_path = ""
    for state in candidate_states:
        source_meta = _read_json(Path(str(state.get("selection_metadata") or "")), {})
        ct_path = str(source_meta.get("ct_path") or state.get("ct_path") or ct_path)
    ct = Path(ct_path) if ct_path else Path("missing_ct.nii.gz")
    selection_dir = output_root / "selection_runs" / re_safe(case_id)
    selected: dict[str, Any] | None = None
    try:
        selected, selection = _select_candidate(
            ct=ct,
            organ=target,
            candidates=candidates,
            out=selection_dir,
            case_id=case_id,
            enable_critic=bool(len(candidates) > 1),
            critic_backend="labelcritic",
            critic_base_url=critic_base_url,
            critic_port=int(critic_port),
            timeout_sec=int(timeout_sec),
            dry_run=False,
            strict_labelcritic_selection=True,
            formal_72b_selection_ready=True,
        )
    except Exception as exc:
        state = {
            "status": "RETRY_PENDING",
            "case_id": case_id,
            "target": target,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "updated_at": utc_now(),
        }
        atomic_write_json(_state_path(output_root, case_id, target, kind="target"), state)
        release_claim(output_root, claim_kind="case_target", claim_key=f"{case_id}|{target}")
        return state
    terminal_state = _terminal_from_selection(selection, selected, len(candidates))
    final_mask = ""
    normalization = {}
    if selected and selected.get("prediction") and Path(str(selected["prediction"])).is_file():
        final_dir = output_root / "annotation_versions" / case_id / "updated"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{target}.nii.gz"
        normalization = write_binary_mask_nifti_from_source(Path(str(selected["prediction"])), final_path)
        final_mask = str(final_path)
    result = {
        "status": terminal_state,
        "case_id": case_id,
        "ct_path": ct_path,
        "target": target,
        "organ": target,
        "selected_model": (selected or {}).get("model") or selection.get("selected_model"),
        "selected_candidate_id": (selected or {}).get("candidate_id"),
        "selected_prediction": (selected or {}).get("prediction") or selection.get("selected_prediction"),
        "final_mask": final_mask,
        "mask_path": final_mask,
        "selection": selection,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "teacher_names": [candidate["model"] for candidate in candidates],
        "labelcritic_compare_used": bool(selection.get("labelcritic_records") or selection.get("critic_records")),
        "labelcritic_records": selection.get("labelcritic_records") or selection.get("critic_records") or [],
        "formal_mask_contract": normalization,
        "updated_at": utc_now(),
    }
    atomic_write_json(_state_path(output_root, case_id, target, kind="selection"), result)
    atomic_write_json(_state_path(output_root, case_id, target, kind="target"), result)
    _append_event(output_root, {"event": "case_target_selected", "case_id": case_id, "target": target, "status": terminal_state, "selected_model": result.get("selected_model")})
    release_claim(output_root, claim_kind="case_target", claim_key=f"{case_id}|{target}")
    return result


def run_labelcritic_selection_worker(
    output_root: Path,
    *,
    worker_id: str,
    critic_base_url: str,
    critic_port: int,
    target_config: Path | None = None,
    poll_sec: int = 30,
    max_idle_sec: int = 300,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    started = time.time()
    idle_since: float | None = None
    completed = 0
    wait_labelcritic = 0
    while True:
        claim = claim_next_case_target(output_root, worker_id=worker_id, labelcritic_ready=True)
        if claim["status"] == "CLAIMED":
            idle_since = None
            case_target = claim["case_target"]
            result = select_case_target(
                output_root,
                case_id=str(case_target["case_id"]),
                target=str(case_target["target"]),
                critic_base_url=critic_base_url,
                critic_port=critic_port,
                worker_id=worker_id,
                target_config=target_config,
                timeout_sec=timeout_sec,
            )
            completed += 1 if result.get("status") in VALID_TERMINAL_TARGET_STATES else 0
            continue
        if claim["status"] == "WAITING_FOR_LABELCRITIC":
            wait_labelcritic += 1
        if idle_since is None:
            idle_since = time.time()
        if time.time() - idle_since >= max_idle_sec:
            telemetry = build_estep_telemetry(output_root)
            return {
                "status": "IDLE_EXIT",
                "worker_id": worker_id,
                "completed": completed,
                "wait_labelcritic_count": wait_labelcritic,
                "runtime_sec": round(time.time() - started, 3),
                "telemetry": telemetry,
            }
        time.sleep(max(1, int(poll_sec)))


def build_estep_telemetry(output_root: Path) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    candidate_total = int(scope.get("total_logical_candidate_tasks") or len(scope.get("task_rows") or []))
    case_target_total = int(scope.get("case_count") or 0) * int(scope.get("canonical_target_count") or 0)
    candidate_counts: dict[str, int] = {}
    for path in _queue_paths(output_root)["candidate_states"].glob("*/*/*.json"):
        status = str(_read_json(path, {}).get("status") or "PENDING")
        candidate_counts[status] = candidate_counts.get(status, 0) + 1
    target_counts: dict[str, int] = {}
    for path in _queue_paths(output_root)["case_target_states"].glob("*/*.json"):
        status = str(_read_json(path, {}).get("status") or "WAITING_FOR_CANDIDATES")
        target_counts[status] = target_counts.get(status, 0) + 1
    terminal_target_count = sum(target_counts.get(state, 0) for state in VALID_TERMINAL_TARGET_STATES)
    queue_depth = target_counts.get("CANDIDATES_READY", 0) + target_counts.get("WAITING_FOR_LABELCRITIC", 0)
    telemetry = {
        "status": "READY",
        "updated_at": utc_now(),
        "task_ownership": "shared_queue",
        "profile_binding": False,
        "teacher_candidate": {
            "total": candidate_total,
            "ready_candidates": candidate_counts.get("READY", 0),
            "claimed_candidates": candidate_counts.get("CLAIMED", 0),
            "running_candidates": candidate_counts.get("RUNNING", 0),
            "terminal_candidates": sum(candidate_counts.get(state, 0) for state in TERMINAL_CANDIDATE_STATES),
            "success": candidate_counts.get("SUCCESS", 0),
            "running": candidate_counts.get("RUNNING", 0) + candidate_counts.get("CLAIMED", 0),
            "pending": max(0, candidate_total - sum(candidate_counts.values())),
            "retry": candidate_counts.get("RETRY_PENDING", 0) + candidate_counts.get("BACKPRESSURED", 0),
            "terminal": sum(candidate_counts.get(state, 0) for state in TERMINAL_CANDIDATE_STATES),
            "counts": candidate_counts,
        },
        "case_target": {
            "total": case_target_total,
            "waiting_candidates": max(0, case_target_total - sum(target_counts.values())) + target_counts.get("WAITING_FOR_CANDIDATES", 0),
            "candidates_ready": target_counts.get("CANDIDATES_READY", 0),
            "labelcritic_running": target_counts.get("LABELCRITIC_RUNNING", 0),
            "selected": target_counts.get("SELECTED", 0),
            "single_teacher_accepted": target_counts.get("VALID_SINGLE_TEACHER_ACCEPTED", 0),
            "absent": target_counts.get("ABSENT", 0) + target_counts.get("ABSENT_NEGATIVE", 0) + target_counts.get("NEGATIVE_ABSENT", 0),
            "failed": target_counts.get("FAILED_FINAL", 0),
            "terminal": terminal_target_count,
            "counts": target_counts,
        },
        "labelcritic": {
            "queue_depth": queue_depth,
            "running_requests": target_counts.get("LABELCRITIC_RUNNING", 0),
            "completed": target_counts.get("SELECTED", 0),
            "retry": target_counts.get("RETRY_PENDING", 0),
        },
    }
    atomic_write_json(_queue_paths(output_root)["telemetry"], telemetry)
    return telemetry


def _selection_training_item(selection: dict[str, Any], *, target_config: Path | None = None) -> dict[str, Any] | None:
    if not selection.get("final_mask") or not Path(str(selection.get("final_mask"))).is_file():
        return None
    targets, prompts = _load_target_config_prompts(target_config or (REPO_ROOT / "configs/student_3d_prompt_target_organs.json"))
    target = str(selection.get("target") or selection.get("organ") or "")
    try:
        target_id = targets.index(target)
    except ValueError:
        target_id = -1
    item = {
        "case_id": str(selection.get("case_id") or ""),
        "image": str(selection.get("ct_path") or ""),
        "ct_path": str(selection.get("ct_path") or ""),
        "organ": target,
        "canonical_organ": target,
        "requested_canonical_id": target,
        "resolved_canonical_id": target,
        "prompt": prompts.get(target, target.replace("_", " ")),
        "mask": str(selection.get("final_mask")),
        "mask_path": str(selection.get("final_mask")),
        "supervision_type": "positive",
        "target_type": "positive_hard",
        "label_role": "selected_pseudo_label",
        "supervision_role": "selected_pseudo_label",
        "distillation_role": "positive",
        "dataset_role": "pseudo_label",
        "source_model": str(selection.get("selected_model") or ""),
        "selected_model": str(selection.get("selected_model") or ""),
        "origin_provider": str(selection.get("selected_model") or ""),
        "ground_truth_status": "selected_pseudo_label_not_expert_gt",
        "scoring_schema_version": "autolabel_core_v3",
        "grade": "A",
        "training_weight": 1.0,
        "distillation_eligible": True,
        "training_eligible": True,
        "student_target_id": target_id,
        "source_stage": "full_373_multiteacher_round1_estep",
    }
    return canonicalize_training_record(item, round_index=1, project_root=REPO_ROOT, strict_soft=True)


def _case_manifest_context(output_root: Path) -> dict[str, dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest in sorted((output_root / "slurm").glob("*.csv")):
        try:
            rows.extend(read_csv_rows(manifest))
        except Exception:
            continue
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        entry = out.setdefault(case_id, {})
        for key in ("ct_path", "original_annotation_folder", "canonical_annotation_folder"):
            value = str(row.get(key) or "").strip()
            if value and not entry.get(key):
                entry[key] = value
    return out


def _task1_source_training_item(
    *,
    case_id: str,
    target: str,
    context: dict[str, str],
    target_config: Path | None = None,
) -> dict[str, Any] | None:
    canonical_dir = Path(str(context.get("canonical_annotation_folder") or ""))
    mask = canonical_dir / f"{target}{NIFTI_SUFFIX}"
    ct_path = str(context.get("ct_path") or "")
    if not canonical_dir.is_dir() or not mask.is_file() or not ct_path:
        return None
    targets, prompts = _load_target_config_prompts(target_config or (REPO_ROOT / "configs/student_3d_prompt_target_organs.json"))
    try:
        target_id = targets.index(target)
    except ValueError:
        target_id = -1
    item = {
        "case_id": case_id,
        "image": ct_path,
        "ct_path": ct_path,
        "organ": target,
        "canonical_organ": target,
        "requested_canonical_id": target,
        "resolved_canonical_id": target,
        "prompt": prompts.get(target, target.replace("_", " ")),
        "mask": str(mask),
        "mask_path": str(mask),
        "supervision_type": "positive",
        "target_type": "positive_hard",
        "label_role": "task1_source_label",
        "supervision_role": "task1_source_label",
        "distillation_role": "positive",
        "dataset_role": "source_label",
        "source": "task1_source",
        "source_model": "task1_source",
        "selected_model": "task1_source",
        "origin_provider": "task1_source",
        "selected_teacher": "",
        "eligible_teachers": [],
        "ground_truth_status": "task1_source_not_expert_gt",
        "scoring_schema_version": "autolabel_core_v3_task1_source",
        "grade": "A",
        "training_weight": 1.0,
        "distillation_eligible": True,
        "training_eligible": True,
        "student_target_id": target_id,
        "source_stage": "task1_373_canonical_source_mask",
        "task1_source_mask_sha256": sha256_file_if_exists(mask),
    }
    return canonicalize_training_record(item, round_index=1, project_root=REPO_ROOT, strict_soft=True)


def _merged_training_item(
    *,
    output_root: Path,
    case_id: str,
    target: str,
    state: dict[str, Any],
    case_context: dict[str, dict[str, str]],
    target_config: Path | None = None,
) -> dict[str, Any] | None:
    task1_item = _task1_source_training_item(
        case_id=case_id,
        target=target,
        context=case_context.get(case_id, {}),
        target_config=target_config,
    )
    if task1_item:
        return task1_item
    item = _selection_training_item(state, target_config=target_config)
    if item:
        item["source"] = "task2_teacher"
        item["selected_teacher"] = item.get("selected_model") or item.get("source_model") or ""
        item["eligible_teachers"] = list((state.get("teacher_names") or state.get("eligible_teachers") or []))
    return item


def aggregate_full373_estep(output_root: Path, *, expected_cases: int = 103, expected_targets: int = 373) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {})
    if int(scope.get("case_count") or 0) > 0:
        expected_cases = int(scope.get("case_count") or expected_cases)
    if int(scope.get("canonical_target_count") or 0) > 0:
        expected_targets = int(scope.get("canonical_target_count") or expected_targets)
    cases = sorted({
        str(row.get("case_id") or "")
        for row in (scope.get("task_rows") or [])
        if isinstance(row, dict) and row.get("case_id")
    })
    if not cases:
        manifest_rows: list[dict[str, str]] = []
        for task_manifest in sorted((output_root / "slurm").glob("full373_task_manifest_*.csv")):
            manifest_rows.extend(read_csv_rows(task_manifest))
        cases = sorted({str(row.get("case_id") or "") for row in manifest_rows if row.get("case_id")})
    telemetry = build_estep_telemetry(output_root)
    state_items = []
    training_items = []
    pending = []
    failed = []
    case_context = _case_manifest_context(output_root)
    state_files = list(_queue_paths(output_root)["case_target_states"].glob("*/*.json"))
    if state_files:
        for case_id in cases:
            for target in _target_rows_from_scope(scope):
                state = _read_json(_state_path(output_root, case_id, target, kind="target"), {})
                status = str(state.get("status") or "WAITING_FOR_CANDIDATES")
                if status in VALID_TERMINAL_TARGET_STATES:
                    state_items.append({"case_id": case_id, "organ": target, "terminal_state": status, **state})
                    item = _merged_training_item(output_root=output_root, case_id=case_id, target=target, state=state, case_context=case_context)
                    if item:
                        training_items.append(item)
                elif status in NON_TERMINAL_TARGET_STATES:
                    pending.append({"case_id": case_id, "organ": target, "reason": status})
                else:
                    failed.append({"case_id": case_id, "organ": target, "reason": status})
        manifest_targets = len(state_items)
        expected_total = expected_cases * expected_targets
        status = "PASSED" if len(cases) == expected_cases and manifest_targets == expected_total and not pending and not failed else "RUNNING"
        if failed and not pending:
            status = "FAILED"
        trainable = [
            item for item in training_items
            if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0
        ]
        training_manifest = {
            "version": "full_373_multiteacher_round1_voxtell_manifest_v1",
            "stage": "round1_mstep_manifest",
            "status": "success" if status == "PASSED" else "pending",
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "source_formal_root": str(output_root),
            "num_items": len(training_items),
            "num_cases": len({str(item.get("case_id") or "") for item in training_items}),
            "num_distillation_eligible_items": len(trainable),
            "items": training_items,
        }
        if status == "PASSED":
            write_json(output_root / "training_manifest.json", training_manifest)
        report = {
            "stage": "full_373_multiteacher_round1_estep_gate",
            "status": status,
            "case_count": len(cases),
            "expected_case_count": expected_cases,
            "canonical_target_count": expected_targets,
            "expected_targets": expected_total,
            "manifest_targets": manifest_targets,
            "complete_case_373": manifest_targets == expected_total,
            "pending_cases": pending[:100],
            "failed_cases": failed[:100],
            "allowed_terminal_states": sorted(VALID_TERMINAL_TARGET_STATES),
            "training_manifest": str(output_root / "training_manifest.json") if training_items else "",
            "num_training_items": len(training_items),
            "num_distillation_eligible_items": len(trainable),
            "telemetry": telemetry,
        }
        write_json(output_root / "full_case_373_manifest.json", {"stage": "full_case_373_estep_manifest", "status": "success" if status == "PASSED" else status.lower(), "items": state_items})
        write_json(output_root / "full_373_estep_status.json", report)
        return report

    items = []
    training_items = []
    pending = []
    failed = []
    for case_id in cases:
        case_root = output_root / "run_loop_cases" / case_id
        manifest = _read_json(case_root / "full_case_373_manifest.json", {})
        summary = _read_json(case_root / "case_373_target_summary.json", {})
        task_state = _read_json(output_root / "task_states" / f"{case_id}.json", {})
        if task_state.get("status") == "FAILED":
            failed.append({"case_id": case_id, "reason": "run_loop_task_failed", "task_state": task_state})
            continue
        if manifest.get("status") != "success" or not summary.get("complete_case_373"):
            pending.append({"case_id": case_id, "reason": "full_case_373_not_complete", "manifest_status": manifest.get("status"), "complete_case_373": summary.get("complete_case_373")})
            continue
        items.extend(manifest.get("items") or [])
        tm = _read_json(case_root / "training_manifest.json", {})
        training_items.extend(tm.get("items") or [])
    manifest_targets = len(items)
    expected_total = expected_cases * expected_targets
    status = "PASSED" if len(cases) == expected_cases and manifest_targets == expected_total and not pending and not failed else "RUNNING"
    if failed and not pending:
        status = "FAILED"
    trainable = [
        item for item in training_items
        if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0
    ]
    training_manifest = {
        "version": "full_373_multiteacher_round1_voxtell_manifest_v1",
        "stage": "round1_mstep_manifest",
        "status": "success" if training_items else "pending",
        "source_formal_root": str(output_root),
        "num_items": len(training_items),
        "num_cases": len({str(item.get("case_id") or "") for item in training_items}),
        "num_distillation_eligible_items": len(trainable),
        "items": training_items,
    }
    if training_items:
        write_json(output_root / "training_manifest.json", training_manifest)
    report = {
        "stage": "full_373_multiteacher_round1_estep_gate",
        "status": status,
        "case_count": len(cases),
        "expected_case_count": expected_cases,
        "canonical_target_count": expected_targets,
        "expected_targets": expected_total,
        "manifest_targets": manifest_targets,
        "complete_case_373": manifest_targets == expected_total,
        "pending_cases": pending[:100],
        "failed_cases": failed[:100],
        "allowed_terminal_states": sorted(VALID_TERMINAL_TARGET_STATES),
        "training_manifest": str(output_root / "training_manifest.json") if training_items else "",
        "num_training_items": len(training_items),
        "num_distillation_eligible_items": len(trainable),
    }
    write_json(output_root / "full_373_estep_status.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Full 373 multi-Teacher Round1 launcher.")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs/model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs/student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--checkpoint-root", default=REPO_ROOT / "checkpoints", type=Path)
    parser.add_argument("--nnunet-predict-executable", default=Path("nnUNetv2_predict"), type=Path)
    parser.add_argument("--unest-python-executable", default=Path(sys.executable), type=Path)
    parser.add_argument("--totalsegmentator-executable", default=resolve_totalsegmentator_executable_arg())
    parser.add_argument("--cache-root", action="append", default=[], type=Path)
    parser.add_argument("--execute-task-index", type=int)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--selection-worker", action="store_true")
    parser.add_argument("--queue-worker", action="store_true")
    parser.add_argument("--worker-profile", default="")
    parser.add_argument("--worker-resource-class", default="GPU_INFERENCE_COMPATIBLE")
    parser.add_argument("--queue-poll-sec", default=int(os.getenv("TEACHER_QUEUE_POLL_SEC", "15")), type=int)
    parser.add_argument("--queue-max-idle-sec", default=int(os.getenv("TEACHER_QUEUE_MAX_IDLE_SEC", "180")), type=int)
    parser.add_argument("--queue-max-tasks", default=int(os.getenv("TEACHER_QUEUE_MAX_TASKS_PER_WORKER", "8")), type=int)
    parser.add_argument("--claim-lease-sec", default=int(os.getenv("TEACHER_QUEUE_CLAIM_LEASE_SEC", "7200")), type=int)
    parser.add_argument("--critic-base-url", default=os.getenv("LABELCRITIC_BASE_URL", "http://localhost"))
    parser.add_argument("--critic-port", default=int(os.getenv("LABELCRITIC_PORT", "8000")), type=int)
    parser.add_argument("--selection-poll-sec", default=int(os.getenv("LABELCRITIC_SELECTION_POLL_SEC", "30")), type=int)
    parser.add_argument("--selection-max-idle-sec", default=int(os.getenv("LABELCRITIC_SELECTION_MAX_IDLE_SEC", "300")), type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.execute_task_index is not None:
        if not args.task_manifest:
            raise SystemExit("--task-manifest is required with --execute-task-index")
        try:
            result = execute_task_index(args.execute_task_index, args.task_manifest.resolve(), output_root, worker_id=args.worker_id, state_root=args.state_root)
        except Exception as exc:
            crash = record_worker_crash(output_root, worker_id=args.worker_id or f"task_{os.getpid()}", exc=exc, extra={"mode": "execute_task_index", "task_index": args.execute_task_index})
            print(json.dumps(crash, indent=2, default=str))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.queue_worker:
        if not args.task_manifest:
            raise SystemExit("--task-manifest is required with --queue-worker")
        worker_id = args.worker_id or f"queue_{os.getpid()}"
        try:
            result = run_candidate_queue_worker(
                output_root,
                task_manifest=args.task_manifest.resolve(),
                worker_id=worker_id,
                profile=args.worker_profile or "generic_gpu",
                resource_class=args.worker_resource_class,
                state_root=args.state_root,
                poll_sec=args.queue_poll_sec,
                max_idle_sec=args.queue_max_idle_sec,
                max_tasks=args.queue_max_tasks,
                lease_sec=args.claim_lease_sec,
            )
        except Exception as exc:
            crash = record_worker_crash(output_root, worker_id=worker_id, exc=exc, extra={"mode": "queue_worker", "profile": args.worker_profile or "generic_gpu", "resource_class": args.worker_resource_class})
            print(json.dumps(crash, indent=2, default=str))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.selection_worker:
        print(json.dumps(run_labelcritic_selection_worker(
            output_root,
            worker_id=args.worker_id or f"selection_{os.getpid()}",
            critic_base_url=args.critic_base_url,
            critic_port=args.critic_port,
            target_config=args.target_config,
            poll_sec=args.selection_poll_sec,
            max_idle_sec=args.selection_max_idle_sec,
        ), indent=2, default=str))
        return 0
    if args.aggregate:
        report = aggregate_full373_estep(output_root)
        print(json.dumps({"status": report["status"], "manifest_targets": report["manifest_targets"], "expected_targets": report["expected_targets"]}, indent=2))
        return 0 if report["status"] in {"PASSED", "RUNNING"} else 2
    if not args.case_manifest:
        raise SystemExit("--case-manifest is required")
    summary = build_submission_manifest(
        case_manifest=args.case_manifest.resolve(),
        output_root=output_root,
        registry_path=args.registry.resolve(),
        target_config=args.target_config.resolve(),
        python=args.python,
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        unest_python_executable=args.unest_python_executable,
        totalsegmentator_executable=args.totalsegmentator_executable,
        state_root=args.state_root,
        cache_roots=[path.resolve() for path in args.cache_root],
    )
    print(json.dumps({"status": summary["status"], "task_count": summary["task_count"], "scope": summary["scope"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
