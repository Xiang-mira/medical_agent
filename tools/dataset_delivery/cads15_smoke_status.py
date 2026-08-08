#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root  # noqa: E402


FAIL_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY", "BOOT_FAIL"}
DEPENDENCY_UNSATISFIED_STATES = {"DependencyNeverSatisfied", "DEPENDENCYNEVERSATISFIED"}
RUNNING_STATES = {"RUNNING", "CONFIGURING", "COMPLETING"}
PENDING_STATES = {"PENDING", "REQUEUED", "SUSPENDED"}
FAILED_WORKFLOW_STATUSES = {
    "PANEL_RESOURCE_INVALID",
    "PANEL_SCRIPT_INVALID",
    "PANEL_NOT_SUBMITTED",
    "PANEL_SUBMISSION_REJECTED",
    "PANEL_FAILED",
    "GPU_RESOURCE_INVALID",
    "GPU_SCRIPT_INVALID",
    "GPU_NOT_SUBMITTED",
    "PANEL_JOB_ID_INVALID",
    "GPU_JOB_ID_INVALID",
    "GPU_DEPENDENCY_UNSATISFIED",
    "GPU_SUBMISSION_REJECTED",
    "GPU_SMOKE_FAILED",
    "VALIDATION_FAILED",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _sacct_state(job_id: str) -> dict[str, str]:
    if not job_id:
        return {"known": "false", "state": "", "exit_code": ""}
    if not str(job_id).isdigit():
        return {"known": "false", "state": "", "exit_code": "", "reason": "malformed_job_id"}
    try:
        proc = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobIDRaw,State,ExitCode", "-n", "-P"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return {"known": "false", "state": "", "exit_code": "", "reason": "sacct_not_found"}
    if proc.returncode != 0:
        return {"known": "false", "state": "", "exit_code": "", "reason": proc.stderr.strip()}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if parts and parts[0] == job_id:
            return {"known": "true", "state": parts[1] if len(parts) > 1 else "", "exit_code": parts[2] if len(parts) > 2 else ""}
    return {"known": "false", "state": "", "exit_code": ""}


def _job_state(job_id: str, *, use_sacct: bool) -> dict[str, str]:
    if job_id and not str(job_id).isdigit():
        return {"known": "false", "state": "", "exit_code": "", "reason": "malformed_job_id"}
    return _sacct_state(job_id) if use_sacct else {"known": "false", "state": "", "exit_code": ""}


def smoke_status(*, smoke_root: Path | None, use_sacct: bool = True) -> dict[str, Any]:
    if smoke_root is None or not smoke_root.exists():
        return {"status": "NOT_SUBMITTED", "gpu_smoke_started": False}
    manifest = _read_json(smoke_root / "submission_manifest.json")
    if not manifest:
        return {"status": "NOT_SUBMITTED", "smoke_root": str(smoke_root), "gpu_smoke_started": False}
    panel_job_id = str(manifest.get("panel", {}).get("job_id") or _read_text(smoke_root / "panel_job_id.txt"))
    gpu_job_ids = list(manifest.get("gpu", {}).get("job_ids") or [])
    if not gpu_job_ids:
        text = _read_text(smoke_root / "gpu_job_ids.txt")
        gpu_job_ids = [line.strip() for line in text.splitlines() if line.strip()]
    panel_progress = _read_json(smoke_root / "preflight" / "panel_progress.json") or _read_json(smoke_root / "panel_progress.json")
    panel_state_json = _read_json(smoke_root / "preflight" / "panel_state.json") or _read_json(smoke_root / "panel_state.json")
    manifest_status = str(manifest.get("status") or "")
    panel_job = _job_state(panel_job_id, use_sacct=use_sacct)
    base = {
        "smoke_root": str(smoke_root),
        "panel_job_id": panel_job_id,
        "gpu_job_ids": gpu_job_ids,
        "panel_cases_completed": int(panel_progress.get("panel_cases_completed") or panel_state_json.get("panel_cases_completed") or 0),
        "panel_cases_total": int(panel_progress.get("panel_cases_total") or panel_state_json.get("panel_cases_total") or 0),
        "gpu_smoke_started": bool(gpu_job_ids),
        "panel_slurm": panel_job,
    }
    if manifest_status in {
        "PANEL_RESOURCE_INVALID",
        "PANEL_SCRIPT_INVALID",
        "PANEL_NOT_SUBMITTED",
        "PANEL_SUBMISSION_REJECTED",
        "GPU_RESOURCE_INVALID",
        "GPU_SCRIPT_INVALID",
        "GPU_NOT_SUBMITTED",
        "GPU_SUBMISSION_REJECTED",
    }:
        return {**base, "status": manifest_status, "reason": (manifest.get("resource_preflight") or {}).get("failure_reason", "")}
    if not panel_job_id and manifest_status in {"", "NOT_SUBMITTED"}:
        return {**base, "status": "NOT_SUBMITTED", "reason": "no_panel_job_id"}
    if not panel_job_id and panel_state_json.get("status") == "NOT_SUBMITTED":
        return {**base, "status": "NOT_SUBMITTED", "reason": "panel_not_submitted"}
    if panel_job_id and not panel_job_id.isdigit():
        return {**base, "status": "PANEL_JOB_ID_INVALID", "reason": "malformed_job_id"}
    if (smoke_root / "PANEL_FAILED").exists() or panel_state_json.get("status") == "PANEL_FAILED" or panel_job.get("state") in FAIL_STATES:
        return {**base, "status": "PANEL_FAILED"}
    panel_completed = (
        (smoke_root / "PANEL_COMPLETED").exists()
        or panel_state_json.get("status") == "PANEL_COMPLETED"
        or (panel_job.get("state") == "COMPLETED" and panel_job.get("exit_code") == "0:0")
    )
    if not panel_completed:
        if panel_job.get("state") in RUNNING_STATES or panel_state_json.get("status") == "PANEL_RUNNING":
            return {**base, "status": "PANEL_RUNNING"}
        return {**base, "status": "PANEL_PENDING"}
    if not gpu_job_ids:
        return {**base, "status": "GPU_SMOKE_PENDING"}
    invalid_gpu_job_ids = [str(job_id) for job_id in gpu_job_ids if not str(job_id).isdigit()]
    if invalid_gpu_job_ids:
        return {**base, "status": "GPU_JOB_ID_INVALID", "reason": "malformed_job_id", "invalid_gpu_job_ids": invalid_gpu_job_ids}
    gpu_states = [_job_state(str(job_id), use_sacct=use_sacct) for job_id in gpu_job_ids]
    gpu_progress = _read_json(smoke_root / "gpu_progress.json")
    gpu_state_json = _read_json(smoke_root / "gpu_state.json")
    gpu_base = {**base, "gpu_slurm": gpu_states, "gpu_progress": gpu_progress}
    if any(row.get("state") in DEPENDENCY_UNSATISFIED_STATES for row in gpu_states):
        return {**gpu_base, "status": "GPU_DEPENDENCY_UNSATISFIED"}
    if (smoke_root / "GPU_SMOKE_FAILED").exists() or gpu_state_json.get("status") == "GPU_SMOKE_FAILED" or any(row.get("state") in FAIL_STATES for row in gpu_states):
        return {**gpu_base, "status": "GPU_SMOKE_FAILED"}
    known_gpu_states = [row for row in gpu_states if row.get("known") == "true"]
    gpu_completed = (
        (smoke_root / "GPU_SMOKE_COMPLETED").exists()
        or gpu_state_json.get("status") == "GPU_SMOKE_COMPLETED"
        or (
            bool(known_gpu_states)
            and all(row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0" for row in known_gpu_states)
        )
    )
    if not gpu_completed:
        if any(row.get("state") in RUNNING_STATES for row in gpu_states) or gpu_state_json.get("status") == "GPU_SMOKE_RUNNING":
            return {**gpu_base, "status": "GPU_SMOKE_RUNNING"}
        return {**gpu_base, "status": "GPU_SMOKE_PENDING"}
    report = validate_smoke_root(
        smoke_root=smoke_root,
        groups=["cads15"],
        panel_json=smoke_root / "preflight" / "cads15_smoke_case_panel.json",
        write_outputs=False,
    )
    if report.get("status") in {"passed", "PASSED"}:
        return {**gpu_base, "status": "PASSED", "validation": report.get("cads15_summary", {})}
    return {**gpu_base, "status": "VALIDATION_FAILED", "validation": report.get("cads15_summary", {}), "failed_groups": report.get("failed_groups", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Report CADS15 smoke state without mutating smoke outputs.")
    parser.add_argument("--smoke-root", default=None, type=Path)
    parser.add_argument("--no-sacct", action="store_true")
    args = parser.parse_args()
    report = smoke_status(smoke_root=args.smoke_root.resolve() if args.smoke_root else None, use_sacct=not bool(args.no_sacct))
    print(json.dumps(report, indent=2))
    return 1 if report["status"] in FAILED_WORKFLOW_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
