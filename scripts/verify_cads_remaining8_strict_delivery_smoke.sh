#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CADS_SMOKE_ROOT=${CADS_SMOKE_ROOT:-${OUT_ROOT:-}}
CADS_SMOKE_JOB_ID=${CADS_SMOKE_JOB_ID:-}

if [ -z "$CADS_SMOKE_ROOT" ]; then
  echo "CADS_SMOKE_ROOT or OUT_ROOT is required" >&2
  exit 2
fi
if [ -z "$CADS_SMOKE_JOB_ID" ] && [ -f "$CADS_SMOKE_ROOT/cads_remaining8_smoke_submission.env" ]; then
  # shellcheck disable=SC1090
  source "$CADS_SMOKE_ROOT/cads_remaining8_smoke_submission.env"
fi

if [ -n "$CADS_SMOKE_JOB_ID" ]; then
  if ! command -v sacct >/dev/null 2>&1; then
    echo "sacct is required when CADS_SMOKE_JOB_ID is provided" >&2
    exit 2
  fi
  SACCT_LINE=$(sacct -j "$CADS_SMOKE_JOB_ID" --format=JobIDRaw,State,ExitCode -n -P | awk -F'|' -v job="$CADS_SMOKE_JOB_ID" '$1 == job {print; exit}')
  if [ -z "$SACCT_LINE" ]; then
    echo "No sacct row found for CADS_SMOKE_JOB_ID=$CADS_SMOKE_JOB_ID" >&2
    exit 2
  fi
  SLURM_STATE=$(printf "%s" "$SACCT_LINE" | awk -F'|' '{print $2}')
  SLURM_EXIT_CODE=$(printf "%s" "$SACCT_LINE" | awk -F'|' '{print $3}')
else
  SLURM_STATE=${SLURM_STATE:-not_checked}
  SLURM_EXIT_CODE=${SLURM_EXIT_CODE:-not_checked}
fi

export CADS_SMOKE_ROOT CADS_SMOKE_JOB_ID SLURM_STATE SLURM_EXIT_CODE
"$PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root = Path(os.environ["CADS_SMOKE_ROOT"])
verdict_path = root / "smoke_verdict.json"
failures: list[str] = []
if not verdict_path.exists():
    failures.append("smoke_verdict_missing")
    verdict = {}
else:
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

if os.environ.get("CADS_SMOKE_JOB_ID"):
    if os.environ.get("SLURM_STATE") != "COMPLETED":
        failures.append(f"slurm_state_not_completed:{os.environ.get('SLURM_STATE')}")
    if os.environ.get("SLURM_EXIT_CODE") != "0:0":
        failures.append(f"slurm_exit_code_not_zero:{os.environ.get('SLURM_EXIT_CODE')}")
if verdict and verdict.get("status") != "passed":
    failures.append(f"smoke_status:{verdict.get('status')}")
if verdict:
    failed = verdict.get("failed_targets") or []
    if failed:
        failures.append("failed_targets:" + ",".join(str(item) for item in failed))
    if len(verdict.get("passed_targets") or []) != len(verdict.get("requested_targets") or []):
        failures.append("not_all_requested_targets_passed")

payload = {
    "status": "passed" if not failures else "failed",
    "smoke_root": str(root),
    "job_id": os.environ.get("CADS_SMOKE_JOB_ID") or "",
    "slurm_state": os.environ.get("SLURM_STATE"),
    "slurm_exit_code": os.environ.get("SLURM_EXIT_CODE"),
    "verdict_path": str(verdict_path),
    "failures": failures,
}
(root / "CADS_REMAINING8_STRICT_SMOKE_PASS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
(root / "CADS_REMAINING8_STRICT_SMOKE_PASS.md").write_text(
    "\n".join([
        "# CADS Remaining8 Strict-delivery Smoke",
        "",
        f"- Status: `{payload['status']}`",
        f"- Slurm: `{payload['slurm_state']}` / `{payload['slurm_exit_code']}`",
        f"- Smoke root: `{root}`",
        f"- Failures: `{';'.join(failures) if failures else 'none'}`",
    ]) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
raise SystemExit(0 if not failures else 1)
PY
