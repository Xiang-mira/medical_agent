#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_ID=${CASE_ID:-BDMAP_00000120}
ATM_SMOKE_JOB_ID=${ATM_SMOKE_JOB_ID:-}
ATM_SMOKE_RUN_OUT=${ATM_SMOKE_RUN_OUT:-${RUN_OUT:-}}
ATM_SMOKE_ROOT=${ATM_SMOKE_ROOT:-}

if [ -z "$ATM_SMOKE_RUN_OUT" ]; then
  echo "ATM_SMOKE_RUN_OUT or RUN_OUT is required" >&2
  exit 2
fi
if [ -z "$ATM_SMOKE_ROOT" ]; then
  ATM_SMOKE_ROOT=$(dirname "$ATM_SMOKE_RUN_OUT")
fi
if [ -z "$ATM_SMOKE_JOB_ID" ]; then
  if [ -f "$ATM_SMOKE_ROOT/atm_smoke_submission.env" ]; then
    # shellcheck disable=SC1090
    source "$ATM_SMOKE_ROOT/atm_smoke_submission.env"
  fi
fi
if [ -z "$ATM_SMOKE_JOB_ID" ]; then
  echo "ATM_SMOKE_JOB_ID is required for the Slurm completion gate" >&2
  exit 2
fi

if ! command -v sacct >/dev/null 2>&1; then
  echo "sacct is required for the Slurm completion gate" >&2
  exit 2
fi

SACCT_LINE=$(sacct -j "$ATM_SMOKE_JOB_ID" --format=JobIDRaw,State,ExitCode -n -P | awk -F'|' -v job="$ATM_SMOKE_JOB_ID" '$1 == job {print; exit}')
if [ -z "$SACCT_LINE" ]; then
  echo "No sacct row found for ATM_SMOKE_JOB_ID=$ATM_SMOKE_JOB_ID" >&2
  exit 2
fi
SLURM_STATE=$(printf "%s" "$SACCT_LINE" | awk -F'|' '{print $2}')
SLURM_EXIT_CODE=$(printf "%s" "$SACCT_LINE" | awk -F'|' '{print $3}')

export ATM_SMOKE_JOB_ID ATM_SMOKE_RUN_OUT ATM_SMOKE_ROOT CASE_ID SLURM_STATE SLURM_EXIT_CODE
"$PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


root = Path(os.environ["ATM_SMOKE_ROOT"])
run_out = Path(os.environ["ATM_SMOKE_RUN_OUT"])
case_id = os.environ["CASE_ID"]
job_id = os.environ["ATM_SMOKE_JOB_ID"]
failures: list[str] = []

if os.environ.get("SLURM_STATE") != "COMPLETED":
    failures.append(f"slurm_state_not_completed:{os.environ.get('SLURM_STATE')}")
if os.environ.get("SLURM_EXIT_CODE") != "0:0":
    failures.append(f"slurm_exit_code_not_zero:{os.environ.get('SLURM_EXIT_CODE')}")

summary = read_json(run_out / "run_summary.json")
if summary.get("status") != "success":
    failures.append(f"run_summary_status_not_success:{summary.get('status')}")
if int(summary.get("strict_delivery_failure_count") or 0) != 0:
    failures.append(f"strict_delivery_failure_count_nonzero:{summary.get('strict_delivery_failure_count')}")
if int(summary.get("total_updated") or 0) != 1:
    failures.append(f"total_updated_not_one:{summary.get('total_updated')}")
if "airway_tree" not in set(summary.get("strict_delivery_fov_override_organs") or []):
    failures.append("summary_override_organs_missing_airway_tree")
if not any(
    row.get("organ") == "airway_tree" and row.get("decision") == "applied"
    for row in summary.get("strict_delivery_fov_override_rows") or []
):
    failures.append("summary_override_applied_missing_airway_tree")
if any(
    row.get("organ") == "airway_tree" and row.get("reason") == "requested_organs_pruned_by_fov"
    for row in summary.get("strict_delivery_failures") or []
):
    failures.append("airway_tree_pruned_failure_present")

plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
plan = read_json(plan_path)
if "atm" not in set(plan.get("teacher_run_list") or []):
    failures.append("teacher_run_list_missing_atm")
if "airway_tree" not in set((plan.get("fov_override") or {}).get("applied_organs") or []):
    failures.append("plan_override_applied_missing_airway_tree")

inference = read_json(run_out / "inference_results.json")
inference_rows = inference if isinstance(inference, list) else inference.get("rows", [])
atm_rows = [row for row in inference_rows if row.get("model_key") == "atm" or row.get("model") == "atm"]
if not atm_rows:
    failures.append("inference_results_missing_atm")
elif not any(row.get("status") == "success" for row in atm_rows):
    failures.append("atm_inference_status_not_success")

mask_path = run_out / "annotation_versions" / case_id / "updated" / "airway_tree.nii.gz"
ct_path = Path(str(plan.get("ct_path") or ""))
mask_metrics: dict[str, object] = {"path": str(mask_path)}
try:
    mask_img = nib.load(str(mask_path))
    ct_img = nib.load(str(ct_path))
    arr = np.asanyarray(mask_img.dataobj)
    values = set(np.unique(arr).astype(int).tolist())
    foreground_voxels = int((arr > 0).sum())
    mask_metrics.update({
        "exists": mask_path.exists(),
        "foreground_voxels": foreground_voxels,
        "binary": values.issubset({0, 1}),
        "shape_matches_ct": mask_img.shape[:3] == ct_img.shape[:3],
        "spacing_matches_ct": tuple(round(float(x), 6) for x in mask_img.header.get_zooms()[:3])
        == tuple(round(float(x), 6) for x in ct_img.header.get_zooms()[:3]),
        "affine_matches_ct": bool(np.allclose(mask_img.affine, ct_img.affine)),
    })
except Exception as exc:
    failures.append(f"nifti_validation_error:{type(exc).__name__}:{exc}")
else:
    for key in ["exists", "binary", "shape_matches_ct", "spacing_matches_ct", "affine_matches_ct"]:
        if not mask_metrics.get(key):
            failures.append(f"mask_{key}_failed")
    if int(mask_metrics.get("foreground_voxels") or 0) <= 0:
        failures.append("mask_foreground_voxels_zero")

payload = {
    "status": "passed" if not failures else "failed",
    "job_id": job_id,
    "slurm_state": os.environ.get("SLURM_STATE"),
    "slurm_exit_code": os.environ.get("SLURM_EXIT_CODE"),
    "run_out": str(run_out),
    "case_id": case_id,
    "checks": {
        "run_summary_status": summary.get("status"),
        "strict_delivery_failure_count": summary.get("strict_delivery_failure_count"),
        "total_updated": summary.get("total_updated"),
        "teacher_run_list": plan.get("teacher_run_list"),
        "override": plan.get("fov_override"),
        "mask": mask_metrics,
    },
    "failures": failures,
}
root.mkdir(parents=True, exist_ok=True)
(root / "ATM_STRICT_SMOKE_PASS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
md = [
    "# ATM Strict-delivery Smoke",
    "",
    f"- Status: `{payload['status']}`",
    f"- Slurm: `{payload['slurm_state']}` / `{payload['slurm_exit_code']}`",
    f"- Run output: `{run_out}`",
    f"- Failures: `{';'.join(failures) if failures else 'none'}`",
]
(root / "ATM_STRICT_SMOKE_PASS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if not failures else 1)
PY
