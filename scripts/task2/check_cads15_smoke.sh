#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
SMOKE_ROOT=${SMOKE_ROOT:-${CADS15_SMOKE_ROOT:-}}

if [ -z "$SMOKE_ROOT" ]; then
  latest=$(ls -td /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/cads15_teacher_smoke_* 2>/dev/null | head -1 || true)
  if [ -z "$latest" ]; then
    echo "Set SMOKE_ROOT or CADS15_SMOKE_ROOT; no cads15_teacher_smoke_* directory found." >&2
    exit 2
  fi
  SMOKE_ROOT=$latest
fi

cd "$CODE_ROOT"
JOBS_FILE=$SMOKE_ROOT/cads15_smoke_jobs.csv
SLURM_STATUS=$SMOKE_ROOT/slurm_status.csv
printf "group,case_id,targets,job_id,state,exit_code\n" > "$SLURM_STATUS"

if [ -f "$JOBS_FILE" ]; then
  if ! command -v sacct >/dev/null 2>&1; then
    echo "sacct is required to validate submitted CADS15 smoke jobs." >&2
    exit 2
  fi
  tail -n +2 "$JOBS_FILE" | while IFS=, read -r group case_id targets job_id sbatch_file; do
    [ -n "$group" ] || continue
    line=$(sacct -j "$job_id" --format=JobIDRaw,State,ExitCode -n -P | awk -F'|' -v job="$job_id" '$1 == job {print; exit}')
    if [ -z "$line" ]; then
      printf "%s,%s,%s,%s,%s,%s\n" "$group" "$case_id" "$targets" "$job_id" "UNKNOWN" "UNKNOWN" >> "$SLURM_STATUS"
    else
      state=$(printf "%s" "$line" | awk -F'|' '{print $2}')
      exit_code=$(printf "%s" "$line" | awk -F'|' '{print $3}')
      printf "%s,%s,%s,%s,%s,%s\n" "$group" "$case_id" "$targets" "$job_id" "$state" "$exit_code" >> "$SLURM_STATUS"
    fi
  done
fi

"$PYTHON" tools/dataset_delivery/task2_smoke_validator.py \
  --smoke-root "$SMOKE_ROOT" \
  --groups cads15 \
  --panel-json "$SMOKE_ROOT/preflight/cads15_smoke_case_panel.json" \
  --slurm-status-csv "$SLURM_STATUS"

echo "CADS15_SMOKE_VERDICT=$SMOKE_ROOT/task2_smoke_verdict.json"
