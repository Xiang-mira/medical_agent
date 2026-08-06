#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
SMOKE_ROOT=${SMOKE_ROOT:-${TASK2_SMOKE_ROOT:-}}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
TASK2_SMOKE_GROUPS=${TASK2_SMOKE_GROUPS:-atm,airrc,unest}

if [ -z "$SMOKE_ROOT" ]; then
  if [ -f "$STATE_ROOT/.last_task2_teacher_smoke" ]; then
    SMOKE_ROOT=$(cat "$STATE_ROOT/.last_task2_teacher_smoke")
  else
    echo '{"status":"NOT_SUBMITTED","failed_groups":[],"groups":{}}'
    exit 0
  fi
fi

cd "$CODE_ROOT"
JOBS_FILE=$SMOKE_ROOT/smoke_jobs.csv
SLURM_STATUS=$SMOKE_ROOT/slurm_status.csv
printf "group,job_id,state,exit_code,node\n" > "$SLURM_STATUS"

if [ -f "$JOBS_FILE" ]; then
  tail -n +2 "$JOBS_FILE" | while IFS=, read -r group job_id sbatch_file status submission_status return_code stdout stderr; do
    [ -n "$group" ] || continue
    [ -n "$job_id" ] || continue
    if command -v squeue >/dev/null 2>&1; then
      line=$(squeue -h -j "$job_id" -o "%T|%N" 2>/dev/null | head -1 || true)
      if [ -n "$line" ]; then
        state=$(printf "%s" "$line" | awk -F'|' '{print $1}')
        node=$(printf "%s" "$line" | awk -F'|' '{print $2}')
        printf "%s,%s,%s,%s,%s\n" "$group" "$job_id" "$state" "" "$node" >> "$SLURM_STATUS"
        continue
      fi
    fi
    if command -v sacct >/dev/null 2>&1; then
      line=$(sacct -j "$job_id" --format=JobIDRaw,State,ExitCode,NodeList -n -P 2>/dev/null | awk -F'|' -v job="$job_id" '$1 == job {print; exit}')
      if [ -n "$line" ]; then
        state=$(printf "%s" "$line" | awk -F'|' '{print $2}')
        exit_code=$(printf "%s" "$line" | awk -F'|' '{print $3}')
        node=$(printf "%s" "$line" | awk -F'|' '{print $4}')
        printf "%s,%s,%s,%s,%s\n" "$group" "$job_id" "$state" "$exit_code" "$node" >> "$SLURM_STATUS"
        continue
      fi
    fi
    if [ -n "${status:-}" ]; then
      printf "%s,%s,%s,%s,%s\n" "$group" "$job_id" "$status" "" "" >> "$SLURM_STATUS"
    else
      printf "%s,%s,%s,%s,%s\n" "$group" "$job_id" "UNKNOWN" "UNKNOWN" "" >> "$SLURM_STATUS"
    fi
  done
fi

"$PYTHON" tools/dataset_delivery/task2_smoke_validator.py \
  --smoke-root "$SMOKE_ROOT" \
  --groups "$TASK2_SMOKE_GROUPS" \
  --slurm-status-csv "$SLURM_STATUS"

echo "TASK2_SMOKE_VERDICT=$SMOKE_ROOT/task2_smoke_verdict.json"
