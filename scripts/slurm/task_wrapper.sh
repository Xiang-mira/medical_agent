#!/bin/bash
# Generic scheduler task wrapper template. Concrete per-task sbatch scripts are
# generated under runs/<run_id>/generated_slurm/.
set -euo pipefail

echo "[scheduler] job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "[scheduler] host=$(hostname) cwd=$(pwd)"
echo "[scheduler] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

exec "$@"
