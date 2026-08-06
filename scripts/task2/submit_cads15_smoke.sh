#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
OUT_ROOT=${OUT_ROOT:-$OUT_PARENT/cads15_teacher_smoke_$(date +%Y%m%d_%H%M%S)}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
PANEL_PARTITION=${PANEL_PARTITION:-shared}
PANEL_CPUS_PER_TASK=${PANEL_CPUS_PER_TASK:-8}
PANEL_MEM=${PANEL_MEM:-32G}
PANEL_TIME_LIMIT=${PANEL_TIME_LIMIT:-02:00:00}
GPU_PARTITION=${GPU_PARTITION:-gpu}
GPU_GRES=${GPU_GRES:-gpu:t4:1}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-8}
GPU_MEM=${GPU_MEM:-64G}
GPU_TIME_LIMIT=${GPU_TIME_LIMIT:-06:00:00}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

cd "$CODE_ROOT"
git fetch origin
if [ "$(git status --porcelain --untracked-files=no)" != "" ]; then
  echo "Refusing CADS15 smoke submission with tracked worktree changes." >&2
  exit 2
fi
git checkout main
git pull --ff-only origin main

branch=$(git branch --show-current)
if [ "$branch" != "main" ]; then
  echo "Refusing CADS15 smoke submission from branch $branch; expected main." >&2
  exit 2
fi
if [ "$(git status --porcelain --untracked-files=no)" != "" ]; then
  echo "Refusing CADS15 smoke submission with tracked worktree changes after pull." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required on HPC to submit CADS15 smoke jobs." >&2
  exit 2
fi

mkdir -p "$OUT_PARENT" "$STATE_ROOT"

"$PYTHON" tools/dataset_delivery/cads15_smoke_launcher.py \
  --prepare-orchestration \
  --smoke-root "$OUT_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --code-root "$CODE_ROOT" \
  --python "$PYTHON" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --timeout-sec "$TIMEOUT_SEC" \
  --panel-partition "$PANEL_PARTITION" \
  --panel-cpus-per-task "$PANEL_CPUS_PER_TASK" \
  --panel-mem "$PANEL_MEM" \
  --panel-time-limit "$PANEL_TIME_LIMIT" \
  --gpu-partition "$GPU_PARTITION" \
  --gpu-gres "$GPU_GRES" \
  --gpu-cpus-per-task "$GPU_CPUS_PER_TASK" \
  --gpu-mem "$GPU_MEM" \
  --gpu-time-limit "$GPU_TIME_LIMIT" \
  --allow-heavy-ct-fov

PANEL_SBATCH=$OUT_ROOT/slurm/cads15_panel_prepare.sbatch
GPU_SBATCH=$OUT_ROOT/slurm/cads15_gpu_smoke.sbatch
PANEL_JOB_ID=$(sbatch --parsable "$PANEL_SBATCH")
GPU_JOB_ID=$(sbatch --parsable --dependency=afterok:"$PANEL_JOB_ID" "$GPU_SBATCH")

printf "%s\n" "$PANEL_JOB_ID" > "$OUT_ROOT/panel_job_id.txt"
printf "%s\n" "$GPU_JOB_ID" > "$OUT_ROOT/gpu_job_ids.txt"
printf "group,case_id,targets,job_id,sbatch_file,dependency\n" > "$OUT_ROOT/cads15_smoke_jobs.csv"
printf "cads15_panel,,,\"%s\",\"%s\",\n" "$PANEL_JOB_ID" "$PANEL_SBATCH" >> "$OUT_ROOT/cads15_smoke_jobs.csv"
printf "cads15_gpu,*,panel_selected,\"%s\",\"%s\",\"afterok:%s\"\n" "$GPU_JOB_ID" "$GPU_SBATCH" "$PANEL_JOB_ID" >> "$OUT_ROOT/cads15_smoke_jobs.csv"

"$PYTHON" - "$OUT_ROOT/submission_manifest.json" "$PANEL_JOB_ID" "$GPU_JOB_ID" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
panel_job = sys.argv[2]
gpu_job = sys.argv[3]
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "PANEL_PENDING"
doc["panel"]["job_id"] = panel_job
doc["panel"]["state"] = "PANEL_PENDING"
doc["gpu"]["job_ids"] = [gpu_job]
doc["gpu"]["dependency"] = f"afterok:{panel_job}"
doc["gpu"]["state"] = "GPU_SMOKE_PENDING"
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY

LAST_FILE=$STATE_ROOT/.last_cads15_smoke
tmp_last=$STATE_ROOT/.last_cads15_smoke.tmp.$$
printf "%s\n" "$OUT_ROOT" > "$tmp_last"
mv "$tmp_last" "$LAST_FILE"

echo "CADS15_SMOKE_ROOT=$OUT_ROOT"
echo "CADS15_PANEL_JOB_ID=$PANEL_JOB_ID"
echo "CADS15_GPU_JOB_ID=$GPU_JOB_ID"
