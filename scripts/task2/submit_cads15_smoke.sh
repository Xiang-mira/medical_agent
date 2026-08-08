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
PANEL_PARTITION=${PANEL_PARTITION:-${TASK2_CPU_PARTITION:-cpu}}
PANEL_CPUS_PER_TASK=${PANEL_CPUS_PER_TASK:-${TASK2_CADS_PANEL_CPUS:-8}}
PANEL_MEM=${PANEL_MEM:-${TASK2_CADS_PANEL_MEMORY:-32G}}
PANEL_TIME_LIMIT=${PANEL_TIME_LIMIT:-${TASK2_CADS_PANEL_TIME:-02:00:00}}
GPU_PARTITION=${GPU_PARTITION:-${TASK2_CADS_GPU_PARTITION:-gpu}}
GPU_GRES=${GPU_GRES:-${TASK2_CADS_GPU_GRES:-gpu:T4:1}}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-${TASK2_CADS_GPU_CPUS:-8}}
GPU_MEM=${GPU_MEM:-${TASK2_CADS_GPU_MEMORY:-64G}}
GPU_TIME_LIMIT=${GPU_TIME_LIMIT:-${TASK2_CADS_GPU_TIME:-06:00:00}}
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
PREP_STATUS=$("$PYTHON" - "$OUT_ROOT/submission_manifest.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
print(doc.get("status") or "")
PY
)
if [ "$PREP_STATUS" != "NOT_SUBMITTED" ]; then
  echo "CADS15 smoke resource preflight blocked submission: $PREP_STATUS" >&2
  exit 2
fi
PANEL_SUBMIT_ERR=$OUT_ROOT/panel_sbatch_submit.err
if ! PANEL_JOB_ID=$(sbatch --parsable "$PANEL_SBATCH" 2>"$PANEL_SUBMIT_ERR"); then
  "$PYTHON" - "$OUT_ROOT/submission_manifest.json" "$PANEL_SUBMIT_ERR" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
stderr_path = Path(sys.argv[2])
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "PANEL_SUBMISSION_REJECTED"
doc["panel"]["state"] = "PANEL_SUBMISSION_REJECTED"
doc["panel"]["submission_stderr"] = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
  echo "CADS15 panel sbatch submission was rejected; GPU smoke will not be submitted." >&2
  exit 2
fi
if [ -z "$PANEL_JOB_ID" ]; then
  "$PYTHON" - "$OUT_ROOT/submission_manifest.json" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "PANEL_NOT_SUBMITTED"
doc["panel"]["state"] = "PANEL_NOT_SUBMITTED"
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
  echo "CADS15 panel submission did not return a job id; GPU smoke will not be submitted." >&2
  exit 2
fi
if ! [[ "$PANEL_JOB_ID" =~ ^[0-9]+$ ]]; then
  "$PYTHON" - "$OUT_ROOT/submission_manifest.json" "$PANEL_JOB_ID" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
panel_job = sys.argv[2]
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "PANEL_JOB_ID_INVALID"
doc["panel"]["job_id"] = panel_job
doc["panel"]["state"] = "PANEL_JOB_ID_INVALID"
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
  echo "CADS15 panel sbatch returned malformed job id '$PANEL_JOB_ID'; GPU smoke will not be submitted." >&2
  exit 2
fi
printf "%s\n" "$PANEL_JOB_ID" > "$OUT_ROOT/panel_job_id.txt"
"$PYTHON" - "$OUT_ROOT/submission_manifest.json" "$PANEL_JOB_ID" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
panel_job = sys.argv[2]
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "PANEL_PENDING"
doc["panel"]["job_id"] = panel_job
doc["panel"]["state"] = "PANEL_PENDING"
doc["gpu"]["state"] = "GPU_NOT_SUBMITTED"
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
GPU_SUBMIT_ERR=$OUT_ROOT/gpu_sbatch_submit.err
if ! GPU_JOB_ID=$(sbatch --parsable --dependency=afterok:"$PANEL_JOB_ID" "$GPU_SBATCH" 2>"$GPU_SUBMIT_ERR"); then
  "$PYTHON" - "$OUT_ROOT/submission_manifest.json" "$PANEL_JOB_ID" "$GPU_SUBMIT_ERR" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
panel_job = sys.argv[2]
stderr_path = Path(sys.argv[3])
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
doc["status"] = "GPU_SUBMISSION_REJECTED"
doc["panel"]["job_id"] = panel_job
doc["panel"]["state"] = "PANEL_PENDING"
doc["gpu"]["dependency"] = f"afterok:{panel_job}"
doc["gpu"]["state"] = "GPU_SUBMISSION_REJECTED"
doc["gpu"]["submission_stderr"] = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
  echo "CADS15 GPU sbatch submission was rejected after panel job $PANEL_JOB_ID; dependency smoke was not submitted." >&2
  exit 2
fi
if [ -z "$GPU_JOB_ID" ] || ! [[ "$GPU_JOB_ID" =~ ^[0-9]+$ ]]; then
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
doc["status"] = "GPU_JOB_ID_INVALID"
doc["panel"]["job_id"] = panel_job
doc["panel"]["state"] = "PANEL_PENDING"
doc["gpu"]["job_ids"] = [gpu_job] if gpu_job else []
doc["gpu"]["dependency"] = f"afterok:{panel_job}"
doc["gpu"]["state"] = "GPU_JOB_ID_INVALID"
tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, manifest_path)
PY
  echo "CADS15 GPU sbatch returned malformed job id '$GPU_JOB_ID' after panel job $PANEL_JOB_ID." >&2
  exit 2
fi
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
