#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CASE_ID=${CASE_ID:-BDMAP_00000120}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
UNEST_PYTHON_EXECUTABLE=${UNEST_PYTHON_EXECUTABLE:-/home/xhan74/envs/medical_agent_train_py311/bin/python}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/task2_teacher_smokes_$(date +%Y%m%d_%H%M%S)}
GROUPS=${GROUPS:-atm,cads,unest}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:t4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-64G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

cd "$CODE_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main

branch=$(git branch --show-current)
if [ "$branch" != "main" ]; then
  echo "Refusing smoke submission from branch $branch; expected main." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
"$PYTHON" tools/dataset_delivery/task2_smoke_launcher.py \
  --smoke-root "$OUT_ROOT" \
  --groups "$GROUPS" \
  --case-manifest "$CASE_MANIFEST" \
  --case-id "$CASE_ID" \
  --code-root "$CODE_ROOT" \
  --python "$PYTHON" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --unest-python-executable "$UNEST_PYTHON_EXECUTABLE" \
  --formal-mode \
  --timeout-sec "$TIMEOUT_SEC" \
  --partition "$PARTITION" \
  --gres "$GRES" \
  --cpus-per-task "$CPUS_PER_TASK" \
  --mem "$MEM" \
  --time-limit "$TIME_LIMIT"

JOBS_FILE=$OUT_ROOT/smoke_jobs.csv
printf "group,job_id,sbatch_file\n" > "$JOBS_FILE"
"$PYTHON" - "$OUT_ROOT/prepare_summary.json" "$JOBS_FILE" <<'PY'
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
jobs_file = Path(sys.argv[2])
rows = []
for group in summary["groups"]:
    sbatch = group["sbatch_file"]
    proc = subprocess.run(["sbatch", "--parsable", sbatch], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"sbatch failed for {group['group']}: {proc.stderr.strip()}")
    rows.append({"group": group["group"], "job_id": proc.stdout.strip(), "sbatch_file": sbatch})
with jobs_file.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["group", "job_id", "sbatch_file"])
    writer.writerows(rows)
print(json.dumps({"status": "submitted", "jobs": rows}, indent=2))
PY

echo "TASK2_SMOKE_ROOT=$OUT_ROOT"
echo "TASK2_SMOKE_JOBS=$JOBS_FILE"
