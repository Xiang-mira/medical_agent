#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/cads15_teacher_smoke_$(date +%Y%m%d_%H%M%S)}
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
  echo "Refusing CADS15 smoke submission from branch $branch; expected main." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/preflight"

"$PYTHON" tools/dataset_delivery/cads15_contract_audit.py \
  --output-root "$OUT_ROOT/preflight/route_audit" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE"

"$PYTHON" tools/dataset_delivery/cads15_smoke_panel.py \
  --case-manifest "$CASE_MANIFEST" \
  --output-root "$OUT_ROOT/preflight"

"$PYTHON" tools/dataset_delivery/cads15_smoke_launcher.py \
  --smoke-root "$OUT_ROOT" \
  --panel-json "$OUT_ROOT/preflight/cads15_smoke_case_panel.json" \
  --code-root "$CODE_ROOT" \
  --python "$PYTHON" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --timeout-sec "$TIMEOUT_SEC" \
  --partition "$PARTITION" \
  --gres "$GRES" \
  --cpus-per-task "$CPUS_PER_TASK" \
  --mem "$MEM" \
  --time-limit "$TIME_LIMIT"

JOBS_FILE=$OUT_ROOT/cads15_smoke_jobs.csv
printf "group,case_id,targets,job_id,sbatch_file\n" > "$JOBS_FILE"
"$PYTHON" - "$OUT_ROOT/cads15_sbatch_manifest.csv" "$JOBS_FILE" <<'PY'
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
jobs_file = Path(sys.argv[2])
rows = []
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        proc = subprocess.run(["sbatch", "--parsable", row["sbatch_file"]], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"sbatch failed for {row['case_id']}: {proc.stderr.strip()}")
        rows.append({
            "group": row["group"],
            "case_id": row["case_id"],
            "targets": row["targets"].replace(",", ";"),
            "job_id": proc.stdout.strip(),
            "sbatch_file": row["sbatch_file"],
        })
with jobs_file.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["group", "case_id", "targets", "job_id", "sbatch_file"])
    writer.writerows(rows)
print(json.dumps({"status": "submitted", "jobs": rows}, indent=2))
PY

echo "CADS15_SMOKE_ROOT=$OUT_ROOT"
echo "CADS15_SMOKE_JOBS=$JOBS_FILE"
