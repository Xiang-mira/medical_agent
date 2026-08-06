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
STATE_ROOT=${STATE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state}
TASK2_SMOKE_GROUPS=${TASK2_SMOKE_GROUPS:-atm,airrc,unest}
TASK2_SUBMIT_READY_GROUPS=${TASK2_SUBMIT_READY_GROUPS:-0}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:T4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-64G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
ACCOUNT=${ACCOUNT:-}
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
SUBMIT_READY_ARGS=()
if [ "$TASK2_SUBMIT_READY_GROUPS" = "1" ]; then
  SUBMIT_READY_ARGS+=(--submit-ready-groups)
fi
ACCOUNT_ARGS=()
if [ -n "$ACCOUNT" ]; then
  ACCOUNT_ARGS+=(--account "$ACCOUNT")
fi

"$PYTHON" tools/dataset_delivery/task2_smoke_launcher.py \
  --smoke-root "$OUT_ROOT" \
  --groups "$TASK2_SMOKE_GROUPS" \
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
  --time-limit "$TIME_LIMIT" \
  "${ACCOUNT_ARGS[@]}" \
  --run-slurm-test-only \
  --submit \
  --runtime-state-root "$STATE_ROOT" \
  "${SUBMIT_READY_ARGS[@]}"

echo "TASK2_SMOKE_ROOT=$OUT_ROOT"
echo "TASK2_SMOKE_JOBS=$OUT_ROOT/smoke_jobs.csv"
echo "TASK2_SMOKE_SUBMISSION_MANIFEST=$OUT_ROOT/submission_manifest.json"
