#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
BASE_MANIFEST=${BASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
FORMAL_APPEND_CASES=${FORMAL_APPEND_CASES:-configs/dataset_delivery/task2_formal_append_cases.csv}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_103cases_work/cases_103_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
UNEST_PYTHON_EXECUTABLE=${UNEST_PYTHON_EXECUTABLE:-/home/xhan74/envs/medical_agent_train_py311/bin/python}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
FORMAL_OUT_ROOT=${FORMAL_OUT_ROOT:-$OUT_PARENT/formal_task2_22targets_103cases_$(date +%Y%m%d_%H%M%S)}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
TARGET_CONFIG=${TARGET_CONFIG:-configs/student_3d_prompt_target_organs.json}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:T4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-64G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
GROUPS=${GROUPS:-cads,atm,airrc,unest}
CADS_CONCURRENCY=${CADS_CONCURRENCY:-4}
ATM_CONCURRENCY=${ATM_CONCURRENCY:-2}
AIRRC_CONCURRENCY=${AIRRC_CONCURRENCY:-2}
UNEST_CONCURRENCY=${UNEST_CONCURRENCY:-2}
DRY_RUN=${DRY_RUN:-1}
RESUME=${RESUME:-0}
RETRY_FAILED=${RETRY_FAILED:-0}
CASE_ID=${CASE_ID:-}
RUN_TESTS_BEFORE_LAUNCH=${RUN_TESTS_BEFORE_LAUNCH:-0}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

cd "$CODE_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  echo "submit_task2_formal_103cases.sh must run on main, got $BRANCH" >&2
  exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Refusing formal Task2 launch with tracked worktree changes." >&2
  git status --porcelain --untracked-files=no >&2
  exit 2
fi

MANIFEST_WORK_ROOT=$(dirname "$CASE_MANIFEST")
mkdir -p "$MANIFEST_WORK_ROOT"
"$PYTHON" tools/dataset_delivery/task2_formal_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --append-cases "$FORMAL_APPEND_CASES" \
  --output-manifest "$CASE_MANIFEST" \
  --audit-json "$MANIFEST_WORK_ROOT/cases_103_manifest.audit.json" \
  --audit-csv "$MANIFEST_WORK_ROOT/cases_103_manifest.audit.csv"

"$PYTHON" -m py_compile \
  tools/dataset_delivery/task2_formal_manifest.py \
  tools/dataset_delivery/task2_formal_launcher.py \
  tools/dataset_delivery/task2_formal_validator.py \
  tools/dataset_delivery/task2_preflight.py \
  tools/dataset_delivery/task2_smoke_validator.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py

if [ "$RUN_TESTS_BEFORE_LAUNCH" = "1" ]; then
  "$PYTHON" -m pytest -q \
    tests/dataset_delivery/test_task2_runtime_repair.py \
    tests/dataset_delivery/test_task2_formal_production.py
fi

ARGS=(
  --output-root "$FORMAL_OUT_ROOT"
  --case-manifest "$CASE_MANIFEST"
  --base-manifest "$BASE_MANIFEST"
  --code-root "$CODE_ROOT"
  --python "$PYTHON"
  --registry "$REGISTRY"
  --target-config "$TARGET_CONFIG"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE"
  --unest-python-executable "$UNEST_PYTHON_EXECUTABLE"
  --groups "$GROUPS"
  --gpu-partition "$PARTITION"
  --gpu-gres "$GRES"
  --gpu-cpus-per-task "$CPUS_PER_TASK"
  --gpu-mem "$MEM"
  --gpu-time-limit "$TIME_LIMIT"
  --timeout-sec "$TIMEOUT_SEC"
)
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)
[ "$RESUME" = "1" ] && ARGS+=(--resume)
[ "$RETRY_FAILED" = "1" ] && ARGS+=(--retry-failed)
[ -n "$CASE_ID" ] && ARGS+=(--case-id "$CASE_ID")

"$PYTHON" tools/dataset_delivery/task2_formal_launcher.py "${ARGS[@]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "TASK2_FORMAL_DRY_RUN_ROOT=$FORMAL_OUT_ROOT"
  echo "TASK2_FORMAL_PREFLIGHT=$FORMAL_OUT_ROOT/formal_task2_preflight.json"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for formal Task2 launch." >&2
  exit 2
fi

SUMMARY="$FORMAL_OUT_ROOT/formal_task2_submission_manifest.json"
READY=$("$PYTHON" - "$SUMMARY" <<'PY'
import json, sys
doc=json.load(open(sys.argv[1], encoding="utf-8"))
print(doc.get("status") or "")
PY
)
if [ "$READY" != "READY" ]; then
  echo "Formal Task2 preflight is not READY: $READY" >&2
  exit 2
fi

mkdir -p "$FORMAL_OUT_ROOT/slurm"
echo "model_group,job_id,task_count,sbatch_file" > "$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"

submit_group() {
  local group=$1
  local concurrency=$2
  local task_count
  local sbatch_file
  task_count=$("$PYTHON" - "$SUMMARY" "$group" <<'PY'
import json, sys
doc=json.load(open(sys.argv[1], encoding="utf-8"))
print(int((doc.get("groups") or {}).get(sys.argv[2], {}).get("task_count") or 0))
PY
)
  sbatch_file=$("$PYTHON" - "$SUMMARY" "$group" <<'PY'
import json, sys
doc=json.load(open(sys.argv[1], encoding="utf-8"))
print((doc.get("groups") or {}).get(sys.argv[2], {}).get("sbatch_file") or "")
PY
)
  if [ "$task_count" -le 0 ]; then
    echo "$group: no planned tasks, skipping"
    return
  fi
  local max_index=$((task_count - 1))
  local job_id
  job_id=$(sbatch --parsable --array=0-"$max_index"%"$concurrency" "$sbatch_file")
  echo "$group,$job_id,$task_count,$sbatch_file" >> "$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"
  echo "${group}_JOB_ID=$job_id"
}

submit_group cads "$CADS_CONCURRENCY"
submit_group atm "$ATM_CONCURRENCY"
submit_group airrc "$AIRRC_CONCURRENCY"
submit_group unest "$UNEST_CONCURRENCY"

tmp_last=$STATE_ROOT/.last_task2_formal.tmp.$$
mkdir -p "$STATE_ROOT"
printf "%s\n" "$FORMAL_OUT_ROOT" > "$tmp_last"
mv "$tmp_last" "$STATE_ROOT/.last_task2_formal"
echo "TASK2_FORMAL_ROOT=$FORMAL_OUT_ROOT"
echo "TASK2_FORMAL_JOBS=$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"
