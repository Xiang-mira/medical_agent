#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812}
BASE_MANIFEST=${BASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
FORMAL_APPEND_CASES=${FORMAL_APPEND_CASES:-configs/dataset_delivery/task2_formal_append_cases.csv}
CASE_MANIFEST=${CASE_MANIFEST:-$WORKSPACE_ROOT/manifests/cases_103_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
UNEST_PYTHON_EXECUTABLE=${UNEST_PYTHON_EXECUTABLE:-/home/xhan74/envs/medical_agent_train_py311/bin/python}
LABELCRITIC_BASE_URL=${LABELCRITIC_BASE_URL:-}
LABELCRITIC_PORT=${LABELCRITIC_PORT:-8000}
LABELCRITIC_MODEL_ID=${LABELCRITIC_MODEL_ID:-Qwen/Qwen2-VL-72B-Instruct-AWQ}
MEDAI_FORMAL_LABELCRITIC_72B_SELECTION_READY=${MEDAI_FORMAL_LABELCRITIC_72B_SELECTION_READY:-1}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
LABELCRITIC_SERVICE_ROOT=${LABELCRITIC_SERVICE_ROOT:-}
FORMAL_OUT_ROOT=${FORMAL_OUT_ROOT:-$WORKSPACE_ROOT/teacher/formal_task2_22targets_103cases_$(date +%Y%m%d_%H%M%S)}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
TARGET_CONFIG=${TARGET_CONFIG:-configs/student_3d_prompt_target_organs.json}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:1}
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
RUNTIME_NO_GIT=${RUNTIME_NO_GIT:-0}
SKIP_GIT_SYNC=${SKIP_GIT_SYNC:-0}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-}
DYNAMIC_GPU_SCHEDULER=${DYNAMIC_GPU_SCHEDULER:-1}
GPU_TARGET_WORKERS=${GPU_TARGET_WORKERS:-30}
GPU_OVERREQUEST_WORKERS=${GPU_OVERREQUEST_WORKERS:-}
GPU_GROUP_WEIGHTS=${GPU_GROUP_WEIGHTS:-cads=0.45,atm=0.15,airrc=0.20,unest=0.20}
DYNAMIC_SBATCH_TEST_ONLY=${DYNAMIC_SBATCH_TEST_ONLY:-1}
REQUIRE_LABELCRITIC_HEALTH=${REQUIRE_LABELCRITIC_HEALTH:-1}
WAIT_LABELCRITIC_SEC=${WAIT_LABELCRITIC_SEC:-600}
ALLOW_LOCALHOST_LABELCRITIC=${ALLOW_LOCALHOST_LABELCRITIC:-0}
STAGE_WORKSPACE=${STAGE_WORKSPACE:-0}
SOURCE_CASE_MANIFEST=${SOURCE_CASE_MANIFEST:-}
IMAGE_ROOT=${IMAGE_ROOT:-/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro}
MASK_ROOT=${MASK_ROOT:-/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro}
if [ -z "${GPU_PROFILE_SPECS:-}" ]; then
  GPU_PROFILE_SPECS='generic_gpu|gpu|gpu:1|8|64G|06:00:00'
fi

LABELCRITIC_SERVICE_ROOT=${LABELCRITIC_SERVICE_ROOT:-$STATE_ROOT/labelcritic_72b_service}
if [ -z "${LABELCRITIC_BASE_URL:-}" ] && [ -s "$LABELCRITIC_SERVICE_ROOT/base_url.txt" ]; then
  LABELCRITIC_BASE_URL=$(cat "$LABELCRITIC_SERVICE_ROOT/base_url.txt")
fi
if [ -s "$LABELCRITIC_SERVICE_ROOT/port.txt" ]; then
  LABELCRITIC_PORT=$(cat "$LABELCRITIC_SERVICE_ROOT/port.txt")
fi
if [ -s "$LABELCRITIC_SERVICE_ROOT/endpoint.host" ]; then
  LABELCRITIC_HOST=$(cat "$LABELCRITIC_SERVICE_ROOT/endpoint.host")
  export NO_PROXY="${NO_PROXY:-},$LABELCRITIC_HOST"
  export no_proxy="${no_proxy:-},$LABELCRITIC_HOST"
fi
LABELCRITIC_BASE_URL=${LABELCRITIC_BASE_URL:-http://localhost}
MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC=${MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC:-$WAIT_LABELCRITIC_SEC}
export LABELCRITIC_BASE_URL LABELCRITIC_PORT LABELCRITIC_SERVICE_ROOT LABELCRITIC_MODEL_ID MEDAI_FORMAL_LABELCRITIC_72B_SELECTION_READY WAIT_LABELCRITIC_SEC MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC

cd "$CODE_ROOT"
if [ "$RUNTIME_NO_GIT" != "1" ] && [ "$SKIP_GIT_SYNC" != "1" ]; then
  git fetch origin
  git switch main
  git pull --ff-only origin main
fi
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  echo "submit_task2_formal_103cases.sh must run on main, got $BRANCH" >&2
  exit 2
fi
CURRENT_GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$EXPECTED_GIT_COMMIT" ] && [ "$CURRENT_GIT_COMMIT" != "$EXPECTED_GIT_COMMIT" ]; then
  echo "submit_task2_formal_103cases.sh expected git commit $EXPECTED_GIT_COMMIT, got $CURRENT_GIT_COMMIT" >&2
  exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Refusing formal Task2 launch with tracked worktree changes." >&2
  git status --porcelain --untracked-files=no >&2
  exit 2
fi

MANIFEST_WORK_ROOT=$(dirname "$CASE_MANIFEST")
mkdir -p "$MANIFEST_WORK_ROOT"
if [ "$STAGE_WORKSPACE" = "1" ]; then
  if [ -z "$SOURCE_CASE_MANIFEST" ]; then
    SOURCE_CASE_MANIFEST="$MANIFEST_WORK_ROOT/cases_103_source_manifest.csv"
    "$PYTHON" tools/dataset_delivery/task2_formal_manifest.py \
      --base-manifest "$BASE_MANIFEST" \
      --append-cases "$FORMAL_APPEND_CASES" \
      --output-manifest "$SOURCE_CASE_MANIFEST" \
      --audit-json "$MANIFEST_WORK_ROOT/cases_103_source_manifest.audit.json" \
      --audit-csv "$MANIFEST_WORK_ROOT/cases_103_source_manifest.audit.csv"
  fi
  "$PYTHON" tools/dataset_delivery/task2_workspace_staging.py \
    --cases-manifest "$SOURCE_CASE_MANIFEST" \
    --workspace-root "$WORKSPACE_ROOT" \
    --image-root "$IMAGE_ROOT" \
    --mask-root "$MASK_ROOT" \
    --base-manifest "$BASE_MANIFEST" \
    --resume
elif [ ! -f "$CASE_MANIFEST" ]; then
  "$PYTHON" tools/dataset_delivery/task2_formal_manifest.py \
    --base-manifest "$BASE_MANIFEST" \
    --append-cases "$FORMAL_APPEND_CASES" \
    --output-manifest "$CASE_MANIFEST" \
    --audit-json "$MANIFEST_WORK_ROOT/cases_103_manifest.audit.json" \
    --audit-csv "$MANIFEST_WORK_ROOT/cases_103_manifest.audit.csv"
fi

"$PYTHON" -m py_compile \
  tools/dataset_delivery/task2_formal_manifest.py \
  tools/dataset_delivery/task2_formal_launcher.py \
  tools/dataset_delivery/task2_formal_validator.py \
  tools/dataset_delivery/task2_manifest_masks.py \
  tools/dataset_delivery/task2_preflight.py \
  tools/dataset_delivery/task2_recovery.py \
  tools/dataset_delivery/task2_workspace_staging.py \
  tools/dataset_delivery/task2_dynamic_gpu_submitter.py \
  tools/dataset_delivery/formal_round1_preflight.py \
  tools/dataset_delivery/labelcritic_resource_preflight.py \
  tools/dataset_delivery/task2_smoke_validator.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py

"$PYTHON" tools/dataset_delivery/formal_round1_preflight.py \
  --workspace-root "$WORKSPACE_ROOT" \
  --cases-manifest "$CASE_MANIFEST" \
  --base-manifest "$BASE_MANIFEST" \
  --code-root "$CODE_ROOT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-json "$WORKSPACE_ROOT/manifests/formal_round1_preflight.json"

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

if [ "$REQUIRE_LABELCRITIC_HEALTH" = "1" ]; then
  "$PYTHON" - "$LABELCRITIC_BASE_URL" "$LABELCRITIC_PORT" "$LABELCRITIC_MODEL_ID" "$WAIT_LABELCRITIC_SEC" "$ALLOW_LOCALHOST_LABELCRITIC" <<'PY'
import json
import re
import sys
import time
import urllib.error
import urllib.request

base_url, port, expected_model, wait_sec, allow_localhost = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
base = re.sub(r"/v1/?$", "", (base_url or "http://localhost").rstrip("/"))
match = re.match(r"^(https?://[^/:]+):(\d+)$", base)
if match:
    base, port = match.group(1), int(match.group(2))
host = re.sub(r"^https?://", "", base).split("/", 1)[0]
if host in {"localhost", "127.0.0.1"} and allow_localhost != "1":
    raise SystemExit(
        "Formal Task2 refuses localhost LabelCritic. Start scripts/task2/submit_labelcritic_72b_service.sh "
        "and let this script read STATE_ROOT/labelcritic_72b_service/base_url.txt, or set ALLOW_LOCALHOST_LABELCRITIC=1 only for same-node debug."
    )
deadline = time.time() + max(1, wait_sec)
last_error = ""
models = []
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"{base}:{port}/health", timeout=10).read()
        payload = urllib.request.urlopen(f"{base}:{port}/v1/models", timeout=20).read()
        models = [str(item.get("id", "")) for item in json.loads(payload.decode("utf-8")).get("data", [])]
        if expected_model in models:
            print(json.dumps({"status": "LABELCRITIC_READY", "base_url": base, "port": port, "model": expected_model}))
            raise SystemExit(0)
        last_error = f"served_model_mismatch expected={expected_model} served={models}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(10)
raise SystemExit(f"Formal LabelCritic 72B health/model check failed after {wait_sec}s: {last_error}")
PY
fi

mkdir -p "$FORMAL_OUT_ROOT/slurm"
echo "model_group,job_id,task_count,sbatch_file" > "$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"

if [ "$DYNAMIC_GPU_SCHEDULER" = "1" ]; then
  DYNAMIC_ARGS=(
    --summary "$SUMMARY"
    --output-root "$FORMAL_OUT_ROOT"
    --state-root "$STATE_ROOT"
    --target-workers "$GPU_TARGET_WORKERS"
    --profile-specs "$GPU_PROFILE_SPECS"
    --groups "$GROUPS"
    --group-weights "$GPU_GROUP_WEIGHTS"
  )
  [ -n "$GPU_OVERREQUEST_WORKERS" ] && DYNAMIC_ARGS+=(--overrequest-workers "$GPU_OVERREQUEST_WORKERS")
  [ "$DYNAMIC_SBATCH_TEST_ONLY" != "1" ] && DYNAMIC_ARGS+=(--skip-sbatch-test-only)
  "$PYTHON" tools/dataset_delivery/task2_dynamic_gpu_submitter.py "${DYNAMIC_ARGS[@]}"
  echo "TASK2_FORMAL_ROOT=$FORMAL_OUT_ROOT"
  echo "TASK2_FORMAL_JOBS=$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"
  echo "TASK2_FORMAL_DYNAMIC_GPU_PLAN=$FORMAL_OUT_ROOT/slurm/dynamic_gpu_submission_plan.json"
  exit 0
fi

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
