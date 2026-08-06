#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
SMOKE_ROOT=${SMOKE_ROOT:-${CADS15_SMOKE_ROOT:-}}
OUT_ROOT=${OUT_ROOT:-$OUT_PARENT/cads15_formal_100cases_$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-1}
RESUME=${RESUME:-0}
RETRY_FAILED=${RETRY_FAILED:-0}
CASE_ID=${CASE_ID:-}
MODELS=${MODELS:-}
GPU_PARTITION=${GPU_PARTITION:-${TASK2_CADS_GPU_PARTITION:-gpu}}
GPU_GRES=${GPU_GRES:-${TASK2_CADS_GPU_GRES:-gpu:T4:1}}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-${TASK2_CADS_GPU_CPUS:-8}}
GPU_MEM=${GPU_MEM:-${TASK2_CADS_GPU_MEMORY:-64G}}
GPU_TIME_LIMIT=${GPU_TIME_LIMIT:-${TASK2_CADS_GPU_TIME:-06:00:00}}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

if [ -z "$SMOKE_ROOT" ] && [ -f "$STATE_ROOT/.last_cads15_smoke" ]; then
  SMOKE_ROOT=$(cat "$STATE_ROOT/.last_cads15_smoke")
fi
if [ -z "$SMOKE_ROOT" ]; then
  echo "Set SMOKE_ROOT or run CADS15 smoke first; formal launcher requires PASSED smoke." >&2
  exit 2
fi

cd "$CODE_ROOT"
git fetch origin
if [ "$(git status --porcelain --untracked-files=no)" != "" ]; then
  echo "Refusing formal CADS15 launch with tracked worktree changes." >&2
  exit 2
fi
git checkout main
git pull --ff-only origin main

ARGS=(
  --output-root "$OUT_ROOT"
  --case-manifest "$CASE_MANIFEST"
  --smoke-root "$SMOKE_ROOT"
  --code-root "$CODE_ROOT"
  --python "$PYTHON"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE"
  --gpu-partition "$GPU_PARTITION"
  --gpu-gres "$GPU_GRES"
  --gpu-cpus-per-task "$GPU_CPUS_PER_TASK"
  --gpu-mem "$GPU_MEM"
  --gpu-time-limit "$GPU_TIME_LIMIT"
)
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)
[ "$RESUME" = "1" ] && ARGS+=(--resume)
[ "$RETRY_FAILED" = "1" ] && ARGS+=(--retry-failed)
[ -n "$CASE_ID" ] && ARGS+=(--case-id "$CASE_ID")
[ -n "$MODELS" ] && ARGS+=(--models "$MODELS")

"$PYTHON" tools/dataset_delivery/cads15_formal_launcher.py "${ARGS[@]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "CADS15_FORMAL_DRY_RUN_ROOT=$OUT_ROOT"
  echo "CADS15_FORMAL_PREFLIGHT=$OUT_ROOT/cads15_formal_preflight.json"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for formal CADS15 launch." >&2
  exit 2
fi
TASK_COUNT=$("$PYTHON" - "$OUT_ROOT/cads15_formal_preflight.json" <<'PY'
import json, sys
doc=json.load(open(sys.argv[1], encoding="utf-8"))
print(int(doc.get("task_count") or 0))
PY
)
if [ "$TASK_COUNT" -le 0 ]; then
  echo "No formal CADS15 tasks to submit." >&2
  exit 2
fi
PREFLIGHT_STATUS=$("$PYTHON" - "$OUT_ROOT/cads15_formal_preflight.json" <<'PY'
import json, sys
doc=json.load(open(sys.argv[1], encoding="utf-8"))
preflight=doc.get("resource_preflight") or {}
print(preflight.get("status") or "")
PY
)
if [ "$PREFLIGHT_STATUS" != "READY" ]; then
  echo "CADS15 formal Slurm resource preflight blocked submission: $PREFLIGHT_STATUS" >&2
  exit 2
fi
ARRAY_MAX=$((TASK_COUNT - 1))
JOB_ID=$(sbatch --parsable --array=0-"$ARRAY_MAX" "$OUT_ROOT/slurm/cads15_formal_array.sbatch")
printf "%s\n" "$JOB_ID" > "$OUT_ROOT/formal_job_id.txt"
tmp_last=$STATE_ROOT/.last_cads15_formal.tmp.$$
mkdir -p "$STATE_ROOT"
printf "%s\n" "$OUT_ROOT" > "$tmp_last"
mv "$tmp_last" "$STATE_ROOT/.last_cads15_formal"
echo "CADS15_FORMAL_ROOT=$OUT_ROOT"
echo "CADS15_FORMAL_JOB_ID=$JOB_ID"
