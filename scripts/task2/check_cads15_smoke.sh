#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
SMOKE_ROOT=${SMOKE_ROOT:-${CADS15_SMOKE_ROOT:-}}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

if [ -z "$SMOKE_ROOT" ]; then
  if [ -f "$STATE_ROOT/.last_cads15_smoke" ]; then
    SMOKE_ROOT=$(cat "$STATE_ROOT/.last_cads15_smoke")
  else
    latest=$(ls -td "$OUT_PARENT"/cads15_teacher_smoke_* 2>/dev/null | head -1 || true)
    SMOKE_ROOT=$latest
  fi
fi

cd "$CODE_ROOT"
SACCT_ARG=()
if ! command -v sacct >/dev/null 2>&1; then
  SACCT_ARG=(--no-sacct)
fi

if [ -z "${SMOKE_ROOT:-}" ]; then
  "$PYTHON" tools/dataset_delivery/cads15_smoke_status.py "${SACCT_ARG[@]}"
else
  "$PYTHON" tools/dataset_delivery/cads15_smoke_status.py \
    --smoke-root "$SMOKE_ROOT" \
    "${SACCT_ARG[@]}"
fi
