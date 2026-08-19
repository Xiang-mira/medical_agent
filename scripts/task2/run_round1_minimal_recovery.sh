#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
STATE_ROOT=${STATE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state}
FORMAL_ROOT=${FORMAL_ROOT:-$STATE_ROOT/round1_orchestrated/full_373_multiteacher_round1}
ROUND1_RUN_ID=${ROUND1_RUN_ID:-round1_4995446918298643158}
GPU_TARGET_WORKERS=2
GPU_OVERREQUEST_WORKERS=2
export GPU_TARGET_WORKERS GPU_OVERREQUEST_WORKERS

cd "$CODE_ROOT"
exec "$PYTHON" tools/dataset_delivery/task2_round1_orchestrator.py minimal-recovery \
  --state-root "$STATE_ROOT" \
  --formal-root "$FORMAL_ROOT" \
  --scientific-run-id "$ROUND1_RUN_ID" \
  --python "$PYTHON" \
  "$@"
