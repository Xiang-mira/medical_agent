#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_103cases_work/cases_103_manifest.csv}
BASE_MANIFEST=${BASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
OUT_PARENT=${OUT_PARENT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373}
STATE_ROOT=${STATE_ROOT:-$OUT_PARENT/runtime_state}
FORMAL_ROOT=${FORMAL_ROOT:-${TASK2_FORMAL_ROOT:-}}

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

if [ -z "$FORMAL_ROOT" ] && [ -f "$STATE_ROOT/.last_task2_formal" ]; then
  FORMAL_ROOT=$(cat "$STATE_ROOT/.last_task2_formal")
fi
if [ -z "$FORMAL_ROOT" ]; then
  echo '{"status":"NOT_SUBMITTED","workflow":"task2_formal_103cases"}'
  exit 0
fi

cd "$CODE_ROOT"
"$PYTHON" tools/dataset_delivery/task2_formal_validator.py \
  --output-root "$FORMAL_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --base-manifest "$BASE_MANIFEST"
