#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
STATE_ROOT=${STATE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812}
FORMAL_ROOT=${FORMAL_ROOT:-$STATE_ROOT/round1_orchestrated/full_373_multiteacher_round1}
CASE_MANIFEST=${CASE_MANIFEST:-$WORKSPACE_ROOT/manifests/cases_103_manifest.csv}
BASE_MANIFEST=${BASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
UNEST_PYTHON_EXECUTABLE=${UNEST_PYTHON_EXECUTABLE:-/home/xhan74/envs/medical_agent_train_py311/bin/python}
ROUND1_RUN_ID=${ROUND1_RUN_ID:-round1_4995446918298643158}
CONTROLLER_MEM=${CONTROLLER_MEM:-32G}
export ROUND1_RUN_ID CONTROLLER_MEM

cd "$CODE_ROOT"
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:-$(git rev-parse HEAD)}
export EXPECTED_GIT_COMMIT

exec "$PYTHON" tools/dataset_delivery/task2_round1_orchestrator.py resume-formal \
  --state-root "$STATE_ROOT" \
  --workspace-root "$WORKSPACE_ROOT" \
  --formal-root "$FORMAL_ROOT" \
  --scientific-run-id "$ROUND1_RUN_ID" \
  --case-manifest "$CASE_MANIFEST" \
  --base-manifest "$BASE_MANIFEST" \
  --python "$PYTHON" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --unest-python-executable "$UNEST_PYTHON_EXECUTABLE" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  "$@"
