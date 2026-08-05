#!/usr/bin/env bash
set -euo pipefail

: "${CODE_ROOT:?CODE_ROOT is required}"
: "${PYTHON:?PYTHON is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${CASE_MANIFEST:?CASE_MANIFEST is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"

cd "$CODE_ROOT"
mkdir -p "$OUT_ROOT"

"$PYTHON" tools/dataset_delivery/propose_task1_finalization.py \
  --data-root "$DATA_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --output-root "$OUT_ROOT" \
  --mapping configs/dataset_delivery/task1/organ_rename_mapping_373.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --boundary-classification configs/dataset_delivery/task1/task_boundary_classification.csv \
  --alias-groups configs/dataset_delivery/task1/task1_alias_groups.csv \
  --task2-targets configs/dataset_delivery/task1/task2_generate_targets_23.csv \
  --semantic-evidence configs/dataset_delivery/task1/taxonomy_semantic_evidence.csv

echo "TASK1_PROPOSAL_AUDIT_OUTPUT=$OUT_ROOT"
