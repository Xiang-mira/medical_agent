#!/bin/bash
#SBATCH --job-name=teacher-missing-100
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Example only. Set account/partition/qos/gres/mem/time/array on submission.
# Suggested T4 shape:
# sbatch --partition=<gpu_partition> --gres=gpu:T4:1 --mem=48G --time=08:00:00 --array=0-99%4 slurm/teacher_missing_labels_100.sbatch

set -euo pipefail

: "${CODE_ROOT:?Set CODE_ROOT, e.g. /projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}"
: "${RUN_DIR:?Set RUN_DIR outside IMAGE_ROOT, MASK_ROOT, and CODE_ROOT}"
: "${CASES_100_MANIFEST:?Set CASES_100_MANIFEST}"
: "${ORGAN_GAP_RESOLUTION:?Set ORGAN_GAP_RESOLUTION}"
: "${TARGET_CONFIG:=configs/student_3d_prompt_target_organs.json}"
: "${MEDAI_TEACHER_MODELS:=epai_20250421,vsmtrans}"
: "${MEDAI_TEACHER_TIMEOUT_SEC:=1800}"
: "${PYTHON:=python}"

cd "$CODE_ROOT"

"$PYTHON" tools/dataset_delivery/run_teacher_case.py \
  --manifest "$CASES_100_MANIFEST" \
  --gap-file "$ORGAN_GAP_RESOLUTION" \
  --taxonomy "$TARGET_CONFIG" \
  --run-dir "$RUN_DIR" \
  --code-root "$CODE_ROOT" \
  --python "$PYTHON" \
  --models "$MEDAI_TEACHER_MODELS" \
  --target-config "$TARGET_CONFIG" \
  --timeout-sec "$MEDAI_TEACHER_TIMEOUT_SEC"

