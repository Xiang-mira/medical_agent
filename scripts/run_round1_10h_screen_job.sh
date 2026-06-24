#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/JHU-project1/medical_agent

ROUND_ROOT="outputs/stage4b_round1_50cases_20260611/round1"
LOG_DIR="${ROUND_ROOT}/mstep/round1_10h_patch_experiments"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/round1_10h_screen_job.log"

{
  echo "==== Round1 10h repair experiment started: $(date -Is) ===="
  echo "cwd=$(pwd)"
  echo "round_root=${ROUND_ROOT}"
  echo "log_file=${LOG_FILE}"
  echo

  echo "---- Step 1/4: rebuild manifest and audit ----"
  python scripts/run_round1_10h_patch_experiments.py \
    --round-root "${ROUND_ROOT}" \
    --rebuild-manifest

  echo
  echo "---- Step 2/4: short pilot M-step checkpoint ----"
  python scripts/train_voxtell_prompt_student.py \
    --manifest "${ROUND_ROOT}/mstep/voxtell_prompt_student_manifest.json" \
    --model-dir checkpoints/VoxTell/voxtell_v1.1 \
    --text-encoding-model checkpoints/Qwen/Qwen3-Embedding-4B \
    --output-dir "${ROUND_ROOT}/mstep" \
    --epochs 1 \
    --max-items 24 \
    --max-steps 2 \
    --freeze-encoder \
    --device cuda

  echo
  echo "---- Step 3/4: automatic sanity and mini quality gates ----"
  python scripts/run_round1_10h_patch_experiments.py \
    --round-root "${ROUND_ROOT}" \
    --run-existing-gates

  echo
  echo "---- Step 4/4: prompt robustness and negative suppression ----"
  python scripts/run_round1_10h_patch_experiments.py \
    --round-root "${ROUND_ROOT}" \
    --run-prompt-robustness \
    --run-negative-suppression \
    --max-robustness-cases 1 \
    --max-negative-cases 1 \
    --max-negatives-per-source 3

  echo
  echo "==== Round1 10h repair experiment finished: $(date -Is) ===="
} 2>&1 | tee "${LOG_FILE}"
