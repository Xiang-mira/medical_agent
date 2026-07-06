#!/usr/bin/env bash
set -euo pipefail

# Screen-friendly launcher for the Round2 +10 experiment chain.
#
# Usage inside screen:
#   cd /home/teacher1/JHU-project1/medical_agent
#   bash scripts/run_round2_plus10_chain_screen.sh
#
# If you already have a safe teacher-cache command for the selected new 10
# cases, pass it through TEACHER_CACHE_CMD. Example:
#   TEACHER_CACHE_CMD='python scripts/<your_teacher_cache_builder>.py ...' \
#     bash scripts/run_round2_plus10_chain_screen.sh
#
# Without TEACHER_CACHE_CMD, the chain deliberately stops after writing
# teacher_cache_required.json so it does not fake a formal Round2.

ROOT="/home/teacher1/JHU-project1/medical_agent"
CHAIN_ROOT="${CHAIN_ROOT:-${ROOT}/outputs/screen_experiment_chains/round2_plus10_$(date -u +%Y%m%dT%H%M%SZ)}"
CASE_PLAN_DIR="${CASE_PLAN_DIR:-${ROOT}/outputs/round2_plus10_case_plan_20260705}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/em_round2_plus10_20260705}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${ROOT}/outputs/em_round_pure_cached_10case_formal_lite_20260703}"
LOG_DIR="${CHAIN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/screen_chain.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

CMD=(
  python scripts/run_screen_experiment_chain.py
  --chain-root "${CHAIN_ROOT}"
  --case-plan-dir "${CASE_PLAN_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --baseline-run-root "${BASELINE_RUN_ROOT}"
  --max-retries "${MAX_RETRIES:-2}"
  --retry-sleep-sec "${RETRY_SLEEP_SEC:-60}"
)

if [[ -n "${TEACHER_CACHE_CMD:-}" ]]; then
  CMD+=(--teacher-cache-cmd "${TEACHER_CACHE_CMD}")
fi

{
  echo "==== Round2 +10 screen chain started: $(date -Is) ===="
  echo "CHAIN_ROOT=${CHAIN_ROOT}"
  echo "CASE_PLAN_DIR=${CASE_PLAN_DIR}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "BASELINE_RUN_ROOT=${BASELINE_RUN_ROOT}"
  if [[ -n "${TEACHER_CACHE_CMD:-}" ]]; then
    echo "TEACHER_CACHE_CMD=${TEACHER_CACHE_CMD}"
  else
    echo "TEACHER_CACHE_CMD is empty; chain will stop safely at teacher-cache gate."
  fi
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
  rc=$?
  echo "==== Round2 +10 screen chain finished: $(date -Is), rc=${rc} ===="
  exit "${rc}"
} 2>&1 | tee -a "${LOG_FILE}"
