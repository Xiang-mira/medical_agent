#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/JHU-project1/medical_agent

export PYTHONUNBUFFERED=1
export PYTHONPATH="agent-harness:${PYTHONPATH:-}"
export MEDAI_FORMAL_LITE_RUN_ROOT="${MEDAI_FORMAL_LITE_RUN_ROOT:-/home/teacher1/JHU-project1/medical_agent/outputs/em_round_pure_cached_10case_formal_lite_20260703}"

log_dir="${MEDAI_FORMAL_LITE_RUN_ROOT}/logs"
log_file="${log_dir}/formal_lite_pipeline.log"
mkdir -p "${log_dir}"

{
  echo "==== Pure cached 10-case formal-lite EM pipeline started: $(date -Is) ===="
  echo "cwd=$(pwd)"
  echo "run_root=${MEDAI_FORMAL_LITE_RUN_ROOT}"
  python scripts/run_pure_cached_10case_formal_lite_em.py
  rc=$?
  echo "==== Pure cached 10-case formal-lite EM pipeline finished: $(date -Is), rc=${rc} ===="
  exit "${rc}"
} 2>&1 | tee -a "${log_file}"
