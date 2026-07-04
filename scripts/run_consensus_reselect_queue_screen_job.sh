#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/JHU-project1/medical_agent

export PYTHONUNBUFFERED=1
export PYTHONPATH="agent-harness:${PYTHONPATH:-}"
export MEDAI_CONSENSUS_QUEUE_ROOT="${MEDAI_CONSENSUS_QUEUE_ROOT:-/home/teacher1/JHU-project1/medical_agent/outputs/consensus_reselect_queue_20260703}"

log_dir="${MEDAI_CONSENSUS_QUEUE_ROOT}/logs"
log_file="${log_dir}/consensus_reselect_queue.log"
mkdir -p "${log_dir}"

{
  echo "==== Consensus reselect experiment queue started: $(date -Is) ===="
  echo "cwd=$(pwd)"
  echo "queue_root=${MEDAI_CONSENSUS_QUEUE_ROOT}"
  echo "log_file=${log_file}"
  echo
  python scripts/consensus_reselect_queue_20260703.py
  rc=$?
  echo
  echo "==== Consensus reselect experiment queue finished: $(date -Is), rc=${rc} ===="
  exit "${rc}"
} 2>&1 | tee -a "${log_file}"
