#!/bin/bash
# Template for running vLLM and the LabelCritic client inside one SLURM job.
set -euo pipefail

VLLM_PID=
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

echo "[scheduler] LabelCritic job=${SLURM_JOB_ID:-local}"
echo "[scheduler] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

: "${VLLM_CONTAINER:?VLLM_CONTAINER is required}"
: "${LABELCRITIC_MODEL_DIR:?LABELCRITIC_MODEL_DIR is required}"
: "${LABELCRITIC_MODEL_ID:?LABELCRITIC_MODEL_ID is required}"
: "${LABELCRITIC_TENSOR_PARALLEL_SIZE:?LABELCRITIC_TENSOR_PARALLEL_SIZE is required}"
LABELCRITIC_PORT="${LABELCRITIC_PORT:-8000}"

apptainer exec --nv "${VLLM_CONTAINER}" python -m vllm.entrypoints.openai.api_server \
  --model "${LABELCRITIC_MODEL_DIR}" \
  --served-model-name "${LABELCRITIC_MODEL_ID}" \
  --tensor-parallel-size "${LABELCRITIC_TENSOR_PARALLEL_SIZE}" \
  --host 127.0.0.1 \
  --port "${LABELCRITIC_PORT}" &
VLLM_PID=$!

for _ in $(seq 1 120); do
  curl --noproxy "*" -fsS "http://127.0.0.1:${LABELCRITIC_PORT}/health" >/dev/null && break
  sleep 5
done
curl --noproxy "*" -fsS "http://127.0.0.1:${LABELCRITIC_PORT}/health" >/dev/null

exec "$@"
