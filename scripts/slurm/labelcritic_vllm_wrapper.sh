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
VLLM_PYTHON="${VLLM_PYTHON:-python3}"
LABELCRITIC_PORT="${LABELCRITIC_PORT:-8000}"
LABELCRITIC_STARTUP_TIMEOUT_SEC="${LABELCRITIC_STARTUP_TIMEOUT_SEC:-2400}"
LABELCRITIC_PROJECT_BIND="${LABELCRITIC_PROJECT_BIND:-/projects/bodymaps/users/xhan74/medical_agent:/projects/bodymaps/users/xhan74/medical_agent}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found inside LabelCritic allocation" >&2
  exit 127
fi
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "FATAL: LabelCritic requires at least 2 visible GPUs, saw ${gpu_count}" >&2
  exit 127
fi
apptainer exec --nv --bind "${LABELCRITIC_PROJECT_BIND}" "${VLLM_CONTAINER}" "${VLLM_PYTHON}" --version
apptainer exec --nv --bind "${LABELCRITIC_PROJECT_BIND}" "${VLLM_CONTAINER}" "${VLLM_PYTHON}" -c 'import vllm, importlib.util; raise SystemExit(0 if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is not None else 3)'
python3 tools/dataset_delivery/labelcritic_service_contract.py model-files --container "${VLLM_CONTAINER}" --model-path "${LABELCRITIC_MODEL_DIR}" --python "${VLLM_PYTHON}" --output-json "${LABELCRITIC_SERVICE_ROOT:-.}/model_files_preflight.json"
python3 tools/dataset_delivery/labelcritic_service_contract.py gpu-topology --container "${VLLM_CONTAINER}" --python "${VLLM_PYTHON}" --expected-count 2 --expected-type H100 --tp "${LABELCRITIC_TENSOR_PARALLEL_SIZE}" --output-json "${LABELCRITIC_SERVICE_ROOT:-.}/gpu_topology_preflight.json"

apptainer exec --nv --bind "${LABELCRITIC_PROJECT_BIND}" "${VLLM_CONTAINER}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${LABELCRITIC_MODEL_DIR}" \
  --served-model-name "${LABELCRITIC_MODEL_ID}" \
  --tensor-parallel-size "${LABELCRITIC_TENSOR_PARALLEL_SIZE}" \
  --host 127.0.0.1 \
  --port "${LABELCRITIC_PORT}" &
VLLM_PID=$!

for _ in $(seq 1 $((LABELCRITIC_STARTUP_TIMEOUT_SEC / 5))); do
  curl --noproxy "*" -fsS "http://127.0.0.1:${LABELCRITIC_PORT}/health" >/dev/null && break
  sleep 5
done
curl --noproxy "*" -fsS "http://127.0.0.1:${LABELCRITIC_PORT}/health" >/dev/null

exec "$@"
