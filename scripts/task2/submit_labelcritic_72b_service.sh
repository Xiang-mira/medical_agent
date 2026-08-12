#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
STATE_ROOT=${STATE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state}
LABELCRITIC_SERVICE_ROOT=${LABELCRITIC_SERVICE_ROOT:-$STATE_ROOT/labelcritic_72b_service}
LABELCRITIC_PARTITION=${LABELCRITIC_PARTITION:-gpuh100}
LABELCRITIC_GRES=${LABELCRITIC_GRES:-gpu:H100:2}
LABELCRITIC_CPUS=${LABELCRITIC_CPUS:-16}
LABELCRITIC_MEM=${LABELCRITIC_MEM:-192G}
LABELCRITIC_TIME=${LABELCRITIC_TIME:-08:00:00}
LABELCRITIC_PORT=${LABELCRITIC_PORT:-8000}
LABELCRITIC_MODEL_ID=${LABELCRITIC_MODEL_ID:-Qwen/Qwen2-VL-72B-Instruct-AWQ}
LABELCRITIC_TENSOR_PARALLEL_SIZE=${LABELCRITIC_TENSOR_PARALLEL_SIZE:-2}
LABELCRITIC_MODEL_DIR=${LABELCRITIC_MODEL_DIR:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ}
VLLM_CONTAINER=${VLLM_CONTAINER:-/home/xhan74/containers/vllm-openai-v0.19.1.sif}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
WAIT_READY=${WAIT_READY:-1}
WAIT_READY_SEC=${WAIT_READY_SEC:-900}

mkdir -p "$LABELCRITIC_SERVICE_ROOT/logs"
SBATCH_FILE="$LABELCRITIC_SERVICE_ROOT/labelcritic_72b_service.sbatch"
ENDPOINT_URL_FILE="$LABELCRITIC_SERVICE_ROOT/endpoint.url"
ENDPOINT_HOST_FILE="$LABELCRITIC_SERVICE_ROOT/endpoint.host"
JOB_ID_FILE="$LABELCRITIC_SERVICE_ROOT/job_id.txt"

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=labelcritic_72b_service
#SBATCH --partition=$LABELCRITIC_PARTITION
#SBATCH --gres=$LABELCRITIC_GRES
#SBATCH --cpus-per-task=$LABELCRITIC_CPUS
#SBATCH --mem=$LABELCRITIC_MEM
#SBATCH --time=$LABELCRITIC_TIME
#SBATCH --output=$LABELCRITIC_SERVICE_ROOT/logs/labelcritic_72b_%j.out
#SBATCH --error=$LABELCRITIC_SERVICE_ROOT/logs/labelcritic_72b_%j.err

set -euo pipefail
cd "$CODE_ROOT"
host=\$(hostname -f 2>/dev/null || hostname)
printf "%s\n" "\$host" > "$ENDPOINT_HOST_FILE"
printf "http://%s:%s\n" "\$host" "$LABELCRITIC_PORT" > "$ENDPOINT_URL_FILE"
export NO_PROXY="\${NO_PROXY:-},127.0.0.1,localhost,\$host"
export no_proxy="\${no_proxy:-},127.0.0.1,localhost,\$host"
echo "[labelcritic] job=\${SLURM_JOB_ID:-local} host=\$host port=$LABELCRITIC_PORT model=$LABELCRITIC_MODEL_ID"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
apptainer exec --nv "$VLLM_CONTAINER" python -m vllm.entrypoints.openai.api_server \\
  --model "$LABELCRITIC_MODEL_DIR" \\
  --served-model-name "$LABELCRITIC_MODEL_ID" \\
  --tensor-parallel-size "$LABELCRITIC_TENSOR_PARALLEL_SIZE" \\
  --host 0.0.0.0 \\
  --port "$LABELCRITIC_PORT" \\
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \\
  --max-model-len "$VLLM_MAX_MODEL_LEN"
EOF

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for LabelCritic 72B service launch." >&2
  exit 2
fi

JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
printf "%s\n" "$JOB_ID" > "$JOB_ID_FILE"
echo "LABELCRITIC_JOB_ID=$JOB_ID"
echo "LABELCRITIC_ENDPOINT_URL_FILE=$ENDPOINT_URL_FILE"
echo "LABELCRITIC_ENDPOINT_HOST_FILE=$ENDPOINT_HOST_FILE"

if [ "$WAIT_READY" = "1" ]; then
  deadline=$((SECONDS + WAIT_READY_SEC))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -s "$ENDPOINT_URL_FILE" ]; then
      url=$(cat "$ENDPOINT_URL_FILE")
      if curl --noproxy "*" -fsS "$url/health" >/dev/null 2>&1; then
        echo "LABELCRITIC_BASE_URL=$url"
        exit 0
      fi
    fi
    sleep 10
  done
  echo "LabelCritic service submitted but health check did not pass within ${WAIT_READY_SEC}s." >&2
  echo "After it starts, export LABELCRITIC_BASE_URL=\$(cat $ENDPOINT_URL_FILE)" >&2
  exit 1
fi
