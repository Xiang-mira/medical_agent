#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

HOST_PYTHON=${PYTHON:-python3}
CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
STATE_ROOT=${STATE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state}
LABELCRITIC_SERVICE_ROOT=${LABELCRITIC_SERVICE_ROOT:-$STATE_ROOT/labelcritic_72b_service}
LABELCRITIC_PARTITION=${LABELCRITIC_PARTITION:-gpuh100}
LABELCRITIC_GRES=${LABELCRITIC_GRES:-gpu:H100:2}
LABELCRITIC_ACCOUNT=${LABELCRITIC_ACCOUNT:-}
LABELCRITIC_QOS=${LABELCRITIC_QOS:-}
LABELCRITIC_CPUS=${LABELCRITIC_CPUS:-16}
LABELCRITIC_MEM=${LABELCRITIC_MEM:-192G}
LABELCRITIC_TIME=${LABELCRITIC_TIME:-08:00:00}
LABELCRITIC_PORT=${LABELCRITIC_PORT:-8000}
LABELCRITIC_MODEL_ID=${LABELCRITIC_MODEL_ID:-Qwen/Qwen2-VL-72B-Instruct-AWQ}
LABELCRITIC_TENSOR_PARALLEL_SIZE=${LABELCRITIC_TENSOR_PARALLEL_SIZE:-2}
LABELCRITIC_MODEL_DIR=${LABELCRITIC_MODEL_DIR:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ}
VLLM_CONTAINER=${VLLM_CONTAINER:-/home/xhan74/containers/vllm-openai-v0.19.1.sif}
VLLM_PYTHON=${VLLM_PYTHON:-python3}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
LABELCRITIC_STARTUP_TIMEOUT_SEC=${LABELCRITIC_STARTUP_TIMEOUT_SEC:-2400}
WAIT_READY=${WAIT_READY:-0}
WAIT_READY_SEC=${WAIT_READY_SEC:-900}

mkdir -p "$LABELCRITIC_SERVICE_ROOT/logs"
SBATCH_FILE="$LABELCRITIC_SERVICE_ROOT/labelcritic_72b_service.sbatch"
ENDPOINT_URL_FILE="$LABELCRITIC_SERVICE_ROOT/endpoint.url"
ENDPOINT_BASE_URL_FILE="$LABELCRITIC_SERVICE_ROOT/base_url.txt"
ENDPOINT_PORT_FILE="$LABELCRITIC_SERVICE_ROOT/port.txt"
ENDPOINT_HOST_FILE="$LABELCRITIC_SERVICE_ROOT/endpoint.host"
JOB_ID_FILE="$LABELCRITIC_SERVICE_ROOT/job_id.txt"
READY_FILE="$LABELCRITIC_SERVICE_ROOT/service_ready.json"
RUNTIME_PREFLIGHT_FILE="$LABELCRITIC_SERVICE_ROOT/labelcritic_runtime_preflight.json"
SPEC_HASH_FILE="$LABELCRITIC_SERVICE_ROOT/service_spec_hash.txt"
rm -f "$ENDPOINT_URL_FILE" "$ENDPOINT_BASE_URL_FILE" "$ENDPOINT_PORT_FILE" "$ENDPOINT_HOST_FILE" "$JOB_ID_FILE" "$READY_FILE" "$RUNTIME_PREFLIGHT_FILE"

"$HOST_PYTHON" tools/dataset_delivery/labelcritic_service_contract.py write-spec \
  --service-root "$LABELCRITIC_SERVICE_ROOT" \
  --generated-script "$SBATCH_FILE" >/dev/null
SERVICE_SPEC_HASH=$(cat "$SPEC_HASH_FILE")

ACCOUNT_LINE=""
QOS_LINE=""
[ -n "$LABELCRITIC_ACCOUNT" ] && ACCOUNT_LINE="#SBATCH --account=$LABELCRITIC_ACCOUNT"
[ -n "$LABELCRITIC_QOS" ] && QOS_LINE="#SBATCH --qos=$LABELCRITIC_QOS"

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=labelcritic_72b_service
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=$LABELCRITIC_PARTITION
$ACCOUNT_LINE
$QOS_LINE
#SBATCH --gres=$LABELCRITIC_GRES
#SBATCH --cpus-per-task=$LABELCRITIC_CPUS
#SBATCH --mem=$LABELCRITIC_MEM
#SBATCH --time=$LABELCRITIC_TIME
#SBATCH --output=$LABELCRITIC_SERVICE_ROOT/logs/labelcritic_72b_%j.out
#SBATCH --error=$LABELCRITIC_SERVICE_ROOT/logs/labelcritic_72b_%j.err

set -euo pipefail
export VLLM_PYTHON="$VLLM_PYTHON"
cd "$CODE_ROOT"
host=\$(hostname -f 2>/dev/null || hostname)
printf "%s\n" "\$host" > "$ENDPOINT_HOST_FILE"
printf "http://%s:%s\n" "\$host" "$LABELCRITIC_PORT" > "$ENDPOINT_URL_FILE"
printf "http://%s\n" "\$host" > "$ENDPOINT_BASE_URL_FILE"
printf "%s\n" "$LABELCRITIC_PORT" > "$ENDPOINT_PORT_FILE"
export NO_PROXY="\${NO_PROXY:-},127.0.0.1,localhost,\$host"
export no_proxy="\${no_proxy:-},127.0.0.1,localhost,\$host"
echo "[labelcritic] job=\${SLURM_JOB_ID:-local} host=\$host port=$LABELCRITIC_PORT model=$LABELCRITIC_MODEL_ID spec_hash=$SERVICE_SPEC_HASH"
runtime_status=PASSED
runtime_reason=
gpu_count=0
if ! command -v nvidia-smi >/dev/null 2>&1; then
  runtime_status=FAILED
  runtime_reason=nvidia_smi_not_found
else
  nvidia-smi || true
  gpu_count=\$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  if [ "\$gpu_count" -lt 2 ]; then
    runtime_status=FAILED
    runtime_reason=insufficient_visible_gpus
  fi
fi
if [ "\$runtime_status" = "PASSED" ]; then
  if ! apptainer exec --nv "$VLLM_CONTAINER" "\$VLLM_PYTHON" --version >/tmp/labelcritic_python_version_\${SLURM_JOB_ID:-local}.txt 2>/tmp/labelcritic_python_version_\${SLURM_JOB_ID:-local}.err; then
    runtime_status=FAILED
    runtime_reason=configured_vllm_python_missing
  elif ! apptainer exec --nv "$VLLM_CONTAINER" "\$VLLM_PYTHON" -c 'import vllm, importlib.util; raise SystemExit(0 if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is not None else 3)' >/tmp/labelcritic_vllm_import_\${SLURM_JOB_ID:-local}.txt 2>/tmp/labelcritic_vllm_import_\${SLURM_JOB_ID:-local}.err; then
    runtime_status=FAILED
    runtime_reason=vllm_or_api_server_import_failed
  fi
fi
python3 - <<RUNTIME_JSON
import json, os
payload = {
  "status": "\$runtime_status",
  "failure_reason": "\$runtime_reason",
  "job_id": os.environ.get("SLURM_JOB_ID", ""),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "nvidia_smi_gpu_count": int("\$gpu_count" or 0),
  "container": "$VLLM_CONTAINER",
  "python_executable": "$VLLM_PYTHON",
  "service_spec_hash": "$SERVICE_SPEC_HASH",
}
open("$RUNTIME_PREFLIGHT_FILE", "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\\n")
RUNTIME_JSON
if [ "\$runtime_status" != "PASSED" ]; then
  echo "FATAL: LabelCritic runtime preflight failed: \$runtime_reason" >&2
  exit 127
fi

apptainer exec --nv "$VLLM_CONTAINER" "\$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \\
  --model "$LABELCRITIC_MODEL_DIR" \\
  --served-model-name "$LABELCRITIC_MODEL_ID" \\
  --tensor-parallel-size "$LABELCRITIC_TENSOR_PARALLEL_SIZE" \\
  --host 0.0.0.0 \\
  --port "$LABELCRITIC_PORT" \\
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \\
  --max-model-len "$VLLM_MAX_MODEL_LEN" &
vllm_pid=\$!
deadline=\$((SECONDS + $LABELCRITIC_STARTUP_TIMEOUT_SEC))
ready=0
while [ "\$SECONDS" -lt "\$deadline" ]; do
  if ! kill -0 "\$vllm_pid" >/dev/null 2>&1; then
    wait "\$vllm_pid"
    exit_code=\$?
    echo "FATAL: vLLM exited before readiness with code \$exit_code" >&2
    exit "\$exit_code"
  fi
  if curl --noproxy "*" -fsS "http://127.0.0.1:$LABELCRITIC_PORT/v1/models" | grep -F "$LABELCRITIC_MODEL_ID" >/dev/null 2>&1; then
    ready=1
    python3 - <<READY_JSON
import json, os
payload = {
  "status": "READY",
  "job_id": os.environ.get("SLURM_JOB_ID", ""),
  "base_url": "http://\$host",
  "port": int("$LABELCRITIC_PORT"),
  "model": "$LABELCRITIC_MODEL_ID",
  "service_spec_hash": "$SERVICE_SPEC_HASH",
}
open("$READY_FILE", "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\\n")
READY_JSON
    break
  fi
  sleep 10
done
if [ "\$ready" != "1" ]; then
  echo "FATAL: vLLM did not become READY within $LABELCRITIC_STARTUP_TIMEOUT_SEC seconds" >&2
  kill "\$vllm_pid" >/dev/null 2>&1 || true
  wait "\$vllm_pid" >/dev/null 2>&1 || true
  exit 124
fi
wait "\$vllm_pid"
EOF

"$HOST_PYTHON" tools/dataset_delivery/labelcritic_service_contract.py preflight \
  --service-root "$LABELCRITIC_SERVICE_ROOT" \
  --generated-script "$SBATCH_FILE" >/dev/null

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for LabelCritic 72B service launch." >&2
  exit 2
fi

JOB_ID=$(sbatch --parsable --comment "medical_agent:labelcritic_72b:$SERVICE_SPEC_HASH" "$SBATCH_FILE")
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
