#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEARCH_SCRIPT="${PROJECT_ROOT}/tools/dataset_delivery/task2_fov_candidate_search.py"
HPC_DEFAULT_PYTHON="/home/xhan74/envs/medical_agent/bin/python"

IMAGE_ROOT="/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
MASK_ROOT="/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro"
OUTPUT_ROOT="/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/fov_candidate_search_$(date +%Y%m%d_%H%M%S)"
SCAN_LIMIT="2000"
MAX_DIRECTORIES_EXAMINED="50000"
PROGRESS_EVERY="50"
CHECKPOINT_EVERY="10"
DRY_RUN="${DRY_RUN:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEM="${MEM:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
JOB_NAME="${JOB_NAME:-task2_fov_search}"
PYTHON_FROM_CLI=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-executable)
      PYTHON_FROM_CLI="${2:?--python-executable requires a value}"
      shift 2
      ;;
    --image-root)
      IMAGE_ROOT="${2:?--image-root requires a value}"
      shift 2
      ;;
    --mask-root)
      MASK_ROOT="${2:?--mask-root requires a value}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a value}"
      shift 2
      ;;
    --scan-limit)
      SCAN_LIMIT="${2:?--scan-limit requires a value}"
      shift 2
      ;;
    --max-directories-examined)
      MAX_DIRECTORIES_EXAMINED="${2:?--max-directories-examined requires a value}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:?--progress-every requires a value}"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:?--checkpoint-every requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 64
      ;;
  esac
done

resolve_python() {
  if [[ -n "${PYTHON_FROM_CLI}" ]]; then
    printf '%s\n' "${PYTHON_FROM_CLI}"
  elif [[ -n "${TASK2_PYTHON_EXECUTABLE:-}" ]]; then
    printf '%s\n' "${TASK2_PYTHON_EXECUTABLE}"
  elif [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x "${HPC_DEFAULT_PYTHON}" ]]; then
    printf '%s\n' "${HPC_DEFAULT_PYTHON}"
  else
    command -v python3 || command -v python
  fi
}

TASK2_PYTHON_EXECUTABLE="$(resolve_python)"
if [[ ! -x "${TASK2_PYTHON_EXECUTABLE}" ]]; then
  echo "RUNTIME_PREFLIGHT_FAILED python_not_executable=${TASK2_PYTHON_EXECUTABLE}" >&2
  exit 70
fi

if ! "${TASK2_PYTHON_EXECUTABLE}" -c 'import sys,nibabel,numpy; print("TASK2_FOV_PREFLIGHT_OK", sys.executable, nibabel.__version__, numpy.__version__)'; then
  echo "RUNTIME_PREFLIGHT_FAILED python_executable=${TASK2_PYTHON_EXECUTABLE}" >&2
  exit 70
fi

mkdir -p "${OUTPUT_ROOT}"
SBATCH_SCRIPT="${OUTPUT_ROOT}/task2_fov_candidate_search.sbatch"
LOG_PATH="${OUTPUT_ROOT}/task2_fov_candidate_search.%j.log"

SBATCH_OPTIONS=(
  "#SBATCH --job-name=${JOB_NAME}"
  "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
  "#SBATCH --mem=${MEM}"
  "#SBATCH --time=${TIME_LIMIT}"
  "#SBATCH --output=${LOG_PATH}"
)
if [[ -n "${PARTITION}" ]]; then
  SBATCH_OPTIONS+=("#SBATCH --partition=${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_OPTIONS+=("#SBATCH --account=${ACCOUNT}")
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  for option in "${SBATCH_OPTIONS[@]}"; do
    printf '%s\n' "${option}"
  done
  printf '\n'
  printf 'export TASK2_PYTHON_EXECUTABLE=%q\n' "${TASK2_PYTHON_EXECUTABLE}"
  printf 'echo "TASK2_FOV_SEARCH_PYTHON=${TASK2_PYTHON_EXECUTABLE}"\n'
  printf '"${TASK2_PYTHON_EXECUTABLE}" -c %q\n' 'import sys,nibabel,numpy; print("TASK2_FOV_COMPUTE_PREFLIGHT_OK", sys.executable, nibabel.__version__, numpy.__version__)'
  printf 'exec "${TASK2_PYTHON_EXECUTABLE}" %q \\\n' "${SEARCH_SCRIPT}"
  printf '  --image-root %q \\\n' "${IMAGE_ROOT}"
  printf '  --mask-root %q \\\n' "${MASK_ROOT}"
  printf '  --output-root %q \\\n' "${OUTPUT_ROOT}"
  printf '  --scan-limit %q \\\n' "${SCAN_LIMIT}"
  printf '  --max-directories-examined %q \\\n' "${MAX_DIRECTORIES_EXAMINED}"
  printf '  --progress-every %q \\\n' "${PROGRESS_EVERY}"
  printf '  --checkpoint-every %q\n' "${CHECKPOINT_EVERY}"
} > "${SBATCH_SCRIPT}"

echo "TASK2_FOV_SEARCH_OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TASK2_FOV_SEARCH_SBATCH=${SBATCH_SCRIPT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${SBATCH_SCRIPT}"
  exit 0
fi

sbatch "${SBATCH_SCRIPT}"
