#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
PREFERRED_CASE_ID=${PREFERRED_CASE_ID:-BDMAP_00000120}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:t4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-48G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/cads_remaining8_strict_smoke_$(date +%Y%m%d_%H%M%S)}

cd "$CODE_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main

SLURM_DIR=$OUT_ROOT/slurm
mkdir -p "$SLURM_DIR"
SBATCH_FILE=$SLURM_DIR/cads_remaining8_strict_smoke.sbatch

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=cads_rem8_smoke
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_DIR}/cads_remaining8_smoke_%j.out
#SBATCH --error=${SLURM_DIR}/cads_remaining8_smoke_%j.err

set -euo pipefail
if [ -f "\$HOME/.bodymaps_env" ]; then
  source "\$HOME/.bodymaps_env"
fi
export CODE_ROOT="$CODE_ROOT"
export PYTHON="$PYTHON"
export CASE_MANIFEST="$CASE_MANIFEST"
export PREFERRED_CASE_ID="$PREFERRED_CASE_ID"
export OUT_ROOT="$OUT_ROOT"
export REGISTRY="$REGISTRY"
export TIMEOUT_SEC="${TIMEOUT_SEC:-14400}"
bash "$CODE_ROOT/scripts/run_cads_remaining8_strict_delivery_smoke_hpc.sh"
EOF

JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
cat > "$OUT_ROOT/cads_remaining8_smoke_submission.env" <<EOF
CADS_SMOKE_JOB_ID=$JOB_ID
CADS_SMOKE_ROOT=$OUT_ROOT
CADS_SMOKE_RUN_OUT=$OUT_ROOT/run_loop
CADS_SMOKE_CASE_MANIFEST=$CASE_MANIFEST
EOF

echo "CADS_SMOKE_JOB_ID=$JOB_ID"
echo "CADS_SMOKE_ROOT=$OUT_ROOT"
echo "CADS_SMOKE_RUN_OUT=$OUT_ROOT/run_loop"
