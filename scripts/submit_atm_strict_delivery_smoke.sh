#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CASE_ID=${CASE_ID:-BDMAP_00000120}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:T4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-48G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/atm_strict_delivery_smoke_$(date +%Y%m%d_%H%M%S)}

cd "$CODE_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main

SMOKE_ROOT=$OUT_ROOT
RUN_OUT=$SMOKE_ROOT/run_loop
CASE_CSV=$SMOKE_ROOT/case_${CASE_ID}.csv
SLURM_DIR=$SMOKE_ROOT/slurm
mkdir -p "$RUN_OUT" "$SLURM_DIR"

"$PYTHON" - "$CASE_MANIFEST" "$CASE_CSV" "$CASE_ID" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
case_id = sys.argv[3]
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
match = next((row for row in rows if (row.get("case_id") or row.get("id")) == case_id), None)
if match is None:
    raise SystemExit(f"case_id not found in manifest: {case_id}")
ct_path = match.get("ct_path") or match.get("image_path")
ref_dir = match.get("annotation_folder") or match.get("reference_mask_dir") or match.get("mask_dir")
if not ct_path:
    raise SystemExit(f"case {case_id} is missing ct_path/image_path")
if not ref_dir:
    raise SystemExit(f"case {case_id} is missing annotation_folder/reference_mask_dir")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=["case_id", "ct_path", "annotation_folder"])
    writer.writeheader()
    writer.writerow({"case_id": case_id, "ct_path": ct_path, "annotation_folder": ref_dir})
PY

SBATCH_FILE=$SLURM_DIR/atm_strict_delivery_smoke.sbatch
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=atm_strict_smoke
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_DIR}/atm_strict_smoke_%j.out
#SBATCH --error=${SLURM_DIR}/atm_strict_smoke_%j.err

set -euo pipefail
if [ -f "\$HOME/.bodymaps_env" ]; then
  source "\$HOME/.bodymaps_env"
fi
cd "$CODE_ROOT"
mkdir -p "$RUN_OUT"
export MEDAI_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export NNUNETV2_PREDICT_EXECUTABLE="$NNUNETV2_PREDICT_EXECUTABLE"
export MEDAI_NNUNETV2_PREDICT="$NNUNETV2_PREDICT_EXECUTABLE"
export PATH="$(dirname "$NNUNETV2_PREDICT_EXECUTABLE"):\$PATH"

"$PYTHON" run_medai_cli.py --json run-loop \\
  --case-list "$CASE_CSV" \\
  --models atm \\
  --organs airway_tree \\
  --registry "$REGISTRY" \\
  --output "$RUN_OUT" \\
  --checkpoint-root "$CHECKPOINT_ROOT" \\
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \\
  --timeout-sec 14400 \\
  --teacher-inference-mode full_volume \\
  --no-enable-shapekit \\
  --debug-allow-no-shapekit \\
  --no-enable-critic \\
  --strict-delivery-targets \\
  --strict-delivery-fov-override-organs airway_tree \\
  --log-file "$RUN_OUT/run_loop.log"
EOF

JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
cat > "$SMOKE_ROOT/atm_smoke_submission.env" <<EOF
ATM_SMOKE_JOB_ID=$JOB_ID
ATM_SMOKE_ROOT=$SMOKE_ROOT
ATM_SMOKE_RUN_OUT=$RUN_OUT
ATM_SMOKE_CASE_CSV=$CASE_CSV
EOF

echo "ATM_SMOKE_JOB_ID=$JOB_ID"
echo "ATM_SMOKE_ROOT=$SMOKE_ROOT"
echo "ATM_SMOKE_RUN_OUT=$RUN_OUT"
