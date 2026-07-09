#!/bin/bash
#SBATCH --job-name=medai-hf-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/medai-hf-%j.out
#SBATCH --error=logs/medai-hf-%j.err

set -euo pipefail

CT=${1:?Usage: sbatch medai_infer.sbatch CT_PATH OUTPUT_DIR MODEL_KEY}
OUT=${2:?Usage: sbatch medai_infer.sbatch CT_PATH OUTPUT_DIR MODEL_KEY}
MODEL=${3:-atlasnet}
HF_ASSET_ROOT=${HF_ASSET_ROOT:-$PWD/checkpoints/MedIA-Agentic-AI}

mkdir -p logs "$OUT"
export PYTHONPATH="$PWD/agent-harness:${PYTHONPATH:-}"

case "$MODEL" in
  atlasnet)
    python scripts/atlasnet_predict_and_split.py --image "$CT" --output "$OUT" --atlas-root "$HF_ASSET_ROOT/teacher_models/atlasnet" --label-map configs/atlasnet_label_map.json --device cuda:0 --postprocess-mode auto
    ;;
  unest)
    python scripts/unest_predict_and_split.py --image "$CT" --output "$OUT" --unest-root "$HF_ASSET_ROOT/teacher_models/unest"
    ;;
  vista3d)
    python scripts/vista3d_predict_and_split.py --image "$CT" --output "$OUT" --vista-root "$HF_ASSET_ROOT/teacher_models/vista3d"
    ;;
  totalsegmentator)
    python run_medai_cli.py --json infer --image "$CT" --output-folder "$OUT" --backend totalseg
    ;;
  *)
    echo "For nnUNet model $MODEL, use configs/hf_model_manifest.yaml to copy the exact command." >&2
    exit 2
    ;;
esac
