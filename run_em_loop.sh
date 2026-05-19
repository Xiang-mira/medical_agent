#!/bin/bash
# EM Loop: E-step -> M-step -> E-step (Round 2)
# Run inside a screen session: screen -S medai bash run_em_loop.sh
# Detach: Ctrl+A D  |  Reattach: screen -r medai

set -euo pipefail

WORKDIR="/root/autodl-tmp/medai_v14"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$WORKDIR/outputs/logs_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Color output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_DIR/pipeline.log"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*" | tee -a "$LOG_DIR/pipeline.log"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_DIR/pipeline.log"; exit 1; }

cd "$WORKDIR"

# ─────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────
log "=== Pre-flight checks ==="

# VLM server
if ! curl -s --max-time 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
    warn "LabelCritic VLM server not responding on :8000"
    warn "Starting lmdeploy server..."
    HF_HOME=/root/autodl-tmp/.hf_cache \
    HF_ENDPOINT=https://hf-mirror.com \
    OMP_NUM_THREADS=1 \
    lmdeploy serve api_server Qwen/Qwen2.5-VL-7B-Instruct \
        --server-port 8000 --tp 1 --cache-max-entry-count 0.4 \
        &>> "$LOG_DIR/lmdeploy_server.log" &
    LMDEPLOY_PID=$!
    log "Waiting for VLM server to start (PID $LMDEPLOY_PID)..."
    for i in $(seq 1 24); do
        sleep 5
        if curl -s --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
            log "VLM server ready."
            break
        fi
        if [ $i -eq 24 ]; then
            die "VLM server failed to start after 2 minutes. Check $LOG_DIR/lmdeploy_server.log"
        fi
    done
else
    log "VLM server already running on :8000."
fi

# GPU memory
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "Free VRAM: ${FREE_VRAM} MiB (need ~25000 for training)"
if [ "$FREE_VRAM" -lt 20000 ]; then
    die "Not enough free VRAM (${FREE_VRAM} MiB). Kill other GPU processes first."
fi

# Disk space
FREE_DISK=$(df /root/autodl-tmp --output=avail -BG | tail -1 | tr -d 'G ')
log "Free disk (autodl-tmp): ${FREE_DISK} GB"
if [ "$FREE_DISK" -lt 5 ]; then
    die "Less than 5GB free on /root/autodl-tmp. Free up space first."
fi

log "Pre-flight OK."
echo ""

# ─────────────────────────────────────────────
# Step 1: E-step Round 1
# ─────────────────────────────────────────────
E1_OUTPUT="outputs/run_pants50_round1_${TIMESTAMP}"
log "=== STEP 1/3: E-step Round 1 ==="
log "Output: $E1_OUTPUT"
log "Estimated time: ~2.2 hours (50 cases x 2 models)"
echo ""

OMP_NUM_THREADS=1 python3 run_medai_cli.py --json run-loop \
    --case-list data_manifest/case_list_50_tumor.csv \
    --models epai_20250421,vsmtrans \
    --organs pancreas,liver,spleen,kidney_left,kidney_right,aorta,postcava,duodenum,stomach \
    --output "$E1_OUTPUT" \
    --enable-shapekit \
    --enable-critic \
    --critic-backend labelcritic \
    --critic-base-url http://localhost \
    --critic-port 8000 \
    2>&1 | tee "$LOG_DIR/e_step_round1.log"

# Check E-step succeeded
if [ ! -f "$E1_OUTPUT/training_manifest.json" ]; then
    die "E-step Round 1 failed: training_manifest.json not found."
fi

MANIFEST_ITEMS=$(python3 -c "import json; d=json.load(open('$E1_OUTPUT/training_manifest.json')); print(len(d))" 2>/dev/null || echo 0)
log "E-step Round 1 complete. Training manifest: $MANIFEST_ITEMS items."

# Quick DICE summary
python3 -c "
import csv
with open('$E1_OUTPUT/dice_metrics.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['dice'] and 'shapekit' not in r['model']]
if rows:
    dices = [float(r['dice']) for r in rows]
    print(f'  DICE summary: mean={sum(dices)/len(dices):.3f}  min={min(dices):.3f}  max={max(dices):.3f}  n={len(dices)}')
    accepted = sum(1 for r in rows if r['decision'] == 'accept')
    print(f'  Accepted: {accepted}/{len(rows)} ({100*accepted//len(rows)}%)')
" 2>/dev/null | tee -a "$LOG_DIR/pipeline.log"
echo ""

# ─────────────────────────────────────────────
# Step 2: M-step (50-epoch finetune)
# ─────────────────────────────────────────────
M_OUTPUT="outputs/mstep_round1_${TIMESTAMP}"
PRETRAINED="checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"
log "=== STEP 2/3: M-step (50-epoch finetune) ==="
log "Output: $M_OUTPUT"
log "Pretrained weights: $PRETRAINED"
log "Estimated time: ~2.9 hours"
echo ""

OMP_NUM_THREADS=1 python3 run_medai_cli.py --json mstep-update \
    --training-manifest "$E1_OUTPUT/training_manifest.json" \
    --output-folder "$M_OUTPUT" \
    --target-model epai_20250421 \
    --ct-source-root third_party/PanTS-main/data \
    --pretrained-weights "$PRETRAINED" \
    2>&1 | tee "$LOG_DIR/m_step.log"

# Check M-step succeeded
MSTEP_STATUS=$(python3 -c "
import json, glob
files = glob.glob('$M_OUTPUT/**/mstep_training_result.json', recursive=True)
if files:
    d = json.load(open(files[0]))
    print(d.get('status','unknown'))
else:
    print('no_result_file')
" 2>/dev/null)

if [ "$MSTEP_STATUS" != "success" ]; then
    warn "M-step status: $MSTEP_STATUS"
    warn "Checking for checkpoint anyway..."
fi

NEW_CKPT=$(find "$M_OUTPUT" -name "checkpoint_best.pth" 2>/dev/null | head -1)
if [ -z "$NEW_CKPT" ]; then
    NEW_CKPT=$(find "$M_OUTPUT" -name "checkpoint_final.pth" 2>/dev/null | head -1)
fi

if [ -z "$NEW_CKPT" ]; then
    die "M-step failed: no checkpoint found in $M_OUTPUT"
fi

log "M-step complete. New checkpoint: $NEW_CKPT"
echo ""

# ─────────────────────────────────────────────
# Step 3: E-step Round 2 (with updated model)
# ─────────────────────────────────────────────
E2_OUTPUT="outputs/run_pants50_round2_${TIMESTAMP}"
log "=== STEP 3/3: E-step Round 2 (updated model) ==="
log "Using checkpoint: $NEW_CKPT"
log "Output: $E2_OUTPUT"
log "Estimated time: ~2.2 hours"
echo ""

# Temporarily override the epai checkpoint for this run via extra context
# The mstep-update auto-updates the registry; if it didn't, pass checkpoint explicitly
OMP_NUM_THREADS=1 python3 run_medai_cli.py --json run-loop \
    --case-list data_manifest/case_list_50_tumor.csv \
    --models epai_20250421,vsmtrans \
    --organs pancreas,liver,spleen,kidney_left,kidney_right,aorta,postcava,duodenum,stomach \
    --output "$E2_OUTPUT" \
    --enable-shapekit \
    --enable-critic \
    --critic-backend labelcritic \
    --critic-base-url http://localhost \
    --critic-port 8000 \
    2>&1 | tee "$LOG_DIR/e_step_round2.log"

if [ ! -f "$E2_OUTPUT/dice_metrics.csv" ]; then
    die "E-step Round 2 failed: dice_metrics.csv not found."
fi

log "E-step Round 2 complete."

# ─────────────────────────────────────────────
# Final comparison: Round 1 vs Round 2
# ─────────────────────────────────────────────
echo ""
log "=== FINAL COMPARISON ==="
python3 -c "
import csv

def summarize(path, label):
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r['dice'] and 'shapekit' not in r['model']]
        if not rows:
            print(f'  {label}: no DICE data')
            return
        dices = [float(r['dice']) for r in rows]
        accepted = sum(1 for r in rows if r['decision'] == 'accept')
        by_organ = {}
        for r in rows:
            by_organ.setdefault(r['organ'], []).append(float(r['dice']))
        print(f'  {label}:')
        print(f'    overall  mean={sum(dices)/len(dices):.4f}  accepted={accepted}/{len(rows)}')
        for organ in sorted(by_organ):
            vals = by_organ[organ]
            print(f'    {organ:20s} mean={sum(vals)/len(vals):.4f}  n={len(vals)}')
    except Exception as e:
        print(f'  {label}: error - {e}')

summarize('$E1_OUTPUT/dice_metrics.csv', 'Round 1 (baseline)')
print()
summarize('$E2_OUTPUT/dice_metrics.csv', 'Round 2 (after M-step)')
" 2>/dev/null | tee -a "$LOG_DIR/pipeline.log"

echo ""
log "=== ALL DONE ==="
log "Logs:     $LOG_DIR/"
log "Round 1:  $E1_OUTPUT/"
log "M-step:   $M_OUTPUT/"
log "Round 2:  $E2_OUTPUT/"
log "New ckpt: $NEW_CKPT"
