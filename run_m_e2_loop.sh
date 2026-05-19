    #!/bin/bash
# M-step + E-step Round 2 (continuation from completed E-step Round 1)
# Run: screen -S medai_cont -dm bash run_m_e2_loop.sh

set -euo pipefail

WORKDIR="/root/autodl-tmp/medai_v14"
TIMESTAMP="20260517_195010"
E1_DIR="$WORKDIR/outputs/run_pants50_round1_${TIMESTAMP}"
LOG_DIR="$WORKDIR/outputs/logs_${TIMESTAMP}"
M_OUTPUT="$WORKDIR/outputs/mstep_round1_${TIMESTAMP}"
E2_OUTPUT="$WORKDIR/outputs/run_pants50_round2_${TIMESTAMP}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_DIR/pipeline_cont.log"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*" | tee -a "$LOG_DIR/pipeline_cont.log"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_DIR/pipeline_cont.log"; exit 1; }

cd "$WORKDIR"
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────
log "=== Pre-flight checks ==="

[ -f "$E1_DIR/training_manifest.json" ] || die "E步骤结果不存在: $E1_DIR/training_manifest.json"

MANIFEST_ITEMS=$(python3 -c "import json; d=json.load(open('$E1_DIR/training_manifest.json')); print(len(d))" 2>/dev/null)
log "E步骤训练清单: $MANIFEST_ITEMS 条"

FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "Free VRAM: ${FREE_VRAM} MiB"
[ "$FREE_VRAM" -lt 20000 ] && die "显存不足 (${FREE_VRAM} MiB)"

FREE_DISK=$(df /root/autodl-tmp --output=avail -BG | tail -1 | tr -d 'G ')
log "Free disk: ${FREE_DISK} GB"
[ "$FREE_DISK" -lt 5 ] && die "磁盘空间不足 (${FREE_DISK} GB)"

if ! curl -s --max-time 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
    warn "VLM server not running, starting..."
    HF_HOME=/root/autodl-tmp/.hf_cache \
    HF_ENDPOINT=https://hf-mirror.com \
    OMP_NUM_THREADS=1 \
    lmdeploy serve api_server Qwen/Qwen2.5-VL-7B-Instruct \
        --server-port 8000 --tp 1 --cache-max-entry-count 0.4 \
        &>> "$LOG_DIR/lmdeploy_server.log" &
    for i in $(seq 1 24); do
        sleep 5
        curl -s --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1 && { log "VLM server ready."; break; }
        [ $i -eq 24 ] && die "VLM server 启动失败"
    done
else
    log "VLM server OK."
fi

log "Pre-flight OK."
echo ""

# ─────────────────────────────────────────────
# M-step (50 epoch, from scratch — label set
# differs from pretrained checkpoint)
# ─────────────────────────────────────────────
log "=== STEP 2/3: M-step (50 epoch, 从头训练) ==="
log "训练数据: $E1_DIR/training_manifest.json"
log "输出目录: $M_OUTPUT"
log "预计时间: ~1.8 小时"
echo ""

OMP_NUM_THREADS=1 python3 run_medai_cli.py --json mstep-update \
    --training-manifest "$E1_DIR/training_manifest.json" \
    --output-folder "$M_OUTPUT" \
    --target-model epai_20250421 \
    --ct-source-root third_party/PanTS-main/data \
    2>&1 | tee "$LOG_DIR/m_step.log"

# Find checkpoint
NEW_CKPT=$(find "$M_OUTPUT" -name "checkpoint_best.pth" 2>/dev/null | head -1)
[ -z "$NEW_CKPT" ] && NEW_CKPT=$(find "$M_OUTPUT" -name "checkpoint_final.pth" 2>/dev/null | head -1)
[ -z "$NEW_CKPT" ] && die "M步骤失败: 未找到 checkpoint"

log "M步骤完成. Checkpoint: $NEW_CKPT"

# Show training curve summary
python3 -c "
import re
log_file = '$(find $M_OUTPUT -name training_log_*.txt 2>/dev/null | head -1)'
if not log_file: exit()
with open(log_file) as f:
    text = f.read()
epochs = re.findall(r'Epoch (\d+)', text)
dices  = re.findall(r'EMA pseudo Dice: ([\d.]+)', text)
losses = re.findall(r'train_loss ([-\d.]+)', text)
if epochs:
    print(f'  训练 epoch 数: {len(set(epochs))}')
if dices:
    print(f'  最终 EMA Dice: {dices[-1]}')
if losses:
    print(f'  最终 train loss: {losses[-1]}')
" 2>/dev/null | tee -a "$LOG_DIR/pipeline_cont.log"
echo ""

# ─────────────────────────────────────────────
# E-step Round 2
# ─────────────────────────────────────────────
log "=== STEP 3/3: E-step Round 2 (更新后的模型) ==="
log "输出目录: $E2_OUTPUT"
log "预计时间: ~2.2 小时"
echo ""

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

[ -f "$E2_OUTPUT/dice_metrics.csv" ] || die "E步骤 Round 2 失败"
log "E步骤 Round 2 完成."

# ─────────────────────────────────────────────
# Final comparison
# ─────────────────────────────────────────────
echo ""
log "=== 最终对比: Round 1 vs Round 2 ==="
python3 -c "
import csv

def summarize(path, label):
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)
                    if r['dice'] and 'shapekit' not in r['model']]
        if not rows:
            print(f'  {label}: 无 DICE 数据'); return
        dices = [float(r['dice']) for r in rows]
        accepted = sum(1 for r in rows if r['decision'] == 'accept')
        by_organ = {}
        for r in rows:
            by_organ.setdefault(r['organ'], []).append(float(r['dice']))
        print(f'  {label}:')
        print(f'    总体  mean={sum(dices)/len(dices):.4f}  accepted={accepted}/{len(rows)}')
        for organ in sorted(by_organ):
            vals = by_organ[organ]
            print(f'    {organ:<22} mean={sum(vals)/len(vals):.4f}  n={len(vals)}')
    except Exception as e:
        print(f'  {label}: 读取失败 - {e}')

summarize('$E1_DIR/dice_metrics.csv',  'Round 1 (baseline)')
print()
summarize('$E2_OUTPUT/dice_metrics.csv', 'Round 2 (M步骤后)')
" 2>/dev/null | tee -a "$LOG_DIR/pipeline_cont.log"

echo ""
log "=== 全部完成 ==="
log "E步骤 Round 1: $E1_DIR"
log "M步骤:         $M_OUTPUT"
log "E步骤 Round 2: $E2_OUTPUT"
log "新 checkpoint: $NEW_CKPT"
log "日志目录:      $LOG_DIR"
