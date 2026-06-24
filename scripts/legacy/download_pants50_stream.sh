#!/bin/bash
# PanTS 第一个tar包下载 + 流式提取50个选定case
set -e

DATA_DIR="/home/teacher1/JHU-project1/medical_agent/third_party/PanTS-main/data"
mkdir -p "$DATA_DIR/ImageTr"
mkdir -p "$DATA_DIR/LabelTr"

LOG="/tmp/pants_download.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "开始时间: $(date)"
echo "========================================"

# ── 步骤1：流式下载tar1并提取50个case ──
echo ""
echo "[1/2] 流式下载 PanTSMini_ImageTr_00000001_00001000.tar.gz 并提取目标case..."
echo "      预计耗时: 30-40小时（取决于网速）"
echo ""

CASES=(
    PanTS_00000026 PanTS_00000029 PanTS_00000031 PanTS_00000035
    PanTS_00000047 PanTS_00000049 PanTS_00000074 PanTS_00000086
    PanTS_00000100 PanTS_00000145 PanTS_00000162 PanTS_00000224
    PanTS_00000246 PanTS_00000270 PanTS_00000363 PanTS_00000368
    PanTS_00000416 PanTS_00000423 PanTS_00000449 PanTS_00000451
    PanTS_00000465 PanTS_00000482 PanTS_00000485 PanTS_00000488
    PanTS_00000554 PanTS_00000564 PanTS_00000574 PanTS_00000592
    PanTS_00000626 PanTS_00000654 PanTS_00000674 PanTS_00000693
    PanTS_00000696 PanTS_00000727 PanTS_00000730 PanTS_00000750
    PanTS_00000797 PanTS_00000806 PanTS_00000811 PanTS_00000814
    PanTS_00000836 PanTS_00000849 PanTS_00000855 PanTS_00000860
    PanTS_00000871 PanTS_00000878 PanTS_00000881 PanTS_00000927
    PanTS_00000930 PanTS_00000966
)

# 构建tar wildcards
WILDCARDS=()
for c in "${CASES[@]}"; do
    WILDCARDS+=("--wildcards" "*/${c}/*")
done

URL="https://hf-mirror.com/datasets/BodyMaps/PanTSMini/resolve/main/PanTSMini_ImageTr_00000001_00001000.tar.gz"

# 流式下载+解压，不保存整个tar包到磁盘
# --keep-newer-files 避免重复覆盖已提取的文件
curl -L --retry 10 --retry-delay 30 --retry-max-time 3600 \
     --connect-timeout 30 --max-time 0 \
     --progress-bar \
     "$URL" \
  | tar -xz "${WILDCARDS[@]}" \
        --keep-newer-files \
        -C "$DATA_DIR/ImageTr" \
  && echo "" && echo "[1/2] CT提取完成: $(date)"

# 统计提取结果
echo "已提取case数: $(ls "$DATA_DIR/ImageTr" | wc -l)"

# ── 步骤2：等待标注包（需手动放置或另行下载）──
echo ""
echo "[2/2] 检查标注包..."
LABEL_TAR="$DATA_DIR/PanTSMini_Label.tar.gz"
if [ -f "$LABEL_TAR" ]; then
    echo "发现标注包，开始解压..."
    mkdir -p "$DATA_DIR/LabelAll"
    tar -xzf "$LABEL_TAR" -C "$DATA_DIR/LabelAll" \
        --checkpoint=1000 --checkpoint-action=echo="."
    # 只保留选中的50个case的标注
    for c in "${CASES[@]}"; do
        if [ -d "$DATA_DIR/LabelAll/$c" ]; then
            mv "$DATA_DIR/LabelAll/$c" "$DATA_DIR/LabelTr/"
        fi
    done
    rmdir "$DATA_DIR/LabelAll" 2>/dev/null || true
    echo "标注解压完成: $(ls "$DATA_DIR/LabelTr" | wc -l) 个case"
else
    echo "标注包未找到，请将 PanTSMini_Label.tar.gz 放到:"
    echo "  $LABEL_TAR"
    echo "然后重新运行本脚本的步骤2"
fi

echo ""
echo "========================================"
echo "完成时间: $(date)"
echo "========================================"
