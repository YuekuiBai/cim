#!/bin/bash
# SOTA冲刺实验 - GPU2
# 赛题1: 冲击90%+ (ResNet34/50 + 混合Alpha + Mixup + EMA)
# 赛题2: 冲击90%+ (ResNet34 + 时空噪声 + Mixup + Label Smoothing + EMA + 300epochs)

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sota_rerun_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

NUM_WORKERS=4
BATCH_SIZE=128

{
    echo "==========================================="
    echo "SOTA冲刺实验 - 冲击90%+"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Started at: $(date)"
    echo "==========================================="
    echo ""

    echo "[赛题1] SOTA冲刺实验 (ResNet34/50 + 混合Alpha + Mixup + EMA)..."
    cd "$PROBLEM1_DIR"
    python run_sota_p1.py \
        --device cuda \
        --epochs 100 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed 42 \
        --save_dir results/sota_p1

    echo ""
    echo "[赛题2] SOTA V3冲刺实验 (ResNet34 + 时空噪声 + Mixup + Label Smoothing + EMA + 300epochs)..."
    cd "$PROBLEM2_DIR"
    python run_sota_v3.py \
        --device cuda \
        --epochs 300 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed 42 \
        --save_dir results/sota_v3

    echo ""
    echo "==========================================="
    echo "SOTA冲刺实验完成 at: $(date)"
    echo "==========================================="

} > "$LOG_FILE" 2>&1 &

PID=$!
echo "SOTA冲刺实验已启动 (PID: $PID)"
echo "日志文件: $LOG_FILE"
echo "监控: tail -f $LOG_FILE"
