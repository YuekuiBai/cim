#!/bin/bash
# GPU1专利验证实验 - 赛题2专利五 + 赛题1专利三

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

{
    echo "==========================================="
    echo "GPU1实验启动 at: $(date)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "==========================================="
    echo ""

    echo "[赛题2 专利五] ResNet34 + 时空噪声 + Mixup + LS + EMA + 300epochs..."
    cd "$PROBLEM2_DIR"
    python run_sota_v3.py \
        --device cuda \
        --epochs 300 \
        --batch_size 128 \
        --num_workers 4 \
        --seed 42 \
        --save_dir results/sota_v3

    echo ""
    echo "GPU1实验完成 at: $(date)"
} > "$LOG_DIR/gpu1_patent5_${TIMESTAMP}.log" 2>&1 &

PID_GPU1=$!
echo "GPU1实验已启动 (PID: $PID_GPU1)"
echo "日志: $LOG_DIR/gpu1_patent5_${TIMESTAMP}.log"
echo "监控: tail -f $LOG_DIR/gpu1_patent5_${TIMESTAMP}.log"
