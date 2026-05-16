#!/bin/bash
# 双GPU并行实验启动脚本 - 专利验证+SOTA冲刺
# GPU1: 赛题1主攻 + 赛题2专利五
# GPU2: 赛题2主攻 + 赛题1专利二
# GPU0: 禁止使用

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

NUM_WORKERS=4
BATCH_SIZE=128

# ============================================
# GPU1 实验
# ============================================
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

# 赛题2专利五：层次化噪声注入（ResNet34 + 时空噪声 + Mixup + LS + EMA + 300epochs）
LOG_GPU1="$LOG_DIR/gpu1_patent5_$(date +%Y%m%d_%H%M%S).log"

{
    echo "==========================================="
    echo "GPU1实验启动 at: $(date)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "==========================================="
    
    echo "[专利五] ResNet34 + 时空噪声 + Mixup + LS + EMA + 300epochs..."
    cd "$PROBLEM2_DIR"
    python run_sota_v3.py \
        --device cuda \
        --epochs 300 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed 42 \
        --save_dir results/sota_v3

    echo ""
    echo "GPU1实验完成 at: $(date)"
} > "$LOG_GPU1" 2>&1 &

PID_GPU1=$!
echo "GPU1实验已启动 (PID: $PID_GPU1)"
echo "日志: $LOG_GPU1"

# ============================================
# GPU2 实验（等待当前SOTA P1完成后执行）
# ============================================
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

# 赛题2专利四/五/六验证
LOG_GPU2="$LOG_DIR/gpu2_patent456_$(date +%Y%m%d_%H%M%S).log"

{
    echo "==========================================="
    echo "GPU2实验启动 at: $(date)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "==========================================="
    
    echo "[专利四/五/六] 验证实验..."
    cd "$PROBLEM2_DIR"
    python run_patent_verification_p2.py \
        --device cuda \
        --epochs 100 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed 42 \
        --save_dir results/patent_verification_p2

    echo ""
    echo "GPU2实验完成 at: $(date)"
} > "$LOG_GPU2" 2>&1 &

PID_GPU2=$!
echo "GPU2实验已启动 (PID: $PID_GPU2)"
echo "日志: $LOG_GPU2"

# ============================================
# 监控信息
# ============================================
echo ""
echo "==========================================="
echo "双GPU实验已启动"
echo "==========================================="
echo "GPU1 PID: $PID_GPU1 (日志: $LOG_GPU1)"
echo "GPU2 PID: $PID_GPU2 (日志: $LOG_GPU2)"
echo ""
echo "监控命令:"
echo "  GPU1: tail -f $LOG_GPU1"
echo "  GPU2: tail -f $LOG_GPU2"
echo "  GPU利用率: watch -n 5 nvidia-smi"
echo "==========================================="
