#!/bin/bash
# GPU并行实验启动脚本 - 修复EMA Bug + 充分利用GPU1+GPU2
# GPU1: 专利一+诊断框架组合实验（赛题1 SOTA冲刺）
# GPU2: 专利五（ResNet34+时空噪声+300epochs，修复EMA Bug后重新运行）

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================
# GPU1 实验 - 专利一+诊断框架组合
# ============================================
LOG_GPU1="$LOG_DIR/gpu1_patent1_2_combined_${TIMESTAMP}.log"

(
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=1
    
    echo "==========================================="
    echo "GPU1实验启动 at: $(date)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "实验：专利一+诊断框架组合（ResNet34+分层Alpha）"
    echo "==========================================="
    
    cd "$PROBLEM1_DIR"
    python run_patent1_2_combined.py \
        --device cuda \
        --epochs 100 \
        --batch_size 128 \
        --num_workers 4 \
        --seed 42 \
        --save_dir results/patent1_2_combined

    echo ""
    echo "GPU1实验完成 at: $(date)"
) > "$LOG_GPU1" 2>&1 &

PID_GPU1=$!
echo "GPU1实验已启动 (PID: $PID_GPU1)"
echo "日志: $LOG_GPU1"
echo "监控: tail -f $LOG_GPU1"
echo ""

# ============================================
# GPU2 实验 - 专利五（ResNet34+时空噪声+300epochs）
# ============================================
LOG_GPU2="$LOG_DIR/gpu2_patent5_sota_v3_${TIMESTAMP}.log"

(
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=2
    
    echo "==========================================="
    echo "GPU2实验启动 at: $(date)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "实验：专利五（ResNet34+时空噪声+Mixup+EMA，修复EMA Bug）"
    echo "==========================================="
    
    cd "$PROBLEM2_DIR"
    python run_sota_v3.py \
        --device cuda \
        --epochs 300 \
        --batch_size 128 \
        --num_workers 4 \
        --seed 42 \
        --save_dir results/sota_v3_fixed

    echo ""
    echo "GPU2实验完成 at: $(date)"
) > "$LOG_GPU2" 2>&1 &

PID_GPU2=$!
echo "GPU2实验已启动 (PID: $PID_GPU2)"
echo "日志: $LOG_GPU2"
echo "监控: tail -f $LOG_GPU2"
echo ""

echo "==========================================="
echo "双GPU实验状态"
echo "==========================================="
echo "GPU1: 专利一+诊断框架组合 (PID: $PID_GPU1)"
echo "GPU2: 专利五300epochs-修复EMA (PID: $PID_GPU2)"
echo "==========================================="
echo ""
echo "等待GPU2实验完成..."
wait $PID_GPU2
echo "GPU2实验完成!"

echo ""
echo "等待GPU1实验完成..."
wait $PID_GPU1
echo "GPU1实验完成!"

echo ""
echo "==========================================="
echo "所有实验完成 at: $(date)"
echo "==========================================="
