#!/bin/bash
# 修复版实验启动脚本 - 修复除零错误 + 明确日志路径

LOG_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# GPU1 实验 - 赛题1 专利一+诊断框架组合（已修复除零错误）
export CUDA_VISIBLE_DEVICES=1
python /mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码/run_patent1_2_combined.py \
    --device cuda \
    --epochs 100 \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42 \
    --save_dir results/patent1_2_combined \
    > "$LOG_DIR/gpu1_patent1_2_combined_${TIMESTAMP}.log" 2>&1 &

GPU1_PID=$!
echo "GPU1实验已启动 (PID: $GPU1_PID)"
echo "日志: $LOG_DIR/gpu1_patent1_2_combined_${TIMESTAMP}.log"

# GPU2 实验 - 赛题2 专利五（已修复EMA）
export CUDA_VISIBLE_DEVICES=2
python /mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码/run_sota_v3.py \
    --device cuda \
    --epochs 300 \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42 \
    --save_dir results/sota_v3_fixed \
    > "$LOG_DIR/gpu2_patent5_sota_v3_${TIMESTAMP}.log" 2>&1 &

GPU2_PID=$!
echo "GPU2实验已启动 (PID: $GPU2_PID)"
echo "日志: $LOG_DIR/gpu2_patent5_sota_v3_${TIMESTAMP}.log"

echo ""
echo "==========================================="
echo "双GPU实验状态"
echo "==========================================="
echo "GPU1: 专利一+诊断框架组合 (PID: $GPU1_PID)"
echo "GPU2: 专利五300epochs (PID: $GPU2_PID)"
echo "==========================================="
echo ""
echo "监控命令："
echo "  GPU1: tail -f $LOG_DIR/gpu1_patent1_2_combined_${TIMESTAMP}.log"
echo "  GPU2: tail -f $LOG_DIR/gpu2_patent5_sota_v3_${TIMESTAMP}.log"
echo ""
