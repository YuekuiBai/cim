#!/bin/bash
# Run optimization experiments
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
LOG_FILE="$LOG_DIR/optimization_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

{
    echo "=========================================="
    echo "Optimization Experiments"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""

    echo "[OPT-1/1] Running regularizer_v2_ablation (fixed)..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type regularizer_v2_ablation \
        --device cuda --epochs 50 --batch_size 256 --num_workers 4 --seed 42

    echo ""
    echo "=========================================="
    echo "Optimization Completed at: $(date)"
    echo "=========================================="

} > "$LOG_FILE" 2>&1 &

PID=$!
echo "Optimization started (PID: $PID)"
echo "Log file: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
