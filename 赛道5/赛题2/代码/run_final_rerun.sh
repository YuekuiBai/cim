#!/bin/bash
# Run fixed experiments for Problem 2
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
LOG_FILE="$LOG_DIR/final_rerun_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

NUM_WORKERS=4
BATCH_SIZE=256

{
    echo "=========================================="
    echo "Final Re-run: Fixed P2-2 and P2-3"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""

    echo "[FIX-1/3] P2-2 gradient_variance_adaptive (fixed device mismatch)..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type gradient_variance_adaptive \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[FIX-2/3] P2-3 zero_order_correction (fixed device mismatch)..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type zero_order_correction \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[FIX-3/3] P2-6 regularizer_v2_ablation (fixed normalization)..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type regularizer_v2_ablation \
        --device cuda --epochs 50 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "=========================================="
    echo "Final Re-run Completed at: $(date)"
    echo "=========================================="

} > "$LOG_FILE" 2>&1 &

PID=$!
echo "Final re-run started (PID: $PID)"
echo "Log file: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
