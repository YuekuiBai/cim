#!/bin/bash
# Re-run failed experiments
# Force GPU 2
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
LOG_FILE="$LOG_DIR/rerun_failed_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

{
    echo "=========================================="
    echo "Re-running Failed Experiments"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""

    # P1-5: Quantization experiment
    echo "[P1-5/4] Running quantization experiment..."
    cd "$PROBLEM1_DIR"
    python scripts/run_extended_quantization.py --device cuda --batch_size 256 --num_workers 4

    echo ""
    echo "[P2-1/4] Running STE baseline experiment..."
    cd "$PROBLEM2_DIR"
    python scripts/run_ste.py --task task2 --device cuda --batch_size 256 --num_workers 4 --save_dir ../结果

    echo ""
    echo "[P2-2/4] Running gradient variance adaptive experiment..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type gradient_variance_adaptive \
        --device cuda --epochs 30 --batch_size 256 --num_workers 4 --seed 42

    echo ""
    echo "[P2-3/4] Running zero-order correction experiment..."
    cd "$PROBLEM2_DIR"
    python scripts/run_enhanced_experiments.py --experiment_type zero_order_correction \
        --device cuda --epochs 30 --batch_size 256 --num_workers 4 --seed 42

    echo ""
    echo "=========================================="
    echo "Re-run Completed at: $(date)"
    echo "=========================================="

} > "$LOG_FILE" 2>&1 &

PID=$!
echo "Re-run started (PID: $PID)"
echo "Log file: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
