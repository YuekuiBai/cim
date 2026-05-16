#!/bin/bash
# Run ALL experiments (Problem 1 + Problem 2) in background on GPU 2
# Force override any existing CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

PROBLEM1_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码"
PROBLEM2_DIR="/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码"
LOG_DIR="$PROBLEM2_DIR/logs"
LOG_FILE="$LOG_DIR/all_experiments_$(date +%Y%m%d_%H%M%S).log"

# Memory-safe settings
NUM_WORKERS=4
BATCH_SIZE=256

mkdir -p "$LOG_DIR"

{
    echo "=========================================="
    echo "ALL Experiments on GPU 2 (physical)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "NUM_WORKERS=$NUM_WORKERS"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""

    # ========== Problem 1 ==========
    echo "=========================================="
    echo "PROBLEM 1 - Nonlinear Error Experiments"
    echo "=========================================="
    cd "$PROBLEM1_DIR"

    echo ""
    echo "[P1-1/7] Running sensitivity analysis (task2, mixed mode)..."
    python scripts/run_sensitivity.py --task task2 --training_mode mixed --alpha_max 0.5 \
        --total_epochs 20 --seed 42 --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-2/7] Running scratch training (rigorous)..."
    python scripts/run_scratch_rigorous.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-3/7] Running extended experiment (gaussian vs nonlinear)..."
    python scripts/run_extended_gaussian_vs_nonlinear.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-4/7] Running extended experiment (network structure)..."
    python scripts/run_extended_network_structure.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-5/7] Running extended experiment (quantization)..."
    python scripts/run_extended_quantization.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-6/7] Running extended experiment (quantization v2)..."
    python scripts/run_extended_quantization_v2.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    echo ""
    echo "[P1-7/7] Running baseline training..."
    python scripts/train_baseline.py --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

    # ========== Problem 2 ==========
    echo ""
    echo "=========================================="
    echo "PROBLEM 2 - STE Noise Training Experiments"
    echo "=========================================="
    cd "$PROBLEM2_DIR"

    echo ""
    echo "[P2-1/6] Running STE baseline experiment..."
    python scripts/run_ste.py --task task2 --device cuda --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --save_dir ../结果

    echo ""
    echo "[P2-2/6] Running gradient variance adaptive experiment (Patent 4)..."
    python scripts/run_enhanced_experiments.py --experiment_type gradient_variance_adaptive \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[P2-3/6] Running zero-order correction experiment (Patent 4)..."
    python scripts/run_enhanced_experiments.py --experiment_type zero_order_correction \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[P2-4/6] Running spatiotemporal noise experiment (Patent 5)..."
    python scripts/run_enhanced_experiments.py --experiment_type spatiotemporal_noise \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[P2-5/6] Running decoupled bias correction experiment (Patent 6)..."
    python scripts/run_enhanced_experiments.py --experiment_type decoupled_bias_correction \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "[P2-6/6] Running regularizer v2 ablation experiment (Patent 6)..."
    python scripts/run_enhanced_experiments.py --experiment_type regularizer_v2_ablation \
        --device cuda --epochs 30 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed 42

    echo ""
    echo "=========================================="
    echo "ALL Experiments Completed at: $(date)"
    echo "=========================================="

} > "$LOG_FILE" 2>&1 &

PID=$!
echo "All experiments started in background on GPU 2 (PID: $PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Stop experiments:"
echo "  kill $PID"
