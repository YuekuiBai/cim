#!/bin/bash
# Run Problem 2 experiments in background on GPU 2
export CUDA_VISIBLE_DEVICES=2
LOG_FILE="logs/experiments_$(date +%Y%m%d_%H%M%S)_problem2.log"

echo "Starting Problem 2 experiments on cuda:0 (physical GPU 2)..."
echo "Log file: $LOG_FILE"

cd /mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码

# Run all experiments sequentially
{
    echo "=== Problem 2 Experiment Started at $(date) ==="
    echo ""

    echo "[1/6] Running STE baseline experiment..."
    python scripts/run_ste.py --task task2 --noise_strength 1.0 --schedule exp \
        --epochs 30 --seed 42 --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[2/6] Running gradient variance adaptive experiment (Patent 4)..."
    python scripts/run_enhanced_experiments.py --experiment_type gradient_variance_adaptive \
        --device cuda --epochs 30 --batch_size 512 --num_workers 8 --seed 42
    echo ""

    echo "[3/6] Running zero-order correction experiment (Patent 4)..."
    python scripts/run_enhanced_experiments.py --experiment_type zero_order_correction \
        --device cuda --epochs 30 --batch_size 512 --num_workers 8 --seed 42
    echo ""

    echo "[4/6] Running spatiotemporal noise experiment (Patent 5)..."
    python scripts/run_enhanced_experiments.py --experiment_type spatiotemporal_noise \
        --device cuda --epochs 30 --batch_size 512 --num_workers 8 --seed 42
    echo ""

    echo "[5/6] Running decoupled bias correction experiment (Patent 6)..."
    python scripts/run_enhanced_experiments.py --experiment_type decoupled_bias_correction \
        --device cuda --epochs 30 --batch_size 512 --num_workers 8 --seed 42
    echo ""

    echo "[6/6] Running regularizer v2 ablation experiment (Patent 6)..."
    python scripts/run_enhanced_experiments.py --experiment_type regularizer_v2_ablation \
        --device cuda --epochs 30 --batch_size 512 --num_workers 8 --seed 42
    echo ""

    echo "=== Problem 2 Experiment Completed at $(date) ==="
} > "$LOG_FILE" 2>&1 &

echo "Problem 2 experiments started in background (PID: $!)"
echo "Log file: $LOG_FILE"
