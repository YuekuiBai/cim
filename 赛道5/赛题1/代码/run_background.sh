#!/bin/bash
# Run Problem 1 experiments in background on GPU 2
export CUDA_VISIBLE_DEVICES=2
LOG_FILE="logs/experiments_$(date +%Y%m%d_%H%M%S)_problem1.log"

echo "Starting Problem 1 experiments on cuda:0 (physical GPU 2)..."
echo "Log file: $LOG_FILE"

cd /mnt/storage2/zyc/CIM比赛/赛道5/赛题1/代码

# Run all experiments sequentially
{
    echo "=== Problem 1 Experiment Started at $(date) ==="
    echo ""

    echo "[1/7] Running sensitivity analysis (task2, mixed mode)..."
    python scripts/run_sensitivity.py --task task2 --training_mode mixed --alpha_max 0.5 \
        --epochs 20 --seed 42 --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[2/7] Running scratch training (rigorous)..."
    python scripts/run_scratch_rigorous.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[3/7] Running extended experiment (gaussian vs nonlinear)..."
    python scripts/run_extended_gaussian_vs_nonlinear.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[4/7] Running extended experiment (network structure)..."
    python scripts/run_extended_network_structure.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[5/7] Running extended experiment (quantization)..."
    python scripts/run_extended_quantization.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[6/7] Running extended experiment (quantization v2)..."
    python scripts/run_extended_quantization_v2.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "[7/7] Running baseline training..."
    python scripts/train_baseline.py --device cuda --batch_size 512 --num_workers 8
    echo ""

    echo "=== Problem 1 Experiment Completed at $(date) ==="
} > "$LOG_FILE" 2>&1 &

echo "Problem 1 experiments started in background (PID: $!)"
echo "Log file: $LOG_FILE"
