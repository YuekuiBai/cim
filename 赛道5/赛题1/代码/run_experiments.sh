#!/bin/bash
# Track 5 Problem 1 - Unified Experiment Launcher
# Usage: ./run_experiments.sh [gpu_id] [default: cuda:0]

GPU_DEVICE=${1:-cuda:0}
LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Track 5 Problem 1 - Experiment Launcher"
echo "GPU Device: $GPU_DEVICE"
echo "Log Directory: $LOG_DIR"
echo "=========================================="

# High Priority Experiments
echo ""
echo "[1/7] Running sensitivity analysis (task2, mixed mode)..."
python scripts/run_sensitivity.py --task task2 --training_mode mixed --alpha_max 0.5 \
    --epochs 20 --seed 42 --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/sensitivity_mixed.log"

echo ""
echo "[2/7] Running scratch training (rigorous)..."
python scripts/run_scratch_rigorous.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/scratch_rigorous.log"

echo ""
echo "[3/7] Running extended experiment (gaussian vs nonlinear)..."
python scripts/run_extended_gaussian_vs_nonlinear.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/extended_gaussian_vs_nonlinear.log"

echo ""
echo "[4/7] Running extended experiment (network structure)..."
python scripts/run_extended_network_structure.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/extended_network_structure.log"

echo ""
echo "[5/7] Running extended experiment (quantization)..."
python scripts/run_extended_quantization.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/extended_quantization.log"

echo ""
echo "[6/7] Running extended experiment (quantization v2)..."
python scripts/run_extended_quantization_v2.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/extended_quantization_v2.log"

echo ""
echo "[7/7] Running baseline training..."
python scripts/train_baseline.py --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/baseline.log"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $LOG_DIR"
echo "=========================================="
