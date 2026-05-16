#!/bin/bash
# Track 5 Problem 2 - Unified Experiment Launcher
# Usage: ./run_experiments.sh [gpu_id] [default: cuda:2]

GPU_DEVICE=${1:-cuda:2}
LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Track 5 Problem 2 - Experiment Launcher"
echo "GPU Device: $GPU_DEVICE"
echo "Log Directory: $LOG_DIR"
echo "=========================================="

# High Priority Experiments (Patent 4)
echo ""
echo "[1/6] Running STE baseline experiment..."
python scripts/run_ste.py --task task2 --noise_strength 1.0 --schedule exp \
    --epochs 30 --seed 42 --device "$GPU_DEVICE" --batch_size 512 --num_workers 8 \
    2>&1 | tee "$LOG_DIR/ste_baseline.log"

echo ""
echo "[2/6] Running gradient variance adaptive experiment (Patent 4)..."
python scripts/run_enhanced_experiments.py --experiment_type gradient_variance_adaptive \
    --device "$GPU_DEVICE" --epochs 30 --batch_size 512 --num_workers 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/gradient_variance_adaptive.log"

echo ""
echo "[3/6] Running zero-order correction experiment (Patent 4)..."
python scripts/run_enhanced_experiments.py --experiment_type zero_order_correction \
    --device "$GPU_DEVICE" --epochs 30 --batch_size 512 --num_workers 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/zero_order_correction.log"

# High Priority Experiments (Patent 5)
echo ""
echo "[4/6] Running spatiotemporal noise experiment (Patent 5)..."
python scripts/run_enhanced_experiments.py --experiment_type spatiotemporal_noise \
    --device "$GPU_DEVICE" --epochs 30 --batch_size 512 --num_workers 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/spatiotemporal_noise.log"

# High Priority Experiments (Patent 6)
echo ""
echo "[5/6] Running decoupled bias correction experiment (Patent 6)..."
python scripts/run_enhanced_experiments.py --experiment_type decoupled_bias_correction \
    --device "$GPU_DEVICE" --epochs 30 --batch_size 512 --num_workers 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/decoupled_bias_correction.log"

echo ""
echo "[6/6] Running regularizer v2 ablation experiment (Patent 6)..."
python scripts/run_enhanced_experiments.py --experiment_type regularizer_v2_ablation \
    --device "$GPU_DEVICE" --epochs 30 --batch_size 512 --num_workers 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/regularizer_v2_ablation.log"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $LOG_DIR"
echo "=========================================="
