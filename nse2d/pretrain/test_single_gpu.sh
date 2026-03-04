#!/bin/bash
# =============================================================================
# Quick test: Single GPU training to verify code changes
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="test_single_gpu"
GPU_ID=${1:-0}

echo "========================================================================="
echo "  Quick Test: Single GPU (use_accelerate=0)"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Iterations: 100 (quick test)"
echo "========================================================================="
echo ""

/zhangtao/envs/rae/bin/python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "online" \
    --is_grf_overfitting_test 0 \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --grf_use_radial_mask 1 \
    --time_start 250.0 \
    --time_end 260.0 \
    --dt_data 1.0 \
    --Nx 512 --Ny 256 \
    --modes_x 64 --modes_y 64 \
    --width 40 \
    --n_layers 4 \
    --K 2 \
    --output_dim 5 \
    --rollout_dt 0.2 \
    --time_integrator euler \
    --lr 0.001 \
    --batch_size 2 \
    --num_iterations 100 \
    --log_every 20 \
    --eval_every 50 \
    --eval_rollout_steps 3 \
    --seed 42 \
    --use_accelerate 0

echo ""
echo "Single GPU test completed successfully!"
