#!/bin/bash
# =============================================================================
# Exp10: Self-Training Mode with Multi-GPU (4 GPUs via Accelerate)
# =============================================================================
# 
# Training Strategy:
# - Phase 1 (steps 0-9999): Train with raw GRF data (physics loss)
# - Phase 2 (steps 10000+): Train with model-evolved GRF data
#   - GRF -> Model (10 NFE) -> evolved state -> Training
#   - Generator model weights refreshed every 10000 steps
#
# Configuration:
# - Total iterations: 100000
# - Self-training activation: step 10000
# - Weight update interval: 10000 steps
# - Rollout NFE: 10 steps
# - Model dt: 0.1, output_dim: 10
# - Evaluation: mhd_sim test set (trajectory rollout L2 error)
# - 4 GPU training via Accelerate
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Data paths
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp10_self_training_4gpu"
GPU_IDS=${1:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "========================================================================="
echo "  Exp10: Self-Training with Multi-GPU"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  Total iterations: 100000"
echo "  Phase 1 (0-9999): Raw GRF training"
echo "  Phase 2 (10000+): Model-evolved GRF training"
echo "  Weight update every: 10000 steps"
echo "  Model rollout: dt=0.1, output_dim=10, NFE=10"
echo "  Eval: mhd_sim test set"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU
CUDA_VISIBLE_DEVICES=$GPU_IDS /zhangtao/envs/rae/bin/accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    main.py \
    --mode train \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "online" \
    --self_training_start_step 10000 \
    --self_training_update_every 10000 \
    --self_training_rollout_steps 10 \
    --is_grf_overfitting_test 0 \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 1 \
    --time_start 250.0 \
    --time_end 300.0 \
    --dt_data 1.0 \
    --Nx 512 --Ny 256 \
    --modes_x 128 --modes_y 128 \
    --width 80 \
    --n_layers 12 \
    --K 4 \
    --output_dim 10 \
    --rollout_dt 0.1 \
    --time_integrator crank_nicolson \
    --input_noise_scale 0.0 \
    --lr 0.002 \
    --batch_size 10 \
    --num_iterations 100000 \
    --log_every 500 \
    --eval_every 5000 \
    --eval_rollout_steps 10 \
    --seed 42 \
    --use_accelerate 1
