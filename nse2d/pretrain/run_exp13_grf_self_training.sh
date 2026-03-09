#!/bin/bash
# =============================================================================
# Exp13: GRF + Self-Training (Multi-GPU via Accelerate)
# 
# Phase 1 (steps 0-99999): Pure GRF training with physics loss
# Phase 2 (steps 100000+): Self-training mode
#   - GRF -> Model(3 steps) -> evolved state as training input
#   - Model weights refreshed every 20000 steps
# 
# Time integrator: Crank-Nicolson (fixed)
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp13_grf_self_training"
GPU_IDS=${1:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Training configuration
NUM_ITERATIONS=200000
SELF_TRAINING_START=100000
SELF_TRAINING_UPDATE_EVERY=20000
SELF_TRAINING_ROLLOUT_STEPS=3

echo "========================================================================="
echo "  Exp13: GRF + Self-Training (Multi-GPU via Accelerate)"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  Phase 1: steps 0-$((SELF_TRAINING_START-1))"
echo "    - Pure GRF input, PDE loss (Crank-Nicolson)"
echo ""
echo "  Phase 2: steps ${SELF_TRAINING_START}-$((NUM_ITERATIONS-1))"
echo "    - Self-training: GRF -> Model($SELF_TRAINING_ROLLOUT_STEPS steps) -> input"
echo "    - Model weights updated every $SELF_TRAINING_UPDATE_EVERY steps"
echo ""
echo "  Time integrator: Crank-Nicolson (fixed)"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU with specified GPUs
CUDA_VISIBLE_DEVICES=$GPU_IDS /zhangtao/envs/rae/bin/accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    --main_process_port=29513 \
    main.py \
    --mode train \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --eval_grf_data_path "$EVAL_GRF_DATA_PATH" \
    --data_mode "online" \
    --is_grf_overfitting_test 0 \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 1 \
    --self_training_start_step $SELF_TRAINING_START \
    --self_training_update_every $SELF_TRAINING_UPDATE_EVERY \
    --self_training_rollout_steps $SELF_TRAINING_ROLLOUT_STEPS \
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
    --num_iterations $NUM_ITERATIONS \
    --log_every 500 \
    --eval_every 5000 \
    --eval_rollout_steps 10 \
    --seed 42 \
    --use_accelerate 1
