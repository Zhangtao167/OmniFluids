#!/bin/bash
# =============================================================================
# Exp6: GRF Online Warmup -> Offline Staged Training (Multi-GPU via Accelerate)
# Stage 1: 50000 steps online (GRF, physics loss only)
# Stage 2: offline (mhd_sim data, physics loss only)
# Eval: mhd_sim test set (trajectory rollout L2 error)
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp6_grf_staged_multigpu"
GPU_IDS=${1:-"1,2"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "========================================================================="
echo "  Exp6: GRF Staged Training (Multi-GPU via Accelerate)"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  Stage 1: 500 steps online (GRF, physics loss) - SHORT TEST"
echo "  Stage 2: offline (mhd_sim data, physics loss)"
echo "  Eval: mhd_sim test set"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU with specified GPUs
CUDA_VISIBLE_DEVICES=$GPU_IDS /zhangtao/envs/rae/bin/accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    main.py \
    --mode train \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "staged" \
    --online_warmup_steps 300 \
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
    --batch_size 4 \
    --num_iterations 500 \
    --log_every 50 \
    --eval_every 200 \
    --eval_rollout_steps 5 \
    --seed 42 \
    --use_accelerate 1
