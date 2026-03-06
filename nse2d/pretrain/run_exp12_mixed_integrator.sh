#!/bin/bash
# =============================================================================
# Exp12: Mixed Integrator (Euler -> CN) with GRF Staged Training
# 
# Mixed Integrator Settings:
#   - Euler weight starts at 1.0 (pure Euler)
#   - Annealing starts at step 20000
#   - Half-life = 20000 steps (exponential decay)
#   - Decays towards 0.0 (pure CN)
#   - At step 80000: 12.5% Euler + 87.5% CN
#
# Data Mode: staged (same as exp6)
#   - Stage 1: online GRF warmup
#   - Stage 2: offline mhd_sim data
#
# GPUs: 0,1 (2 GPUs via Accelerate)
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp12_mixed_integrator"
GPU_IDS=${1:-"0,1"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "========================================================================="
echo "  Exp12: Mixed Integrator (Euler -> CN) with GRF Staged Training"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  Mixed Integrator:"
echo "    - euler_weight_init: 1.0 (pure Euler)"
echo "    - euler_weight_min: 0.0 (pure CN)"
echo "    - euler_anneal_start: 20000"
echo "    - euler_half_life: 20000"
echo ""
echo "  Data Mode: staged"
echo "    - Stage 1: 100000 steps online (GRF, physics loss)"
echo "    - Stage 2: offline (mhd_sim data, physics loss)"
echo "  Eval: mhd_sim test set"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU with specified GPUs
CUDA_VISIBLE_DEVICES=$GPU_IDS /zhangtao/envs/rae/bin/accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    --main_process_port=29502 \
    main.py \
    --mode train \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "staged" \
    --online_warmup_steps 100000 \
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
    --use_mixed_integrator 1 \
    --euler_weight_init 1.0 \
    --euler_weight_min 0.0 \
    --euler_anneal_start 20000 \
    --euler_half_life 20000 \
    --input_noise_scale 0.0 \
    --lr 0.002 \
    --batch_size 10 \
    --num_iterations 150000 \
    --log_every 500 \
    --eval_every 5000 \
    --eval_rollout_steps 10 \
    --seed 42 \
    --use_accelerate 1
