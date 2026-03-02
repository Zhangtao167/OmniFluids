#!/bin/bash
# Staged训练：50000 step online warmup -> offline
# mae_weight=0.1, 无radial mask
# 
# 用法: bash launch_staged_no_radial.sh [GPU_ID]

set -e

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="mhd5_staged_mae01_no_radial"
GPU_ID=${1:-0}

echo "=== Staged Training (50000 online + offline) ==="
echo "  mae_weight=0.1, NO radial mask, GPU=$GPU_ID"
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "staged" \
    --online_warmup_steps 50000 \
    --mae_weight 0.1 \
    --grf_use_radial_mask 0 \
    --grf_use_abs_constraint 1 \
    --time_start 250.0 \
    --time_end 300.0 \
    --Nx 512 --Ny 256 \
    --modes_x 128 --modes_y 128 \
    --width 80 \
    --n_layers 12 \
    --K 4 \
    --output_dim 10 \
    --rollout_dt 0.1 \
    --time_integrator crank_nicolson \
    --input_noise_scale 0.001 \
    --lr 0.002 \
    --batch_size 8 \
    --num_iterations 200000 \
    --log_every 100 \
    --eval_every 500 \
    --eval_rollout_steps 10 \
    --seed 42
