#!/bin/bash
# =============================================================================
# Exp6: GRF Online Warmup -> Offline Staged Training
# Stage 1: 50000 steps online (GRF, physics loss only)
# Stage 2: offline (mhd_sim data, physics loss only)
# Eval: mhd_sim test set (trajectory rollout L2 error)
# Radial mask ON, no abs constraint on n/Ti
# =============================================================================

set -e

# source /opt/conda/bin/activate
# conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp6_grf_staged"
GPU_ID=${1:-0}

echo "========================================================================="
echo "  Exp6: GRF Online Warmup -> Offline Staged Training"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Stage 1: 50000 steps online (GRF, physics loss)"
echo "  Stage 2: offline (mhd_sim data, physics loss)"
echo "  Eval: mhd_sim test set (trajectory rollout L2)"
echo "  Radial mask: ON (x=[180,330])"
echo "  Abs constraint: OFF"
echo "========================================================================="
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "staged" \
    --online_warmup_steps 50000 \
    --is_grf_overfitting_test 0 \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 1 \
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
    --input_noise_scale 0.0 \
    --lr 0.002 \
    --batch_size 4 \
    --num_iterations 200000 \
    --log_every 100 \
    --eval_every 2000 \
    --eval_rollout_steps 10 \
    --seed 42
