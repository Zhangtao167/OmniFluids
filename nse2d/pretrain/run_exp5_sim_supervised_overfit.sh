#!/bin/bash
# =============================================================================
# Exp5: Single Trajectory Supervised Overfitting Test
# Train on ONE simulation trajectory, supervised loss only, no PDE loss
# model_dt = data_dt = 1.0s, output_dim = 1 (no interpolation needed)
# Goal: verify the model can overfit a single trajectory with supervised loss
# =============================================================================

# set -e

source /opt/conda/bin/activate
# conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp5_sim_supervised_overfit"
GPU_ID=${1:-0}
TRAJ_IDX=${2:-0}   # which trajectory to overfit (default: 0)

echo "========================================================================="
echo "  Exp5: Single Trajectory Supervised Overfitting Test"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Train data: ONE simulation trajectory (traj_idx=$TRAJ_IDX)"
echo "  Eval data: mhd_sim test set"
echo "  model_dt = dt_data = 1.0s, output_dim = 1 (no interpolation)"
echo "  Physics Loss: DISABLED"
echo "  Supervised Loss: MSE only (weight=1.0)"
echo "  Expected: training MSE should converge to near zero (model memorizes)"
echo "========================================================================="
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "offline" \
    --is_overfitting_test 1 \
    --overfitting_traj_idx "$TRAJ_IDX" \
    --physics_loss_weight 0.0 \
    --supervised_loss_weight 1.0 \
    --supervised_mse_weight 1.0 \
    --supervised_mae_weight 0.0 \
    --mae_weight 0.0 \
    --time_start 250.0 \
    --time_end 300.0 \
    --dt_data 1.0 \
    --Nx 512 --Ny 256 \
    --modes_x 128 --modes_y 128 \
    --width 80 \
    --n_layers 12 \
    --K 4 \
    --output_dim 1 \
    --rollout_dt 1.0 \
    --time_integrator crank_nicolson \
    --input_noise_scale 0.0 \
    --lr 0.002 \
    --batch_size 4 \
    --num_iterations 20000 \
    --log_every 100 \
    --eval_every 500 \
    --eval_rollout_steps 10 \
    --seed 42
