#!/bin/bash
# =============================================================================
# 实验1a: 纯监督Loss训练 (无PDE loss)
# rollout_dt=0.1s, output_dim=10, 线性插值10帧target
# 使用Offline仿真数据，只用MSE监督loss
# =============================================================================

set -e

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

EXP_NAME="exp1a_pure_supervised_dt01s"
GPU_ID=${1:-0}

echo "========================================================================="
echo "  实验1a: 纯监督Loss (rollout_dt=0.1s, output_dim=10, 线性插值10帧)"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  rollout_dt=0.1s, dt_data=1.0s → 10 interpolated target frames per step"
echo "  Physics Loss: DISABLED  |  Supervised Loss: MSE (weight=1.0)"
echo "========================================================================="
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "offline" \
    --physics_loss_weight 0.0 \
    --supervised_loss_weight 1.0 \
    --supervised_mse_weight 1.0 \
    --supervised_mae_weight 0.0 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 0 \
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
