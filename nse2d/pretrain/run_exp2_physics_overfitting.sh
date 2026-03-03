#!/bin/bash
# =============================================================================
# 实验2: PDE Loss Overfitting测试
# 使用单条轨迹，只用PDE loss，检测模型拟合能力
# =============================================================================

set -e

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# 使用训练数据作为测试数据（overfitting test）
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"

EXP_NAME="exp2_physics_overfitting"
GPU_ID=${1:-0}

echo "========================================================================="
echo "  实验2: PDE Loss Overfitting测试"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Data: Single trajectory #0 (train = eval)"
echo "  Physics Loss: ENABLED (weight=1.0)"
echo "  Supervised Loss: DISABLED (weight=0)"
echo "  Expected: train/eval loss should converge to near zero"
echo "========================================================================="
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$DATA_PATH" \
    --is_overfitting_test 1 \
    --overfitting_traj_idx 0 \
    --data_mode "offline" \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
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
    --batch_size 4 \
    --num_iterations 50000 \
    --log_every 100 \
    --eval_every 500 \
    --eval_rollout_steps 10 \
    --seed 42
