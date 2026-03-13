#!/bin/bash
# =============================================================================
# 实验1b: 纯监督Loss训练 (无PDE loss) - 多卡版本
# rollout_dt=1.0s (model_dt=1.0s), output_dim=10
# 使用Offline仿真数据，只用MSE监督loss
# 训练/评估时间尺度匹配：rollout_dt=dt_data=1.0s → n_substeps=1
# =============================================================================

set -e

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp1b_pure_supervised_dt1s_signle_gpu"
GPUS=${1:-"1"}

echo "========================================================================="
echo "  实验1b: 纯监督Loss (rollout_dt=1.0s, output_dim=10) - 多卡"
echo "========================================================================="
echo "  GPUs: $GPUS"
echo "  rollout_dt=1.0s, dt_data=1.0s → n_substeps=1 (训练/评估时间尺度匹配)"
echo "  Physics Loss: DISABLED  |  Supervised Loss: MSE (weight=1.0)"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPUS

/zhangtao/envs/rae/bin/python main.py \
    --mode train \
    --use_accelerate 0 \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --eval_grf_data_path "$EVAL_GRF_PATH" \
    --data_mode "offline" \
    --physics_loss_weight 0.0 \
    --supervised_loss_weight 1.0 \
    --supervised_mse_weight 1.0 \
    --supervised_mae_weight 0.0 \
    --supervised_n_substeps 1 \
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
    --rollout_dt 1.0 \
    --time_integrator crank_nicolson \
    --input_noise_scale 0.001 \
    --lr 0.002 \
    --batch_size 8 \
    --num_iterations 200000 \
    --log_every 100 \
    --eval_every 500 \
    --eval_rollout_steps 10 \
    --seed 42
