#!/bin/bash
# =============================================================================
# 实验1c: 纯监督 Loss + 线性插值子步监督 - 多卡版本
# model_dt=0.1s (rollout_dt=0.1s), dt_data=1.0s
# 用 x_t 和 x_{t+1} 之间的线性插值，在线采样 10 段 0.1s 一步伪样本
# 设计上使用 output_dim=1，不做带梯度的10步自回归训练，显存更稳
# =============================================================================

set -e

# source /opt/conda/bin/activate
# conda activate /zhangtao/envs/rae || true
# cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp1c_pure_supervised_dt0p1_interp_multigpu"
GPUS=${1:-"0,1,2,3"}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_PROCESSES=${#GPU_ARR[@]}

echo "========================================================================="
echo "  实验1c: 纯监督Loss + 线性插值子步监督 (model_dt=0.1s) - 多卡"
echo "========================================================================="
echo "  GPUs: $GPUS"
echo "  num_processes: $NUM_PROCESSES"
echo "  rollout_dt=0.1s, dt_data=1.0s -> sampled one-step pseudo-pairs from 10 linear segments"
echo "  Physics Loss: DISABLED"
echo "  Supervised Loss: MSE (weight=1.0)"
echo "  Intermediate targets: on-the-fly linear interpolation between x_t and x_{t+1}"
echo "  output_dim=1 -> training/inference horizons are aligned at 0.1s"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPUS

accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" --main_process_port=29504 main.py \
    --mode train \
    --use_accelerate 1 \
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
    --supervised_use_interpolation 0 \
    --supervised_pair_interp_steps 10 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 0 \
    --time_start 250.0 \
    --time_end 300.0 \
    --Nx 512 --Ny 256 \
    --modes_x 128 --modes_y 128 \
    --width 80 \
    --n_layers 12 \
    --K 4 \
    --output_dim 1 \
    --rollout_dt 0.1 \
    --time_integrator crank_nicolson \
    --input_noise_scale 0.001 \
    --lr 0.002 \
    --batch_size 10 \
    --num_iterations 150000 \
    --log_every 100 \
    --eval_every 1000 \
    --eval_rollout_steps 10 \
    --seed 42
