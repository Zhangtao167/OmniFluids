#!/bin/bash
# =============================================================================
# Exp15: Multi-step PDE Loss (对标 exp14，但不使用 learnable GRF)
# 
# 验证多步 rollout PDE loss 是否有助于训练
# 配置：rollout 2 步，不 detach 中间状态（完整梯度链）
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp15_multistep_pde"
GPU_IDS=${1:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Training configuration (对标 exp14)
NUM_ITERATIONS=200000

# Multi-step PDE loss configuration
MULTI_STEP_N=2
MULTI_STEP_DETACH=0  # 0=完整梯度链

echo "========================================================================="
echo "  Exp15: Multi-step PDE Loss (对标 exp14，无 learnable GRF)"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  Multi-step PDE loss: N=$MULTI_STEP_N, detach=$MULTI_STEP_DETACH"
echo "  Time integrator: Crank-Nicolson"
echo "  与 exp14 的区别: 使用 multi-step PDE loss，不使用 learnable GRF"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/zhangtao/envs/rae/bin/accelerate launch --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    --main_process_port=29515 \
    main.py \
    --mode train \
    --use_accelerate 1 \
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
    --multi_step_pde_loss 1 \
    --multi_step_pde_n $MULTI_STEP_N \
    --multi_step_pde_detach $MULTI_STEP_DETACH \
    --learnable_grf 0 \
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
    --lr 0.001 \
    --lr_end 1e-7 \
    --batch_size 10 \
    --num_iterations $NUM_ITERATIONS \
    --log_every 100 \
    --eval_every 2000 \
    --eval_rollout_steps 10 \
    --checkpoint_every 10000 \
    --seed 42
