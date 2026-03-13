#!/bin/bash
# =============================================================================
# Exp16: Soft-L∞ Loss (对标 exp14，但不使用 learnable GRF)
# 
# 验证 soft-L∞ loss 是否有助于控制最大误差
# 配置：soft_linf_weight=0.02, beta=10
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp16_soft_linf"
GPU_IDS=${1:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Training configuration (对标 exp14)
NUM_ITERATIONS=200000

# Soft-L∞ loss configuration
SOFT_LINF_WEIGHT=0.02
SOFT_LINF_BETA=10.0

echo "========================================================================="
echo "  Exp16: Soft-L∞ Loss (对标 exp14，无 learnable GRF)"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  Soft-L∞ loss: weight=$SOFT_LINF_WEIGHT, beta=$SOFT_LINF_BETA"
echo "  Time integrator: Crank-Nicolson"
echo "  与 exp14 的区别: 使用 soft-L∞ loss，不使用 learnable GRF"
echo "========================================================================="
echo ""

# Use accelerate launch for multi-GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/zhangtao/envs/rae/bin/accelerate launch --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    --main_process_port=29516 \
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
    --soft_linf_weight $SOFT_LINF_WEIGHT \
    --soft_linf_beta $SOFT_LINF_BETA \
    --grf_use_radial_mask 1 \
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
