#!/bin/bash
# =============================================================================
# Exp20: Learnable GRF (10k) + Soft-L∞ Loss (Multi-GPU via Accelerate)
# 
# Based on exp19 + soft-L∞ regularization.
# soft-L∞ is now RMS-normalized and per-sample (updated implementation).
# 
# Phase 1 (steps 0-9999): Pure GRF training with PDE loss + soft-L∞
#   - GRF parameters (alpha/tau) are fixed
# Phase 2 (steps 10000+): Learnable GRF mode
#   - GRF alpha/tau become trainable via PDE loss gradient
#   - Separate optimizer with smaller lr (model_lr * 0.1)
# 
# Time integrator: Crank-Nicolson (fixed)
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp20_learnable_grf_10k_soft_linf"
GPU_IDS=${1:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Training configuration
NUM_ITERATIONS=200000

# Learnable GRF configuration
LEARNABLE_GRF_START=10000
LEARNABLE_GRF_LR_RATIO=0.1
LEARNABLE_GRF_REG_WEIGHT=0.01
LEARNABLE_GRF_LOG_EVERY=500

# Soft-L∞ loss configuration (RMS-normalized, per-sample)
SOFT_LINF_WEIGHT=0.1
SOFT_LINF_BETA=10.0

echo "========================================================================="
echo "  Exp20: Learnable GRF (10k) + Soft-L∞ (Multi-GPU via Accelerate)"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  Soft-L∞ loss: weight=$SOFT_LINF_WEIGHT, beta=$SOFT_LINF_BETA"
echo "    (RMS-normalized, per-sample, per-field)"
echo ""
echo "  Phase 1: steps 0-$((LEARNABLE_GRF_START-1))"
echo "    - Pure GRF input, PDE loss + soft-L∞ (Crank-Nicolson)"
echo "    - GRF parameters (alpha/tau) are FIXED"
echo ""
echo "  Phase 2: steps ${LEARNABLE_GRF_START}-$((NUM_ITERATIONS-1))"
echo "    - GRF alpha/tau become LEARNABLE"
echo "    - GRF lr = model_lr * $LEARNABLE_GRF_LR_RATIO"
echo "    - Regularization weight: $LEARNABLE_GRF_REG_WEIGHT"
echo ""
echo "  Time integrator: Crank-Nicolson (fixed)"
echo "  Eval every 2000 steps, checkpoint every 10000 steps"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/zhangtao/envs/rae/bin/accelerate launch --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    --main_process_port=29520 \
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
    --learnable_grf 1 \
    --learnable_grf_start_step $LEARNABLE_GRF_START \
    --learnable_grf_lr_ratio $LEARNABLE_GRF_LR_RATIO \
    --learnable_grf_alpha_min 1.0 \
    --learnable_grf_alpha_max 6.0 \
    --learnable_grf_tau_min 0.5 \
    --learnable_grf_tau_max 20.0 \
    --learnable_grf_reg_weight $LEARNABLE_GRF_REG_WEIGHT \
    --learnable_grf_accum_steps 1 \
    --learnable_grf_log_every $LEARNABLE_GRF_LOG_EVERY \
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
