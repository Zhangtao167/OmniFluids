#!/bin/bash
# =============================================================================
# 实验: 纯 Offline + PDE Loss 训练
# 
# 特点:
#   - 完全使用 offline 仿真数据 (5field_mhd_batch)
#   - 纯 PDE loss (physics_loss_weight=1.0, supervised_loss_weight=0.0)
#   - 同时在 MHD 和 GRF 测试集上评估
#   - 每 5000 step 评估和保存 checkpoint
# =============================================================================

set -e

# Activate environment
# source /opt/conda/bin/activate
# conda activate /zhangtao/envs/rae || true
# cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Data paths
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

# Experiment name
EXP_NAME="exp_offline_pde_only"

# GPU configuration (可以通过参数覆盖)
GPUS=${1:-"0,1,2,3"}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_PROCESSES=${#GPU_ARR[@]}

echo "========================================================================="
echo "  实验: 纯 Offline + PDE Loss 训练"
echo "========================================================================="
echo "  GPUs: $GPUS (num_processes: $NUM_PROCESSES)"
echo "  Training data: $DATA_PATH"
echo "  MHD test data: $EVAL_DATA_PATH"
echo "  GRF test data: $EVAL_GRF_PATH"
echo ""
echo "  Loss: PDE loss only (physics_loss_weight=1.0)"
echo "  Eval: every 5000 steps on both MHD and GRF test sets"
echo "  Checkpoint: every 5000 steps"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPUS

accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" --main_process_port=29500 main.py \
    --mode train \
    --use_accelerate 1 \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --eval_grf_data_path "$EVAL_GRF_PATH" \
    --data_mode "offline" \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
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
    --dealias_input 1 \
    --dealias_rhs 0 \
    --input_noise_scale 0.001 \
    --lr 0.002 \
    --batch_size 10 \
    --num_iterations 200000 \
    --log_every 100 \
    --eval_every 5000 \
    --checkpoint_every 5000 \
    --eval_rollout_steps 10 \
    --seed 42

echo ""
echo "========================================================================="
echo "  Training complete!"
echo "========================================================================="
