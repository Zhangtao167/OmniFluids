#!/bin/bash
# =============================================================================
# Exp4: Fixed GRF Overfitting Test
# Train on a single fixed GRF sample to test model fitting capacity
# Radial mask ON, no abs constraint on n/Ti
# Evaluation: mhd_sim test set (for comparison with other experiments)
# =============================================================================

set -e

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

EXP_NAME="exp4_fixed_grf_overfitting"
GPU_ID=${1:-0}

# Data paths: DATA_PATH for GRF scale stats, EVAL_DATA_PATH for test evaluation
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

echo "========================================================================="
echo "  Exp4: Fixed GRF Overfitting Test"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Training Data: Single FIXED GRF sample (seed=42)"
echo "  Eval Data: mhd_sim test set (for fair comparison)"
echo "  Mode: online (using GRF generator)"
echo "  Radial mask: ON (x=[180,330])"
echo "  Abs constraint: OFF"
echo "  Physics Loss: ENABLED (weight=1.0)"
echo "  Supervised Loss: DISABLED"
echo "  Expected: training loss converges, but eval error may not decrease"
echo "========================================================================="
echo ""

python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "online" \
    --is_grf_overfitting_test 1 \
    --grf_overfitting_seed 42 \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
    --grf_use_radial_mask 1 \
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
    --lr 0.002 \
    --batch_size 4 \
    --num_iterations 20000 \
    --log_every 100 \
    --eval_every 500 \
    --eval_rollout_steps 10 \
    --seed 42
