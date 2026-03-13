#!/bin/bash
# =============================================================================
# Exp25: Offline PDE-only control with exp22-style batch/multistep settings
#
# Purpose:
#   - Keep the offline PDE-only setup from run_exp_offline_pde.sh
#   - Only align batch_size and training multi-step rollout settings with exp22
#   - Help isolate whether batch_size=2 and multi-step PDE rollout itself
#     changes behavior, without self-training / learnable GRF / soft-Linf
#
# Compared with run_exp_offline_pde.sh:
#   - batch_size: 8 -> 2
#   - multi_step_pde_loss: 0 -> 1
#   - multi_step_pde_n: 3
#   - multi_step_pde_detach: 1
# Everything else is kept the same as the offline PDE baseline.
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Data paths
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

# Experiment name
EXP_NAME="exp25_offline_pde_multistep_batch2_single_gpu"

# GPU configuration (single-process control run)
GPU=${1:-"0"}

# Match exp22 only on batch size and multi-step rollout settings
BATCH_SIZE=2
MULTI_STEP_PDE_LOSS=1
MULTI_STEP_PDE_N=3
MULTI_STEP_PDE_DETACH=1

echo "========================================================================="
echo "  Exp25: Offline PDE-only control (batch/multistep matched to exp22)"
echo "========================================================================="
echo "  GPU: $GPU (single-process control run)"
echo "  Training data: $DATA_PATH"
echo "  MHD test data: $EVAL_DATA_PATH"
echo "  GRF test data: $EVAL_GRF_PATH"
echo ""
echo "  Base setup: same as run_exp_offline_pde.sh"
echo "  Control changes vs baseline:"
echo "    batch_size=$BATCH_SIZE"
echo "    multi_step_pde_loss=$MULTI_STEP_PDE_LOSS"
echo "    multi_step_pde_n=$MULTI_STEP_PDE_N"
echo "    multi_step_pde_detach=$MULTI_STEP_PDE_DETACH"
echo ""
echo "  Loss: PDE loss only (physics_loss_weight=1.0)"
echo "  Eval: every 5000 steps on both MHD and GRF test sets"
echo "  Checkpoint: every 5000 steps"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPU

/zhangtao/envs/rae/bin/python main.py \
    --mode train \
    --use_accelerate 0 \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --eval_grf_data_path "$EVAL_GRF_PATH" \
    --data_mode "offline" \
    --physics_loss_weight 1.0 \
    --supervised_loss_weight 0.0 \
    --mae_weight 0.0 \
    --multi_step_pde_loss $MULTI_STEP_PDE_LOSS \
    --multi_step_pde_n $MULTI_STEP_PDE_N \
    --multi_step_pde_detach $MULTI_STEP_PDE_DETACH \
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
    --batch_size $BATCH_SIZE \
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
