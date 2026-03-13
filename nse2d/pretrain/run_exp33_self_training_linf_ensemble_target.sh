#!/bin/bash
# =============================================================================
# Exp33: Self-Training + Soft-Linf + Ensemble RHS Target Smoothing
#
# Based on exp32, with one key addition:
#   - use_ensemble_target=1
#   - Expand each clean sample into a local noisy ensemble
#   - Use the ensemble-mean RHS as the physics target
#
# Ensemble protocol:
#   - 1 clean member + (E-1) noisy members per clean sample
#   - Gaussian perturbation only
#   - Physics-only path (no supervised loss)
#   - No temporal-window averaging
#
# We keep exp32's no-multi-step setting to isolate the effect of
# ensemble RHS smoothing as much as possible.
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"
EVAL_GRF_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt"

EXP_NAME="exp33_self_training_linf_ensemble_target"
GPU_IDS=${1:-"2,3"} 
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
MAIN_PROCESS_PORT=29333

# Training configuration
NUM_ITERATIONS=200000

# Self-training configuration (same as exp32)
SELF_TRAINING_START=30000
SELF_TRAINING_UPDATE_EVERY=20000
SELF_TRAINING_ROLLOUT_STEPS=5

# Soft-Linf loss configuration (same as exp32)
SOFT_LINF_WEIGHT=0.1
SOFT_LINF_BETA=10.0

# Multi-step PDE loss configuration (same as exp32: OFF)
MULTI_STEP_PDE_LOSS=0
MULTI_STEP_PDE_N=3
MULTI_STEP_PDE_DETACH=1

# Ensemble RHS smoothing configuration
USE_ENSEMBLE_TARGET=1
ENSEMBLE_NUM_SAMPLES=4
ENSEMBLE_NOISE_SCALE=0.001
ENSEMBLE_KEEP_CLEAN=1

# One clean sample per rank, then expand locally to E members.
# Effective forward batch per rank is roughly ENSEMBLE_NUM_SAMPLES.
BATCH_SIZE=1

echo "========================================================================="
echo "  Exp33: Self-Training + Soft-Linf + Ensemble RHS Target Smoothing"
echo "========================================================================="
echo "  GPU_IDS: $GPU_IDS"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""
echo "  *** Ensemble RHS Target Smoothing ***"
echo "    use_ensemble_target=$USE_ENSEMBLE_TARGET"
echo "    ensemble_num_samples=$ENSEMBLE_NUM_SAMPLES"
echo "    ensemble_noise_scale=$ENSEMBLE_NOISE_SCALE"
echo "    ensemble_keep_clean=$ENSEMBLE_KEEP_CLEAN (1 clean + E-1 noisy)"
echo "    batch_size=$BATCH_SIZE clean sample(s) per rank"
echo ""
echo "  *** Soft-Linf Loss ***"
echo "    soft_linf_weight=$SOFT_LINF_WEIGHT"
echo "    soft_linf_beta=$SOFT_LINF_BETA"
echo ""
echo "  *** Multi-Step PDE Loss ***"
echo "    multi_step_pde_loss=$MULTI_STEP_PDE_LOSS (kept same as exp32)"
echo "    multi_step_pde_n=$MULTI_STEP_PDE_N"
echo "    multi_step_pde_detach=$MULTI_STEP_PDE_DETACH"
echo ""
echo "  Phase 1: steps 0-$((SELF_TRAINING_START-1))"
echo "    - Pure GRF input, PDE + Soft-Linf + Ensemble RHS smoothing"
echo ""
echo "  Phase 2: steps ${SELF_TRAINING_START}-$((NUM_ITERATIONS-1))"
echo "    - Self-training: GRF -> Model($SELF_TRAINING_ROLLOUT_STEPS steps) -> input"
echo "    - Model weights updated every $SELF_TRAINING_UPDATE_EVERY steps"
echo ""
echo "  Time integrator: Crank-Nicolson (fixed)"
echo "  Eval every 2000 steps, checkpoint every 10000 steps"
echo "========================================================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
    --mode train
    --exp_name "$EXP_NAME"
    --data_path "$DATA_PATH"
    --eval_data_path "$EVAL_DATA_PATH"
    --eval_grf_data_path "$EVAL_GRF_DATA_PATH"
    --data_mode "online"
    --is_grf_overfitting_test 0
    --physics_loss_weight 1.0
    --supervised_loss_weight 0.0
    --mae_weight 0.0
    --soft_linf_weight $SOFT_LINF_WEIGHT
    --soft_linf_beta $SOFT_LINF_BETA
    --multi_step_pde_loss $MULTI_STEP_PDE_LOSS
    --multi_step_pde_n $MULTI_STEP_PDE_N
    --multi_step_pde_detach $MULTI_STEP_PDE_DETACH
    --use_ensemble_target $USE_ENSEMBLE_TARGET
    --ensemble_num_samples $ENSEMBLE_NUM_SAMPLES
    --ensemble_noise_scale $ENSEMBLE_NOISE_SCALE
    --ensemble_keep_clean $ENSEMBLE_KEEP_CLEAN
    --grf_use_radial_mask 1
    --self_training_start_step $SELF_TRAINING_START
    --self_training_update_every $SELF_TRAINING_UPDATE_EVERY
    --self_training_rollout_steps $SELF_TRAINING_ROLLOUT_STEPS
    --time_start 250.0
    --time_end 300.0
    --dt_data 1.0
    --Nx 512 --Ny 256
    --modes_x 128 --modes_y 128
    --width 80
    --n_layers 12
    --K 4
    --output_dim 10
    --rollout_dt 0.1
    --time_integrator crank_nicolson
    --input_noise_scale 0.0
    --lr 0.001
    --lr_end 1e-7
    --batch_size $BATCH_SIZE
    --num_iterations $NUM_ITERATIONS
    --log_every 100
    --eval_every 2000
    --eval_rollout_steps 10
    --checkpoint_every 10000
    --seed 42
)

if [ "$NUM_GPUS" -gt 1 ]; then
    /zhangtao/envs/rae/bin/accelerate launch --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=no \
        --main_process_port=$MAIN_PROCESS_PORT \
        main.py \
        --use_accelerate 1 \
        "${COMMON_ARGS[@]}"
else
    /zhangtao/envs/rae/bin/python main.py \
        --use_accelerate 0 \
        "${COMMON_ARGS[@]}"
fi
