#!/bin/bash
# 5-field MHD OmniFluids: Training and Inference
#
# Usage:
#   bash run.sh train      [GPU_ID] [EXP_NAME]
#   bash run.sh inference  [GPU_ID] [CHECKPOINT_PATH] [EXP_NAME]
#
# To switch data mode, change DATA_MODE below:
#   offline     — mhd_sim pre-generated data only (default)
#   online      — GRF random initial conditions only
#   staged      — online warmup then offline
#   alternating — cycle online/offline

set -e

ARGS=("$@")
set --

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

cd "$(dirname "$0")"

MODE=${ARGS[0]:-train}
GPU=${ARGS[1]:-0}
ARG3=${ARGS[2]:-mhd5_omnifluids_v1}

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

# ---- Data mode config (edit here to switch) ----
DATA_MODE="offline"
ONLINE_WARMUP_STEPS=50000
ALTERNATE_ONLINE_STEPS=20000
ALTERNATE_OFFLINE_STEPS=10000

if [ "$MODE" = "train" ]; then
    EXP_NAME="$ARG3"
    echo "=== Training [$DATA_MODE]: $EXP_NAME on cuda:$GPU ==="
    python main.py \
        --mode train \
        --device "cuda:$GPU" \
        --exp_name "$EXP_NAME" \
        --data_path "$DATA_PATH" \
        --eval_data_path "$EVAL_DATA_PATH" \
        --data_mode "$DATA_MODE" \
        --online_warmup_steps "$ONLINE_WARMUP_STEPS" \
        --alternate_online_steps "$ALTERNATE_ONLINE_STEPS" \
        --alternate_offline_steps "$ALTERNATE_OFFLINE_STEPS" \
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
        --batch_size 8 \
        --num_iterations 200000 \
        --log_every 100 \
        --eval_every 500 \
        --eval_rollout_steps 10 \
        --seed 0

elif [ "$MODE" = "inference" ]; then
    CKPT="$ARG3"
    EXP_NAME=${ARGS[3]:-mhd5_omnifluids_v1}
    echo "=== Inference from $CKPT on cuda:$GPU ==="
    python main.py \
        --mode inference \
        --device "cuda:$GPU" \
        --checkpoint "$CKPT" \
        --data_path "$DATA_PATH" \
        --eval_data_path "$EVAL_DATA_PATH" \
        --time_start 250.0 \
        --time_end 300.0 \
        --rollout_dt 0.1 \
        --eval_rollout_steps 50 \
        --exp_name "$EXP_NAME"

else
    echo "Usage:"
    echo "  bash run.sh train      [GPU_ID] [EXP_NAME]"
    echo "  bash run.sh inference  [GPU_ID] [CHECKPOINT_PATH] [EXP_NAME]"
    exit 1
fi
