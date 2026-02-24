#!/bin/bash
# 5-field MHD OmniFluids: Training and Inference
#
# Usage:
#   bash run.sh train      [GPU_ID] [EXP_NAME]
#   bash run.sh inference  [GPU_ID] [CHECKPOINT_PATH] [EXP_NAME]
#
# dt hierarchy:
#   rollout_dt=0.1  (= mhd_sim delta_t, model inference step)
#   train_dt=0.01   (= rollout_dt / output_dim, physics supervision step)
#   dt_data=1.0     (data snapshot interval)
#   n_substeps=10   (= dt_data / rollout_dt, NFE per data step at inference)

set -e

# Save positional arguments before conda activation (which clobbers $@)
ARGS=("$@")
set --

source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae

cd "$(dirname "$0")"

MODE=${ARGS[0]:-train}
GPU=${ARGS[1]:-0}
ARG3=${ARGS[2]:-mhd5_omnifluids_v1}

DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"

if [ "$MODE" = "train" ]; then
    EXP_NAME="$ARG3"
    echo "=== Training: $EXP_NAME on cuda:$GPU ==="
    python main.py \
        --mode train \
        --device "cuda:$GPU" \
        --exp_name "$EXP_NAME" \
        --data_path "$DATA_PATH" \
        --time_start 250.0 \
        --time_end 300.0 \
        --Nx 512 --Ny 256 \
        --modes_x 32 --modes_y 32 \
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
