#!/bin/bash
# Experiment 8: External pretrained model mode (Transfer Learning)
#
# This experiment uses an external pretrained model to generate training data:
# - GRF -> Pretrained Model (10 steps) -> evolved state -> Training
# - The pretrained model is fixed (not updated during training)
#
# Use this when you have a good pretrained model and want to:
# 1. Fine-tune a new model with model-evolved data
# 2. Transfer learning from a previous experiment
#
# USAGE: Replace CHECKPOINT_PATH with your actual checkpoint path

set -e
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Use rae conda environment
PYTHON=/opt/conda/envs/rae/bin/python

# Example checkpoint path - replace with your actual path
CHECKPOINT_PATH="results/exp6_grf_staged/YOUR_RUN_TAG/model/best-YOUR_RUN_TAG.pt"

$PYTHON main.py --mode train \
    --data_mode online \
    --pretrained_model_path ${CHECKPOINT_PATH} \
    --pretrained_rollout_steps 10 \
    --physics_loss_weight 1.0 \
    --num_iterations 50000 \
    --batch_size 4 \
    --lr 0.002 \
    --eval_every 1000 \
    --log_every 100 \
    --device cuda:0 \
    --exp_name exp8_pretrained_transfer
