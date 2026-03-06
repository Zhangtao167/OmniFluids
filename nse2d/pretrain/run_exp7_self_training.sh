#!/bin/bash
# Experiment 7: Self-training mode
# 
# This experiment demonstrates the self-training data generation mode:
# - Phase 1 (steps 0-9999): Train with raw GRF data
# - Phase 2 (steps 10000+): Train with model-evolved GRF data
#   - GRF -> Current Model (10 steps) -> evolved state -> Training
#   - Model weights in generator are refreshed every 5000 steps
#
# The idea is that:
# 1. Raw GRF may not lie on the physical manifold (doesn't satisfy PDE)
# 2. After model evolution, states are closer to physical solutions
# 3. Self-improvement: better model -> better data -> even better model

set -e
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Use rae conda environment
PYTHON=/opt/conda/envs/rae/bin/python

$PYTHON main.py --mode train \
    --data_mode online \
    --self_training_start_step 10000 \
    --self_training_update_every 5000 \
    --self_training_rollout_steps 10 \
    --physics_loss_weight 1.0 \
    --num_iterations 50000 \
    --batch_size 4 \
    --lr 0.002 \
    --eval_every 1000 \
    --log_every 100 \
    --device cuda:0 \
    --exp_name exp7_self_training
