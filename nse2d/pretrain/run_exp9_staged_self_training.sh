#!/bin/bash
# Experiment 9: Staged mode + Self-training
#
# This experiment combines staged training with self-training:
# - Phase 1 (steps 0-9999): Raw GRF + PDE loss
# - Phase 2 (steps 10000-29999): Model-evolved GRF + PDE loss (self-training)
#   - Model weights refreshed every 5000 steps
# - Phase 3 (steps 30000+): Switch to offline real simulation data
#
# This is a curriculum learning approach:
# 1. Start with simple random data (GRF)
# 2. Transition to self-generated physical-like data
# 3. Finally fine-tune on real simulation data

set -e
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Use rae conda environment
PYTHON=/opt/conda/envs/rae/bin/python

$PYTHON main.py --mode train \
    --data_mode staged \
    --online_warmup_steps 30000 \
    --self_training_start_step 10000 \
    --self_training_update_every 5000 \
    --self_training_rollout_steps 10 \
    --physics_loss_weight 1.0 \
    --num_iterations 100000 \
    --batch_size 4 \
    --lr 0.002 \
    --eval_every 1000 \
    --log_every 100 \
    --device cuda:0 \
    --exp_name exp9_staged_self_training
