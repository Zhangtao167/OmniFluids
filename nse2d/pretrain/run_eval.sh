#!/bin/bash
# =============================================================================
# Universal evaluation script for 5-field MHD OmniFluids
# 
# Usage:
#   ./run_eval.sh /path/to/checkpoint.pt [GPU]
#   
# Examples:
#   ./run_eval.sh results/exp10_self_training_4gpu/.../model/latest-xxx.pt
#   ./run_eval.sh results/exp10_self_training_4gpu/.../model/latest-xxx.pt cuda:1
#
# Auto-find checkpoint:
#   ./run_eval.sh results/exp10_self_training_4gpu/  # finds best/latest automatically
# =============================================================================

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path_or_exp_dir> [device]"
    echo ""
    echo "Examples:"
    echo "  $0 results/exp10_self_training_4gpu/.../model/best-xxx.pt"
    echo "  $0 results/exp10_self_training_4gpu/  # auto-find checkpoint"
    echo "  $0 results/exp10_self_training_4gpu/ cuda:1"
    exit 1
fi

# Activate environment
source /opt/conda/bin/activate 2>/dev/null || true
conda activate /zhangtao/envs/rae 2>/dev/null || true
cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

CKPT_INPUT="$1"
DEVICE=${2:-"cuda:0"}

# Function to find checkpoint in a directory
find_checkpoint() {
    local dir="$1"
    local ckpt=""
    
    # Try to find best checkpoint first
    ckpt=$(find "$dir" -name "best-*.pt" 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        echo "$ckpt"
        return
    fi
    
    # Fallback to latest
    ckpt=$(find "$dir" -name "latest-*.pt" 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        echo "$ckpt"
        return
    fi
    
    # Try any .pt file in model/ subdirectory
    ckpt=$(find "$dir" -path "*/model/*.pt" 2>/dev/null | head -1)
    echo "$ckpt"
}

# Resolve checkpoint path
if [ -f "$CKPT_INPUT" ]; then
    # Direct path to checkpoint file
    CKPT_PATH="$CKPT_INPUT"
elif [ -d "$CKPT_INPUT" ]; then
    # Directory - try to find checkpoint
    echo "Searching for checkpoint in: $CKPT_INPUT"
    CKPT_PATH=$(find_checkpoint "$CKPT_INPUT")
    if [ -z "$CKPT_PATH" ]; then
        echo "ERROR: No checkpoint found in $CKPT_INPUT"
        exit 1
    fi
else
    echo "ERROR: Path not found: $CKPT_INPUT"
    exit 1
fi

echo "========================================================================="
echo "  5-field MHD OmniFluids Evaluation"
echo "========================================================================="
echo "  Checkpoint: $CKPT_PATH"
echo "  Device: $DEVICE"
echo "========================================================================="
echo ""

# Run inference (evaluates on both MHD and GRF test sets by default)
python inference.py \
    --checkpoint "$CKPT_PATH" \
    --n_rollout_steps 10 \
    --time_start 250.0 \
    --dt_data 1.0 \
    --device "$DEVICE" \
    --save_plots 1

echo ""
echo "========================================================================="
echo "  Evaluation complete!"
echo "========================================================================="
