#!/bin/bash
# =============================================================================
# Generate GRF-initialized test set for 5-field MHD
# Uses GRF for initial conditions + physical solver for evolution
# =============================================================================
# Usage:
#   ./run_generate_grf_testset.sh [GPU_ID] [N_SAMPLES] [N_STEPS]
#
# Examples:
#   ./run_generate_grf_testset.sh 4          # GPU 4, 10 samples, 50 steps
#   ./run_generate_grf_testset.sh 5 10 50    # GPU 5, 10 samples, 50 steps
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

GPU=${1:-0}
N_SAMPLES=${2:-10}
N_STEPS=${3:-50}

# Training data path (for deriving GRF field_scales to match training)
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"

# Calculate estimated runtime (rough: ~50 sec per trajectory for 50 steps)
# dt_sim=0.002 (default), steps_per_frame=500, total 500*50=25000 RK4 steps per traj
EST_SECONDS=$((N_SAMPLES * 50))
EST_MINUTES=$(( (EST_SECONDS + 59) / 60 ))

echo "========================================================================="
echo "  Generating GRF Test Set"
echo "========================================================================="
echo "  GPU: cuda:$GPU"
echo "  Samples: $N_SAMPLES"
echo "  Steps: $N_STEPS (dt_data=1.0s, total ${N_STEPS}s simulation per traj)"
echo "  Output: /zhangtao/project2026/OmniFluids/nse2d/data/grf_testset"
echo ""
echo "  GRF field_scales derived from: $DATA_PATH"
echo "  (This ensures GRF matches training data distribution)"
echo ""
echo "  Estimated time: ~${EST_MINUTES} minutes"
echo "========================================================================="
echo ""

# Check GPU memory
echo "GPU Status (GPU $GPU):"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv -i $GPU 2>/dev/null || \
    echo "  [WARNING] Could not query GPU $GPU"
echo ""

# Set CUDA_VISIBLE_DEVICES to use only the specified GPU
# This maps physical GPU $GPU to cuda:0 inside Python
export CUDA_VISIBLE_DEVICES=$GPU

/zhangtao/envs/rae/bin/python generate_grf_testset.py \
    --n-samples $N_SAMPLES \
    --n-steps $N_STEPS \
    --dt-data 1.0 \
    --device "cuda:0" \
    --output-dir /zhangtao/project2026/OmniFluids/nse2d/data/grf_testset \
    --base-seed 1000 \
    --data-path "$DATA_PATH"

echo ""
echo "Done! Dataset saved to /zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/"
