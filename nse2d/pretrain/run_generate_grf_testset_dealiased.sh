#!/bin/bash
# =============================================================================
# Generate GRF Test Set with DEALIASED Initial Conditions
# 
# IMPORTANT: dealias_init=True ensures test set matches training (dealias_input=True)
# 
# Output file: grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt
# =============================================================================

set -e

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# Configuration
GPU_ID=${1:-0}
N_SAMPLES=10
N_STEPS=50
DT_DATA=1.0
BASE_SEED=1000
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
OUTPUT_DIR="/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset"

echo "========================================================================="
echo "  Generate GRF Test Set (DEALIASED)"
echo "========================================================================="
echo "  GPU: cuda:${GPU_ID}"
echo "  N_SAMPLES: ${N_SAMPLES}"
echo "  N_STEPS: ${N_STEPS}"
echo "  DT_DATA: ${DT_DATA}s"
echo "  BASE_SEED: ${BASE_SEED}"
echo "  dealias_init: TRUE (matches training dealias_input=True)"
echo "========================================================================="
echo ""

/zhangtao/envs/rae/bin/python generate_grf_testset.py \
    --n-samples ${N_SAMPLES} \
    --n-steps ${N_STEPS} \
    --dt-data ${DT_DATA} \
    --device "cuda:${GPU_ID}" \
    --base-seed ${BASE_SEED} \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done! Test set generated with DEALIASED initial conditions."

