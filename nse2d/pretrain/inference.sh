#!/bin/bash
set -e
ARGS=("$@"); set --
source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae || true
cd "$(dirname "$0")"

CKPT=${ARGS[0]:-"results/mhd5_omnifluids_v1/d10ea74a-02_24_07_51_41-K4-mx128-w80-L12-od10/model/best-d10ea74a-02_24_07_51_41-K4-mx128-w80-L12-od10.pt"}
GPU=${ARGS[1]:-1}

python main.py \
    --mode inference \
    --device "cuda:$GPU" \
    --checkpoint "$CKPT" \
    --data_path "/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt" \
    --eval_data_path "/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt" \
    --time_start 250.0 --time_end 300.0 \
    --rollout_dt 0.1 \
    --eval_rollout_steps 10 \
    --exp_name mhd5_omnifluids_v1
