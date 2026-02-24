#!/bin/bash
# 完整流程: 生成数据 → 训练 OmniFluids
# 使用方法: bash run_all.sh
source /opt/conda/bin/activate
conda activate /zhangtao/envs/rae
cd /zhangtao/project2026/OmniFluids/kse2d/pretrain

set -e

MHD_SIM_ROOT="/zhangtao/project2026/mhd_sim"
OMNIFLUIDS_ROOT="/zhangtao/project2026/OmniFluids"
PRETRAIN_DIR="${OMNIFLUIDS_ROOT}/kse2d/pretrain"

# 数据输出路径 (在 mhd_sim 内)
TRAIN_DATA_DIR="${MHD_SIM_ROOT}/outputs/numerical/data/hw_train"
TEST_DATA_DIR="${MHD_SIM_ROOT}/outputs/numerical/data/hw_test"
TRAIN_DATA="${TRAIN_DATA_DIR}/hw_dataset.pt"
TEST_DATA="${TEST_DATA_DIR}/hw_dataset.pt"

DEVICE="cuda:1"

# ---- 实验控制 ----
# train_loss_type:          mhd_sim(默认/当前最优) | omnifluids
# operator_discretization:  mhd_sim(默认/当前最优) | spectral
# USE_INFERENCE_TRAIN:      0=Tp路径(当前最优) | 1=inference rollout路径
# TRAIN_UNROLL_STEPS:       仅 USE_INFERENCE_TRAIN=1 时生效
# MODEL_SIZE:               large(68M原始) | small(2M轻量)
TRAIN_LOSS_TYPE="omnifluids"
OPERATOR_DISC="spectral"
USE_INFERENCE_TRAIN=0
TRAIN_UNROLL_STEPS=1
MODEL_SIZE="large"   # large=68M | small=2M

# ---- 模型架构参数 ----
if [ "${MODEL_SIZE}" = "small" ]; then
    # 轻量版 ~2M params (对标 mhd_sim UNet)
    WIDTH=64
    MODES=16
    N_LAYERS=4
    K=2
    F_NU_HIDDEN=64
    OUTPUT_DIM=40
else
    # 原始版 ~68M params
    WIDTH=128
    MODES=32
    N_LAYERS=8
    K=4
    F_NU_HIDDEN=128
    OUTPUT_DIM=40
fi

# ===========================================================
# Step 1: 生成训练数据 (100 samples, base-seed=0)
# ===========================================================
# echo "===== Step 1: Generating training data (100 samples) ====="
# cd "${MHD_SIM_ROOT}"
# python -m numerical.scripts.hw_eq_batch \
#     --n-samples 100 \
#     --base-seed 0 \
#     --output-dir "${TRAIN_DATA_DIR}" \
#     --device "${DEVICE}"
# echo "Training data saved: ${TRAIN_DATA}"

# # ===========================================================
# # Step 2: 生成测试数据 (100 samples, base-seed=1000)
# # ===========================================================
# echo "===== Step 2: Generating test data (100 samples) ====="
# python -m numerical.scripts.hw_eq_batch \
#     --n-samples 100 \
#     --base-seed 1000 \
#     --output-dir "${TEST_DATA_DIR}" \
#     --device "${DEVICE}"
# echo "Test data saved: ${TEST_DATA}"

# ===========================================================
# Step 3: 训练 OmniFluids
# ===========================================================
echo "===== Step 3: Training OmniFluids ====="
echo "MODEL_SIZE=${MODEL_SIZE}: width=${WIDTH}, modes=${MODES}, layers=${N_LAYERS}, K=${K}"
cd "${PRETRAIN_DIR}"
CMD=(python main.py \
    --train_data "${TRAIN_DATA}" \
    --test_data "${TEST_DATA}" \
    --device "${DEVICE}" \
    --rollout_DT 0.1 \
    --output_dim "${OUTPUT_DIM}" \
    --test_steps 10 \
    --lr 5e-4 \
    --input_noise_std 0.001 \
    --train_loss_type "${TRAIN_LOSS_TYPE}" \
    --rhs_loss_weight 0.0 \
    --operator_discretization "${OPERATOR_DISC}" \
    --mhd_sim_root "${MHD_SIM_ROOT}" \
    --dt_data 1.0 \
    --width "${WIDTH}" \
    --modes "${MODES}" \
    --n_layers "${N_LAYERS}" \
    --K "${K}" \
    --f_nu_hidden "${F_NU_HIDDEN}")

# 训练路径开关: 0=旧逻辑(可回退), 1=训练走 inference=True 路径
if [ "${USE_INFERENCE_TRAIN}" -eq 1 ]; then
    CMD+=(--train_use_inference_path --train_unroll_steps "${TRAIN_UNROLL_STEPS}")
fi

"${CMD[@]}"

# ===========================================================
# Step 4: 评估 (训练完成后手动指定 checkpoint)
# 用法: 把 CKPT 改为实际 .pt 路径后取消注释运行
# ===========================================================
# CKPT="model/<your_model_name>.pt"
# python eval.py \
#     --ckpt "${CKPT}" \
#     --test_data "${TEST_DATA}" \
#     --device "${DEVICE}" \
#     --rollout_DT 0.1 \
#     --dt_data 1.0 \
#     --output_dim "${OUTPUT_DIM}" \
#     --width "${WIDTH}" \
#     --modes "${MODES}" \
#     --n_layers "${N_LAYERS}" \
#     --K "${K}" \
#     --f_nu_hidden "${F_NU_HIDDEN}" \
#     --simulation_steps 10 \
#     --out_dir eval_results
