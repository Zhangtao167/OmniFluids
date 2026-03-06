cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

# 默认设置（exp10 模型，50 步演化）
python visualize_model_rollout.py

# 自定义参数
python visualize_model_rollout.py \
    --checkpoint /path/to/model.pt \
    --n_steps 100 \
    --save_dir ./my_vis \
    --seed 123 \
    --batch_size 1