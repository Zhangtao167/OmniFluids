#!/bin/bash
# =============================================================================
# 自动设置并启动 MHD5 Staged 训练实验
# 配置: 50000 online -> offline, mae_weight=0.1, 无radial mask
# 用法: bash setup_and_launch.sh [GPU_ID] [EXP_NAME_SUFFIX]
# =============================================================================

set -e

# 参数设置
GPU_ID=${1:-0}
EXP_SUFFIX=${2:-""}
EXP_NAME="mhd5_staged_mae01_no_radial${EXP_SUFFIX:+_$EXP_SUFFIX}"

echo "========================================================================="
echo "  MHD5 Staged Training Setup & Launch"
echo "========================================================================="
echo "  GPU: cuda:$GPU_ID"
echo "  Exp Name: $EXP_NAME"
echo "  Mode: staged (50000 online warmup -> offline)"
echo "  MAE weight: 0.1"
echo "  Radial mask: DISABLED"
echo "========================================================================="
echo ""

# 激活环境
source /opt/conda/bin/activate 2>/dev/null || true
conda activate /zhangtao/envs/rae 2>/dev/null || {
    echo "Error: Cannot activate conda environment /zhangtao/envs/rae"
    echo "Please modify the script to use your correct environment path"
    exit 1
}

cd /zhangtao/project2026/OmniFluids/nse2d/pretrain

echo "[1/3] Checking code modifications..."

# 检查并修改 train.py - convergence图增加训练loss
if ! grep -q "train_loss" train.py 2>/dev/null; then
    echo "  -> Modifying train.py to add training loss curve..."
    
    # 备份原文件
    cp train.py train.py.bak.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    # 修改 _plot_convergence 函数
    python3 << 'PYEOF'
import re

with open('train.py', 'r') as f:
    content = f.read()

# 替换 _plot_convergence 函数
old_func = '''def _plot_convergence(eval_history, save_path):
    """Plot mean relative L2 error vs training step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h['step'] for h in eval_history]
    mean_l2 = [h['mean_rel_l2'] for h in eval_history]
    best_idx = int(np.argmin(mean_l2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_l2, 'b-o', markersize=3, label='mean rel L2')
    ax.plot(steps[best_idx], mean_l2[best_idx], 'r*', markersize=12,
            label=f'best={mean_l2[best_idx]:.6f} @ step {steps[best_idx]}')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Mean relative L2 error')
    ax.set_title('Evaluation Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(mean_l2) > 1 and max(mean_l2) / max(min(mean_l2), 1e-10) > 10:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)'''

new_func = '''def _plot_convergence(eval_history, save_path):
    """Plot mean relative L2 error and training loss vs training step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h['step'] for h in eval_history]
    mean_l2 = [h['mean_rel_l2'] for h in eval_history]
    best_idx = int(np.argmin(mean_l2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Evaluation relative L2 error
    ax1 = axes[0]
    ax1.plot(steps, mean_l2, 'b-o', markersize=3, label='mean rel L2')
    ax1.plot(steps[best_idx], mean_l2[best_idx], 'r*', markersize=12,
            label=f'best={mean_l2[best_idx]:.6f} @ step {steps[best_idx]}')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Mean relative L2 error')
    ax1.set_title('Evaluation Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if len(mean_l2) > 1 and max(mean_l2) / max(min(mean_l2), 1e-10) > 10:
        ax1.set_yscale('log')
    
    # Plot 2: Training loss
    ax2 = axes[1]
    if 'train_loss' in eval_history[0]:
        train_losses = [h.get('train_loss', float('nan')) for h in eval_history]
        ax2.plot(steps, train_losses, 'g-s', markersize=3, label='training loss')
        ax2.set_xlabel('Training step')
        ax2.set_ylabel('Training loss')
        ax2.set_title('Training Loss Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if len(train_losses) > 1 and max(train_losses) / max(min(train_losses), 1e-10) > 10:
            ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No training loss data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Loss (not available)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("  -> _plot_convergence function updated")
else:
    print("  -> _plot_convergence already modified or different format")

# 修改 _log_and_eval 函数中记录eval_history的部分
old_eval = "eval_history.append({'step': global_step, 'mean_rel_l2': current_loss})"
new_eval = """avg_train_loss = running_loss / max(running_count, 1) if running_count > 0 else float('nan')
        eval_history.append({'step': global_step, 'mean_rel_l2': current_loss, 'train_loss': avg_train_loss})"""

if old_eval in content and 'avg_train_loss' not in content:
    content = content.replace(old_eval, new_eval)
    print("  -> eval_history recording updated with train_loss")
else:
    print("  -> eval_history already records train_loss")

with open('train.py', 'w') as f:
    f.write(content)

print("  -> train.py modifications complete")
PYEOF
else
    echo "  -> train.py already has training loss curve support"
fi

# 检查并修改 tools.py - GRF超参数控制
if ! grep -q "use_radial_mask" tools.py 2>/dev/null; then
    echo "  -> Modifying tools.py to add GRF control parameters..."
    
    cp tools.py tools.py.bak.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    python3 << 'PYEOF'
import re

with open('tools.py', 'r') as f:
    content = f.read()

# 1. 修改 __init__ 参数
old_init = '''def __init__(self, Nx=512, Ny=256, alpha=None, tau=None,
                 field_scales=None, device='cpu',
                 x_active_range=(180, 330), x_edge_width=20.0):'''

new_init = '''def __init__(self, Nx=512, Ny=256, alpha=None, tau=None,
                 field_scales=None, device='cpu',
                 x_active_range=(180, 330), x_edge_width=20.0,
                 use_radial_mask=True, use_abs_constraint=True):
        self.use_abs_constraint = use_abs_constraint'''

if old_init in content:
    content = content.replace(old_init, new_init)
    print("  -> MHD5FieldGRF.__init__ updated")

# 2. 修改 radial window 逻辑
old_radial = '''        # Radial window: Dirichlet BC * optional active-band mask
        idx = torch.arange(Nx, dtype=torch.float32, device=device)
        dirichlet = torch.sin(math.pi * idx / (Nx - 1))  # (Nx,)

        if x_active_range is not None:
            lo, hi = x_active_range
            ew = max(x_edge_width, 1.0)
            # smooth bump: ~1 inside [lo, hi], ~0 outside
            radial_mask = 0.5 * (torch.tanh((idx - lo) / ew)
                                 - torch.tanh((idx - hi) / ew))
            self.radial_window = dirichlet * radial_mask
            print(f'GRF radial mask: active x=[{lo}, {hi}], edge_width={ew}')
        else:
            self.radial_window = dirichlet'''

new_radial = '''        # Radial window: Dirichlet BC * optional active-band mask
        idx = torch.arange(Nx, dtype=torch.float32, device=device)
        dirichlet = torch.sin(math.pi * idx / (Nx - 1))  # (Nx,)

        if use_radial_mask and x_active_range is not None:
            lo, hi = x_active_range
            ew = max(x_edge_width, 1.0)
            # smooth bump: ~1 inside [lo, hi], ~0 outside
            radial_mask = 0.5 * (torch.tanh((idx - lo) / ew)
                                 - torch.tanh((idx - hi) / ew))
            self.radial_window = dirichlet * radial_mask
            print(f'GRF radial mask: active x=[{lo}, {hi}], edge_width={ew}')
        else:
            self.radial_window = dirichlet
            if not use_radial_mask:
                print('GRF radial mask: disabled (full Dirichlet window)')
            elif x_active_range is None:
                print('GRF radial mask: disabled (x_active_range is None)')'''

if old_radial in content:
    content = content.replace(old_radial, new_radial)
    print("  -> Radial window logic updated")

# 3. 修改 __call__ 中的 abs 约束
old_call = '''        out = torch.stack(fields, dim=-1)  # (B, Nx, Ny, 5)

        # Positivity constraints: n (idx 0) and Ti (idx 4) must be >= 0
        out[..., 0] = torch.abs(out[..., 0])
        out[..., 4] = torch.abs(out[..., 4])

        return out'''

new_call = '''        out = torch.stack(fields, dim=-1)  # (B, Nx, Ny, 5)

        # Positivity constraints: n (idx 0) and Ti (idx 4) must be >= 0
        if self.use_abs_constraint:
            out[..., 0] = torch.abs(out[..., 0])
            out[..., 4] = torch.abs(out[..., 4])

        return out'''

if old_call in content:
    content = content.replace(old_call, new_call)
    print("  -> __call__ abs constraint updated")

# 4. 修改 from_data_stats
old_from = '''    @staticmethod
    def from_data_stats(data_path, Nx=512, Ny=256, alpha=None, tau=None,
                        device='cpu', time_start=250.0, time_end=300.0,
                        dt_data=1.0, x_active_range=(180, 330),
                        x_edge_width=20.0):'''

new_from = '''    @staticmethod
    def from_data_stats(data_path, Nx=512, Ny=256, alpha=None, tau=None,
                        device='cpu', time_start=250.0, time_end=300.0,
                        dt_data=1.0, x_active_range=(180, 330),
                        x_edge_width=20.0,
                        use_radial_mask=True, use_abs_constraint=True):'''

if old_from in content:
    content = content.replace(old_from, new_from)
    # 还需要修改return语句
    old_return = '''return MHD5FieldGRF(Nx, Ny, alpha, tau, field_scales, device,
                            x_active_range=x_active_range,
                            x_edge_width=x_edge_width)'''
    new_return = '''return MHD5FieldGRF(Nx, Ny, alpha, tau, field_scales, device,
                            x_active_range=x_active_range,
                            x_edge_width=x_edge_width,
                            use_radial_mask=use_radial_mask,
                            use_abs_constraint=use_abs_constraint)'''
    if old_return in content:
        content = content.replace(old_return, new_return)
    print("  -> from_data_stats updated")

with open('tools.py', 'w') as f:
    f.write(content)

print("  -> tools.py modifications complete")
PYEOF
else
    echo "  -> tools.py already has GRF control parameters"
fi

# 检查并修改 main.py - 添加命令行参数
if ! grep -q "grf_use_radial_mask" main.py 2>/dev/null; then
    echo "  -> Modifying main.py to add command line arguments..."
    
    cp main.py main.py.bak.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    python3 << 'PYEOF'
with open('main.py', 'r') as f:
    content = f.read()

# 添加新的命令行参数
old_args = '''    parser.add_argument('--grf_scale_from_data', type=int, default=1,
                        help='1=derive GRF field_scales from data stats, 0=use defaults')'''

new_args = '''    parser.add_argument('--grf_scale_from_data', type=int, default=1,
                        help='1=derive GRF field_scales from data stats, 0=use defaults')
    parser.add_argument('--grf_use_radial_mask', type=int, default=1,
                        help='1=use radial mask for GRF (default), 0=disable (full Dirichlet)')
    parser.add_argument('--grf_use_abs_constraint', type=int, default=1,
                        help='1=apply abs() to n and Ti for positivity (default), 0=disable')'''

if old_args in content:
    content = content.replace(old_args, new_args)
    print("  -> Command line arguments added")

with open('main.py', 'w') as f:
    f.write(content)

print("  -> main.py modifications complete")
PYEOF
else
    echo "  -> main.py already has GRF control arguments"
fi

# 检查并修改 train.py - _make_grf_generator 传递参数
if ! grep -q "use_radial_mask=use_radial_mask" train.py 2>/dev/null; then
    echo "  -> Modifying train.py _make_grf_generator to pass new parameters..."
    
    python3 << 'PYEOF'
with open('train.py', 'r') as f:
    content = f.read()

old_grf = '''def _make_grf_generator(config):
    """Build GRF random field generator."""
    # None -> use per-field defaults inside MHD5FieldGRF
    alpha = getattr(config, 'grf_alpha', None)
    tau = getattr(config, 'grf_tau', None)
    if getattr(config, 'grf_scale_from_data', 1):
        grf = MHD5FieldGRF.from_data_stats(
            config.data_path, Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=config.device,
            time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data)
    else:
        grf = MHD5FieldGRF(
            Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=config.device)
    return grf'''

new_grf = '''def _make_grf_generator(config):
    """Build GRF random field generator."""
    # None -> use per-field defaults inside MHD5FieldGRF
    alpha = getattr(config, 'grf_alpha', None)
    tau = getattr(config, 'grf_tau', None)
    use_radial_mask = bool(getattr(config, 'grf_use_radial_mask', 1))
    use_abs_constraint = bool(getattr(config, 'grf_use_abs_constraint', 1))
    if getattr(config, 'grf_scale_from_data', 1):
        grf = MHD5FieldGRF.from_data_stats(
            config.data_path, Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=config.device,
            time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data,
            use_radial_mask=use_radial_mask,
            use_abs_constraint=use_abs_constraint)
    else:
        grf = MHD5FieldGRF(
            Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=config.device,
            use_radial_mask=use_radial_mask,
            use_abs_constraint=use_abs_constraint)
    return grf'''

if old_grf in content:
    content = content.replace(old_grf, new_grf)
    print("  -> _make_grf_generator updated")

with open('train.py', 'w') as f:
    f.write(content)

print("  -> _make_grf_generator modifications complete")
PYEOF
else
    echo "  -> train.py _make_grf_generator already passes new parameters"
fi

echo ""
echo "[2/3] All code modifications verified!"
echo ""
echo "[3/3] Launching training experiment..."
echo ""

# 数据路径
DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt"
EVAL_DATA_PATH="/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt"

# 启动训练
python main.py \
    --mode train \
    --device "cuda:$GPU_ID" \
    --exp_name "$EXP_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --data_mode "staged" \
    --online_warmup_steps 50000 \
    --mae_weight 0.1 \
    --grf_use_radial_mask 0 \
    --grf_use_abs_constraint 1 \
    --time_start 250.0 \
    --time_end 300.0 \
    --Nx 512 --Ny 256 \
    --modes_x 128 --modes_y 128 \
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
    --seed 42

echo ""
echo "========================================================================="
echo "  Training Complete!"
echo "========================================================================="
echo "  Results saved to: results/$EXP_NAME/"
echo "========================================================================="
