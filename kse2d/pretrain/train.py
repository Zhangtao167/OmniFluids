from tools import Init_generation, HW_Init_generation
from psm_loss import PSM_loss, rhs_anchored_loss

import math
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


EPS = 1e-7


# ===== mhd_sim 数据加载 =====

def load_mhd_sim_data(path, time_start, time_end, size):
    """加载 mhd_sim hw_dataset.pt, 转为 OmniFluids channels-last 格式.

    Args:
        path: hw_dataset.pt 路径
        time_start: 训练窗口起始帧 (含), 如 200
        time_end: 训练窗口结束帧 (含), 如 249
        size: 模型空间分辨率 (如 64)

    Returns:
        pool: [N_pool, S, S, 2]  训练快照 (channels-last)
        traj: [B, S, S, T_win, 2]  轨迹 (用于评估)
        param: [B, 3]  (alpha, kappa, log10(nu))
        hw_cfg: dict  原始 HW 配置
    """
    data = torch.load(path, map_location='cpu', weights_only=False)
    zeta = data['traj_zeta'].float()        # [B, T, H, W]
    n_field = data['traj_n'].float()        # [B, T, H, W]
    hw_cfg = data['config']
    B, T_total, H, W = zeta.shape

    # 时间窗口 [time_start, time_end] (inclusive)
    t0, t1 = time_start, time_end + 1
    zeta_win = zeta[:, t0:t1]              # [B, T_win, H, W]
    n_win = n_field[:, t0:t1]
    T_win = zeta_win.shape[1]

    # 堆叠通道: [B, T_win, 2, H, W]
    states = torch.stack([zeta_win, n_win], dim=2)

    # --- pool: 展平 batch×time → [N_pool, 2, H, W] ---
    pool_cf = states.reshape(-1, 2, H, W)
    if H != size:
        pool_cf = F.interpolate(pool_cf, size=(size, size),
                                mode='bilinear', align_corners=False)
    pool = pool_cf.permute(0, 2, 3, 1).contiguous()   # [N_pool, S, S, 2]

    # --- traj: [B, T_win, 2, H, W] → [B, S, S, T_win, 2] ---
    traj_cf = states
    if H != size:
        traj_cf = F.interpolate(
            traj_cf.reshape(B * T_win, 2, H, W),
            size=(size, size), mode='bilinear', align_corners=False
        ).reshape(B, T_win, 2, size, size)
    traj = traj_cf.permute(0, 3, 4, 1, 2).contiguous()  # [B, S, S, T_win, 2]

    # --- param ---
    alpha = hw_cfg['alpha']
    kappa = hw_cfg['kappa']
    lognu = math.log10(hw_cfg['nu'])
    param = torch.zeros(B, 3)
    param[:, 0] = alpha
    param[:, 1] = kappa
    param[:, 2] = lognu

    return pool, traj, param, hw_cfg


def _make_full_param(alpha, kappa, lognu, N, device):
    """构造 [N, 5] 完整参数向量"""
    p = torch.zeros(N, 5, device=device)
    p[:, 0] = alpha
    p[:, 1] = kappa
    p[:, 2] = lognu
    p[:, 3] = 0.75   # hyper_order / 4
    p[:, 4] = 0.15   # k0
    return p


# ===== test / val / train =====

def _build_train_sequence(config, net, w0, param):
    """构建训练时序列，支持旧路径和 inference 路径两种模式。"""
    if not getattr(config, 'train_use_inference_path', False):
        w_pred = net(w0, param)                                      # [B,S,S,Tp,2]
        w_all = torch.cat([w0.unsqueeze(-2), w_pred], dim=-2)       # [B,S,S,Tp+1,2]
        return w_all, w_pred[:, :, :, -1, :], config.rollout_DT

    n_steps = max(1, int(getattr(config, 'train_unroll_steps', 1)))
    current = w0
    preds = [current]
    for _ in range(n_steps):
        current = net(current, param, inference=True)
        preds.append(current)
    w_all = torch.stack(preds, dim=3)                                # [B,S,S,n_steps+1,2]
    return w_all, current, config.rollout_DT * n_steps


def _compute_training_loss(config, w_all, param, train_interval, loss_mode):
    """按配置计算训练损失，支持 mhd_sim / OmniFluids 两种loss."""
    loss_type = getattr(config, 'train_loss_type', 'mhd_sim')
    operator_discretization = getattr(config, 'operator_discretization', 'mhd_sim')
    mhd_sim_root = getattr(config, 'mhd_sim_root', '/zhangtao/project2026/mhd_sim')

    if loss_type == 'omnifluids':
        loss = PSM_loss(
            w_all, param, train_interval, loss_mode,
            operator_discretization=operator_discretization,
            mhd_sim_root=mhd_sim_root
        )
        rhs_w = getattr(config, 'rhs_loss_weight', 0.0)
        if rhs_w > 0:
            n_steps = w_all.shape[3] - 1
            step_dt = train_interval / n_steps
            rhs_sum = 0.0
            for t in range(n_steps):
                rhs_sum = rhs_sum + rhs_anchored_loss(
                    w_all[:, :, :, t, :], w_all[:, :, :, t + 1, :], param, step_dt,
                    operator_discretization=operator_discretization,
                    mhd_sim_root=mhd_sim_root
                )
            loss = loss + rhs_w * (rhs_sum / n_steps)
        return loss

    n_steps = w_all.shape[3] - 1
    step_dt = train_interval / n_steps
    loss_sum = 0.0
    for t in range(n_steps):
        loss_sum = loss_sum + rhs_anchored_loss(
            w_all[:, :, :, t, :], w_all[:, :, :, t + 1, :], param, step_dt,
            operator_discretization=operator_discretization,
            mhd_sim_root=mhd_sim_root
        )
    return loss_sum / n_steps

def test(config, net, test_data, test_param):
    """
    test_data: [N, S, S, T_win, 2]
    test_param: [N, 3] (α, κ, logν)
    """
    device = config.device
    rollout_DT = config.rollout_DT
    T_win = test_data.shape[3]
    total_iter = T_win - 1     # frame 0 是初始条件
    if hasattr(config, 'test_steps') and config.test_steps > 0:
        total_iter = min(total_iter, config.test_steps)

    N = test_param.shape[0]
    param_full = _make_full_param(
        test_param[:, 0], test_param[:, 1], test_param[:, 2], N, device)

    dt_data = getattr(config, 'dt_data', 1.0)
    substeps = max(1, round(dt_data / config.rollout_DT))

    w_current = test_data[:, :, :, 0, :].to(device)   # [N,S,S,2]
    predictions = [w_current]
    net.eval()
    with torch.no_grad():
        for _ in range(total_iter):
            for _ in range(substeps):
                w_current = net(w_current, param_full, inference=True).detach()
            predictions.append(w_current)
    w_pre = torch.stack(predictions, dim=3)            # [N,S,S,total_iter+1,2]

    rela_err = []
    print('_________Test__________')
    for t in range(1, total_iter + 1):
        w = w_pre[:, :, :, t, :]
        w_t = test_data[:, :, :, t, :].to(device)
        rela_err.append(
            (torch.norm((w - w_t).reshape(N, -1), dim=1)
             / torch.norm(w_t.reshape(N, -1), dim=1).clamp(min=1e-12)
             ).mean().item())
        if t % 10 == 0 or t == 1:
            print(f'  step {t}/{total_iter}  rel_L2={rela_err[-1]:.6f}')
    mean_err = np.mean(rela_err)
    print(f'Mean Relative L2 Error: {mean_err:.6f}')
    return mean_err


def val(config, net, w_0, val_param):
    """w_0: [B,S,S,2], val_param: [B,5]"""
    net.eval()
    with torch.no_grad():
        w_pre, _, train_interval = _build_train_sequence(config, net, w_0, val_param)
    physics_loss = _compute_training_loss(config, w_pre, val_param, train_interval, config.loss_mode)
    return physics_loss.item()


def train(config, net):
    device = config.device
    size = config.size
    batch_size = config.batch_size

    # ---- 加载 mhd_sim 数据 ----
    t_start = config.time_start
    t_end = config.time_end
    print(f'Loading data: train={config.train_data}, test={config.test_data}')
    print(f'Time window: [{t_start}, {t_end}] (inclusive)')

    train_pool, _, _, hw_cfg = load_mhd_sim_data(
        config.train_data, t_start, t_end, size)
    _, test_traj, test_param, _ = load_mhd_sim_data(
        config.test_data, t_start, t_end, size)

    pool = train_pool.to(device)
    test_data = test_traj
    T_win = test_data.shape[3]

    alpha = hw_cfg['alpha']
    kappa = hw_cfg['kappa']
    lognu = math.log10(hw_cfg['nu'])

    print(f'Pool: {list(pool.shape)} ({pool.shape[0]} snapshots)')
    print(f'Test: {list(test_data.shape)} ({test_data.shape[0]} traj × {T_win} frames)')
    dt_data = getattr(config, 'dt_data', 1.0)
    substeps = max(1, round(dt_data / config.rollout_DT))
    train_use_infer = getattr(config, 'train_use_inference_path', False)
    train_unroll_steps = max(1, int(getattr(config, 'train_unroll_steps', 1)))
    print(f'train_loss_type={config.train_loss_type}, operator_discretization={config.operator_discretization}')
    print(f'HW: alpha={alpha}, kappa={kappa}, nu={hw_cfg["nu"]:.2e}, lognu={lognu:.2f}')
    print(f'rollout_DT={config.rollout_DT}, dt_data={dt_data}, substeps={substeps}')
    print(f'train_use_inference_path={train_use_infer}, train_unroll_steps={train_unroll_steps}')

    # ---- 验证集初值 (从 pool 采样) ----
    val_size = config.val_size
    idx_v = torch.randint(0, pool.shape[0], (val_size,))
    w0_val = pool[idx_v]
    val_param = _make_full_param(alpha, kappa, lognu, val_size, device)

    # ---- 训练 ----
    loss_mode = config.loss_mode
    assert loss_mode in ('cn', 'mid')
    rollout_DT = config.rollout_DT

    num_iterations = config.num_iterations
    net = net.to(device)
    val_error = 1e10
    optimizer = optim.Adam(net.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=num_iterations + 1, max_lr=config.lr)

    vis_dir = f'log/log_{config.file_name}/vis_{config.model_name}'
    os.makedirs(vis_dir, exist_ok=True)

    run_hash = getattr(config, 'run_hash', 'unknown')
    current_loss = 0.0
    best_rel_l2 = float('inf')
    
    pbar = tqdm(range(num_iterations + 1), desc=f'[{run_hash}] Training',
                file=sys.stderr, ncols=120)
    
    for step in pbar:
        net.train()

        # 从 pool 采样初值
        idx = torch.randint(0, pool.shape[0], (batch_size,))
        w0_train = pool[idx]                                          # [B,S,S,2]

        # 输入噪声增强
        noise_std = getattr(config, 'input_noise_std', 0.0)
        if noise_std > 0:
            w0_train = w0_train + noise_std * torch.randn_like(w0_train)

        param = _make_full_param(alpha, kappa, lognu, batch_size, device)

        w_all, _, train_interval = _build_train_sequence(config, net, w0_train, param)
        loss = _compute_training_loss(config, w_all, param, train_interval, loss_mode)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        current_loss = loss.detach().item()
        pbar.set_postfix_str(f'loss={current_loss:.4f} best_L2={best_rel_l2:.4f}')

        if step % 100 == 0:
            print(f'training loss {step} {current_loss:.6f}')
            val_err_now = val(config, net, w0_val, val_param)
            print(f'validation physics loss {step} {val_err_now:.6f}')
            if val_err_now < val_error:
                val_error = val_err_now
                print('-----------SAVING NEW MODEL-----------')
                torch.save(net.state_dict(), f'model/{config.model_name}.pt')
                rel_l2 = test(config, net, test_data, test_param)
                if rel_l2 is not None:
                    best_rel_l2 = rel_l2

                if step % 500 == 0 or step == 0:
                    _visualize(config, net, test_data, test_param, vis_dir, step, device)
            sys.stdout.flush()

    print('----------------------------FINAL_RESULT-----------------------------')
    net.load_state_dict(torch.load(f'model/{config.model_name}.pt'))
    test(config, net, test_data, test_param)
    sys.stdout.flush()


def _visualize(config, net, test_data, test_param, vis_dir, step, device):
    """生成 true vs pred 可视化"""
    try:
        rollout_DT = config.rollout_DT
        T_win = test_data.shape[3]
        total_iter = T_win - 1
        if hasattr(config, 'test_steps') and config.test_steps > 0:
            total_iter = min(total_iter, config.test_steps)
        n_vis = min(10, total_iter)

        param_full = _make_full_param(
            test_param[0:1, 0], test_param[0:1, 1], test_param[0:1, 2], 1, device)

        dt_data = getattr(config, 'dt_data', 1.0)
        substeps = max(1, round(dt_data / config.rollout_DT))

        w_current = test_data[0:1, :, :, 0, :].to(device)
        preds = [w_current.cpu()]
        net.eval()
        with torch.no_grad():
            for _ in range(n_vis):
                for _ in range(substeps):
                    w_current = net(w_current, param_full, inference=True).detach()
                preds.append(w_current.cpu())

        w_pred = torch.stack(preds, dim=3)       # [1,S,S,n_vis+1,2]
        w_true = test_data[0:1, :, :, :n_vis+1, :].cpu()

        fig, axes = plt.subplots(4, n_vis + 1, figsize=(3 * (n_vis + 1), 12))
        for ti in range(n_vis + 1):
            vmin_z = min(w_pred[0,:,:,ti,0].min(), w_true[0,:,:,ti,0].min())
            vmax_z = max(w_pred[0,:,:,ti,0].max(), w_true[0,:,:,ti,0].max())
            vmin_n = min(w_pred[0,:,:,ti,1].min(), w_true[0,:,:,ti,1].min())
            vmax_n = max(w_pred[0,:,:,ti,1].max(), w_true[0,:,:,ti,1].max())

            im = axes[0, ti].imshow(w_true[0,:,:,ti,0].numpy(), cmap='RdBu_r', vmin=vmin_z, vmax=vmax_z)
            axes[0, ti].set_title(f't={ti}s (z True)', fontsize=8); axes[0, ti].axis('off')
            plt.colorbar(im, ax=axes[0, ti], fraction=0.046)

            im = axes[1, ti].imshow(w_pred[0,:,:,ti,0].numpy(), cmap='RdBu_r', vmin=vmin_z, vmax=vmax_z)
            axes[1, ti].set_title(f't={ti}s (z Pred)', fontsize=8); axes[1, ti].axis('off')
            plt.colorbar(im, ax=axes[1, ti], fraction=0.046)

            im = axes[2, ti].imshow(w_true[0,:,:,ti,1].numpy(), cmap='viridis', vmin=vmin_n, vmax=vmax_n)
            axes[2, ti].set_title(f't={ti}s (n True)', fontsize=8); axes[2, ti].axis('off')
            plt.colorbar(im, ax=axes[2, ti], fraction=0.046)

            im = axes[3, ti].imshow(w_pred[0,:,:,ti,1].numpy(), cmap='viridis', vmin=vmin_n, vmax=vmax_n)
            axes[3, ti].set_title(f't={ti}s (n Pred)', fontsize=8); axes[3, ti].axis('off')
            plt.colorbar(im, ax=axes[3, ti], fraction=0.046)

        plt.suptitle(f'Step {step}: a={test_param[0,0]:.2f} k={test_param[0,1]:.2f} lv={test_param[0,2]:.2f}',
                     fontsize=12, y=0.995)
        plt.tight_layout()
        save_path = f'{vis_dir}/vis_step_{step:06d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f'    Visualization saved: {save_path}')
    except Exception as e:
        print(f'    Visualization failed: {e}')
