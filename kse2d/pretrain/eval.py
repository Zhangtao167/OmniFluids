"""OmniFluids HW 评估脚本: 加载 checkpoint → 自回归 rollout → 指标 + 可视化.

指标 (参考 mhd_sim eval_hw.py):
  - Per-step relative L2 error
  - Per-step Pearson correlation
  - 动能 E_kin, 势能 E_pot, 总能量 E_total
  - 粒子通量 Gamma_n

可视化:
  - 预测场 vs 真实场 (zeta & n)
  - 绝对误差场
  - Per-step rel L2 曲线
  - 能量 & 粒子通量随时间演化

用法:
  python eval.py \
      --ckpt model/xxx.pt \
      --test_data /path/to/hw_dataset.pt \
      --device cuda:0
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import OmniFluids2D


# ===================== 数据加载 =====================

def load_mhd_sim_data(path, time_start, time_end, size):
    """加载 mhd_sim hw_dataset.pt, 返回 channels-last 轨迹和参数."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    zeta = data['traj_zeta'].float()
    n_field = data['traj_n'].float()
    hw_cfg = data['config']
    B, T_total, H, W = zeta.shape

    t0, t1 = time_start, time_end + 1
    zeta_win = zeta[:, t0:t1]
    n_win = n_field[:, t0:t1]
    T_win = zeta_win.shape[1]

    states = torch.stack([zeta_win, n_win], dim=2)  # [B, T, 2, H, W]

    if H != size:
        states = F.interpolate(
            states.reshape(B * T_win, 2, H, W),
            size=(size, size), mode='bilinear', align_corners=False
        ).reshape(B, T_win, 2, size, size)

    # channels-last: [B, S, S, T, 2]
    traj = states.permute(0, 3, 4, 1, 2).contiguous()

    alpha = hw_cfg['alpha']
    kappa = hw_cfg['kappa']
    lognu = math.log10(hw_cfg['nu'])
    return traj, alpha, kappa, lognu, hw_cfg


def _make_full_param(alpha, kappa, lognu, N, device):
    p = torch.zeros(N, 5, device=device)
    p[:, 0] = alpha
    p[:, 1] = kappa
    p[:, 2] = lognu
    p[:, 3] = 0.75   # hyper_order / 4
    p[:, 4] = 0.15   # k0
    return p


# ===================== 自回归 rollout =====================

@torch.no_grad()
def autoregressive_rollout(net, w0, param, n_data_steps, substeps=1):
    """w0: [B,S,S,2], param: [B,5] → predictions at data-frame boundaries.
    每个 data step 内做 substeps 次模型调用.
    Returns: [B,S,S,n_data_steps+1,2]"""
    net.eval()
    preds = [w0]
    current = w0
    for _ in range(n_data_steps):
        for _ in range(substeps):
            current = net(current, param, inference=True).detach()
        preds.append(current)
    return torch.stack(preds, dim=3)


# ===================== 指标计算 =====================

def compute_rel_l2(pred, true):
    """pred, true: [B,S,S,T,2], 返回 per-step rel L2 (长度 T-1)."""
    B = pred.shape[0]
    errors = []
    T = pred.shape[3]
    for t in range(1, T):
        diff = (pred[:, :, :, t, :] - true[:, :, :, t, :]).reshape(B, -1)
        ref = true[:, :, :, t, :].reshape(B, -1)
        errors.append(
            (torch.norm(diff, dim=1) / torch.norm(ref, dim=1).clamp(min=1e-12)
             ).mean().item())
    return errors


def compute_correlation(pred, true):
    """Per-step Pearson correlation, 返回长度 T-1 的 list."""
    B = pred.shape[0]
    T = pred.shape[3]
    corrs = []
    for t in range(1, T):
        p = pred[:, :, :, t, :].reshape(B, -1)
        g = true[:, :, :, t, :].reshape(B, -1)
        p_c = p - p.mean(dim=1, keepdim=True)
        g_c = g - g.mean(dim=1, keepdim=True)
        cov = (p_c * g_c).sum(dim=1)
        corrs.append(
            (cov / (p_c.norm(dim=1).clamp(min=1e-12) * g_c.norm(dim=1).clamp(min=1e-12))
             ).mean().item())
    return corrs


@torch.no_grad()
def compute_physics_diagnostics(traj_cl, k0, device):
    """从 channels-last [B,S,S,T,2] 计算能量和粒子通量.

    Returns: dict with keys gamma_n, E_kin, E_pot, E_total, 各为 [T] numpy array.
    """
    B, S, _, T, _ = traj_cl.shape
    # [B, S, S, T, 2] -> [B, T, 2, S, S]
    states = traj_cl.permute(0, 3, 4, 1, 2).contiguous()

    zeta = states[:, :, 0]  # [B, T, S, S]
    n = states[:, :, 1]

    Lx = 2 * math.pi / k0
    dx = Lx / S
    kx_1d = torch.fft.fftfreq(S, d=dx / (2 * math.pi), device=device, dtype=torch.float32)
    ky_1d = torch.fft.fftfreq(S, d=dx / (2 * math.pi), device=device, dtype=torch.float32)
    kx, ky = torch.meshgrid(kx_1d, ky_1d, indexing='ij')
    k2 = kx ** 2 + ky ** 2
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0

    zeta = zeta.to(device)
    n = n.to(device)

    # Poisson: nabla^2 phi = zeta => phi_k = -zeta_k / k^2
    zeta_k = torch.fft.fft2(zeta)
    phi_k = -zeta_k / k2_safe
    phi_k[..., 0, 0] = 0
    phi = torch.fft.ifft2(phi_k).real

    dphi_dx = torch.fft.ifft2(1j * kx * torch.fft.fft2(phi)).real
    dphi_dy = torch.fft.ifft2(1j * ky * torch.fft.fft2(phi)).real

    # 粒子通量 Gamma_n = <n * (-dphi/dy)>
    gamma_n = (n * (-dphi_dy)).mean(dim=(-2, -1))  # [B, T]
    E_kin = 0.5 * (dphi_dx ** 2 + dphi_dy ** 2).mean(dim=(-2, -1))
    E_pot = 0.5 * (n ** 2).mean(dim=(-2, -1))
    E_total = E_kin + E_pot

    return {
        'gamma_n': gamma_n.mean(dim=0).cpu().numpy(),
        'E_kin': E_kin.mean(dim=0).cpu().numpy(),
        'E_pot': E_pot.mean(dim=0).cpu().numpy(),
        'E_total': E_total.mean(dim=0).cpu().numpy(),
    }


# ===================== 可视化 =====================

def plot_fields(pred_cl, true_cl, sample_idx, steps_to_show, t_offset, save_path):
    """预测场 vs 真实场 (zeta 和 n), 以及绝对误差场."""
    n_cols = len(steps_to_show)
    fig, axes = plt.subplots(6, n_cols, figsize=(3.5 * n_cols, 18))

    for ci, t in enumerate(steps_to_show):
        z_true = true_cl[sample_idx, :, :, t, 0].cpu().numpy()
        z_pred = pred_cl[sample_idx, :, :, t, 0].cpu().numpy()
        n_true = true_cl[sample_idx, :, :, t, 1].cpu().numpy()
        n_pred = pred_cl[sample_idx, :, :, t, 1].cpu().numpy()
        phys_t = t_offset + t

        vmin_z = min(z_true.min(), z_pred.min())
        vmax_z = max(z_true.max(), z_pred.max())
        vmin_n = min(n_true.min(), n_pred.min())
        vmax_n = max(n_true.max(), n_pred.max())

        # Row 0: zeta true
        im = axes[0, ci].imshow(z_true, cmap='RdBu_r', vmin=vmin_z, vmax=vmax_z)
        axes[0, ci].set_title(f't={phys_t}s ζ True', fontsize=8); axes[0, ci].axis('off')
        plt.colorbar(im, ax=axes[0, ci], fraction=0.046)
        # Row 1: zeta pred
        im = axes[1, ci].imshow(z_pred, cmap='RdBu_r', vmin=vmin_z, vmax=vmax_z)
        axes[1, ci].set_title(f't={phys_t}s ζ Pred', fontsize=8); axes[1, ci].axis('off')
        plt.colorbar(im, ax=axes[1, ci], fraction=0.046)
        # Row 2: zeta |err|
        im = axes[2, ci].imshow(np.abs(z_pred - z_true), cmap='hot')
        axes[2, ci].set_title(f't={phys_t}s |err| ζ', fontsize=8); axes[2, ci].axis('off')
        plt.colorbar(im, ax=axes[2, ci], fraction=0.046)
        # Row 3: n true
        im = axes[3, ci].imshow(n_true, cmap='viridis', vmin=vmin_n, vmax=vmax_n)
        axes[3, ci].set_title(f't={phys_t}s n True', fontsize=8); axes[3, ci].axis('off')
        plt.colorbar(im, ax=axes[3, ci], fraction=0.046)
        # Row 4: n pred
        im = axes[4, ci].imshow(n_pred, cmap='viridis', vmin=vmin_n, vmax=vmax_n)
        axes[4, ci].set_title(f't={phys_t}s n Pred', fontsize=8); axes[4, ci].axis('off')
        plt.colorbar(im, ax=axes[4, ci], fraction=0.046)
        # Row 5: n |err|
        im = axes[5, ci].imshow(np.abs(n_pred - n_true), cmap='hot')
        axes[5, ci].set_title(f't={phys_t}s |err| n', fontsize=8); axes[5, ci].axis('off')
        plt.colorbar(im, ax=axes[5, ci], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fields saved: {save_path}')


def plot_metrics(rel_l2, corrs, pred_diag, true_diag, t_offset, rollout_DT, save_path):
    """4 子图: rel L2, correlation, energy, particle flux."""
    n_steps = len(rel_l2)
    t_axis = [t_offset + (i + 1) * rollout_DT for i in range(n_steps)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Per-step Relative L2
    axes[0, 0].plot(t_axis, rel_l2, 'b-o', markersize=3)
    axes[0, 0].set_xlabel('Physical time (s)')
    axes[0, 0].set_ylabel('Relative L2 error')
    axes[0, 0].set_title(f'Per-step Relative L2 Error (mean={np.mean(rel_l2):.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Per-step Correlation
    axes[0, 1].plot(t_axis, corrs, 'g-o', markersize=3)
    axes[0, 1].set_xlabel('Physical time (s)')
    axes[0, 1].set_ylabel('Pearson Correlation')
    axes[0, 1].set_title(f'Per-step Correlation (mean={np.mean(corrs):.4f})')
    axes[0, 1].set_ylim([-0.1, 1.05])
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) Energy
    T_diag = len(pred_diag['E_total'])
    t_diag = [t_offset + i * rollout_DT for i in range(T_diag)]
    axes[1, 0].plot(t_diag, true_diag['E_kin'], 'b-', label='E_kin true')
    axes[1, 0].plot(t_diag, pred_diag['E_kin'], 'b--', label='E_kin pred')
    axes[1, 0].plot(t_diag, true_diag['E_pot'], 'r-', label='E_pot true')
    axes[1, 0].plot(t_diag, pred_diag['E_pot'], 'r--', label='E_pot pred')
    axes[1, 0].plot(t_diag, true_diag['E_total'], 'k-', label='E_total true')
    axes[1, 0].plot(t_diag, pred_diag['E_total'], 'k--', label='E_total pred')
    axes[1, 0].set_xlabel('Physical time (s)')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Evolution')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Particle flux
    axes[1, 1].plot(t_diag, true_diag['gamma_n'], 'b-', label='Γ_n true')
    axes[1, 1].plot(t_diag, pred_diag['gamma_n'], 'r--', label='Γ_n pred')
    axes[1, 1].set_xlabel('Physical time (s)')
    axes[1, 1].set_ylabel('Particle flux Γ_n')
    axes[1, 1].set_title('Particle Flux Evolution')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Metrics saved: {save_path}')


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description='OmniFluids HW Evaluation')
    parser.add_argument('--ckpt', type=str, required=True, help='model checkpoint .pt')
    parser.add_argument('--test_data', type=str, required=True, help='mhd_sim hw_dataset.pt')
    parser.add_argument('--time_start', type=int, default=200)
    parser.add_argument('--time_end', type=int, default=249)
    parser.add_argument('--simulation_steps', type=int, default=10,
                        help='autoregressive rollout steps')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out_dir', type=str, default='eval_results')
    # 模型结构参数 (需与训练一致)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=40)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--f_nu_hidden', type=int, default=128,
                        help='hidden dim for MoE weight generator')
    parser.add_argument('--rollout_DT', type=float, default=0.1)
    parser.add_argument('--dt_data', type=float, default=1.0,
                        help='physical time between data frames (s)')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载数据 ----
    print(f'Loading test data: {args.test_data}')
    traj, alpha, kappa, lognu, hw_cfg = load_mhd_sim_data(
        args.test_data, args.time_start, args.time_end, args.size)
    B, S, _, T_win, _ = traj.shape
    n_steps = min(args.simulation_steps, T_win - 1)
    substeps = max(1, round(args.dt_data / args.rollout_DT))
    print(f'  {B} samples, {S}x{S}, T_win={T_win}, rollout={n_steps} data steps, substeps={substeps}')

    # ---- 加载模型 ----
    print(f'Loading model: {args.ckpt}')
    net = OmniFluids2D(s=args.size, K=args.K, modes=args.modes, width=args.width,
                       output_dim=args.output_dim, n_layers=args.n_layers,
                       n_fields=2, n_params=5, f_nu_hidden=args.f_nu_hidden).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    net.eval()
    print(f'  Model loaded, params={sum(p.numel() for p in net.parameters()):,}')

    # ---- Rollout ----
    param = _make_full_param(alpha, kappa, lognu, B, device)
    w0 = traj[:, :, :, 0, :].to(device)
    print(f'Running autoregressive rollout ({n_steps} data steps × {substeps} substeps) ...')
    pred = autoregressive_rollout(net, w0, param, n_steps, substeps)  # [B,S,S,n_steps+1,2]
    true = traj[:, :, :, :n_steps + 1, :].to(device)

    # ---- 指标 ----
    rel_l2 = compute_rel_l2(pred, true)
    corrs = compute_correlation(pred, true)
    print(f'  Mean rel L2:      {np.mean(rel_l2):.6f}')
    print(f'  Mean correlation: {np.mean(corrs):.6f}')
    for i, (e, c) in enumerate(zip(rel_l2, corrs)):
        print(f'    step {i+1:3d}  rel_L2={e:.6f}  corr={c:.6f}')

    # ---- 物理诊断 ----
    k0 = hw_cfg.get('k0', 0.15)
    pred_diag = compute_physics_diagnostics(pred, k0, device)
    true_diag = compute_physics_diagnostics(true, k0, device)

    # ---- 可视化: 场 ----
    t_offset = args.time_start
    n_show = min(5, n_steps + 1)
    steps_to_show = np.linspace(0, n_steps, n_show, dtype=int).tolist()
    plot_fields(pred, true, sample_idx=0, steps_to_show=steps_to_show,
                t_offset=t_offset,
                save_path=os.path.join(args.out_dir, 'fields.png'))

    # ---- 可视化: 指标曲线 ----
    plot_metrics(rel_l2, corrs, pred_diag, true_diag,
                 t_offset=t_offset, rollout_DT=args.dt_data,
                 save_path=os.path.join(args.out_dir, 'metrics.png'))

    # ---- 保存数值结果 ----
    import json
    results = {
        'rel_l2_per_step': rel_l2,
        'mean_rel_l2': float(np.mean(rel_l2)),
        'correlation_per_step': corrs,
        'mean_correlation': float(np.mean(corrs)),
        'pred_gamma_n': pred_diag['gamma_n'].tolist(),
        'true_gamma_n': true_diag['gamma_n'].tolist(),
        'pred_E_total': pred_diag['E_total'].tolist(),
        'true_E_total': true_diag['E_total'].tolist(),
        'n_samples': B,
        'n_steps': n_steps,
        'time_start': args.time_start,
        'time_end': args.time_end,
        'rollout_DT': args.rollout_DT,
    }
    json_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved: {json_path}')
    print('Done.')


if __name__ == '__main__':
    main()
