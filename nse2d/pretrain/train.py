"""Training and evaluation for 5-field MHD OmniFluids.

Training: offline dataset + physics loss (mhd_sim RHS).
Evaluation: autoregressive rollout with relative L2 error + visualization.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tools import (load_mhd5_snapshots, load_mhd5_trajectories, MHD5FieldGRF, 
                   FixedGRFDataSampler, ModelEvolvedGRFGenerator,
                   compute_metrics_and_visualize)
from psm_loss import build_mhd_instance, make_mhd5_rhs_fn, compute_physics_loss, dealias_state

# Multi-GPU support via Accelerate
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    Accelerator = None

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']


# ---------------------------------------------------------------------------
# Visualization helpers (aligned with mhd_sim/eval_5field.py style)
# ---------------------------------------------------------------------------

def save_eval_plots(pred_traj, gt_traj, rel_l2_total, rel_l2_per_field,
                    save_dir, tag='eval',
                    time_start=250.0, rollout_dt=0.1, n_substeps=10):
    """Save evaluation plots: error curves, GT/Pred/Error snapshots, energy, spectrum.

    Args:
        pred_traj: (B, T, C, Nx, Ny) predicted trajectory
        gt_traj:   (B, T, C, Nx, Ny) ground truth trajectory
        time_start: physical start time (seconds)
        rollout_dt: model forward dt (seconds per NFE step)
        n_substeps: model forward passes per data step (NFE per data step)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    n_steps = len(rel_l2_total)
    B, T_total, C, Nx, Ny = gt_traj.shape

    # ===== 1. Error curves (dual x-axis: NFE + physical time) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nfe = np.arange(1, n_steps + 1) * n_substeps

    axes[0].plot(nfe, rel_l2_total, 'k-o', markersize=3, label='total')
    axes[0].set_xlabel('Model forward steps (NFE)')
    axes[0].set_ylabel('Relative L2 error')
    axes[0].set_title(f'Total rel L2 ({tag})')
    axes[0].grid(True, alpha=0.3)
    ax0_top = axes[0].twiny()
    nfe_lo, nfe_hi = axes[0].get_xlim()
    ax0_top.set_xlim(time_start + nfe_lo * rollout_dt,
                     time_start + nfe_hi * rollout_dt)
    ax0_top.set_xlabel('Physical time (s)')

    for name in FIELD_NAMES:
        axes[1].plot(nfe, rel_l2_per_field[name], '-o', markersize=2, label=name)
    axes[1].set_xlabel('Model forward steps (NFE)')
    axes[1].set_ylabel('Relative L2 error')
    axes[1].set_title(f'Per-field rel L2 ({tag})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    ax1_top = axes[1].twiny()
    nfe_lo, nfe_hi = axes[1].get_xlim()
    ax1_top.set_xlim(time_start + nfe_lo * rollout_dt,
                     time_start + nfe_hi * rollout_dt)
    ax1_top.set_xlabel('Physical time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'error_curves_{tag}.png'), dpi=150)
    plt.close(fig)

    # ===== 2. GT / Pred / Error / GT Residual / Pred Residual snapshots (5 rows per field) =====
    plot_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    plot_steps = sorted(set(s for s in plot_steps if s < T_total))
    b_idx = 0
    n_cols = len(plot_steps)

    for c, name in enumerate(FIELD_NAMES):
        fig, axes = plt.subplots(5, n_cols, figsize=(4 * n_cols, 16))
        if n_cols == 1:
            axes = axes.reshape(5, 1)
        for j, t in enumerate(plot_steps):
            gt_snap = gt_traj[b_idx, t, c].cpu().numpy()
            pred_snap = pred_traj[b_idx, t, c].cpu().numpy()
            err_snap = pred_snap - gt_snap
            vmin, vmax = gt_snap.min(), gt_snap.max()
            err_abs = max(abs(err_snap.min()), abs(err_snap.max()), 1e-10)
            t_phys = time_start + t * n_substeps * rollout_dt
            ic_tag = ' (IC)' if t == 0 else ''

            # GT residual: gt[t] - gt[t-1], first frame = 0
            if t > 0:
                gt_prev = gt_traj[b_idx, t - 1, c].cpu().numpy()
                gt_residual = gt_snap - gt_prev
            else:
                gt_residual = np.zeros_like(gt_snap)
            
            # Pred residual: pred[t] - pred[t-1], first frame = 0
            if t > 0:
                pred_prev = pred_traj[b_idx, t - 1, c].cpu().numpy()
                pred_residual = pred_snap - pred_prev
            else:
                pred_residual = np.zeros_like(pred_snap)
            
            # Shared residual color scale
            res_abs = max(abs(gt_residual).max(), abs(pred_residual).max(), 1e-10)

            # Row 0: GT
            im0 = axes[0, j].imshow(gt_snap, aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[0, j].set_title(f'GT t={t_phys:.1f}s{ic_tag}', fontsize=9)
            fig.colorbar(im0, ax=axes[0, j], fraction=0.046)

            # Row 1: Pred
            im1 = axes[1, j].imshow(pred_snap, aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1, j].set_title(f'Pred t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im1, ax=axes[1, j], fraction=0.046)

            # Row 2: Error (Pred - GT)
            im2 = axes[2, j].imshow(err_snap, aspect='auto',
                                    vmin=-err_abs, vmax=err_abs, cmap='bwr')
            axes[2, j].set_title(f'Err t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im2, ax=axes[2, j], fraction=0.046)

            # Row 3: GT Residual (gt[t] - gt[t-1])
            im3 = axes[3, j].imshow(gt_residual, aspect='auto',
                                    vmin=-res_abs, vmax=res_abs, cmap='PuOr')
            axes[3, j].set_title(f'GT Δ t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im3, ax=axes[3, j], fraction=0.046)

            # Row 4: Pred Residual (pred[t] - pred[t-1])
            im4 = axes[4, j].imshow(pred_residual, aspect='auto',
                                    vmin=-res_abs, vmax=res_abs, cmap='PuOr')
            axes[4, j].set_title(f'Pred Δ t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im4, ax=axes[4, j], fraction=0.046)

            if j == 0:
                axes[0, j].set_ylabel('GT\nx (radial)')
                axes[1, j].set_ylabel('Pred\nx (radial)')
                axes[2, j].set_ylabel('Error\nx (radial)')
                axes[3, j].set_ylabel('GT Δ\nx (radial)')
                axes[4, j].set_ylabel('Pred Δ\nx (radial)')
            axes[4, j].set_xlabel('y (binormal)')

        fig.suptitle(f'{name} ({tag})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'snapshots_{name}_{tag}.png'), dpi=150)
        plt.close(fig)

    # ===== 3. Per-field energy over time =====
    gt_np = gt_traj[b_idx, :n_steps + 1].cpu().numpy()   # (T, C, Nx, Ny)
    pred_np = pred_traj[b_idx, :n_steps + 1].cpu().numpy()
    t_axis = time_start + np.arange(n_steps + 1) * n_substeps * rollout_dt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for c, name in enumerate(FIELD_NAMES):
        gt_energy = (gt_np[:, c] ** 2).mean(axis=(-2, -1))
        pred_energy = (pred_np[:, c] ** 2).mean(axis=(-2, -1))
        axes[c].plot(t_axis, gt_energy, 'b-', label='GT')
        axes[c].plot(t_axis, pred_energy, 'r--', label='Pred')
        axes[c].set_title(f'{name}  <f²>')
        axes[c].set_xlabel('Physical time (s)')
        axes[c].set_ylabel('Mean energy')
        axes[c].legend(fontsize=8)
        axes[c].grid(True, alpha=0.3)
    axes[5].axis('off')
    fig.suptitle(f'Per-field mean energy ({tag})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'energy_{tag}.png'), dpi=150)
    plt.close(fig)

    # ===== 4. Spatial power spectrum at selected time steps =====
    spec_steps = [0, n_steps // 2, n_steps]
    spec_steps = sorted(set(s for s in spec_steps if s < T_total))

    fig, axes = plt.subplots(len(FIELD_NAMES), len(spec_steps),
                              figsize=(5 * len(spec_steps), 3.5 * len(FIELD_NAMES)))
    if len(spec_steps) == 1:
        axes = axes.reshape(-1, 1)

    for c, name in enumerate(FIELD_NAMES):
        for j, t in enumerate(spec_steps):
            gt_f = gt_traj[b_idx, t, c].cpu().numpy()
            pred_f = pred_traj[b_idx, t, c].cpu().numpy()
            gt_spec_2d = np.abs(np.fft.rfft2(gt_f)) ** 2
            pred_spec_2d = np.abs(np.fft.rfft2(pred_f)) ** 2
            gt_spec_1d = gt_spec_2d.mean(axis=0)
            pred_spec_1d = pred_spec_2d.mean(axis=0)
            k = np.arange(len(gt_spec_1d))

            t_phys = time_start + t * n_substeps * rollout_dt
            axes[c, j].semilogy(k[1:], gt_spec_1d[1:], 'b-', label='GT', alpha=0.8)
            axes[c, j].semilogy(k[1:], pred_spec_1d[1:], 'r--', label='Pred', alpha=0.8)
            axes[c, j].set_title(f'{name}  t={t_phys:.1f}s', fontsize=9)
            axes[c, j].set_xlabel('k_y')
            axes[c, j].grid(True, alpha=0.3)
            if j == 0:
                axes[c, j].set_ylabel('Power')
            axes[c, j].legend(fontsize=7)

    fig.suptitle(f'Spatial power spectrum (y-direction) ({tag})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spectrum_{tag}.png'), dpi=150)
    plt.close(fig)

    print(f'  Plots saved to {save_dir}/')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, net, eval_data, mhd, n_rollout_steps=10,
             save_plots=False, step_tag='eval'):
    """Autoregressive rollout evaluation with optional visualization.

    Uses n_substeps = round(dt_data / rollout_dt) sub-steps per data step
    to match mhd_sim NFE.

    Args:
        eval_data: pre-loaded (B, T, 5, Nx, Ny) float32 tensor on CPU,
                   or a file path string (legacy, will load on the fly).
    """
    device = config.device
    net.eval()

    n_substeps = max(1, round(config.dt_data / config.rollout_dt))

    if isinstance(eval_data, str):
        trajectories, _ = load_mhd5_trajectories(
            eval_data, time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data)
    else:
        trajectories = eval_data

    B, T, C, Nx, Ny = trajectories.shape
    n_steps = min(n_rollout_steps, T - 1)

    traj_slice = trajectories[:, :n_steps + 1]

    x_0 = traj_slice[:, 0].permute(0, 2, 3, 1).to(device)  # (B, Nx, Ny, 5)
    gt = traj_slice.to(device)

    pred_list = [trajectories[:, 0:1].to(device)]
    current = x_0
    total_nfe = n_steps * n_substeps
    print(f'  Rollout: {n_steps} data steps x {n_substeps} substeps = {total_nfe} NFE, B={B}')
    with torch.no_grad():
        for _data_step in range(n_steps):
            for _sub in range(n_substeps):
                out = net(current, inference=True)
                current = out[..., -1]
            pred_cf = current.permute(0, 3, 1, 2).unsqueeze(1)
            pred_list.append(pred_cf)
            if (_data_step + 1) % max(1, n_steps // 10) == 0 or _data_step == 0:
                print(f'    step {_data_step + 1}/{n_steps}', flush=True)

    pred_traj = torch.cat(pred_list, dim=1)  # (B, n_steps+1, 5, Nx, Ny)

    rel_l2_total = []
    rel_l2_per_field = {name: [] for name in FIELD_NAMES}

    for t in range(1, n_steps + 1):
        pred_t = pred_traj[:, t]
        gt_t = gt[:, t]
        diff_norm = torch.norm((pred_t - gt_t).reshape(B, -1), dim=1)
        true_norm = torch.norm(gt_t.reshape(B, -1), dim=1)
        rel_l2_total.append((diff_norm / true_norm.clamp(min=1e-8)).mean().item())
        for c, name in enumerate(FIELD_NAMES):
            diff_c = torch.norm((pred_t[:, c] - gt_t[:, c]).reshape(B, -1), dim=1)
            true_c = torch.norm(gt_t[:, c].reshape(B, -1), dim=1)
            rel_l2_per_field[name].append(
                (diff_c / true_c.clamp(min=1e-8)).mean().item())

    results = {
        'rel_l2_total': rel_l2_total,
        'rel_l2_per_field': rel_l2_per_field,
        'mean_rel_l2': float(np.mean(rel_l2_total)),  # Convert to Python float for JSON
        'n_steps': n_steps,
        'n_substeps': n_substeps,
    }

    print(f'  Eval [{step_tag}]: mean rel L2 = {results["mean_rel_l2"]:.6f} '
          f'({n_steps} steps x {n_substeps} substeps = {n_steps * n_substeps} NFE)')
    for t in range(min(5, n_steps)):
        fields_str = ' | '.join(
            f'{name}={rel_l2_per_field[name][t]:.5f}' for name in FIELD_NAMES)
        print(f'    step {t+1}: total={rel_l2_total[t]:.6f} | {fields_str}')

    if save_plots:
        run_dir = getattr(config, 'run_dir',
                          os.path.join('results', config.exp_name,
                                       getattr(config, 'run_tag', config.exp_name)))
        # Create step-specific subdirectory
        vis_dir = os.path.join(run_dir, 'vis', step_tag)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Convert to (B, T, Nx, Ny, C) format for compute_metrics_and_visualize
        gt_for_vis = gt.permute(0, 1, 3, 4, 2).cpu()  # (B, T, Nx, Ny, 5)
        pred_for_vis = pred_traj.permute(0, 1, 3, 4, 2).cpu()  # (B, T, Nx, Ny, 5)
        
        # Use compute_metrics_and_visualize for consistent visualization
        compute_metrics_and_visualize(
            gt_traj=gt_for_vis,
            pred_traj=pred_for_vis,
            metric_step_list=[1, 3, 5, 10] if n_steps >= 10 else list(range(1, n_steps + 1)),
            plot_step_list=[0, 1, 3, 5, 10] if n_steps >= 10 else list(range(n_steps + 1)),
            visualize=True,
            save_dir=vis_dir,
            time_start=config.time_start,
            dt_data=config.dt_data,
            sample_idx=0,
            eval_key=step_tag,
        )
        print(f'  Plots saved to {vis_dir}/')

    net.train()
    return results


# ---------------------------------------------------------------------------
# GRF Overfitting Evaluation (uses physics loss instead of trajectory L2)
# ---------------------------------------------------------------------------

def evaluate_grf_overfitting(config, net, grf_data, rhs_fn, step_tag='eval', n_eval_batches=4):
    """Evaluate on fixed GRF data using physics loss (no ground truth trajectory).
    
    For GRF overfitting test, we don't have a "true" trajectory to compare against.
    Instead, we measure the PDE residual (physics loss) as the evaluation metric.
    
    Args:
        grf_data: (B, Nx, Ny, 5) fixed GRF initial conditions
        rhs_fn: physics RHS function
        step_tag: tag for logging
        n_eval_batches: number of batches to average over (for more stable evaluation)
    """
    device = config.device
    net.eval()
    
    x_all = grf_data.to(device)
    B_total = x_all.shape[0]
    batch_size = max(1, B_total // n_eval_batches)
    
    phys_losses = []
    field_rms_all = {name: [] for name in FIELD_NAMES}
    
    with torch.no_grad():
        # Multi-batch evaluation for more stable metrics
        for i in range(0, B_total, batch_size):
            x_0 = x_all[i:i+batch_size]
            if x_0.shape[0] == 0:
                continue
            
            pred_traj = net(x_0)  # (B, Nx, Ny, 5, output_dim)
            
            # Compute physics loss as evaluation metric
            phys_loss = compute_physics_loss(
                pred_traj, x_0, rhs_fn,
                rollout_dt=config.rollout_dt,
                output_dim=config.output_dim,
                time_integrator=config.time_integrator,
                mae_weight=0.0)
            phys_losses.append(phys_loss.item())
            
            # Per-field RMS of predictions
            pred_last = pred_traj[..., -1]  # (B, Nx, Ny, 5)
            for c, name in enumerate(FIELD_NAMES):
                field_rms_all[name].append(pred_last[..., c].pow(2).mean().sqrt().item())
    
    # Average over batches
    avg_phys_loss = sum(phys_losses) / len(phys_losses) if phys_losses else 0.0
    field_rms = {name: sum(vals) / len(vals) if vals else 0.0 
                 for name, vals in field_rms_all.items()}
    
    results = {
        'physics_loss': avg_phys_loss,
        'mean_rel_l2': avg_phys_loss,  # For compatibility with convergence plotting
        'field_rms': field_rms,
        'n_steps': 1,
        'n_substeps': config.output_dim,
        'n_eval_batches': len(phys_losses),
        'eval_type': 'grf_physics_loss',  # Explicitly mark evaluation type
        'is_grf_eval': True,
    }
    
    print(f'  [GRF Overfit Eval] physics_loss = {avg_phys_loss:.6f} (avg over {len(phys_losses)} batches)')
    print(f'    Field RMS: ' + ' | '.join(f'{k}={v:.4f}' for k, v in field_rms.items()))

    # --- Snapshot visualization: input GRF + predicted output ---
    if getattr(config, 'run_dir', None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            vis_dir = os.path.join(config.run_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            # Use first sample for visualization
            x_in = x_all[0].cpu()   # (Nx, Ny, 5) GRF input
            with torch.no_grad():
                pred_single = net(x_all[:1].to(device))   # (1, Nx, Ny, 5, T)
            x_pred = pred_single[0, ..., -1].cpu()        # (Nx, Ny, 5) last predicted frame

            fig, axes = plt.subplots(2, len(FIELD_NAMES), figsize=(18, 6))
            row_labels = ['GRF Input', f'Pred (+{config.output_dim * config.rollout_dt:.1f}s)']
            for row, (label, snap) in enumerate(zip(row_labels, [x_in, x_pred])):
                for col, fname in enumerate(FIELD_NAMES):
                    im = axes[row, col].imshow(
                        snap[:, :, col].numpy().T,
                        origin='lower', aspect='auto', cmap='RdBu_r')
                    if row == 0:
                        axes[row, col].set_title(fname)
                    if col == 0:
                        axes[row, col].set_ylabel(label)
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            fig.suptitle(f'GRF Overfit [{step_tag}]  physics_loss={avg_phys_loss:.4f}', fontsize=12)
            plt.tight_layout()
            out_path = os.path.join(vis_dir, f'grf_snapshot_{step_tag}.png')
            fig.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f'  Snapshot saved: {out_path}')
        except Exception as e:
            print(f'  [WARNING] GRF snapshot plot failed: {e}')

    net.train()
    return results


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_offline_loader(config):
    """Load mhd_sim data and build DataLoader."""
    is_overfitting = bool(getattr(config, 'is_overfitting_test', 0))
    traj_idx = getattr(config, 'overfitting_traj_idx', 0)
    
    snapshots, meta = load_mhd5_snapshots(
        config.data_path,
        time_start=config.time_start,
        time_end=config.time_end,
        dt_data=config.dt_data,
        single_trajectory=is_overfitting,
        traj_idx=traj_idx)
    Nx, Ny = meta['Nx'], meta['Ny']
    assert Nx == config.Nx and Ny == config.Ny, \
        f'Grid mismatch: data ({Nx},{Ny}) vs config ({config.Nx},{config.Ny})'
    dataset = TensorDataset(snapshots)
    
    # For overfitting test: keep batch_size unchanged, randomly sample from available snapshots
    if is_overfitting:
        # Use same batch_size, sample with replacement if dataset is smaller than batch_size
        n_samples = len(dataset)
        # Calculate total samples needed for all iterations
        total_iters = getattr(config, 'num_iterations', 10000)
        num_samples_needed = config.batch_size * total_iters
        print(f'[OVERFITTING TEST] Dataset size: {n_samples}, batch_size={config.batch_size}, '
              f'num_iterations={total_iters} (random sampling with replacement)')
        # RandomSampler with replacement allows sampling more than dataset size
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_needed)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
    return loader, meta


def _load_supervised_pairs(config):
    """Load mhd_sim data as (x_t, x_{t+1}) pairs for supervised training.
    
    Returns:
        pairs: list of (x_current, x_next) tuples
        meta: dict with metadata
    """
    is_overfitting = bool(getattr(config, 'is_overfitting_test', 0))
    traj_idx = getattr(config, 'overfitting_traj_idx', 0)
    
    # Use mmap=True to avoid loading entire file into RAM
    print(f'  Loading supervised pairs (mmap) ...', end=' ', flush=True)
    data = torch.load(config.data_path, map_location='cpu', weights_only=False, mmap=True)
    
    # Calculate time indices FIRST, then only clone the needed slice
    t_start_idx = int(round(config.time_start / config.dt_data))
    t_end_idx = int(round(config.time_end / config.dt_data))
    
    # Clone ONLY the slice we need (not the full tensor) - this is the key to mmap efficiency
    fields = [data[name][:, t_start_idx:t_end_idx + 1].clone() for name in FIELD_NAMES]
    del data
    print('done.', flush=True)
    states = torch.stack(fields, dim=2)  # (B, T_slice, 5, Nx, Ny)
    
    B, T, C, Nx, Ny = states.shape
    
    # For overfitting test: use only one trajectory
    if is_overfitting:
        if traj_idx >= B:
            traj_idx = 0
        states = states[traj_idx:traj_idx+1]  # (1, T, 5, Nx, Ny)
        B = 1
        print(f'[OVERFITTING TEST] Loading supervised pairs from trajectory #{traj_idx}: {T} timesteps, {C} fields, {Nx}x{Ny}')
    else:
        print(f'Loading supervised pairs: {B} trajectories, {T} timesteps, {C} fields, {Nx}x{Ny}')
    
    # Create (x_t, x_{t+1}) pairs from consecutive timesteps
    pairs = []
    for b in range(B):
        for t in range(T - 1):  # T-1 pairs per trajectory
            x_current = states[b, t].permute(1, 2, 0).float()  # (Nx, Ny, 5)
            x_next = states[b, t + 1].permute(1, 2, 0).float()  # (Nx, Ny, 5)
            pairs.append((x_current, x_next))
    
    meta = dict(Nx=Nx, Ny=Ny, n_pairs=len(pairs))
    print(f'Created {len(pairs)} supervised (x_t, x_{{t+1}}) pairs')
    return pairs, meta


class _SupervisedOfflineSampler:
    """Infinite iterator over supervised pairs, auto-resets on exhaustion.

    Optionally converts each real pair (x_t, x_{t+1}) into a random linear-
    interpolation one-step pseudo-pair, so a 0.1s model can be trained from
    1.0s-spaced data without storing all intermediate states.
    """
    def __init__(self, pairs, batch_size, device='cpu', is_overfitting=False,
                 interp_steps=1):
        self.pairs = pairs
        self.device = device
        self.n_pairs = len(pairs)
        self.batch_size = batch_size  # Keep batch_size unchanged
        self.is_overfitting = is_overfitting
        self.interp_steps = max(1, int(interp_steps))
        if is_overfitting:
            print(f'[OVERFITTING TEST] Supervised sampler: {self.n_pairs} pairs, '
                  f'batch_size={batch_size} (random sampling with replacement)')
        self._shuffle()
        self._idx = 0
    
    def _shuffle(self):
        import random
        random.shuffle(self.pairs)
    
    def next_batch(self):
        """Returns (x_0_batch, x_target_batch)"""
        import random
        
        if self.is_overfitting:
            # For overfitting: randomly sample with replacement to fill batch_size
            batch_pairs = random.choices(self.pairs, k=self.batch_size)
        else:
            # Normal mode: sequential with shuffle on epoch end
            if self._idx + self.batch_size > self.n_pairs:
                self._shuffle()
                self._idx = 0
            batch_pairs = self.pairs[self._idx:self._idx + self.batch_size]
            self._idx += self.batch_size
        
        if self.interp_steps > 1:
            x_0_list = []
            x_target_list = []
            for x_start, x_end in batch_pairs:
                sub_idx = random.randrange(self.interp_steps)
                alpha_0 = float(sub_idx) / float(self.interp_steps)
                alpha_1 = float(sub_idx + 1) / float(self.interp_steps)
                x_0_list.append(torch.lerp(x_start, x_end, alpha_0))
                x_target_list.append(torch.lerp(x_start, x_end, alpha_1))
        else:
            x_0_list = [p[0] for p in batch_pairs]
            x_target_list = [p[1] for p in batch_pairs]
        
        x_0 = torch.stack(x_0_list).to(self.device)
        x_target = torch.stack(x_target_list).to(self.device)
        
        return x_0, x_target


def _make_grf_generator(config, device=None):
    """Build GRF random field generator.
    
    Args:
        config: Configuration object
        device: Override device (for multi-GPU, use accelerator.device)
    """
    # None -> use per-field defaults inside MHD5FieldGRF
    alpha = getattr(config, 'grf_alpha', None)
    tau = getattr(config, 'grf_tau', None)
    use_radial_mask = bool(getattr(config, 'grf_use_radial_mask', 1))
    target_device = device if device is not None else config.device
    if getattr(config, 'grf_scale_from_data', 1):
        grf = MHD5FieldGRF.from_data_stats(
            config.data_path, Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=target_device,
            time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data,
            use_radial_mask=use_radial_mask)
    else:
        grf = MHD5FieldGRF(
            Nx=config.Nx, Ny=config.Ny,
            alpha=alpha, tau=tau,
            device=target_device,
            use_radial_mask=use_radial_mask)
    return grf


def _make_pretrained_generator(grf, ckpt_path, config, device='cpu', verbose=True):
    """Create ModelEvolvedGRFGenerator with external pretrained model.
    
    Args:
        grf: MHD5FieldGRF instance
        ckpt_path: Path to pretrained model checkpoint
        config: Configuration object
        device: Target device
        verbose: Whether to print status messages (set False for non-main processes)
    
    Returns:
        ModelEvolvedGRFGenerator with loaded pretrained model (activated)
    """
    from model import OmniFluids2D
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', {})
    
    # Reconstruct model architecture from checkpoint config
    pretrained_net = OmniFluids2D(
        Nx=saved_cfg.get('Nx', config.Nx),
        Ny=saved_cfg.get('Ny', config.Ny),
        K=saved_cfg.get('K', config.K),
        T=saved_cfg.get('temperature', 10.0),
        modes_x=saved_cfg.get('modes_x', config.modes_x),
        modes_y=saved_cfg.get('modes_y', config.modes_y),
        width=saved_cfg.get('width', config.width),
        output_dim=saved_cfg.get('output_dim', config.output_dim),
        n_fields=5,
        n_params=saved_cfg.get('n_params', config.n_params),
        n_layers=saved_cfg.get('n_layers', config.n_layers),
        factor=saved_cfg.get('factor', config.factor),
        n_ff_layers=saved_cfg.get('n_ff_layers', config.n_ff_layers),
        layer_norm=saved_cfg.get('layer_norm', True))
    
    pretrained_net.load_state_dict(ckpt['model_state_dict'])
    pretrained_net = pretrained_net.to(device)
    
    rollout_steps = getattr(config, 'pretrained_rollout_steps', 10)
    
    if verbose:
        print(f'[External Model Mode] Loaded pretrained model from step {ckpt.get("step", "?")}')
        print(f'  Model will evolve GRF for {rollout_steps} steps')
    
    generator = ModelEvolvedGRFGenerator(
        pretrained_net, grf, rollout_steps=rollout_steps, device=device, verbose=verbose)
    generator.activate()  # External model is activated immediately
    return generator


def _make_self_training_generator(grf, training_net, config, device='cpu', verbose=True):
    """Create ModelEvolvedGRFGenerator using a COPY of the training model.
    
    IMPORTANT: Creates a separate model instance to avoid gradient interference.
    The generator's model is completely isolated from the training model.
    
    Args:
        grf: MHD5FieldGRF instance
        training_net: The training model (weights will be copied, not referenced)
        config: Configuration object
        device: Target device
        verbose: Whether to print status messages (set False for non-main processes)
    
    Returns:
        ModelEvolvedGRFGenerator (NOT activated - will be activated at start_step)
    """
    from model import OmniFluids2D
    
    # Create a separate model instance (same architecture as training model)
    generator_net = OmniFluids2D(
        Nx=config.Nx, Ny=config.Ny, K=config.K, T=config.temperature,
        modes_x=config.modes_x, modes_y=config.modes_y,
        width=config.width, output_dim=config.output_dim,
        n_fields=5, n_params=config.n_params,
        n_layers=config.n_layers, factor=config.factor,
        n_ff_layers=config.n_ff_layers, layer_norm=config.layer_norm)
    
    # Copy current weights (will be updated periodically)
    generator_net.load_state_dict(training_net.state_dict())
    generator_net = generator_net.to(device)
    
    rollout_steps = getattr(config, 'self_training_rollout_steps', 10)
    
    if verbose:
        print(f'[Self-Training Mode] Created generator model (copy of training model)')
        print(f'  Will activate at step {config.self_training_start_step}')
        print(f'  Rollout steps: {rollout_steps}')
        update_every = getattr(config, 'self_training_update_every', 0)
        if update_every > 0:
            print(f'  Will update weights every {update_every} steps')
    
    # NOT activated here - will be activated in training loop at start_step
    return ModelEvolvedGRFGenerator(
        generator_net, grf, rollout_steps=rollout_steps, device=device, verbose=verbose)


def _make_scheduler(optimizer, lr, total_steps):
    """Create a fresh OneCycleLR scheduler."""
    return optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps + 1)


class _OfflineSampler:
    """Infinite iterator over a DataLoader, auto-resets on exhaustion."""
    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def next_batch(self):
        try:
            (x_batch,) = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            (x_batch,) = next(self._iter)
        return x_batch


# ---------------------------------------------------------------------------
# Supervised loss computation
# ---------------------------------------------------------------------------

def _compute_supervised_loss(pred, target, mse_weight=1.0, mae_weight=0.0):
    """Compute supervised MSE and MAE loss between prediction and target.
    
    Args:
        pred: (B, Nx, Ny, 5) predicted state
        target: (B, Nx, Ny, 5) target state
        mse_weight: weight for MSE term
        mae_weight: weight for MAE term
    
    Returns:
        loss: combined loss value
        metrics: dict with mse, mae components
    """
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    loss = mse_weight * mse + mae_weight * mae
    return loss, {'mse': mse.item(), 'mae': mae.item()}


# ---------------------------------------------------------------------------
# Mixed integrator (Euler/CN) weight schedule
# ---------------------------------------------------------------------------

def _get_euler_weight(step, config):
    """Compute current Euler weight for mixed integrator mode.
    
    Uses exponential decay: w(t) = min + (init - min) * 0.5^((t - start) / half_life)
    
    Args:
        step: Current training step
        config: Configuration object with euler_weight_* parameters
    
    Returns:
        float: Current Euler weight, or None if mixed integrator is disabled
    """
    if not getattr(config, 'use_mixed_integrator', 0):
        return None
    
    init = getattr(config, 'euler_weight_init', 1.0)
    min_w = getattr(config, 'euler_weight_min', 0.0)
    anneal_start = getattr(config, 'euler_anneal_start', 0)
    half_life = getattr(config, 'euler_half_life', 10000)
    
    if step < anneal_start:
        return init
    
    if half_life <= 0:
        return min_w
    
    t = step - anneal_start
    decay = 0.5 ** (t / half_life)
    w = min_w + (init - min_w) * decay
    return w


# ---------------------------------------------------------------------------
# Unified training step
# ---------------------------------------------------------------------------

def _train_step(net, x_0, rhs_fn, optimizer, scheduler, config, x_target=None, 
                accelerator=None, euler_weight=None, mhd=None):
    """Execute one training step. Returns (loss_value, loss_components).
    
    Args:
        x_target: Optional target state for supervised loss. If provided,
                  computes MSE/MAE between autoregressive prediction and x_target.
                  When supervised_n_substeps > 1, the model is called multiple times
                  autoregressively to cover sup_n_substeps * rollout_dt time span.
        accelerator: Optional Accelerator instance for multi-GPU training.
        euler_weight: If not None, use mixed integrator mode with this weight.
        mhd: FiveFieldMHD instance for dealias_input (required if dealias_input=True).
    
    Returns:
        (loss_value, loss_components): loss value and dict with component losses
    """
    x_ref = x_0
    noise_scale = getattr(config, 'input_noise_scale', 0.0)
    if noise_scale > 0:
        x_0 = x_0 + noise_scale * torch.randn_like(x_0)
    
    # Dealias input to prevent aliasing in nonlinear RHS terms
    if getattr(config, 'dealias_input', True) and mhd is not None:
        x_0 = dealias_state(mhd, x_0)
    
    pred_traj = net(x_0)

    # Physics loss (PDE loss)
    phys_loss_weight = getattr(config, 'physics_loss_weight', 1.0)
    loss_components = {}
    
    if phys_loss_weight > 0:
        if euler_weight is not None:
            # Mixed integrator mode: returns (combined_loss, {'euler': ..., 'cn': ...})
            phys_loss, integrator_losses = compute_physics_loss(
                pred_traj, x_0, rhs_fn,
                rollout_dt=config.rollout_dt,
                output_dim=config.output_dim,
                time_integrator=config.time_integrator,
                mae_weight=getattr(config, 'mae_weight', 0.0),
                euler_weight=euler_weight)
            loss_components['phys'] = phys_loss.item()
            loss_components['euler_loss'] = integrator_losses['euler']
            loss_components['cn_loss'] = integrator_losses['cn']
            loss_components['euler_weight'] = euler_weight
        else:
            # Single integrator mode
            phys_loss = compute_physics_loss(
                pred_traj, x_0, rhs_fn,
                rollout_dt=config.rollout_dt,
                output_dim=config.output_dim,
                time_integrator=config.time_integrator,
                mae_weight=getattr(config, 'mae_weight', 0.0))
            loss_components['phys'] = phys_loss.item()
        
        total_loss = phys_loss_weight * phys_loss
    else:
        total_loss = torch.tensor(0.0, device=pred_traj.device, dtype=pred_traj.dtype)
        loss_components['phys'] = 0.0
    
    # Supervised loss (only when target is provided)
    if x_target is not None:
        sup_weight = getattr(config, 'supervised_loss_weight', 0.0)
        
        if sup_weight > 0:
            # Get n_substeps for autoregressive rollout to match dt_data
            sup_n_substeps = getattr(config, 'supervised_n_substeps', 1)
            use_interp = bool(getattr(config, 'supervised_use_interpolation', 0))
            sup_mse_weight = getattr(config, 'supervised_mse_weight', 1.0)
            sup_mae_weight = getattr(config, 'supervised_mae_weight', 0.0)
            
            if sup_n_substeps > 1 and use_interp:
                # Supervise each 0.1-style autoregressive substep using a linearly
                # interpolated target between the two data frames.
                current = x_0
                total_sup_loss = torch.tensor(0.0, device=pred_traj.device, dtype=pred_traj.dtype)
                total_sup_mse = 0.0
                total_sup_mae = 0.0

                for sub_idx in range(sup_n_substeps):
                    if sub_idx == 0:
                        pred_next = pred_traj[..., -1]
                    else:
                        pred_sub = net(current)
                        pred_next = pred_sub[..., -1]

                    alpha = float(sub_idx + 1) / float(sup_n_substeps)
                    interp_target = torch.lerp(x_ref, x_target, alpha)
                    step_loss, step_metrics = _compute_supervised_loss(
                        pred_next, interp_target,
                        mse_weight=sup_mse_weight,
                        mae_weight=sup_mae_weight)
                    total_sup_loss = total_sup_loss + step_loss
                    total_sup_mse += step_metrics['mse']
                    total_sup_mae += step_metrics['mae']
                    current = pred_next

                sup_loss = total_sup_loss / sup_n_substeps
                sup_metrics = {
                    'mse': total_sup_mse / sup_n_substeps,
                    'mae': total_sup_mae / sup_n_substeps,
                }
                pred_next = current
            elif sup_n_substeps > 1:
                # Autoregressive rollout: iterate model sup_n_substeps times
                # This covers sup_n_substeps * rollout_dt time span
                current = x_0
                for sub_idx in range(sup_n_substeps):
                    if sub_idx == 0:
                        current = pred_traj[..., -1]
                    else:
                        pred_sub = net(current)
                        current = pred_sub[..., -1]
                pred_next = current  # (B, Nx, Ny, 5) at t + sup_n_substeps * rollout_dt
                sup_loss, sup_metrics = _compute_supervised_loss(
                    pred_next, x_target,
                    mse_weight=sup_mse_weight,
                    mae_weight=sup_mae_weight)
            else:
                # Single step: use last frame from initial forward pass
                # Prediction is at t + rollout_dt
                pred_next = pred_traj[..., -1]  # (B, Nx, Ny, 5)
                sup_loss, sup_metrics = _compute_supervised_loss(
                    pred_next, x_target,
                    mse_weight=sup_mse_weight,
                    mae_weight=sup_mae_weight)
            total_loss = total_loss + sup_weight * sup_loss
            loss_components['sup'] = sup_loss.item()
            loss_components['sup_mse'] = sup_metrics['mse']
            loss_components['sup_mae'] = sup_metrics['mae']
            loss_components['sup_n_substeps'] = sup_n_substeps
            loss_components['sup_interp'] = float(use_interp and sup_n_substeps > 1)

    # Backward pass: use accelerator if available
    if accelerator is not None:
        accelerator.backward(total_loss)
    else:
        total_loss.backward()

    if config.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    return total_loss.item(), loss_components


# ---------------------------------------------------------------------------
# Convergence curve (updated after each evaluation)
# ---------------------------------------------------------------------------

def _plot_convergence(eval_history, save_path, train_loss_history=None, 
                      eval_grf_history=None):
    """Plot evaluation and training loss vs training step.
    
    Args:
        eval_history: list of dicts with 'step' and 'mean_rel_l2' for MHD test set
        save_path: path to save the plot
        train_loss_history: list of dicts with training loss info
        eval_grf_history: optional list of dicts for GRF test set evaluation
    
    If mixed integrator is used, also plots Euler loss, CN loss, and euler_weight.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h['step'] for h in eval_history]
    mean_l2 = [h['mean_rel_l2'] for h in eval_history]
    best_idx = int(np.argmin(mean_l2))
    
    # Check if we have mixed integrator data
    has_mixed = (train_loss_history and len(train_loss_history) > 0 and
                 'euler_loss' in train_loss_history[0])
    
    if has_mixed:
        # Two subplots: (1) losses, (2) euler_weight
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax_weight = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot MHD test set evaluation loss
    ax.plot(steps, mean_l2, 'b-o', markersize=3, label='eval L2 (MHD)')
    ax.plot(steps[best_idx], mean_l2[best_idx], 'r*', markersize=12,
            label=f'best MHD={mean_l2[best_idx]:.6f} @ {steps[best_idx]}')
    
    # Plot GRF test set evaluation loss if available
    if eval_grf_history and len(eval_grf_history) > 0:
        grf_steps = [h['step'] for h in eval_grf_history]
        grf_l2 = [h['mean_rel_l2'] for h in eval_grf_history]
        grf_best_idx = int(np.argmin(grf_l2))
        ax.plot(grf_steps, grf_l2, 'c-s', markersize=3, label='eval L2 (GRF)')
        ax.plot(grf_steps[grf_best_idx], grf_l2[grf_best_idx], 'm*', markersize=10,
                label=f'best GRF={grf_l2[grf_best_idx]:.6f} @ {grf_steps[grf_best_idx]}')
    
    # Plot training loss if available
    if train_loss_history and len(train_loss_history) > 0:
        train_steps = [h['step'] for h in train_loss_history]
        train_loss = [h['train_loss'] for h in train_loss_history]
        ax.plot(train_steps, train_loss, 'g-', alpha=0.6, linewidth=1, label='train loss (avg)')
        
        # Plot Euler and CN losses if mixed integrator
        if has_mixed:
            euler_losses = [h.get('euler_loss', 0) for h in train_loss_history]
            cn_losses = [h.get('cn_loss', 0) for h in train_loss_history]
            ax.plot(train_steps, euler_losses, 'm-', alpha=0.5, linewidth=1, label='euler loss')
            ax.plot(train_steps, cn_losses, 'y-', alpha=0.5, linewidth=1, label='cn loss')
            
            # Plot euler_weight on secondary axis
            euler_weights = [h.get('euler_weight', 1.0) for h in train_loss_history]
            ax_weight.plot(train_steps, euler_weights, 'k-', linewidth=2, label='euler_weight')
            ax_weight.set_xlabel('Training step')
            ax_weight.set_ylabel('Euler Weight')
            ax_weight.set_title('Euler Weight Schedule (Exponential Decay)')
            ax_weight.set_ylim(-0.05, 1.05)
            ax_weight.grid(True, alpha=0.3)
            ax_weight.legend()
            ax_weight.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Evaluation Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Use log scale if loss varies significantly
    all_values = mean_l2[:]
    if eval_grf_history:
        all_values.extend([h['mean_rel_l2'] for h in eval_grf_history])
    if train_loss_history:
        all_values.extend([h['train_loss'] for h in train_loss_history])
    if len(all_values) > 1 and max(all_values) / max(min(all_values), 1e-10) > 10:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Logging / checkpointing (shared across all modes)
# ---------------------------------------------------------------------------

def _log_and_eval(config, net, global_step, loss_val, running_loss,
                  running_count, t_start, eval_data, mhd,
                  best_loss, run_tag, source_tag, eval_history,
                  train_loss_history=None, optimizer=None, loss_components=None,
                  eval_grf_data=None, eval_grf_history=None):
    """Handle periodic logging and evaluation. Returns updated best_loss.
    
    Evaluates on both MHD test set and GRF test set (if provided).
    
    Args:
        eval_data: MHD simulation test data (required)
        train_loss_history: list to append training loss for convergence plot
        optimizer: optimizer instance (required for lr logging in multi-GPU mode)
        loss_components: dict with individual loss components (euler_loss, cn_loss, euler_weight)
        eval_grf_data: optional GRF test data for additional evaluation
        eval_grf_history: optional list to append GRF evaluation results
    """
    if global_step % config.log_every == 0:
        avg = running_loss / max(running_count, 1)
        lr_now = optimizer.param_groups[0]['lr'] if optimizer else 0.0
        elapsed = time.time() - t_start
        it_s = global_step / max(elapsed, 1e-6)
        eta = (config.num_iterations - global_step) / max(it_s, 1e-6)
        
        # Base log message
        log_msg = (f'[{run_tag[:8]}] step {global_step:6d}/{config.num_iterations} '
                   f'| loss {loss_val:.6f} | avg {avg:.6f} '
                   f'| lr {lr_now:.2e} | {it_s:.1f} it/s | ETA {eta/60:.0f}m '
                   f'| src={source_tag}')
        
        # Add mixed integrator info if available
        if loss_components and 'euler_weight' in loss_components:
            euler_w = loss_components.get('euler_weight', 1.0)
            euler_l = loss_components.get('euler_loss', 0.0)
            cn_l = loss_components.get('cn_loss', 0.0)
            log_msg += f' | ew={euler_w:.3f} el={euler_l:.4f} cn={cn_l:.4f}'
        
        print(log_msg)
        sys.stdout.flush()
        
        # Record training loss for convergence plot
        if train_loss_history is not None:
            history_entry = {'step': global_step, 'train_loss': avg}
            # Add mixed integrator data if available
            if loss_components and 'euler_weight' in loss_components:
                history_entry['euler_loss'] = loss_components.get('euler_loss', 0.0)
                history_entry['cn_loss'] = loss_components.get('cn_loss', 0.0)
                history_entry['euler_weight'] = loss_components.get('euler_weight', 1.0)
            train_loss_history.append(history_entry)

    if global_step % config.eval_every == 0:
        do_plots = (global_step % (config.eval_every * 5) == 0)
        
        # === Evaluate on MHD test set ===
        print(f'  [Eval MHD] step {global_step}:')
        eval_results = evaluate(
            config, net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=do_plots,
            step_tag=f'step_{global_step}_mhd')

        current_loss = eval_results['mean_rel_l2']
        if current_loss < best_loss:
            best_loss = current_loss
            save_path = os.path.join(config.run_dir, 'model', f'best-{run_tag}.pt')
            torch.save({
                'model_state_dict': net.state_dict(),
                'config': vars(config),
                'step': global_step,
                'best_loss': best_loss,
                'eval_results': eval_results,
            }, save_path)
            print(f'  -> New best (MHD): {save_path} (L2={best_loss:.6f})')

        torch.save({
            'model_state_dict': net.state_dict(),
            'config': vars(config),
            'step': global_step,
            'best_loss': best_loss,
        }, os.path.join(config.run_dir, 'model', f'latest-{run_tag}.pt'))

        # Save intermediate checkpoint at specified intervals
        ckpt_every = getattr(config, 'checkpoint_every', 0)
        if ckpt_every > 0 and global_step % ckpt_every == 0:
            ckpt_path = os.path.join(config.run_dir, 'model', f'step_{global_step}-{run_tag}.pt')
            torch.save({
                'model_state_dict': net.state_dict(),
                'config': vars(config),
                'step': global_step,
                'best_loss': best_loss,
                'eval_results': eval_results,
            }, ckpt_path)
            print(f'  -> Saved intermediate checkpoint: {ckpt_path}')

        eval_history.append({'step': global_step, 'mean_rel_l2': current_loss})
        
        # === Evaluate on GRF test set (if available) ===
        if eval_grf_data is not None and eval_grf_history is not None:
            print(f'  [Eval GRF] step {global_step}:')
            grf_results = evaluate(
                config, net, eval_grf_data, mhd,
                n_rollout_steps=config.eval_rollout_steps,
                save_plots=do_plots,
                step_tag=f'step_{global_step}_grf')
            grf_loss = grf_results['mean_rel_l2']
            eval_grf_history.append({'step': global_step, 'mean_rel_l2': grf_loss})
            print(f'  [Summary] MHD L2={current_loss:.6f}, GRF L2={grf_loss:.6f}')
        
        # Update convergence plot with both histories
        conv_path = os.path.join(config.run_dir, 'vis', 'convergence.png')
        _plot_convergence(eval_history, conv_path, train_loss_history, eval_grf_history)

        net.train()
        sys.stdout.flush()

    return best_loss


# ---------------------------------------------------------------------------
# Main training loop (supports offline / online / staged / alternating)
# ---------------------------------------------------------------------------

def train(config, net, accelerator=None):
    """Main training loop with multiple data modes.

    data_mode:
        offline     — train exclusively on mhd_sim pre-generated data
        online      — train exclusively on GRF random initial conditions
        staged      — online for online_warmup_steps, then offline for remainder
        alternating — cycle: alternate_online_steps online, then
                      alternate_offline_steps offline, repeat
    
    Args:
        accelerator: Optional Accelerator instance for multi-GPU training.
    """
    # --- Multi-GPU setup via Accelerate ---
    is_main_process = True
    
    if accelerator is not None:
        is_main_process = accelerator.is_main_process
        device = accelerator.device
        if is_main_process:
            print(f'[Accelerate] Using {accelerator.num_processes} GPU(s), device={device}')
    else:
        device = config.device
    
    run_tag = getattr(config, 'run_tag', config.exp_name)
    data_mode = getattr(config, 'data_mode', 'offline')
    eval_data_path = getattr(config, 'eval_data_path', config.data_path)

    if is_main_process:
        if eval_data_path == config.data_path:
            print('  WARNING: eval_data_path not set, using training data for evaluation!')
        else:
            print(f'Evaluation will use test set: {eval_data_path}')

    # Training data overfitting flags
    is_sim_overfitting = bool(getattr(config, 'is_overfitting_test', 0))  # Simulation data overfitting
    overfitting_traj_idx = getattr(config, 'overfitting_traj_idx', 0)

    # Pre-load eval data once to avoid repeated disk I/O
    # For simulation overfitting test: use TRAINING data (same trajectory) to verify overfitting
    # For other modes: use test set for fair comparison
    if is_sim_overfitting:
        if is_main_process:
            print(f'Pre-loading evaluation data (TRAINING trajectory #{overfitting_traj_idx} for overfitting test)...')
        eval_data, eval_meta = load_mhd5_trajectories(
            config.data_path,  # Use training data path!
            time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data,
            single_trajectory=True,
            traj_idx=overfitting_traj_idx)
    else:
        if is_main_process:
            print(f'Pre-loading evaluation data (full test set)...')
        eval_data, eval_meta = load_mhd5_trajectories(
            eval_data_path, time_start=config.time_start, time_end=config.time_end,
            dt_data=config.dt_data,
            single_trajectory=False,
            traj_idx=0)
    
    # Load GRF test set if specified
    eval_grf_data_path = getattr(config, 'eval_grf_data_path', '')
    eval_grf_data = None
    if eval_grf_data_path and os.path.exists(eval_grf_data_path):
        if is_main_process:
            print(f'Pre-loading GRF test set: {eval_grf_data_path}')
        # GRF test set uses the same format as mhd_sim data
        grf_file = os.path.join(eval_grf_data_path, 'grf_testset.pt') if os.path.isdir(eval_grf_data_path) else eval_grf_data_path
        if os.path.exists(grf_file):
            eval_grf_data, grf_meta = load_mhd5_trajectories(
                grf_file, time_start=config.time_start, time_end=config.time_end,
                dt_data=config.dt_data,
                single_trajectory=False,
                traj_idx=0)
            if is_main_process:
                print(f'  GRF test set loaded: {eval_grf_data.shape}')
        else:
            if is_main_process:
                print(f'  WARNING: GRF test file not found: {grf_file}')
    elif eval_grf_data_path and is_main_process:
        print(f'  WARNING: eval_grf_data_path set but path does not exist: {eval_grf_data_path}')
    
    # Alias for backward compatibility in training data loading
    is_overfitting = is_sim_overfitting

    # --- Build components needed by each mode ---
    offline_sampler = None
    supervised_sampler = None
    grf = None
    fixed_grf_sampler = None

    need_offline = data_mode in ('offline', 'staged', 'alternating')
    need_online = data_mode in ('online', 'staged', 'alternating')
    
    # Check if supervised loss is enabled
    use_supervised = getattr(config, 'supervised_loss_weight', 0.0) > 0
    
    # Check if fixed GRF overfitting test is enabled
    use_fixed_grf = bool(getattr(config, 'is_grf_overfitting_test', 0))
    fixed_grf_seed = getattr(config, 'grf_overfitting_seed', 42)

    if need_offline:
        if is_main_process:
            print(f'Loading training data from {config.data_path}...')
        loader, meta = _make_offline_loader(config)
        offline_sampler = _OfflineSampler(loader)
        
        # Load supervised pairs if supervised loss is enabled
        if use_supervised:
            if is_main_process:
                print('Loading supervised (x_t, x_{t+1}) pairs for offline training...')
            sup_pairs, sup_meta = _load_supervised_pairs(config)
            pair_interp_steps = getattr(config, 'supervised_pair_interp_steps', 1)
            supervised_sampler = _SupervisedOfflineSampler(
                sup_pairs, config.batch_size, device=device,
                is_overfitting=is_overfitting, interp_steps=pair_interp_steps)
            if is_main_process:
                sup_n_substeps = getattr(config, 'supervised_n_substeps', 1)
                use_interp = bool(getattr(config, 'supervised_use_interpolation', 0))
                print(f'  Supervised pair interpolation steps: {pair_interp_steps}')
                expected_n_substeps = max(1, round(config.dt_data / config.rollout_dt))
                print(f'  Supervised loss weight: {config.supervised_loss_weight}')
                print(f'  Supervised n_substeps: {sup_n_substeps} '
                      f'(expected {expected_n_substeps} to match dt_data={config.dt_data}s)')
                print(f'  Supervised interpolation: {use_interp}')
                if sup_n_substeps != expected_n_substeps:
                    print(f'  [WARNING] supervised_n_substeps ({sup_n_substeps}) != '
                          f'expected ({expected_n_substeps}). '
                          f'Training/eval time scales may be mismatched!')

    # Model-evolved GRF generator (for self-training or external pretrained model)
    model_evolved_generator = None
    
    if need_online:
        if use_fixed_grf:
            if is_main_process:
                print('Building FIXED GRF generator for overfitting test...')
            grf_gen = _make_grf_generator(config, device=device)
            fixed_grf_sampler = FixedGRFDataSampler(grf_gen, fixed_seed=fixed_grf_seed)
            fixed_grf_sampler.generate_fixed_data(config.batch_size)
            grf = fixed_grf_sampler  # Use fixed sampler instead
        else:
            if is_main_process:
                print('Building GRF generator for online data...')
            grf = _make_grf_generator(config, device=device)
            
            # Check for model-evolved GRF modes
            pretrained_path = getattr(config, 'pretrained_model_path', None)
            self_training_start = getattr(config, 'self_training_start_step', 0)
            
            # Warn if both modes are specified (pretrained takes priority)
            if pretrained_path is not None and self_training_start > 0:
                if is_main_process:
                    print(f'[WARNING] Both pretrained_model_path and self_training_start_step are set.')
                    print(f'  pretrained_model_path takes priority, self_training_start_step will be IGNORED.')
            
            # Warn if self_training_start_step >= num_iterations
            if self_training_start > 0 and self_training_start >= config.num_iterations:
                if is_main_process:
                    print(f'[WARNING] self_training_start_step ({self_training_start}) >= num_iterations ({config.num_iterations})')
                    print(f'  Self-training will never be activated!')
            
            # Warn about staged mode + self-training interaction
            if data_mode == 'staged' and self_training_start > 0:
                online_warmup = getattr(config, 'online_warmup_steps', 0)
                if is_main_process:
                    print(f'[INFO] Using staged mode with self-training:')
                    print(f'  - Steps 0 to {self_training_start-1}: raw GRF (online phase)')
                    print(f'  - Steps {self_training_start} to {online_warmup-1}: model-evolved GRF (online phase)')
                    print(f'  - Steps {online_warmup}+: offline data (self-training NOT used)')
                    if self_training_start >= online_warmup:
                        print(f'  [WARNING] self_training_start_step ({self_training_start}) >= online_warmup_steps ({online_warmup})')
                        print(f'    Self-training will never be activated (offline phase starts first)!')
            
            if pretrained_path is not None:
                # External model mode: load pretrained model and activate immediately
                if not os.path.exists(pretrained_path):
                    raise FileNotFoundError(f'Pretrained model not found: {pretrained_path}')
                model_evolved_generator = _make_pretrained_generator(
                    grf, pretrained_path, config, device=device, verbose=is_main_process)
                grf = model_evolved_generator  # Replace GRF with model-evolved generator
                
            elif self_training_start > 0:
                # Self-training mode: use training model's copy
                # NOTE: We create the generator later after model is on device
                if is_main_process:
                    print(f'[Self-Training Mode] Will create generator after model initialization')
                    print(f'  Activation step: {self_training_start}')
                # Mark that we need to create self-training generator after model is ready
                config._need_self_training_generator = True

    if is_main_process:
        print(f'Building MHD instance on {device}...')
    mhd = build_mhd_instance(device=device, Nx=config.Nx, Ny=config.Ny)
    dealias_rhs = bool(getattr(config, 'dealias_rhs', False))
    rhs_fn = make_mhd5_rhs_fn(mhd, model_dtype=torch.float32, dealias=dealias_rhs)
    if is_main_process:
        dealias_input = bool(getattr(config, 'dealias_input', True))
        print(f'  Dealiasing: dealias_input={dealias_input}, dealias_rhs={dealias_rhs}')

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config.lr, config.num_iterations)

    # Wrap with Accelerator if using multi-GPU
    if accelerator is not None:
        net, optimizer, scheduler = accelerator.prepare(net, optimizer, scheduler)

    # Create self-training generator AFTER model is on device and wrapped
    # This ensures we copy the correct initial weights
    need_self_training_gen = getattr(config, '_need_self_training_generator', False)
    if need_self_training_gen and need_online:
        # Get unwrapped model for weight copying
        source_net = accelerator.unwrap_model(net) if accelerator is not None else net
        model_evolved_generator = _make_self_training_generator(
            grf, source_net, config, device=device, verbose=is_main_process)
        grf = model_evolved_generator  # Replace GRF with model-evolved generator
    # Clean up temporary flag (always, to avoid state leakage)
    if hasattr(config, '_need_self_training_generator'):
        del config._need_self_training_generator

    train_dt = config.rollout_dt / config.output_dim
    n_substeps_eval = max(1, round(config.dt_data / config.rollout_dt))
    if is_main_process:
        print(f'  dt hierarchy: rollout_dt={config.rollout_dt}, '
              f'train_dt={train_dt:.4f} (output_dim={config.output_dim}), '
              f'dt_data={config.dt_data}, n_substeps_eval={n_substeps_eval}')
        print(f'  data_mode={data_mode}')
        if data_mode == 'staged':
            print(f'    online_warmup_steps={config.online_warmup_steps}')
        elif data_mode == 'alternating':
            print(f'    alternate: {config.alternate_online_steps} online / '
                  f'{config.alternate_offline_steps} offline per cycle')
        
        # Mixed integrator info
        if getattr(config, 'use_mixed_integrator', 0):
            print(f'  [Mixed Integrator] ENABLED:')
            print(f'    euler_weight_init={config.euler_weight_init}, '
                  f'euler_weight_min={config.euler_weight_min}')
            print(f'    euler_anneal_start={config.euler_anneal_start}, '
                  f'euler_half_life={config.euler_half_life}')
        else:
            print(f'  time_integrator={config.time_integrator}')
        
        print(f'\n[{run_tag}] Training: {config.num_iterations} iters, '
              f'bs={config.batch_size}, lr={config.lr}')
        print('-' * 60)
        sys.stdout.flush()

    # --- Initial evaluation (on both MHD and GRF test sets) ---
    # Only run evaluation on main process to avoid duplicate computation
    init_grf_loss = float('inf')
    if is_main_process:
        print('Initial evaluation (step 0) ...')
        # For DDP, unwrap model for evaluation
        eval_net = accelerator.unwrap_model(net) if accelerator is not None else net
        
        # Evaluate on MHD test set
        print('  [Eval MHD] step 0:')
        init_results = evaluate(
            config, eval_net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=True, step_tag='step_0_mhd')
        best_loss = init_results['mean_rel_l2']
        
        # Evaluate on GRF test set if available
        if eval_grf_data is not None:
            print('  [Eval GRF] step 0:')
            init_grf_results = evaluate(
                config, eval_net, eval_grf_data, mhd,
                n_rollout_steps=config.eval_rollout_steps,
                save_plots=True, step_tag='step_0_grf')
            init_grf_loss = init_grf_results['mean_rel_l2']
            print(f'  [Summary] MHD L2={best_loss:.6f}, GRF L2={init_grf_loss:.6f}')
        
        sys.stdout.flush()
    else:
        best_loss = float('inf')
    
    # Sync all processes after initial evaluation
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    eval_history = [{'step': 0, 'mean_rel_l2': best_loss}]
    eval_grf_history = [{'step': 0, 'mean_rel_l2': init_grf_loss}] if eval_grf_data is not None else None
    train_loss_history = []  # Track training loss for convergence plot

    t_start = time.time()
    global_step = 0
    running_loss = 0.0
    running_count = 0

    # --- Helper: decide data source for current step ---
    def _get_source(step):
        if data_mode == 'offline':
            return 'offline'
        if data_mode == 'online':
            return 'online'
        if data_mode == 'staged':
            return 'online' if step < config.online_warmup_steps else 'offline'
        # alternating
        cycle = config.alternate_online_steps + config.alternate_offline_steps
        pos = step % cycle
        return 'online' if pos < config.alternate_online_steps else 'offline'

    prev_source = None

    # Self-training configuration
    self_training_start = getattr(config, 'self_training_start_step', 0)
    self_training_update_every = getattr(config, 'self_training_update_every', 0)
    
    while global_step < config.num_iterations:
        source = _get_source(global_step)
        
        # === Self-training mode: activation and periodic weight updates ===
        if model_evolved_generator is not None and not use_fixed_grf:
            # Check if we need to activate self-training
            if (not model_evolved_generator.is_active() 
                and self_training_start > 0 
                and global_step >= self_training_start):
                # Update weights from current training model before activation
                source_net = accelerator.unwrap_model(net) if accelerator is not None else net
                model_evolved_generator.update_model_weights(source_net)
                model_evolved_generator.activate()
                if is_main_process:
                    print(f'\n[Self-Training] ACTIVATED at step {global_step}')
                    print(f'  Data will now be: GRF -> Model({self_training_start}) -> evolved state\n')
                    sys.stdout.flush()
            
            # Check if we need to update weights (only for self-training mode, not external model)
            elif (model_evolved_generator.is_active() 
                  and self_training_update_every > 0
                  and self_training_start > 0  # Only self-training mode updates
                  and global_step > self_training_start
                  and (global_step - self_training_start) % self_training_update_every == 0):
                source_net = accelerator.unwrap_model(net) if accelerator is not None else net
                model_evolved_generator.update_model_weights(source_net)
                if is_main_process:
                    print(f'\n[Self-Training] Weights REFRESHED at step {global_step}\n')
                    sys.stdout.flush()

        # Staged mode: reset scheduler at the transition point
        if data_mode == 'staged' and prev_source == 'online' and source == 'offline':
            remaining = config.num_iterations - global_step
            scheduler = _make_scheduler(optimizer, config.lr, remaining)
            # Wrap with accelerator if using multi-GPU (must be done on all processes)
            if accelerator is not None:
                scheduler = accelerator.prepare(scheduler)
            if is_main_process:
                print(f'\n[{run_tag[:8]}] === STAGED SWITCH: online -> offline at step '
                      f'{global_step} (remaining {remaining} steps, lr scheduler reset) ===\n')
            running_loss = 0.0
            running_count = 0
            if is_main_process:
                sys.stdout.flush()
        prev_source = source

        # Compute euler_weight for mixed integrator mode
        euler_weight = _get_euler_weight(global_step, config)

        if source == 'online':
            x_0 = grf(config.batch_size)
            net.train()
            loss_val, loss_components = _train_step(
                net, x_0, rhs_fn, optimizer, scheduler, config,
                accelerator=accelerator, euler_weight=euler_weight, mhd=mhd)
        else:
            # Offline mode: can use supervised loss if enabled
            if use_supervised and supervised_sampler is not None:
                x_0, x_target = supervised_sampler.next_batch()
                net.train()
                loss_val, loss_components = _train_step(
                    net, x_0, rhs_fn, optimizer, scheduler, config,
                    x_target, accelerator=accelerator, euler_weight=euler_weight, mhd=mhd)
            else:
                x_0 = offline_sampler.next_batch().to(device)
                net.train()
                loss_val, loss_components = _train_step(
                    net, x_0, rhs_fn, optimizer, scheduler, config,
                    accelerator=accelerator, euler_weight=euler_weight, mhd=mhd)

        running_loss += loss_val
        running_count += 1
        global_step += 1

        # Sync BEFORE evaluation to prevent deadlock:
        # - Without sync, non-main processes continue to next training step
        # - Next step's backward() requires all processes to participate
        # - But main process is still evaluating -> deadlock
        if accelerator is not None and global_step % config.eval_every == 0:
            accelerator.wait_for_everyone()

        # Only log and eval on main process
        if is_main_process:
            eval_net = accelerator.unwrap_model(net) if accelerator is not None else net
            best_loss = _log_and_eval(
                config, eval_net, global_step, loss_val, running_loss, running_count,
                t_start, eval_data, mhd, best_loss, run_tag, source,
                eval_history, train_loss_history=train_loss_history,
                optimizer=optimizer, loss_components=loss_components,
                eval_grf_data=eval_grf_data, eval_grf_history=eval_grf_history)
        
        # Sync AFTER evaluation so non-main processes wait for main to finish
        if accelerator is not None and global_step % config.eval_every == 0:
            accelerator.wait_for_everyone()

        if global_step % config.log_every == 0:
            running_loss = 0.0
            running_count = 0

    # --- Final evaluation with best model (main process only) ---
    if is_main_process:
        best_path = os.path.join(config.run_dir, 'model', f'best-{run_tag}.pt')
        print(f'\n--- Final Evaluation [{run_tag}] ---')
        eval_net = accelerator.unwrap_model(net) if accelerator is not None else net
        if os.path.exists(best_path):
            eval_net.load_state_dict(
                torch.load(best_path, map_location=device,
                           weights_only=False)['model_state_dict'])
        else:
            print(f'  WARNING: best checkpoint not found at {best_path}, using latest weights')

        # Final evaluation on MHD test set
        print(f'  [Final Eval MHD]:')
        final_results = evaluate(
            config, eval_net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=True, step_tag=f'final-{run_tag}_mhd')

        results_path = os.path.join(config.run_dir, f'eval-{run_tag}_mhd.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f'  MHD results saved to {results_path}')
        
        # Final evaluation on GRF test set if available
        if eval_grf_data is not None:
            print(f'  [Final Eval GRF]:')
            final_grf_results = evaluate(
                config, eval_net, eval_grf_data, mhd,
                n_rollout_steps=config.eval_rollout_steps,
                save_plots=True, step_tag=f'final-{run_tag}_grf')
            
            grf_results_path = os.path.join(config.run_dir, f'eval-{run_tag}_grf.json')
            with open(grf_results_path, 'w') as f:
                json.dump(final_grf_results, f, indent=2)
            print(f'  GRF results saved to {grf_results_path}')
            
            print(f'\n  === Final Summary ===')
            print(f'  MHD Test L2: {final_results["mean_rel_l2"]:.6f}')
            print(f'  GRF Test L2: {final_grf_results["mean_rel_l2"]:.6f}')

        total_time = time.time() - t_start
        print(f'Total training time: {total_time/3600:.1f}h ({total_time/60:.0f}m)')
    sys.stdout.flush()
    
    # Wait for all processes to finish
    if accelerator is not None:
        accelerator.wait_for_everyone()
