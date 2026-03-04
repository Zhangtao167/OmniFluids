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

from tools import load_mhd5_snapshots, load_mhd5_trajectories, MHD5FieldGRF, FixedGRFDataSampler
from psm_loss import build_mhd_instance, make_mhd5_rhs_fn, compute_physics_loss

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

    # ===== 2. GT / Pred / Error snapshots (3 rows per field) =====
    plot_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    plot_steps = sorted(set(s for s in plot_steps if s < T_total))
    b_idx = 0
    n_cols = len(plot_steps)

    for c, name in enumerate(FIELD_NAMES):
        fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(3, 1)
        for j, t in enumerate(plot_steps):
            gt_snap = gt_traj[b_idx, t, c].cpu().numpy()
            pred_snap = pred_traj[b_idx, t, c].cpu().numpy()
            err_snap = pred_snap - gt_snap
            vmin, vmax = gt_snap.min(), gt_snap.max()
            err_abs = max(abs(err_snap.min()), abs(err_snap.max()), 1e-10)
            t_phys = time_start + t * n_substeps * rollout_dt
            ic_tag = ' (IC)' if t == 0 else ''

            im0 = axes[0, j].imshow(gt_snap, aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[0, j].set_title(f'GT t={t_phys:.1f}s{ic_tag}', fontsize=9)
            fig.colorbar(im0, ax=axes[0, j], fraction=0.046)

            im1 = axes[1, j].imshow(pred_snap, aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1, j].set_title(f'Pred t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im1, ax=axes[1, j], fraction=0.046)

            im2 = axes[2, j].imshow(err_snap, aspect='auto',
                                    vmin=-err_abs, vmax=err_abs, cmap='bwr')
            axes[2, j].set_title(f'Err t={t_phys:.1f}s', fontsize=9)
            fig.colorbar(im2, ax=axes[2, j], fraction=0.046)

            if j == 0:
                axes[0, j].set_ylabel('GT\nx (radial)')
                axes[1, j].set_ylabel('Pred\nx (radial)')
                axes[2, j].set_ylabel('Error\nx (radial)')
            axes[2, j].set_xlabel('y (binormal)')

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
        'mean_rel_l2': np.mean(rel_l2_total),
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
        vis_dir = os.path.join(run_dir, 'vis')
        save_eval_plots(pred_traj, gt, rel_l2_total, rel_l2_per_field,
                        vis_dir, tag=step_tag,
                        time_start=config.time_start,
                        rollout_dt=config.rollout_dt,
                        n_substeps=n_substeps)

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
    """Infinite iterator over supervised pairs, auto-resets on exhaustion."""
    def __init__(self, pairs, batch_size, device='cpu', is_overfitting=False):
        self.pairs = pairs
        self.device = device
        self.n_pairs = len(pairs)
        self.batch_size = batch_size  # Keep batch_size unchanged
        self.is_overfitting = is_overfitting
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
# Unified training step
# ---------------------------------------------------------------------------

def _train_step(net, x_0, rhs_fn, optimizer, scheduler, config, x_target=None, accelerator=None):
    """Execute one training step. Returns loss value.
    
    Args:
        x_target: Optional target state for supervised loss. If provided,
                  computes MSE/MAE between pred[..., -1] and x_target.
        accelerator: Optional Accelerator instance for multi-GPU training.
    """
    noise_scale = getattr(config, 'input_noise_scale', 0.0)
    if noise_scale > 0:
        x_0 = x_0 + noise_scale * torch.randn_like(x_0)
    pred_traj = net(x_0)

    # Physics loss (PDE loss)
    phys_loss_weight = getattr(config, 'physics_loss_weight', 1.0)
    if phys_loss_weight > 0:
        phys_loss = compute_physics_loss(
            pred_traj, x_0, rhs_fn,
            rollout_dt=config.rollout_dt,
            output_dim=config.output_dim,
            time_integrator=config.time_integrator,
            mae_weight=getattr(config, 'mae_weight', 0.0))
        total_loss = phys_loss_weight * phys_loss
        loss_components = {'phys': phys_loss.item()}
    else:
        total_loss = torch.tensor(0.0, device=pred_traj.device, dtype=pred_traj.dtype)
        loss_components = {'phys': 0.0}
    
    # Supervised loss (only when target is provided)
    if x_target is not None:
        # Use the last predicted frame as the next state prediction
        pred_next = pred_traj[..., -1]  # (B, Nx, Ny, 5)
        
        sup_weight = getattr(config, 'supervised_loss_weight', 0.0)
        sup_mse_weight = getattr(config, 'supervised_mse_weight', 1.0)
        sup_mae_weight = getattr(config, 'supervised_mae_weight', 0.0)
        
        if sup_weight > 0:
            sup_loss, sup_metrics = _compute_supervised_loss(
                pred_next, x_target, 
                mse_weight=sup_mse_weight,
                mae_weight=sup_mae_weight)
            total_loss = total_loss + sup_weight * sup_loss
            loss_components['sup'] = sup_loss.item()
            loss_components['sup_mse'] = sup_metrics['mse']
            loss_components['sup_mae'] = sup_metrics['mae']

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
    
    return total_loss.item()


# ---------------------------------------------------------------------------
# Convergence curve (updated after each evaluation)
# ---------------------------------------------------------------------------

def _plot_convergence(eval_history, save_path, train_loss_history=None):
    """Plot evaluation and training loss vs training step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h['step'] for h in eval_history]
    mean_l2 = [h['mean_rel_l2'] for h in eval_history]
    best_idx = int(np.argmin(mean_l2))

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot evaluation loss
    ax.plot(steps, mean_l2, 'b-o', markersize=3, label='eval loss')
    ax.plot(steps[best_idx], mean_l2[best_idx], 'r*', markersize=12,
            label=f'best={mean_l2[best_idx]:.6f} @ step {steps[best_idx]}')
    
    # Plot training loss if available
    if train_loss_history and len(train_loss_history) > 0:
        train_steps = [h['step'] for h in train_loss_history]
        train_loss = [h['train_loss'] for h in train_loss_history]
        ax.plot(train_steps, train_loss, 'g-', alpha=0.6, linewidth=1, label='train loss (avg)')
    
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Evaluation Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale if loss varies significantly
    all_values = mean_l2[:]
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
                  train_loss_history=None, optimizer=None):
    """Handle periodic logging and evaluation. Returns updated best_loss.
    
    Evaluation always uses mhd_sim test set for fair comparison, regardless of training data source.
    
    Args:
        train_loss_history: list to append training loss for convergence plot
        optimizer: optimizer instance (required for lr logging in multi-GPU mode)
    """
    if global_step % config.log_every == 0:
        avg = running_loss / max(running_count, 1)
        lr_now = optimizer.param_groups[0]['lr'] if optimizer else 0.0
        elapsed = time.time() - t_start
        it_s = global_step / max(elapsed, 1e-6)
        eta = (config.num_iterations - global_step) / max(it_s, 1e-6)
        print(f'[{run_tag[:8]}] step {global_step:6d}/{config.num_iterations} '
              f'| loss {loss_val:.6f} | avg {avg:.6f} '
              f'| lr {lr_now:.2e} | {it_s:.1f} it/s | ETA {eta/60:.0f}m '
              f'| src={source_tag}')
        sys.stdout.flush()
        
        # Record training loss for convergence plot
        if train_loss_history is not None:
            train_loss_history.append({'step': global_step, 'train_loss': avg})

    if global_step % config.eval_every == 0:
        do_plots = (global_step % (config.eval_every * 5) == 0)
        
        # Always use mhd_sim test set for fair comparison
        eval_results = evaluate(
            config, net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=do_plots,
            step_tag=f'step_{global_step}')

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
            print(f'  -> New best: {save_path} (loss={best_loss:.6f})')

        torch.save({
            'model_state_dict': net.state_dict(),
            'config': vars(config),
            'step': global_step,
            'best_loss': best_loss,
        }, os.path.join(config.run_dir, 'model', f'latest-{run_tag}.pt'))

        eval_history.append({'step': global_step, 'mean_rel_l2': current_loss})
        conv_path = os.path.join(config.run_dir, 'vis', 'convergence.png')
        _plot_convergence(eval_history, conv_path, train_loss_history)

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
            supervised_sampler = _SupervisedOfflineSampler(
                sup_pairs, config.batch_size, device=device, is_overfitting=is_overfitting)
            if is_main_process:
                print(f'  Supervised loss weight: {config.supervised_loss_weight}')

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

    if is_main_process:
        print(f'Building MHD instance on {device}...')
    mhd = build_mhd_instance(device=device, Nx=config.Nx, Ny=config.Ny)
    rhs_fn = make_mhd5_rhs_fn(mhd, model_dtype=torch.float32)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config.lr, config.num_iterations)

    # Wrap with Accelerator if using multi-GPU
    if accelerator is not None:
        net, optimizer, scheduler = accelerator.prepare(net, optimizer, scheduler)

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
        print(f'\n[{run_tag}] Training: {config.num_iterations} iters, '
              f'bs={config.batch_size}, lr={config.lr}')
        print('-' * 60)
        sys.stdout.flush()

    # --- Initial evaluation (always use mhd_sim test set for fair comparison) ---
    # Only run evaluation on main process to avoid duplicate computation
    if is_main_process:
        print('Initial evaluation (step 0) ...')
        # For DDP, unwrap model for evaluation
        eval_net = accelerator.unwrap_model(net) if accelerator is not None else net
        init_results = evaluate(
            config, eval_net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=True, step_tag='step_0')
        best_loss = init_results['mean_rel_l2']
        sys.stdout.flush()
    else:
        best_loss = float('inf')
    
    # Sync all processes after initial evaluation
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    eval_history = [{'step': 0, 'mean_rel_l2': best_loss}]
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

    while global_step < config.num_iterations:
        source = _get_source(global_step)

        # Staged mode: reset scheduler at the transition point
        if data_mode == 'staged' and prev_source == 'online' and source == 'offline':
            remaining = config.num_iterations - global_step
            scheduler = _make_scheduler(optimizer, config.lr, remaining)
            if is_main_process:
                print(f'\n[{run_tag[:8]}] === STAGED SWITCH: online -> offline at step '
                      f'{global_step} (remaining {remaining} steps, lr scheduler reset) ===\n')
            running_loss = 0.0
            running_count = 0
            if is_main_process:
                sys.stdout.flush()
        prev_source = source

        if source == 'online':
            x_0 = grf(config.batch_size)
            net.train()
            loss_val = _train_step(net, x_0, rhs_fn, optimizer, scheduler, config, accelerator=accelerator)
        else:
            # Offline mode: can use supervised loss if enabled
            if use_supervised and supervised_sampler is not None:
                x_0, x_target = supervised_sampler.next_batch()
                net.train()
                loss_val = _train_step(net, x_0, rhs_fn, optimizer, scheduler, config, x_target, accelerator=accelerator)
            else:
                x_0 = offline_sampler.next_batch().to(device)
                net.train()
                loss_val = _train_step(net, x_0, rhs_fn, optimizer, scheduler, config, accelerator=accelerator)

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
                eval_history, train_loss_history=train_loss_history, optimizer=optimizer)
        
        # Sync AFTER evaluation so non-main processes wait for main to finish
        if accelerator is not None and global_step % config.eval_every == 0:
            accelerator.wait_for_everyone()

        if is_main_process and global_step % config.log_every == 0:
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

        # Final evaluation always uses mhd_sim test set
        final_results = evaluate(
            config, eval_net, eval_data, mhd,
            n_rollout_steps=config.eval_rollout_steps,
            save_plots=True, step_tag=f'final-{run_tag}')

        results_path = os.path.join(config.run_dir, f'eval-{run_tag}.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f'Results saved to {results_path}')

        total_time = time.time() - t_start
        print(f'Total training time: {total_time/3600:.1f}h ({total_time/60:.0f}m)')
        sys.stdout.flush()
    
    # Wait for all processes to finish
    if accelerator is not None:
        accelerator.wait_for_everyone()
