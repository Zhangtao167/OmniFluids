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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tools import load_mhd5_snapshots, load_mhd5_trajectories, MHD5FieldGRF
from psm_loss import build_mhd_instance, make_mhd5_rhs_fn, compute_physics_loss

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
# Training helpers
# ---------------------------------------------------------------------------

def _make_offline_loader(config):
    """Load mhd_sim data and build DataLoader."""
    snapshots, meta = load_mhd5_snapshots(
        config.data_path,
        time_start=config.time_start,
        time_end=config.time_end,
        dt_data=config.dt_data)
    Nx, Ny = meta['Nx'], meta['Ny']
    assert Nx == config.Nx and Ny == config.Ny, \
        f'Grid mismatch: data ({Nx},{Ny}) vs config ({config.Nx},{config.Ny})'
    dataset = TensorDataset(snapshots)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    return loader, meta


def _make_grf_generator(config):
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
# Unified training step
# ---------------------------------------------------------------------------

def _train_step(net, x_0, rhs_fn, optimizer, scheduler, config):
    """Execute one training step. Returns loss value."""
    noise_scale = getattr(config, 'input_noise_scale', 0.0)
    if noise_scale > 0:
        x_0 = x_0 + noise_scale * torch.randn_like(x_0)
    pred_traj = net(x_0)

    loss = compute_physics_loss(
        pred_traj, x_0, rhs_fn,
        rollout_dt=config.rollout_dt,
        output_dim=config.output_dim,
        time_integrator=config.time_integrator,
        mae_weight=getattr(config, 'mae_weight', 0.0))

    loss.backward()
    if config.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Convergence curve (updated after each evaluation)
# ---------------------------------------------------------------------------

def _plot_convergence(eval_history, save_path):
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
    plt.close(fig)


# ---------------------------------------------------------------------------
# Logging / checkpointing (shared across all modes)
# ---------------------------------------------------------------------------

def _log_and_eval(config, net, global_step, loss_val, running_loss,
                  running_count, t_start, eval_data, mhd,
                  best_loss, run_tag, source_tag, eval_history):
    """Handle periodic logging and evaluation. Returns updated best_loss."""
    if global_step % config.log_every == 0:
        avg = running_loss / max(running_count, 1)
        lr_now = net._optimizer_ref.param_groups[0]['lr']
        elapsed = time.time() - t_start
        it_s = global_step / max(elapsed, 1e-6)
        eta = (config.num_iterations - global_step) / max(it_s, 1e-6)
        print(f'[{run_tag[:8]}] step {global_step:6d}/{config.num_iterations} '
              f'| loss {loss_val:.6f} | avg {avg:.6f} '
              f'| lr {lr_now:.2e} | {it_s:.1f} it/s | ETA {eta/60:.0f}m '
              f'| src={source_tag}')
        sys.stdout.flush()

    if global_step % config.eval_every == 0:
        do_plots = (global_step % (config.eval_every * 5) == 0)
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
            print(f'  -> New best: {save_path} (rel_l2={best_loss:.6f})')

        torch.save({
            'model_state_dict': net.state_dict(),
            'config': vars(config),
            'step': global_step,
            'best_loss': best_loss,
        }, os.path.join(config.run_dir, 'model', f'latest-{run_tag}.pt'))

        # Record training loss (average since last eval)
        avg_train_loss = running_loss / max(running_count, 1) if running_count > 0 else float('nan')
        eval_history.append({'step': global_step, 'mean_rel_l2': current_loss, 'train_loss': avg_train_loss})
        conv_path = os.path.join(config.run_dir, 'vis', 'convergence.png')
        _plot_convergence(eval_history, conv_path)

        net.train()
        sys.stdout.flush()

    return best_loss


# ---------------------------------------------------------------------------
# Main training loop (supports offline / online / staged / alternating)
# ---------------------------------------------------------------------------

def train(config, net):
    """Main training loop with multiple data modes.

    data_mode:
        offline     — train exclusively on mhd_sim pre-generated data
        online      — train exclusively on GRF random initial conditions
        staged      — online for online_warmup_steps, then offline for remainder
        alternating — cycle: alternate_online_steps online, then
                      alternate_offline_steps offline, repeat
    """
    device = config.device
    run_tag = getattr(config, 'run_tag', config.exp_name)
    data_mode = getattr(config, 'data_mode', 'offline')
    eval_data_path = getattr(config, 'eval_data_path', config.data_path)

    if eval_data_path == config.data_path:
        print('  WARNING: eval_data_path not set, using training data for evaluation!')
    else:
        print(f'Evaluation will use test set: {eval_data_path}')

    # Pre-load eval data once to avoid repeated disk I/O
    print(f'Pre-loading evaluation data...')
    eval_data, eval_meta = load_mhd5_trajectories(
        eval_data_path, time_start=config.time_start, time_end=config.time_end,
        dt_data=config.dt_data)

    # --- Build components needed by each mode ---
    offline_sampler = None
    grf = None

    need_offline = data_mode in ('offline', 'staged', 'alternating')
    need_online = data_mode in ('online', 'staged', 'alternating')

    if need_offline:
        print(f'Loading training data from {config.data_path}...')
        loader, meta = _make_offline_loader(config)
        offline_sampler = _OfflineSampler(loader)

    if need_online:
        print('Building GRF generator for online data...')
        grf = _make_grf_generator(config)

    print(f'Building MHD instance on {device}...')
    mhd = build_mhd_instance(device=device, Nx=config.Nx, Ny=config.Ny)
    rhs_fn = make_mhd5_rhs_fn(mhd, model_dtype=torch.float32)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config.lr, config.num_iterations)

    # Expose optimizer for logging helper
    net._optimizer_ref = optimizer

    train_dt = config.rollout_dt / config.output_dim
    n_substeps_eval = max(1, round(config.dt_data / config.rollout_dt))
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

    # --- Initial evaluation ---
    print('Initial evaluation (step 0) ...')
    init_results = evaluate(
        config, net, eval_data, mhd,
        n_rollout_steps=config.eval_rollout_steps,
        save_plots=True, step_tag='step_0')
    best_loss = init_results['mean_rel_l2']
    eval_history = [{'step': 0, 'mean_rel_l2': best_loss, 'train_loss': float('nan')}]
    sys.stdout.flush()

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
            print(f'\n[{run_tag[:8]}] === STAGED SWITCH: online -> offline at step '
                  f'{global_step} (remaining {remaining} steps, lr scheduler reset) ===\n')
            running_loss = 0.0
            running_count = 0
            sys.stdout.flush()
        prev_source = source

        if source == 'online':
            x_0 = grf(config.batch_size)
        else:
            x_0 = offline_sampler.next_batch().to(device)

        net.train()
        loss_val = _train_step(net, x_0, rhs_fn, optimizer, scheduler, config)

        running_loss += loss_val
        running_count += 1
        global_step += 1

        best_loss = _log_and_eval(
            config, net, global_step, loss_val, running_loss, running_count,
            t_start, eval_data, mhd, best_loss, run_tag, source,
            eval_history)

        if global_step % config.log_every == 0:
            running_loss = 0.0
            running_count = 0

    # --- Final evaluation with best model ---
    best_path = os.path.join(config.run_dir, 'model', f'best-{run_tag}.pt')
    print(f'\n--- Final Evaluation [{run_tag}] ---')
    if os.path.exists(best_path):
        net.load_state_dict(
            torch.load(best_path, map_location=device,
                       weights_only=False)['model_state_dict'])
    else:
        print(f'  WARNING: best checkpoint not found at {best_path}, using latest weights')

    final_results = evaluate(
        config, net, eval_data, mhd,
        n_rollout_steps=config.eval_rollout_steps,
        save_plots=True, step_tag=f'final-{run_tag}')

    results_path = os.path.join(config.run_dir, f'eval-{run_tag}.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f'Results saved to {results_path}')

    total_time = time.time() - t_start
    print(f'Total training time: {total_time/3600:.1f}h ({total_time/60:.0f}m)')
    sys.stdout.flush()
