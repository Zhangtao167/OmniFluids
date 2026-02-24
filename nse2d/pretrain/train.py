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

from tools import load_mhd5_snapshots, load_mhd5_trajectories
from psm_loss import build_mhd_instance, make_mhd5_rhs_fn, compute_physics_loss

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']


# ---------------------------------------------------------------------------
# Visualization helpers (aligned with mhd_sim/eval_5field.py style)
# ---------------------------------------------------------------------------

def save_eval_plots(pred_traj, gt_traj, rel_l2_total, rel_l2_per_field,
                    save_dir, tag='eval',
                    time_start=250.0, rollout_dt=0.1, n_substeps=10):
    """Save evaluation plots: per-step error curves + snapshot comparison.

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

    # 1) Per-step error curves with dual x-axis (NFE + physical time)
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

    # 2) Snapshot comparison at selected time steps
    plot_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    plot_steps = sorted(set(s for s in plot_steps if s < pred_traj.shape[1]))
    b_idx = 0

    for c, name in enumerate(FIELD_NAMES):
        fig, axes = plt.subplots(2, len(plot_steps), figsize=(4 * len(plot_steps), 7))
        if len(plot_steps) == 1:
            axes = axes.reshape(2, 1)
        for j, t in enumerate(plot_steps):
            gt_snap = gt_traj[b_idx, t, c].cpu().numpy()    # (Nx, Ny)
            pred_snap = pred_traj[b_idx, t, c].cpu().numpy()
            vmin, vmax = gt_snap.min(), gt_snap.max()
            t_phys = time_start + t * n_substeps * rollout_dt
            ic_tag = ' (IC)' if t == 0 else ''

            im0 = axes[0, j].imshow(gt_snap.T, origin='lower', aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[0, j].set_title(f'GT t={t_phys:.1f}s{ic_tag}', fontsize=9)
            fig.colorbar(im0, ax=axes[0, j], fraction=0.046)
            if j == 0:
                axes[0, j].set_ylabel('y (binormal)')

            im1 = axes[1, j].imshow(pred_snap.T, origin='lower', aspect='auto',
                                    vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1, j].set_title(f'Pred t={t_phys:.1f}s{ic_tag}', fontsize=9)
            fig.colorbar(im1, ax=axes[1, j], fraction=0.046)
            if j == 0:
                axes[1, j].set_ylabel('y (binormal)')
            axes[1, j].set_xlabel('x (radial)')

        fig.suptitle(f'{name} ({tag})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'snapshots_{name}_{tag}.png'), dpi=150)
        plt.close(fig)

    print(f'  Plots saved to {save_dir}/')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, net, eval_data_path, mhd, n_rollout_steps=10,
             save_plots=False, step_tag='eval'):
    """Autoregressive rollout evaluation with optional visualization.

    Uses n_substeps = round(dt_data / rollout_dt) sub-steps per data step
    to match mhd_sim NFE.
    """
    device = config.device
    net.eval()

    trajectories, meta = load_mhd5_trajectories(
        eval_data_path, time_start=config.time_start, time_end=config.time_end,
        dt_data=config.dt_data)

    B, T, C, Nx, Ny = trajectories.shape
    n_steps = min(n_rollout_steps, T - 1)
    n_substeps = max(1, round(config.dt_data / config.rollout_dt))

    x_0 = trajectories[:, 0].permute(0, 2, 3, 1).to(device)  # (B, Nx, Ny, 5)
    gt = trajectories.to(device)

    pred_list = [trajectories[:, 0:1].to(device)]
    current = x_0
    with torch.no_grad():
        for _data_step in range(n_steps):
            for _sub in range(n_substeps):
                out = net(current, inference=True)
                current = out[..., -1]
            pred_cf = current.permute(0, 3, 1, 2).unsqueeze(1)
            pred_list.append(pred_cf)

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
        run_tag = getattr(config, 'run_tag', config.exp_name)
        vis_dir = f'results/{config.exp_name}/vis_{run_tag}'
        save_eval_plots(pred_traj, gt, rel_l2_total, rel_l2_per_field,
                        vis_dir, tag=step_tag,
                        time_start=config.time_start,
                        rollout_dt=config.rollout_dt,
                        n_substeps=n_substeps)

    net.train()
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config, net):
    """Main training loop."""
    device = config.device
    run_tag = getattr(config, 'run_tag', config.exp_name)

    print(f'Loading training data from {config.data_path}...')
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

    print(f'Building MHD instance on {device}...')
    mhd = build_mhd_instance(device=device, Nx=Nx, Ny=Ny)
    rhs_fn = make_mhd5_rhs_fn(mhd, model_dtype=torch.float32)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr,
        total_steps=config.num_iterations + 1)

    train_dt = config.rollout_dt / config.output_dim
    n_substeps_eval = max(1, round(config.dt_data / config.rollout_dt))
    print(f'  dt hierarchy: rollout_dt={config.rollout_dt}, '
          f'train_dt={train_dt:.4f} (output_dim={config.output_dim}), '
          f'dt_data={config.dt_data}, n_substeps_eval={n_substeps_eval}')

    best_loss = float('inf')
    global_step = 0

    print(f'\n[{run_tag}] Training: {config.num_iterations} iters, '
          f'bs={config.batch_size}, lr={config.lr}')
    print('-' * 60)
    sys.stdout.flush()

    t_start = time.time()

    for epoch in range(config.max_epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        for (x_batch,) in loader:
            if global_step > config.num_iterations:
                break

            x_0 = x_batch.to(device)
            noise_scale = getattr(config, 'input_noise_scale', 0.0)
            if noise_scale > 0:
                x_0 = x_0 + noise_scale * torch.randn_like(x_0)
            pred_traj = net(x_0)

            loss = compute_physics_loss(
                pred_traj, x_0, rhs_fn,
                rollout_dt=config.rollout_dt,
                output_dim=config.output_dim,
                time_integrator=config.time_integrator)

            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr_now = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t_start
                it_per_s = global_step / max(elapsed, 1e-6)
                eta = (config.num_iterations - global_step) / max(it_per_s, 1e-6)
                print(f'[{run_tag[:8]}] step {global_step:6d}/{config.num_iterations} '
                      f'| loss {loss.item():.6f} | avg {avg_loss:.6f} '
                      f'| lr {lr_now:.2e} | {it_per_s:.1f} it/s | ETA {eta/60:.0f}m')
                sys.stdout.flush()

            if global_step % config.eval_every == 0:
                do_plots = (global_step % (config.eval_every * 5) == 0)
                eval_results = evaluate(
                    config, net, config.data_path, mhd,
                    n_rollout_steps=config.eval_rollout_steps,
                    save_plots=do_plots,
                    step_tag=f'step_{global_step}')

                current_loss = eval_results['mean_rel_l2']
                if current_loss < best_loss:
                    best_loss = current_loss
                    save_path = f'model/{config.exp_name}/best-{run_tag}.pt'
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
                }, f'model/{config.exp_name}/latest-{run_tag}.pt')

                net.train()
                sys.stdout.flush()

        if global_step > config.num_iterations:
            break

    # Final evaluation with best model
    best_path = f'model/{config.exp_name}/best-{run_tag}.pt'
    print(f'\n--- Final Evaluation [{run_tag}] ---')
    if os.path.exists(best_path):
        net.load_state_dict(
            torch.load(best_path, map_location=device,
                       weights_only=False)['model_state_dict'])
    else:
        print(f'  WARNING: best checkpoint not found at {best_path}, using latest weights')

    final_results = evaluate(
        config, net, config.data_path, mhd,
        n_rollout_steps=config.eval_rollout_steps,
        save_plots=True, step_tag=f'final-{run_tag}')

    results_path = f'results/{config.exp_name}/eval-{run_tag}.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f'Results saved to {results_path}')

    total_time = time.time() - t_start
    print(f'Total training time: {total_time/3600:.1f}h ({total_time/60:.0f}m)')
    sys.stdout.flush()
