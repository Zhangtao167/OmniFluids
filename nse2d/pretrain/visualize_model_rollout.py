"""Visualize model rollout and evaluate on test set.

Load a trained model, evaluate on simulation test data, compute metrics,
and generate visualizations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/zhangtao/project2026/OmniFluids/nse2d/pretrain')
from model import OmniFluids2D
from tools import MHD5FieldGRF, load_mhd5_trajectories, compute_metrics_and_visualize

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']

# Default paths
DEFAULT_CKPT = '/zhangtao/project2026/OmniFluids/nse2d/pretrain/results/mhd5_staged_v1/595fa318-02_25_02_50_50-K4-mx128-w80-L12-od10/model/best-595fa318-02_25_02_50_50-K4-mx128-w80-L12-od10.pt'
DEFAULT_DATA_PATH = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt'
DEFAULT_SAVE_DIR = '/zhangtao/project2026/OmniFluids/nse2d/pretrain/vis_rollout'


def load_model(ckpt_path, device='cuda:0'):
    """Load trained model from checkpoint."""
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', {})
    
    # Print checkpoint info
    print(f'  Step: {ckpt.get("step", "?")}')
    print(f'  Best loss: {ckpt.get("best_loss", "?")}')
    
    # Reconstruct model
    net = OmniFluids2D(
        Nx=saved_cfg.get('Nx', 512),
        Ny=saved_cfg.get('Ny', 256),
        K=saved_cfg.get('K', 4),
        T=saved_cfg.get('temperature', 10.0),
        modes_x=saved_cfg.get('modes_x', 128),
        modes_y=saved_cfg.get('modes_y', 128),
        width=saved_cfg.get('width', 80),
        output_dim=saved_cfg.get('output_dim', 10),
        n_fields=5,
        n_params=saved_cfg.get('n_params', 8),
        n_layers=saved_cfg.get('n_layers', 12),
        factor=saved_cfg.get('factor', 4),
        n_ff_layers=saved_cfg.get('n_ff_layers', 2),
        layer_norm=saved_cfg.get('layer_norm', True))
    
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device)
    net.eval()
    
    print(f'  Model loaded: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params')
    return net, saved_cfg


def build_grf_generator(data_path, device='cuda:0', Nx=512, Ny=256):
    """Build GRF generator with stats from data."""
    print(f'Building GRF generator from: {data_path}')
    grf = MHD5FieldGRF.from_data_stats(
        data_path, Nx=Nx, Ny=Ny,
        device=device,
        time_start=250.0, time_end=300.0, dt_data=1.0,
        use_radial_mask=True)
    return grf


def rollout_model(net, x_0, n_steps, device='cuda:0'):
    """Rollout model for n_steps, collecting all intermediate states.
    
    Args:
        net: OmniFluids2D model
        x_0: (B, Nx, Ny, 5) initial state
        n_steps: Number of inference steps (NFE)
    
    Returns:
        trajectory: list of (B, Nx, Ny, 5) states, length = n_steps + 1 (including x_0)
    """
    trajectory = [x_0.cpu()]
    current = x_0.to(device)
    
    with torch.no_grad():
        for step in range(n_steps):
            # inference=True returns (B, Nx, Ny, 5, 1)
            out = net(current, inference=True)
            current = out[..., -1]  # (B, Nx, Ny, 5)
            trajectory.append(current.cpu())
            if (step + 1) % 10 == 0 or step == n_steps - 1:
                print(f'  Rollout step {step + 1}/{n_steps}')
    
    return trajectory


def evaluate_on_testset(net, data_path, n_steps, device='cuda:0',
                        time_start=250.0, time_end=260.0, dt_data=1.0,
                        model_dt=0.1, save_dir='./vis_rollout',
                        sample_idx=0):
    """Evaluate model on ALL test trajectories and compute metrics.
    
    For each trajectory, takes the first frame as initial condition,
    runs model rollout, aligns with GT at dt_data intervals, then
    computes metrics averaged over all trajectories.
    
    Args:
        net: OmniFluids2D model
        data_path: Path to simulation dataset
        n_steps: Number of rollout steps (NFE)
        device: Device to run on
        time_start: Start time of evaluation window (seconds)
        time_end: End time of evaluation window (seconds)
        dt_data: Time step in simulation data (default 1.0s)
        model_dt: Model's time step per NFE (default 0.1s)
        save_dir: Directory to save results
        sample_idx: Which sample to visualize (default: 0)
    
    Returns:
        metrics: dict with L2 errors
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load ALL trajectories
    print(f'\n=== Loading Test Data ===')
    trajectories, meta = load_mhd5_trajectories(
        data_path, time_start=time_start, time_end=time_end,
        dt_data=dt_data, single_trajectory=False)
    # trajectories: (B, T, 5, Nx, Ny) -> channel-last (B, T, Nx, Ny, 5)
    gt_traj = trajectories.permute(0, 1, 3, 4, 2)  # (B, T, Nx, Ny, 5)
    B, T_gt, Nx, Ny, n_fields = gt_traj.shape
    print(f'  GT trajectory shape: {gt_traj.shape} (B={B}, T={T_gt} frames)')
    
    # Alignment parameters
    steps_per_gt_frame = int(round(dt_data / model_dt))
    n_gt_steps = min(T_gt, n_steps // steps_per_gt_frame + 1)
    
    print(f'\n=== Model Rollout (B={B} trajectories, batched) ===')
    print(f'  {n_steps} NFE (model dt={model_dt}s, total={n_steps * model_dt:.1f}s)')
    print(f'  Model steps per GT frame: {steps_per_gt_frame}')
    print(f'  Aligned time steps: {n_gt_steps} (0 to {n_gt_steps-1})')
    
    # Batch all B initial conditions and rollout in parallel
    x_0_batch = gt_traj[:, 0].to(device)  # (B, Nx, Ny, 5)
    current = x_0_batch
    
    # Collect aligned frames: list of (B, Nx, Ny, 5)
    aligned_frames = [x_0_batch.cpu()]
    frame_counter = 1
    
    with torch.no_grad():
        for s in range(1, n_steps + 1):
            out = net(current, inference=True)  # (B, Nx, Ny, 5, 1)
            current = out[..., -1]               # (B, Nx, Ny, 5)
            if s % steps_per_gt_frame == 0 and frame_counter < n_gt_steps:
                aligned_frames.append(current.cpu())
                frame_counter += 1
            if s % 10 == 0 or s == n_steps:
                print(f'    Step {s}/{n_steps} (collected {frame_counter}/{n_gt_steps} aligned frames)')
    
    # Stack: (n_gt_steps, B, Nx, Ny, 5) -> transpose to (B, n_gt_steps, Nx, Ny, 5)
    pred_aligned = torch.stack(aligned_frames, dim=0).permute(1, 0, 2, 3, 4).numpy()
    gt_aligned = gt_traj[:, :n_gt_steps].numpy()
    
    print(f'\n=== Aligning Trajectories ===')
    print(f'  GT aligned shape: {gt_aligned.shape}')
    print(f'  Pred aligned shape: {pred_aligned.shape}')
    
    # Compute metrics and visualize
    print(f'\n=== Computing Metrics and Visualizing ===')
    metrics = compute_metrics_and_visualize(
        gt_traj=gt_aligned,
        pred_traj=pred_aligned,
        metric_step_list=[1, 3, 5, 10] if n_gt_steps > 10 else list(range(1, n_gt_steps)),
        plot_step_list=[0, 1, 3, 5, 10] if n_gt_steps > 10 else list(range(n_gt_steps)),
        visualize=True,
        save_dir=save_dir,
        time_start=time_start,
        dt_data=dt_data,
        sample_idx=sample_idx
    )
    
    return metrics


def visualize_trajectory(trajectory, rollout_dt=0.1, save_dir='./vis_rollout', 
                         sample_idx=0, n_vis_steps=None):
    """Visualize trajectory evolution (for GRF rollout mode).
    
    Args:
        trajectory: list of (B, Nx, Ny, 5) states
        rollout_dt: Time step per NFE
        save_dir: Directory to save plots
        sample_idx: Which sample in batch to visualize
        n_vis_steps: Number of steps to visualize (None = all, or select evenly spaced)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_total = len(trajectory)
    
    # Select steps to visualize
    if n_vis_steps is None or n_vis_steps >= n_total:
        vis_indices = list(range(n_total))
    else:
        vis_indices = np.linspace(0, n_total - 1, n_vis_steps, dtype=int).tolist()
    
    n_cols = len(vis_indices)
    
    # === Plot 1: All fields evolution ===
    fig, axes = plt.subplots(5, n_cols, figsize=(4 * n_cols, 15))
    if n_cols == 1:
        axes = axes.reshape(5, 1)
    
    for j, step_idx in enumerate(vis_indices):
        state = trajectory[step_idx][sample_idx].numpy()  # (Nx, Ny, 5)
        t_phys = step_idx * rollout_dt
        
        for c, name in enumerate(FIELD_NAMES):
            field = state[:, :, c].T  # (Ny, Nx) for imshow
            vmax = np.abs(field).max()
            vmin = -vmax if name != 'n' else field.min()
            
            im = axes[c, j].imshow(field, aspect='auto', origin='lower',
                                    cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[c, j].set_title(f'{name} t={t_phys:.1f}s', fontsize=10)
            plt.colorbar(im, ax=axes[c, j], fraction=0.046)
            
            if j == 0:
                axes[c, j].set_ylabel(f'{name}\nx (radial)')
            if c == 4:
                axes[c, j].set_xlabel('y (binormal)')
    
    fig.suptitle(f'Model Rollout: GRF -> {(n_total-1)*rollout_dt:.1f}s ({n_total-1} NFE)', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'grf_trajectory_fields.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')
    
    # === Plot 2: Per-field energy evolution ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    t_axis = np.arange(n_total) * rollout_dt
    
    for c, name in enumerate(FIELD_NAMES):
        energies = []
        for state in trajectory:
            field = state[sample_idx, :, :, c].numpy()
            energy = (field ** 2).mean()
            energies.append(energy)
        
        axes[c].plot(t_axis, energies, 'b-o', markersize=2)
        axes[c].set_title(f'{name} mean energy <f²>')
        axes[c].set_xlabel('Time (s)')
        axes[c].set_ylabel('Energy')
        axes[c].grid(True, alpha=0.3)
    
    axes[5].axis('off')
    fig.suptitle('Per-field Mean Energy Evolution', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'grf_energy_evolution.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')
    
    # === Plot 3: Initial vs Final comparison ===
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    
    for c, name in enumerate(FIELD_NAMES):
        # Initial
        init_field = trajectory[0][sample_idx, :, :, c].numpy().T
        vmax = np.abs(init_field).max()
        vmin = -vmax
        
        im0 = axes[0, c].imshow(init_field, aspect='auto', origin='lower',
                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, c].set_title(f'{name} (GRF Input)')
        plt.colorbar(im0, ax=axes[0, c], fraction=0.046)
        
        # Final
        final_field = trajectory[-1][sample_idx, :, :, c].numpy().T
        im1 = axes[1, c].imshow(final_field, aspect='auto', origin='lower',
                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, c].set_title(f'{name} (t={(n_total-1)*rollout_dt:.1f}s)')
        plt.colorbar(im1, ax=axes[1, c], fraction=0.046)
    
    axes[0, 0].set_ylabel('Initial (GRF)')
    axes[1, 0].set_ylabel('Final')
    
    fig.suptitle(f'GRF Input vs Model Output after {n_total-1} NFE', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'grf_initial_vs_final.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')
    
    print(f'\nAll GRF visualizations saved to: {save_dir}/')


def main(args):
    device = args.device
    
    # Load model
    net, saved_cfg = load_model(args.checkpoint, device=device)
    
    if args.mode == 'testset':
        # =====================================================
        # Mode: Evaluate on simulation test set
        # =====================================================
        print('\n' + '=' * 60)
        print('MODE: Test Set Evaluation')
        print('=' * 60)
        
        # Get model dt from config (default 0.1)
        model_dt = saved_cfg.get('rollout_dt', 0.1)
        print(f'  Model rollout_dt from config: {model_dt}s')
        
        metrics = evaluate_on_testset(
            net=net,
            data_path=args.data_path,
            n_steps=args.n_steps,
            device=device,
            time_start=args.time_start,
            time_end=args.time_end,
            dt_data=args.dt_data,
            model_dt=model_dt,
            save_dir=args.save_dir,
            sample_idx=args.sample_idx
        )
        
        print('\n' + '=' * 60)
        print('Evaluation Complete!')
        print('=' * 60)
        print(f'\nResults saved to: {args.save_dir}/')
        
    else:
        # =====================================================
        # Mode: GRF rollout visualization
        # =====================================================
        print('\n' + '=' * 60)
        print('MODE: GRF Rollout Visualization')
        print('=' * 60)
        
        # Build GRF generator
        Nx = saved_cfg.get('Nx', 512)
        Ny = saved_cfg.get('Ny', 256)
        grf = build_grf_generator(args.data_path, device=device, Nx=Nx, Ny=Ny)
        
        # Generate GRF initial condition
        print(f'\nGenerating GRF initial condition (batch_size={args.batch_size})...')
        torch.manual_seed(args.seed)
        x_0 = grf(args.batch_size)  # (B, Nx, Ny, 5)
        print(f'  GRF shape: {x_0.shape}')
        
        # Rollout
        print(f'\nRolling out model for {args.n_steps} steps...')
        rollout_dt = saved_cfg.get('rollout_dt', 0.1)
        trajectory = rollout_model(net, x_0, args.n_steps, device=device)
        print(f'  Trajectory: {len(trajectory)} states, total time = {(len(trajectory)-1)*rollout_dt:.1f}s')
        
        # Visualize
        print(f'\nGenerating visualizations...')
        visualize_trajectory(
            trajectory, 
            rollout_dt=rollout_dt,
            save_dir=args.save_dir,
            sample_idx=args.sample_idx,
            n_vis_steps=args.n_vis_steps)
    
    print('\nDone!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model and visualize rollout')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='testset', choices=['testset', 'grf'],
                        help='Evaluation mode: testset (compare with simulation) or grf (random init)')
    
    # Paths
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to simulation dataset')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='Directory to save results')
    
    # Rollout settings
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of model inference steps (NFE). For testset mode, use 100 for 10s.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    
    # Test set mode settings
    parser.add_argument('--time_start', type=float, default=250.0,
                        help='Start time for test set evaluation (seconds)')
    parser.add_argument('--time_end', type=float, default=260.0,
                        help='End time for test set evaluation (seconds)')
    parser.add_argument('--dt_data', type=float, default=1.0,
                        help='Time step in simulation data (seconds)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which sample to visualize')
    
    # GRF mode settings
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of GRF samples to generate (GRF mode)')
    parser.add_argument('--n_vis_steps', type=int, default=6,
                        help='Number of time steps to show in GRF trajectory plot')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for GRF generation')
    
    args = parser.parse_args()
    main(args)
