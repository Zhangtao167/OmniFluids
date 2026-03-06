"""Visualize model rollout from GRF initial conditions.

Load a trained model and visualize how it evolves GRF-generated initial conditions
over multiple time steps.
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
from tools import MHD5FieldGRF

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']

# Default checkpoint path
DEFAULT_CKPT = '/zhangtao/project2026/OmniFluids/nse2d/pretrain/results/exp11_grf_staged_multigpu/06fff25a-03_04_11_14_58-K4-mx128-w80-L12-od10/model/latest-06fff25a-03_04_11_14_58-K4-mx128-w80-L12-od10.pt'
DEFAULT_DATA_PATH = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt'


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


def visualize_trajectory(trajectory, rollout_dt=0.1, save_dir='./vis_rollout', 
                         sample_idx=0, n_vis_steps=None):
    """Visualize trajectory evolution.
    
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
    save_path = os.path.join(save_dir, 'trajectory_fields.png')
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
    save_path = os.path.join(save_dir, 'energy_evolution.png')
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
    save_path = os.path.join(save_dir, 'initial_vs_final.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')
    
    print(f'\nAll visualizations saved to: {save_dir}/')


def main(args):
    device = args.device
    
    # Load model
    net, saved_cfg = load_model(args.checkpoint, device=device)
    
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
    parser = argparse.ArgumentParser(description='Visualize model rollout from GRF')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to data for GRF stats')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of model inference steps (NFE)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of GRF samples to generate')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which sample in batch to visualize')
    parser.add_argument('--n_vis_steps', type=int, default=6,
                        help='Number of time steps to show in trajectory plot')
    parser.add_argument('--save_dir', type=str, default='./vis_rollout',
                        help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for GRF generation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    
    args = parser.parse_args()
    main(args)
