#!/usr/bin/env python3
"""
Visualize GRF test set trajectories.

Each page shows 5 rows (fields) x 5 columns (time steps).
Output: PDF files saved to the dataset directory.

Usage:
    python visualize_grf_trajectory.py [--data-path PATH] [--traj-idx IDX] [--n-pages N]
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']
FIELD_LABELS = {
    'n': 'Density (n)',
    'U': 'Vorticity (U)',
    'vpar': 'Parallel velocity (v||)',
    'psi': 'Magnetic flux (ψ)',
    'Ti': 'Ion temperature (Ti)'
}

DEFAULT_DATA_PATH = '/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt'


def visualize_trajectory_page(data, traj_idx, start_frame, t_list, n_cols=5, figsize=(20, 16)):
    """Create a single page of trajectory visualization."""
    T_total = data['n'].shape[1]
    end_frame = min(start_frame + n_cols, T_total)
    actual_cols = end_frame - start_frame
    
    fig, axes = plt.subplots(5, actual_cols, figsize=figsize)
    if actual_cols == 1:
        axes = axes.reshape(5, 1)
    
    for col_idx, frame_idx in enumerate(range(start_frame, end_frame)):
        t_val = t_list[frame_idx]
        
        for row_idx, field_name in enumerate(FIELD_NAMES):
            ax = axes[row_idx, col_idx]
            field = data[field_name][traj_idx, frame_idx].numpy()
            
            vmax = np.abs(field).max()
            if field_name == 'n':
                vmin, vmax = field.min(), field.max()
                cmap = 'viridis'
            else:
                vmin = -vmax
                cmap = 'RdBu_r'
            
            im = ax.imshow(field.T, origin='lower', aspect='auto',
                          cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            if row_idx == 0:
                ax.set_title(f't = {t_val:.1f}s\n(frame {frame_idx})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(FIELD_LABELS[field_name], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(f'Trajectory {traj_idx}: Frames {start_frame} - {end_frame-1}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_field_stats(data, traj_idx, t_list):
    """Create field statistics visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, field_name in enumerate(FIELD_NAMES):
        ax = axes[i]
        field_traj = data[field_name][traj_idx]
        
        means = field_traj.mean(dim=(1, 2)).numpy()
        stds = field_traj.std(dim=(1, 2)).numpy()
        maxs = field_traj.amax(dim=(1, 2)).numpy()
        mins = field_traj.amin(dim=(1, 2)).numpy()
        
        ax.fill_between(t_list, mins, maxs, alpha=0.3, label='min-max')
        ax.fill_between(t_list, means - stds, means + stds, alpha=0.5, label='mean±std')
        ax.plot(t_list, means, 'k-', linewidth=1.5, label='mean')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(field_name)
        ax.set_title(FIELD_LABELS[field_name])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[5].axis('off')
    fig.suptitle(f'Field Statistics - Trajectory {traj_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize GRF test set trajectories')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to GRF test set (.pt file)')
    parser.add_argument('--traj-idx', type=int, default=0,
                        help='Trajectory index to visualize (default: 0)')
    parser.add_argument('--n-pages', type=int, default=2,
                        help='Number of pages to visualize (5 frames per page)')
    parser.add_argument('--all-trajs', action='store_true',
                        help='Visualize all trajectories (overrides --traj-idx)')
    args = parser.parse_args()

    print('=' * 60)
    print('GRF Test Set Trajectory Visualization')
    print('=' * 60)

    # Load dataset
    print(f'\nLoading: {args.data_path}')
    data = torch.load(args.data_path, map_location='cpu', weights_only=False)

    # Print metadata
    if 'metadata' in data:
        meta = data['metadata']
        print('\nMetadata:')
        for k, v in meta.items():
            print(f'  {k}: {v}')

    # Get data shape
    n_field = data['n']
    B, T, Nx, Ny = n_field.shape
    print(f'\nData shape: B={B} trajectories, T={T} frames, grid={Nx}x{Ny}')

    # Get time list
    if 't_list' in data:
        t_list = data['t_list'].numpy()
        print(f'Time range: [{t_list[0]:.1f}, {t_list[-1]:.1f}] s')
    else:
        t_list = np.arange(T).astype(float)
        print('No t_list found, using frame indices')

    # Determine output directory (same as data file location)
    data_dir = Path(args.data_path).parent
    
    # Determine which trajectories to visualize
    if args.all_trajs:
        traj_indices = list(range(B))
    else:
        traj_indices = [args.traj_idx]
    
    # Visualization parameters
    n_cols_per_page = 5
    max_pages = (T + n_cols_per_page - 1) // n_cols_per_page
    actual_pages = min(args.n_pages, max_pages)

    for traj_idx in traj_indices:
        if traj_idx >= B:
            print(f'Warning: traj_idx={traj_idx} >= B={B}, skipping')
            continue
            
        # Output PDF path
        pdf_path = data_dir / f'grf_trajectory_vis_traj{traj_idx}.pdf'
        print(f'\n{"="*60}')
        print(f'Trajectory {traj_idx}: Saving to {pdf_path}')
        print(f'{"="*60}')

        with PdfPages(pdf_path) as pdf:
            # Generate trajectory pages
            for page_idx in range(actual_pages):
                start_frame = page_idx * n_cols_per_page
                print(f'  Page {page_idx + 1}/{actual_pages}: frames {start_frame}-{start_frame + n_cols_per_page - 1}')
                
                fig = visualize_trajectory_page(data, traj_idx, start_frame, t_list, n_cols=n_cols_per_page)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add field statistics page
            print(f'  Field statistics page')
            fig_stats = visualize_field_stats(data, traj_idx, t_list)
            pdf.savefig(fig_stats, bbox_inches='tight')
            plt.close(fig_stats)

        print(f'  Saved: {pdf_path}')

    print('\n' + '=' * 60)
    print('Visualization complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
