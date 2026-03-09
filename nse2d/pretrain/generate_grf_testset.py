#!/usr/bin/env python3
"""
Generate GRF-initialized test set for 5-field MHD.

Uses GRF to generate initial conditions, then evolves them using the
physical solver (FiveFieldMHD with RK4 integration).

IMPORTANT: Use --data-path to ensure GRF field_scales match training data!

Usage:
    python generate_grf_testset.py --data-path /path/to/training_data.pt [OPTIONS]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/zhangtao/project2026/mhd_sim')
sys.path.insert(0, '/zhangtao/project2026/OmniFluids/nse2d/pretrain')

import torch
import numpy as np
from tqdm import tqdm

from numerical.equations.five_field_mhd import FiveFieldMHD, FiveFieldMHDConfig
from numerical.scripts.run_5field_mhd import rk4_step
from tools import MHD5FieldGRF

# Default training data path (for deriving GRF field_scales)
DEFAULT_DATA_PATH = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt'


def generate_grf_testset(
    n_samples=10, n_steps=50, dt_data=1.0,
    Nx=512, Ny=256, device='cuda:0',
    output_dir='./data/grf_testset', base_seed=1000,
    use_radial_mask=True,
    data_path=None,
    time_start_for_stats=250.0,
    time_end_for_stats=300.0,
):
    """Generate GRF-initialized test set.
    
    Args:
        n_samples: Number of trajectories to generate
        n_steps: Number of evolution steps (frames = n_steps + 1)
        dt_data: Time interval between saved frames (1.0s to match training)
        Nx, Ny: Grid dimensions
        device: CUDA device
        output_dir: Output directory
        base_seed: Base random seed
        use_radial_mask: Whether to apply radial mask to GRF
        data_path: Path to training data for deriving field_scales (REQUIRED for consistency)
        time_start_for_stats: Start time for computing field_scales from data
        time_end_for_stats: End time for computing field_scales from data
    
    Output format matches mhd_sim batch data:
      - Each field: (B, T, Nx, Ny)
      - t_list: physical time array (virtual time starting at 250.0)
      - metadata: comprehensive info including GRF parameters
    """
    print("=" * 60)
    print("GRF Test Set Generation for 5-Field MHD")
    print("=" * 60)
    print(f"  n_samples: {n_samples}")
    print(f"  n_steps: {n_steps} (total {n_steps + 1} frames)")
    print(f"  dt_data: {dt_data}s")
    print(f"  Total physical time: {n_steps * dt_data}s")
    print(f"  Grid: {Nx} x {Ny}")
    print(f"  Device: {device}")
    print(f"  use_radial_mask: {use_radial_mask}")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize GRF generator with field_scales from training data
    print("\nInitializing GRF generator...")
    if data_path and Path(data_path).exists():
        print(f"  Deriving field_scales from: {data_path}")
        print(f"  Time range for stats: [{time_start_for_stats}, {time_end_for_stats}]")
        grf = MHD5FieldGRF.from_data_stats(
            data_path, Nx=Nx, Ny=Ny,
            device=device,
            time_start=time_start_for_stats,
            time_end=time_end_for_stats,
            dt_data=dt_data,
            use_radial_mask=use_radial_mask)
        grf_scales_source = 'from_data'
        grf_scales = grf.field_scales.tolist()
    else:
        if data_path:
            print(f"  [WARNING] data_path not found: {data_path}")
        print("  [WARNING] Using DEFAULT field_scales (may not match training!)")
        grf = MHD5FieldGRF(Nx=Nx, Ny=Ny, device=device, use_radial_mask=use_radial_mask)
        grf_scales_source = 'default'
        grf_scales = grf.field_scales.tolist()

    # Initialize physical solver
    print("\nInitializing FiveFieldMHD solver...")
    mhd_cfg = FiveFieldMHDConfig()
    mhd_cfg.Nx = Nx
    mhd_cfg.Ny = Ny
    mhd_cfg.device = device
    mhd_cfg.precision = 'fp64'
    dt_sim = mhd_cfg.dt  # Default 0.002s (same as training data generation)
    steps_per_frame = int(round(dt_data / dt_sim))
    print(f"  dt_sim: {dt_sim}s, steps_per_frame: {steps_per_frame}")
    print(f"  Total RK4 steps per trajectory: {steps_per_frame * n_steps}")

    mhd = FiveFieldMHD(mhd_cfg)

    # Pre-allocate storage
    field_names = ['n', 'U', 'vpar', 'psi', 'Ti']
    T = n_steps + 1
    data = {name: torch.zeros(n_samples, T, Nx, Ny, dtype=torch.float32) for name in field_names}
    
    # Track valid frames per trajectory (for NaN detection)
    valid_frames = torch.zeros(n_samples, dtype=torch.int32)

    # Generate trajectories
    print(f"\nGenerating {n_samples} trajectories...")
    
    for b in tqdm(range(n_samples), desc="Trajectories"):
        seed = base_seed + b
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Generate GRF initial condition (B, Nx, Ny, 5)
        x_0 = grf(batch_size=1)
        
        # Convert to state tuple: 5 tensors each (1, Nx, Ny)
        state = tuple(x_0[0, :, :, i].unsqueeze(0).to(torch.float64) for i in range(5))

        # Save t=0
        for i, name in enumerate(field_names):
            data[name][b, 0] = state[i][0].float().cpu()
        
        last_valid_frame = 0

        # Evolve with inner progress bar
        with torch.no_grad():
            pbar_inner = tqdm(range(1, T), desc=f"  Traj {b} frames", leave=False)
            for t_idx in pbar_inner:
                nan_detected = False
                for step in range(steps_per_frame):
                    if torch.isnan(state[0]).any():
                        print(f"\n[WARN] NaN at traj {b}, frame {t_idx}, substep {step}")
                        nan_detected = True
                        break
                    state = rk4_step(mhd.compute_rhs, state, dt_sim)
                    state = mhd.apply_boundary_conditions(state)
                
                if nan_detected:
                    break
                
                for i, name in enumerate(field_names):
                    data[name][b, t_idx] = state[i][0].float().cpu()
                last_valid_frame = t_idx
            pbar_inner.close()
        
        valid_frames[b] = last_valid_frame + 1  # +1 because frame 0 is always valid

        if (b + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Generate t_list (physical time array, compatible with mhd_sim format)
    # Use time_start=250.0 to match mhd_sim convention (allows using default eval params)
    virtual_time_start = 250.0
    t_list = virtual_time_start + torch.arange(T, dtype=torch.float64) * dt_data

    # Build descriptive filename with GRF config info
    mask_tag = 'radial' if use_radial_mask else 'full'
    scales_tag = 'fromdata' if grf_scales_source == 'from_data' else 'default'
    filename = f'grf_testset_B{n_samples}_T{n_steps}_dt{dt_data}_{scales_tag}_{mask_tag}_seed{base_seed}.pt'
    output_file = output_path / filename
    
    # Also create a symlink 'grf_testset.pt' for backward compatibility
    symlink_path = output_path / 'grf_testset.pt'

    # Build metadata
    metadata = {
        'n_samples': n_samples,
        'n_steps': n_steps,
        'dt_data': dt_data,
        'dt_sim': dt_sim,
        'steps_per_frame': steps_per_frame,
        'Nx': Nx,
        'Ny': Ny,
        'base_seed': base_seed,
        'time_start': virtual_time_start,
        'time_end': virtual_time_start + n_steps * dt_data,
        'use_radial_mask': use_radial_mask,
        'field_names': field_names,
        'is_grf_data': True,
        'grf_scales_source': grf_scales_source,
        'grf_field_scales': grf_scales,
        'grf_alphas': MHD5FieldGRF.DEFAULT_ALPHAS,
        'grf_taus': MHD5FieldGRF.DEFAULT_TAUS,
        'data_path_for_scales': data_path if grf_scales_source == 'from_data' else None,
    }
    
    data['t_list'] = t_list
    data['valid_frames'] = valid_frames
    data['metadata'] = metadata
    torch.save(data, output_file)
    
    # Create/update symlink for backward compatibility
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(filename)
    
    # Also save a human-readable config file
    config_file = output_path / f'{filename.replace(".pt", "_config.txt")}'
    with open(config_file, 'w') as f:
        f.write("GRF Test Set Configuration\n")
        f.write("=" * 60 + "\n")
        for k, v in metadata.items():
            f.write(f"  {k}: {v}\n")
        f.write("=" * 60 + "\n")
    
    # Summary
    n_complete = (valid_frames == T).sum().item()
    print(f"\nSaved to: {output_file}")
    print(f"  Symlink: {symlink_path} -> {filename}")
    print(f"  Config: {config_file}")
    print(f"  Complete trajectories: {n_complete}/{n_samples}")
    if n_complete < n_samples:
        print(f"  [WARNING] {n_samples - n_complete} trajectories have NaN (incomplete)")
    print(f"  GRF field_scales ({grf_scales_source}): {dict(zip(field_names, grf_scales))}")
    for name in field_names:
        t = data[name]
        print(f"  {name}: {t.shape}, range=[{t.min():.3f}, {t.max():.3f}]")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate GRF test set for 5-field MHD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate with field_scales derived from training data (RECOMMENDED)
  python generate_grf_testset.py --data-path /path/to/training_data.pt
  
  # Generate 50 steps on GPU 4
  python generate_grf_testset.py --data-path /path/to/data.pt --n-steps 50 --device cuda:4
''')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of trajectories (default: 10)')
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Number of evolution steps (default: 50, gives 51 frames)')
    parser.add_argument('--dt-data', type=float, default=1.0,
                        help='Frame interval in seconds (default: 1.0, same as training data)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str,
                        default='/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset')
    parser.add_argument('--base-seed', type=int, default=1000)
    parser.add_argument('--no-radial-mask', action='store_true',
                        help='Disable radial mask (not recommended)')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to training data for deriving GRF field_scales '
                             '(REQUIRED for consistency with training)')
    parser.add_argument('--time-start-for-stats', type=float, default=250.0,
                        help='Start time for computing field stats from data (default: 250.0)')
    parser.add_argument('--time-end-for-stats', type=float, default=300.0,
                        help='End time for computing field stats from data (default: 300.0)')
    args = parser.parse_args()
    
    generate_grf_testset(
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        dt_data=args.dt_data,
        device=args.device,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        use_radial_mask=not args.no_radial_mask,
        data_path=args.data_path,
        time_start_for_stats=args.time_start_for_stats,
        time_end_for_stats=args.time_end_for_stats,
    )


if __name__ == '__main__':
    main()
