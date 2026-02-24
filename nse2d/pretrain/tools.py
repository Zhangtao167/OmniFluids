"""Utility functions for 5-field MHD OmniFluids training.

Provides:
- setup_seed: reproducibility
- param_count: model parameter counting
- load_mhd5_snapshots: load offline MHD data as channel-last snapshots
"""

import torch
import numpy as np
import random
import os
from functools import reduce
import operator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def param_count(net):
    """Count and print model parameters."""
    params = 0
    for p in net.parameters():
        params += reduce(operator.mul,
                         list(p.size() + (2,) if p.is_complex() else p.size()))
    print(f' params: {params / 1e6:.3f} M')
    return params


def load_mhd5_snapshots(data_path, time_start=250.0, time_end=300.0, dt_data=1.0):
    """Load 5-field MHD dataset and extract training snapshots.

    Expected format: dict with keys 'n', 'U', 'vpar', 'psi', 'Ti',
    each of shape (B, T, Nx, Ny) in float64.

    Args:
        data_path: path to .pt dataset file
        time_start: start time for training window (seconds)
        time_end: end time for training window (seconds)
        dt_data: time interval between consecutive snapshots

    Returns:
        snapshots: (N, Nx, Ny, 5) float32 tensor, channel-last
        metadata: dict with Nx, Ny, dt_data, n_samples, n_timesteps
    """
    data = torch.load(data_path, map_location='cpu', weights_only=False)

    fields = [data[name] for name in FIELD_NAMES]
    states = torch.stack(fields, dim=2)  # (B, T, 5, Nx, Ny)

    t_start_idx = int(round(time_start / dt_data))
    t_end_idx = int(round(time_end / dt_data))
    states = states[:, t_start_idx:t_end_idx + 1]

    B, T, C, Nx, Ny = states.shape
    print(f'Loaded data: {B} samples, {T} timesteps, {C} fields, '
          f'{Nx}x{Ny} grid, time [{time_start}, {time_end}]')

    # Flatten (B, T) -> (B*T,) and convert to channel-last float32
    snapshots = states.reshape(B * T, C, Nx, Ny).permute(0, 2, 3, 1).float()

    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, t_start=time_start, t_end=time_end)
    return snapshots, metadata


def load_mhd5_trajectories(data_path, time_start=250.0, time_end=300.0,
                           dt_data=1.0):
    """Load 5-field MHD dataset as trajectories for evaluation.

    Returns:
        trajectories: (B, T, 5, Nx, Ny) float32 tensor, channel-first
                      (compatible with mhd_sim eval tools)
        metadata: dict
    """
    data = torch.load(data_path, map_location='cpu', weights_only=False)

    fields = [data[name] for name in FIELD_NAMES]
    states = torch.stack(fields, dim=2)  # (B, T, 5, Nx, Ny)

    t_start_idx = int(round(time_start / dt_data))
    t_end_idx = int(round(time_end / dt_data))
    states = states[:, t_start_idx:t_end_idx + 1]

    B, T, C, Nx, Ny = states.shape
    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, t_start=time_start, t_end=time_end)
    return states.float(), metadata
