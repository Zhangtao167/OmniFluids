"""Utility functions for 5-field MHD OmniFluids training.

Provides:
- setup_seed: reproducibility
- param_count: model parameter counting
- load_mhd5_snapshots: load offline MHD data as channel-last snapshots
- MHD5FieldGRF: online random initial condition generator
"""

import math
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


# ---------------------------------------------------------------------------
# Online random initial condition generator for 5-field MHD
# ---------------------------------------------------------------------------

class MHD5FieldGRF:
    """Gaussian Random Field generator for 5-field MHD with mixed BCs.

    Generates random initial conditions on a rectangular grid (Nx, Ny):
      - x direction (Dirichlet): multiply by sin(pi*i/(Nx-1)) to enforce zero BC
      - y direction (periodic): standard FFT generation

    Each of the 5 fields (n, U, vpar, psi, Ti) is generated independently
    with per-field amplitude scaling.

    Args:
        Nx, Ny: grid dimensions
        alpha: spectral power decay exponent (higher = smoother)
        tau: inverse correlation length
        field_scales: per-field amplitude, shape (5,) or scalar
        device: target device
    """

    def __init__(self, Nx=512, Ny=256, alpha=2.5, tau=7.0,
                 field_scales=None, device='cpu'):
        self.Nx = Nx
        self.Ny = Ny
        self.device = device
        self.n_fields = 5

        if field_scales is None:
            field_scales = torch.tensor([1.0, 1.0, 0.5, 5.0, 0.5])
        self.field_scales = field_scales.to(device)

        Lx = 2 * math.pi
        Ly = 2 * math.pi
        cx = (4 * math.pi ** 2) / (Lx ** 2)
        cy = (4 * math.pi ** 2) / (Ly ** 2)

        kx = torch.arange(0, Nx, dtype=torch.float32, device=device)
        ky = torch.arange(0, Ny // 2 + 1, dtype=torch.float32, device=device)
        kx2 = (cx * kx ** 2).unsqueeze(1)                    # (Nx, 1)
        ky2 = (cy * ky ** 2).unsqueeze(0)                     # (1, Ny//2+1)

        sigma = 0.5 * tau ** (0.5 * (2 * alpha - 2.0))
        sqrt_eig = sigma * (kx2 + ky2 + tau ** 2) ** (-alpha / 2.0)
        sqrt_eig[0, 0] = 0.0
        self.sqrt_eig = sqrt_eig * Nx * Ny                   # (Nx, Ny//2+1)

        # Dirichlet window: sin(pi * i / (Nx-1)), zero at i=0 and i=Nx-1
        idx = torch.arange(Nx, dtype=torch.float32, device=device)
        self.dirichlet_window = torch.sin(math.pi * idx / (Nx - 1))  # (Nx,)

    def __call__(self, batch_size):
        """Generate random 5-field initial conditions.

        Returns:
            (batch_size, Nx, Ny, 5) float32 tensor, channel-last
        """
        B, Nx, Ny = batch_size, self.Nx, self.Ny
        fields = []
        for c in range(self.n_fields):
            xi = torch.randn(B, Nx, Ny // 2 + 1, 2,
                              dtype=torch.float32, device=self.device)
            xi[..., 0] *= self.sqrt_eig
            xi[..., 1] *= self.sqrt_eig
            u = torch.fft.irfft2(torch.view_as_complex(xi), s=(Nx, Ny))
            u = u * self.dirichlet_window.reshape(1, Nx, 1)
            u = u * self.field_scales[c]
            fields.append(u)
        return torch.stack(fields, dim=-1)  # (B, Nx, Ny, 5)

    @staticmethod
    def from_data_stats(data_path, Nx=512, Ny=256, alpha=2.5, tau=7.0,
                        device='cpu', time_start=250.0, time_end=300.0,
                        dt_data=1.0):
        """Create GRF with field_scales derived from training data std."""
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        scales = []
        for name in FIELD_NAMES:
            t0 = int(round(time_start / dt_data))
            t1 = int(round(time_end / dt_data))
            field = data[name][:, t0:t1 + 1]
            scales.append(field.std().item())
        field_scales = torch.tensor(scales, dtype=torch.float32)
        print(f'GRF field_scales from data: {dict(zip(FIELD_NAMES, scales))}')
        return MHD5FieldGRF(Nx, Ny, alpha, tau, field_scales, device)
