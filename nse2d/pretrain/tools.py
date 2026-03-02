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
    print(f'  Loading {data_path} ...', end=' ', flush=True)
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    print('done.', flush=True)

    t_start_idx = int(round(time_start / dt_data))
    t_end_idx = int(round(time_end / dt_data))

    # Slice time before stacking to reduce peak memory
    fields = [data[name][:, t_start_idx:t_end_idx + 1] for name in FIELD_NAMES]
    del data
    states = torch.stack(fields, dim=2).float()  # (B, T, 5, Nx, Ny)
    del fields

    B, T, C, Nx, Ny = states.shape
    print(f'  Trajectories: {B} samples, {T} steps, {C} fields, {Nx}x{Ny}')
    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, t_start=time_start, t_end=time_end)
    return states, metadata


# ---------------------------------------------------------------------------
# Online random initial condition generator for 5-field MHD
# ---------------------------------------------------------------------------

class MHD5FieldGRF:
    """Gaussian Random Field generator for 5-field MHD with mixed BCs.

    Generates random initial conditions on a rectangular grid (Nx, Ny):
      - x direction (Dirichlet): multiply by sin(pi*i/(Nx-1)) to enforce zero BC
      - y direction (periodic): standard FFT generation

    Each of the 5 fields (n, U, vpar, psi, Ti) is generated independently
    with per-field amplitude scaling and optional per-field spectral parameters.

    Args:
        Nx, Ny: grid dimensions
        alpha: spectral power decay exponent (scalar or list of 5).
               Higher = smoother fields. Physically, velocity/vorticity fields
               are typically smoother than density fluctuations.
        tau: inverse correlation length (scalar or list of 5).
             Larger = longer correlation / larger structures.
        field_scales: per-field amplitude, shape (5,) or scalar
        device: target device
    """

    # Per-field defaults tuned for 5-field MHD:
    #   n, vpar, Ti: standard turbulence alpha/tau
    #   U (vorticity): slightly smoother (lower alpha = rougher, but U is
    #     Laplacian of phi so it amplifies high-k -> use slightly lower alpha)
    #   psi (magnetic flux): large-scale, smoother structures
    DEFAULT_ALPHAS = [2.5, 2.0, 2.5, 3.0, 2.5]
    DEFAULT_TAUS   = [7.0, 5.0, 7.0, 10.0, 7.0]

    def __init__(self, Nx=512, Ny=256, alpha=None, tau=None,
                 field_scales=None, device='cpu',
                 x_active_range=(180, 330), x_edge_width=20.0,
                 use_radial_mask=True, use_abs_constraint=True):
        """
        Args:
            x_active_range: (lo, hi) radial index range where turbulence is
                            concentrated. Outside this band, field values taper
                            smoothly to near-zero. Set to None to disable
                            (falls back to full Dirichlet window).
            x_edge_width: width (in grid points) of the smooth tanh transition
                          at each edge of the active band.
            use_radial_mask: If False, disable radial mask (use full Dirichlet window).
            use_abs_constraint: If True, apply abs() to n and Ti for positivity.
        """
        self.use_abs_constraint = use_abs_constraint
        self.Nx = Nx
        self.Ny = Ny
        self.device = device
        self.n_fields = 5

        # Default scales approximate data std: n~1.0, U~0.8, vpar~1.5, psi~20, Ti~1.5
        if field_scales is None:
            field_scales = torch.tensor([1.0, 0.8, 1.5, 20.0, 1.5])
        self.field_scales = field_scales.to(device)

        # Parse alpha/tau: scalar -> broadcast, list -> per-field
        if alpha is None:
            alphas = self.DEFAULT_ALPHAS
        elif isinstance(alpha, (int, float)):
            alphas = [float(alpha)] * 5
        else:
            alphas = list(alpha)
        if tau is None:
            taus = self.DEFAULT_TAUS
        elif isinstance(tau, (int, float)):
            taus = [float(tau)] * 5
        else:
            taus = list(tau)

        Lx = 2 * math.pi
        Ly = 2 * math.pi
        cx = (4 * math.pi ** 2) / (Lx ** 2)
        cy = (4 * math.pi ** 2) / (Ly ** 2)

        kx = torch.arange(0, Nx, dtype=torch.float32, device=device)
        ky = torch.arange(0, Ny // 2 + 1, dtype=torch.float32, device=device)
        kx2 = (cx * kx ** 2).unsqueeze(1)                    # (Nx, 1)
        ky2 = (cy * ky ** 2).unsqueeze(0)                     # (1, Ny//2+1)

        # Check if all fields share the same spectral params (fast path)
        self.per_field_spectral = not (len(set(alphas)) == 1 and len(set(taus)) == 1)

        if self.per_field_spectral:
            self.sqrt_eig_per_field = []
            for a, t in zip(alphas, taus):
                sigma = 0.5 * t ** (0.5 * (2 * a - 2.0))
                se = sigma * (kx2 + ky2 + t ** 2) ** (-a / 2.0)
                se[0, 0] = 0.0
                self.sqrt_eig_per_field.append(se * Nx * Ny)
            self.sqrt_eig = None
        else:
            a, t = alphas[0], taus[0]
            sigma = 0.5 * t ** (0.5 * (2 * a - 2.0))
            sqrt_eig = sigma * (kx2 + ky2 + t ** 2) ** (-a / 2.0)
            sqrt_eig[0, 0] = 0.0
            self.sqrt_eig = sqrt_eig * Nx * Ny
            self.sqrt_eig_per_field = None

        # Radial window: Dirichlet BC * optional active-band mask
        idx = torch.arange(Nx, dtype=torch.float32, device=device)
        dirichlet = torch.sin(math.pi * idx / (Nx - 1))  # (Nx,)

        if use_radial_mask and x_active_range is not None:
            lo, hi = x_active_range
            ew = max(x_edge_width, 1.0)
            # smooth bump: ~1 inside [lo, hi], ~0 outside
            radial_mask = 0.5 * (torch.tanh((idx - lo) / ew)
                                 - torch.tanh((idx - hi) / ew))
            self.radial_window = dirichlet * radial_mask
            print(f'GRF radial mask: active x=[{lo}, {hi}], edge_width={ew}')
        else:
            self.radial_window = dirichlet
            if not use_radial_mask:
                print('GRF radial mask: disabled (full Dirichlet window)')
            elif x_active_range is None:
                print('GRF radial mask: disabled (x_active_range is None)')

        print(f'GRF config: alphas={alphas}, taus={taus}, '
              f'scales={self.field_scales.tolist()}')

    def __call__(self, batch_size):
        """Generate random 5-field initial conditions.

        Returns:
            (batch_size, Nx, Ny, 5) float32 tensor, channel-last
        """
        B, Nx, Ny = batch_size, self.Nx, self.Ny
        fields = []
        for c in range(self.n_fields):
            se = self.sqrt_eig_per_field[c] if self.per_field_spectral else self.sqrt_eig
            xi = torch.randn(B, Nx, Ny // 2 + 1, 2,
                              dtype=torch.float32, device=self.device)
            xi[..., 0] *= se
            xi[..., 1] *= se
            u = torch.fft.irfft2(torch.view_as_complex(xi), s=(Nx, Ny))
            u = u * self.radial_window.reshape(1, Nx, 1)
            u = u * self.field_scales[c]
            fields.append(u)
        out = torch.stack(fields, dim=-1)  # (B, Nx, Ny, 5)

        # Positivity constraints: n (idx 0) and Ti (idx 4) must be >= 0
        if self.use_abs_constraint:
            out[..., 0] = torch.abs(out[..., 0])
            out[..., 4] = torch.abs(out[..., 4])

        return out

    @staticmethod
    def from_data_stats(data_path, Nx=512, Ny=256, alpha=None, tau=None,
                        device='cpu', time_start=250.0, time_end=300.0,
                        dt_data=1.0, x_active_range=(180, 330),
                        x_edge_width=20.0,
                        use_radial_mask=True, use_abs_constraint=True):
        """Create GRF with field_scales derived from training data std."""
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        scales = []
        for name in FIELD_NAMES:
            t0 = int(round(time_start / dt_data))
            t1 = int(round(time_end / dt_data))
            field = data[name][:, t0:t1 + 1]
            scales.append(field.std().item())
        del data
        field_scales = torch.tensor(scales, dtype=torch.float32)
        print(f'GRF field_scales from data: {dict(zip(FIELD_NAMES, scales))}')
        return MHD5FieldGRF(Nx, Ny, alpha, tau, field_scales, device,
                            x_active_range=x_active_range,
                            x_edge_width=x_edge_width,
                            use_radial_mask=use_radial_mask,
                            use_abs_constraint=use_abs_constraint)
