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


def load_mhd5_snapshots(data_path, time_start=250.0, time_end=300.0, dt_data=1.0,
                         single_trajectory=False, traj_idx=0):
    """Load 5-field MHD dataset and extract training snapshots.

    Expected format: dict with keys 'n', 'U', 'vpar', 'psi', 'Ti',
    each of shape (B, T, Nx, Ny) in float64.

    Args:
        data_path: path to .pt dataset file
        time_start: start time for training window (seconds)
        time_end: end time for training window (seconds)
        dt_data: time interval between consecutive snapshots
        single_trajectory: If True, load only one trajectory for overfitting test
        traj_idx: Index of trajectory to load when single_trajectory=True

    Returns:
        snapshots: (N, Nx, Ny, 5) float32 tensor, channel-last
        metadata: dict with Nx, Ny, dt_data, n_samples, n_timesteps
    """
    # Use mmap=True to avoid loading entire file into RAM (memory-mapped I/O)
    # This significantly speeds up loading for large files by only reading accessed data
    print(f'  Loading {data_path} (mmap) ...', end=' ', flush=True)
    data = torch.load(data_path, map_location='cpu', weights_only=False, mmap=True)
    print('done.', flush=True)

    # Calculate time indices FIRST, then slice each field before stacking
    # This avoids materializing the full tensor in memory
    t_start_idx = int(round(time_start / dt_data))
    t_end_idx = int(round(time_end / dt_data))
    
    # Slice time window from each field (mmap only reads accessed data)
    fields = [data[name][:, t_start_idx:t_end_idx + 1] for name in FIELD_NAMES]
    states = torch.stack(fields, dim=2)  # (B, T_slice, 5, Nx, Ny)
    del data  # Release mmap handle early

    B, T, C, Nx, Ny = states.shape
    
    # For overfitting test: use only one trajectory
    if single_trajectory:
        if traj_idx >= B:
            traj_idx = 0
        states = states[traj_idx:traj_idx+1]  # Keep dim: (1, T, 5, Nx, Ny)
        B = 1
        print(f'[OVERFITTING TEST] Using single trajectory #{traj_idx}: {T} timesteps, {C} fields, '
              f'{Nx}x{Ny} grid, time [{time_start}, {time_end}]')
    else:
        print(f'Loaded data: {B} samples, {T} timesteps, {C} fields, '
              f'{Nx}x{Ny} grid, time [{time_start}, {time_end}]')

    # Flatten (B, T) -> (B*T,) and convert to channel-last float32
    # Note: .float() creates a new tensor, so data is now in RAM (not mmap'd)
    snapshots = states.reshape(B * T, C, Nx, Ny).permute(0, 2, 3, 1).float()

    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, t_start=time_start, t_end=time_end,
                    single_trajectory=single_trajectory, traj_idx=traj_idx)
    return snapshots, metadata


def load_mhd5_trajectories(data_path, time_start=250.0, time_end=300.0,
                           dt_data=1.0, single_trajectory=False, traj_idx=0):
    """Load 5-field MHD dataset as trajectories for evaluation.

    Args:
        data_path: path to .pt dataset file
        time_start: start time for evaluation window (seconds)
        time_end: end time for evaluation window (seconds)
        dt_data: time interval between consecutive snapshots
        single_trajectory: If True, load only one trajectory for overfitting test
        traj_idx: Index of trajectory to load when single_trajectory=True

    Returns:
        trajectories: (B, T, 5, Nx, Ny) float32 tensor, channel-first
                      (compatible with mhd_sim eval tools)
        metadata: dict
    """
    # Use mmap=True to avoid loading entire file into RAM (memory-mapped I/O)
    print(f'  Loading {data_path} (mmap) ...', end=' ', flush=True)
    data = torch.load(data_path, map_location='cpu', weights_only=False, mmap=True)
    print('done.', flush=True)

    t_start_idx = int(round(time_start / dt_data))
    t_end_idx = int(round(time_end / dt_data))

    # Slice time before stacking to reduce peak memory
    # With mmap, only the sliced portion will be read from disk
    fields = [data[name][:, t_start_idx:t_end_idx + 1].clone() for name in FIELD_NAMES]
    del data
    states = torch.stack(fields, dim=2).float()  # (B, T, 5, Nx, Ny)
    del fields

    B, T, C, Nx, Ny = states.shape
    
    # For overfitting test: use only one trajectory
    if single_trajectory:
        if traj_idx >= B:
            traj_idx = 0
        states = states[traj_idx:traj_idx+1]  # Keep dim: (1, T, 5, Nx, Ny)
        B = 1
        print(f'  [OVERFITTING TEST] Trajectory #{traj_idx}: {T} steps, {C} fields, {Nx}x{Ny}')
    else:
        print(f'  Trajectories: {B} samples, {T} steps, {C} fields, {Nx}x{Ny}')
    
    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, t_start=time_start, t_end=time_end,
                    single_trajectory=single_trajectory, traj_idx=traj_idx)
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
                 use_radial_mask=True):
        """
        Args:
            x_active_range: (lo, hi) radial index range where turbulence is
                            concentrated. Outside this band, field values taper
                            smoothly to near-zero. Set to None to disable
                            (falls back to full Dirichlet window).
            x_edge_width: width (in grid points) of the smooth tanh transition
                          at each edge of the active band.
            use_radial_mask: whether to apply radial window masking (Dirichlet BC + active band)
        """
        self.Nx = Nx
        self.Ny = Ny
        self.device = device
        self.n_fields = 5
        self.use_radial_mask = use_radial_mask

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

        if x_active_range is not None:
            lo, hi = x_active_range
            ew = max(x_edge_width, 1.0)
            # smooth bump: ~1 inside [lo, hi], ~0 outside
            radial_mask = 0.5 * (torch.tanh((idx - lo) / ew)
                                 - torch.tanh((idx - hi) / ew))
            self.radial_window = dirichlet * radial_mask
            print(f'GRF radial mask: active x=[{lo}, {hi}], edge_width={ew}')
        else:
            self.radial_window = dirichlet

        # Store x_active_range for logging
        self.x_active_range = x_active_range
        self.x_edge_width = x_edge_width
        
        print(f'GRF config: alphas={alphas}, taus={taus}, '
              f'scales={self.field_scales.tolist()}')
        print(f'GRF flags: use_radial_mask={self.use_radial_mask}, '
              f'x_active_range={self.x_active_range}, x_edge_width={self.x_edge_width}')

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
            if self.use_radial_mask:
                u = u * self.radial_window.reshape(1, Nx, 1)
            u = u * self.field_scales[c]
            fields.append(u)
        out = torch.stack(fields, dim=-1)  # (B, Nx, Ny, 5)
        return out

    @staticmethod
    def from_data_stats(data_path, Nx=512, Ny=256, alpha=None, tau=None,
                        device='cpu', time_start=250.0, time_end=300.0,
                        dt_data=1.0, x_active_range=(180, 330),
                        x_edge_width=20.0,
                        use_radial_mask=True):
        """Create GRF with field_scales derived from training data std."""
        print(f'  Computing GRF stats from {data_path} (mmap) ...', end=' ', flush=True)
        data = torch.load(data_path, map_location='cpu', weights_only=False, mmap=True)
        scales = []
        for name in FIELD_NAMES:
            t0 = int(round(time_start / dt_data))
            t1 = int(round(time_end / dt_data))
            field = data[name][:, t0:t1 + 1]
            scales.append(field.std().item())
        del data
        print('done.', flush=True)
        field_scales = torch.tensor(scales, dtype=torch.float32)
        print(f'GRF field_scales from data: {dict(zip(FIELD_NAMES, scales))}')
        return MHD5FieldGRF(Nx, Ny, alpha, tau, field_scales, device,
                            x_active_range=x_active_range,
                            x_edge_width=x_edge_width,
                            use_radial_mask=use_radial_mask)


# ---------------------------------------------------------------------------
# Fixed GRF data for overfitting test
# ---------------------------------------------------------------------------

class FixedGRFDataSampler:
    """Generate a single fixed GRF sample and repeat it for overfitting test.
    
    This is used to test if the model can overfit to a single random GRF initial
    condition by repeatedly training on the same data.
    """
    
    def __init__(self, grf_generator, fixed_seed=42):
        """
        Args:
            grf_generator: MHD5FieldGRF instance
            fixed_seed: seed for generating the fixed sample
        """
        self.grf = grf_generator
        self.fixed_seed = fixed_seed
        self.fixed_data = None
        self.Nx = grf_generator.Nx
        self.Ny = grf_generator.Ny
        self.device = grf_generator.device
        
    def generate_fixed_data(self, batch_size):
        """Generate the fixed GRF data that will be reused."""
        # Save current random state
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state(self.device)
        
        # Set fixed seed for reproducibility
        torch.manual_seed(self.fixed_seed)
        
        # Generate single sample
        self.fixed_data = self.grf(batch_size)
        
        # Restore random state
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
        
        print(f'[Fixed GRF Overfitting] Generated fixed GRF data with seed {self.fixed_seed}')
        print(f'  Shape: {self.fixed_data.shape}')
        print(f'  Field ranges: n=[{self.fixed_data[...,0].min():.3f}, {self.fixed_data[...,0].max():.3f}], '
              f'U=[{self.fixed_data[...,1].min():.3f}, {self.fixed_data[...,1].max():.3f}], '
              f'psi=[{self.fixed_data[...,3].min():.3f}, {self.fixed_data[...,3].max():.3f}]')
        
        return self.fixed_data
    
    def next_batch(self, batch_size):
        """Return the fixed GRF data (repeated if needed)."""
        if self.fixed_data is None:
            self.generate_fixed_data(batch_size)
        return self.fixed_data
    
    def __call__(self, batch_size):
        """Make it callable like GRF."""
        return self.next_batch(batch_size)
