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
    print(f'  Loading {data_path} (mmap) ...', end=' ', flush=True)
    data = torch.load(data_path, map_location='cpu', weights_only=False, mmap=True)
    print('done.', flush=True)

    # Determine data's actual time_start for correct index calculation
    # Priority: 1) metadata['time_start'], 2) t_list[0], 3) default 0.0
    data_time_start = 0.0  # Safe default (most raw simulation data starts at t=0)
    if 'metadata' in data and 'time_start' in data['metadata']:
        data_time_start = data['metadata']['time_start']
    elif 't_list' in data:
        data_time_start = float(data['t_list'][0])

    # Calculate time indices relative to data's actual start time
    t_start_idx = int(round((time_start - data_time_start) / dt_data))
    t_end_idx = int(round((time_end - data_time_start) / dt_data))
    
    # Clamp indices to valid range (prevents out-of-bounds for short datasets)
    T_total = data[FIELD_NAMES[0]].shape[1]
    t_start_idx = max(0, min(t_start_idx, T_total - 1))
    t_end_idx = max(t_start_idx, min(t_end_idx, T_total - 1))
    
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

    # Determine data's actual time_start for correct index calculation
    # Priority: 1) metadata['time_start'], 2) t_list[0], 3) default 0.0
    data_time_start = 0.0  # Safe default (most raw simulation data starts at t=0)
    if 'metadata' in data and 'time_start' in data['metadata']:
        data_time_start = data['metadata']['time_start']
    elif 't_list' in data:
        data_time_start = float(data['t_list'][0])

    # Calculate time indices relative to data's actual start time
    t_start_idx = int(round((time_start - data_time_start) / dt_data))
    t_end_idx = int(round((time_end - data_time_start) / dt_data))
    
    # Clamp indices to valid range (prevents out-of-bounds for short datasets)
    T_total = data[FIELD_NAMES[0]].shape[1]
    t_start_idx = max(0, min(t_start_idx, T_total - 1))
    t_end_idx = max(t_start_idx, min(t_end_idx, T_total - 1))

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
    
    actual_time_start = data_time_start + t_start_idx * dt_data
    actual_time_end = data_time_start + t_end_idx * dt_data
    metadata = dict(Nx=Nx, Ny=Ny, dt_data=dt_data,
                    n_samples=B, n_timesteps=T, 
                    t_start=actual_time_start, t_end=actual_time_end,
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


# ---------------------------------------------------------------------------
# Model-evolved GRF data generator for self-training / transfer learning
# ---------------------------------------------------------------------------

class ModelEvolvedGRFGenerator:
    """Generate training data by evolving GRF through a model (pretrained or self).
    
    This generator produces states that lie closer to the physical manifold compared
    to raw GRF samples, which may not satisfy PDE constraints.
    
    IMPORTANT: Gradient isolation is enforced via three mechanisms:
    1. Separate model instance (not a reference to training model)
    2. All parameters have requires_grad=False
    3. Inference wrapped in torch.no_grad()
    4. Output tensor is detached
    
    Supports two modes:
    - External model mode: Use a fixed pretrained model (activate immediately)
    - Self-training mode: Use training model's copy, with dynamic weight updates
    """
    
    def __init__(self, model, grf_generator, rollout_steps=10, device='cpu', verbose=True):
        """
        Args:
            model: OmniFluids2D model instance (will be frozen, no gradients)
            grf_generator: MHD5FieldGRF instance for generating initial conditions
            rollout_steps: Number of model inference steps to evolve GRF
            device: Target device
            verbose: Whether to print status messages (set False for non-main processes)
        """
        if rollout_steps < 1:
            raise ValueError(f'rollout_steps must be >= 1, got {rollout_steps}')
        
        self.model = model
        self.grf = grf_generator
        self.rollout_steps = rollout_steps
        self.device = device
        self._is_active = False  # Delayed activation support
        self._update_count = 0   # Track number of weight updates
        self._verbose = verbose
        
        # Freeze model - NO gradients will flow through this model
        self._freeze_model()
        
        if self._verbose:
            print(f'[ModelEvolvedGRFGenerator] Created: rollout_steps={rollout_steps}, '
                  f'device={device}, active={self._is_active}')
    
    def _freeze_model(self):
        """Freeze model parameters to prevent gradient flow."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def activate(self):
        """Activate the generator (start using model-evolved data)."""
        self._is_active = True
        if self._verbose:
            print(f'[ModelEvolvedGRFGenerator] ACTIVATED - now using model-evolved data')
    
    def is_active(self):
        """Check if generator is currently active."""
        return self._is_active
    
    def update_model_weights(self, source_model):
        """Update internal model weights from source (for self-training mode).
        
        Args:
            source_model: The training model to copy weights from (deep copy)
        """
        # state_dict() returns a copy, not a reference
        self.model.load_state_dict(source_model.state_dict())
        # Re-freeze after update
        self._freeze_model()
        self._update_count += 1
        if self._verbose:
            print(f'[ModelEvolvedGRFGenerator] Weights UPDATED (update #{self._update_count})')
    
    def __call__(self, batch_size):
        """Generate training data with full gradient isolation.
        
        If not active: returns raw GRF samples
        If active: returns model-evolved GRF samples (detached)
        
        Returns:
            (batch_size, Nx, Ny, 5) float32 tensor
        """
        # 1. Generate GRF random initial conditions
        x_0 = self.grf(batch_size)  # (B, Nx, Ny, 5)
        
        # 2. If not active, return raw GRF
        if not self._is_active:
            return x_0
        
        # 3. Evolve through model (no gradient tracking)
        current = x_0
        with torch.no_grad():
            for _ in range(self.rollout_steps):
                # inference=True → output (B, Nx, Ny, 5, 1)
                out = self.model(current, inference=True)
                current = out[..., -1]  # Take last frame → (B, Nx, Ny, 5)
        
        # 4. Detach to ensure no gradient connection (extra safety)
        return current.detach()
    
    @property
    def Nx(self):
        return self.grf.Nx
    
    @property
    def Ny(self):
        return self.grf.Ny


# ---------------------------------------------------------------------------
# Metrics and Visualization
# ---------------------------------------------------------------------------


def compute_metrics_and_visualize(
    gt_traj,
    pred_traj,
    metric_step_list=None,
    plot_step_list=None,
    visualize=True,
    save_dir='./eval_results',
    time_start=250.0,
    dt_data=1.0,
    sample_idx=0,
    field_names=None,
    eval_key='',
):
    """Compute L2 error metrics and visualize GT vs Prediction trajectories.

    Args:
        gt_traj: Ground truth trajectory, shape (B, T, Nx, Ny, n_fields)
        pred_traj: Predicted trajectory, shape (B, T, Nx, Ny, n_fields)
        metric_step_list: List of time steps for metric computation (default: [1, 3, 10])
        plot_step_list: List of time steps for visualization (default: [0, 1, 3, 5, 10])
        visualize: Whether to generate plots (default: True)
        save_dir: Directory to save results
        time_start: Physical start time in seconds (default: 250.0)
        dt_data: Time step between trajectory frames in seconds (default: 1.0)
        sample_idx: Which sample in batch to visualize (default: 0)
        field_names: List of field names (default: module-level FIELD_NAMES)
        eval_key: Evaluation key string appended to saved filenames

    Returns:
        dict: Metrics dictionary with total, per-field, and per-channel-mean
              L2 errors (averaged over batch)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if field_names is None:
        field_names = FIELD_NAMES
    if metric_step_list is None:
        metric_step_list = [1, 3, 10]
    if plot_step_list is None:
        plot_step_list = [0, 1, 3, 5, 10]

    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.cpu().numpy()
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()

    assert gt_traj.shape == pred_traj.shape, \
        f'Shape mismatch: gt_traj {gt_traj.shape} vs pred_traj {pred_traj.shape}'
    assert gt_traj.ndim == 5 and gt_traj.shape[-1] == len(field_names), \
        f'Expected shape (B, T, Nx, Ny, {len(field_names)}), got {gt_traj.shape}'

    B, T, Nx, Ny, n_fields = gt_traj.shape
    os.makedirs(save_dir, exist_ok=True)

    print(f'  Termwise metrics: (B={B}, T={T}, Nx={Nx}, Ny={Ny}, fields={n_fields})')

    # =========================================================================
    # 1. Compute Relative L2 Error Metrics (vectorized over batch)
    # =========================================================================
    metrics = {
        'total': {},
        'per_field': {name: {} for name in field_names},
        'per_channel_mean': {},
    }

    for step in metric_step_list:
        if step >= T:
            print(f'Warning: step {step} >= T={T}, skipping')
            continue

        gt_s = gt_traj[:, step]      # (B, Nx, Ny, n_fields)
        pred_s = pred_traj[:, step]  # (B, Nx, Ny, n_fields)

        # Total relative L2: ||pred-gt||_F / ||gt||_F, per sample then average
        diff_norms = np.sqrt(np.sum((pred_s - gt_s) ** 2, axis=(1, 2, 3)))  # (B,)
        gt_norms = np.sqrt(np.sum(gt_s ** 2, axis=(1, 2, 3)))               # (B,)
        rel_l2 = diff_norms / np.maximum(gt_norms, 1e-10)                   # (B,)
        metrics['total'][step] = float(np.mean(rel_l2))

        # Per-field relative L2
        per_ch_vals = []
        for c, name in enumerate(field_names):
            diff_c = np.sqrt(np.sum((pred_s[:, :, :, c] - gt_s[:, :, :, c]) ** 2, axis=(1, 2)))
            gt_c = np.sqrt(np.sum(gt_s[:, :, :, c] ** 2, axis=(1, 2)))
            rel_c = diff_c / np.maximum(gt_c, 1e-10)
            val = float(np.mean(rel_c))
            metrics['per_field'][name][step] = val
            per_ch_vals.append(val)

        # Per-channel mean: average of per-field rel L2 across channels
        metrics['per_channel_mean'][step] = float(np.mean(per_ch_vals))

    # Compute mean over first 10 time steps (or available steps)
    steps_for_mean = [s for s in range(1, 11) if s in metrics['total']]
    if steps_for_mean:
        metrics['mean_10step'] = float(np.mean([metrics['total'][s] for s in steps_for_mean]))
        metrics['mean_rel_l2'] = metrics['mean_10step']  # Alias for compatibility with evaluate()
        metrics['mean_10step_per_field'] = {
            name: float(np.mean([metrics['per_field'][name][s] for s in steps_for_mean]))
            for name in field_names
        }
        metrics['mean_10step_per_channel_mean'] = float(np.mean(
            [metrics['per_channel_mean'][s] for s in steps_for_mean]
        ))

    # Save metrics to txt file
    suffix = f'_{eval_key}' if eval_key else ''
    txt_path = os.path.join(save_dir, f'termwise_metrics{suffix}.txt')
    with open(txt_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write(f'Relative L2 Error (averaged over B={B} samples)\n')
        f.write('=' * 60 + '\n\n')

        if steps_for_mean:
            steps_str = ','.join(map(str, steps_for_mean))
            f.write(f'Mean Relative L2 Error (over {len(steps_for_mean)} steps: {steps_str}):\n')
            f.write('-' * 40 + '\n')
            f.write(f'  Total:            {metrics["mean_10step"]:.6f}\n')
            f.write(f'  Per-channel mean: {metrics["mean_10step_per_channel_mean"]:.6f}\n')
            for name in field_names:
                f.write(f'  {name:5s}:           {metrics["mean_10step_per_field"][name]:.6f}\n')
            f.write('\n')

        f.write('Per-step metrics (total | per_ch_mean | per-field):\n')
        f.write('-' * 40 + '\n')
        header = ['step', 'total', 'ch_mean'] + field_names
        f.write('\t'.join(header) + '\n')
        for step in sorted(metrics['total'].keys()):
            t_phys = time_start + step * dt_data
            row = [f'{step}']
            row.append(f'{metrics["total"].get(step, 0):.6f}')
            row.append(f'{metrics["per_channel_mean"].get(step, 0):.6f}')
            for name in field_names:
                row.append(f'{metrics["per_field"][name].get(step, 0):.6f}')
            f.write('\t'.join(row) + '\n')
        if steps_for_mean:
            f.write(f'mean({len(steps_for_mean)}steps)\t{metrics["mean_10step"]:.6f}')
            f.write(f'\t{metrics["mean_10step_per_channel_mean"]:.6f}')
            for name in field_names:
                f.write(f'\t{metrics["mean_10step_per_field"][name]:.6f}')
            f.write('\n')

    print(f'  Termwise metrics saved to: {txt_path}')

    # =========================================================================
    # 2. Visualization (if enabled) - use sample_idx
    # =========================================================================
    if not visualize:
        return metrics

    if sample_idx >= B:
        print(f'Warning: sample_idx={sample_idx} >= B={B}, using sample_idx=0')
        sample_idx = 0

    print(f'  Visualizing sample {sample_idx} of {B}')

    gt_vis = gt_traj[sample_idx]      # (T, Nx, Ny, n_fields)
    pred_vis = pred_traj[sample_idx]  # (T, Nx, Ny, n_fields)

    valid_plot_steps = [s for s in plot_step_list if s < T]
    n_cols = len(valid_plot_steps)

    if n_cols == 0:
        print('Warning: No valid plot steps, skipping visualization')
        return metrics

    for c, field_name in enumerate(field_names):
        fig, axes = plt.subplots(5, n_cols, figsize=(4 * n_cols, 16))
        if n_cols == 1:
            axes = axes.reshape(5, 1)

        for j, step in enumerate(valid_plot_steps):
            t_phys = time_start + step * dt_data

            gt_snap = gt_vis[step, :, :, c]
            pred_snap = pred_vis[step, :, :, c]
            err_snap = pred_snap - gt_snap

            if step > 0:
                gt_residual = gt_snap - gt_vis[step - 1, :, :, c]
                pred_residual = pred_snap - pred_vis[step - 1, :, :, c]
            else:
                gt_residual = np.zeros_like(gt_snap)
                pred_residual = np.zeros_like(pred_snap)

            vmin, vmax = gt_snap.min(), gt_snap.max()
            err_abs = max(abs(err_snap.min()), abs(err_snap.max()), 1e-10)
            res_abs = max(abs(gt_residual).max(), abs(pred_residual).max(), 1e-10)

            im0 = axes[0, j].imshow(gt_snap.T, aspect='auto', origin='lower',
                                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, j].set_title(f'GT  t={t_phys:.0f}s', fontsize=10)
            fig.colorbar(im0, ax=axes[0, j], fraction=0.046)

            im1 = axes[1, j].imshow(pred_snap.T, aspect='auto', origin='lower',
                                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[1, j].set_title(f'Pred  t={t_phys:.0f}s', fontsize=10)
            fig.colorbar(im1, ax=axes[1, j], fraction=0.046)

            im2 = axes[2, j].imshow(err_snap.T, aspect='auto', origin='lower',
                                     cmap='bwr', vmin=-err_abs, vmax=err_abs)
            axes[2, j].set_title(f'Error  t={t_phys:.0f}s', fontsize=10)
            fig.colorbar(im2, ax=axes[2, j], fraction=0.046)

            t_prev = t_phys - dt_data if step > 0 else t_phys
            res_title = f'd/dt  [{t_prev:.0f}->{t_phys:.0f}s]' if step > 0 else f't={t_phys:.0f}s (=0)'
            im3 = axes[3, j].imshow(gt_residual.T, aspect='auto', origin='lower',
                                     cmap='PuOr', vmin=-res_abs, vmax=res_abs)
            axes[3, j].set_title(f'GT Res  {res_title}', fontsize=9)
            fig.colorbar(im3, ax=axes[3, j], fraction=0.046)

            im4 = axes[4, j].imshow(pred_residual.T, aspect='auto', origin='lower',
                                     cmap='PuOr', vmin=-res_abs, vmax=res_abs)
            axes[4, j].set_title(f'Pred Res  {res_title}', fontsize=9)
            fig.colorbar(im4, ax=axes[4, j], fraction=0.046)

            if j == 0:
                axes[0, j].set_ylabel('GT\ny (binormal)')
                axes[1, j].set_ylabel('Prediction\ny (binormal)')
                axes[2, j].set_ylabel('Error\ny (binormal)')
                axes[3, j].set_ylabel('GT Residual\ny (binormal)')
                axes[4, j].set_ylabel('Pred Residual\ny (binormal)')

            axes[4, j].set_xlabel('x (radial)')

        fig.suptitle(f'Field: {field_name} (sample {sample_idx})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(save_dir, f'termwise_field_{field_name}{suffix}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {fig_path}')

    return metrics
