"""Physics loss for 5-field MHD using mhd_sim's compute_rhs.

Computes time-differencing loss between OmniFluids multi-frame predictions
and the physical RHS (right-hand side) from the 5-field Landau-fluid equations.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/zhangtao/project2026/mhd_sim')
from numerical.equations.five_field_mhd import FiveFieldMHD, FiveFieldMHDConfig


def make_mhd5_rhs_fn(mhd, model_dtype=torch.float32):
    """Create an RHS adapter: channel-last (B, Nx, Ny, 5) -> (B, Nx, Ny, 5).

    The FiveFieldMHD.compute_rhs expects a tuple of 5 tensors each (B, Nx, Ny)
    and returns a tuple of 5 tensors. This adapter handles the conversion,
    computing RHS in float64 for numerical accuracy.

    Args:
        mhd: FiveFieldMHD instance
        model_dtype: dtype of the model output (default float32)
    """
    def rhs_fn(x):
        # x: (B, Nx, Ny, 5), model_dtype
        x_f64 = x.to(torch.float64)
        state = tuple(x_f64[..., i] for i in range(5))
        rhs_tuple = mhd.compute_rhs(state)
        return torch.stack(rhs_tuple, dim=-1).to(model_dtype)
    return rhs_fn


def build_mhd_instance(device='cpu', Nx=512, Ny=256):
    """Build a FiveFieldMHD instance with default config.

    RHS computation always runs in float64 for numerical accuracy.
    """
    cfg = FiveFieldMHDConfig()
    cfg.Nx = Nx
    cfg.Ny = Ny
    cfg.device = str(device)
    cfg.precision = 'fp64'
    mhd = FiveFieldMHD(cfg)
    return mhd


def compute_physics_loss(pred_traj, x_0, rhs_fn, rollout_dt, output_dim,
                         time_integrator='crank_nicolson'):
    """Multi-frame physics loss using mhd_sim's RHS.

    For each consecutive frame pair in the trajectory, enforce:
        (state[t+1] - state[t]) / dt ≈ target_rhs

    where target_rhs depends on the time integrator.

    Args:
        pred_traj: (B, Nx, Ny, 5, output_dim) multi-frame model output
        x_0: (B, Nx, Ny, 5) initial state
        rhs_fn: callable (B, Nx, Ny, 5) -> (B, Nx, Ny, 5)
        rollout_dt: total time span covered by output_dim frames
        output_dim: number of predicted frames
        time_integrator: 'euler' or 'crank_nicolson'

    Returns:
        scalar loss (MSE between time derivative and RHS)
    """
    dt = rollout_dt / output_dim
    full_traj = torch.cat([x_0.unsqueeze(-1), pred_traj], dim=-1)

    total_loss = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)

    for t in range(output_dim):
        state_t = full_traj[..., t]
        state_tp1 = full_traj[..., t + 1]
        time_diff = (state_tp1 - state_t) / dt

        if time_integrator == 'euler':
            target = rhs_fn(state_t)
        elif time_integrator == 'crank_nicolson':
            target = (rhs_fn(state_t) + rhs_fn(state_tp1)) / 2.0
        else:
            raise ValueError(f"Unknown integrator: {time_integrator}")

        total_loss = total_loss + F.mse_loss(time_diff, target.detach())

    return total_loss / output_dim
