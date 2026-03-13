"""Physics loss for 5-field MHD using mhd_sim's compute_rhs.

Computes time-differencing loss between OmniFluids multi-frame predictions
and the physical RHS (right-hand side) from the 5-field Landau-fluid equations.
"""

import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, '/zhangtao/project2026/mhd_sim')
from numerical.equations.five_field_mhd import FiveFieldMHD, FiveFieldMHDConfig


def dealias_state(mhd, x):
    """Dealias each channel in y-direction using mhd._dealias.
    
    Removes high-frequency Fourier modes above M to prevent aliasing errors
    in nonlinear terms (e.g., Poisson bracket [phi, f]).
    
    Args:
        mhd: FiveFieldMHD instance (has _dealias method and cfg.M)
        x: (B, Nx, Ny, 5) channel-last tensor
        
    Returns:
        Dealiased tensor of same shape
    """
    # Process each field independently
    out_channels = []
    for c in range(5):
        field = x[..., c]  # (B, Nx, Ny)
        field_dealiased = mhd._dealias(field)
        out_channels.append(field_dealiased)
    return torch.stack(out_channels, dim=-1)  # (B, Nx, Ny, 5)


def make_mhd5_rhs_fn(mhd, model_dtype=torch.float32, dealias=False):
    """Create an RHS adapter: channel-last (B, Nx, Ny, 5) -> (B, Nx, Ny, 5).

    The FiveFieldMHD.compute_rhs expects a tuple of 5 tensors each (B, Nx, Ny)
    and returns a tuple of 5 tensors. This adapter handles the conversion,
    computing RHS in float64 for numerical accuracy.

    Args:
        mhd: FiveFieldMHD instance
        model_dtype: dtype of the model output (default float32)
        dealias: if True, dealias input before computing RHS (prevents aliasing
                 from high-frequency noise in neural network outputs)
    """
    def rhs_fn(x):
        # x: (B, Nx, Ny, 5), model_dtype
        x_f64 = x.to(torch.float64)
        if dealias:
            x_f64 = dealias_state(mhd, x_f64)
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


def compute_target_rms_scale(target, eps=1e-8):
    """Compute detached per-field RMS scale from the target RHS.

    Using the target magnitude keeps the relative loss sensitive to actual
    error reduction, unlike normalizing by the residual's own scale.
    """
    return target.detach().pow(2).mean(dim=(0, 1, 2), keepdim=True).sqrt().clamp(min=eps)


def parse_fixed_field_scales(fixed_field_scales, n_fields, device, dtype, eps=1e-8):
    """Parse optional fixed per-field scales for relative PDE loss."""
    if fixed_field_scales is None:
        return None

    if isinstance(fixed_field_scales, str):
        if not fixed_field_scales.strip():
            return None
        try:
            values = [float(v.strip()) for v in fixed_field_scales.split(',') if v.strip()]
        except ValueError as exc:
            raise ValueError(
                "relative_loss_fixed_scales must be a comma-separated list of floats"
            ) from exc
        scales = torch.tensor(values, device=device, dtype=dtype)
    elif torch.is_tensor(fixed_field_scales):
        scales = fixed_field_scales.detach().to(device=device, dtype=dtype)
    else:
        scales = torch.tensor(list(fixed_field_scales), device=device, dtype=dtype)

    if scales.numel() != n_fields:
        raise ValueError(
            f"relative_loss_fixed_scales expects {n_fields} values, got {scales.numel()}"
        )
    return scales.reshape(1, 1, 1, n_fields).detach().clamp(min=eps)


def get_relative_loss_scale(target, fixed_field_scales=None, eps=1e-8):
    """Return per-field normalization scale for relative PDE loss."""
    if fixed_field_scales is not None:
        return fixed_field_scales
    return compute_target_rms_scale(target, eps=eps)


def average_target_over_local_ensemble(target, ensemble_group_size):
    """Average targets within each local clean-sample ensemble group.

    Args:
        target: (B_flat, Nx, Ny, C) flattened local batch
        ensemble_group_size: Number of ensemble members per clean sample.

    Returns:
        Tensor of same shape where each ensemble member in the same group shares
        the group-mean target.
    """
    if ensemble_group_size <= 1:
        return target
    if target.shape[0] % ensemble_group_size != 0:
        raise ValueError(
            f'Batch size {target.shape[0]} is not divisible by ensemble_group_size '
            f'{ensemble_group_size}'
        )
    grouped = target.reshape(
        target.shape[0] // ensemble_group_size,
        ensemble_group_size,
        *target.shape[1:],
    )
    mean_target = grouped.mean(dim=1, keepdim=True)
    return mean_target.expand_as(grouped).reshape_as(target)


def compute_physics_loss(pred_traj, x_0, rhs_fn, rollout_dt, output_dim,
                         time_integrator='crank_nicolson', mae_weight=0.0,
                         soft_linf_weight=0.0, soft_linf_beta=10.0,
                         euler_weight=None, ensemble_group_size=1,
                         use_relative_loss=False,
                         relative_loss_fixed_scales=None):
    """Multi-frame physics loss using mhd_sim's RHS.

    For each consecutive frame pair in the trajectory, enforce:
        (state[t+1] - state[t]) / dt ≈ target_rhs

    where target_rhs depends on the time integrator.

    Total loss = MSE + mae_weight * MAE + soft_linf_weight * soft-L∞ (per-frame averaged).

    When ensemble_group_size > 1, targets are first averaged within each local
    clean-sample ensemble group and that shared mean target is used for the
    residual. This preserves per-sample grouping even when the local batch has
    multiple clean samples.

    When use_relative_loss=True, the MSE/MAE terms are computed on
    residual / scale_per_field. By default, scale_per_field is the detached
    RHS target RMS computed over batch and spatial dimensions. If
    relative_loss_fixed_scales is provided, those fixed per-field values are
    used instead.

    The soft-L∞ loss uses the Log-Sum-Exp (LSE) smooth approximation:
        soft-L∞_c = (1/β) * (logsumexp(β * |r̃_c|) - ln(N_spatial))
    where r̃_c = |residual_c| / scale_c is the field-normalized residual.
    If use_relative_loss=True, scale_c comes from either:
        - RMS(target_c).detach(), or
        - the provided fixed per-field scale.
    Otherwise, scale_c = RMS(residual_c).detach(),
    and β controls sharpness (larger β → closer to true L∞).
    Computed per-sample, per-field, then averaged over batch and summed over fields.

    Args:
        pred_traj: (B, Nx, Ny, 5, output_dim) multi-frame model output
        x_0: (B, Nx, Ny, 5) initial state
        rhs_fn: callable (B, Nx, Ny, 5) -> (B, Nx, Ny, 5)
        rollout_dt: total time span covered by output_dim frames
        output_dim: number of predicted frames
        time_integrator: 'euler' or 'crank_nicolson' (used when euler_weight is None)
        mae_weight: weight for the additional MAE loss term (default 0 = off)
        soft_linf_weight: weight for soft-L∞ loss term (default 0 = off)
        soft_linf_beta: β parameter for soft-L∞ approximation (default 10.0)
        euler_weight: If not None, enables mixed mode:
            - Computes both Euler and CN losses separately
            - Returns combined loss = euler_weight * euler_loss + (1 - euler_weight) * cn_loss
            - Also returns individual losses as dict
        ensemble_group_size: Number of local ensemble members per clean sample.
            When >1, targets are averaged within each ensemble group before
            computing residuals.
        use_relative_loss: If True, normalize residual by detached per-field
            RMS of the target RHS before computing MSE/MAE
        relative_loss_fixed_scales: Optional fixed per-field scales used when
            use_relative_loss=True. Accepts a comma-separated string or an
            iterable/tensor with one value per field in channel order.

    Returns:
        If euler_weight is None: scalar loss
        If euler_weight is not None: (combined_loss, {'euler': euler_loss, 'cn': cn_loss})
    """
    dt = rollout_dt / output_dim
    full_traj = torch.cat([x_0.unsqueeze(-1), pred_traj], dim=-1)
    fixed_field_scales = None
    if use_relative_loss:
        fixed_field_scales = parse_fixed_field_scales(
            relative_loss_fixed_scales,
            n_fields=x_0.shape[-1],
            device=x_0.device,
            dtype=x_0.dtype,
        )

    # Mixed integrator mode: compute both Euler and CN losses separately
    if euler_weight is not None:
        total_euler_mse = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
        total_cn_mse = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
        
        for t in range(output_dim):
            state_t = full_traj[..., t]
            state_tp1 = full_traj[..., t + 1]
            time_diff = (state_tp1 - state_t) / dt
            
            # Euler target: rhs(t)
            rhs_t = rhs_fn(state_t)
            target_euler_raw = rhs_t.detach()
            target_euler = average_target_over_local_ensemble(
                target_euler_raw, ensemble_group_size
            )
            residual_euler = time_diff - target_euler
            if use_relative_loss:
                target_scale = get_relative_loss_scale(
                    target_euler_raw, fixed_field_scales=fixed_field_scales
                )
                residual_euler = residual_euler / target_scale
            total_euler_mse = total_euler_mse + (residual_euler ** 2).mean()
            
            # CN target: (rhs(t) + rhs(t+1)) / 2
            # Only compute rhs_tp1 if we need CN loss (euler_weight < 1)
            if euler_weight < 1.0 - 1e-6:
                rhs_tp1 = rhs_fn(state_tp1)
                target_cn_raw = ((rhs_t + rhs_tp1) / 2.0).detach()
                target_cn = average_target_over_local_ensemble(
                    target_cn_raw, ensemble_group_size
                )
                residual_cn = time_diff - target_cn
                if use_relative_loss:
                    target_scale = get_relative_loss_scale(
                        target_cn_raw, fixed_field_scales=fixed_field_scales
                    )
                    residual_cn = residual_cn / target_scale
                total_cn_mse = total_cn_mse + (residual_cn ** 2).mean()
        
        euler_loss = total_euler_mse / output_dim
        
        if euler_weight >= 1.0 - 1e-6:
            # Pure Euler mode
            cn_loss = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
            combined_loss = euler_loss
        else:
            cn_loss = total_cn_mse / output_dim
            combined_loss = euler_weight * euler_loss + (1.0 - euler_weight) * cn_loss
        
        return combined_loss, {'euler': euler_loss.item(), 'cn': cn_loss.item()}

    # Original single-integrator mode
    total_mse = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
    total_mae = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
    total_soft_linf = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)

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

        target_raw = target.detach()
        target = average_target_over_local_ensemble(target_raw, ensemble_group_size)
        residual = time_diff - target  # (B, Nx, Ny, 5)
        target_scale = None
        if use_relative_loss:
            target_scale = get_relative_loss_scale(
                target_raw, fixed_field_scales=fixed_field_scales
            )
            residual = residual / target_scale
        total_mse = total_mse + (residual ** 2).mean()
        if mae_weight > 0:
            total_mae = total_mae + torch.abs(residual).mean()
        
        # Soft-L∞ loss: per-sample, per-field, normalized by either
        # target RMS (relative mode) or residual RMS (legacy mode).
        if soft_linf_weight > 0:
            soft_linf_residual = time_diff - target  # (B, Nx, Ny, 5)
            B_size = soft_linf_residual.shape[0]
            N_spatial = soft_linf_residual.shape[1] * soft_linf_residual.shape[2]  # Nx * Ny
            ln_N = math.log(N_spatial)
            for c in range(5):
                abs_res_c = torch.abs(soft_linf_residual[..., c])  # (B, Nx, Ny)
                if use_relative_loss:
                    scale_c = target_scale[..., c]
                else:
                    scale_c = abs_res_c.detach().pow(2).mean().sqrt().clamp(min=1e-8)
                normed = abs_res_c / scale_c  # (B, Nx, Ny)
                # logsumexp per sample over spatial dims, then batch mean
                per_sample = (1.0 / soft_linf_beta) * (
                    torch.logsumexp(soft_linf_beta * normed.reshape(B_size, -1), dim=1)
                    - ln_N
                )  # (B,)
                total_soft_linf = total_soft_linf + per_sample.mean()

    loss = total_mse / output_dim
    if mae_weight > 0:
        loss = loss + mae_weight * (total_mae / output_dim)
    if soft_linf_weight > 0:
        loss = loss + soft_linf_weight * (total_soft_linf / output_dim)
    return loss
