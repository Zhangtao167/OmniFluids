
import torch
import math
import sys
EPS = 1e-7
_MHD_SIM_HW_CACHE = {}


def PSM_KS(w, param, t_interval=5.0, loss_mode='cn'):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = 1/4 * torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = 1/4 * torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = - (k_x ** 2 + k_y ** 2)
    L = (lap + lap ** 2) * param[:, 0].reshape(-1, 1 , 1, 1) * w_h

    wx = torch.fft.ifft2(1j * k_x * w_h, dim=[1, 2]).real
    wy = torch.fft.ifft2(1j * k_y * w_h, dim=[1, 2]).real
    N = 0.5 * param[:, 1].reshape(-1, 1 , 1, 1) * torch.fft.fft2(wx ** 2 + wy ** 2, dim=[1, 2]) 
    dt = t_interval / (nt-1)
    if loss_mode=='cn':
        wt = (w[:, :, :, 1:] - w[:, :, :, :-1]) / dt
        Du = torch.fft.ifft2(L + N, dim=[1, 2]).real
        Du1 = wt + (Du[..., :-1] + Du[..., 1:]) * 0.5 
    if loss_mode=='mid':
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
        Du1 = wt + torch.fft.ifft2(L + N, dim=[1, 2]).real[...,1:-1] #- forcing
    return Du1


def PSM_HW(zeta, n, param, t_interval=1.0, loss_mode='cn'):
    """
    HW PDE 残差 (纯谱方法 + CN 时间离散).
    zeta: [B, S, S, Nt]
    n:    [B, S, S, Nt]
    param: [B, 5] → α, κ, log₁₀ν, N_hyper/4, k0
    返回: Du_zeta, Du_n (CN 残差)
    """
    B, S, _, Nt = zeta.shape
    device = zeta.device

    alpha = param[:, 0].reshape(B, 1, 1, 1)
    kappa = param[:, 1].reshape(B, 1, 1, 1)
    nu = (10.0 ** param[:, 2]).reshape(B, 1, 1, 1)
    N_hyper = 3   # fixed (param[:,3]=0.75 → 0.75*4=3)
    k0 = 0.15     # fixed (param[:,4])

    # 波数: k = k0 * freq_index
    half = S // 2
    freq = torch.cat([torch.arange(0, half, device=device),
                      torch.arange(-half, 0, device=device)], dim=0).float()
    kx = (k0 * freq).reshape(1, S, 1, 1)    # [1,S,1,1]
    ky = (k0 * freq).reshape(1, 1, S, 1)    # [1,1,S,1]

    k2 = kx ** 2 + ky ** 2                   # [1,S,S,1]
    k2N = k2 ** N_hyper                       # k^6

    # Poisson 求解: ∇²φ=ζ → φ̂ = -ζ̂/k²
    k2_safe = k2.clone()
    k2_safe[0, 0, 0, 0] = 1.0
    zeta_h = torch.fft.fft2(zeta, dim=[1, 2])
    n_h = torch.fft.fft2(n, dim=[1, 2])
    phi_h = -zeta_h / k2_safe
    phi_h[:, 0, 0, :] = 0.0

    # 谱导数 (6 次 ifft2)
    dphi_dx = torch.fft.ifft2(1j * kx * phi_h, dim=[1, 2]).real
    dphi_dy = torch.fft.ifft2(1j * ky * phi_h, dim=[1, 2]).real
    dzeta_dx = torch.fft.ifft2(1j * kx * zeta_h, dim=[1, 2]).real
    dzeta_dy = torch.fft.ifft2(1j * ky * zeta_h, dim=[1, 2]).real
    dn_dx = torch.fft.ifft2(1j * kx * n_h, dim=[1, 2]).real
    dn_dy = torch.fft.ifft2(1j * ky * n_h, dim=[1, 2]).real

    phi = torch.fft.ifft2(phi_h, dim=[1, 2]).real

    # Poisson 括号: [φ,f] = ∂φ/∂x·∂f/∂y − ∂φ/∂y·∂f/∂x
    pb_zeta = dphi_dx * dzeta_dy - dphi_dy * dzeta_dx
    pb_n = dphi_dx * dn_dy - dphi_dy * dn_dx

    # 耦合项 & 密度梯度驱动
    coupling = alpha * (phi - n)
    kappa_dphi_dy = kappa * dphi_dy

    # 超扩散对空间算子的贡献 (= -D(f) = +ν·k^{2N}·f̂, 正号)
    hyper_zeta = torch.fft.ifft2(nu * k2N * zeta_h, dim=[1, 2]).real
    hyper_n = torch.fft.ifft2(nu * k2N * n_h, dim=[1, 2]).real

    # 空间算子 Sp = -RHS
    Sp_zeta = pb_zeta - coupling + hyper_zeta
    Sp_n = pb_n + kappa_dphi_dy - coupling + hyper_n

    # CN 时间离散
    dt = t_interval / (Nt - 1)
    if loss_mode == 'cn':
        zeta_t = (zeta[:, :, :, 1:] - zeta[:, :, :, :-1]) / dt
        n_t = (n[:, :, :, 1:] - n[:, :, :, :-1]) / dt
        Du_zeta = zeta_t + 0.5 * (Sp_zeta[:, :, :, :-1] + Sp_zeta[:, :, :, 1:])
        Du_n = n_t + 0.5 * (Sp_n[:, :, :, :-1] + Sp_n[:, :, :, 1:])
    elif loss_mode == 'mid':
        zeta_t = (zeta[:, :, :, 2:] - zeta[:, :, :, :-2]) / (2 * dt)
        n_t = (n[:, :, :, 2:] - n[:, :, :, :-2]) / (2 * dt)
        Du_zeta = zeta_t + Sp_zeta[:, :, :, 1:-1]
        Du_n = n_t + Sp_n[:, :, :, 1:-1]

    return Du_zeta, Du_n


def _compute_hw_rhs_spectral(state, param):
    """谱离散 RHS. state: [B,S,S,2], param: [B,5]."""
    B, S, _, _ = state.shape
    device = state.device

    zeta = state[..., 0]
    n = state[..., 1]

    alpha = param[:, 0].reshape(B, 1, 1)
    kappa = param[:, 1].reshape(B, 1, 1)
    nu = (10.0 ** param[:, 2]).reshape(B, 1, 1)
    N_hyper = 3
    k0 = 0.15

    half = S // 2
    freq = torch.cat([torch.arange(0, half, device=device),
                      torch.arange(-half, 0, device=device)], dim=0).float()
    kx = (k0 * freq).reshape(1, S, 1)
    ky = (k0 * freq).reshape(1, 1, S)
    k2 = kx ** 2 + ky ** 2
    k2N = k2 ** N_hyper
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0

    zeta_h = torch.fft.fft2(zeta, dim=[1, 2])
    n_h = torch.fft.fft2(n, dim=[1, 2])
    phi_h = -zeta_h / k2_safe
    phi_h[:, 0, 0] = 0.0

    dphi_dx = torch.fft.ifft2(1j * kx * phi_h, dim=[1, 2]).real
    dphi_dy = torch.fft.ifft2(1j * ky * phi_h, dim=[1, 2]).real
    dzeta_dx = torch.fft.ifft2(1j * kx * zeta_h, dim=[1, 2]).real
    dzeta_dy = torch.fft.ifft2(1j * ky * zeta_h, dim=[1, 2]).real
    dn_dy = torch.fft.ifft2(1j * ky * n_h, dim=[1, 2]).real

    phi = torch.fft.ifft2(phi_h, dim=[1, 2]).real

    pb_zeta = dphi_dx * dzeta_dy - dphi_dy * dzeta_dx
    pb_n = dphi_dx * dn_dy - dphi_dy * torch.fft.ifft2(1j * kx * n_h, dim=[1, 2]).real

    coupling = alpha * (phi - n)
    hyper_zeta = torch.fft.ifft2(nu * k2N * zeta_h, dim=[1, 2]).real
    hyper_n = torch.fft.ifft2(nu * k2N * n_h, dim=[1, 2]).real

    rhs_zeta = -pb_zeta + coupling - hyper_zeta
    rhs_n = -pb_n - kappa * dphi_dy + coupling - hyper_n
    return torch.stack([rhs_zeta, rhs_n], dim=-1)


def _compute_hw_rhs_mhd_sim(state, param, mhd_sim_root):
    """mhd_sim 离散 (Arakawa + FD hyper-diffusion) RHS."""
    if mhd_sim_root not in sys.path:
        sys.path.insert(0, mhd_sim_root)
    from numerical.equations.hasegawa_wakatani import HWConfig, HasegawaWakatani

    B, S, _, _ = state.shape
    if not (torch.allclose(param[:, 0], param[0, 0])
            and torch.allclose(param[:, 1], param[0, 1])
            and torch.allclose(param[:, 2], param[0, 2])):
        raise ValueError('mhd_sim operator backend currently requires same HW params within batch')

    alpha = float(param[0, 0].item())
    kappa = float(param[0, 1].item())
    nu = float((10.0 ** param[0, 2]).item())

    key = (mhd_sim_root, str(state.device), S, alpha, kappa, nu)
    if key not in _MHD_SIM_HW_CACHE:
        cfg = HWConfig(
            Nx=S, Ny=S, alpha=alpha, kappa=kappa, nu=nu,
            hyper_order=3, k0=0.15, diffusion_method='fd',
            device=str(state.device)
        )
        _MHD_SIM_HW_CACHE[key] = HasegawaWakatani(cfg)
    hw = _MHD_SIM_HW_CACHE[key]

    state_cf = state.permute(0, 3, 1, 2).to(dtype=torch.float64)
    rhs_cf = hw.compute_rhs(state_cf)
    return rhs_cf.permute(0, 2, 3, 1).to(dtype=state.dtype)


def compute_hw_rhs(state, param, operator_discretization='spectral',
                   mhd_sim_root='/zhangtao/project2026/mhd_sim'):
    """计算 HW 方程 RHS. state: [B,S,S,2], param: [B,5]."""
    if operator_discretization == 'mhd_sim':
        return _compute_hw_rhs_mhd_sim(state, param, mhd_sim_root)
    return _compute_hw_rhs_spectral(state, param)


def rhs_anchored_loss(w0, w_final, param, rollout_DT, operator_discretization='spectral',
                      mhd_sim_root='/zhangtao/project2026/mhd_sim'):
    """mhd_sim 风格 RHS-anchored CN loss. w0, w_final: [B,S,S,2], param: [B,5]."""
    time_diff = (w_final - w0) / rollout_DT
    with torch.no_grad():
        rhs_0 = compute_hw_rhs(
            w0.detach(), param,
            operator_discretization=operator_discretization,
            mhd_sim_root=mhd_sim_root
        )
    rhs_f = compute_hw_rhs(
        w_final, param,
        operator_discretization=operator_discretization,
        mhd_sim_root=mhd_sim_root
    )
    target = (rhs_0 + rhs_f) / 2
    return ((time_diff - target) ** 2).mean()


def PSM_loss(w, param, t_interval=1.0, loss_mode='cn', operator_discretization='spectral',
             mhd_sim_root='/zhangtao/project2026/mhd_sim'):
    """
    w: [B, S, S, Nt, 2] — 双场 (ζ, n)
    param: [B, 5]
    """
    if operator_discretization == 'spectral':
        zeta = w[..., 0]    # [B,S,S,Nt]
        n = w[..., 1]       # [B,S,S,Nt]
        Du_z, Du_n = PSM_HW(zeta, n, param, t_interval, loss_mode)
        return 0.5 * ((torch.square(Du_z).mean() + EPS).sqrt()
                    + (torch.square(Du_n).mean() + EPS).sqrt())

    B, S, _, Nt, _ = w.shape
    dt = t_interval / (Nt - 1)
    w0 = w[:, :, :, :-1, :]     # [B,S,S,Nt-1,2]
    w1 = w[:, :, :, 1:, :]      # [B,S,S,Nt-1,2]
    wt = (w1 - w0) / dt

    param_rep = param.unsqueeze(1).repeat(1, Nt - 1, 1).reshape(-1, param.shape[-1])
    rhs0 = compute_hw_rhs(
        w0.reshape(-1, S, S, 2), param_rep,
        operator_discretization=operator_discretization,
        mhd_sim_root=mhd_sim_root
    ).reshape(B, Nt - 1, S, S, 2).permute(0, 2, 3, 1, 4)
    rhs1 = compute_hw_rhs(
        w1.reshape(-1, S, S, 2), param_rep,
        operator_discretization=operator_discretization,
        mhd_sim_root=mhd_sim_root
    ).reshape(B, Nt - 1, S, S, 2).permute(0, 2, 3, 1, 4)

    Du = wt - 0.5 * (rhs0 + rhs1)
    Du_z = Du[..., 0]
    Du_n = Du[..., 1]
    return 0.5 * ((torch.square(Du_z).mean() + EPS).sqrt()
                + (torch.square(Du_n).mean() + EPS).sqrt())