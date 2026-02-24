import torch
import math


def ks_2d_rk4(u0, T, param, dt=1e-3, record_steps=100):
    """
    2D Kuramoto-Sivashinsky equation solver using spectral method + RK4 time stepping.
    ∂u/∂t + 1/2 * (∇u)^2 + Δu + Δ^2 u = 0

    Parameters:
        u0: [batch, N, N] tensor, initial condition in physical space
        T: total simulation time
        dt: time step
        record_steps: number of saved steps (default 100)
    Returns:
        sol: [N, N, record_steps] tensor of u in physical space
        times: [record_steps] tensor of time values
    """
    device = u0.device
    param = param.to(device)
    batch = u0.shape[0]
    N = u0.shape[-1]
    steps = math.ceil(T / dt)
    save_every = steps // record_steps

    # Wavenumbers
    k_max = N/2
    kx = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    ky = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    # Spectral operators
    lap = -(kx ** 2 + ky ** 2)
    lap2 = lap ** 2
    L = lap + lap2  # Linear operator
    L = - L[None, :] * param[:, 0].reshape(batch, 1, 1)

    # Initialize
    u_h = torch.fft.fft2(u0)
    sol = torch.zeros(batch, N, N, record_steps+1, device='cpu')
    sol[..., 0] = u0
    t = 0.0
    c = 1

    def nonlinear(u_phys):
        dealias = torch.unsqueeze(torch.logical_and(torch.abs(ky) <= (2.0/3.0)*k_max, torch.abs(kx) <= (2.0/3.0)*k_max).float(), 0)[None, :]
        ux = torch.fft.ifft2(1j * kx * torch.fft.fft2(u_phys)).real
        uy = torch.fft.ifft2(1j * ky * torch.fft.fft2(u_phys)).real
        return -0.5 * param[:, 1].reshape(batch, 1, 1) * torch.fft.fft2(ux ** 2 + uy ** 2) * dealias

    for i in range(steps):
        u_phys = torch.fft.ifft2(u_h).real
        N1 = nonlinear(u_phys)
        k1 = dt * (L * u_h + N1)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k1).real
        N2 = nonlinear(u_phys)
        k2 = dt * (L * (u_h + 0.5 * k1) + N2)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k2).real
        N3 = nonlinear(u_phys)
        k3 = dt * (L * (u_h + 0.5 * k2) + N3)

        u_phys = torch.fft.ifft2(u_h + k3).real
        N4 = nonlinear(u_phys)
        k4 = dt * (L * (u_h + k3) + N4)

        u_h = u_h + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt

        if (i + 1) % save_every == 0:
            sol[:, :, :, c] = torch.fft.ifft2(u_h).real.cpu()
            print(f"Step {i + 1}/{steps}, t={t:.4f}, max(u)={sol[:,:,:,c].max():.4f}")
            c += 1

    return sol


# ===== HW solver 封装: 调用 mhd_sim =====
import sys
sys.path.insert(0, '/zhangtao/project2026/mhd_sim')
from numerical.equations.hasegawa_wakatani import HasegawaWakatani, HWConfig
from numerical.scripts.hw_eq import rk4_step as hw_rk4_step


def hw_2d_rk4(zeta0, n0, T, param, k0=0.15, N_hyper=3,
              dt=0.025, record_steps=250, device='cuda'):
    """
    param: [B,3] → α=param[:,0], κ=param[:,1], log₁₀ν=param[:,2]
    返回: [B, S, S, record_steps+1, 2]  (最后维: 0=ζ, 1=n)
    逐样本调用 mhd_sim solver (各样本参数可不同)
    """
    B = zeta0.shape[0]
    S = zeta0.shape[1]
    steps = int(T / dt)
    save_every = max(1, steps // record_steps)

    sol = torch.zeros(B, S, S, record_steps + 1, 2, device='cpu')

    for b in range(B):
        alpha_val = param[b, 0].item()
        kappa_val = param[b, 1].item()
        nu_val = 10 ** param[b, 2].item()

        cfg = HWConfig(
            Nx=S, Ny=S, k0=k0,
            alpha=alpha_val, kappa=kappa_val,
            nu=nu_val, hyper_order=N_hyper,
            dt=dt, Nt=steps, stamp_interval=save_every,
            device=str(device), output_path=None,
        )
        hw = HasegawaWakatani(cfg)

        # 初始状态 [2, S, S], float64
        state = torch.stack([
            zeta0[b].to(torch.float64).to(device),
            n0[b].to(torch.float64).to(device)
        ], dim=0)

        sol[b, :, :, 0, 0] = state[0].cpu().float()
        sol[b, :, :, 0, 1] = state[1].cpu().float()

        c = 1
        with torch.no_grad():
            for i in range(steps):
                state = hw_rk4_step(hw.compute_rhs, state, dt)
                if (i + 1) % save_every == 0 and c <= record_steps:
                    sol[b, :, :, c, 0] = state[0].cpu().float()
                    sol[b, :, :, c, 1] = state[1].cpu().float()
                    if c % 50 == 0:
                        print(f"  Sample {b}, frame {c}/{record_steps}, "
                              f"t={(i+1)*dt:.2f}s, max|ζ|={state[0].abs().max():.4f}")
                    c += 1

        if torch.isnan(state).any():
            print(f"WARNING: NaN detected in sample {b}")

    return sol