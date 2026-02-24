"""OmniFluids2D model adapted for 5-field MHD with mixed boundary conditions.

Key modifications from nse2d original:
- Rectangular grid (Nx x Ny) instead of square (S x S)
- DST for x-direction (Dirichlet BC), FFT for y-direction (periodic BC)
- 5 independent output heads (one per physical field)
- MoE conditioned on n_params physics parameters
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers - 1
                else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Discrete Sine Transform helpers (Dirichlet BC in x-direction)
# ---------------------------------------------------------------------------

def _dst_forward(x, dim=-2):
    """DST via odd extension + rfft.

    For a signal of length N along `dim` with Dirichlet BC (x[0]=x[N-1]=0),
    create odd-symmetric extension of length 2*(N-1) and apply rfft.
    Returns N complex coefficients along `dim`.
    """
    N = x.shape[dim]
    interior = x.narrow(dim, 1, N - 2)
    x_ext = torch.cat([x, -torch.flip(interior, [dim])], dim=dim)
    return torch.fft.rfft(x_ext, dim=dim, norm='ortho')


def _idst_backward(X, N, dim=-2):
    """Inverse DST: irfft of odd extension then take first N points."""
    x_ext = torch.fft.irfft(X, n=2 * (N - 1), dim=dim, norm='ortho')
    return x_ext.narrow(dim, 0, N)


# ---------------------------------------------------------------------------
# Spectral convolution with mixed BC
# ---------------------------------------------------------------------------

class SpectralConv2d_MHD(nn.Module):
    """Spectral convolution for mixed BC: DST in x (Dirichlet), FFT in y (periodic).

    Replaces SpectralConv2d_dy from original nse2d which used FFT in both directions.
    Uses separate mode counts for x and y to handle rectangular grids.
    """

    def __init__(self, K, in_dim, out_dim, modes_x, modes_y,
                 fourier_weight, factor, n_ff_layers, layer_norm):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y

        if not fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_y, modes_x]:
                weight = torch.FloatTensor(K, in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)
        else:
            self.fourier_weight = fourier_weight

        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm)

    def forward(self, x, att):
        x = self.forward_fourier(x, att)
        b = self.backcast_ff(x)
        return b

    def forward_fourier(self, x, att):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape  # M=Nx, N=Ny
        O = self.out_dim

        weight_y = torch.einsum("bk,kioxy->bioxy", att, self.fourier_weight[0])
        weight_x = torch.einsum("bk,kioxy->bioxy", att, self.fourier_weight[1])

        # --- Y-direction: FFT (periodic BC) ---
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')  # (B, I, Nx, Ny//2+1)
        out_ft_y = x_fty.new_zeros(B, O, M, N // 2 + 1)
        out_ft_y[:, :, :, :self.modes_y] = torch.einsum(
            "bixy,bioy->boxy",
            x_fty[:, :, :, :self.modes_y],
            torch.view_as_complex(weight_y))
        xy = torch.fft.irfft(out_ft_y, n=N, dim=-1, norm='ortho')

        # --- X-direction: DST (Dirichlet BC) via odd extension + rfft ---
        x_ftx = _dst_forward(x, dim=-2)  # (B, I, Nx, Ny) complex
        n_freq_x = x_ftx.shape[-2]
        out_ft_x = x_ftx.new_zeros(B, O, n_freq_x, N)
        out_ft_x[:, :, :self.modes_x, :] = torch.einsum(
            "bixy,biox->boxy",
            x_ftx[:, :, :self.modes_x, :],
            torch.view_as_complex(weight_x))
        xx = _idst_backward(out_ft_x, M, dim=-2)

        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        return x


# ---------------------------------------------------------------------------
# Output head (per-field, train/inference dual pathway)
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """Per-field output head with separate train/inference pathways.

    Training (output_dim=10):
      fc_a (4*10-4=36 dims) + fc_b (34 dims) = 70 dims
      → Conv1d(1,8,12,s=2): 70→30 → Conv1d(8,1,12,s=2): 30→10 = output_dim ✓

    Inference:
      fc_b (34 dims) only
      → Conv1d(1,8,12,s=2): 34→12 → Conv1d(8,1,12,s=2): 12→1 ✓
    """

    def __init__(self, width, output_dim):
        super().__init__()
        self.fc_a = nn.Linear(width, 4 * output_dim - 4)
        self.fc_b = nn.Linear(width, 34)
        self.mlp = nn.Sequential(
            nn.Conv1d(1, 8, 12, stride=2),
            nn.GELU(),
            nn.Conv1d(8, 1, 12, stride=2),
        )

    def forward(self, x, inference=False):
        B, Nx, Ny, _ = x.shape
        if not inference:
            h = torch.cat([self.fc_a(x), self.fc_b(x)], dim=-1)
        else:
            h = self.fc_b(x)
        h = F.gelu(h)
        h = self.mlp(h.reshape(-1, 1, h.shape[-1]))
        return h.reshape(B, Nx, Ny, -1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OmniFluids2D(nn.Module):
    """OmniFluids2D for 5-field MHD.

    Architecture: Input projection → [SpectralConv(DST-x, FFT-y) + MoE] × L
                  → 5 independent OutputHeads → residual + linear interpolation.

    Input:  (B, Nx, Ny, 5)  — 5 physical fields (n, U, vpar, psi, Ti)
    Output: (B, Nx, Ny, 5, T) — T = output_dim (train) or 1 (inference)
    """

    def __init__(self, Nx=512, Ny=256, K=4, T=10, modes_x=128, modes_y=128,
                 width=80, output_dim=10, n_fields=5, n_params=8,
                 n_layers=12, factor=4, n_ff_layers=2, layer_norm=True):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.output_dim = output_dim
        self.n_fields = n_fields
        self.n_layers = n_layers
        self.K = K
        self.T = T

        grid = self._make_grid(Nx, Ny)
        self.register_buffer('grid', grid)

        self.in_proj = nn.Linear(n_fields + 2, width)

        # Default physics params: [eta_i, beta, shear, lam, mass_ratio, Dn, eta, kq]
        default_params = torch.zeros(n_params)
        default_params[:8] = torch.tensor(
            [1.0, 0.01, 0.1, 1.5, 1836.0, 0.01, 0.0, 0.9])
        self.register_buffer('default_params', default_params)

        self.f_nu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_params, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(),
                nn.Linear(128, K))
            for _ in range(n_layers)
        ])

        self.fourier_weight = None
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d_MHD(
                K, in_dim=width, out_dim=width,
                modes_x=modes_x, modes_y=modes_y,
                fourier_weight=self.fourier_weight,
                factor=factor, n_ff_layers=n_ff_layers, layer_norm=layer_norm)
            for _ in range(n_layers)
        ])

        self.output_heads = nn.ModuleList([
            OutputHead(width, output_dim) for _ in range(n_fields)
        ])

    def forward(self, x, params=None, inference=False):
        """
        Args:
            x: (B, Nx, Ny, n_fields) input state
            params: (B, n_params) physics parameters, or None for defaults
            inference: if True, output single frame per field

        Returns:
            (B, Nx, Ny, n_fields, T) multi-frame prediction with residual connection
        """
        B = x.shape[0]
        x_0 = x  # (B, Nx, Ny, 5) for residual

        grid = self.grid.expand(B, -1, -1, -1)
        x = torch.cat([x, grid], dim=-1)  # (B, Nx, Ny, 7)
        x = self.in_proj(x)  # (B, Nx, Ny, width)
        x = F.gelu(x)

        if params is None:
            params = self.default_params.unsqueeze(0).expand(B, -1)

        for i in range(self.n_layers):
            att = self.f_nu[i](params)
            att = F.softmax(att / self.T, dim=-1)
            b = self.spectral_layers[i](x, att)
            x = x + b

        x = F.gelu(b)  # use last spectral output (original nse2d design)

        field_outputs = []
        for head in self.output_heads:
            field_outputs.append(head(x, inference=inference))
        out = torch.stack(field_outputs, dim=-2)  # (B, Nx, Ny, n_fields, T)

        dt_frac = (torch.arange(1, out.shape[-1] + 1, device=out.device,
                                dtype=out.dtype)
                   .reshape(1, 1, 1, 1, -1) / out.shape[-1])
        out = x_0.unsqueeze(-1) + out * dt_frac

        return out

    @staticmethod
    def _make_grid(Nx, Ny):
        """Normalized grid coordinates (1, Nx, Ny, 2) in [0, 2*pi)."""
        gx = torch.linspace(0, 2 * math.pi * (1 - 1.0 / Nx), Nx)
        gy = torch.linspace(0, 2 * math.pi * (1 - 1.0 / Ny), Ny)
        gridx = gx.reshape(1, Nx, 1, 1).expand(1, Nx, Ny, 1)
        gridy = gy.reshape(1, 1, Ny, 1).expand(1, Nx, Ny, 1)
        return torch.cat([gridx, gridy], dim=-1)
