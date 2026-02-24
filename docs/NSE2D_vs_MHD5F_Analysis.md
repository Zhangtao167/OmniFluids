# OmniFluids (nse2d) vs mhd_sim (5-field MHD) 详细对比分析

## 目录
1. [方程细节对比](#1-方程细节对比)
2. [物理场含义](#2-物理场含义)
3. [网络结构详解](#3-网络结构详解)
4. [数据流详解](#4-数据流详解)
5. [损失函数构建](#5-损失函数构建)
6. [训练设置对比](#6-训练设置对比)
7. [仿真设置](#7-仿真设置)
8. [评估与可视化](#8-评估与可视化)
9. [关键差异总结](#9-关键差异总结)
10. [迁移计划](#10-迁移计划) ⬅️ **重点阅读**
    - [重要原则](#重要原则-必读)
    - [时间步长对齐方案](#时间步长对齐方案-关键配置)
    - [实施步骤清单](#阶段-6-实施步骤清单)
11. [附录: 代码参考](#附录-代码参考)

---

## 1. 方程细节对比

### 1.1 nse2d: 2D Navier-Stokes 方程 (涡度形式)

**方程形式:**
```
∂w/∂t + u·∇w = ν∇²w + f
```

其中:
- `w`: 涡度 (vorticity)
- `u = (u_x, u_y)`: 速度场
- `ν`: 动力粘度 (kinematic viscosity)
- `f`: 外力场 (forcing)

**辅助关系:**
- 流函数 `ψ = ∇⁻²w` (通过泊松方程求解)
- 速度场 `u_x = ∂ψ/∂y`, `u_y = -∂ψ/∂x`

**代码实现** (`pretrain/psm_loss.py:7-48`):
```python
# 频域计算速度场
f_h = w_h / lap  # 流函数的傅里叶变换
ux_h = 1j * k_y * f_h  # u_x = ∂ψ/∂y
uy_h = -1j * k_x * f_h  # u_y = -∂ψ/∂x

# 涡度梯度
wx_h = 1j * k_x * w_h
wy_h = 1j * k_y * w_h

# 拉普拉斯算子
wlap_h = -lap * w_h

# 非线性项 + 扩散项
Du = (ux*wx + uy*wy - v*wlap)  # u·∇w - ν∇²w
```

**边界条件:** 周期性边界条件 (periodic BC)

---

### 1.2 mhd_sim: 5-field Landau-Fluid 方程 (⚠️ 已修正: 2D 系统!)

**文件:** `numerical/equations/five_field_mhd.py` (分支 `mhd_sim-5field_sim`)

**方程组** (2D, x-Dirichlet, y-periodic):
```
d/dt = ∂/∂t + v0(x)·∂/∂y + [φ, .]   (对流导数，eq 1,2,3,5)

dn/dt    = -∂φ/∂y - ∇_∥(vpar) + ∇_∥(J_∥) + Dn·∇_⊥²(n)
dU/dt    = (1+η_i)∂U/∂y + ∇_∥(J_∥) + Du·∇_⊥²(U)
dvpar/dt = -2·∇_∥(n) - ∇_∥(Ti) - β(2+η_i)∂ψ/∂y + η_⊥·∇_⊥²(vpar)
dpsi/dt  = (1/β)[∇_∥(φ-n) + β(1-v0)∂ψ/∂y + η·J_∥
           - √(π·me/(2·mi))·|∇_∥|(vpar-J_∥)]         ← 无对流项!
dTi/dt   = -η_i·∂φ/∂y - (2/3)∇_∥(vpar) - (2/3)√(8/π)·|∇_∥|(Ti)
           + χ_⊥·∇_⊥²(Ti)
```

**关键算子 (全部 2D, 无 z 方向):**
- `[φ, f]`: Arakawa Poisson bracket (`arakawa_scheme_2d_padx`) — x-zero-pad
- `∇_∥(f) = s_hat·λ·arcsinh(x/λ)·∂f/∂y` — 通过磁剪切在 y 方向实现
- `∇_⊥²(f) = ∂²f/∂x² + ∂²f/∂y²` — FD(x) + spectral(y)
- `|∇_∥|(f)`: Landau damping — `|k_∥|·f_hat` in rfft space
- `∂/∂y`: spectral via rfft
- `∂²/∂x²`: central FD with Dirichlet zero-pad

**辅助量:**
- `φ = ∇_⊥⁻²(U)` — PoissonSolver2D (rfft-y + tridiag-x)
- `J_∥ = ∇_⊥²(ψ)` — FD(x) + spectral(y)
- `v0(x) = v0_amp·sin(kq·x)` — 剪切流 profile

**边界条件 (2D!):**
- x 方向: **Dirichlet** (f(0,:) = f(Nx-1,:) = 0)，可选阻尼缓冲区
- y 方向: **周期性** (rfft/irfft)

---

## 2. 物理场含义

### 2.1 nse2d 物理场

| 场名 | 符号 | 物理含义 | Shape |
|------|------|----------|-------|
| 涡度 | w | ∇×u，流体旋转程度 | [B, H, W, T] |
| 外力 | f | 驱动流体运动的外加力场 | [B, H, W, 1] |
| 粘度 | ν | 流体阻力系数 | [B, 1, 1, 1] |

**参数表示** (`pretrain/train.py:94`):
```python
param = torch.concat([f_train, nu_train*torch.ones_like(f_train)], dim=-1)
# param shape: [B, H, W, 2], 其中 param[..., 0] 是 f, param[..., 1] 是 log10(ν)
```

### 2.2 5-field MHD 物理场 (⚠️ 已修正: 2D)

**主场 (神经网络输入/输出, 通道顺序 C=5):**

| 通道 | 场名 | 符号 | 物理含义 | Shape |
|------|------|------|----------|-------|
| 0 | 密度扰动 | n | 等离子体密度偏离平衡值 | (..., Nx, Ny) |
| 1 | 涡度 | U | ∇_⊥²φ，ExB 涡旋度 | (..., Nx, Ny) |
| 2 | 平行速度 | vpar | 沿磁场线方向的离子速度 | (..., Nx, Ny) |
| 3 | 磁通函数 | ψ | 磁场扰动 A_∥ | (..., Nx, Ny) |
| 4 | 离子温度 | Ti | 离子温度扰动 | (..., Nx, Ny) |

**辅助场 (不直接进网络):**

| 场名 | 符号 | 物理含义 | 计算方式 |
|------|------|----------|----------|
| 电势 | φ | 静电势 | ∇_⊥²φ = U (Poisson solve) |
| 平行电流 | J_∥ | 沿磁场方向电流 | J_∥ = ∇_⊥²ψ |

**物理参数** (`FiveFieldMHDConfig`, 分支 `mhd_sim-5field_sim`):
```python
eta_i: float = 1.0        # ITG 驱动参数 L_n/L_Ti
beta: float = 0.01        # 等离子体 β
shear: float = 0.1        # 磁剪切 s_hat
lam: float = 1.5          # arcsinh profile shaping λ
mass_ratio: float = 1836  # m_i/m_e (物理值!)
eta: float = 0.0          # 电阻率
v0_amp: float = 0.0       # 剪切流振幅
kq: float = 0.9           # 剪切流波数
# 耗散
Dn: float = 0.01          # 密度扩散
Du: float = 0.01          # 涡度扩散
eta_perp: float = 0.01    # 垂直粘度
chi_perp: float = 0.01    # 热扩散
```

---

## 3. 网络结构详解

### 3.1 OmniFluids2D 网络结构

**文件:** `pretrain/model.py:77-151`

**整体架构:**
```
输入 → in_proj → [SpectralConv2d_dy × n_layers] → fc1a/fc1b → output_mlp → 输出
```

**关键组件:**

#### 3.1.1 输入处理
```python
# model.py:119-126
x = torch.cat((x, self.grid.repeat(batch_size, 1, 1, 1)), dim=-1)  # [B, S, S, 3]
x = torch.cat((x, param), dim=-1)  # [B, S, S, 5]
x = self.in_proj(x)  # Linear(5 → width), [B, S, S, width]
x = F.gelu(x)
nu = param[:, 0, 0, 1:2]  # 提取粘度参数 [B, 1]
```

**输入组成:**
| 通道 | 内容 | 维度 |
|------|------|------|
| 0 | 涡度 w | [B, S, S, 1] |
| 1-2 | 网格坐标 (x, y) | [B, S, S, 2] |
| 3 | 外力 f | [B, S, S, 1] |
| 4 | 粘度 log10(ν) | [B, S, S, 1] |

#### 3.1.2 SpectralConv2d_dy (动态谱卷积层)
```python
# model.py:29-74
class SpectralConv2d_dy(nn.Module):
    def __init__(self, K, in_dim, out_dim, n_modes, ...):
        # K 个算子权重
        self.fourier_weight = nn.ParameterList([
            nn.Parameter(FloatTensor(K, in_dim, out_dim, n_modes, 2))  # 两个方向
            for _ in range(2)
        ])
    
    def forward_fourier(self, x, att):
        # 动态组合 K 个算子
        weight = torch.einsum("bk, kioxy->bioxy", att, self.fourier_weight[0])
        
        # y 方向傅里叶卷积
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft[:, :, :, :n_modes] = torch.einsum("bixy,bioy->boxy", 
                                                  x_fty[:, :, :, :n_modes],
                                                  torch.view_as_complex(weight))
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        
        # x 方向傅里叶卷积 (类似)
        # ...
        return xx + xy
```

#### 3.1.3 动态注意力机制
```python
# model.py:95-99, 127-130
self.f_nu = nn.ModuleList([
    nn.Sequential(nn.Linear(1, 128), nn.GELU(), 
                  nn.Linear(128, 128), nn.GELU(), 
                  nn.Linear(128, K))
    for _ in range(n_layers)
])

# 前向传播时
att = fc(nu)  # nu: [B, 1] → att: [B, K]
att = F.softmax(att/self.T, dim=-1)  # 温度缩放的 softmax
```

**核心创新:** 根据物理参数 (粘度 ν) 动态调整 K 个算子的权重组合。

#### 3.1.4 输出处理
```python
# model.py:135-142
if inference == False:
    x = torch.concat([self.fc1a(x), self.fc1b(x)], dim=-1)  # [B, S, S, 4*output_dim-4+34]
else:
    x = self.fc1b(x)  # [B, S, S, 34]
x = F.gelu(x)
x = self.output_mlp(x.reshape(-1, 1, x.shape[-1]))  # 1D 卷积降维
x = x.reshape(batch_size, size, size, -1)  # [B, S, S, output_dim]

# 时间积分
dt = torch.arange(1, x.shape[-1] + 1, device=x.device).reshape(1, 1, 1, -1) / x.shape[-1]
x = x_o + x * dt  # 残差连接 + 线性时间插值
```

**输出:** `[B, S, S, output_dim]`，预测从 t=0 到 t=rollout_DT 的多个时间帧。

#### 3.1.5 默认超参数
```python
# main.py:66-79
modes = 128        # 傅里叶模式数
width = 80         # 隐藏层宽度
n_layers = 12      # 谱卷积层数
output_dim = 50    # 输出时间帧数
K = 4              # 混合算子数
size = 256         # 空间分辨率
```

---

### 3.2 UNet 网络结构 (mhd_sim)

**文件:** `plasma_sim/models/unet.py:23-99`

**整体架构:**
```
输入 → [Encoder × n_levels] → [Decoder × (n_levels-1)] → out_conv → 输出
```

#### 3.2.1 ConvBlock
```python
# unet.py:10-20
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.act = nn.GELU()
```

#### 3.2.2 编码器-解码器
```python
# unet.py:56-70
channel_mults = (1, 2, 4)  # 每层通道倍数
channels = [base_channels * m for m in channel_mults]  # [64, 128, 256]

# 编码器
for ch in channels:
    self.encoders.append(ConvBlock(in_ch, ch))

# 解码器
for i in range(n_levels - 2, -1, -1):
    dec_in = channels[i + 1] + channels[i]  # skip connection
    self.decoders.append(ConvBlock(dec_in, channels[i]))
```

#### 3.2.3 前向传播
```python
# unet.py:73-99
def forward(self, x, dt=None):
    if self.condition_on_dt:
        dt_ch = dt.reshape(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([x, dt_ch], dim=1)  # 添加 dt 通道
    
    # 编码器
    skips = []
    for i, enc in enumerate(self.encoders):
        h = enc(h)
        if i < len(self.encoders) - 1:
            skips.append(h)
            h = self.pool(h)  # AvgPool2d(2)
    
    # 解码器
    for dec in self.decoders:
        h = F.interpolate(h, scale_factor=2, mode="bilinear")
        h = torch.cat([h, skips.pop()], dim=1)  # skip connection
        h = dec(h)
    
    return self.out_conv(h)
```

#### 3.2.4 默认超参数 (5-field MHD 实际训练设置)
```python
# train_5field.py + scripts/train_5field.sh
arch = "unet"
hidden_channels = 128       # base_channels (比 HW 的 64 更大)
channel_mults = (1, 2, 4, 8)  # 4 层 UNet (也有 (1,2,4) 的实验)
condition_on_dt = True       # 条件化于时间步
predict_mode = "residual"    # 预测残差 x_{t+dt} - x_t
in_channels = 5              # n, U, vpar, psi, Ti
out_channels = 5             # 同上
use_group_norm = False       # 默认不用
zero_init = False            # 默认不用

# Loss
time_integrator = "crank_nicolson"
mse_weight = 1.0, mae_weight = 0.0
unroll_steps = 1, detach_rhs_input = True, unroll_detach = False

# Training (scripts/train_5field.sh)
lr = 1e-4, batch_size = 16, epochs = 1000
model_dtype = "float64", rhs_dtype = "float64"
delta_t = [0.001, 0.01, 0.1]  # 不同实验用不同 delta_t
use_normalizer = False          # --no-use-normalizer
grad_clip = None                # 无梯度裁剪
input_noise_std = 0.0           # 无输入噪声

# 训练数据窗口
data.time_start = 250.0         # 从 t=250 开始取训练数据
data.time_end = 300.0           # 到 t=300，共约 50 个 snapshots (dt_data=1.0)
```

---

## 4. 数据流详解

### 4.1 OmniFluids2D 数据流（MHD5F 版本）

> **注意**：这里描述的是已迁移到 MHD5F 的 `nse2d/pretrain/model.py` 实现，
> 与原始 NSE2D 有以下关键区别：
> - 原始 NSE2D：输入 5 通道 = [涡度(1) + grid(2) + 强迫(1) + log_ν(1)]，参数嵌入在空间场中
> - MHD5F：输入 7 通道 = [5物理场 + grid(2)]，物理参数单独作为向量传入 MoE

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    OmniFluids2D 数据流（MHD5F 版本）                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入（两路分离）:                                                              │
│                                                                              │
│  路径A: 空间场                                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   ┌────────┐ ┌────────┐      │
│  │  n   │ │  U   │ │ vpar │ │ psi  │ │  Ti  │   │ grid_x │ │ grid_y │      │
│  │(B,Nx,│ │(B,Nx,│ │(B,Nx,│ │(B,Nx,│ │(B,Nx,│   │(B,Nx,  │ │(B,Nx,  │      │
│  │Ny,1) │ │Ny,1) │ │Ny,1) │ │Ny,1) │ │Ny,1) │   │Ny,1)   │ │Ny,1)   │      │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘   └───┬────┘ └───┬────┘      │
│     └────────┴────────┴────────┴────────┘           │          │            │
│                        ↓ cat(fields, grid)           │          │            │
│                        └──────────────────────────────┴──────────┘           │
│                                   ↓                                          │
│                    ┌──────────────────────────┐                              │
│                    │  x: (B, Nx, Ny, 7)       │  5场 + 2坐标                  │
│                    └────────────┬─────────────┘                              │
│                                 ↓ in_proj: Linear(7→width=80) + GELU        │
│                    ┌──────────────────────────┐                              │
│                    │  x: (B, Nx, Ny, 80)      │  嵌入空间                     │
│                    └────────────┬─────────────┘                              │
│                                 │                                            │
│  路径B: 物理参数（独立向量）                                                     │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │ params: (B, 8)                                       │                   │
│  │ [eta_i, beta, shear, lam, mass_ratio, Dn, eta, kq]   │                   │
│  └──────────────────────────────────────────────────────┘                   │
│                                 │                                            │
│  ┌──────────────────────────────┼──────────────────────────────────────┐    │
│  │        SpectralConv2d_MHD Layer × n_layers=12                       │    │
│  │                              │                                      │    │
│  │  params(B,8) → f_nu[i]: Linear(8→128→128→K=4) → att(B,4)           │    │
│  │                              ↓ softmax(att/T)                       │    │
│  │                                                                     │    │
│  │  x(B,Nx,Ny,80) → forward_fourier(x, att):                          │    │
│  │    Y方向: FFT → 频域加权(MoE) → iFFT  (周期BC)                        │    │
│  │    X方向: DST → 频域加权(MoE) → iDST  (Dirichlet BC)                 │    │
│  │    xy + xx → (B, Nx, Ny, 80)                                        │    │
│  │    → backcast_ff: Linear(80→320→80) + LayerNorm → b(B,Nx,Ny,80)    │    │
│  │                                                                     │    │
│  │  x = x + b  (残差连接)                                               │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 ↓ GELU(b_last)  ← 注意: 用最后一层的 b        │
│                    ┌──────────────────────────┐                              │
│                    │  x: (B, Nx, Ny, 80)      │                              │
│                    └────────────┬─────────────┘                              │
│                                 │                                            │
│  ┌──────────────────────────────┼──────────────────────────────────────┐    │
│  │        5个独立 OutputHead（每物理场一个）                               │    │
│  │                              │                                      │    │
│  │  训练路径 (inference=False):                                          │    │
│  │    fc_a: Linear(80→36)  ─┐                                          │    │
│  │    fc_b: Linear(80→34)  ─┴→ cat → (B,Nx,Ny,70) → GELU             │    │
│  │    → reshape(-1,1,70) → Conv1d(1,8,12,s=2) → (·,8,30)              │    │
│  │    → Conv1d(8,1,12,s=2) → (·,1,10) → reshape(B,Nx,Ny,10)           │    │
│  │    输出: (B, Nx, Ny, output_dim=10)                                  │    │
│  │                                                                     │    │
│  │  推理路径 (inference=True):                                           │    │
│  │    fc_b: Linear(80→34) → (B,Nx,Ny,34) → GELU                       │    │
│  │    → reshape(-1,1,34) → Conv1d(1,8,12,s=2) → (·,8,12)              │    │
│  │    → Conv1d(8,1,12,s=2) → (·,1,1) → reshape(B,Nx,Ny,1)             │    │
│  │    输出: (B, Nx, Ny, 1)                                              │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 ↓ stack 5个场                                │
│                    ┌──────────────────────────────┐                          │
│                    │ out: (B, Nx, Ny, 5, T)       │  T=10(训练)/T=1(推理)     │
│                    └────────────┬─────────────────┘                          │
│                                 ↓ 残差 + 线性时间插值                           │
│  dt_frac = [1/T, 2/T, ..., 1]  (shape: 1,1,1,1,T)                           │
│  out = x_0.unsqueeze(-1) + out * dt_frac                                     │
│                                 ↓                                            │
│                    ┌──────────────────────────────┐                          │
│                    │ out: (B, Nx, Ny, 5, T)       │  最终输出                  │
│                    └──────────────────────────────┘                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Shape 变化总结（MHD5F）:**
| 阶段 | Shape | 物理意义 |
|------|-------|----------|
| 输入物理场 | (B, Nx=512, Ny=256, 5) | n, U, vpar, psi, Ti |
| 拼接 grid 后 | (B, 512, 256, **7**) | 5场 + grid_x + grid_y |
| in_proj 后 | (B, 512, 256, 80) | 嵌入空间（width=80） |
| 谱卷积层×12 | (B, 512, 256, 80) | 保持维度，MoE 动态加权 |
| 物理参数路径 | (B, 8) → att (B, 4) | 8维参数 → K=4 注意力权重 |
| OutputHead×5 | (B, 512, 256, 10) 各 | 每场独立输出 10 帧 |
| stack 后 | (B, 512, 256, 5, 10) | 5场 × 10帧 |
| 残差后（最终） | (B, 512, 256, 5, **10**) 训练 | 含残差的多帧预测 |
| 残差后（最终） | (B, 512, 256, 5, **1**) 推理 | 单帧预测 |

**与原始 NSE2D 的关键差异：**
| 维度 | 原始 NSE2D | MHD5F 版本 |
|------|-----------|------------|
| 输入通道 | 5 = [w(1) + grid(2) + f(1) + log_ν(1)] | **7 = [fields(5) + grid(2)]** |
| 参数传入方式 | 嵌入在空间场中（与 field 拼接） | **独立向量 (B, 8)，仅进 MoE** |
| f_nu 输入维度 | Linear(**1**→128→K)，输入标量 ν | Linear(**8**→128→K)，输入8维参数 |
| 输出场数 | 1（涡度） | **5（n, U, vpar, psi, Ti）** |
| 输出头 | 1个共享头 | **5个独立头** |
| 输出 shape | (B, S, S, output_dim) | **(B, Nx, Ny, 5, output_dim)** |
| 空间分辨率 | S×S（正方形） | **Nx×Ny = 512×256（非正方形）** |

---

### 4.2 mhd_sim UNet 数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          UNet (mhd_sim) 数据流                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入:                                                                        │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │  x: [B, C, H, W] │    │  dt: [B]        │                                  │
│  │  C=2 (HW) / 5(MHD)│    │  时间步长       │                                  │
│  └────────┬────────┘    └────────┬────────┘                                  │
│           │                      │                                           │
│           │    ┌─────────────────┘                                           │
│           ↓    ↓ (if condition_on_dt)                                        │
│  ┌─────────────────────────────────┐                                         │
│  │ x_aug: [B, C+1, H, W]           │  dt 作为额外通道                          │
│  └──────────────┬──────────────────┘                                         │
│                 │                                                            │
│  ┌──────────────┼──────────────────────────────────────────┐                │
│  │    ENCODER   │                                          │                │
│  │              ↓                                          │                │
│  │  ┌─────────────────┐  skip₀                             │                │
│  │  │ enc₀: [B,64,H,W]│────────────────────────────┐       │                │
│  │  └────────┬────────┘                            │       │                │
│  │           ↓ AvgPool(2)                          │       │                │
│  │  ┌─────────────────────┐  skip₁                 │       │                │
│  │  │enc₁: [B,128,H/2,W/2]│──────────────┐         │       │                │
│  │  └────────┬────────────┘              │         │       │                │
│  │           ↓ AvgPool(2)                │         │       │                │
│  │  ┌─────────────────────┐              │         │       │                │
│  │  │enc₂: [B,256,H/4,W/4]│ (bottleneck) │         │       │                │
│  │  └────────┬────────────┘              │         │       │                │
│  └───────────┼───────────────────────────┼─────────┼───────┘                │
│              │                           │         │                        │
│  ┌───────────┼───────────────────────────┼─────────┼───────┐                │
│  │    DECODER│                           │         │       │                │
│  │           ↓ Upsample(2)               │         │       │                │
│  │  ┌─────────────────────────────────┐  │         │       │                │
│  │  │ cat: [B, 256+128, H/2, W/2]     │←─┘         │       │                │
│  │  └────────┬────────────────────────┘            │       │                │
│  │           ↓ dec₀                                │       │                │
│  │  ┌─────────────────────┐                        │       │                │
│  │  │dec₀: [B,128,H/2,W/2]│                        │       │                │
│  │  └────────┬────────────┘                        │       │                │
│  │           ↓ Upsample(2)                         │       │                │
│  │  ┌─────────────────────────────────┐            │       │                │
│  │  │ cat: [B, 128+64, H, W]          │←───────────┘       │                │
│  │  └────────┬────────────────────────┘                    │                │
│  │           ↓ dec₁                                        │                │
│  │  ┌─────────────────┐                                    │                │
│  │  │dec₁: [B, 64,H,W]│                                    │                │
│  │  └────────┬────────┘                                    │                │
│  └───────────┼─────────────────────────────────────────────┘                │
│              ↓ out_conv (Conv2d 1×1)                                        │
│  ┌─────────────────────┐                                                    │
│  │ raw: [B, C, H, W]   │  预测 (残差或状态)                                   │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│  if predict_mode == "residual":                                             │
│      output = x + raw                                                       │
│  else:                                                                      │
│      output = raw                                                           │
│             ↓                                                               │
│  ┌─────────────────────┐                                                    │
│  │ output: [B, C, H, W]│  预测的下一时刻状态                                  │
│  └─────────────────────┘                                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 损失函数构建

### 5.1 OmniFluids PSM Loss (物理结构匹配)

**文件:** `pretrain/psm_loss.py:51-64`

**原理:** 确保预测的轨迹满足物理方程残差最小。

**公式 (Crank-Nicolson 格式):**
```
wₜ = (w[t+1] - w[t]) / Δt
Du = u·∇w - ν∇²w
Loss = RMSE(wₜ + (Du[t] + Du[t+1])/2 - f)
```

**代码实现:**
```python
# psm_loss.py:51-64
def PSM_loss(u, forcing, v, t_interval=0.50, loss_mode='cn'):
    Du = PSM_NS_vorticity(u, v, t_interval, loss_mode)
    if loss_mode == 'cn':
        forcing = forcing.reshape(-1, nx, ny, 1)
        f = forcing.repeat(1, 1, 1, nt-1)
    return (torch.square(Du - f).mean() + EPS).sqrt()  # RMSE

# psm_loss.py:7-48
def PSM_NS_vorticity(w, v, t_interval=5.0, loss_mode='cn'):
    # 在频域计算所有空间导数
    w_h = torch.fft.fft2(w, dim=[1, 2])
    # ... (计算 ux, uy, wx, wy, wlap)
    
    if loss_mode == 'cn':  # Crank-Nicolson
        wt = (w[:, :, :, 1:] - w[:, :, :, :-1]) / dt
        Du = (ux*wx + uy*wy - v*wlap)
        Du1 = wt + (Du[..., :-1] + Du[..., 1:]) * 0.5  # 时间平均
    if loss_mode == 'mid':  # 中点格式
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
        Du1 = wt + (ux*wx + uy*wy - v*wlap)[..., 1:-1]
    return Du1
```

**关键特点:**
1. **纯物理监督:** 不需要真实数据标签
2. **频域计算:** 高效且精确的空间导数
3. **多帧约束:** 约束整条预测轨迹

---

### 5.2 mhd_sim Physics Loss (时间差分, 通用版)

**文件:** `plasma_sim/train/train_utils.py:357-431`

**原理:** 网络预测应满足时间积分方案。使用通用 `rhs_fn` 接口，HW 和 5-field MHD 共用同一 loss 函数。

**5-field MHD 的 RHS adapter** (`train_5field.py:119-138`):
```python
def make_mhd5_rhs_fn(mhd, rhs_dtype, model_dtype):
    def rhs_fn(x):  # x: (B, C=5, H, W)
        x_rhs = x.to(rhs_dtype)
        state = tuple(x_rhs[:, i] for i in range(5))  # 拆成 5 个 (B,H,W)
        rhs_tuple = mhd.compute_rhs(state)             # 调用物理方程
        return torch.stack(rhs_tuple, dim=1).to(model_dtype)  # → (B,5,H,W)
    return rhs_fn
```

**核心 loss 函数** (`compute_unrolled_physics_loss`):
```python
# 支持 unroll_steps > 1 的多步训练
for _ in range(unroll_steps):
    prediction = model_predict(model, current, delta_t, ...)
    time_diff = (prediction - current) / delta_t

    if integrator == "euler":         target = rhs_fn(current)
    elif integrator == "crank_nicolson": target = (rhs_fn(current) + rhs_fn(prediction)) / 2
    elif integrator == "rk4":
        k1 = rhs_fn(current); k2 = rhs_fn(current + dt/2*k1)
        k3 = rhs_fn(current + dt/2*k2); k4 = rhs_fn(current + dt*k3)
        target = (k1 + 2*k2 + 2*k3 + k4) / 6
    elif integrator == "implicit_rk4":  # Simpson's rule
        target = (rhs_fn(current) + 4*rhs_fn((current+prediction)/2) + rhs_fn(prediction)) / 6

    if detach_rhs_input: target = target.detach()
    loss += mixed_loss(time_diff, target, mse_w, mae_w)
    current = prediction.detach() if unroll_detach else prediction
```

**关键特点:**
1. **通用 rhs_fn 接口:** `(B, C, H, W) -> (B, C, H, W)`，方程无关
2. **多种积分方案:** euler / crank_nicolson / implicit_euler / rk4 / implicit_rk4
3. **unroll 训练:** 可展开多步，梯度流过整条自回归链
4. **detach 控制:** `detach_rhs_input` (RHS 不参与反传), `unroll_detach` (步间截断梯度)

---

### 5.3 损失函数对比

| 特性 | OmniFluids PSM | mhd_sim Physics |
|------|----------------|-----------------|
| 监督类型 | 纯物理无标签 | 物理信息有/无标签 |
| 预测步数 | 多帧同时预测 | 单帧迭代预测 |
| 时间格式 | CN / 中点 | Euler / CN / RK4 |
| RHS 计算 | 内置于 loss | 调用外部方程 |
| 梯度流 | 通过整条轨迹 | 可配置 detach |

---

## 6. 训练设置对比

### 6.1 OmniFluids 预训练设置

**文件:** `pretrain/main.py`, `pretrain/train.py`

```python
# 默认超参数
lr = 0.002                    # 学习率
weight_decay = 0.0            # 权重衰减
num_iterations = 20000        # 训练迭代次数
batch_size = 10               # 批大小
val_size = 10                 # 验证集大小

rollout_DT = 0.2              # 每次预测的时间跨度
loss_mode = 'cn'              # Crank-Nicolson 格式

# 优化器和调度器
optimizer = optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                 total_steps=num_iterations+1, 
                                                 max_lr=lr)

# 训练循环
for step in range(num_iterations+1):
    # 在线生成数据
    w0_train = GRF(batch_size)[..., None]
    f_train = F_Sampler(batch_size)[..., None]
    nu_train = -lognu_min - (lognu_max - lognu_min) * torch.rand(batch_size)
    
    # 前向 + loss
    w_pre = net(w0_train, param)
    w_pre = torch.concat([w0_train, w_pre], dim=-1)
    loss = PSM_loss(w_pre, f_train, 10**nu_train, rollout_DT, loss_mode)
    
    # 每 100 步验证
    if step % 100 == 0:
        val_error = val(config, net, w0_val, val_param)
        if val_error < best_val_error:
            torch.save(net.state_dict(), f'model/{model_name}.pt')
```

### 6.2 mhd_sim 5-field MHD 训练设置

**文件:** `plasma_sim/train/train_5field.py`, `scripts/train_5field.sh`

```python
# 训练超参数 (5-field MHD)
lr = 1e-4                          # 学习率
weight_decay = 0.0                 # 权重衰减
epochs = 1000                      # 训练轮数
batch_size = 16                    # 批大小
model_dtype = "float64"            # 模型精度 (fp64!)
rhs_dtype = "float64"              # RHS 计算精度

# 模型
arch = "unet"
hidden_channels = 128              # UNet base channels
channel_mults = (1, 2, 4, 8)      # 4 层
condition_on_dt = True
predict_mode = "residual"

# Loss
delta_t = 0.001 ~ 0.1             # 多种实验
time_integrator = "crank_nicolson"
mse_weight = 1.0, mae_weight = 0.0
unroll_steps = 1
detach_rhs_input = True            # RHS 输入不参与反传

# 数据
data.time_start = 250.0            # 训练时间窗口起始
data.time_end = 300.0              # 训练时间窗口结束 (约 50 个 snapshots)
use_normalizer = False             # 默认不用

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 训练循环
for epoch in range(epochs):
    for x_t in loader:     # x_t: (B, C=5, Nx, Ny) snapshot
        x_t += input_noise_std * randn_like(x_t)  # 可选噪声
        loss = compute_unrolled_physics_loss(model, rhs_fn, x_t, delta_t, cfg)
        optimizer.zero_grad(); loss.backward()
        if grad_clip: clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    # Evaluation: autoregressive rollout → rel L2 error
    # Checkpointing: save model state dict
```

### 6.3 训练设置对比

| 设置 | OmniFluids nse2d | mhd_sim 5-field |
|------|------------------|-----------------|
| 优化器 | Adam | AdamW |
| 学习率 | 0.002 | 1e-4 |
| 调度器 | OneCycleLR | 无 |
| 迭代方式 | iterations (在线) | epochs (离线数据集) |
| 数据 | 在线随机生成 GRF | 预生成 trajectory |
| 精度 | float32 | **float64** |
| 模型 | SpectralConv (FNO-like) | UNet (CNN) |
| 预测方式 | **多帧** (output_dim=50) | **单帧** (residual) |
| loss | PSM (频域方程残差) | Physics time-differencing |
| dt conditioning | 无 (固定 dt) | 有 (dt 作为输入通道) |
| 验证 | 每100步 | 每 N epochs |
| 字段数 | 1 (w) | **5** (n, U, vpar, psi, Ti) |

### 6.4 实际实现设置对比 (OmniFluids 5-field 修改版 vs mhd_sim baseline)

> **此表是最终实现状态的对比，非原始 nse2d 或 mhd_sim 的默认值。**

| 设置 | OmniFluids (修改后) | mhd_sim baseline | 对齐? | 说明 |
|------|---------------------|------------------|-------|------|
| **方程** | 5-field Landau-fluid (调用 mhd_sim compute_rhs) | 5-field Landau-fluid | ✅ | 相同 RHS 函数 |
| **网格** | Nx=512, Ny=256 | Nx=512, Ny=256 | ✅ | |
| **数据时间窗口** | [250.0, 300.0] | [250.0, 300.0] | ✅ | 来源: `train_5field.sh --data.time-start 250 --data.time-end 300` |
| **dt_data** | 1.0 | 1.0 | ✅ | = dt_sim(2e-3) × stamp_interval(500) |
| **训练数据** | 同一份 5field_mhd_dataset.pt | 同一份 | ✅ | |
| **Time integrator (loss)** | Crank-Nicolson | Crank-Nicolson | ✅ | |
| **Loss 类型** | MSE(time_diff, target) | MSE(time_diff, target) | ✅ | mhd_sim: mse_weight=1.0, mae_weight=0.0 |
| **detach_rhs_input** | ✅ target.detach() | ✅ detach_rhs_input=True | ✅ | 梯度只经过 time_diff, 不经过 RHS target |
| **normalizer** | 无 | 无 (--no-use-normalizer) | ✅ | |
| **input noise** | 0.0 | 0.0 | ✅ | |
| **grad clip** | None | None | ✅ | |
| **评估指标** | per-step rel L2, per-field rel L2 | per-step rel L2, correlation, energy | ✅ | 核心指标一致 |
| **评估方式** | autoregressive rollout | autoregressive rollout | ✅ | |
| **─── 以下为有意差异 ───** | | | | |
| **网络架构** | SpectralConv (DST-x, FFT-y) + MoE | UNet (CNN, circular pad) | ⚡ | 这是对比的核心 |
| **x 方向 BC** | DST (正确 Dirichlet) | circular padding (周期,不匹配!) | ⚡ | OmniFluids 更物理准确 |
| **输入格式** | channel-last (B, Nx, Ny, 5) | channel-first (B, 5, Nx, Ny) | ⚡ | 架构选择 |
| **预测方式** | 训练多帧(10), 推理单帧(dt=0.1) | 单帧 (dt=0.1, residual) | ⚡ | 推理dt一致, 训练时OmniFluids有更细的约束 |
| **模型精度** | float32 (RHS 内部 float64) | float64 | ⚡ | 用户选择, 后续可改 |
| **优化器** | Adam, lr=0.002 | AdamW, lr=1e-4 | ⚡ | 不同架构需要不同 lr |
| **调度器** | OneCycleLR | 无 | ⚡ | OmniFluids 原有设计 |
| **模型参数量** | **158.2M** | **31.1M** | ⚡ | OmniFluids 5.1x 更大 (谱卷积权重主导) |
| **模型参数显存** | 0.60 GB (fp32) | 0.23 GB (fp64) | ⚡ | |
| **优化器状态显存** | 1.20 GB | 0.46 GB | ⚡ | Adam/AdamW momentum+variance |
| **静态显存合计** | ~2.4 GB | ~0.9 GB | ⚡ | params + gradients + optimizer |
| **每样本激活显存** | ~5.5 GB (fp32) | ~0.8 GB (fp64, UNet 下采样) | ⚡ | OmniFluids: 12层谱卷积全分辨率; UNet: 4级下采样 |
| **batch_size** | 4 | 16 | ⚡ | OmniFluids 显存瓶颈: 谱卷积12层×全分辨率激活 |
| **估算总峰值显存** | ~25 GB (bs=4) | ~14 GB (bs=16) | ⚡ | |
| **训练量** | 200000 iterations | 1000 epochs ≈ 3000 iters | ⚡ | 待实验调整 |

**dt 体系对齐详解 (⚠️ 关键设计):**

mhd_sim 的三层 dt:
```
dt_sim = 2e-3              (仿真时间步长)
delta_t = 0.1              (神经网络 dt, 模型预测一步覆盖的时间)
dt_data = 1.0              (数据快照间隔 = dt_sim × stamp_interval)
```

OmniFluids 的三层 dt (与 mhd_sim 对齐):
```
rollout_dt = 0.1           (= mhd_sim delta_t, 推理时每步覆盖的时间)
train_dt = 0.1 / 10 = 0.01 (= rollout_dt / output_dim, 训练物理loss的dt)
dt_data = 1.0              (= mhd_sim dt_data, 数据快照间隔)
n_substeps = 10            (= dt_data / rollout_dt, 推进一个数据步需要的推理次数)
```

| 量 | OmniFluids | mhd_sim (delta_t=0.1) | 对齐 |
|----|-----------|----------------------|------|
| 推理 dt (每次 forward) | rollout_dt = **0.1** | delta_t = **0.1** | ✅ |
| 训练 physics loss dt | train_dt = **0.01** | delta_t = **0.1** | ⚡ 更细 |
| 推进 dt_data=1.0 的 NFE | **10** 次 forward | **10** 次 forward | ✅ |
| 50 步评估总 NFE | **500** 次 | **500** 次 | ✅ |
| 训练每次 forward 的约束数 | **10** (output_dim) | **1** (unroll_steps=1) | ⚡ |

**梯度流对比:**
```
mhd_sim:  ∂loss/∂θ ← ∂[(pred-x)/dt - target.detach()]/∂θ
                    ← ∂pred/∂θ × (1/dt)
                    (梯度经过 prediction, 不经过 RHS target)

OmniFluids: ∂loss/∂θ ← ∂[(pred[t+1]-pred[t])/dt - target.detach()]/∂θ
                      ← ∂pred/∂θ × (1/dt)
                      (梯度经过 multi-frame predictions, 不经过 RHS target)
                      效果等价: 都是只通过时间差分传梯度
```

### 6.5 多帧输出 (T=10 训练 vs T=1 推理) 详细解释

#### 核心机制：OutputHead 双通路设计

OmniFluids 的每个 OutputHead 内部有 **训练通路** 和 **推理通路**，共享部分权重：

```
训练通路 (inference=False):
  spectral特征: (B, Nx, Ny, width=80)
       ↓
  fc_a(80 → 36) ─┐
  fc_b(80 → 34) ─┤ ← fc_b 权重被两条通路共享!
       ↓          
  concat: (B, Nx, Ny, 70)     [36+34=70]
       ↓ GELU
  Conv1d(1→8, k=12, s=2): 70→30
       ↓ GELU
  Conv1d(8→1, k=12, s=2): 30→10   ← output_dim = 10 帧
       ↓
  输出: (B, Nx, Ny, 10)        每场 10 帧

推理通路 (inference=True):
  spectral特征: (B, Nx, Ny, width=80)
       ↓
  fc_b(80 → 34) 仅用fc_b   ← 同一组 fc_b 权重!
       ↓ GELU
  Conv1d(1→8, k=12, s=2): 34→12   ← 同一组 Conv1d 权重!
       ↓ GELU
  Conv1d(8→1, k=12, s=2): 12→1    ← 直接输出 1 帧
       ↓
  输出: (B, Nx, Ny, 1)         每场 1 帧
```

**关键**: `fc_b` 和 `Conv1d` 的权重在两条通路之间**完全共享**。
Conv1d 是卷积层，对不同长度的输入自然产生不同长度的输出（输入 70→输出 10, 输入 34→输出 1）。

#### 残差连接 + 线性时间插值

```python
# 5个场的输出 stack → (B, Nx, Ny, 5, T)

# 线性时间插值因子
# 训练: dt_frac = [1/10, 2/10, ..., 10/10] = [0.1, 0.2, ..., 1.0]
# 推理: dt_frac = [1/1] = [1.0]

output = x_0 + pred * dt_frac
```

- **训练 T=10**: 第 k 帧 = `x_0 + pred_k × (k/10)`
  - 第 1 帧 ≈ `x_0 + 小扰动` （接近初始状态）
  - 第 10 帧 ≈ `x_0 + pred_10 × 1.0` （完整的 rollout_dt=0.1 步预测）
  - 物理含义：10 帧是从 t 到 t+rollout_dt 之间的**等间距中间状态**
  - 每帧间隔 train_dt = rollout_dt/output_dim = 0.1/10 = **0.01**

- **推理 T=1**: 唯一帧 = `x_0 + pred_1 × 1.0`
  - 预测 t+rollout_dt = t+0.1 的状态
  - 要推进 dt_data=1.0，需要链式调用 n_substeps=10 次

#### 为什么这样设计？训练 vs 推理的联系

```
                         训练阶段                                    推理阶段
  ┌─────────────────────────────────────────┐       ┌──────────────────────────────────────┐
  │  x_0 ──→ 网络 ──→ 10帧                  │       │  x_0 ──→ 网络 ──→ 1帧                │
  │           (fc_a + fc_b → Conv1d)         │       │        (fc_b → Conv1d)               │
  │                                         │       │                                      │
  │  帧0  帧1  帧2  ...  帧9  帧10           │       │  帧1 (= t+0.1 的预测)                │
  │  ↕    ↕    ↕         ↕    ↕             │       │       ↓ (链式调用 10 次)               │
  │  物理loss 物理loss ... 物理loss          │       │  t → t+0.1 → t+0.2 → ... → t+1.0   │
  │  (10个约束, train_dt=0.01 each)          │       │  (10次 forward = mhd_sim 的 NFE)     │
  └─────────────────────────────────────────┘       └──────────────────────────────────────┘
                    │                                              ↑
                    │  fc_b 和 Conv1d 权重共享                       │
                    └──────────────────────────────────────────────┘

  训练时: 10个物理约束 (train_dt=0.01) → 迫使 fc_b 学到精细的动力学编码
  推理时: fc_b 编码 → 每步预测 rollout_dt=0.1 → 链式 10 步 = dt_data=1.0
  NFE 对齐: 推进 1.0 时间 = 10次 forward，与 mhd_sim (delta_t=0.1) 完全一致!
```

**本质**: 训练是"**监督放大器**" — 同一个 forward pass 提供 10 倍的物理约束（train_dt=0.01 比 mhd_sim 的 delta_t=0.1 更细），
迫使共享权重 (fc_b, Conv1d) 学到更准确的动力学表示。推理时，模型的 dt 与 mhd_sim 完全一致，NFE 相同，公平比较。

#### 对比 mhd_sim 的单帧预测

| 维度 | OmniFluids (多帧训练) | mhd_sim (单帧) |
|------|----------------------|----------------|
| **推理 dt (每次 forward)** | **0.1** (= rollout_dt) | **0.1** (= delta_t) |
| **推理 NFE (每 dt_data=1.0)** | **10 次** forward | **10 次** forward |
| **推理 NFE 对齐** | ✅ **完全一致** | ✅ |
| **训练 physics dt** | **0.01** (= 0.1/10, 更细!) | 0.1 (= delta_t) |
| **训练每次 forward 约束数** | 10 个 (output_dim=10) | 1 个 (unroll_steps=1) |
| **训练优势** | 同 NFE 下更多物理约束 | — |

### 6.6 评估方式对齐 (⚠️ 重要)

#### 自回归 Rollout 对比 (⚠️ NFE 完全一致!)

两种方法在相同 GT 时间点 (t=250, 251, ..., 300) 评估，**推理方式完全对等**:

```
mhd_sim (delta_t=0.1):
  t=250 ─[0.1]→ ─[0.1]→ ... ─[0.1]→ t=251 ─[0.1]→ ... ─[0.1]→ t=252 ─ ...
         ╰───── 10次 forward ──────╯        ╰─── 10次 forward ───╯
  每 dt_data=1.0: 10次 forward, 共 50 数据步 → 500次 forward

OmniFluids (rollout_dt=0.1, n_substeps=10):
  t=250 ─[0.1]→ ─[0.1]→ ... ─[0.1]→ t=251 ─[0.1]→ ... ─[0.1]→ t=252 ─ ...
         ╰───── 10次 forward ──────╯        ╰─── 10次 forward ───╯
  每 dt_data=1.0: 10次 forward, 共 50 数据步 → 500次 forward

  两者 NFE 完全相同!
  唯一区别: 网络架构不同 (SpectralConv vs UNet)
```

#### 评估步数对齐

| 评估场景 | OmniFluids | mhd_sim | 对齐 |
|---------|-----------|---------|------|
| **推理 dt** | rollout_dt = 0.1 | delta_t = 0.1 | ✅ |
| **每数据步 substeps** | n_substeps = 10 | n_substeps = 10 | ✅ |
| **训练中快速评估** | 10 数据步 × 10 substeps = 100 NFE | 未启用 | — |
| **最终评估** | 50 数据步 × 10 substeps = **500 NFE** | 50 数据步 × 10 substeps = **500 NFE** | ✅ |
| **评估数据窗口** | [250, 300] | [250, 300] | ✅ |
| **评估指标** | per-step rel L2, per-field rel L2 | per-step rel L2, correlation, energy | ✅ 核心一致 |
| **GT 比较时间点** | t=251,252,...,300 (每 1.0) | t=251,252,...,300 (每 1.0) | ✅ |

---

## 7. 仿真设置

### 7.1 nse2d 仿真设置

**文件:** `data/generate_data.py`, `data/nse.py`

```python
# 默认仿真参数
s = 1024                      # 原始分辨率
sub = 4                       # 下采样因子 → 实际 256×256
T = 10.0                      # 总仿真时间
dt = 1e-4                     # 时间步长
record_ratio = 10             # 记录间隔 (record_steps = T * record_ratio)

# Reynolds 数范围 (对应粘度范围)
re = (500, 2500)              # 对于 _multi 数据集
# ν = 1/Re, 所以 lognu_min = log10(1/2500), lognu_max = log10(1/500)

# 外力参数
max_frequency = 4             # 外力最大频率
amplitude = 0.1               # 外力振幅范围 [-0.1, 0.1]

# 数值方法
method = "Crank-Nicolson"     # 时间积分
spatial = "FFT"               # 频域空间离散
dealiasing = "2/3 rule"       # 去混叠

# 数据格式
data shape: [N, H, W, T]      # N 样本，H×W 空间，T 时间帧
param shape: [N, H, W, 2]     # f 和 ν
```

### 7.2 5-field MHD 仿真设置 (⚠️ 已基于 mhd_sim-5field_sim 分支修正)

**文件:** `numerical/equations/five_field_mhd.py`, `numerical/scripts/run_5field_mhd_batch.py`

**关键: 这是 2D 系统！** 不是 3D！

```python
# FiveFieldMHDConfig 默认参数

# 网格参数 (2D!)
Nx = 512                      # x 方向网格点 (radial, Dirichlet BC)
Ny = 256                      # y 方向网格点 (binormal, periodic BC)
Lx = 100.0                    # x 方向物理长度 (rho_i)
Ly = 20.0 * pi                # y 方向物理长度 (rho_i)
M = 48                        # dealiasing 保留 Fourier 模式数

# 场 shape: (..., Nx, Ny) — 支持 batch 维度

# 时间参数
dt = 2e-3                     # 仿真时间步长
Nt = 500 * 2000 = 1_000_000   # 总步数 (batch 模式默认)
stamp_interval = 500           # 每 500 步保存一帧
# → dt_data = 2e-3 * 500 = 1.0 (连续 snapshot 间隔)

# 物理参数
eta_i = 1.0                   # ITG 驱动 (L_n / L_Ti)
beta = 0.01                   # 等离子体 beta
shear = 0.1                   # 磁剪切 s_hat
lam = 1.5                     # arcsinh 参数 lambda
mass_ratio = 1836.0           # m_i/m_e (物理值)
eta = 0.0                     # 电阻率
v0_amp = 0.0                  # 剪切流振幅 (用户设置)
kq = 0.9                      # 剪切流波数

# 耗散系数
Dn = 0.01; Du = 0.01; eta_perp = 0.01; chi_perp = 0.01

# 数值方法
method = "RK4"                # 时间积分
x方向 = "有限差分 (FD)"        # Dirichlet BC: zero-pad + central difference
y方向 = "伪谱 (rfft/irfft)"   # Periodic BC: rfft → iky → irfft
Poisson solver = "rfft(y) + tridiag solve(x)"  # 混合 BC
Arakawa = "arakawa_scheme_2d_padx"  # Poisson bracket 保守格式

# 数据生成: run_5field_mhd_batch.py
# 支持多 GPU 并行生成, 每个 sample 不同 seed
# 输出: {"n": (B,T,Nx,Ny), "U": ..., "vpar": ..., "psi": ..., "Ti": ..., "phi": ...}
```

### 7.3 仿真设置对比

| 设置 | nse2d | 5-field MHD |
|------|-------|-------------|
| **维度** | **2D** | **2D** (Nx, Ny) |
| 分辨率 | 256×256 (方形) | **512×256** (矩形) |
| dt_sim | 1e-4 | **2e-3** |
| dt_data | 取决于 record_ratio | **1.0** (= 2e-3 × 500) |
| 总时间 | T=10 | T=1000 (默认 Nt×dt) |
| 时间积分 | Crank-Nicolson | **RK4** |
| x 方向 BC | **周期** | **Dirichlet** (零边界) |
| y 方向 BC | 周期 | 周期 |
| x 方向离散 | FFT | **有限差分** (zero-pad) |
| y 方向离散 | FFT | **rfft/irfft** |
| 字段数 | 1 (w) | **5** (n, U, vpar, psi, Ti) |
| 非线性项 | u·∇w | **[φ, f] Arakawa** |
| 参数范围 | Re: 500-2500 (随机) | 固定参数 |
| 精度 | float32 | **float64** |

---

## 8. 评估与可视化

### 8.1 OmniFluids 评估

**文件:** `pretrain/train.py:20-44`, `finetune/train.py:19-43`

**评估指标:** Relative L2 Error

```python
def test(config, net, test_data, test_param, test_data_dict):
    # 自回归预测
    w_pre = w_0
    for _ in range(total_iter):
        w_0 = net(w_pre[..., -1:], test_param)[..., -1:]
        w_pre = torch.concat([w_pre, w_0], dim=-1)
    
    # 计算每步相对 L2 误差
    for time_step in range(1, total_iter+1):
        w = w_pre[..., time_step]
        w_t = test_data[..., sub * time_step]
        rela_err = (torch.norm((w-w_t).reshape(B,-1), dim=1) / 
                    torch.norm(w_t.reshape(B,-1), dim=1)).mean()
    
    return np.mean(rela_err)
```

### 8.2 mhd_sim 评估

**文件:** `plasma_sim/train/eval_hw.py`

**评估指标:**
1. Relative L2 Error (每步)
2. Pearson Correlation (每步)
3. Physics Diagnostics (粒子通量、能量)
4. Energy Spectrum Deviation

```python
def evaluate(cfg):
    # 自回归预测
    pred_traj = autoregressive_rollout(model, traj[:, 0], n_steps, ...)
    
    # Relative L2 Error
    for t in range(1, n_steps + 1):
        diff_norm = torch.norm((pred_traj[:, t] - gt_traj[:, t]).reshape(B, -1), dim=1)
        true_norm = torch.norm(gt_traj[:, t].reshape(B, -1), dim=1)
        rel_l2_errors.append((diff_norm / true_norm).mean())
    
    # Correlation
    correlations = compute_per_step_correlation(pred_traj, gt_traj)
    
    # Physics diagnostics
    pred_diag = compute_hw_diagnostics(pred_traj, spectral)
    
    # Energy spectrum
    for t in range(1, n_steps + 1):
        _, _, _, E_tot_gt = compute_hw_energy_spectrum(gt_state, Lx, Ly)
        _, _, _, E_tot_pr = compute_hw_energy_spectrum(pr_state, Lx, Ly)
        spec_dev = abs(log(E_pr) - log(E_gt)) / abs(log(E_gt))
```

### 8.3 可视化 (已实现)

**mhd_sim 5-field 可视化/诊断** (`eval_5field.py`):
- `_plot_perstep_metrics()`: per-step rel L2 error + correlation (2 subplots)
- `_plot_perfield_rel_l2()`: 5 条曲线, 每场 rel L2 vs physical time
- `_plot_snapshots()`: GT vs Pred 2D imshow, 每场单独一张图, 选取 5 个等间距时间点
- `_plot_snapshots(residual=True)`: 同上, 但显示 Δf = f(t) - f(t-1) 残差
- `save_eval_plots()`: 统一入口, 生成全部图

**OmniFluids 5-field 可视化** (`pretrain/train.py`, 已对齐 mhd_sim):
- `plot_perstep_metrics()`: per-step total rel L2 + per-field rel L2 (两张图)
- `plot_snapshots()`: GT vs Pred 2D imshow, 每场一张图, 5 个等间距时间点, `RdBu_r` colormap
- `save_eval_plots()`: 统一入口
- 训练中每 `eval_every * 5` 步自动生成可视化 (避免过于频繁)
- 最终评估 (`step_tag='final'`) 始终生成完整可视化

**输出目录结构:**
```
results/{exp_name}/
  step_1000/        # 训练中间评估
    perstep_rel_l2.png
    perfield_rel_l2.png
    snapshot_n.png
    snapshot_U.png
    snapshot_vpar.png
    snapshot_psi.png
    snapshot_Ti.png
  final/            # 最终评估
    (同上)
  eval_results.json
```

`mhd5_diagnostics.py` 能量计算:
```python
E_kin = 0.5 * <|grad_perp phi|^2>   # phi 由 Poisson 求解 U
E_mag = (beta/2) * <|nabla_perp psi|^2>
E_th  = 0.5 * <Ti^2>
E_tot = E_kin + E_mag + E_th
```

---

## 9. 关键差异总结 (⚠️ 已修正)

| 方面 | OmniFluids (nse2d) | mhd_sim (5-field MHD) |
|------|--------------------|-----------------------|
| **方程** | 2D NS (涡度形式) | **2D 5-field Landau-fluid** |
| **物理场数** | 1 (涡度) + 2 参数 | **5 主场** (n,U,vpar,psi,Ti) + 辅助场 |
| **维度** | **2D** | **2D** (Nx, Ny) |
| **分辨率** | 256×256 (方形) | **512×256** (矩形) |
| **网络** | OmniFluids2D (谱卷积 + MoE) | UNet (CNN) |
| **输入格式** | [B, H, W, C] (channels last) | [B, C, H, W] (channels first) |
| **输出** | **多帧** [B, H, W, T] | **单帧** [B, C, H, W] (residual) |
| **损失** | PSM (频域方程残差) | Physics time-differencing |
| **时间积分** | CN / mid | euler / CN / RK4 / implicit |
| **参数条件** | 粘度 ν → MoE | dt → extra channel |
| **数据** | 在线随机 GRF | 离线 trajectory 数据集 |
| **x 方向 BC** | **周期 (FFT)** | **Dirichlet (FD zero-pad)** |
| **y 方向 BC** | 周期 (FFT) | 周期 (rfft) |
| **精度** | float32 | **float64** |
| **边界条件** | 双周期 | 混合边界 |

---

## 10. 迁移计划

### 重要原则 (⚠️ 必读)

> **原则 1: 不修改 mhd_sim 代码！** 
> - mhd_sim 代码库保持不变，作为 baseline 方法
> - 通过 **导入/调用** mhd_sim 的功能（RHS计算、评估函数等），而不是修改它
> - 这样可以保证公平比较：相同数据、相同评估、不同方法

> **原则 1.5: 直接在 nse2d 代码上修改**
> - nse2d 已有备份 (`nse2d_old/`)，直接在 `nse2d/` 上修改，方便对照
> - 不创建新的 `mhd5f/` 目录

> **原则 2: 渐进式实现，先用 mhd_sim 的 loss**
> - **第一阶段**：保留 OmniFluids 网络架构，**直接调用 mhd_sim 的物理损失函数**
> - **第二阶段（可选）**：实现 OmniFluids 风格的 PSM loss
> - 这样可以隔离变量，先验证网络架构的效果

> **原则 3: 时间步长严格对齐 (⚠️ 关键！)**
> - 所有 dt 设置与 mhd_sim 保持一致，确保公平比较
> - 详见下方 "时间步长对齐方案"

---

### 设计决策 Q&A (⚠️ 关键设计记录)

> **⚠️ 重要更正 (2026-02-23)**: 以下分析基于 `mhd_sim-5field_sim` 分支。
> 该分支上的 5-field MHD 是 **2D** (Nx, Ny) 系统，不是 3D！
> 之前基于 master 分支的 3D 分析全部作废。

#### mhd_sim 5-field MHD 神经网络的完整做法

mhd_sim (`mhd_sim-5field_sim` 分支) 已有完整的 5-field MHD 训练 pipeline:

| 组件 | 文件 | 说明 |
|------|------|------|
| 方程 (2D) | `numerical/equations/five_field_mhd.py` | 2D Landau-fluid, (..., Nx, Ny) |
| 训练 | `plasma_sim/train/train_5field.py` | 通用 physics loss + UNet |
| 数据集 | `plasma_sim/train/mhd5_dataset.py` | (B,T,C=5,Nx,Ny) |
| 评估 | `plasma_sim/train/eval_5field.py` | autoregressive rollout |
| 数据生成 | `numerical/scripts/run_5field_mhd_batch.py` | 批量 RK4 仿真 |
| 诊断 | `plasma_sim/train/mhd5_diagnostics.py` | E_kin, E_mag, E_th |
| 通用工具 | `plasma_sim/train/train_utils.py` | model_predict, loss, normalizer |

**网络 Input/Output** (mhd_sim baseline):
- **Input**: `[B, C=5, Nx, Ny]`，C=5 对应 **(n, U, vpar, psi, Ti)** — 注意 channel-first!
- **Output**: `[B, C=5, Nx, Ny]` (residual mode: 输出 `x_{t+dt} - x_t`)
- **网络**: UNet, base_channels=128, channel_mults=(1,2,4,8), condition_on_dt=True
- **Physics loss**: `|(pred - x_t)/dt - rhs(x_t)|` (Crank-Nicolson time-differencing)

**数据格式** (来自 `run_5field_mhd_batch.py`):
```python
dataset = {
    "n":     (B, T, Nx, Ny),   # 密度扰动
    "U":     (B, T, Nx, Ny),   # 涡度 (nabla_perp^2 phi = U)
    "vpar":  (B, T, Nx, Ny),   # 平行速度
    "psi":   (B, T, Nx, Ny),   # 磁通函数
    "Ti":    (B, T, Nx, Ny),   # 离子温度
    "phi":   (B, T, Nx, Ny),   # 电势 (辅助场, 由 U Poisson 求解)
    "t_list": (T,),            # 时间戳
    "seeds":  [int, ...],
}
# 加载后 stack: states = (B, T, C=5, Nx, Ny)，通道顺序 [n, U, vpar, psi, Ti]
```

**默认物理/数值参数** (`FiveFieldMHDConfig`):
- Grid: Nx=512, Ny=256, Lx=100, Ly=20π
  (⚠️ 注: mhd_sim 内部 equations doc 写 Ny=128, 但代码默认是 Ny=256, 以代码为准)
- M=48 (dealiasing 保留模式数)
- dt_sim=2e-3, stamp_interval=500, **dt_data=1.0**
- 训练 delta_t: 0.001 ~ 0.1 (多种实验)
- eta_i=1.0, beta=0.01, shear=0.1, mass_ratio=1836
- Dn=Du=eta_perp=chi_perp=0.01

**方程离散化** (mixed FD-spectral, 2D):
- x 方向: **有限差分 (FD)**，Dirichlet BC (zero-pad + narrow)
- y 方向: **伪谱方法 (rfft/irfft)**，Periodic BC
- nabla_par(f) = s_hat * λ * arcsinh(x/λ) * ∂f/∂y (纯 2D，无 z 导数)
- landau_damping: |k_∥| * u，其中 k_∥ = s_hat * λ * arcsinh(x/λ) * ky

#### Q1: 维度问题 — 不存在！

**修正**: 5-field MHD 是 **2D** 系统 `(Nx, Ny)`，与 OmniFluids nse2d 的 2D `(S, S)` 完全匹配。
无需考虑 z 维度、z-as-batch、3D 算子耦合等问题。
唯一的形状差异是 **矩形网格** (Nx ≠ Ny) vs nse2d 的方形网格 (S × S)。

#### Q2: 边界条件 — x 方向用 DST 替代 FFT

**问题**: OmniFluids 的 `SpectralConv2d_dy` 在两个方向都用 FFT:
```python
x_fty = torch.fft.rfft(x, dim=-1)   # y 方向 → periodic → 正确
x_ftx = torch.fft.rfft(x, dim=-2)   # x 方向 → assumes periodic → 与 Dirichlet BC 不匹配!
```

**mhd_sim 的做法**: x 方向完全不用 FFT，而是用 FD (有限差分)。
OmniFluids 的 spectral conv 必须在 x 方向做变换来学习频域特征。

**⚠️ 重要发现: mhd_sim UNet baseline 也有同样的 BC 不匹配！**
mhd_sim 的 UNet 使用 `padding_mode="circular"` (双周期 padding):
```python
# plasma_sim/models/unet.py:15-16
self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular")
self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular")
```
这意味着 mhd_sim baseline 在 x 方向也假设了周期性，和实际的 Dirichlet BC 不一致。
所以如果 OmniFluids 用 DST 正确处理 x 方向 BC，理论上比 UNet baseline 更物理正确。

**决策**: x 方向改用 **DST (离散正弦变换)** 替代 FFT。
- DST 展开为 `f(x) = Σ a_k * sin(kπx/L)` — 自然满足 Dirichlet BC (f(0)=f(L)=0)
- y 方向保持 FFT (周期性，正确)
- 这比 mhd_sim baseline 的 circular padding 更物理正确

**实现**: PyTorch 无原生 DST，通过奇延拓 + rfft 实现。

#### Q3: MoE 条件输入适配

**问题**: nse2d 中 MoE 条件输入为 ν (标量) + f (空间场)；5-field MHD 参数暂时固定且无外部 forcing。

**决策**:
- 保留 MoE 结构，用物理参数向量 `[eta_i, beta, shear, ...]` 代替 ν 作为条件输入
- 当前参数固定 → attention weights 会收敛到固定值，但保留结构方便后续扩展
- `param` 不再拼接到空间输入中（无空间变化的 forcing）
- 输入: **5 fields + 2 grid = 7 channels** → `in_proj(7, width)`

#### Q4: 数据来源

**决策**: 由同事提供已生成好的数据。
**数据路径**: `/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt`
**数据配置确认**: Nx=512, Ny=256, dt=0.002, stamp_interval=500, n_samples=10, Nt=1000000
**数据格式**: dict with keys n, U, vpar, psi, Ti, each (B=10, T=2001, 512, 256)

⚠️ **待处理**: 数据文件加载报错 (zip archive corrupt?)，需要和同事确认重新传输。

#### Q5-Q8: 最终实现决策 (2026-02-23)

| 决策项 | 选择 | 说明 |
|--------|------|------|
| modes_x | **128** | 与 modes_y 相同，后续可调 |
| MoE 条件 | **保留 MoE** | 用 physics params 向量, 固定参数但保留结构 |
| 空间输入 | **7 channels** | 5 fields + 2 grid, 不拼接参数 |
| 输出头 | **5 独立头** | 每个 field 独立的 (fc_a, fc_b, Conv1d) |
| output_dim | **10** | 超参数，后续可调 |
| 精度 | **fp32** | 网络 fp32, RHS 计算 fp64 |
| 时间窗口 | **t∈[250, 300]** | 与 mhd_sim baseline 一致 (train_5field.sh:28-29) |
| 数据分辨率 | **512×256** | 从 batch_config.yaml 确认 |

---

### 时间步长对齐方案 (⚠️ 关键配置)

#### 对比原有设置 (⚠️ 已基于 mhd_sim-5field_sim 分支修正)

| 设置项 | nse2d (原) | mhd_sim 5-field | **OmniFluids 适配 (对齐 mhd_sim)** |
|--------|-----------|-----------------|-------------------------------------|
| 仿真 dt | 1e-4 | **2e-3** | (不涉及，数据由同事提供) |
| stamp_interval | N/A | **500** | N/A |
| dt_data (数据间隔) | N/A | **1.0** (= 2e-3 × 500) | **1.0** |
| 神经网络 delta_t | rollout_DT / output_dim | **0.001 ~ 0.1** (实验) | 与 mhd_sim 实验对齐 |
| Grid | 256×256 | **512×256** | **与数据一致** |
| M (dealiasing) | N/A | **48** | N/A (spectral conv 自带 mode truncation) |

#### 明确的 dt 设置

```python
# OmniFluids 5-field MHD 的时间配置 (与 mhd_sim 严格对齐)

# 1. 仿真 dt (数据生成用，由同事提供数据)
SIMULATION_DT = 2e-3  # FiveFieldMHDConfig.dt

# 2. 数据 dt (连续 snapshot 间隔)
STAMP_INTERVAL = 500  # FiveFieldMHDConfig.stamp_interval
DT_DATA = SIMULATION_DT * STAMP_INTERVAL  # = 1.0

# 3. 神经网络 delta_t (与 mhd_sim 训练实验对齐)
DELTA_T = 1.0  # 等于 dt_data，每次预测一个数据步
# 注意: mhd_sim 实验中也尝试了 0.001, 0.01, 0.1 等

# 4. OmniFluids 多帧预测配置
# OmniFluids 一次前向输出 output_dim 帧，覆盖 rollout_DT 时间
ROLLOUT_DT = DELTA_T  # = 1.0
TRAIN_TIME_INTERVAL = SIMULATION_DT  # = 2e-3 (内部帧间隔与仿真 dt 对齐)
OUTPUT_DIM = int(ROLLOUT_DT / TRAIN_TIME_INTERVAL)  # = 500 帧 (非常多!)

# ⚠️ 注意: 500 帧的 output_dim 太大了，可能需要调整
# 备选方案: 减少 output_dim，增大 TRAIN_TIME_INTERVAL
# 例如: TRAIN_TIME_INTERVAL = 0.04 → OUTPUT_DIM = 25
```

#### 时间设置图解

```
mhd_sim 仿真 timeline (dt_sim = 2e-3):
|-----|-----|-----|-----|-----|... (每 500 步保存一帧)
0  2e-3  4e-3  6e-3  ...         t=1.0 (第 500 步 → snapshot 1)

mhd_sim 数据: dt_data = 1.0
|==================|==================|==================|
t=0               t=1.0             t=2.0             t=3.0
snapshot 0        snapshot 1        snapshot 2        snapshot 3

mhd_sim UNet baseline: delta_t = 1.0 (或 0.001, 0.01, 0.1)
input: x_t → model → output: x_{t+delta_t}
(单帧预测，可以 sub-step: 多次 delta_t 步进到 dt_data)

OmniFluids 适配方案 (output_dim = 25, TRAIN_TIME_INTERVAL = 0.04):
输入: state @ t=0
输出: [state@0.04, state@0.08, ..., state@1.0]  (共 25 帧)
                                                ↑
                                     最后一帧 = t + 1.0 (= dt_data)

OmniFluids 推理时 (自回归):
step 1: input(t=0)   → output(t=1.0)   取最后帧
step 2: input(t=1.0) → output(t=2.0)   取最后帧
step 3: input(t=2.0) → output(t=3.0)   取最后帧
...
```

---

### 代码组织结构

```
OmniFluids/
├── nse2d_old/                # nse2d 备份 (原始 NS 代码，不动)
├── nse2d/                    # 【直接修改】改为 5-field MHD
│   ├── pretrain/
│   │   ├── model.py          # OmniFluids2D → 适配 5-field MHD
│   │   ├── psm_loss.py       # → 改为调用 mhd_sim RHS 的 physics loss
│   │   ├── train.py          # → 适配 MHD 数据加载和训练流程
│   │   ├── main.py           # → 更新超参数和配置
│   │   └── tools.py          # → 更新工具函数
│   ├── data/                 # 数据由同事提供，无需自行生成
│   └── ...
├── kse2d/                    # KS 代码 (不修改)
└── docs/

mhd_sim/                      # 【不修改】保持原样作为 baseline
├── numerical/                # 调用其仿真和 RHS 计算
└── plasma_sim/               # 参考其训练和评估逻辑
```

---

### 阶段 1: 数据加载与格式适配

**数据**: 由同事提供，格式与 `run_5field_mhd_batch.py` 输出一致。
**不需要自己生成数据。**

```python
# 直接复用 mhd_sim 的数据加载逻辑
import sys; sys.path.append('/zhangtao/project2026/mhd_sim')
from plasma_sim.train.mhd5_dataset import load_raw_mhd5_data

# 返回: states (B, T, C=5, Nx, Ny), dt_data (float), stamp_interval (int)
states, dt_data, stamp_interval = load_raw_mhd5_data(data_path)
```

**需要做的格式转换**:
| mhd_sim 格式 | OmniFluids 格式 | 转换方法 |
|--------------|-----------------|----------|
| `(B, T, C=5, Nx, Ny)` channel-first | `(B, Nx, Ny, 5)` channel-last | `permute(0, 3, 4, 2)` per snapshot |
| 2D `(Nx, Ny)` | 2D `(Nx, Ny)` | 无需转换 (都是 2D!) |

**关键**: OmniFluids 训练只需要 snapshot (不需要连续对)，因为用 physics loss。

---

### 阶段 2: 网络架构适配 (修改 `nse2d/pretrain/model.py`)

保留 OmniFluids 核心架构（谱卷积 + MoE 注意力），修改以下几点:

#### 2.1 SpectralConv2d_dy: x 方向 FFT → DST

```python
# 原始: 两个方向都用 rfft (假设双周期)
x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')  # y 方向, OK (periodic)
x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')  # x 方向, WRONG (Dirichlet!)

# 修改: x 方向改用 DST (离散正弦变换)
x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')     # y 方向: 保持 FFT (periodic)
x_ftx = dst(x, dim=-2)                               # x 方向: 改用 DST (Dirichlet)
# DST 实现: 奇延拓 + rfft，或 scipy.fft.dstn
```

**DST 实现方案** (用 rfft 实现 DST, 概念伪代码):
```python
def dst_via_rfft(x, dim=-2):
    """DST via odd extension + rfft (概念伪代码, 实际实现需仔细处理 indexing).
    
    对长度 N 的 Dirichlet 信号 (x[0]=x[-1]=0):
    1. 取内部 N-2 个点: x[1], ..., x[N-2]
    2. 奇延拓为长度 2(N-1) 的序列: [0, x[1], ..., x[N-2], 0, -x[N-2], ..., -x[1]]
    3. rfft → 虚部给出正弦系数
    
    返回 N-2 个 DST 系数 (对应 sin(k*pi*x/L), k=1,...,N-2)
    """
    # 具体实现待编码阶段确定
    pass

def idst_via_irfft(X_dst, N, dim=-2):
    """逆 DST: DST 系数 → 实空间 (含零边界)"""
    pass
```
**注**: PyTorch 无原生 DST，需通过奇延拓 + rfft 实现。具体 indexing 和归一化在编码阶段仔细处理。

#### 2.2 输入/输出通道数

| 组件 | nse2d (原) | 5-field MHD (新) |
|------|-----------|------------------|
| 物理场数 | 1 (w) | **5** (n, U, vpar, psi, Ti) |
| 输入通道 | 5 = 1(w) + 2(grid) + 1(f) + 1(ν) | **7** = 5(fields) + 2(grid) |
| MoE 条件 | 1 (log10 ν) | **n_params** ([eta_i, beta, ...]) |
| 输出 | `[B, S, S, output_dim]` (1 field × T) | `[B, Nx, Ny, 5*output_dim]` (5 fields × T) |

#### 2.3 矩形网格支持

nse2d 假设方形 `(S, S)`，需要修改为支持 `(Nx, Ny)`:
- grid 坐标分别在两个方向归一化
- SpectralConv 的 modes 数可能不同: modes_x, modes_y
- 输出层 reshape 时考虑 Nx ≠ Ny

#### 2.4 完整修改清单 (model.py)

```
1. __init__:
   - in_proj: Linear(7, width)  [5 fields + 2 grid]
   - f_nu → f_param: MLP(n_params → K)  [物理参数 → attention weights]
   - grid: (Nx, Ny, 2) 非方形
   - SpectralConv: modes_x, modes_y 可不同; x 方向 DST, y 方向 FFT
   - 输出层: width → 5*output_dim (5 fields × T frames)
   
2. forward:
   - 输入: x=[B, Nx, Ny, 5], cond=[n_params] (标量向量)
   - 拼接 grid → [B, Nx, Ny, 7]
   - in_proj → [B, Nx, Ny, width]
   - SpectralConv × n_layers (DST-x + FFT-y)
   - 输出: [B, Nx, Ny, 5*output_dim] → reshape → [B, Nx, Ny, 5, output_dim]
   - 残差: x_0[:, :, :, :, None] + pred * dt_linspace  (对每个 field)
```

---

### 阶段 3: 损失函数 (修改 `nse2d/pretrain/psm_loss.py`)

**第一阶段: 直接调用 mhd_sim 的 `compute_rhs()` 作为 physics loss target**

```python
import sys; sys.path.append('/zhangtao/project2026/mhd_sim')
from numerical.equations.five_field_mhd import FiveFieldMHD, FiveFieldMHDConfig

def make_mhd5_rhs_fn(mhd: FiveFieldMHD):
    """RHS adapter: (B, Nx, Ny, 5) → (B, Nx, Ny, 5)
    
    注意: OmniFluids 用 channel-last, mhd_sim 用 tuple-of-5 (B, Nx, Ny)
    """
    def rhs_fn(x):  # x: (B, Nx, Ny, 5)
        state = tuple(x[..., i] for i in range(5))   # 5 个 (B, Nx, Ny)
        rhs_tuple = mhd.compute_rhs(state)
        return torch.stack(rhs_tuple, dim=-1)         # (B, Nx, Ny, 5)
    return rhs_fn

def compute_mhd5f_physics_loss(pred_traj, x_0, rhs_fn, rollout_DT, output_dim,
                                time_integrator='crank_nicolson'):
    """
    OmniFluids 风格的 physics loss，使用 mhd_sim 的 RHS。
    
    pred_traj: [B, Nx, Ny, 5, output_dim] — OmniFluids 多帧输出
    x_0: [B, Nx, Ny, 5] — 初始状态
    
    对 pred_traj 的每相邻帧对施加 time-differencing loss:
      (pred[t+1] - pred[t]) / dt ≈ rhs(pred[t]) or (rhs(t) + rhs(t+1))/2
    """
    dt = rollout_DT / output_dim
    full_traj = torch.cat([x_0.unsqueeze(-1), pred_traj], dim=-1)  # [B,Nx,Ny,5,od+1]
    
    total_loss = 0.0
    for t in range(output_dim):
        state_t = full_traj[..., t]      # [B, Nx, Ny, 5]
        state_tp1 = full_traj[..., t+1]  # [B, Nx, Ny, 5]
        time_diff = (state_tp1 - state_t) / dt
        
        if time_integrator == 'crank_nicolson':
            target = (rhs_fn(state_t) + rhs_fn(state_tp1)) / 2
        elif time_integrator == 'euler':
            target = rhs_fn(state_t)
        
        total_loss += F.mse_loss(time_diff, target.detach())
    
    return total_loss / output_dim
```

**为什么先用 mhd_sim 的 RHS**:
1. 保证物理约束与 baseline 完全一致
2. 避免 5-field 方程的重新实现错误
3. 隔离变量: 效果差异只来自网络架构

---

### 阶段 4: 训练流程适配 (修改 `nse2d/pretrain/train.py`, `main.py`)

#### 4.1 代码结构 (直接修改 nse2d/)

```
OmniFluids/nse2d/             # 【直接修改】
├── pretrain/
│   ├── model.py              # OmniFluids2D → 5-field MHD 适配
│   ├── psm_loss.py           # NS PSM → mhd_sim RHS physics loss
│   ├── train.py              # 训练循环 (改为离线数据集)
│   ├── main.py               # 超参数配置
│   └── tools.py              # 数据加载/工具函数
└── ...
```

#### 4.2 训练配置 (main.py)

```python
# 与 mhd_sim 对齐的配置
modes = 128          # spectral modes (≤ Ny//2)
width = 80           # hidden width
n_layers = 12        # spectral conv layers
K = 4                # MoE operators
output_dim = 25      # 多帧: rollout_DT / train_time_interval
n_fields = 5         # physical fields

# Grid (与数据一致)
Nx = 512; Ny = 256   # 或数据实际分辨率

# 时间 (与 mhd_sim 对齐)
rollout_DT = 1.0     # = dt_data
output_dim = 25      # 内部帧数
# train_time_interval = rollout_DT / output_dim = 0.04

# 训练
lr = 0.002
num_iterations = 20000
batch_size = 8       # 512×256 显存较大，可能需要减小

# 精度 (与 mhd_sim 对齐)
dtype = torch.float64

# Physics loss
time_integrator = 'crank_nicolson'  # 与 mhd_sim 对齐
```

#### 4.3 训练循环改动

**从 "在线数据生成" 改为 "离线数据集":**

```python
# 原 nse2d: 在线生成
w0_train = GRF(batch_size); f_train = F_Sampler(batch_size)  # 每步随机

# 新 5-field: 离线数据集
states, dt_data, _ = load_raw_mhd5_data(data_path)  # (B, T, 5, Nx, Ny)
# 取时间窗口内的 snapshots 作为训练集
train_snapshots = extract_snapshots(states, t_start, t_end)  # (N, 5, Nx, Ny)
# permute to channel-last for OmniFluids
train_snapshots = train_snapshots.permute(0, 2, 3, 1)  # (N, Nx, Ny, 5)
loader = DataLoader(train_snapshots, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    for x_0 in loader:
        pred_traj = model(x_0, param)  # [B, Nx, Ny, 5, output_dim]
        loss = compute_mhd5f_physics_loss(pred_traj, x_0, rhs_fn, rollout_DT, output_dim)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**验证**: 与 mhd_sim 对齐，使用 autoregressive rollout 计算 rel L2 error。

---

### 阶段 5: 评估与可视化对齐

#### 5.1 评估指标 (与 mhd_sim eval_5field.py 完全对齐)

```python
# 复用 mhd_sim 的评估工具
from plasma_sim.train.train_utils import compute_per_step_correlation
from plasma_sim.train.mhd5_diagnostics import compute_mhd5_diagnostics

# 指标列表 (与 mhd_sim 一致):
# 1. Per-step relative L2 error (总体 + 每场分别)
# 2. Per-step Pearson correlation
# 3. Energy diagnostics: E_kin, E_mag, E_th, E_total
# 4. Snapshot visualization: pred vs gt 对比
```

#### 5.2 OmniFluids → mhd_sim 格式转换 (评估时)

```python
def omnifluids_to_mhd5_eval(pred_omnifluids):
    """OmniFluids rollout 输出 → mhd_sim 评估格式
    
    OmniFluids: [B, Nx, Ny, 5, T] (channel-last, multi-frame)
    mhd_sim:    [B, T, C=5, Nx, Ny] (channel-first, trajectory)
    """
    return pred_omnifluids.permute(0, 4, 3, 1, 2)  # [B, T, 5, Nx, Ny]
```

---

### 关键注意事项

1. **不修改 mhd_sim 代码** — 只 import 使用 (sys.path.insert)
2. **直接修改 nse2d/** — 已有 `nse2d_old/` 备份
3. **数据对齐**: ✅ 使用完全相同的 `5field_mhd_dataset.pt`, 相同 train/eval 时间窗口 [250, 300]
4. **dt 对齐**: ✅ rollout_dt = dt_data = 1.0, output_dim=10, 每帧 dt=0.1 (匹配 mhd_sim dt=0.1 实验)
5. **精度**: 网络 float32, RHS 内部 float64 (用户选择, 不同于 mhd_sim 全 fp64)
6. **评估对齐**: ✅ 相同的 rel L2 metric, 相同的 autoregressive rollout, 可视化格式对齐
7. **命名规范**:
   ```
   log/log_{exp_name}/log-{timestamp}.csv
   model/{exp_name}/best.pt, latest.pt
   results/{exp_name}/eval_results.json
   results/{exp_name}/step_N/  (可视化)
   ```
8. **⚠️ 待解决**: 数据文件 `5field_mhd_dataset.pt` 加载失败 (可能损坏), 需重新获取

---

## 附录: 代码参考

### A.1 关键文件列表

**OmniFluids (nse2d) — 待修改:**
- `pretrain/model.py` - 网络定义 (→ 5-field + DST + 矩形网格)
- `pretrain/psm_loss.py` - 物理损失 (→ mhd_sim RHS)
- `pretrain/train.py` - 训练逻辑 (→ 离线数据集)
- `pretrain/main.py` - 入口和配置 (→ 新超参数)
- `pretrain/tools.py` - 工具函数 (→ 数据加载)

**mhd_sim (5-field MHD) — 只 import, 不修改:**
- `numerical/equations/five_field_mhd.py` - **2D 方程 + compute_rhs()**
- `numerical/operators/spectral.py` - **2D 谱算子 (rfft)**
- `numerical/operators/derivatives.py` - **2D FD 算子 (Dirichlet-x)**
- `numerical/operators/poisson_solver.py` - **2D Poisson (混合 BC)**
- `numerical/operators/arakawa.py` - **2D Arakawa bracket**
- `plasma_sim/models/unet.py` - UNet baseline 模型
- `plasma_sim/train/train_5field.py` - 5-field 训练脚本 (参考)
- `plasma_sim/train/train_utils.py` - **通用 loss/eval 工具**
- `plasma_sim/train/mhd5_dataset.py` - **数据加载**
- `plasma_sim/train/eval_5field.py` - **评估脚本**
- `plasma_sim/train/mhd5_diagnostics.py` - **能量诊断**

### A.2 Shape 快速参考

**OmniFluids nse2d (原始):**
```
输入: x=[B, S, S, 1], param=[B, S, S, 2]  → concat → [B, S, S, 5]
内部: [B, S, S, width]
输出: [B, S, S, output_dim]  — 1 field × T frames
```

**mhd_sim UNet baseline (5-field MHD):**
```
输入: x=[B, C=5, Nx, Ny], dt=[B]    — channel-first, (n,U,vpar,psi,Ti)
内部: UNet [B, ch, Nx/k, Ny/k]
输出: [B, C=5, Nx, Ny]              — residual
```

**OmniFluids 适配 5-field MHD (目标):**
```
输入: x=[B, Nx, Ny, 5]              — channel-last, (n,U,vpar,psi,Ti)
拼接 grid: [B, Nx, Ny, 7]          — 5 fields + 2 grid coords
in_proj:   [B, Nx, Ny, width]
SpectralConv (DST-x, FFT-y) × n_layers: [B, Nx, Ny, width]
输出层:    [B, Nx, Ny, 5*output_dim] → reshape [B, Nx, Ny, 5, output_dim]
残差+插值: x_0 + pred * dt_linspace → [B, Nx, Ny, 5, output_dim]
```
