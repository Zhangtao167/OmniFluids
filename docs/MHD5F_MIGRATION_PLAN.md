# OmniFluids 迁移至 5-Field MHD 详细文档

## 目录
1. [NSE2D (OmniFluids) 详细分析](#1-nse2d-omnifluids-详细分析)
2. [5-Field MHD (mhd_sim) 详细分析](#2-5-field-mhd-mhd_sim-详细分析)
3. [关键差异对比](#3-关键差异对比)
4. [迁移开发计划](#4-迁移开发计划)

---

## 1. NSE2D (OmniFluids) 详细分析

### 1.1 方程细节

**2D Navier-Stokes 涡度方程**：
```
∂ω/∂t + u·∇ω = ν∇²ω + f
```

其中：
- `ω`: 涡度 (vorticity)，标量场
- `u = (u_x, u_y)`: 速度场，通过流函数 `ψ` 计算：`u_x = ∂ψ/∂y`, `u_y = -∂ψ/∂x`
- `ψ`: 流函数，满足 `∇²ψ = ω`
- `ν`: 粘性系数
- `f`: 外力项

**物理场数量**: 1 (只有涡度 ω)

### 1.2 可调超参数

| 参数 | 代码位置 | 默认值 | 含义 |
|------|---------|--------|------|
| `ν` (nu) | `train.py` | 10^(-lognu) | 粘性系数，以 log10 形式采样 |
| `lognu_range` | data config | [-3, -5] | log10(ν) 的范围 |
| `f` (forcing) | `Force_generation` | 随机 | 外力场，低频随机场 |
| `max_frequency` | data config | 4 | 外力最大频率 |
| `amplitude_range` | data config | (-0.1, 0.1) | 外力振幅范围 |

### 1.3 仿真和模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 空间分辨率 | 256×256 (可配置) | 默认 `size=256` |
| rollout_DT | 0.2s | 每次预测的物理时间跨度 |
| 训练数据来源 | GRF 在线生成 | 无需预生成数据 |
| 数据精度 | float32 | 训练使用 |

### 1.4 网络结构 (OmniFluids2D)

**输入**：
```python
# x: [B, S, S, 1]      - 涡度场 ω
# param: [B, S, S, 2]  - (外力 f, log(ν)) 广播到空间维度

# 拼接后输入 in_proj:
x_input = concat(x, grid, param)  # [B, S, S, 5]
# 维度: 1(ω) + 2(grid_x, grid_y) + 2(f, logν) = 5
```

**架构核心组件**：

```
┌─────────────────────────────────────────────────────────────────┐
│                      OmniFluids2D 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入 x: [B, S, S, 1] ───┐                                     │
│  param: [B, S, S, 2] ────┤                                     │
│  grid: [1, S, S, 2] ─────┴──► concat ──► in_proj ──► [B,S,S,W] │
│                                          (Linear 5→W)          │
│                                               │                 │
│                                               ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              n_layers × SpectralConv2d_dy                │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  1. f_nu MLP: logν → attention weights [B, K]      │  │  │
│  │  │     (Linear 1→128→128→K, with softmax/T)           │  │  │
│  │  │                                                    │  │  │
│  │  │  2. Mixture of K Fourier kernels:                  │  │  │
│  │  │     weight = Σ att_k × kernel_k                    │  │  │
│  │  │                                                    │  │  │
│  │  │  3. Factorized FFT (x-dir + y-dir):               │  │  │
│  │  │     x_ft = rfft(x) → kernel × x_ft → irfft        │  │  │
│  │  │                                                    │  │  │
│  │  │  4. Residual: x = x + FeedForward(spectral_out)   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  训练路径 (inference=False):                             │  │
│  │    fc1a: [B,S,S,W] → [B,S,S,4*Tp-4]                     │  │
│  │    fc1b: [B,S,S,W] → [B,S,S,34]                         │  │
│  │    concat → [B,S,S,4*Tp+30] → output_mlp → [B,S,S,Tp]   │  │
│  │                                                          │  │
│  │  推理路径 (inference=True):                              │  │
│  │    fc1b: [B,S,S,W] → [B,S,S,34] → output_mlp → [B,S,S,1]│  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  残差 + dt ramp:                                         │  │
│  │    dt = [1/Tp, 2/Tp, ..., 1.0]                          │  │
│  │    output = x_0 + prediction × dt                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**输出**：
```python
# 训练时 (inference=False): [B, S, S, Tp]  - Tp 个子帧
# 推理时 (inference=True):  [B, S, S, 1]   - 单帧预测

# 物理意义: w(t + k×dt) = w(t) + Δw_pred × (k/Tp)
```

**参数量** (默认配置 s=256, K=4, modes=128, width=80, n_layers=12):
- 约 68M 参数

### 1.5 Loss 构建

**PSM_loss (Physics Structure Matching)**：

```python
# nse2d/pretrain/psm_loss.py

def PSM_NS_vorticity(w, v, t_interval, loss_mode='cn'):
    """
    计算 NS 涡度方程的 PDE 残差
    
    w: [B, S, S, Nt] - 涡度序列
    v: [B] - 粘性系数 ν
    t_interval: 总物理时间跨度
    """
    # FFT 到频域
    w_h = fft2(w)
    
    # 波数
    k_x, k_y = wavenumbers(S)
    lap = k_x² + k_y²
    
    # 求流函数: ψ̂ = ω̂/k²
    f_h = w_h / lap
    
    # 速度场 (谱导数)
    u_x = ifft(i*k_y * f_h)  # ∂ψ/∂y
    u_y = ifft(-i*k_x * f_h) # -∂ψ/∂x
    
    # 涡度梯度
    w_x = ifft(i*k_x * w_h)
    w_y = ifft(i*k_y * w_h)
    
    # 扩散项
    w_lap = ifft(-lap * w_h)
    
    # 空间算子: u·∇ω - ν∇²ω
    Du = u_x*w_x + u_y*w_y - ν*w_lap
    
    # Crank-Nicolson 时间离散
    dt = t_interval / (Nt - 1)
    w_t = (w[..., 1:] - w[..., :-1]) / dt
    residual = w_t + 0.5*(Du[..., :-1] + Du[..., 1:])
    
    return residual  # 如果满足方程，residual → 0

def PSM_loss(u, forcing, v, t_interval, loss_mode='cn'):
    """
    PSM Loss = √(mean((残差 - 外力)²))
    """
    Du = PSM_NS_vorticity(u, v, t_interval, loss_mode)
    f = forcing.repeat(1, 1, 1, Nt-1)
    return sqrt(mean((Du - f)²) + ε)
```

**Loss 特点**：
- 自监督（不需要真实数据）
- √MSE 形式（梯度压缩）
- 允许退化解（预测不变）

### 1.6 训练 Setting

```python
# nse2d/pretrain/main.py 默认配置

batch_size = 10
val_size = 10
lr = 0.002
weight_decay = 0.0
num_iterations = 20000
rollout_DT = 0.2  # 秒
loss_mode = 'cn'  # Crank-Nicolson

# 数据生成
GRF = Init_generation(size)      # 高斯随机场初始条件
F_Sampler = Force_generation(size)  # 随机外力场

# 训练流程
for step in range(num_iterations):
    w0 = GRF(batch_size)   # 在线生成初始条件
    f = F_Sampler(batch_size)  # 在线生成外力
    nu = 10^(-random(lognu_min, lognu_max))  # 随机粘性
    
    w_pred = model(w0, param)  # 预测多子帧
    w_all = concat(w0, w_pred)  # [B,S,S,Tp+1]
    
    loss = PSM_loss(w_all, f, nu, rollout_DT)
    loss.backward()
```

---

## 2. 5-Field MHD (mhd_sim) 详细分析

### 2.1 方程细节

**5-Field MHD 方程组**（3D）：

```
dU/dt = -[φ, U] + (1+η_i)∂U/∂y + ∇_∥(j_∥) + D_U∇_⊥²U + 曲率项
dv_∥/dt = -[φ, v_∥] - (1+τ)∇_∥(n) - ∇_∥(T_i) - β(1+τ+η_i)∂ψ/∂y + η_⊥∇_⊥²v_∥
dn/dt = -[φ, n] - ∂φ/∂y - ∇_∥(v_∥) + ∇_∥(j_∥) + D_n∇_⊥²n
dψ/dt = -[φ, ψ] + (1/β)(∇_∥(φ-n) + β∂ψ/∂y - C_L|∇_∥|(v_∥+j_∥) - ηj_∥)
dT_i/dt = -[φ, T_i] - η_i∂φ/∂y - (Γ-1)∇_∥(v_∥) - C_T|∇_∥|(T_i) + χ∇_⊥²T_i
```

其中：
- `U = ∇_⊥²φ`: 涡度
- `v_∥`: 平行速度
- `n`: 密度扰动
- `ψ`: 磁通函数
- `T_i`: 离子温度
- `j_∥ = -∇_⊥²ψ`: 平行电流
- `φ`: 电势（通过 Poisson 方程求解）
- `[f, g]`: Poisson 括号（Arakawa 格式）
- `∇_∥`: 平行梯度算子
- `∇_⊥²`: 垂直 Laplacian

**物理场数量**: 5 (U, v_∥, n, ψ, T_i) + 2 辅助场 (φ, j_∥)

### 2.2 可调超参数

| 参数 | 代码位置 | 默认值 | 含义 |
|------|---------|--------|------|
| `η_i` (eta_i) | PhysicsConfig | 3.0 | ITG 驱动 (>1.5 不稳定) |
| `β` (beta) | PhysicsConfig | 0.01 | 等离子体 beta |
| `shear` | PhysicsConfig | 0.0 | 磁剪切 |
| `τ` (tau) | PhysicsConfig | 1.0 | T_e/T_i 比值 |
| `Γ` (Gamma) | PhysicsConfig | 5/3 | 绝热指数 |
| `mass_ratio` | PhysicsConfig | 100.0 | m_i/m_e 质量比 |
| `D_n` | DiffusionConfig | 1e-2 | 密度扩散 |
| `D_U` | DiffusionConfig | 1e-2 | 涡度扩散 |
| `η_⊥` (eta_perp) | DiffusionConfig | 1e-2 | 垂直粘性 |
| `χ` (chi) | DiffusionConfig | 1e-2 | 热扩散率 |
| `η` (eta) | DiffusionConfig | 0.01 | 电阻率 |
| `hyper_nu` | DiffusionConfig | 0.0 | 超扩散系数 |
| `buffer_width` | BufferConfig | 0.2 | 边界缓冲区宽度 |
| `buffer_amp` | BufferConfig | 2.0 | 缓冲区阻尼强度 |
| `noise_level` | InitConfig | 1e-5 | 初始噪声水平 |
| `k_peak` | InitConfig | 4.0 | 初始噪声峰值波数 |
| `curvature_coeff` | InitConfig | 2.0 | 曲率驱动系数 |

### 2.3 仿真和模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 空间分辨率 | 128×64×32 (Nx×Ny×Nz) | 3D 网格 |
| 域大小 | 60×20×32 (Lx×Ly×Lz) | 物理尺寸 |
| dt | 1e-3 | 仿真时间步 |
| num_steps | 2000 | 总步数 |
| save_interval | 1 | 保存间隔 |
| 数据精度 | float64 | 仿真使用 |

### 2.4 网络结构 (mhd_sim UNet - HW 参考)

**注意**: mhd_sim 目前只有 HW 2D 的训练代码，没有 5-field MHD 的神经网络。以下是 HW UNet 的结构，作为参考。

**输入**：
```python
# x: [B, C=2, H, W]  - (ζ, n) channels-first
# dt: [B] (可选)    - 时间步长 (如果 condition_on_dt=True)

# 如果 condition_on_dt:
x_input = concat(x, dt_channel)  # [B, C+1, H, W]
```

**UNet 架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        mhd_sim UNet                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入 x: [B, 2, H, W] ──────────────────────────────────────┐  │
│  dt: [B] (可选) ─► expand to [B,1,H,W] ─► concat ──► [B,3,H,W] │
│                                                        │        │
│                                                        ▼        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Encoder Path                          │  │
│  │  Level 0: ConvBlock(in→64)  ────────────────► skip_0     │  │
│  │           AvgPool2d ↓                                    │  │
│  │  Level 1: ConvBlock(64→128) ────────────────► skip_1     │  │
│  │           AvgPool2d ↓                                    │  │
│  │  Level 2: ConvBlock(128→256) (bottleneck)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Decoder Path                          │  │
│  │  Level 1: Upsample → concat(skip_1) → ConvBlock(384→128) │  │
│  │  Level 0: Upsample → concat(skip_0) → ConvBlock(192→64)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  out_conv: Conv2d(64→2, kernel=1) ──────────► [B, 2, H, W]     │
│                                                                 │
│  if predict_mode == 'residual':                                │
│      output = x + raw_output                                    │
│  else:                                                          │
│      output = raw_output                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

ConvBlock(in_ch, out_ch):
    Conv2d(in, out, 3, padding=1, padding_mode='circular')
    GELU()
    Conv2d(out, out, 3, padding=1, padding_mode='circular')
    GELU()
```

**参数量**: 约 2M (base_channels=64, channel_mults=(1,2,4))

### 2.5 Loss 构建 (HW 参考)

**Physics-informed Loss (RHS-anchored)**：

```python
# mhd_sim/plasma_sim/train/train_hw.py

def compute_physics_loss(model, hw, x_t, delta_t, cfg):
    """
    物理信息损失 - 基于时间差分
    
    x_t: [B, C, H, W] - 当前状态
    delta_t: float - 时间步长
    """
    # 模型预测
    prediction = model_predict(model, x_t, delta_t)  # x_{t+dt}
    
    # 时间差分: (x_{t+dt} - x_t) / dt
    time_diff = (prediction - x_t) / delta_t
    
    # RHS 计算 (detach 输入)
    with torch.no_grad():
        rhs_xt = hw.compute_rhs(x_t.detach())  # 固定锚点
    
    # Crank-Nicolson: target = (RHS(x_t) + RHS(x_{t+dt})) / 2
    rhs_pred = hw.compute_rhs(prediction)  # 有梯度
    target = (rhs_xt + rhs_pred) / 2
    
    # MSE Loss
    return MSE(time_diff, target)
```

**Loss 特点**：
- 有锚点（RHS_0 固定）
- MSE 形式（梯度正常）
- 不允许退化解

### 2.6 训练 Setting (HW 参考)

```python
# mhd_sim/plasma_sim/train/train_hw.py 默认配置

delta_t = 0.025          # 模型预测步长
lr = 1e-4                # 学习率
weight_decay = 0.0
epochs = 100
batch_size = 16
grad_clip = None
input_noise_std = 0.0    # 输入噪声

# 数据
time_start = 200.0       # 训练起始时间 (spin-up 后)
time_end = 249.0         # 训练结束时间
dt_data = 1.0            # 数据帧间隔

# 精度
model_dtype = float64
rhs_dtype = float64

# Loss 配置
time_integrator = 'crank_nicolson'
mse_weight = 1.0
mae_weight = 0.0
unroll_steps = 1         # 多步训练
unroll_detach = False    # 梯度流过整个链
detach_rhs_input = True  # RHS 输入 detach
```

---

## 3. 关键差异对比

### 3.1 方程对比

| 维度 | NSE2D (OmniFluids) | 5-Field MHD (mhd_sim) |
|------|-------------------|----------------------|
| 空间维度 | 2D | 3D |
| 场数量 | 1 (ω) | 5 (U, v_∥, n, ψ, T_i) + 2 辅助 |
| 边界条件 | 周期 | x: Dirichlet, y/z: 周期 |
| 方程耦合 | 简单 (单场) | 复杂 (多场耦合) |
| 物理效应 | 粘性、外力 | 磁场、电流、温度、Landau 阻尼等 |

### 3.2 网络对比

| 维度 | OmniFluids2D | mhd_sim UNet |
|------|-------------|--------------|
| 架构类型 | FFNO + MoE | U-Net |
| 输入通道 | 1 (ω) + 2 (grid) + 2 (param) = 5 | 2 (ζ, n) + 1 (dt 可选) = 2~3 |
| 输出形式 | 残差 × dt_ramp | 残差 或 直接预测 |
| 参数条件化 | ν → attention weights | dt → 通道拼接 |
| 训练/推理路径 | 不同 (fc1a+fc1b vs fc1b) | 相同 |
| 参数量 | ~68M (大) | ~2M (小) |
| 周期边界 | 隐式 (FFT) | circular padding |

### 3.3 Loss 对比

| 维度 | OmniFluids PSM | mhd_sim RHS-anchored |
|------|---------------|---------------------|
| 形式 | √MSE | MSE |
| 锚点 | 无 (自监督) | 有 (RHS_0 固定) |
| 退化解 | 允许 | 不允许 |
| RHS 计算 | 谱方法 | Arakawa + FD |
| 需要数据 | 否 (在线生成) | 是 (预生成轨迹) |

### 3.4 训练流程对比

| 维度 | OmniFluids | mhd_sim |
|------|-----------|---------|
| 数据来源 | GRF 在线生成 | 数值模拟预生成 |
| 参数采样 | 随机 (ν, f) | 固定 |
| 学习率 | 0.002 | 1e-4 |
| 迭代次数 | 20000 | 100 epochs |
| 精度 | float32 | float64 |
| 多步训练 | 隐式 (Tp 子帧) | 显式 (unroll_steps) |

---

## 4. 迁移开发计划

### 4.1 总体目标

将 OmniFluids 的 FFNO+MoE 架构应用到 mhd_sim 的 5-field MHD 场景，实现：
1. 使用 mhd_sim 生成的数据训练
2. 与 mhd_sim 的评估方法对齐
3. 公平对比 OmniFluids 和 UNet 的性能

### 4.2 开发阶段

#### 阶段 1: 项目结构搭建

**目标**: 在 mhd_sim 仓库中创建 OmniFluids 训练框架

**任务清单**:

- [ ] **1.1** 创建目录结构
  ```
  mhd_sim/plasma_sim/
  ├── models/
  │   ├── omnifluids.py       # 新增: OmniFluids 3D 模型
  │   └── ...
  ├── train/
  │   ├── train_5field.py     # 新增: 5-field MHD 训练脚本
  │   ├── eval_5field.py      # 新增: 5-field MHD 评估脚本
  │   ├── mhd_dataset.py      # 新增: 5-field MHD 数据集类
  │   └── ...
  ```

- [ ] **1.2** 实现数据集类 `mhd_dataset.py`
  - 加载 5-field MHD 轨迹数据
  - 支持时间窗口切片
  - 输出格式: `[B, T, C=5, Nx, Ny, Nz]`

**验证**: 能正确加载和遍历数据

#### 阶段 2: 模型架构迁移

**目标**: 将 OmniFluids2D 扩展为 OmniFluids3D

**任务清单**:

- [ ] **2.1** 实现 `SpectralConv3d_dy`
  - 扩展 FFT 到 3D: `rfft` → `rfftn`
  - 修改波数处理
  - 注意 x 方向非周期边界处理

- [ ] **2.2** 实现 `OmniFluids3D` 模型
  ```python
  class OmniFluids3D(nn.Module):
      def __init__(self, 
                   s=(128, 64, 32),      # (Nx, Ny, Nz)
                   K=4,                   # MoE 核数
                   modes=(16, 16, 8),     # 傅里叶模数
                   width=64,              # 隐层宽度
                   n_layers=8,            # 层数
                   n_fields=5,            # 输入/输出场数
                   n_params=10):          # 方程参数数
          # ...
      
      def forward(self, x, param, inference=False):
          """
          x: [B, Nx, Ny, Nz, 5]  - 5 个物理场
          param: [B, n_params]   - 方程参数
          
          Returns: [B, Nx, Ny, Nz, Tp, 5] 或 [B, Nx, Ny, Nz, 5]
          """
  ```

- [ ] **2.3** 参数条件化设计
  - 输入参数: `[η_i, β, shear, τ, Γ, D_n, D_U, η_⊥, χ, η]`
  - 通过 MLP 生成 attention weights

**验证**: 
- 模型能正确前向传播
- 输出 shape 正确
- 参数量合理 (~2M 与 UNet 对标)

#### 阶段 3: Loss 函数实现

**目标**: 实现适配 5-field MHD 的 Loss

**任务清单**:

- [ ] **3.1** 实现 `PSM_5field` (OmniFluids 风格)
  - 计算 5 个场的 PDE 残差
  - CN 时间离散
  - Loss = Σ √MSE(residual_i)

- [ ] **3.2** 实现 `rhs_anchored_5field` (mhd_sim 风格)
  - 复用 `FiveFieldMHD.compute_rhs()`
  - Loss = MSE(time_diff, target_rhs)

- [ ] **3.3** 超参数控制
  ```python
  train_loss_type: str = 'mhd_sim'  # 'omnifluids' | 'mhd_sim'
  rhs_loss_weight: float = 0.0      # 混合 loss 权重
  ```

**验证**: Loss 能正确计算且梯度流动正常

#### 阶段 4: 训练脚本实现

**目标**: 完整的训练流程

**任务清单**:

- [ ] **4.1** 实现 `train_5field.py`
  - 配置类 `MHDTrainConfig`
  - 数据加载
  - 训练循环
  - 验证评估
  - Checkpoint 保存

- [ ] **4.2** 训练设置对齐
  ```python
  # 与 mhd_sim UNet 对齐
  delta_t: float = 0.025      # 或根据数据调整
  lr: float = 1e-4
  batch_size: int = 16
  model_dtype: str = 'float64'
  unroll_steps: int = 1
  ```

- [ ] **4.3** 保存路径命名
  ```
  outputs/plasma_sim/checkpoint/{exp_name}/
  ├── config.yaml
  ├── epoch_{N}.pt
  └── latest.pt
  
  outputs/plasma_sim/results/{exp_name}/
  ├── train_log.txt
  └── eval_results.json
  ```

- [ ] **4.4** 中间评估和可视化
  - 每 N epoch 评估一次
  - 保存 rel L2 per step
  - 保存能量诊断

**验证**: 能完成一轮完整训练

#### 阶段 5: 评估脚本实现

**目标**: 与 mhd_sim 对齐的评估流程

**任务清单**:

- [ ] **5.1** 实现 `eval_5field.py`
  - 加载 checkpoint
  - 自回归 rollout
  - 计算指标

- [ ] **5.2** 评估指标
  - Per-step relative L2 error
  - Mean relative L2 error
  - Per-step correlation
  - 能量诊断 (各场能量、总能量)
  - 粒子通量

- [ ] **5.3** 可视化
  - 2D 切片可视化 (z=const)
  - 误差场可视化
  - 指标时间演化曲线

**验证**: 评估结果与 UNet 可对比

#### 阶段 6: 数据生成和实验

**目标**: 生成数据并运行对比实验

**任务清单**:

- [ ] **6.1** 生成 5-field MHD 训练数据
  - 使用 `run_5field_mhd.py`
  - 保存格式与 HW 类似
  - 多个初始条件/参数

- [ ] **6.2** 训练 OmniFluids 模型
  - 使用 mhd_sim loss
  - 记录训练日志

- [ ] **6.3** 对比实验
  - OmniFluids vs UNet
  - 不同超参数
  - 记录并对比结果

**验证**: 完成公平对比

### 4.3 关键技术决策

| 决策点 | 建议方案 | 理由 |
|--------|---------|------|
| 空间离散 | 保留 Arakawa+FD | 与数据一致 |
| 边界条件 | x 方向特殊处理 | Dirichlet BC |
| Loss 类型 | 默认 mhd_sim | 更稳定 |
| 模型大小 | ~2M 参数 | 与 UNet 对标 |
| 精度 | float64 | 与 mhd_sim 对齐 |
| 多步训练 | unroll_steps=1 | 简单起步 |

### 4.4 风险和应对

| 风险 | 影响 | 应对措施 |
|------|-----|---------|
| 3D FFT 效率低 | 训练慢 | 减小分辨率或使用混合精度 |
| x 方向非周期 | 边界伪影 | 使用 padding 或特殊处理 |
| 5 场耦合复杂 | Loss 不稳定 | 逐步增加场数 |
| 参数量差异大 | 不公平对比 | 调整架构匹配参数量 |

### 4.5 时间估计

| 阶段 | 预计时间 |
|------|---------|
| 阶段 1: 项目结构 | 0.5 天 |
| 阶段 2: 模型架构 | 1-2 天 |
| 阶段 3: Loss 函数 | 0.5 天 |
| 阶段 4: 训练脚本 | 1 天 |
| 阶段 5: 评估脚本 | 0.5 天 |
| 阶段 6: 实验运行 | 2-3 天 |
| **总计** | **~1 周** |

---

## 附录

### A. 文件路径参考

```
mhd_sim/
├── numerical/
│   ├── equations/
│   │   └── five_field_mhd.py    # 5-field MHD 方程
│   ├── operators/
│   │   ├── arakawa.py           # Arakawa 格式
│   │   └── derivatives.py       # FD 算子
│   ├── solvers/
│   │   ├── config.py            # 仿真配置
│   │   └── runner.py            # 仿真运行器
│   └── scripts/
│       └── run_5field_mhd.py    # 仿真脚本
├── plasma_sim/
│   ├── models/
│   │   ├── unet.py              # UNet 模型
│   │   └── omnifluids.py        # [待创建] OmniFluids 模型
│   └── train/
│       ├── train_hw.py          # HW 训练 (参考)
│       ├── eval_hw.py           # HW 评估 (参考)
│       ├── train_5field.py      # [待创建] 5-field 训练
│       └── eval_5field.py       # [待创建] 5-field 评估

OmniFluids/
└── nse2d/
    └── pretrain/
        ├── model.py             # OmniFluids2D (参考)
        ├── train.py             # 训练逻辑 (参考)
        └── psm_loss.py          # PSM Loss (参考)
```

### B. 数据格式

**5-field MHD 轨迹数据** (建议格式):
```python
{
    "U":     (B, T, Nx, Ny, Nz),
    "vpar":  (B, T, Nx, Ny, Nz),
    "n":     (B, T, Nx, Ny, Nz),
    "psi":   (B, T, Nx, Ny, Nz),
    "Ti":    (B, T, Nx, Ny, Nz),
    "phi":   (B, T, Nx, Ny, Nz),  # 辅助
    "j_par": (B, T, Nx, Ny, Nz),  # 辅助
    "t":     (T,),
    "config": {...},
}
```
