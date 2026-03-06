# OmniFluids2D 网络架构详解

## 1. 概述

OmniFluids2D 是一个基于 **Fourier Neural Operator (FNO)** 的神经网络，专门设计用于求解 5 场磁流体动力学 (5-field MHD) 问题。

### 核心特点
- **混合边界条件**: x 方向 Dirichlet BC (DST)，y 方向周期性 BC (FFT)
- **Mixture-of-Experts (MoE)**: 通过物理参数动态调制谱权重
- **多帧预测**: 单次前向传播输出多个时间步
- **独立输出头**: 每个物理场有独立的输出通道

---

## 2. 整体架构流程图

```
输入: (B, Nx, Ny, 5)  ← 5个物理场 [n, U, vpar, psi, Ti]
        ↓
   + Grid坐标 → (B, Nx, Ny, 7)
        ↓
   Input Projection (Linear: 7 → width)
        ↓
      GELU
        ↓
   ┌─────────────────────────────────────┐
   │  SpectralConv2d_MHD + MoE  (× L层)  │  ← 物理参数 → f_ν → softmax → att
   │     DST(x) + FFT(y) + FeedForward   │
   │           + Residual                │
   └─────────────────────────────────────┘
        ↓
      GELU
        ↓
   5× OutputHead (独立)
        ↓
   Stack → (B, Nx, Ny, 5, T)
        ↓
   + x_0 * dt_frac (残差 + 线性插值)
        ↓
输出: (B, Nx, Ny, 5, T)  ← T帧预测
```

---

## 3. 各模块详解

### 3.1 输入处理

```python
# 输入
x: (B, Nx, Ny, 5)         # 5个物理场

# 添加位置编码 (归一化到 [0, 2π))
grid: (1, Nx, Ny, 2)      # [gx, gy] 坐标
x = cat([x, grid])        # → (B, Nx, Ny, 7)

# 线性投影
x = in_proj(x)            # Linear(7 → 80) → (B, Nx, Ny, 80)
x = gelu(x)
```

**物理意义**:
- 位置编码让模型知道每个点的空间位置
- 投影到高维空间 (width=80) 以捕获复杂的场间耦合

---

### 3.2 SpectralConv2d_MHD (核心模块)

每层执行混合边界条件的谱卷积：

```
输入: (B, Nx, Ny, width)
        ↓
  rearrange → (B, width, Nx, Ny)
        ↓
  ┌──────────────────────────────────┐
  │   Y方向: FFT (周期性边界)        │
  │   x_fty = rfft(x, dim=-1)       │
  │   → (B, I, Nx, Ny//2+1) complex │
  │   截取 modes_y 个低频模态        │
  │   × weight_y (MoE调制)          │
  │   → irfft → xy                  │
  ├──────────────────────────────────┤
  │   X方向: DST (Dirichlet边界)    │
  │   x_ftx = dst(x, dim=-2)        │
  │   → (B, I, Nx, Ny) complex      │
  │   截取 modes_x 个低频模态        │
  │   × weight_x (MoE调制)          │
  │   → idst → xx                   │
  └──────────────────────────────────┘
        ↓
  x = xx + xy              # 两个方向的贡献相加
        ↓
  rearrange → (B, Nx, Ny, width)
        ↓
  FeedForward (width → width*4 → width)
        ↓
输出: (B, Nx, Ny, width)
```

#### DST 实现 (Dirichlet BC)

```python
def _dst_forward(x, dim=-2):
    """
    通过奇对称扩展 + rfft 实现 DST
    
    原始信号: [x_0, x_1, ..., x_{N-1}]  (假设 x_0=0, x_{N-1}=0)
    奇扩展:   [x_0, x_1, ..., x_{N-1}, -x_{N-2}, ..., -x_1]
    长度:     N + (N-2) = 2*(N-1)
    """
    N = x.shape[dim]
    interior = x.narrow(dim, 1, N - 2)  # 取 [x_1, ..., x_{N-2}]
    x_ext = torch.cat([x, -torch.flip(interior, [dim])], dim=dim)
    return torch.fft.rfft(x_ext, dim=dim, norm='ortho')
```

**物理意义**:
- Dirichlet BC: 边界处场值为零 (如等离子体边缘)
- DST 自然满足这个约束，不需要额外处理

#### MoE 调制机制

```python
# 物理参数 → 注意力权重
params: (B, 8)              # [eta_i, beta, shear, lam, mass_ratio, Dn, eta, kq]
        ↓
att = f_nu[i](params)       # MLP: 8 → 128 → 128 → K
att = softmax(att / T)      # (B, K) 归一化权重

# 动态组合 K 个专家的权重
weight_y = einsum("bk,kioxy->bioxy", att, fourier_weight[0])  # (B, I, O, modes_y, 2)
weight_x = einsum("bk,kioxy->bioxy", att, fourier_weight[1])  # (B, I, O, modes_x, 2)
```

**物理意义**:
- 不同物理参数配置需要不同的算子
- MoE 让模型根据输入参数自适应选择合适的谱权重组合
- K=4 个专家，温度 T=10 控制 softmax 平滑度

---

### 3.3 Spectral Layer 堆叠 (12层)

```python
for i in range(n_layers):           # n_layers = 12
    att = f_nu[i](params)           # 每层独立的 MoE
    att = softmax(att / T, dim=-1)
    b = spectral_layers[i](x, att)  # 谱卷积
    x = x + b                       # 残差连接

x = gelu(b)  # 注意: 使用最后一层的输出 b，而不是累积的 x
```

**Shape 变化**: 始终保持 `(B, Nx, Ny, width)`

**物理意义**:
- 每层捕获不同尺度的空间相关性
- 残差连接保持梯度流动
- 12 层足够深以学习复杂的非线性 PDE 动力学

---

### 3.4 OutputHead (每场独立)

```python
class OutputHead:
    def __init__(self, width=80, output_dim=10):
        self.fc_a = Linear(80, 36)   # 4*10-4 = 36
        self.fc_b = Linear(80, 34)
        self.mlp = Sequential(
            Conv1d(1, 8, kernel=12, stride=2),  # 输入长度 → (L-12)/2+1
            GELU(),
            Conv1d(8, 1, kernel=12, stride=2),
        )
```

#### 训练模式 (output_dim=10)
```
x: (B, Nx, Ny, 80)
        ↓
fc_a(x): (B, Nx, Ny, 36)
fc_b(x): (B, Nx, Ny, 34)
        ↓
cat → (B, Nx, Ny, 70)
        ↓
gelu
        ↓
reshape → (B*Nx*Ny, 1, 70)
        ↓
Conv1d(1,8,12,s=2): (70-12)/2+1 = 30 → (B*Nx*Ny, 8, 30)
        ↓
Conv1d(8,1,12,s=2): (30-12)/2+1 = 10 → (B*Nx*Ny, 1, 10)
        ↓
reshape → (B, Nx, Ny, 10)
```

#### 推理模式 (output_dim=1)
```
x: (B, Nx, Ny, 80)
        ↓
fc_b(x): (B, Nx, Ny, 34)   # 只用 fc_b
        ↓
gelu
        ↓
reshape → (B*Nx*Ny, 1, 34)
        ↓
Conv1d(1,8,12,s=2): (34-12)/2+1 = 12 → (B*Nx*Ny, 8, 12)
        ↓
Conv1d(8,1,12,s=2): (12-12)/2+1 = 1  → (B*Nx*Ny, 1, 1)
        ↓
reshape → (B, Nx, Ny, 1)
```

**物理意义**:
- 每个物理场有独立的输出头，可以学习场特定的动力学
- 训练时预测多帧 (10)，提供更丰富的监督信号
- 推理时单帧输出，用于自回归 rollout

---

### 3.5 残差连接 + 线性插值

```python
# 多帧输出
out = stack([head(x) for head in output_heads], dim=-2)  # (B, Nx, Ny, 5, T)

# 线性插值系数
dt_frac = [1/T, 2/T, ..., T/T]  # [0.1, 0.2, ..., 1.0] for T=10

# 残差 + 插值
out = x_0.unsqueeze(-1) + out * dt_frac
```

**物理意义**:
- 残差连接: 模型只需学习 `Δx = x(t+dt) - x(t)`，而不是绝对值
- 线性插值: 第 i 帧 ≈ `x_0 + (i/T) * Δx`，强制输出从初始条件平滑过渡
- 这是 **线性外推假设**，假设短时间内变化近似线性

---

## 4. 参数统计

| 模块 | 参数数量 | 公式 |
|------|----------|------|
| in_proj | 7 × 80 + 80 = 640 | Linear(7, 80) |
| f_nu (每层) | 8×128 + 128×128 + 128×4 = 18,432 | MLP(8→128→128→K) |
| f_nu (12层) | 18,432 × 12 = 221,184 | |
| spectral_weight (每层) | K × I × O × (modes_y + modes_x) × 2 = 4×80×80×256×2 = 13,107,200 | |
| spectral_weight (12层) | 13,107,200 × 12 = 157,286,400 | |
| backcast_ff (每层) | 80×320 + 320×80 = 51,200 | width→4×width→width |
| backcast_ff (12层) | 51,200 × 12 = 614,400 | |
| output_heads | 5 × (80×36 + 80×34 + conv_params) ≈ 34,000 | |
| **总计** | **~158M** | |

**主要参数**: 谱权重占 ~99%

---

## 5. 潜在问题分析

### 5.1 ⚠️ 第 256 行: 使用 `b` 而不是 `x`

```python
x = F.gelu(b)  # use last spectral output (original nse2d design)
```

**问题**: 丢弃了前面层的累积信息，只用最后一层的输出
**影响**: 可能限制模型表达能力
**建议**: 考虑改为 `x = F.gelu(x)` 使用完整的残差累积

### 5.2 ⚠️ OutputHead 的硬编码维度

```python
self.fc_b = nn.Linear(width, 34)  # 硬编码 34
```

**问题**: 34 是为了配合 Conv1d 输出 1（推理）或 10（训练），但不够灵活
**影响**: 修改 `output_dim` 时需要手动调整
**建议**: 动态计算所需维度

### 5.3 ⚠️ DST 实现的边界假设

```python
interior = x.narrow(dim, 1, N - 2)  # 假设 x[0] 和 x[N-1] 已经是 0
```

**问题**: 假设输入已满足 Dirichlet BC
**影响**: 如果输入不满足（如 GRF），边界处会有数值误差
**建议**: 显式强制 `x[:, 0, :, :] = 0` 和 `x[:, -1, :, :] = 0`

### 5.4 ⚠️ MoE 权重没有共享

```python
self.fourier_weight = None  # 每层独立的权重
```

**问题**: 每层有独立的 K×I×O×modes×2 参数，总共 ~157M
**影响**: 参数量大，训练慢
**建议**: 考虑跨层共享部分权重

### 5.5 ⚠️ 线性插值假设

```python
out = x_0.unsqueeze(-1) + out * dt_frac
```

**问题**: 假设 `x(t) = x_0 + t * f(x_0)`，是一阶 Taylor 展开
**影响**: 对于高度非线性动力学，短时间内可能不准确
**建议**: 考虑更高阶的时间离散化

### 5.6 ✅ 合理的设计

- **混合 BC 处理**: DST + FFT 正确处理了 Dirichlet + Periodic 边界
- **独立输出头**: 允许每个物理场学习不同的动力学
- **MoE 机制**: 使模型能够适应不同的物理参数配置
- **残差连接**: 有助于梯度流动和训练稳定性

---

## 6. 建议改进

### 6.1 修复第 256 行

```python
# 当前
x = F.gelu(b)

# 建议
x = F.gelu(x)  # 使用完整的残差累积
```

### 6.2 动态 OutputHead 维度

```python
def __init__(self, width, output_dim):
    # 计算需要的输入维度使 Conv1d 输出 output_dim
    # Conv1d(k=12, s=2): out = (in - 12) / 2 + 1
    # 两层: out2 = ((in - 12) / 2 + 1 - 12) / 2 + 1
    # 逆推: in = 4 * output_dim - 4 + 34 = 4 * output_dim + 30 (for T > 1)
    ...
```

### 6.3 显式边界强制

```python
def forward(self, x, ...):
    # 强制 Dirichlet BC
    x = x.clone()
    x[:, 0, :, :] = 0
    x[:, -1, :, :] = 0
    ...
```

---

## 7. 总结

OmniFluids2D 是一个精心设计的 PDE 求解器，核心创新点：

1. **混合边界条件谱方法**: DST + FFT 正确处理物理边界
2. **MoE 机制**: 根据物理参数动态调整算子
3. **多帧预测**: 训练时提供更丰富的监督信号
4. **残差 + 插值**: 简化学习任务

主要瓶颈是 **谱权重参数量大** (~157M/158M)，可以考虑权重共享或低秩分解来优化。
