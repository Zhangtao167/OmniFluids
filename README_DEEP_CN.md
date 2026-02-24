# OmniFluids 代码深度导读与复现迁移指南

> 本文档基于仓库所有源代码逐行比对整理，保证每个 shape、公式、调用链与代码一致。

## 1. 仓库结构

```
OmniFluids/
├── kse2d/           # 2D Kuramoto-Sivashinsky
│   ├── data/        # 数据生成
│   ├── pretrain/    # 纯物理预训练
│   ├── distillation/# 教师->学生蒸馏
│   └── finetune/    # 少样本微调
├── nse2d/           # 2D Navier-Stokes（涡量形式）
│   └── (同上四个子目录)
├── requirements.txt
└── README.md
```

两条并行主线（KSE / NSE），结构完全对称，差异仅在：条件参数形式、PSM 物理残差公式、数据生成求解器。

## 2. 复现命令

```bash
pip install -r requirements.txt   # einops, matplotlib, numpy, thop, torch

# KSE 全流程
cd kse2d/data      && python main.py
cd ../pretrain     && python main.py
cd ../distillation && python main.py --model_path ../pretrain/model/<name>.pt
cd ../finetune     && python main.py --model_path ../distillation/model/<name>.pt
```

NSE 同理，目录换为 `nse2d/*`。

产物：各阶段 `model/*.pt`（权重）和 `log/log_*/*.csv`（日志）。

---

## 3. 数据生成（精确到每一步）

### 3.1 KSE 数据

#### 3.1.1 PDE 与求解器 (`kse.py::ks_2d_rk4`)

方程（docstring 原始形式）：`du/dt + param1*(Delta u + Delta^2 u) + 0.5*param2*|grad u|^2 = 0`

代码实现细节：

- **空间域 `[0, 8π] x [0, 8π]`，周期边界**（⚠️ 不是 `[0,2π]`，见下方推导）
- **波数有 1/4 缩放**：`kx = (1/4) * fftfreq(N, d=1/N)`，物理波数 = 整数波数 x 1/4
- 拉普拉斯：`lap = -(kx^2 + ky^2)` （**负号**，即 lap 代表 `-Delta`）
- 线性算子（RK4 右端）：`L = -(lap + lap^2) * param1`，展开后 = `(kx^2+ky^2 - (kx^2+ky^2)^2) * param1`
- 非线性项（物理空间）：`ux = ifft2(1j*kx*fft2(u)).real`，`uy` 同理，然后 `-0.5*param2*fft2(ux^2+uy^2)` 乘 dealias 掩码（2/3 规则去混叠）
- 时间推进：经典 RK4，频域中 `u_h += (k1+2*k2+2*k3+k4)/6`

> **域大小与波数缩放的数学关系（关键知识）**
>
> 谱方法在周期域 `[0, L]` 上的 Fourier 基函数是 `exp(2πinx/L)`，
> 对空间求导的乘子为 `i · (2πn/L)`，其中 `n` 是整数模态序号。
> `torch.fft.fftfreq(N, d=1/N)` 返回的恰好是整数 `n = [0,1,...,N/2-1, -N/2,...,-1]`。
> 代码中用 `ifft2(i*kx*fft2(u))` 计算 `du/dx`，因此 `kx` 就是物理波数 `2πn/L`。
>
> | PDE | 代码中的 kx | 等式 `kx = 2πn/L` | 解出 L |
> |-----|-----------|-------------------|--------|
> | **KSE** | `(1/4) * n` | `n/4 = 2πn/L` | **L = 8π** |
> | **NSE** | `n`（整数） | `n = 2πn/L` | **L = 2π** |
>
> **为什么 KSE 选 8π 而不是 2π？**
>
> KSE 方程 `∂u/∂t + param1·(Δu + Δ²u) + 0.5·param2·|∇u|² = 0`
> 在 param1=param2=1 时，线性色散关系为 `σ(k) = k² - k⁴`，
> 当 `|k| < 1` 时模态线性不稳定（`σ > 0`）。
>
> - 在 `[0, 2π]` 上，物理波数 = 整数 n，最小非零 |k| = 1，恰好在临界点，**几乎无不稳定模态**
> - 在 `[0, 8π]` 上，物理波数 = n/4，最小非零 |k| = 1/4，**|k| < 1 的不稳定模态有 n = 1,2,3**（每个方向），足以激发时空混沌
>
> 域越大，不稳定模态越多，动力学越丰富。8π 是一个经典选择，能在可控分辨率下产生充分的混沌行为。
>
> **模型中 `get_grid` 的 `[0, 2π)` 坐标是什么？**
>
> 模型的 `get_grid` 对 KSE 和 NSE 都生成 `[0, 2π)` 归一化坐标——这只是神经网络的**位置编码**，
> 不等同于 PDE 物理域大小。Neural Operator 用坐标通道告诉网络"每个点在哪"即可，
> 不需要与物理尺度严格对应。对于 KSE，物理坐标应是 `[0, 8π)` 但网格编码用 `[0, 2π)` 作归一化表示。

关键参数：`dt=1e-5, T=5.0, record_steps = int(record_ratio * T) = 50`

输出 shape：`[batch, N, N, record_steps+1]`（含初始帧，即 51 帧）

#### 3.1.2 初值采样器 (`sampler.py::Init_generation`)

高斯随机场（GRF），通过频域着色白噪声生成：

```
sqrt_eig[k1,k2] = s1*s2 * sigma * (const1*k1^2 + const2*k2^2 + tau^2)^(-alpha/2)
sqrt_eig[0,0] = 0   # 零均值
```

调用：`xi ~ N(0,1) shape [N, s, s//2+1, 2]` -> `xi *= sqrt_eig` -> `u = irfft2(complex(xi))` -> shape `[N, s, s]`

**关键发现：不同阶段的 GRF 参数不同！**

| 位置 | alpha | tau | sigma 公式 |
|------|-------|-----|-----------|
| `kse2d/data/sampler.py` | 4 | 8.0 | **0.5** * tau^(alpha-1) |
| `kse2d/pretrain/tools.py` | 4 | 8.0 | tau^(alpha-1)（**无 0.5**） |
| `kse2d/distillation/tools.py` | 4 | 8.0 | tau^(alpha-1)（**无 0.5**） |
| `kse2d/finetune/tools.py` | **2.5** | **7.0** | 0.5 * tau^(alpha-1) |
| `nse2d/*/tools.py` 及 `sampler.py` | 2.5 | 7.0 | 0.5 * tau^(alpha-1) |

这意味着：预训练/蒸馏的在线采样初值振幅是数据生成时的 **2 倍**（无 0.5 系数）。微调和 NSE 用更平滑的初值（alpha=2.5 < 4）。

#### 3.1.3 生成流程 (`generate_data.py::generate_ks_data`)

```
按 mode 固定 seed 和样本数：train seed=0 N=2, test seed=1 N=10, val seed=2 N=2
-> w0 = GRF(bsize)                     [bsize, 128, 128]
-> param[:, 0] ~ U(param1[0], param1[1])
   param[:, 1] ~ U(param2[0], param2[1])   [bsize, 2]
-> sol = ks_2d_rk4(w0, T, param, dt, record_steps)  [bsize, 128, 128, 51]
-> sol = sol[:, ::sub, ::sub, :]       (空间下采样, 默认 sub=1)
-> 保存 dataset/{name}     [N, 128, 128, 51]
        dataset/param_{name}  [N, 2]
```

`main.py` 硬编码生成三组：`test_multi`(param 范围), `*_0.2_0.5`(固定), `*_1_1`(固定)。

### 3.2 NSE 数据

#### 3.2.1 PDE 与求解器 (`nse.py::navier_stokes_2d`)

涡量-流函数形式：`dw/dt + u*grad(w) = nu*Delta(w) + f`

- **空间域 `[0, 2π] x [0, 2π]`，周期边界**（波数 = 整数 n，与 `2πn/L = n` → `L=2π` 一致）
- `psi_h = w_h / lap`（泊松求流函数，`lap[0,0]=1` 防除零）
- 速度：`u_x = iky*psi_h`, `u_y = -ikx*psi_h`
- 对流项物理空间算后回频域，dealias
- 时间推进 **Crank-Nicolson 半隐式**：`w_h^{n+1} = (-dt*F_h + dt*f_h + (1-0.5*dt*nu*lap)*w_h) / (1+0.5*dt*nu*lap)`

参数：`s=1024, sub=4, dt=1e-4, T=10.0, record_ratio=10 -> record_steps=100`

#### 3.2.2 Forcing 采样器 (`sampler.py::Force_generation`)

```
f(x,y) = (1/max_freq) * sum_{u,v=1}^{max_freq} [a_r*cos(ux+vy) + a_i*sin(ux+vy)]
其中 a_r, a_i ~ U(-amplitude, amplitude)
```

默认 `max_frequency=10, amplitude=0.5`。

#### 3.2.3 黏性采样（精确公式）

```python
lognu_min = log10(re[0])      # 如 log10(500) ~ 2.699
lognu_max = log10(re[1])      # 如 log10(2500) ~ 3.398
visc = -lognu_min - (lognu_max - lognu_min) * rand()
# visc 范围 [-lognu_max, -lognu_min]
# 实际物理黏性 nu = 10^visc 范围 [1/re_max, 1/re_min]
```

**param 中存的是 visc（-log10(Re) 尺度），传给 PSM_loss 时做 `10**visc` 转成物理黏性。**

#### 3.2.4 保存格式

```
dataset/data_{name}   [N, 256, 256, 101]   (s/sub=1024/4=256)
dataset/f_{name}      [N, 256, 256, 2]
  [..., 0] = forcing 场
  [..., 1] = visc 标量广播成全场
```

---

## 4. 模型架构（逐层 shape 追踪）

### 4.1 OmniFluids2D（教师模型）

以 KSE 默认 `B=20, S=64, width=64, modes=32, n_layers=8, output_dim=100, K=4` 为例。

#### 4.1.1 输入拼接

```
x                                [20, 64, 64, 1]   当前物理场
grid = get_grid(...)             [1, 64, 64, 2]    坐标 in [0, 2pi)
cat(x, grid.repeat(B,...))       [20, 64, 64, 3]

KSE: param [20,2] -> reshape+repeat -> [20,64,64,2]
cat -> [20, 64, 64, 5]

NSE: param 本身 [B,S,S,2] 直接 cat -> [B, S, S, 5]

in_proj(x)                      [20, 64, 64, 64]   Linear(5 -> width)
gelu(x)                         [20, 64, 64, 64]
```

#### 4.1.2 条件专家路由（每层独立的 MLP）——Mixture of Operators 的核心

**设计动机：** 不同物理参数（如 KSE 的 param1/param2、NSE 的黏性 nu）对应不同的 PDE 动力学行为。
一个固定的 Fourier 卷积核无法同时适配所有参数区间。因此引入 **K 组并行 Fourier 核**（"专家"），
由一个以物理参数为输入的小型 MLP 动态决定每组专家的混合权重（"路由"），实现参数自适应。

**每层一个独立 MLP（不共享权重）：**
`f_nu` 是一个 `ModuleList`，长度 = `n_layers`（默认 16 层），每个元素是一个独立的 3 层 MLP。
不共享权重意味着每层 Fourier 块可以学到不同的参数-专家映射策略——浅层可能更关注低频特征的参数依赖，
深层可能更关注非线性结构的参数依赖。

**网络结构与 shape 追踪：**

```
KSE: f_nu[i] = Sequential(
    Linear(2 -> 128),    # 输入: PDE 参数 param [B, 2]，即 (param1, param2)
    GELU(),
    Linear(128 -> 128),
    GELU(),
    Linear(128 -> K)     # 输出: 原始 logits [B, K]，K=4（默认）
)

NSE: f_nu[i] = Sequential(
    Linear(1 -> 128),    # 输入: 仅黏性系数 nu [B, 1]，从 param[:,0,0,1:2] 提取
    GELU(),
    Linear(128 -> 128),
    GELU(),
    Linear(128 -> K)     # 输出: 原始 logits [B, K]
)
```

**调用与输出含义：**

```python
# forward() 循环体内（model.py 第 126~132 行）:
fc = self.f_nu[i]             # 取第 i 层的路由 MLP
att = fc(param)               # [B, 2] -> [B, K]   原始 logits（未归一化，有正有负）
att = F.softmax(att / self.T, dim=-1)
# att shape: [B, K]
# att 含义: batch 中每个样本对 K 个专家的混合权重，和为 1
# att[b, k] 表示 "样本 b 的第 k 个专家的贡献比例"
```

`att` 随后传入 `SpectralConv2d_dy.forward_fourier`，用于混合 K 组 Fourier 权重：

```python
# SpectralConv2d_dy.forward_fourier（model.py 第 57~58 行）:
weight = torch.einsum("bk, kioxy->bioxy", att, self.fourier_weight[0])
# fourier_weight[0] shape: [K, width, width, modes, 2]
#   K 组专家，每组是一个 [width, width, modes] 的复数核
# att shape: [B, K]
# 输出 weight shape: [B, width, width, modes, 2]
#   对每个样本 b: weight[b] = sum_k att[b,k] * fourier_weight[k]
#   即：加权平均 K 组专家核，得到该样本专属的动态 Fourier 核
```

**效果总结：同一层的 K 个 Fourier 核被 softmax 权重线性混合，每个样本根据自己的物理参数得到一个"专属"核。**

**温度参数 T 的作用：**

```
T = 10（默认）: softmax(logits/10) → 权重非常平滑，接近均匀分布 [0.25, 0.25, 0.25, 0.25]
  → 各专家均匀分担，混合核≈所有核的平均
T → 0: softmax 退化为 argmax，只激活权重最大的一个专家 → 硬路由
T = 1: 标准 softmax，权重分布取决于 logits 的差异大小
```

T=10 这个较大的温度使训练初期各专家都能得到梯度（避免"赢者通吃"），
随着训练推进，不同参数区间会自然产生略有差异的权重分布。

**KSE 与 NSE 的输入差异：**

| | KSE | NSE |
|---|---|---|
| 路由输入 | `param` [B, 2] = (param1, param2) | `nu` [B, 1] = param[:,0,0,1:2]（仅黏性） |
| 输入维度 | `Linear(2->128)` | `Linear(1->128)` |
| 为什么 | KSE 有两个独立可变参数 | NSE 中 forcing 是空间场已通过 in_proj 编码，只有 nu 是全局标量需要路由 |

**迁移提示：** 如果你的目标 PDE 有 M 个全局标量参数（如 Re, Pr, Ma 等），
将 `f_nu` 的第一层改为 `Linear(M -> 128)` 即可。如果参数是空间场，
建议仍提取其全局统计量（均值、方差等）作为标量输入路由 MLP。

#### 4.1.3 动态频域层 SpectralConv2d_dy（分离式 1D rFFT + MoE）

**设计动机：** 标准 FNO 用一次 rFFT2 做二维频域卷积，参数量 `O(width² × modes_x × modes_y)`。
本代码改用**分离式设计**：对 y 方向和 x 方向分别做一次 1D rFFT，再相加。
这将参数量降为 `O(2 × width² × modes)`，同时保持了对两个方向频谱特征的独立建模能力。

每层维护 2 组权重（y 方向 + x 方向）。每组 shape `[K, width, width, modes, 2]`（2 = 复数实虚部）。

**完整前向流程（以 KSE 默认 B=20, S=64, width=64, modes=32, K=4 为例）：**

```
输入: x [20, 64, 64, 64]   （来自残差累加），att [20, 4]（来自 f_nu 的 softmax 路由权重）

┌─────────────────── forward_fourier ───────────────────┐
│                                                        │
│  ① 转置为通道优先:                                       │
│     x = rearrange('b m n i -> b i m n')                │
│     x: [20, 64, 64, 64]  (B, width, S_y, S_x)        │
│                                                        │
│  ② 混合专家权重（两个方向各做一次）:                        │
│     weight_y = einsum("bk, kioxy->bioxy",              │
│                       att, fourier_weight[0])           │
│     att:             [20, 4]                            │
│     fourier_weight:  [4, 64, 64, 32, 2]  (K组专家核)    │
│     weight_y:        [20, 64, 64, 32, 2]  (混合后的核)   │
│     → view_as_complex → [20, 64, 64, 32] 复数           │
│                                                        │
│     物理含义: 每个样本根据自己的 PDE 参数，                  │
│     得到一个"专属"频域线性变换核                             │
│                                                        │
│  ③ y 方向处理:                                           │
│     x_fty = rfft(x, dim=-1, norm='ortho')              │
│     x_fty: [20, 64, 64, 33]  (S_x//2+1=33 个频率分量)   │
│                                                        │
│     截取低频 modes 个模态做矩阵乘:                         │
│     out_ft = zeros [20, 64, 64, 33]                    │
│     out_ft[:,:,:,:32] = einsum("bixy,bioy->boxy",      │
│         x_fty[:,:,:,:32], weight_y_complex)             │
│     含义: 对每个空间位置(y行)的前32个频率分量,               │
│           做一次 width→width 的通道混合                    │
│                                                        │
│     xy = irfft(out_ft, n=64, dim=-1)                   │
│     xy: [20, 64, 64, 64]  回到物理空间                   │
│                                                        │
│  ④ x 方向处理（完全对称，对 dim=-2 做 rfft）:               │
│     x_ftx = rfft(x, dim=-2, norm='ortho')              │
│     x_ftx: [20, 64, 33, 64]  (S_y//2+1=33)            │
│     out_ft[:,:,:32,:] = einsum(x_ftx[:,:,:32,:], w_x)  │
│     xx = irfft(out_ft, n=64, dim=-2)                   │
│     xx: [20, 64, 64, 64]                               │
│                                                        │
│  ⑤ 两方向相加 + 转回通道末尾:                              │
│     x = xx + xy                  [20, 64, 64, 64]      │
│     x = rearrange('b i m n -> b m n i')                │
│     x: [20, 64, 64, 64]                                │
│                                                        │
│     物理含义: x 方向和 y 方向的频域特征变换各自独立完成后,     │
│     在物理空间叠加——类比分离式有限差分中 x/y 算子可加性        │
│                                                        │
└────────────────────────────────────────────────────────┘

┌─────────────────── backcast_ff (后处理 MLP) ──────────────┐
│                                                           │
│  b = backcast_ff(x)                                       │
│    FeedForward(width=64, factor=4, n_layers=2):           │
│      Linear(64 -> 256) -> ReLU                            │
│      Linear(256 -> 64) -> LayerNorm(64)                   │
│    b: [20, 64, 64, 64]                                    │
│                                                           │
│    物理含义: 逐点 MLP 对频域变换后的特征做非线性混合,           │
│    LayerNorm 稳定训练。这是频域线性变换后的"非线性补偿"        │
│                                                           │
└───────────────────────────────────────────────────────────┘

最终输出: b [20, 64, 64, 64]
```

**关键细节：**
- `norm='ortho'` 使 FFT 和 IFFT 各乘 `1/sqrt(N)`，避免幅度因 N 膨胀
- 高于 `modes` 的频率分量被置零（隐式低通滤波），`modes` 越大保留越多高频细节
- `einsum("bixy,bioy->boxy")` 本质是在频域对 `width` 个通道做线性变换（b=batch, i/o=in/out channel, x=freq, y=spatial）

#### 4.1.4 残差连接循环（关键细节！）

```python
for i in range(n_layers):        # 8 层
    att = softmax(f_nu[i](param) / T)
    b = spectral_layers[i](x, att)    # [B, S, S, width]
    x = x + b                         # 残差累加
# 循环结束后:
x = F.gelu(b)    # !!!用的是最后一层的 b，不是累加后的 x!!!
```

**这意味着：经过所有层残差累加的 `x` 在此处被丢弃，后续输出头只看最后一层频域块的输出 `b` 经过 GELU 激活的结果。前面层的信息只通过影响最后一层的输入间接作用。**

#### 4.1.5 多帧输出头（训练模式 inference=False）

**设计动机：** 预训练时网络需要一次输出多帧（如 100 帧），以便 PSM Loss 在时间维上计算 PDE 残差。
但如果直接用 `Linear(64 -> 100)`，参数效率低且无法建模帧间时间结构。
因此设计了**先升维再用 1D 卷积沿时间维压缩**的方案，让卷积核学到帧间的时间局部模式。

**完整 shape 追踪（KSE 默认 output_dim=100, width=64）：**

```
x: [B, S, S, 64]   (来自 gelu(b)，最后一层频域块的输出经激活)

┌─ 训练模式 (inference=False) ──────────────────────────────┐
│                                                           │
│  ① 两路 Linear 并行升维:                                    │
│     fc1a(x) = Linear(64 -> 396)   → [B, S, S, 396]       │
│     fc1b(x) = Linear(64 -> 34)    → [B, S, S, 34]        │
│     concat(fc1a, fc1b, dim=-1)    → [B, S, S, 430]       │
│     gelu                          → [B, S, S, 430]       │
│                                                           │
│     为什么是 396+34=430？                                   │
│     Conv1d(k=12,s=2) 的输出长度: out=(in-12)/2+1           │
│     两层串联: out=((in-12)/2+1-12)/2+1                     │
│     令 out=100(output_dim): 反解 in=430                    │
│     分成两个 Linear 是为了复用 fc1b 给推理模式                │
│                                                           │
│  ② 逐像素 1D 卷积沿"时间种子维"压缩:                         │
│     reshape → [B*S*S, 1, 430]   (把空间展平，时间维=430)     │
│                                                           │
│     Conv1d(in_ch=1, out_ch=8, kernel=12, stride=2)        │
│       → [B*S*S, 8, 210]   out_len=(430-12)/2+1=210       │
│     GELU                                                  │
│     Conv1d(in_ch=8, out_ch=1, kernel=12, stride=2)        │
│       → [B*S*S, 1, 100]   out_len=(210-12)/2+1=100       │
│                                                           │
│     reshape → [B, S, S, 100]                              │
│                                                           │
│     物理含义: 每个空间点输出 100 个时间帧的增量预测。           │
│     Conv1d 的 kernel=12 意味着每个输出帧感受 12 个"种子特征", │
│     让相邻帧之间共享信息，保证时间连续性                        │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ 推理模式 (inference=True) ──────────────────────────────┐
│                                                          │
│  ① 只用 fc1b:                                             │
│     fc1b(x) = Linear(64 -> 34)    → [B, S, S, 34]       │
│     gelu                          → [B, S, S, 34]       │
│                                                          │
│  ② 同一个 output_mlp:                                     │
│     reshape → [B*S*S, 1, 34]                             │
│     Conv1d(1->8, k=12, s=2) → [B*S*S, 8, 12]            │
│     GELU                                                 │
│     Conv1d(8->1, k=12, s=2) → [B*S*S, 1, 1]             │
│     reshape → [B, S, S, 1]                               │
│                                                          │
│     物理含义: 只输出下一个时间步的场增量                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### 4.1.6 时间权重增量输出（线性内插先验）

```python
# model.py 第 140~141 行
dt = arange(1, Tp+1) / Tp     # [1/Tp, 2/Tp, ..., 1.0]
x = x_o + x * dt              # x_o: [B,S,S,1], x: [B,S,S,Tp]
```

**逐帧展开（以 Tp=100 为例）：**

```
frame[0] = x_o + delta[0] * (1/100)     第1帧：增量贡献极小，≈初始场
frame[49] = x_o + delta[49] * (50/100)   中间帧：增量贡献一半
frame[99] = x_o + delta[99] * (100/100)  最后帧：增量完全体现
```

**物理含义：** 这是一个**线性时间先验**——假设物理场在短时间内近似线性演化。
早期帧的增量权重小，保证预测不会偏离初始场太远；
后期帧的增量权重大，允许更大的变化。
这比直接输出绝对值更稳定，因为网络只需学习**偏离初始场的残差**。

**推理模式（Tp=1）：** `dt = [1.0]`，即 `x = x_o + delta * 1.0`，等价于无权重的残差连接。

### 4.2 Student2D（学生模型）

与教师差异对照：

| 对比项 | OmniFluids2D（教师） | Student2D（学生） |
|-------|---------------------|------------------|
| 输出头 | fc1a+fc1b+Conv1d 多帧 | Linear(width->4*width)->GELU->Linear(4*width->1) |
| 输出帧数 | output_dim(训练) / 1(推理) | 始终 1 |
| 增量形式 | `x_o + delta * dt_ramp` | **`x_o + delta`（无时间权重）** |
| 典型分辨率 | size=64(KSE) / 256(NSE) | student_size=32(KSE) / 128(NSE) |

---

## 5. PSM Loss（物理结构匹配损失）——无监督训练的核心

**核心思想：** 不需要任何真实轨迹数据。网络输出一段时间序列后，将其代入 PDE 残差：
如果输出完美满足方程，残差为零。训练目标就是让残差 → 0。

### 5.1 KSE 的 PSM（`kse2d/pretrain/psm_loss.py`）

**调用入口：** `PSM_loss(u, param, t_interval, loss_mode)` → 标量 loss

**函数调用链：** `PSM_loss` → `PSM_KS` → 返回残差张量 → 外层算 RMSE

**完整 shape 追踪（KSE 默认 B=20, S=64, T=101）：**

```
输入:
  u:          [20, 64, 64, 101]  网络输出的完整轨迹（w0 + 100帧预测）
  param:      [20, 2]            (param1, param2) PDE 系数
  t_interval: 0.2                rollout_DT，即 101 帧覆盖的物理时间

┌─────────────── PSM_KS 内部 ──────────────────────────────┐
│                                                           │
│  ① FFT 变换到频域（空间维，时间维保留）:                      │
│     w_h = fft2(u, dim=[1,2])                              │
│     w_h: [20, 64, 64, 101] 复数                           │
│     物理含义: 每帧独立做空间 FFT，得到频域表示                  │
│                                                           │
│  ② 构造波数（注意 1/4 缩放 → 域 [0,8π]）:                   │
│     k_x = (1/4) * [0,1,...,31, -32,...,-1]                │
│     k_x: reshape → [1, 64, 64, 1]  (广播到 batch 和时间)   │
│     k_y: 同理                                              │
│                                                           │
│  ③ 拉普拉斯算子:                                            │
│     lap = -(k_x² + k_y²)                                 │
│     lap: [1, 64, 64, 1]                                   │
│     物理含义: 负拉普拉斯 -Δ 的频域表示                        │
│                                                           │
│  ④ 线性项（频域，含 w_h）:                                   │
│     L = (lap + lap²) * param[:,0].reshape(-1,1,1,1) * w_h │
│     展开: (lap + lap²) = (-k² + k⁴)                       │
│     L: [20, 64, 64, 101] 复数                              │
│     物理含义: param1·(Δu + Δ²u) 的频域表示，                 │
│     即 KSE 方程中耗散项 + 超扩散项作用于每帧                   │
│                                                           │
│  ⑤ 非线性项（物理空间计算再回频域）:                           │
│     wx = ifft2(1j·k_x·w_h, dim=[1,2]).real                │
│     wy = ifft2(1j·k_y·w_h, dim=[1,2]).real                │
│     wx, wy: [20, 64, 64, 101]  每帧的空间梯度               │
│     物理含义: ∂u/∂x 和 ∂u/∂y                               │
│                                                           │
│     N = 0.5 * param[:,1].reshape(-1,1,1,1)                │
│         * fft2(wx² + wy², dim=[1,2])                      │
│     N: [20, 64, 64, 101] 复数                              │
│     物理含义: 0.5·param2·|∇u|² 的频域表示                   │
│     ⚠️ 与求解器不同，PSM 中 **无 dealias 掩码**              │
│                                                           │
│  ⑥ PDE 空间算子合并 → 物理空间:                              │
│     Du = ifft2(L + N, dim=[1,2]).real                     │
│     Du: [20, 64, 64, 101]                                 │
│     物理含义: 每帧的 param1·(Δu+Δ²u) + 0.5·param2·|∇u|²   │
│     即 PDE 右端项（不含 ∂u/∂t）                              │
│                                                           │
│  ⑦ 时间导数（cn = Crank-Nicolson 模式）:                    │
│     dt_step = 0.2 / (101-1) = 0.002                      │
│     wt = (u[...,1:] - u[...,:-1]) / dt_step              │
│     wt: [20, 64, 64, 100]   前向差分近似 ∂u/∂t             │
│                                                           │
│  ⑧ 组装完整 PDE 残差:                                       │
│     Du1 = wt + 0.5*(Du[...,:-1] + Du[...,1:])            │
│     Du1: [20, 64, 64, 100]                                │
│     物理含义: ∂u/∂t + 0.5*(F(u_n) + F(u_{n+1})) ≈ 0       │
│     这是 Crank-Nicolson 格式：对空间算子取相邻帧的平均，       │
│     比前向 Euler 更精确（二阶），比中点法少一帧边界损失          │
│                                                           │
│  返回 Du1: [20, 64, 64, 100]                               │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─────────────── PSM_loss 外层 ────────────────────────────┐
│                                                           │
│  loss = sqrt( mean(Du1²) + 1e-7 )                        │
│  = sqrt( (1/(20·64·64·100)) · Σ Du1² + 1e-7 )           │
│                                                           │
│  物理含义: 所有时空点 PDE 残差的 RMS（均方根）               │
│  +1e-7 防止残差恰好为 0 时 sqrt 梯度爆炸                    │
│  完美满足 PDE → loss ≈ 1e-3.5 ≈ 0.0003                   │
│                                                           │
│  返回: 标量 loss                                           │
└───────────────────────────────────────────────────────────┘
```

**mid 模式**（中心差分）：`wt = (u[...,2:] - u[...,:-2]) / (2*dt)`，残差 shape `[B,S,S,T-2]`（首尾各丢一帧）。
精度同为二阶，但不对空间项取平均——适合空间算子计算代价高时减少一半计算量。

### 5.2 NSE 的 PSM（`nse2d/pretrain/psm_loss.py`）

**调用入口：** `PSM_loss(u, forcing, v, t_interval, loss_mode)` → 标量 loss

**函数调用链：** `PSM_loss` → `PSM_NS_vorticity` → 返回残差张量 → 外层减去 forcing 算 RMSE

**完整 shape 追踪（NSE 默认 B=10, S=256, T=51）：**

```
输入:
  u:        [10, 256, 256, 51]  涡量时间序列
  forcing:  [10, 256, 256, 1]   外力场（时间不变）
  v:        [10]                物理黏性 nu = 10^visc
  t_interval: 0.2

┌────────── PSM_NS_vorticity 内部 ──────────────────────────┐
│                                                            │
│  ① FFT:                                                    │
│     w_h = fft2(u, dim=[1,2])      [10, 256, 256, 51] 复数  │
│                                                            │
│  ② 波数（无缩放 → 域 [0,2π]）:                               │
│     k_x, k_y: 整数波数 [1, 256, 256, 1]                    │
│     lap = k_x² + k_y²   ← ⚠️ 正号！（与 KSE 的负号不同）    │
│     lap[0,0,0,0] = 1.0  ← 防除零                           │
│                                                            │
│  ③ 从涡量反算速度（泊松方程 → 流函数 → 速度）:                 │
│     psi_h = w_h / lap              流函数 ψ = Δ⁻¹ω          │
│     ux_h  = 1j·k_y·psi_h          u_x = ∂ψ/∂y             │
│     uy_h  = -1j·k_x·psi_h         u_y = -∂ψ/∂x            │
│     物理含义: 不可压缩流的速度由流函数唯一确定                   │
│                                                            │
│  ④ 涡量梯度和拉普拉斯:                                       │
│     wx_h   = 1j·k_x·w_h           ∂ω/∂x                   │
│     wy_h   = 1j·k_y·w_h           ∂ω/∂y                   │
│     wlap_h = -lap·w_h             Δω（注意负号！因 lap>0）   │
│                                                            │
│  ⑤ 转物理空间（用 irfft2 截半频加速）:                        │
│     ux   = irfft2(ux_h[:,:,:129])      [10, 256, 256, 51]  │
│     uy   = irfft2(uy_h[:,:,:129])      同上                 │
│     wx   = irfft2(wx_h[:,:,:129])      同上                 │
│     wy   = irfft2(wy_h[:,:,:129])      同上                 │
│     wlap = irfft2(wlap_h[:,:,:129])    同上                 │
│                                                            │
│  ⑥ 组装空间算子（物理空间）:                                  │
│     Du = ux·wx + uy·wy - v·wlap                            │
│     Du: [10, 256, 256, 51]                                 │
│     物理含义: u·∇ω - ν·Δω（对流项 - 扩散项）                 │
│     ⚠️ 注意：此处 **不含 forcing**，forcing 在外层减去         │
│                                                            │
│  ⑦ 时间导数 + CN 平均:                                      │
│     dt = 0.2 / 50 = 0.004                                 │
│     wt = (u[...,1:] - u[...,:-1]) / dt    [10,256,256,50] │
│     Du1 = wt + 0.5*(Du[...,:-1] + Du[...,1:])             │
│     Du1: [10, 256, 256, 50]                                │
│     物理含义: ∂ω/∂t + u·∇ω - ν·Δω 的 CN 离散               │
│                                                            │
│  返回 Du1: [10, 256, 256, 50]                               │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌────────── PSM_loss 外层（forcing 在这里减去！） ──────────────┐
│                                                              │
│  f = forcing.reshape(-1,256,256,1).repeat(1,1,1,50)         │
│  f: [10, 256, 256, 50]                                      │
│                                                              │
│  loss = sqrt( mean((Du1 - f)²) + 1e-7 )                    │
│                                                              │
│  物理含义: ∂ω/∂t + u·∇ω - ν·Δω - f ≈ 0                     │
│  完整 NSE 涡量方程的残差 RMS                                  │
│                                                              │
│  ⚠️ 关键设计: forcing 不在 PSM_NS_vorticity 内减去,           │
│  而在 PSM_loss 中做 (Du1-f)，这样 PSM_NS_vorticity 可以      │
│  复用于无 forcing 场景                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**KSE vs NSE PSM 差异总结：**

| | KSE | NSE |
|---|---|---|
| 波数缩放 | k = n/4（域 8π） | k = n（域 2π） |
| lap 符号 | `-(k²+k²)` 负 | `k²+k²` 正 |
| 需要反算速度？ | 否（标量方程） | 是（涡量→流函数→速度）|
| Forcing | 无 | `loss = RMSE(Du1 - f)` |
| Dealias | 无（与求解器不同！） | 无 |
| FFT 类型 | fft2（全频）| fft2 + irfft2（半频优化）|

---

## 6. 训练流程详解——从数据采样到 loss 的完整链路

### 6.1 KSE 预训练：纯物理驱动，无真实数据

**文件：** `kse2d/pretrain/train.py::train`
**特点：** 训练循环中**不加载任何轨迹文件**，只在线采样初值和参数。

**单步迭代完整链路：**

```
┌─── Step 1: 在线采样（不使用保存的数据！）──────────────────────┐
│                                                              │
│  GRF = Init_generation(size=64, alpha=4, tau=8, sigma无0.5)  │
│                                        ↑ pretrain/tools.py   │
│                                        ↑ 注意：比 data/sampler│
│                                          的 sigma 大 2 倍     │
│                                                              │
│  w0_train = GRF(batch_size=20)[..., None]                    │
│  shape: [20, 64, 64] → unsqueeze → [20, 64, 64, 1]          │
│  物理含义: 20 个随机初始物理场，分辨率 64×64                     │
│                                                              │
│  param = ones(20, 2)                                         │
│  param[:,0] = U(param1[0], param1[1])   如 U(0.1, 0.5)       │
│  param[:,1] = U(param2[0], param2[1])   如 U(0.1, 0.5)       │
│  shape: [20, 2]                                              │
│  物理含义: 20 组随机 PDE 系数 (线性系数, 非线性系数)             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 2: 模型前向（训练模式，输出 100 帧）─────────────────────┐
│                                                                │
│  w_pred = net(w0_train, param)       # inference=False（默认）  │
│                                                                │
│  net.forward 内部:                                              │
│    ① x = w0_train                          [20, 64, 64, 1]    │
│       x_o = x（保存原始输入用于残差连接）                          │
│    ② cat(x, grid[0,2π))                   [20, 64, 64, 3]     │
│    ③ cat(x, param.reshape→repeat)         [20, 64, 64, 5]     │
│    ④ in_proj: Linear(5→64) + GELU         [20, 64, 64, 64]    │
│    ⑤ for i in range(8):  ← n_layers=8                         │
│        att = softmax(f_nu[i](param)/T)    [20, 4]             │
│        b = spectral_layers[i](x, att)     [20, 64, 64, 64]    │
│        x = x + b                          残差累加              │
│    ⑥ x = gelu(b)                          丢弃累加x，只用最后b   │
│                                            [20, 64, 64, 64]    │
│    ⑦ concat(fc1a(x)[396], fc1b(x)[34])   [20, 64, 64, 430]   │
│    ⑧ gelu → Conv1d×2 → reshape            [20, 64, 64, 100]   │
│    ⑨ dt_ramp = [1/100, 2/100, ..., 1.0]                       │
│       x = x_o + x * dt_ramp               [20, 64, 64, 100]   │
│       物理含义: 初始场 + 时间加权增量 = 100帧预测                 │
│                                                                │
│  w_pred: [20, 64, 64, 100]                                     │
│  物理含义: 预测 t=dt 到 t=rollout_DT 的 100 帧物理场演化         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 3: 拼接初始帧 ─────────────────────────────────────────┐
│                                                                │
│  w_pre = concat([w0_train, w_pred], dim=-1)                    │
│  shape: [20, 64, 64, 1] + [20, 64, 64, 100] → [20, 64, 64, 101]│
│  物理含义: 完整轨迹（帧0=初始，帧1~100=预测）                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 4: PSM Loss（代入 PDE 算残差）─────────────────────────┐
│                                                                │
│  loss = PSM_loss(w_pre, param, rollout_DT=0.2, 'cn')          │
│                                                                │
│  PSM_loss 调用 PSM_KS:                                         │
│    fft2(w_pre)                         [20,64,64,101] 复数     │
│    波数 k = n/4（1/4 缩放）                                     │
│    线性项 L = (lap+lap²)·param1·w_h   [20,64,64,101]          │
│    非线性项 N = 0.5·param2·fft2(wx²+wy²) [20,64,64,101]       │
│    Du = ifft2(L+N).real               [20,64,64,101]           │
│    dt = 0.2/100 = 0.002                                       │
│    wt = 前向差分                       [20,64,64,100]           │
│    Du1 = wt + 0.5·(Du_left+Du_right)  [20,64,64,100]          │
│                                                                │
│  loss = sqrt(mean(Du1²) + 1e-7) → 标量                         │
│  物理含义: 预测轨迹在所有时空点的 PDE 残差 RMS                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 5: 反向传播 + 更新 ─────────────────────────────────────┐
│                                                                 │
│  loss.backward()                                                │
│  optimizer.step()          # Adam, lr=0.005                     │
│  optimizer.zero_grad()     # ⚠️ 先 step 再 zero_grad（非常规但等价）│
│  scheduler.step()          # OneCycleLR, 20001 步               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**验证与测试（每 100 步）：**

```
val():
  ① 固定的 w0_val[val_size,64,64,1] + val_param（训练开始前采样一次）
  ② w_pre = cat(w0_val, net(w0_val, val_param))   [val_size,64,64,101]
  ③ physics_loss = PSM_loss(w_pre, val_param, 0.2, 'cn')  → 标量
  ④ 如果 physics_loss < 历史最优 → 保存 checkpoint → 调用 test()

test():
  ① 加载真实轨迹 test_data [N_test, 64, 64, 51]
  ② model_ratio = round(1.0/0.2) = 5
     sub = round(record_ratio / model_ratio) = round(10/5) = 2
     total_iter = T_final/rollout_DT = 5.0/0.2 = 25
  ③ 自回归 rollout 25 步（推理模式，每步取最后 1 帧）:
     for _ in range(25):
         w_next = net(w_pre[...,-1:], param)[...,-1:]  → [N,64,64,1]
         w_pre = cat(w_pre, w_next)
  ④ 每步对齐真实数据: 预测第 k 步 ↔ 数据第 2k 帧
     rela_err[k] = ||pred_k - data_{2k}||₂ / ||data_{2k}||₂
  ⑤ 输出: 每步相对 L2 误差 + 平均值
```

### 6.2 NSE 预训练：增加 forcing 和黏性采样

**文件：** `nse2d/pretrain/train.py::train`

**与 KSE 的关键差异在数据采样和 loss 调用，模型结构已在 4.1 节说明。单步链路：**

```
┌─── Step 1: 在线采样（含 forcing 和黏性）──────────────────────┐
│                                                              │
│  w0 = GRF(B=10)[..., None]            [10, 256, 256, 1]     │
│  物理含义: 随机涡量初始场                                      │
│                                                              │
│  f = F_Sampler(B=10)[..., None]       [10, 256, 256, 1]     │
│  物理含义: 随机外力场（傅里叶级数叠加）                          │
│                                                              │
│  nu_train = -lognu_min - (lognu_max-lognu_min)*rand(10)      │
│  shape: [10]                                                 │
│  范围: [-log10(re_max), -log10(re_min)]                      │
│  物理含义: -log10(Re) 尺度的黏性系数                            │
│                                                              │
│  param = concat([f, nu·ones_like(f)], dim=-1)                │
│  shape: [10, 256, 256, 2]                                    │
│  param[...,0] = forcing 场（空间变化）                         │
│  param[...,1] = nu 标量（广播到全空间）                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 2: 模型前向 ─────────────────────────────────────────┐
│                                                              │
│  w_pred = net(w0, param)              [10, 256, 256, 50]     │
│                                                              │
│  关键差异（vs KSE）:                                           │
│    ② cat(x, param)     ← param 已是 [B,S,S,2]，直接 cat      │
│    ⑤ nu = param[:,0,0,1:2]  → [10,1]                        │
│       att = softmax(f_nu[i](nu)/T)   ← 路由输入只有 nu       │
│                                                              │
│  w_pred: [10, 256, 256, 50]   (NSE output_dim=50)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 3~4: 拼接 + PSM Loss ────────────────────────────────┐
│                                                              │
│  w_pre = concat([w0, w_pred], dim=-1)  [10, 256, 256, 51]   │
│                                                              │
│  loss = PSM_loss(w_pre,                                      │
│                  f_train,          ← forcing [10,256,256,1]  │
│                  10**nu_train,     ← ⚠️ 从 log 转物理黏性     │
│                  rollout_DT=0.2,                             │
│                  'cn')                                       │
│                                                              │
│  PSM_loss 内部:                                               │
│    Du1 = PSM_NS_vorticity(w_pre, 10**nu_train, 0.2, 'cn')   │
│    f = forcing.repeat(1,1,1,50)        [10,256,256,50]       │
│    loss = sqrt(mean((Du1 - f)²) + 1e-7)                     │
│                                                              │
│  物理含义: ∂ω/∂t + u·∇ω - ν·Δω - f ≈ 0 的残差 RMS           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
      Step 5: backward → step → zero_grad → scheduler.step()
```

### 6.3 蒸馏：教师多步 rollout → 学生单步拟合

**文件：** `kse2d/distillation/train.py::train`
**目标：** 大分辨率教师模型压缩成小分辨率学生模型，学生一步跨教师五步。

**单步完整链路：**

```
┌─── Step 1: 采样（教师分辨率）──────────────────────────────────┐
│                                                               │
│  GRF = Init_generation(size=64, alpha=4, tau=8, 无0.5)        │
│  w0_train = GRF(B=10)[..., None]       [10, 64, 64, 1]       │
│  param[:, 0] ~ U(param1 range)         [10, 2]               │
│  param[:, 1] ~ U(param2 range)                                │
│  物理含义: 教师分辨率 64×64 的随机初值和参数                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 2: 教师 rollout 生成目标（推理模式，无梯度）──────────────┐
│                                                                │
│  teacher_step = round(rollout_DT / rollout_DT_teacher)         │
│               = round(1.0 / 0.2) = 5                          │
│  含义: 学生一步跨 1.0 秒，教师每步 0.2 秒，需 5 步              │
│                                                                │
│  w_gth = copy(w0_train)                [10, 64, 64, 1]        │
│  with no_grad():                                               │
│    for i in range(5):                                          │
│      w_gth = net_t(w_gth, param)[..., -1:]                    │
│              ↑ OmniFluids2D 推理模式                             │
│              ↑ 输出 [10,64,64,1]，取最后1帧                     │
│                                                                │
│  w_gth: [10, 64, 64, 1]                                       │
│  物理含义: 教师从 t=0 推演到 t=1.0 秒的场                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 3: 空间下采样到学生分辨率 ──────────────────────────────┐
│                                                                │
│  student_sub = size // student_size = 64 // 32 = 2            │
│                                                                │
│  w_gth    = w_gth[:, ::2, ::2, ...]    [10, 32, 32, 1]       │
│  w0_train = w0_train[:, ::2, ::2, ...] [10, 32, 32, 1]       │
│                                                                │
│  物理含义: 隔行隔列取点，64→32 分辨率                            │
│  学生在粗网格上学习教师在细网格上的推演结果                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 4: 学生 10 次内循环训练 ───────────────────────────────┐
│                                                                │
│  for _ in range(10):     ← 同一对 (w0, w_gth) 复用 10 次       │
│                                                                │
│    net_s.train()                                               │
│    w_s = net_s(w0_train, param)         [10, 32, 32, 1]       │
│                                                                │
│    net_s.forward 内部（Student2D）:                              │
│      ① cat(x, grid, param.reshape)     [10, 32, 32, 5]       │
│      ② in_proj + gelu                  [10, 32, 32, 64]      │
│      ③ for i in range(8): att→spectral→residual               │
│      ④ x = gelu(b)                     [10, 32, 32, 64]      │
│      ⑤ output_mlp_student:                                    │
│           Linear(64→256) → GELU → Linear(256→1)              │
│                                         [10, 32, 32, 1]      │
│      ⑥ x = x_o + x   ← ⚠️ 无 dt_ramp 时间权重！              │
│                                                                │
│    loss = sqrt(mean((w_s - w_gth)²) + 1e-7)   → 标量 RMSE    │
│    物理含义: 学生预测与教师 5 步 rollout 目标的均方根误差          │
│                                                                │
│    loss.backward()                                             │
│    optimizer.step()        # Adam, lr=0.001                    │
│    optimizer.zero_grad()                                       │
│                                                                │
│  为什么 10 次内循环？                                            │
│  教师 5 步前向代价高（5×大模型推理），                             │
│  生成一次目标后让学生多次更新，摊薄教师开销                        │
│                                                                │
│  scheduler.step()   ← 每外循环 step 1 次（不是内循环每次 step）  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─── 验证与保存 ────────────────────────────────────────────────┐
│                                                                │
│  每 10 个外循环:                                                │
│    ① 直接保存 checkpoint（⚠️ 无 val 选模！每 10 步覆盖保存）     │
│    ② test(): 加载真实数据，rollout_DT=1.0                      │
│       sub = round(10 / (1/1.0)) = 10                          │
│       total_iter = 5.0/1.0 = 5                                │
│       5 步自回归 rollout，每步对齐数据第 10k 帧                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.4 微调：真实数据少样本监督

**文件：** `kse2d/finetune/train.py::train`
**目标：** 在 2 条真实轨迹上微调蒸馏好的学生模型，提升特定参数区间精度。

**数据加载（训练开始前）：**

```
┌─── 加载真实数据 ─────────────────────────────────────────────┐
│                                                              │
│  train_data = load('dataset/train_0.2_0.5')                  │
│    [:num_train, ::sub, ::sub, ...]                           │
│  shape: [2, 32, 32, 51]   (2条轨迹, 空间下采样, 51帧)         │
│  物理含义: 2 条固定参数 (param1=0.2, param2=0.5) 的真实解      │
│                                                              │
│  train_param = load('dataset/param_train_0.2_0.5')[:2]       │
│  shape: [2, 2]                                               │
│                                                              │
│  val_data, test_data: 同理加载                                 │
│                                                              │
│  train_step = round(record_ratio / (1/rollout_DT))           │
│             = round(10 / (1/1.0)) = 10                       │
│  含义: 模型单步跨 rollout_DT=1.0 秒 = 数据 10 帧间隔           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**单步迭代完整链路：**

```
┌─── Step 1: 随机时间采样 ──────────────────────────────────────┐
│                                                               │
│  w0   = zeros(2, 32, 32, 1)                                  │
│  w_gth1 = zeros(2, 32, 32, 1)                                │
│  w_gth2 = zeros(2, 32, 32, 1)                                │
│                                                               │
│  for i in range(num_train=2):                                 │
│    t = randint(0, 51 - 2*10 - 1)    ← 即 randint(0, 30)      │
│    w0[i]     = train_data[i, ..., t]                          │
│    w_gth1[i] = train_data[i, ..., t+10]                      │
│    w_gth2[i] = train_data[i, ..., t+20]                      │
│                                                               │
│  物理含义: 每条轨迹随机选一个起点 t，                            │
│  取 (t, t+1秒, t+2秒) 三帧作为 (输入, 目标1, 目标2)            │
│  每次迭代采不同 t → 数据增强，避免过拟合                         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 2: 第一步预测 ────────────────────────────────────────┐
│                                                               │
│  pred1 = net(w0, train_param)[..., -1:]                       │
│  shape: [2, 32, 32, 1]                                       │
│  物理含义: 从 t 预测 t+1秒 的物理场                             │
│                                                               │
│  loss1 = sqrt(mean((pred1 - w_gth1)²) + 1e-7)                │
│  物理含义: 第一步预测的 RMSE                                    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 3: 第二步自回归预测 ──────────────────────────────────┐
│                                                               │
│  pred2 = net(pred1, train_param)[..., -1:]                    │
│  shape: [2, 32, 32, 1]                                       │
│  物理含义: 从 pred1（而非 gth1）继续预测 t+2秒                  │
│  ⚠️ 用预测值（非真实值）做输入 → 自回归 → 训练模型容忍误差累积     │
│                                                               │
│  loss2 = sqrt(mean((pred2 - w_gth2)²) + 1e-7)                │
│  物理含义: 第二步预测的 RMSE（含误差传播）                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─── Step 4: 总 loss + 更新 ────────────────────────────────────┐
│                                                               │
│  loss = loss1 + loss2                                         │
│  loss.backward()       # 梯度回传穿过两步自回归                  │
│  optimizer.step()      # Adam, lr=0.002                       │
│  optimizer.zero_grad()                                        │
│  ⚠️ 无 scheduler（与 pretrain/distillation 不同！）             │
│                                                               │
│  物理含义: 同时优化单步精度和两步自回归精度,                      │
│  惩罚误差累积,提升长期预测稳定性                                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─── 验证与保存（每 10 步）──────────────────────────────────────┐
│                                                               │
│  val_error = test(net, val_data, val_param, val_dict)         │
│    rollout: total_iter = 5.0/1.0 = 5 步                      │
│    sub = 10（对齐数据帧）                                       │
│    返回: 平均相对 L2 误差                                       │
│                                                               │
│  if val_error < best:                                         │
│    best = val_error                                           │
│    test(net, test_data, ...)   ← 跑 test 集                   │
│    save checkpoint             ← ⚠️ 只在 val 改善时保存!       │
│                                                               │
│  vs 蒸馏: 蒸馏每 10 步无条件保存，微调有 val 选模                │
│  原因: 微调数据极少(2条)易过拟合，需 val 集做 early stopping     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**微调 main.py 额外注意：** 它对两个数据集 `_0.2_0.5` 和 `_1_1` **分别**调用 `train(cfg)`。
即同一个模型先在参数组 1 上微调，保存后再在参数组 2 上重新加载蒸馏权重微调。最终得到**两个独立 checkpoint**。

### 6.5 三阶段数据流对比总览

| | 预训练 | 蒸馏 | 微调 |
|---|---|---|---|
| **数据来源** | 在线 GRF 采样 | 在线 GRF → 教师 rollout | 真实轨迹文件 |
| **输入 shape** | `[B,S,S,1]` 随机 | `[B,s,s,1]` 下采样 | `[N,s,s,1]` 随机时间片 |
| **模型** | OmniFluids2D（教师）| Student2D（学生）| Student2D（继续）|
| **输出 shape** | `[B,S,S,Tp]` 多帧 | `[B,s,s,1]` 单帧 | `[N,s,s,1]` 单帧 |
| **Loss 类型** | PSM（PDE 残差）| RMSE（拟合教师）| RMSE（拟合真实）|
| **Loss 公式** | `√(mean(Du²)+ε)` | `√(mean((w_s-w_gth)²)+ε)` | `√(mean((pred-gth)²)+ε)` |
| **监督步数** | 100帧一次 loss | 1步 | 2步自回归 |
| **Scheduler** | OneCycleLR | OneCycleLR | 无 |
| **选模策略** | val physics loss | 每10步无条件保存 | val rollout L2 |
| **典型 lr** | 0.005 / 0.002 | 0.001 / 0.002 | 0.002 / 5e-5 |

---

## 7. 超参数完整字典

### 7.1 数据生成

| 参数 | KSE | NSE | 含义 |
|------|-----|-----|------|
| s | 128 | 1024 | 求解器网格 |
| sub | 1 | 4 | 空间下采样倍数 |
| dt | 1e-5 | 1e-4 | 求解器时间步 |
| T | 5.0 | 10.0 | 总时间 |
| record_ratio | 10 | 10 | 每单位时间帧数 |
| param1 | [0.1,0.5] | - | 线性项系数范围 |
| param2 | [0.1,0.5] | - | 非线性项系数范围 |
| re | - | [500,2500] | 雷诺数范围 |
| max_frequency | - | 10 | forcing 最高频 |
| amplitude | - | 0.5 | forcing 幅值 |

### 7.2 模型结构

| 参数 | KSE pretrain | NSE pretrain | KSE distill学生 | NSE distill学生 |
|------|-------------|-------------|----------------|----------------|
| size | 64 | 256 | 32 | 128 |
| modes | 32 | 128 | 16 | 64 |
| width | 64 | 80 | 64 | 80 |
| n_layers | 8 | 12 | 8 | 12 |
| output_dim | 100 | 50 | - | - |
| K | 4 | 4 | 4 | 4 |

### 7.3 训练配置

| 参数 | KSE pre | NSE pre | KSE dist | NSE dist | KSE ft | NSE ft |
|------|---------|---------|----------|----------|--------|--------|
| batch_size | 20 | 10 | 10 | 10 | 全量(2) | 全量(10) |
| lr | 0.005 | 0.002 | 0.001 | 0.002 | 0.002 | 5e-5 |
| iterations | 20000 | 20000 | 2000 | 2000 | 1000 | 200 |
| rollout_DT | 0.2 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 |
| DT_teacher | - | - | 0.2 | 0.2 | - | - |
| scheduler | OneCycle | OneCycle | OneCycle | OneCycle | 无 | 无 |

---

## 8. 时间对齐规则

### 8.1 预训练阶段

教师输出 Tp 帧覆盖 rollout_DT 时间：

- KSE: dt_psm = 0.2/100 = 0.002
- NSE: dt_psm = 0.2/50 = 0.004

### 8.2 Test rollout 对齐

```
total_iter = T_final / rollout_DT
sub = record_ratio / (1/rollout_DT)     # 数据帧间隔
第 k 步预测 <-> 数据第 sub*k 帧
```

| 阶段 | rollout_DT | total_iter | sub |
|------|-----------|------------|-----|
| KSE pretrain | 0.2 | 25 | 2 |
| KSE distill/ft | 1.0 | 5 | 10 |
| NSE pretrain | 0.2 | 50 | 2 |
| NSE distill/ft | 1.0 | 10 | 10 |

---

## 9. KSE 与 NSE 差异对照

| 维度 | KSE | NSE |
|------|-----|-----|
| 条件参数 shape | [B, 2] 标量 | [B, S, S, 2] 场+标量 |
| in_proj 拼接 | param reshape+broadcast | param 直接 cat |
| 专家路由输入 | param [B,2] Linear(2->) | nu [B,1] Linear(1->) |
| 物理域大小 | `[0, 8π]²`（k = n/4） | `[0, 2π]²`（k = n） |
| PSM forcing | 无（残差=0） | loss=sqrt(mean((Du-f)^2)) |
| nu 处理 | param 直接用 | param 存 log, PSM 时 10**visc |
| 数据命名 | test_multi / param_test_multi | data_test_multi / f_test_multi |

---

## 10. 排错决策树

| 现象 | 首先检查 | 然后尝试 |
|------|---------|---------|
| loss nan/inf | 数据 sol.max() 是否爆炸 | 降 lr 5~10 倍; 减 rollout_DT |
| physics loss 降但 rollout 不降 | 时间对齐是否整除 | 提高 modes; 减小 rollout_DT |
| 蒸馏效果弱 | teacher_step 是否够 | 增大学生 modes/width |
| 微调过拟合 | lr 是否过大 | 降 lr; 增 num_train; 减容量 |

---

## 11. 迁移最小改造清单

### 11.1 仅换 PDE

- [ ] 新增 `data/<solver>.py`
- [ ] 修改 `data/generate_data.py`
- [ ] 重写 `pretrain/psm_loss.py`（核心：你的 PDE 残差）
- [ ] 调 `rollout_DT / record_ratio / dt / T`

### 11.2 条件参数也变

还需改：

- [ ] `model.py` 的 `in_proj` 输入维度
- [ ] `f_nu` 的输入维度
- [ ] `train.py` 的 param 组装
- [ ] 数据保存/加载格式

### 11.3 验收

1. pretrain physics loss 可下降
2. distillation 学生 rollout 接近教师
3. finetune 后 val/test 相对 L2 改善

---

## 12. 阅读顺序建议

| 序号 | 文件 | 读完要掌握 |
|------|------|-----------|
| 1 | `kse2d/data/kse.py` | RK4 谱方法、波数 1/4 缩放、去混叠 |
| 2 | `kse2d/data/sampler.py` | GRF 着色、alpha/tau 影响 |
| 3 | `kse2d/pretrain/model.py` | 输入拼接、专家路由、分离式 FFT、gelu(b) 丢弃残差 x、多帧输出头 Conv1d 压缩 |
| 4 | `kse2d/pretrain/psm_loss.py` | 残差各项与 PDE 对应、cn/mid 差异 |
| 5 | `kse2d/pretrain/train.py` | 在线采样、physics-only 训练、rollout 评估 |
| 6 | `kse2d/distillation/train.py` | 教师多步->学生单步、10 次内循环、空间下采样 |
| 7 | `kse2d/finetune/train.py` | 随机时间片、两步监督 |
| 8 | `nse2d/pretrain/psm_loss.py` | 涡量方程残差、forcing 减法在 PSM_loss 层 |
| 9 | `nse2d/pretrain/train.py` | nu 的 log 采样与转换 |
| 10 | `nse2d/pretrain/model.py` | 条件编码只取 nu 标量 |

---

## 13. 掌握全部代码的 Checklist（完成即毕业）

本节设计为"通关清单"：按顺序完成每个 checkpoint，全部打勾后你就完整掌握了这个 repo。

---

### Phase 0: 环境与直觉（约 30 分钟）

- [ ] **C0.1** 安装依赖 `pip install -r requirements.txt`，确认 `import torch, einops, thop` 无报错
- [ ] **C0.2** 通读本文档第 1~2 节，在纸上画出 `data -> pretrain -> distill -> finetune` 的四阶段流水线图
- [ ] **C0.3** 回答问题：预训练阶段的监督信号从哪来？（答案：不来自数据标签，来自 PDE 残差）

---

### Phase 1: 数据生成（约 1 小时）

#### 阅读

- [ ] **C1.1** 打开 `kse2d/data/kse.py`，找到波数定义行 `kx = 1/4 * ...`，推导 `n/4 = 2πn/L` → `L=8π`，理解 KSE 域是 `[0,8π]`
- [ ] **C1.2** 在 `ks_2d_rk4` 中，用笔写出 RK4 的四个子步 `k1~k4`，确认线性项 `L` 和非线性项 `N` 分别对应 PDE 哪一部分
- [ ] **C1.3** 找到 `dealias` 掩码所在行，确认它实现了 2/3 规则
- [ ] **C1.4** 打开 `kse2d/data/sampler.py`，写出 `sqrt_eig` 公式，理解 `alpha` 越大高频衰减越快
- [ ] **C1.5** 打开 `kse2d/data/generate_data.py`，确认 `record_steps = int(record_ratio * T) = 50`，输出 shape `[N, 128, 128, 51]`
- [ ] **C1.6** 打开 `nse2d/data/nse.py`，找到 Crank-Nicolson 更新公式那一行（约第 82 行），对照涡量方程确认正确
- [ ] **C1.7** 在 `nse2d/data/sampler.py` 中找到 `Force_generation.__call__`，确认 forcing 是随机傅里叶级数叠加

#### 动手

- [ ] **C1.8** 运行 `cd kse2d/data && python main.py`，确认 `dataset/` 下出现文件且 `test_multi` 大小合理
- [ ] **C1.9** 写 3 行代码加载一条 KSE 轨迹，打印 shape 和 max/min，确认无 nan/inf：
  ```python
  import torch
  u = torch.load('dataset/test_multi'); p = torch.load('dataset/param_test_multi')
  print(u.shape, u.max(), u.min(), p.shape, p)
  ```
- [ ] **C1.10**（可选）对 NSE 做同样检查

---

### Phase 2: 模型架构（约 1.5 小时）

#### 输入拼接

- [ ] **C2.1** 打开 `kse2d/pretrain/model.py` 第 119~125 行，追踪 shape：`[B,S,S,1]` -> cat grid -> `[B,S,S,3]` -> cat param -> `[B,S,S,5]` -> in_proj -> `[B,S,S,width]`
- [ ] **C2.2** 打开 `nse2d/pretrain/model.py` 第 119~126 行，确认 NSE 的 param 是 `[B,S,S,2]` 直接 cat（不做 reshape+repeat），专家路由只取 `nu = param[:,0,0,1:2]`

#### 频域层

- [ ] **C2.3** 在 `SpectralConv2d_dy.forward_fourier` 中，确认它做的是 **两次 1D rFFT**（dim=-1 和 dim=-2），不是一次 rFFT2
- [ ] **C2.4** 确认 `fourier_weight` shape 是 `[K, width, width, modes, 2]`，理解 `einsum("bk, kioxy->bioxy", att, weight)` 是用 softmax 权重混合 K 组专家
- [ ] **C2.5** 确认 `backcast_ff` 是一个 2 层 MLP（width -> width*4 -> width，含 ReLU+LayerNorm），跟在频域变换之后
- [ ] **C2.6** 确认频域层的 `forward` 返回的是 backcast_ff 的输出 `b`，不是频域变换的直接输出

#### 残差循环与输出头

- [ ] **C2.7** 在第 126~133 行，确认循环中 `x = x + b`（残差累加），但循环后 `x = F.gelu(b)`（**丢弃累加的 x，只用最后一层 b**）
- [ ] **C2.8** 手算训练模式输出维度：fc1a 输出 `4*100-4=396`，fc1b 输出 `34`，拼接 `430`，经两层 Conv1d(k=12,s=2) 压成 `100`。写出公式验证
- [ ] **C2.9** 手算推理模式输出维度：fc1b 输出 `34`，经同样两层 Conv1d 压成 `1`
- [ ] **C2.10** 确认时间权重 `dt = arange(1, Tp+1) / Tp`，第一帧权重 1/100，最后一帧权重 1.0

#### 学生模型

- [ ] **C2.11** 打开 `kse2d/distillation/model.py` 第 154~211 行，确认 Student2D 输出头是 `Linear(width -> 4*width) -> GELU -> Linear(4*width -> 1)`
- [ ] **C2.12** 确认 Student2D 的输出是 `x_o + x`（**无时间权重 ramp**），与教师的 `x_o + x * dt` 不同

---

### Phase 3: PSM Loss（约 1 小时）

#### KSE

- [ ] **C3.1** 打开 `kse2d/pretrain/psm_loss.py`，确认波数有 `1/4` 缩放因子（第 19~22 行）
- [ ] **C3.2** 写出 `lap = -(k_x^2 + k_y^2)`，确认 `L = (lap + lap^2) * param1 * w_h` 对应方程的线性项
- [ ] **C3.3** 确认非线性项 `N = 0.5 * param2 * fft2(wx^2 + wy^2)`（注意：PSM 中**无 dealias**，与求解器不同）
- [ ] **C3.4** 对照 cn 模式：`wt = 前向差分`，`Du1 = wt + 0.5*(Du_left + Du_right)`，确认这是 Crank-Nicolson 风格
- [ ] **C3.5** 确认最终 loss = `sqrt(mean(Du1^2) + 1e-7)`，理解 +eps 防止梯度爆炸

#### NSE

- [ ] **C3.6** 打开 `nse2d/pretrain/psm_loss.py`，确认 NSE 波数**无 1/4 缩放**（第 19~22 行，直接整数波数）
- [ ] **C3.7** 确认 `lap = k_x^2 + k_y^2`（**正号**，与 KSE 的负号不同！）
- [ ] **C3.8** 追踪：`psi_h = w_h/lap` -> `ux_h = 1j*ky*psi_h` -> `uy_h = -1j*kx*psi_h`，对照涡量方程
- [ ] **C3.9** 确认 `PSM_NS_vorticity` 返回的 `Du1` **不含 forcing**，forcing 减法在外层 `PSM_loss` 中做：`loss = sqrt(mean((Du1 - f)^2) + eps)`

---

### Phase 4: 训练流程（约 1 小时）

#### 预训练

- [ ] **C4.1** 打开 `kse2d/pretrain/train.py::train`，确认训练循环**不加载任何轨迹数据**（只在线采样 w0 和 param）
- [ ] **C4.2** 确认 GRF 来自 `pretrain/tools.py`（alpha=4, tau=8, **无 0.5**），与 `data/sampler.py`（**有 0.5**）不同
- [ ] **C4.3** 追踪一次迭代：`w0[20,64,64,1]` -> `net()` -> `[20,64,64,100]` -> concat w0 -> `[20,64,64,101]` -> PSM_loss -> 标量
- [ ] **C4.4** 确认 optimizer 顺序是 `backward -> step -> zero_grad`（非常规顺序，但首轮梯度从零开始所以等价）
- [ ] **C4.5** 确认 val() 函数也是 physics loss（在线采样），不是轨迹误差
- [ ] **C4.6** 确认 test() 函数是 rollout 相对 L2，用 `net(...)[...,-1:]` 即推理模式

#### 蒸馏

- [ ] **C4.7** 打开 `kse2d/distillation/train.py::train`，确认 `teacher_step = round(rollout_DT / rollout_DT_teacher) = 5`
- [ ] **C4.8** 确认教师 rollout 是逐步调用 `net_t(w_gth, param)[...,-1:]`（推理模式，每步取最后1帧）
- [ ] **C4.9** 确认空间下采样 `student_sub = size // student_size = 2`，w0 和 w_gth 都下采样
- [ ] **C4.10** 确认 10 次内循环：同一组 `(w0, w_gth)` 对学生更新 10 次，scheduler 在外循环 step
- [ ] **C4.11** 确认蒸馏**没有 val 选模**，每 10 步直接保存并 test

#### 微调

- [ ] **C4.12** 打开 `kse2d/finetune/train.py::train`，确认 `train_step = round(record_ratio / (1/rollout_DT)) = 10`
- [ ] **C4.13** 确认随机采样：每个样本随机选 `t`，取 `(t, t+10, t+20)` 三帧
- [ ] **C4.14** 确认两步监督：`loss = RMSE(pred1, gth1) + RMSE(pred2, gth2)`
- [ ] **C4.15** 确认有 val 选模：val rollout 改善时才保存 checkpoint
- [ ] **C4.16** 打开 `kse2d/finetune/main.py`，确认它对 `_0.2_0.5` 和 `_1_1` 两个数据集分别调用 `main(cfg)`

---

### Phase 5: NSE 差异确认（约 30 分钟）

- [ ] **C5.1** 确认 `nse2d/pretrain/model.py` 的 `f_nu` 输入是 `Linear(1->128)`（不是 KSE 的 `Linear(2->128)`）
- [ ] **C5.2** 确认 `forward` 中 param 直接 cat（第 123 行 `x = torch.cat((x, param), dim=-1)`），不做 reshape+repeat
- [ ] **C5.3** 确认 `nse2d/pretrain/train.py` 第 93 行 nu 采样公式：`nu_train = -lognu_min - (lognu_max-lognu_min)*rand(B)`
- [ ] **C5.4** 确认 PSM_loss 调用时传入 `10**nu_train` 作为物理黏性
- [ ] **C5.5** 确认 NSE 数据命名用 `data_*` / `f_*` 前缀（KSE 无前缀 / `param_*` 前缀）

---

### Phase 6: 端到端复现（约半天~1天）

- [ ] **C6.1** KSE 数据生成完成，检查无 nan
- [ ] **C6.2** KSE pretrain 跑 2000 步（先冒烟），确认 physics loss 下降
- [ ] **C6.3** KSE pretrain 跑完 20000 步，记录最终 test rollout 相对 L2
- [ ] **C6.4** KSE distillation 跑完，确认学生 rollout 接近教师
- [ ] **C6.5** KSE finetune 跑完两个数据集，确认 val 误差下降
- [ ] **C6.6** 整理一张表：各阶段 test 相对 L2，对比论文报告值
- [ ] **C6.7**（可选）NSE 全流程复现

---

### Phase 7: 准备迁移（约 1 小时思考）

- [ ] **C7.1** 写出你的目标 PDE 残差形式（连续方程）
- [ ] **C7.2** 选择时间离散方式（cn 或 mid）
- [ ] **C7.3** 确定空间导数离散方式（频域是否可行？是否需要有限差分？）
- [ ] **C7.4** 确定条件参数形式（标量？场？维度？）
- [ ] **C7.5** 列出需要修改的文件清单（对照本文档第 11 节）
- [ ] **C7.6** 在小分辨率上跑通 data + pretrain（确认 physics loss 可下降）
- [ ] **C7.7** 跑通 distillation + finetune
- [ ] **C7.8** 固化实验记录模板：配置、日志、ckpt 路径、val/test 指标

---

### 通关自测题（全答对说明你真正掌握了）

1. 预训练时网络一次输出 100 帧（KSE），这 100 帧覆盖多长物理时间？（答：rollout_DT = 0.2 秒）
2. PSM loss 的 cn 模式中 `dt` 是多少？（答：rollout_DT / (num_frames - 1) = 0.2 / 100 = 0.002）
3. 教师输出头为什么分 `fc1a(396维)` 和 `fc1b(34维)` 两个 Linear？（答：推理时只用 fc1b，经 Conv1d 输出 1 帧；训练时拼接后经 Conv1d 输出 100 帧）
4. 循环结束后 `x = F.gelu(b)` 为什么用 `b` 不用 `x`？（答：代码设计如此，只取最后频域层输出做激活，前面层通过影响输入间接作用）
5. Student2D 与 OmniFluids2D 的增量输出有什么区别？（答：教师用 `x_o + delta * dt_ramp` 有线性时间权重，学生用 `x_o + delta` 无权重）
6. 蒸馏中为什么做 10 次内循环？（答：教师前向成本高，复用同一目标多次更新学生更高效）
7. NSE 的 param 中存的 nu 是什么尺度？（答：`-log10(Re)` 尺度，使用时 `10**visc` 转成物理黏性）
8. KSE 的 PSM 波数为什么乘 1/4？（答：KSE 物理域是 `[0, 8π]`，物理波数 = `2πn/L = n/4`，与求解器一致）
