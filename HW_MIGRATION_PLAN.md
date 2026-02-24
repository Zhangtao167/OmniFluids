# HW 方程迁移：详细实施计划

> **状态**: ✅ 所有决策已确认，可以执行
> **方式**: 直接增改 `kse2d/` 下的现有代码，使其适用于 HW 方程
> **目标**: 将 OmniFluids KSE pipeline 迁移为 HW 方程，保持三阶段架构

---

## 0. 关键设计决策 (已确认)

### 固定参数 (与 mhd_sim 一致)

| 参数 | 值 | 来源 |
|------|-----|------|
| k0 | 0.15 | HWConfig 默认 |
| N (hyper_order) | 3 | HWConfig 默认 |
| Nx = Ny | 128 | HWConfig 默认 |
| dt (solver) | 0.025 | HWConfig 默认 |
| diffusion_method | "fd" (solver) / spectral (PSM) | — |

### 可配置训练参数范围

| 参数 | 范围 | 默认值 |
|------|------|--------|
| α (耦合强度) | **[0.5, 1.5]** | 1.0 |
| κ (密度梯度) | **[0.5, 1.5]** | 1.0 |
| log₁₀(ν) (超黏性) | **[-4.7, -3.3]** | -4.3 (ν=5e-5) |

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| rollout_DT | **1.0s** | 匹配 mhd_sim 数据帧间距 (dt×stamp_interval=0.025×40) |
| output_dim | 100 | 100 帧内部子步，每帧 0.01s |
| batch_size | 4~8 | HW PSM 计算量约为 KSE 的 4 倍 |
| size (pretrain) | 64 | 模型工作分辨率，与 KSE 保持一致；solver 128→下采样2× |

### 时间步链路全图 ★必须理解★

```
┌─────────────────── HW Solver ───────────────────┐
│ dt_solver = 0.025s, Nt = 10000 步                │
│ stamp_interval = 40 → dt_data = 1.0s             │
│ 共 250 stamps + 初始帧 = 251 个数据点             │
│ 总物理时间 = 250s                                 │
└──────────────────────────────────────────────────┘

┌─────────────────── OmniFluids 数据 ─────────────────┐
│ record_ratio = 1 (fps, 每秒1帧)                      │
│ T = 250 (总物理时间)                                  │
│ record_steps = record_ratio × T = 250                │
│ 数据 shape: [N, S, S, 251, 2] (含初始帧)             │
└──────────────────────────────────────────────────────┘

┌──────────── Pretrain (PSM, 无监督) ─────────────┐
│ rollout_DT = 1.0s (模型一次预测覆盖的物理时间)    │
│ output_dim = 100 (训练时模型输出100帧)            │
│                                                   │
│ 训练 (模型内部):                                  │
│   100帧 / 1.0s = PSM内部dt = 0.01s               │
│   PSM_KS代码: dt = t_interval/(nt-1) = 1.0/100   │
│   dt_ramp: [0.01, 0.02, ..., 1.0]s               │
│                                                   │
│ 推理 (inference=True):                            │
│   模型输出1帧 = 状态@t+1.0s                       │
│                                                   │
│ 测试:                                             │
│   model_ratio = round(1/1.0) = 1                  │
│   sub = round(record_ratio/model_ratio) = 1       │
│   total_iter = round(T/rollout_DT) = 250          │
│   → 自回归250步 × 1.0s = 250s rollout             │
└───────────────────────────────────────────────────┘

┌─────── Distillation (Teacher→Student) ──────────┐
│ rollout_DT = 1.0s                                │
│ rollout_DT_teacher = 1.0s (★改 KSE的0.2→1.0)    │
│ teacher_step = round(1.0/1.0) = 1                │
│ → Teacher 做 1步×1.0s = Student学 1步×1.0s       │
└──────────────────────────────────────────────────┘

┌──────────── Finetune (有监督) ──────────────────┐
│ rollout_DT = 1.0s                                │
│ train_step = round(record_ratio × rollout_DT)    │
│            = round(1 × 1.0) = 1                  │
│ → 1模型步 = 1数据帧 = 1.0s                       │
└──────────────────────────────────────────────────┘
```

**对比 KSE (供参考)**:
```
KSE: dt_solver=1e-5, record_ratio=10(fps), T=5.0
     pretrain rollout_DT=0.2 → PSM内部dt=0.002s, test: sub=2, total_iter=25
     distillation rollout_DT_teacher=0.2, teacher_step=5
     finetune train_step=10
```

### D1: f_nu (MoO) 输入 — 5 维

| 参数 | 编码 | 训练时取值 | 说明 |
|------|------|-----------|------|
| α | 直接 | U[0.5, 1.5] | 耦合强度 |
| κ | 直接 | U[0.5, 1.5] | 密度梯度 |
| log₁₀(ν) | log变换 | U[-4.7, -3.3] | 超黏性 |
| N/4 | 归一化 | 0.75 (固定) | 超扩散阶数 |
| k0 | 直接 | 0.15 (固定) | 基频 |

初版 k0=0.15、N=3 固定，但仍输入 f_nu（架构预留泛化能力）。

### D2: in_proj — 9 维

`[ζ(1), n(1), grid_x(1), grid_y(1), α, κ, logν, N/4, k0]`

### D3: 输出头 — Conv1d 输出通道 1→2

Teacher: `Conv1d(8,1,12,2)` → `Conv1d(8,2,12,2)`  
Student: `Linear(4w,1)` → `Linear(4w,2)`

### D4: PSM 残差 — 分别 RMSE 后平均

`loss = 0.5*(sqrt(mean(Du_ζ²)+ε) + sqrt(mean(Du_n²)+ε))`

### D5: 初值 — 独立 GRF（域大小 L=2π/k0）

### D6: 波数 — 固定 k0=0.15

PSM 硬编码 `kx = k0 * freq`，get_grid 保持 [0,2π) 归一化

---

## Phase 1: 数据生成模块 — 改 `kse2d/data/`

### Step 1.1: 改 `kse.py` → 增加 HW solver 封装

**文件**: `kse2d/data/kse.py`  
**改法**: 保留原 `ks_2d_rk4` 不删，在同文件末尾增加 `hw_2d_rk4`，直接调用 mhd_sim solver。

```python
# ===== 新增: HW solver 封装 (文件末尾追加) =====
import sys
sys.path.insert(0, '/zhangtao/project2026/mhd_sim')
from numerical.equations.hasegawa_wakatani import HasegawaWakatani, HWConfig
from numerical.scripts.hw_eq import rk4_step

def hw_2d_rk4(zeta0, n0, T, param, k0=0.15, N_hyper=3,
              dt=0.025, record_steps=100, device='cuda'):
    """
    param: [B,3] → α=param[:,0], κ=param[:,1], log₁₀ν=param[:,2]
    返回: [B, S, S, record_steps+1, 2]  (最后维: 0=ζ, 1=n)
    """
    ...  # 逐样本构造 HWConfig → HasegawaWakatani → RK4 积分
```

**验证**: 单样本运行，输出 shape 正确、无 NaN

### Step 1.2: 改 `sampler.py` — GRF 域大小

**文件**: `kse2d/data/sampler.py`  
**改法**: `Init_generation` 的默认域大小从 `L=2π` 改为 `L=2π/k0`

| 行号 | 原始 | 修改 |
|------|------|------|
| 8 | `L1=2*math.pi, L2=2*math.pi` | `L1=2*math.pi/0.15, L2=2*math.pi/0.15` |

同时新增双场封装:
```python
class HW_Init_generation:
    def __init__(self, size, k0=0.15, **kwargs):
        L = 2 * math.pi / k0
        self.grf_zeta = Init_generation(size, L1=L, L2=L, **kwargs)
        self.grf_n = Init_generation(size, L1=L, L2=L, **kwargs)
    def __call__(self, N):
        return self.grf_zeta(N), self.grf_n(N)
```

### Step 1.3: 改 `generate_data.py` — 参数维度与双场

**文件**: `kse2d/data/generate_data.py`

| 行号 | 原始 (KSE) | 修改 (HW) | 原因 |
|------|-----------|----------|------|
| 55 | `param_record = torch.zeros(N, 2)` | `torch.zeros(N, 3)` | 3 参数 |
| 60 | `param = torch.ones(bsize, 2, device=device)` | `torch.zeros(bsize, 3, device=device)` | α,κ,logν |
| 61 | `param[:, 0] = U(param1_range)` | `param[:, 0] = U(alpha_range)` | α |
| 62 | `param[:, 1] = U(param2_range)` | `param[:, 1] = U(kappa_range)` | κ |
| 新增 | — | `param[:, 2] = U(lognu_range)` | log₁₀ν |
| 59 | `w0 = GRF(bsize)` | `z0, n0 = HW_GRF(bsize)` | 双场 |
| 63 | `sol = ks_2d_rk4(w0, T, param, ...)` | `sol = hw_2d_rk4(z0, n0, T, param, ...)` | 调用 HW solver |
| 54 | `u = torch.zeros(N, s//sub, s//sub, T+1)` | `u = torch.zeros(N, s//sub, s//sub, T+1, 2)` | 双场 |

### Step 1.4: 改 `main.py` — 参数名、范围、★时间步参数★

**文件**: `kse2d/data/main.py`

| 行号 | 原始 | 修改 | 原因 |
|------|------|------|------|
| 37 | `--T 5.0` | `--T 250` | HW 总物理时间 250s |
| 40 | `--dt 1e-5` | `--dt 0.025` | HW solver 步长 |
| 43 | `--record_ratio 10` | `--record_ratio 1` | 1fps (每秒1帧, 即 dt_data=1.0s) |
| 46-49 | `--param1/--param2` | `--alpha_range/--kappa_range/--lognu_range` | 3参数 |
| 56-57 | `cfg.param1='[0.1,0.5]'` | `cfg.alpha_range='[0.5,1.5]'` 等 | 范围 |

**★ record_ratio 定义对齐 ★**:
- KSE: `record_ratio=10` → `record_steps = 10 × 5.0 = 50` → 数据51帧，帧间0.1s
- HW:  `record_ratio=1`  → `record_steps = 1 × 250 = 250` → 数据251帧，帧间1.0s
- 公式: `record_steps = record_ratio × T`，`record_ratio` 含义 = 每秒帧数 (fps)

**★ 测试数据配置文件 (JSON) ★**:

`generate_data.py` 会将 `cfg.__dict__` 保存为 `log/cfg_{name}.txt`，test 函数通过它读取 `T`, `record_ratio` 等。HW 配置文件需包含:
```json
{
  "T": 250,
  "record_ratio": 1,
  "s": 128,
  "sub": 1,
  "alpha_range": [0.5, 1.5],
  "kappa_range": [0.5, 1.5],
  "lognu_range": [-4.7, -3.3]
}
```

test 函数使用逻辑:
```python
total_iter = round(T / rollout_DT) = round(250 / 1.0) = 250   # 推理250步
sub = round(record_ratio / model_ratio) = round(1 / 1) = 1     # 每步跳1帧
# 最大数据索引: sub × total_iter = 1 × 250 = 250               # 需要251帧数据
```

**验证**: 
```bash
cd kse2d/data && python main.py
# 检查:
# 1. dataset/ 下文件 shape: data=[N,S,S,251,2], param=[N,3]
# 2. log/cfg_*.txt 中 T=250, record_ratio=1
# 3. 数据帧0 = 初始条件, 帧250 = t=250s
```

---

## Phase 2: PSM Loss — 改 `kse2d/pretrain/psm_loss.py` ★最关键★

### Step 2.1: 重写 `PSM_KS` → `PSM_HW`

**文件**: `kse2d/pretrain/psm_loss.py`  
**改法**: 保留原 `PSM_KS` 不删，新增 `PSM_HW`；将 `PSM_loss` 改为调用 `PSM_HW`。

HW 残差计算链路 (每帧):
```
1. FFT: ζ_h = fft2(ζ),  n_h = fft2(n)
2. Poisson: φ_h = -ζ_h / k²   (k²[0,0]=1, φ_h[0,0]=0)
3. 谱导数 (6次ifft2): ∂φ/∂x, ∂φ/∂y, ∂ζ/∂x, ∂ζ/∂y, ∂n/∂x, ∂n/∂y
4. Poisson 括号: [φ,ζ] = ∂φ/∂x·∂ζ/∂y - ∂φ/∂y·∂ζ/∂x,  同理 [φ,n]
5. 耦合: coupling = α(φ-n),  梯度驱动: κ·∂φ/∂y
6. 超扩散 (见下方符号说明)
7. 组装空间算子 Sp_ζ, Sp_n
8. CN 时间离散: res = f_t + 0.5*(Sp[t]+Sp[t+1])
```

**超扩散符号说明 (防止实现时搞混)**:

HW 方程 RHS 的扩散项是 `D(f) = ifft2(-ν·k^{2N}·f̂)`，它是耗散项（值为负，消除能量）。
PSM 计算 `-RHS`（空间算子），所以扩散对空间算子的贡献是 `-D(f) = ifft2(+ν·k^{2N}·f̂)`。

具体地，在代码中：
```python
# 超扩散对空间算子的贡献（注意是正号）
hyper_zeta = torch.fft.ifft2(nu * k2N * zeta_h, dim=[1,2]).real  # +ν·k^{2N}·ζ̂
hyper_n    = torch.fft.ifft2(nu * k2N * n_h, dim=[1,2]).real     # +ν·k^{2N}·n̂

# 组装空间算子 (= -RHS)
Sp_zeta = pb_zeta - coupling + hyper_zeta   # [φ,ζ] - α(φ-n) + ν·k^{2N}·ζ̂
Sp_n    = pb_n + kappa_dphi_dy - coupling + hyper_n  # [φ,n] + κ∂φ/∂y - α(φ-n) + ν·k^{2N}·n̂
```

波数: `kx = k0 * cat(arange(0,S/2), arange(-S/2,0))`（类比 KSE 的 `1/4 * cat(...)`）

**PSM_loss 改动** (原签名 `PSM_loss(u, param, t_interval, loss_mode)`):
```python
def PSM_loss(w, param, t_interval=0.50, loss_mode='cn'):
    # w: [B,S,S,Nt,2] (双场)
    zeta, n = w[..., 0], w[..., 1]  # 各 [B,S,S,Nt]
    Du_z, Du_n = PSM_HW(zeta, n, param, t_interval, loss_mode)
    EPS = 1e-7
    return 0.5 * ((torch.square(Du_z).mean() + EPS).sqrt()
                 + (torch.square(Du_n).mean() + EPS).sqrt())
```

### Step 2.2: PSM 正确性验证 ★必须通过★

用 solver 精确轨迹代入 PSM，检查残差:
```python
sol = hw_2d_rk4(z0, n0, T=1.0, param, record_steps=100)
Du_z, Du_n = PSM_HW(sol[...,0], sol[...,1], param, T)
relative_residual = rmse(Du) / rmse(field)  # 应 < 0.01
```

---

## Phase 3: 模型架构 — 改 `kse2d/*/model.py`

涉及 3 个 model.py: `pretrain/model.py`, `distillation/model.py`, `finetune/model.py`

### Step 3.1: OmniFluids2D 改动 (pretrain + distillation)

**`__init__` 改动**:

| 位置 | KSE 原始 | HW 修改 |
|------|---------|--------|
| 签名 | `def __init__(self, s=256, K=4, ...)` | 增加 `n_fields=2, n_params=5` |
| 第 82 行 | — | `self.n_fields = n_fields; self.n_params = n_params` |
| 第 93 行 `in_proj` | `nn.Linear(5, self.width)` | `nn.Linear(n_fields+2+n_params, self.width)` |
| 第 97 行 `f_nu` | `nn.Linear(2, 128)` | `nn.Linear(n_params, 128)` |
| 第 116 行 `output_mlp` | `nn.Conv1d(8, 1, 12, stride=2)` | `nn.Conv1d(8, n_fields, 12, stride=2)` |

**★ `forward` 改动 — 关键：多场输出 reshape 逻辑 ★**

KSE 原始 (第 139-141 行):
```python
# Conv1d 输出: (B*S*S, 1, Tp) → reshape → (B, S, S, Tp)
x = self.output_mlp(x.reshape(-1, 1, x.shape[-1])).reshape(batch_size, size, size, -1)
dt = torch.arange(1, x.shape[-1] + 1, ...).reshape(1, 1, 1, -1) / x.shape[-1]
x = x_o + x * dt  # x_o:[B,S,S,1], x:[B,S,S,Tp]
```

HW 修改 — **不能直接用 `.reshape(B,S,S,-1)`，否则会把 n_fields 和 Tp 混在一起**:
```python
# Conv1d 输出: (B*S*S, n_fields, Tp) — n_fields=2, Tp=100(训练)/1(推理)
x = self.output_mlp(x.reshape(-1, 1, x.shape[-1]))
# 必须显式拆分 n_fields 和 Tp 维度
x = x.reshape(batch_size, size, size, self.n_fields, -1)  # [B,S,S,2,Tp]
x = x.permute(0, 1, 2, 4, 3)                               # [B,S,S,Tp,2]

# dt ramp: 需要在 Tp 维度上做，同时对两个场广播
Tp = x.shape[3]
dt = torch.arange(1, Tp + 1, device=x.device).reshape(1,1,1,-1,1).float() / Tp
# dt: [1,1,1,Tp,1] — 在 batch, spatial, field 维度广播
x = x_o.unsqueeze(-2) + x * dt
# x_o: [B,S,S,2] → unsqueeze(-2) → [B,S,S,1,2]
# x*dt: [B,S,S,Tp,2]
# 结果: [B,S,S,Tp,2]

if inference:
    x = x.squeeze(-2)  # [B,S,S,1,2] → [B,S,S,2]
return x
```

**param cat** (第 123 行):
```python
# KSE: param.reshape(batch_size, 1, 1, 2).repeat(1, size, size, 1)
# HW:  param.reshape(batch_size, 1, 1, self.n_params).repeat(1, size, size, 1)
```

**尺寸验算 (output_dim=100, n_fields=2)**:
- 训练: fc1a(396)+fc1b(34)=430 → Conv1d(1→8, L=210) → Conv1d(8→2, L=100)
  - reshape → (B,S,S,2,100) → permute → (B,S,S,100,2) ✓
- 推理: fc1b(34) → Conv1d(1→8, L=12) → Conv1d(8→2, L=1)
  - reshape → (B,S,S,2,1) → permute → (B,S,S,1,2) → squeeze → (B,S,S,2) ✓

### Step 3.2: Student2D 改动 (distillation + finetune)

| 位置 | KSE 原始 | HW 修改 |
|------|---------|--------|
| 签名 | `def __init__(self, ...)` | 增加 `n_fields=2, n_params=5` |
| `in_proj` | `nn.Linear(5, width)` | `nn.Linear(n_fields+2+n_params, width)` |
| `f_nu` | `nn.Linear(2, 128)` | `nn.Linear(n_params, 128)` |
| param reshape | `.reshape(B,1,1,2)` | `.reshape(B,1,1,n_params)` |
| output | `nn.Linear(4*width, 1)` | `nn.Linear(4*width, n_fields)` |
| 残差 | `x_o + x` (x_o:[B,S,S,1]) | `x_o + x` (x_o:[B,S,S,2], 自动兼容) |

**注意**: Student2D 没有 `inference` 参数，forward 始终输出 `[B,S,S,n_fields]`。
下游代码中所有 `net(w, param)[..., -1:]` 必须改为 `net(w, param)` (去掉 `[..., -1:]`)。

### Step 3.3: 验证

```python
net = OmniFluids2D(s=64, n_fields=2, n_params=5, ...)
x = randn(2,64,64,2); p = randn(2,5)
assert net(x,p).shape == (2,64,64,100,2)            # 训练
assert net(x,p,inference=True).shape == (2,64,64,2)  # 推理

stu = Student2D(s=32, n_fields=2, n_params=5, ...)
x = randn(2,32,32,2); p = randn(2,5)
assert stu(x,p).shape == (2,32,32,2)                # 始终单步
```

---

## Phase 4: 预训练循环 — 改 `kse2d/pretrain/`

### Step 4.1: 改 `tools.py`

| 改动 | 原始 | 修改 |
|------|------|------|
| `Init_generation` 域大小 | `L1=L2=2π` | `L1=L2=2π/k0` (≈41.9) |
| `param_flops` dummy | `dummy_input2 = randn(1,128,128,2)` | `randn(1,128,128,5)` |
| 新增 | — | `HW_Init_generation` 双场 GRF 封装 |

### Step 4.2: 改 `train.py` ★重点：test/val/train 三个函数都需改★

#### 4.2.1 train 函数 — 在线采样 (第 88-97 行)

```python
# KSE 原始:
w0_train = GRF(batch_size)[..., None]           # [B,S,S,1]
param = torch.ones(batch_size, 2, device=device)
param[:, 0] = U(param1[0], param1[1])
param[:, 1] = U(param2[0], param2[1])
w_pre = net(w0_train, param)                    # [B,S,S,Tp]
w_pre = torch.concat([w0_train, w_pre], dim=-1) # [B,S,S,Tp+1]
loss = PSM_loss(w_pre, param, rollout_DT, loss_mode)

# HW 修改:
z0, n0 = HW_GRF(batch_size)                     # [B,S,S], [B,S,S]
w0_train = torch.stack([z0, n0], dim=-1)         # [B,S,S,2]
param = torch.zeros(batch_size, 5, device=device)
param[:,0] = U(alpha_min, alpha_max)              # α
param[:,1] = U(kappa_min, kappa_max)              # κ
param[:,2] = U(lognu_min, lognu_max)              # log₁₀ν
param[:,3] = 0.75                                  # N/4 (固定)
param[:,4] = 0.15                                  # k0 (固定)

w_pred = net(w0_train, param)                                   # [B,S,S,Tp,2]
w_all = torch.cat([w0_train.unsqueeze(-2), w_pred], dim=-2)     # [B,S,S,Tp+1,2]
loss = PSM_loss(w_all, param, rollout_DT, loss_mode)
```

#### 4.2.2 val 函数 — 需同步改 cat 逻辑 (第 47-51 行)

```python
# KSE 原始:
w_pre = torch.cat([w_0, net(w_0, val_param).detach()], dim=-1)  # dim=-1

# HW 修改:
w_pred = net(w_0, val_param).detach()                              # [B,S,S,Tp,2]
w_pre = torch.cat([w_0.unsqueeze(-2), w_pred], dim=-2)            # [B,S,S,Tp+1,2]
physics_loss = PSM_loss(w_pre, val_param, config.rollout_DT, config.loss_mode)
```

#### 4.2.3 ★ test 函数 — 完全重写 rollout 逻辑 (第 20-43 行) ★

KSE 原始使用 `[..., -1:]` + `torch.concat(dim=-1)` 在时间维累积。
HW 多场下 `[..., -1:]` 取的是最后一个**场**而非最后一个**时间步**，此模式完全失效。

```python
# KSE 原始:
w_0 = test_data[:, :, :, 0:1]                    # [N,S,S,1]
w_pre = w_0
for _ in range(total_iter):
    w_0 = net(w_pre[..., -1:], param).detach()[..., -1:]
    w_pre = torch.concat([w_pre, w_0], dim=-1)   # 在 dim=-1 累积
# w_pre: [N,S,S,T+1]，取 w_pre[..., t] 比较

# HW 修改:
w_current = test_data[:, :, :, 0, :]              # [N,S,S,2] — 初始帧
predictions = [w_current]
net.eval()
with torch.no_grad():
    for _ in range(total_iter):
        w_current = net(w_current, param, inference=True).detach()  # [N,S,S,2]
        predictions.append(w_current)
w_pre = torch.stack(predictions, dim=3)            # [N,S,S,T+1,2]

# 误差计算: 对两场联合计算 relative L2
for time_step in range(1, total_iter+1):
    w = w_pre[:, :, :, time_step, :]               # [N,S,S,2]
    w_t = test_data[:, :, :, sub*time_step, :]      # [N,S,S,2]
    # reshape 为 [N, S*S*2] 再算 norm
    rela_err.append((torch.norm((w-w_t).reshape(N,-1), dim=1)
                    / torch.norm(w_t.reshape(N,-1), dim=1)).mean().item())
```

#### 4.2.4 val 参数初始化 (第 73-76 行)

```python
# KSE 原始:
w0_val = GRF(val_size)[..., None]                  # [B,S,S,1]
val_param = torch.ones(val_size, 2, device=device)

# HW 修改:
z0, n0 = HW_GRF(val_size)
w0_val = torch.stack([z0, n0], dim=-1)             # [B,S,S,2]
val_param = torch.zeros(val_size, 5, device=device)
# ... 同训练采样的 5 维 param
```

#### 4.2.5 参数加载 (第 61 行)

```python
# KSE: param1, param2 = test_data_dict['param1'], test_data_dict['param2']
# HW:  alpha_range = test_data_dict['alpha_range']
#      kappa_range = test_data_dict['kappa_range']
#      lognu_range = test_data_dict['lognu_range']
```

### Step 4.3: 改 `main.py`

- 模型实例化增加 `n_fields=2, n_params=5`
- argparse 增加 `--alpha_min/max`, `--kappa_min/max`, `--lognu_min/max` (可配置)
- ★ `--rollout_DT` 默认值从 **0.2** 改为 **1.0** (与 dt_data=1.0s 匹配)
- `--size` 默认保持 64 (128→下采样2×)

### Step 4.4: 端到端验证

```bash
cd kse2d/pretrain && python main.py --num_iterations 100 --batch_size 4
# 检查: loss 下降, 无 NaN, GPU 显存稳定
```

---

## Phase 5: 蒸馏 — 改 `kse2d/distillation/`

### Step 5.1: 改 `model.py`

Phase 3 已覆盖 (OmniFluids2D + Student2D 同步改)

### Step 5.2: 改 `train.py`

#### 5.2.1 train 函数 (第 72-92 行)

| 行号 | 原始 | 修改 | 说明 |
|------|------|------|------|
| 73 | `w0_train = GRF(batch_size)[..., None]` | `z0,n0=HW_GRF(B); w0=stack([z0,n0],dim=-1)` | [B,S,S,2] |
| 74 | `param = torch.ones(B, 2)` | `torch.zeros(B, 5)` + 3 参数采样 + 2 固定 | 5维 |
| 77 | `w_gth = copy.copy(w0_train)` | `w_gth = w0_train.clone()` | 同 |
| 81 | `w_gth = net_t(w_gth, param).detach()[..., -1:]` | `w_gth = net_t(w_gth, param, inference=True).detach()` | **去掉 `[..., -1:]`** — Teacher 推理返回 [B,S,S,2] |
| 82 | `w_gth[:, ::sub, ::sub, ...]` | 不变 | 空间下采样对双场自动生效 |
| 86 | `w_s = net_s(w0_train, param)` | 不变 | Student 输出 [B,S,S,2] |
| 87 | `(w_s - w_gth)**2` | 不变 | 自然支持双场 |

#### 5.2.2 ★ test 函数 — 完全重写 rollout 逻辑 (第 20-43 行) ★

与 pretrain 的 test 函数改法完全一致（见 Phase 4 Step 4.2.3），但注意：
- 蒸馏 test 使用 **Student2D**，无 `inference` 参数 → 直接 `net(w_current, param)` 即可
- `[..., -1:]` 必须去掉，因为 Student 输出 `[B,S,S,2]`

```python
# KSE 原始:
w_0 = net(w_pre[..., -1:], test_param).detach()[..., -1:]

# HW 修改:
w_current = net(w_current, test_param).detach()    # [N,S,S,2]
```

### Step 5.3: 改 `main.py` — ★ rollout_DT_teacher 需要同步修改 ★

| 行号 | 原始 | 修改 | 原因 |
|------|------|------|------|
| 126-127 | `--rollout_DT_teacher 0.2` | `--rollout_DT_teacher 1.0` | HW pretrain 的 rollout_DT=1.0，对应 teacher_step=1 |
| 45-52 | `OmniFluids2D(s=s, K=K, ...)` | 增加 `n_fields=2, n_params=5` | 参数化 |
| 56 | `Student2D(...)` | 增加 `n_fields=2, n_params=5` | 参数化 |

KSE 时 teacher 做 5 步 (0.2s×5=1.0s)，HW 时 teacher 做 1 步 (1.0s×1=1.0s)。

### Step 5.4: 改 `tools.py`

同 pretrain 修改：GRF 域大小 + 双场封装

### Step 5.5: 验证

```bash
cd kse2d/distillation && python main.py --num_iterations 100
# MSE loss 下降, student rollout L2 合理
```

---

## Phase 6: 微调 — 改 `kse2d/finetune/`

### Step 6.1: 改 `model.py`

Phase 3 已覆盖 (Student2D)

### Step 6.2: 改 `train.py`

#### 6.2.1 train 函数 (第 91-105 行)

| 行号 | 原始 | 修改 | 说明 |
|------|------|------|------|
| 93 | `w0 = zeros(N, S, S, 1)` | `zeros(N, S, S, 2)` | **双场** |
| 94 | `w_gth1 = zeros(N, S, S, 1)` | `zeros(N, S, S, 2)` | **双场** |
| 95 | `w_gth2 = zeros(N, S, S, 1)` | `zeros(N, S, S, 2)` | **双场** |
| 98 | `w0[i, ..., 0] = data[i, ..., t]` | `w0[i,:,:,:] = data[i,:,:,t,:]` | 时间索引在倒数第2维 |
| 99 | `w_gth1[i, ..., 0] = data[i, ..., t+s]` | `w_gth1[i,:,:,:] = data[i,:,:,t+s,:]` | 同上 |
| 100 | `w_gth2[i, ..., 0] = data[i, ..., t+2s]` | `w_gth2[i,:,:,:] = data[i,:,:,t+2s,:]` | 同上 |
| 101 | `net(w0, param)[..., -1:]` | `net(w0, param)` | **去掉 `[..., -1:]`**，Student 输出 [B,S,S,2] |
| 104 | `net(w_pre, param)[..., -1:]` | `net(w_pre, param)` | 同上 |
| 72 | `train_param [N,2]` | `[N,3]` → 补全为 `[N,5]` (加固定 N/4=0.75, k0=0.15) | param 维度 |

#### 6.2.2 ★ test 函数 — 同 Phase 4 Step 4.2.3 完全重写 ★

#### 6.2.3 数据加载 — permute

finetune 中数据由 `hw_eq_batch.py` 生成，保存格式为 `[B, T, S, S]`（mhd_sim 约定）。
加载后需要 permute 为 OmniFluids 约定 `[N, S, S, T, 2]`：

```python
# 在数据加载处添加:
# mhd_sim 原始: traj_zeta [B,T,S,S], traj_n [B,T,S,S]
# 目标: [B, S, S, T, 2]
train_data = torch.stack([traj_zeta, traj_n], dim=-1)  # [B,T,S,S,2]
train_data = train_data.permute(0, 2, 3, 1, 4)         # [B,S,S,T,2]
```

### Step 6.3: 改 `tools.py`

GRF 域大小 + `parse_filename` 无需改动（解析模型名）

### Step 6.4: 改 `main.py`

数据文件名相关参数

### Step 6.5: 验证

```bash
cd kse2d/finetune && python main.py --num_iterations 100
# loss 下降, test L2 优于蒸馏
```

---

## 附录 A: 全文件改动一览 (均在 `kse2d/` 下就地修改)

| 文件路径 | 改动方式 | 具体改什么 |
|---------|---------|-----------|
| `data/kse.py` | **追加** hw_2d_rk4 函数 | 封装 mhd_sim solver |
| `data/sampler.py` | **改** GRF 域大小 + **追加** 双场封装 | L=2π/k0, HW_Init_generation |
| `data/generate_data.py` | **改** | param 维度 2→3, 数据增场维, 调用 hw solver |
| `data/main.py` | **改** | 参数名 param1/2 → alpha/kappa/lognu |
| `pretrain/psm_loss.py` | **追加** PSM_HW + **改** PSM_loss | HW 双方程残差 |
| `pretrain/model.py` | **改** OmniFluids2D | in_proj(5→9), f_nu(2→5), Conv1d(1→2), **forward 输出 reshape+permute** |
| `pretrain/train.py` | **改** train + **重写** test + **改** val | 双场采样/PSM/rollout 全面改写 |
| `pretrain/tools.py` | **改** + **追加** | GRF 域大小, 双场封装, dummy 维度 |
| `pretrain/main.py` | **改** | 模型参数 n_fields/n_params, argparse, rollout_DT 默认值 |
| `distillation/model.py` | **改** OmniFluids2D + Student2D | 同 pretrain |
| `distillation/train.py` | **改** train + **重写** test | 参数维度, Teacher inference, 去掉 `[..., -1:]` |
| `distillation/tools.py` | **改** | GRF 域, dummy 维度 |
| `distillation/main.py` | **改** | n_fields/n_params, **rollout_DT_teacher 0.2→1.0** |
| `finetune/model.py` | **改** Student2D | in_proj/f_nu/output |
| `finetune/train.py` | **改** train + **重写** test | w0/gth shape 1→2, 去掉 `[..., -1:]`, **数据 permute** |
| `finetune/tools.py` | **改** | GRF 域大小 |
| `finetune/main.py` | **改** | 数据命名, 参数 |

## 附录 B: 易错点速查表 (Devil's Advocate 审查结果)

| # | 严重度 | 位置 | 问题 | 正确做法 |
|---|--------|------|------|---------|
| 1 | ★★★ | OmniFluids2D forward L139 | `.reshape(B,S,S,-1)` 把 n_fields×Tp 混合 | 必须 `.reshape(B,S,S,n_fields,-1).permute(0,1,2,4,3)` |
| 2 | ★★★ | 所有 test() (3文件×4处) | `[..., -1:]` + `concat(dim=-1)` 把场维当时间维 | 重写为 `predictions` 列表 + `stack(dim=3)` |
| 3 | ★★ | pretrain/main.py | `--rollout_DT` 默认 0.2，HW 必须改为 1.0 | 否则 PSM 内部 dt=0.002s 与物理不匹配 |
| 4 | ★★ | distillation/main.py | `rollout_DT_teacher=0.2` 与 HW pretrain rollout_DT=1.0 不匹配 | 改为 `1.0`，teacher_step 变为 1 |
| 5 | ★★ | data/main.py | KSE: `record_ratio=10, T=5.0, dt=1e-5` 全部需改 | HW: `record_ratio=1, T=250, dt=0.025` |
| 6 | ★★ | finetune/train.py L93-95 | `zeros(N,S,S,1)` 单场 | `zeros(N,S,S,2)` 双场 |
| 7 | ★★ | finetune/distillation train | `net(w,p)[..., -1:]` 对 Student2D 取错维度 | 去掉 `[..., -1:]` |
| 8 | ★★ | finetune/train.py | mhd_sim 数据 `[B,T,S,S]` 需要 permute | 加载后 `permute(0,2,3,1,4)` |
| 9 | ★★ | 测试数据配置 JSON | `T/record_ratio` 决定 test 的 total_iter 和 sub | HW: `T=250, record_ratio=1` → total_iter=250, sub=1 |
| 10 | ★★ | 4个 tools.py | GRF 的 alpha/tau/sigma 跨文件不一致 | 统一为同一组参数，或明确保留差异 |
| 11 | ★ | PSM 超扩散 | `-D(ζ)` 双重否定易混淆 | 代码中直接写 `+ν·k^{2N}` 带注释 |

## 附录 C: GRF 参数不一致说明

当前 KSE 代码中 4 个 `Init_generation` 实例参数不一致（这是 KSE 原始代码就有的）：

| 文件 | alpha | tau | sigma 系数 | 备注 |
|------|-------|-----|-----------|------|
| `data/sampler.py` | 4 | 8.0 | 0.5× | 数据生成用 |
| `pretrain/tools.py` | 4 | 8.0 | 1× | 预训练在线采样用 |
| `distillation/tools.py` | 4 | 8.0 | 1× | 蒸馏用 |
| `finetune/tools.py` | 2.5 | 7.0 | 0.5× | 微调用，更平滑 |

HW 方程对初值幅度可能比 KSE 更敏感（mhd_sim 用 `init_noise_level=1e-6`）。
建议: 初版保持与 KSE 一致的 GRF 参数不做改动，如果 pretrain 前 100 步 loss 爆炸再考虑缩小幅度。

## 附录 D: 测试数据生成

### 方式一: 通过 OmniFluids 的 generate_data.py (pretrain/distillation test 用)

调用 `hw_2d_rk4` wrapper，直接生成 OmniFluids 格式:
- `record_ratio = 1` (1fps)，`T = 250`
- 输出: `[N, S, S, 251, 2]` (含初始帧, record_steps+1=251帧)
- 10 个样本，每个随机 (α, κ, ν)
- 配套 JSON 配置: `T=250, record_ratio=1, s=128, sub=1`
- **验证**: `total_iter = round(250/1.0) = 250`，最大数据索引 = 250 < 251 ✓

### 方式二: 通过 mhd_sim 的 hw_eq_batch.py (finetune 用)

- T_final=250s (Nt=10000, dt=0.025)，前 200s spin-up
- mhd_sim 输出: `traj_zeta [B,250,S,S]`, `traj_n [B,250,S,S]` (无初始帧, 250帧)
- **格式转换** (在 finetune/train.py 数据加载处):
  ```python
  # mhd_sim → OmniFluids
  data = torch.stack([traj_zeta, traj_n], dim=-1)  # [B,250,S,S,2]
  data = data.permute(0, 2, 3, 1, 4)               # [B,S,S,250,2]
  ```
- finetune 的配套 JSON 配置需要:
  - `T = 249` (250帧无初始帧, 从帧0可推理249步)
  - `record_ratio = 1`
  - 或: 在数据前面补初始帧使之变为251帧, 然后 `T=250`

## 附录 E: mhd_sim PDE loss 参考

`mhd_sim/plasma_sim/train/train_hw.py` 中有 HW 物理损失实现 (`compute_physics_loss`)，
CN 格式与 OmniFluids PSM 数学等价:
```
residual = (x_pred - x_t)/dt - 0.5*(rhs(x_t) + rhs(x_pred))
```
差异: mhd_sim 调用 solver 的 `compute_rhs` (Arakawa FD)，我们的 PSM 用纯谱方法。
可作为正确性交叉验证的参考。

## 附录 F: 风险与应对

| 风险 | 检测 | 应对 |
|------|------|------|
| PSM残差偏大(谱vsArakawa) | Step2.2 | PSM中实现Arakawa |
| GRF初值训练不稳定 | 前100步loss | 缩小GRF sigma/约束GRF/平衡态池 |
| ν跨量级残差不平衡 | 监控per-ν loss | per-sample归一化 |
| ζ/n残差量级差异 | 分别打印 | 自适应权重 |
| GPU OOM | 报错 | 减batch_size/output_dim |
| Conv1d reshape 遗漏 permute | shape assert | Phase 3 验证必过 |
| test rollout `[..., -1:]` 残留 | rollout 输出全零 | 全局搜索 `[..., -1:]` |
