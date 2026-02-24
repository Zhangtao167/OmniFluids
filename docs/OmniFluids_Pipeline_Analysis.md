# OmniFluids 三阶段训练流程深度分析

> 基于 `nse2d/` 和 `nse2d_old/` 代码库，结合 `kse2d/` 对照分析。
> 文档目标：完整说明 Pretrain → Distillation → Finetune 三阶段的具体做法、数据流、损失函数、模型关系。

---

## 总览：三阶段的关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: Pretrain (预训练)                                                    │
│  模型: OmniFluids2D (大模型, 高分辨率 s=256, output_dim=40~50)                │
│  数据: 随机采样初始条件 + 随机强迫 (无需预存轨迹)                               │
│  损失: PSM 物理结构匹配损失 (频域 PDE 残差, 无 GT 标签)                         │
│  目标: 学会在广泛参数范围 (Re, forcing) 下的流体动力学                           │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │  保存 pretrained OmniFluids2D checkpoint
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: Distillation (蒸馏)                                                  │
│  Teacher: 加载 pretrained OmniFluids2D (大模型, 细 dt, 高分辨率)               │
│  Student: 新建 Student2D (小模型, 粗 dt, 低分辨率 s=128)                       │
│  数据: 在线随机采样 (同 Pretrain, 无预存数据)                                   │
│  损失: Teacher 多步 rollout 输出作为软标签, Student 单步 MSE 对齐               │
│  目标: 将大模型的知识压缩到小模型, 同时扩大推理 dt (加速推理)                    │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │  保存 distilled Student2D checkpoint
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 3: Finetune (微调)                                                      │
│  模型: 加载 distilled Student2D, 继续训练                                      │
│  数据: 预存的真实轨迹数据 (train/val/test split)                                │
│  损失: 2步 autoregressive MSE (对齐 GT 轨迹)                                   │
│  目标: 在特定参数 (Re, forcing) 的真实数据上精调, 提升绝对精度                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心设计思路**：
- Pretrain 解决"泛化性"问题：无需大量数据，用物理约束覆盖宽参数空间
- Distillation 解决"效率"问题：大模型细步长 → 小模型粗步长，推理加速
- Finetune 解决"精度"问题：在具体场景的真实数据上对齐

---

## 阶段 1：Pretrain（预训练）

### 1.1 代码位置

```
nse2d/
├── pretrain/
│   ├── model.py   — OmniFluids2D 定义
│   ├── train.py   — 训练/测试循环
│   ├── psm_loss.py — PSM 物理损失
│   └── main.py    — 入口
```

### 1.2 模型：OmniFluids2D（大模型）

**原始 NSE2D 配置**（`nse2d_old/pretrain/model.py`）：

```
输入: (B, S, S, 5)   — [vorticity_field, grid_x, grid_y, forcing_x, forcing_y]
      S=256, 5 = 1场 + 2坐标 + 2强迫分量

架构:
  in_proj: Linear(5, width)
  × n_layers:
    f_nu[i]: Linear(1→128→128→K)  — 从标量 ν (粘度) 生成 K 个注意力权重
    SpectralConv2d_dy:
      fourier_weight: (K, W, W, modes, 2) × 2方向  — 核心参数
      MoE 加权: att = softmax(f_nu(ν)/T)
              weight = Σ_k att_k * fourier_weight_k
      FFT(x方向) + FFT(y方向) 频域卷积
      backcast_ff: FeedForward(W, factor=4, n_layers=2)
  output:
    训练: fc1a(W→4*od-4) ‖ fc1b(W→34) → Conv1d(1,8,12,s=2) → Conv1d(8,1,12,s=2) → (od 帧)
    推理: fc1b(W→34) → Conv1d → Conv1d → (1 帧)
  残差: x_out = x_in + Δx * dt_frac  (线性插值残差)

输出: (B, S, S, output_dim)  — output_dim=40~50 个时间帧
```

**参数量**（典型配置 modes=128, width=80, n_layers=12）：
- fourier_weight：**99.5%** 的参数，约 157M
- 总计：约 158M

### 1.3 数据：在线随机采样（无预存轨迹）

```python
# 每个训练步在线生成：
GRF = Init_generation(size)          # 高斯随机场采样器（初始涡度 w0）
F_Sampler = Force_generation(size)   # 随机强迫场采样器

w0_train = GRF(batch_size)           # (B, S, S) 随机初始条件
f_train  = F_Sampler(batch_size)     # (B, S, S) 随机强迫
nu_train = 10^(uniform(logν_min, logν_max))  # 随机粘度（Re 的倒数）

param = cat([f_train, nu_train * ones_like(f_train)])  # (B, S, S, 2)
```

**关键特点**：
- 每步都是新数据，不重复
- 覆盖宽参数范围（Re 从低到高）
- 不需要预先运行数值模拟存储数据

### 1.4 损失函数：PSM（Physical Structure Matching）

**核心思想**：不用 GT 轨迹，而是检验预测轨迹是否满足 NS 方程的 PDE 残差。

```python
# nse2d_old/pretrain/psm_loss.py

def PSM_NS_vorticity(w, v, t_interval, loss_mode):
    # w: (B, S, S, T+1) — 输入帧 + 预测的 T 帧
    # v: 粘度 (标量)
    
    w_h = fft2(w)                    # 转到频域
    
    # 从涡度计算速度场（Biot-Savart）
    ux = irfft2(1j * k_y * w_h / lap)   # x方向速度
    uy = irfft2(-1j * k_x * w_h / lap)  # y方向速度
    wx = irfft2(1j * k_x * w_h)         # ∂w/∂x
    wy = irfft2(1j * k_y * w_h)         # ∂w/∂y
    wlap = irfft2(-lap * w_h)            # ∇²w (扩散项)
    
    # 对流+扩散项: Du = u·∇w - ν∇²w
    Du = ux*wx + uy*wy - v * wlap
    
    if loss_mode == 'cn':  # Crank-Nicolson
        wt = (w[..., 1:] - w[..., :-1]) / dt          # 时间导数
        residual = wt + (Du[..., :-1] + Du[..., 1:]) / 2  # CN格式
    if loss_mode == 'mid':  # 中心差分
        wt = (w[..., 2:] - w[..., :-2]) / (2*dt)
        residual = wt + Du[..., 1:-1]
    
    return residual  # 应趋近于 forcing f

def PSM_loss(u, forcing, v, t_interval, loss_mode):
    Du = PSM_NS_vorticity(u, v, t_interval, loss_mode)
    return sqrt(mean((Du - f)^2) + EPS)
```

**训练循环**：

```python
for step in range(num_iterations):
    w0, f = GRF(bs), F_Sampler(bs)          # 随机采样
    param = cat([f, ν * ones])

    w_pre = net(w0, param)                   # (B, S, S, output_dim) 多帧预测
    w_pre = cat([w0, w_pre], dim=-1)         # 拼接初始帧: (B, S, S, T+1)
    
    loss = PSM_loss(w_pre, f, ν, rollout_DT, loss_mode)  # PDE 残差损失
    
    loss.backward()
    optimizer.step()
    scheduler.step()  # OneCycleLR
```

**验证**：用 PSM loss（物理残差）而非 GT 误差来选择最优模型。

### 1.5 关键超参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `rollout_DT` | 0.2~0.5 | 模型推理的时间步长（较细） |
| `output_dim` | 40~50 | 训练时输出的帧数 |
| `loss_mode` | `cn` | Crank-Nicolson 格式的 PDE 残差 |
| `size` | 256 | 空间分辨率 |
| `modes` | 128 | 谱卷积截断模数 |

---

## 阶段 2：Distillation（蒸馏）

### 2.1 代码位置

```
nse2d/
├── distillation/
│   ├── model.py   — OmniFluids2D (Teacher) + Student2D (Student)
│   ├── train.py   — 蒸馏训练循环
│   └── main.py    — 入口（加载 Teacher checkpoint）
```

### 2.2 模型对比：Teacher vs Student

| 特征 | Teacher (OmniFluids2D) | Student (Student2D) |
|------|------------------------|---------------------|
| 来源 | 加载 Pretrain checkpoint | 随机初始化 |
| 分辨率 | `size=256` | `size=128`（更低） |
| 输出头 | `fc1a + fc1b → Conv1d`（多帧/单帧双路） | `output_mlp_student: Linear(W→4W→1)`（单帧） |
| `output_dim` | 40~50 | 1（直接输出单帧） |
| 推理 dt | `rollout_DT_teacher`（细，如 0.2） | `rollout_DT`（粗，如 1.0） |
| 参数量 | ~158M | 更小（modes/width 可缩减） |
| 训练状态 | **frozen**（`eval()`, `no_grad`） | **被训练** |

**Student2D 输出头的关键区别**：
```python
# Teacher (OmniFluids2D): 双路输出头，支持多帧/单帧
self.fc1a = Linear(width, 4*output_dim - 4)   # 训练路径
self.fc1b = Linear(width, 34)                  # 推理路径
self.output_mlp = Conv1d(1,8,12,s=2) → Conv1d(8,1,12,s=2)

# Student2D: 简单 MLP，直接输出单帧
self.output_mlp_student = Sequential(
    Linear(width, 4*width), GELU(), Linear(4*width, 1)
)
```

### 2.3 蒸馏训练过程

**核心机制**：Teacher 用细步长 rollout 多步，得到粗步长对应的"软标签"；Student 用粗步长单步预测，对齐这个软标签。

```python
# nse2d/distillation/train.py

teacher_step = round(rollout_DT / rollout_DT_teacher)
# 例: rollout_DT=1.0, rollout_DT_teacher=0.2 → teacher_step=5

for step in range(num_iterations):
    # 1. 在线采样初始条件（同 Pretrain）
    w0, f = GRF(bs), F_Sampler(bs)
    ν = uniform(logν_min, logν_max)
    param = cat([f, ν * ones])

    # 2. Teacher 多步 rollout → 软标签
    w_gth = copy(w0)
    net_t.eval()
    with no_grad():
        for i in range(teacher_step):          # 5步 × dt=0.2 = 1.0s
            w_gth = net_t(w_gth, param)[..., -1:]  # 每步取最后帧
    # w_gth: Teacher 推进 1.0s 后的状态（低分辨率下采样）
    w_gth = w_gth[:, ::student_sub, ::student_sub]  # 降采样到 128×128

    # 3. Student 单步预测
    w0_s = w0[:, ::student_sub, ::student_sub]
    param_s = param[:, ::student_sub, ::student_sub]
    
    for _ in range(10):  # 每个样本更新 10 次梯度
        net_s.train()
        w_s = net_s(w0_s, param_s)             # Student 单步, dt=1.0
        loss = sqrt(mean((w_s - w_gth)^2) + EPS)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # OneCycleLR
```

**时间步长对齐图示**：
```
Teacher: w0 →[dt=0.2]→ w1 →[dt=0.2]→ w2 →[dt=0.2]→ w3 →[dt=0.2]→ w4 →[dt=0.2]→ w_gth
                                                                                      ↑
Student: w0 ──────────────────────────[dt=1.0]──────────────────────────────────→ w_s
                                                                                      ↓
                                                              loss = MSE(w_s, w_gth)
```

**关键设计**：
- Teacher 的多步 rollout 提供了"1.0s 后应该是什么状态"的软标签
- Student 学会用单步（粗 dt）直接跳到相同位置
- 这是一种**时间步长扩展**（time-step expansion）的蒸馏

### 2.4 关键超参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `rollout_DT` | 1.0 | Student 推理步长（粗） |
| `rollout_DT_teacher` | 0.2 | Teacher 推理步长（细） |
| `teacher_step` | 5 | Teacher 每个 Student 步需要走的步数 |
| `student_size` | 128 | Student 空间分辨率 |
| `size` | 256 | Teacher 空间分辨率 |
| `num_iterations` | 2000 | 蒸馏迭代次数（比 Pretrain 少很多） |

---

## 阶段 3：Finetune（微调）

### 3.1 代码位置

```
nse2d/
├── finetune/
│   ├── model.py   — Student2D（与 distillation 相同）
│   ├── train.py   — 微调训练循环
│   └── main.py    — 入口（加载 distilled Student2D checkpoint）
```

### 3.2 数据：预存真实轨迹

与 Pretrain/Distillation 的在线采样不同，Finetune 使用**预先存储的数值模拟轨迹**：

```python
# nse2d/finetune/train.py
train_data = torch.load(f'dataset/data_train{data_name}')  # (N, S, S, T_total)
train_param = torch.load(f'dataset/f_train{data_name}')    # (N, S, S, 2)
val_data   = torch.load(f'dataset/data_val{data_name}')
test_data  = torch.load(f'dataset/data_test{data_name}')
```

数据格式：`(N_samples, S, S, T_total)` — 完整时间序列轨迹。

### 3.3 微调训练过程

**损失**：2步 autoregressive MSE（对齐真实轨迹的两个连续时刻）：

```python
for step in range(num_iterations):
    net.train()
    
    # 随机采样训练样本的起始时刻
    for i in range(num_train):
        t = randint(0, T_total - 2*train_step - 1)
        w0[i]    = train_data[i, ..., t]           # 起始帧
        w_gth1[i] = train_data[i, ..., t+train_step]   # 1步后的 GT
        w_gth2[i] = train_data[i, ..., t+2*train_step] # 2步后的 GT
    
    # 第1步预测
    w_pre = net(w0, train_param)[..., -1:]
    loss = sqrt(mean((w_pre - w_gth1)^2) + EPS)
    
    # 第2步预测（autoregressive）
    w_pre = net(w_pre, train_param)[..., -1:]
    loss += sqrt(mean((w_pre - w_gth2)^2) + EPS)
    
    loss.backward()
    optimizer.step()
    
    # 每10步评估一次 val 误差，保存最优模型
    if step % 10 == 0:
        val_error = test(config, net, val_data, val_param, val_data_dict)
        if val_error < best_val_error:
            save(net)
```

**关键特点**：
- 用真实 GT 轨迹监督（有监督学习），而非物理残差
- 2步 autoregressive 训练，防止单步过拟合
- 用 val 误差（相对 L2）选择最优模型，而非 train loss
- 学习率极小（`lr=5e-5`），防止破坏 distillation 学到的结构

### 3.4 评估（test 函数）

三个阶段共用相同的 autoregressive rollout 评估逻辑：

```python
def test(config, net, test_data, test_param, test_data_dict):
    rollout_DT = config.rollout_DT
    total_iter = round(T_final / rollout_DT)   # 总推理步数
    sub = round(data_ratio / rollout_DT)        # 数据对齐步长
    
    w_pre = w_0  # 初始条件
    for _ in range(total_iter):
        w_0 = net(w_pre[..., -1:], param)[..., -1:]  # 单步推理
        w_pre = cat([w_pre, w_0], dim=-1)
    
    # 与 GT 对比，计算每步相对 L2 误差
    for t in range(1, total_iter+1):
        w_gt = test_data[..., sub * t]
        rel_err = norm(w_pre[t] - w_gt) / norm(w_gt)
```

### 3.5 关键超参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `rollout_DT` | 1.0 | 推理步长（继承自 Distillation） |
| `lr` | 5e-5 | 极小学习率（微调） |
| `num_iterations` | 200 | 很少迭代（防止过拟合） |
| `num_train` | 10 | 少量真实轨迹样本 |
| `train_step` | 由数据决定 | 训练时的步长对齐 |

---

## 三阶段对比总结

| 维度 | Pretrain | Distillation | Finetune |
|------|----------|--------------|----------|
| **模型** | OmniFluids2D（大） | Teacher=OmniFluids2D + Student2D（小） | Student2D（小） |
| **数据来源** | 在线随机采样 | 在线随机采样 | 预存真实轨迹 |
| **损失类型** | PSM 物理残差（无监督） | Teacher 软标签 MSE（知识蒸馏） | GT 轨迹 MSE（有监督） |
| **监督信号** | PDE 方程约束 | Teacher 模型输出 | 数值模拟数据 |
| **推理 dt** | 细（0.2） | Teacher 细→Student 粗 | 粗（1.0） |
| **分辨率** | 高（256） | Teacher 高→Student 低（128） | 低（128） |
| **迭代次数** | 多（~200k） | 中（~2k） | 少（~200） |
| **学习率** | 大（0.002） | 中（0.002） | 小（5e-5） |
| **验证指标** | PSM 物理残差 | 相对 L2 误差 | 相对 L2 误差 |
| **目标** | 泛化到宽参数空间 | 加速推理（dt 扩展） | 特定场景精度提升 |

---

## 数据流图（完整）

### Pretrain 数据流

```
随机采样 w0 (B,256,256,1)
随机采样 f  (B,256,256,1)
随机采样 ν  (B,)
          │
          ▼
param = [f, ν·1] (B,256,256,2)
          │
          ▼
OmniFluids2D.forward(w0, param)
  in_proj: cat(w0, grid, param) → Linear(5,W) → (B,256,256,W)
  × 12层 SpectralConv(DST/FFT) + MoE(ν→att)
  output_head: fc1a‖fc1b → Conv1d → (B,256,256,output_dim)
  残差: w0 + Δw * dt_frac
          │
          ▼
w_pre (B,256,256,output_dim)
cat([w0, w_pre]) → (B,256,256,T+1)
          │
          ▼
PSM_loss: fft2 → 计算 PDE 残差 → MSE(残差 - forcing)
          │
          ▼
backward → Adam + OneCycleLR
```

### Distillation 数据流

```
随机采样 w0, f, ν (同 Pretrain)
          │
          ├──────────────────────────────────────┐
          │  Teacher path (frozen)               │  Student path
          ▼                                      ▼
OmniFluids2D × teacher_step 步               Student2D × 1 步
  每步: net_t(w, param)[..., -1:]              net_s(w0_s, param_s)
  推进 teacher_step × dt_teacher = dt_student  推进 dt_student
          │                                      │
          ▼                                      ▼
w_gth (B,128,128,1)  ←── 下采样 ──           w_s (B,128,128,1)
          │                                      │
          └──────────────── MSE ─────────────────┘
                            │
                            ▼
                    loss = sqrt(MSE(w_s, w_gth))
                            │
                            ▼
                    backward → Adam + OneCycleLR
```

### Finetune 数据流

```
预存轨迹 data_train (N,128,128,T_total)
          │
          ▼
随机采样起始时刻 t
w0     = data[..., t]
w_gth1 = data[..., t + train_step]
w_gth2 = data[..., t + 2*train_step]
          │
          ▼
Student2D.forward(w0, param) → w_pre1 (单步)
loss1 = sqrt(MSE(w_pre1, w_gth1))
          │
          ▼
Student2D.forward(w_pre1, param) → w_pre2 (第2步, autoregressive)
loss2 = sqrt(MSE(w_pre2, w_gth2))
          │
          ▼
loss = loss1 + loss2
          │
          ▼
backward → Adam（lr=5e-5，极小）
```

---

## 关键设计决策解析

### 1. 为什么 Pretrain 用物理损失而非 GT 数据？

- **优点**：不需要预先运行大量数值模拟，可以在线生成无限多样本
- **优点**：物理约束天然泛化到未见参数（Re, forcing）
- **代价**：物理损失需要计算 PDE 残差（FFT 频域操作），计算量较大
- **代价**：不能直接优化预测精度，只能优化物理一致性

### 2. 为什么 Distillation 用软标签而非直接训练 Student？

- Teacher 的多步 rollout 提供了"粗 dt 下的正确答案"
- 直接用 PSM loss 训练 Student 也可以，但 Teacher 软标签更稳定
- 软标签包含了 Teacher 学到的隐式知识（不仅是最终状态，还有轨迹的"感觉"）

### 3. 为什么 Distillation 每个样本更新 10 次梯度？

```python
for _ in range(10):  # inner loop
    w_s = net_s(w0_s, param_s)
    loss = ...
    loss.backward()
    optimizer.step()
```

- 对同一个 (w0, w_gth) 对做多次更新，充分利用 Teacher 计算的软标签
- Teacher 推理（多步 rollout）计算成本高，因此尽量多用每个软标签

### 4. 为什么 Finetune 用 2步 autoregressive 而非单步？

- 单步训练容易过拟合到单步误差，但 autoregressive rollout 时误差累积
- 2步训练让模型学会"自己的输出作为下一步输入时也要正确"
- 防止模型在单步上精确但多步快速发散

### 5. 输出头的 train/inference 双路设计

```python
# 训练时（output_dim=40 帧）：
h = cat([fc1a(x), fc1b(x)])  # (W→36) ‖ (W→34) = 70 维
h → Conv1d(1,8,12,s=2) → Conv1d(8,1,12,s=2) → 40 帧

# 推理时（1 帧）：
h = fc1b(x)                  # 34 维
h → Conv1d(1,8,12,s=2) → Conv1d(8,1,12,s=2) → 1 帧
```

- 训练时输出多帧，使得 PSM loss 可以计算多个时间点的 PDE 残差（更稳定的梯度）
- 推理时只需要单帧，节省计算
- 两路共享 `fc1b` 和 `output_mlp`，保证推理路径与训练路径的一致性

---

## 与 MHD5F 迁移的关系

当前 `nse2d/pretrain/` 中已经实现了针对 5-field MHD 的 Pretrain 阶段，主要修改：

| 修改点 | 原始 NSE2D | MHD5F 版本 |
|--------|-----------|------------|
| 物理场数 | 1（涡度） | 5（n, U, vpar, psi, Ti） |
| 边界条件 | 全周期 FFT | x: DST（Dirichlet）, y: FFT（周期） |
| 物理损失 | PSM（频域 NS 残差） | `compute_rhs`（mhd_sim 时间差分残差） |
| 参数输入 | 标量 ν（粘度） | 8维参数向量（MHD 物理参数） |
| 分辨率 | 256×256 | 512×256（非正方形） |
| 数据来源 | 在线随机采样 | 预存 mhd_sim 轨迹（无在线采样器） |
| 输出头 | 单个共享头 | 5个独立头（每场一个） |

**Distillation 和 Finetune 阶段尚未针对 MHD5F 实现**，如需实现，参考上述分析：
- Distillation：需要一个 Student 模型（更小 modes/width），Teacher 多步 rollout 生成软标签
- Finetune：使用 mhd_sim 真实轨迹数据，2步 autoregressive MSE 微调

---

*文档生成时间：2026-02-24*
*基于代码：`/zhangtao/project2026/OmniFluids/nse2d/` 及 `nse2d_old/`*
