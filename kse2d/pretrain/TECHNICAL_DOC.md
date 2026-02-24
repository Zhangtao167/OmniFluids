# OmniFluids-HW 技术文档

> 当前配置: `TRAIN_LOSS_TYPE="mhd_sim"`, `OPERATOR_DISC="mhd_sim"`

## 1. 系统概述

当前系统是 **OmniFluids 架构 + mhd_sim 物理约束** 的混合方案：

| 组件 | 来源 |
|------|------|
| 神经网络架构 | OmniFluids (Factorized FFNO + MoE) |
| 数据生成 | mhd_sim (hw_eq_batch.py) |
| Loss 函数 | mhd_sim (rhs_anchored_loss) |
| 物理算子离散 | mhd_sim (Arakawa + FD) |
| 数据格式 | OmniFluids (channels-last) |

---

## 2. 数据

### 2.1 数据来源与格式

数据由 `mhd_sim/numerical/scripts/hw_eq_batch.py` 生成:

```
hw_dataset.pt = {
    'traj_zeta': [B, T, H, W],   # 涡度场
    'traj_n':    [B, T, H, W],   # 密度扰动
    'config':    dict            # HW 参数
}
```

### 2.2 数据参数

| 参数 | 值 |
|------|-----|
| 分辨率 | 128×128 |
| 总时长 | 0-250s (dt_data=1.0s) |
| 训练窗口 | 200-249s (spin-up后) |
| 样本数 | 100轨迹 × 50帧 = 5000快照 |

### 2.3 格式转换 (train.py: load_mhd_sim_data)

```python
# mhd_sim [B,T,H,W] → OmniFluids [B,S,S,C]
states = torch.stack([zeta_win, n_win], dim=2)  # [B,T,2,H,W]
pool = states.reshape(-1,2,H,W).permute(0,2,3,1)  # [N,S,S,2]
```

---

## 3. 网络架构 (OmniFluids)

### 3.1 整体结构

```
OmniFluids2D (68.3M params)
├── in_proj: Linear(n_fields+2+n_params → width)  # 9→128
├── f_nu[n_layers]: MLP(n_params → K)             # MoE权重生成
├── spectral_layers[n_layers]: SpectralConv2d_dy  # 主干
├── fc1a: Linear(width → 4*output_dim-4)          # 训练头
├── fc1b: Linear(width → 34)                      # 推理头
└── output_mlp: Conv1d(1→8→n_fields)              # 时间解码
```

### 3.2 SpectralConv2d_dy (Factorized FFT + MoE)

```python
# 动态加权多组傅里叶核
weight = einsum("bk, kioxy->bioxy", att, fourier_weight[K组])
x_ft = rfft(x, dim=-1)
out[:,:,:,:n_modes] = einsum("bixy,bioy->boxy", x_ft[:,:,:,:n_modes], weight)
return irfft(out)
```

### 3.3 训练/推理路径差异 (关键!)

| 模式 | 输出头 | 输出维度 | Tp |
|------|--------|----------|-----|
| 训练 (inference=False) | fc1a + fc1b | 4*40+30=190 | 40帧 |
| 推理 (inference=True) | fc1b only | 34 | 1帧 |

**前向传播**:
```python
def forward(self, x, param, inference=False):
    # 主干
    for i in range(n_layers):
        att = softmax(f_nu[i](param) / T)  # MoE权重
        x = x + spectral_layers[i](x, att)
    
    # 输出头 (路径不同!)
    if not inference:
        x = cat([fc1a(x), fc1b(x)])  # 训练: 190维
    else:
        x = fc1b(x)                   # 推理: 34维
    
    # 时间解码 + 残差
    x = output_mlp(x)                 # → [B,S,S,F,Tp]
    x = x_o + x * dt_ramp             # 残差连接
```

---

## 4. Loss 构建

### 4.1 rhs_anchored_loss (当前使用)

```python
def rhs_anchored_loss(w0, w_final, param, rollout_DT):
    # 模型预测的时间导数
    time_diff = (w_final - w0) / rollout_DT
    
    # 物理约束目标 (CN格式)
    with torch.no_grad():
        rhs_0 = compute_hw_rhs(w0)    # 输入端RHS (detach)
    rhs_f = compute_hw_rhs(w_final)    # 输出端RHS (梯度流通)
    target = (rhs_0 + rhs_f) / 2
    
    return MSE(time_diff, target)
```

**数学形式**:
$$\mathcal{L} = \left\| \frac{w_{pred} - w_0}{dt} - \frac{RHS(w_0) + RHS(w_{pred})}{2} \right\|^2$$

### 4.2 物理算子 (mhd_sim)

```python
def _compute_hw_rhs_mhd_sim(state, param, mhd_sim_root):
    hw = HasegawaWakatani(HWConfig(
        diffusion_method='fd'  # 有限差分超扩散
    ))
    # Arakawa scheme for Poisson bracket
    rhs = hw.compute_rhs(state.to(float64))
    return rhs.to(state.dtype)
```

### 4.3 Loss 计算流程 (train.py)

```python
def _compute_training_loss(config, w_all, param, train_interval):
    # w_all: [B, S, S, Tp+1, 2]
    n_steps = Tp  # 40
    step_dt = train_interval / n_steps  # 0.1/40 = 0.0025s
    
    loss_sum = 0.0
    for t in range(n_steps):
        loss_sum += rhs_anchored_loss(
            w_all[:,:,:,t,:],      # w_t
            w_all[:,:,:,t+1,:],    # w_{t+1}
            param, step_dt
        )
    return loss_sum / n_steps
```

---

## 5. 训练设置

### 5.1 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| rollout_DT | 0.1s | 物理时间跨度 |
| output_dim | 40 | 训练Tp=40帧 |
| step_dt | 0.0025s | 每子帧时间 |
| lr | 5e-4 | 学习率 |
| batch_size | 20 | 批大小 |
| noise_std | 0.001 | 输入噪声 |
| optimizer | Adam + OneCycleLR | 优化器 |

### 5.2 训练循环

```python
for step in range(500000):
    # 1. 随机采样
    w0 = pool[randint(5000, (batch_size,))]
    
    # 2. 输入噪声
    w0 = w0 + 0.001 * randn_like(w0)
    
    # 3. 前向 (Tp路径)
    w_pred = net(w0, param, inference=False)  # [B,S,S,40,2]
    w_all = cat([w0.unsqueeze(-2), w_pred])   # [B,S,S,41,2]
    
    # 4. 计算loss (40步)
    loss = _compute_training_loss(w_all, param, 0.1, 'cn')
    
    # 5. 反向传播
    loss.backward()
    optimizer.step()
```

---

## 6. 推理流程

```python
def test(config, net, test_data, test_param):
    substeps = round(dt_data / rollout_DT)  # 1.0/0.1 = 10
    
    w_current = test_data[:,:,:,0,:]  # 初始帧
    for _ in range(total_iter):
        # 每个数据帧需要 10 次模型推理
        for _ in range(substeps):
            w_current = net(w_current, param, inference=True)
        predictions.append(w_current)
```

---

## 7. 与原始 OmniFluids 区别

### 7.1 已改变

| 项目 | 原始 OmniFluids | 当前系统 |
|------|----------------|----------|
| 方程 | KS (单场) | HW (双场) |
| 数据 | 在线GRF生成 | mhd_sim离线数据 |
| Loss | PSM_loss (自监督残差) | rhs_anchored_loss (锚定) |
| 算子 | 纯谱方法 | Arakawa + FD |
| 初始条件 | 随机GRF | 200s后湍流态 |

### 7.2 保留的 OmniFluids 贡献

| 贡献 | 说明 |
|------|------|
| 网络架构 | Factorized FFNO + MoE 完整保留 |
| 参数条件化 | 5维向量 (α, κ, logν, N/4, k0) 注入 |
| 位置编码 | 2D周期grid嵌入 |
| Tp帧输出 | 训练时40帧密集监督 |
| dt ramp | 残差输出×时间权重 |
| 数据格式 | channels-last [B,S,S,C] |

---

## 8. 与 mhd_sim 区别

### 8.1 网络架构

| 项目 | mhd_sim | 当前系统 |
|------|---------|----------|
| 架构 | UNet/ConvNet | FFNO+MoE |
| 参数量 | ~2M | ~68M |
| 输入格式 | channels-first | channels-last |
| 参数条件 | dt通道 | 5维向量 |

### 8.2 Loss (相同)

| 项目 | mhd_sim | 当前系统 |
|------|---------|----------|
| 形式 | MSE(time_diff, target) | **相同** |
| RHS_0 detach | 是 | **相同** |
| 算子 | Arakawa + FD | **相同** |

### 8.3 训练策略

| 项目 | mhd_sim | 当前系统 |
|------|---------|----------|
| 优化器 | AdamW | Adam |
| lr | 1e-4 固定 | 5e-4 OneCycle |
| 精度 | float64 | float32 |
| 训练dt | 0.025s | 0.0025s |
| 路径一致性 | 是 | **否** |

---

## 9. 数据流图

```
训练阶段:
hw_dataset.pt [B,T,H,W]
    ↓ load_mhd_sim_data()
pool [5000, 128, 128, 2]
    ↓ 随机采样
w0 [20, 128, 128, 2]
    ↓ + noise
    ↓ OmniFluids(inference=False)
w_pred [20, 128, 128, 40, 2]
    ↓ cat([w0, w_pred])
w_all [20, 128, 128, 41, 2]
    ↓ rhs_anchored_loss × 40步
loss
    ↓ backward()

推理阶段:
w [100, 128, 128, 2]
    ↓ OmniFluids(inference=True) × 10 (substeps)
w [100, 128, 128, 2]  # 1s后状态
    ↓ 重复
predictions
```

---

## 10. 实验结果

| loss_type | operator | Mean Rel L2 (10s) |
|-----------|----------|-------------------|
| mhd_sim | mhd_sim | **0.128** |
| omnifluids | mhd_sim | 1.27 |
| mhd_sim | spectral | 1.45 |

**结论**: 锚定loss + 一致算子 = 最佳效果

---

## 11. 关键改进空间

1. **训练/推理路径不一致**: 训练用fc1a+fc1b, 推理只用fc1b
2. **精度**: 当前float32, mhd_sim用float64
3. **学习率**: OneCycleLR可能后期过低

---

## 12. 文件结构

```
OmniFluids/kse2d/pretrain/
├── main.py          # 入口
├── model.py         # OmniFluids2D
├── train.py         # 训练循环
├── psm_loss.py      # Loss函数
├── eval.py          # 评估
├── run_all.sh       # 一键脚本
└── TECHNICAL_DOC.md # 本文档
```
