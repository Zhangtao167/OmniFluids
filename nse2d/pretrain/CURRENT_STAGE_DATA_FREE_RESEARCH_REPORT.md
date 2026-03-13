# 当前阶段研究总结：`nse2d/pretrain` 中的 5-field MHD OmniFluids

## 0. 这份文档的目的

这不是一份单纯的代码说明，而是一份面向后续科研决策的阶段性研究文档。目标是回答五个问题：

1. 这套代码现在到底在做什么，不在做什么。
2. 目前已经有哪些结论可以直接从代码和结果中确认。
3. 你想要的“彻底 data-free 训练一个仿真模型”，距离当前实现还有多远。
4. 当前最核心的科学瓶颈是什么。
5. 接下来最值得投入的方向是什么，包括代码、实验、理论、文献和与老师讨论的问题。

## 0.1 本轮修订说明

这版文档相比前一版，刻意做了三件事：

1. 把内容分成 **代码硬事实 / 当前日志证据 / 历史线索 / 工作假设** 四个层级，避免把推测写成结论。
2. 把所有 `combined/supervised` 相关的正面判断降级处理，因为当前脚本树下存在 **监督时间尺度可能错配** 的风险，在没有重新核实时不能把它当成硬证据。
3. 补充几个会直接干扰科学结论的实现混杂因素：  
   `FiveFieldMHDConfig` 主要靠默认值、不同 loader 的时间索引假设不完全一致、offline 多卡没有 `DistributedSampler`、rollout 阶段缺少统一的 dealias/boundary wrapper、模型与 RHS 之间是 `fp32 -> fp64 -> fp32` 的精度链路。

因此，这份文档现在更强调：

- 哪些结论是已经能站住的；
- 哪些只是当前最强工作假设；
- 哪些实验必须先清理协议之后才能继续解释。

---

## 1. 先给结论

### 1.1 当前系统的真实定位

当前 `OmniFluids/nse2d/pretrain/main.py` 已经不是原始的 `nse2d` NS 预训练入口，而是一个 **2D 5-field MHD OmniFluids 实验平台**。  
它做的是：

- 用 `OmniFluids` 风格的谱算子 + MoE 主干网络，
- 去学习 `mhd_sim` 中的 2D 5-field MHD 流映射，
- 并支持多种训练模式：`offline / online / staged / alternating / self-training / learnable-GRF / combined loss`。

换句话说，当前仓库名叫 `nse2d`，但主入口的真实任务已经是 **5-field MHD surrogate training**。

### 1.2 当前最重要的科学判断

从代码和现有结果看，当前最值得优先检验的，并不是“继续堆更强的 PDE loss”，而是下面几类更基础的问题：

1. **训练状态分布很可能不对**  
   这是我当前最强的工作假设之一。physics loss 在强锚定的真实轨迹附近可以工作，但在 raw GRF 这种离物理流形很远的分布上，loss 下降并不等价于 rollout 变好。  
   但它还不能被表述成“唯一已证实根因”，因为当前实现里还有其他未清干净的混杂因素。

2. **训练目标和推理路径不完全一致**  
   尤其是 `output_dim > 1` 时，训练优化的是“单次前向产生一段子帧轨迹”，评估却是“`inference=True` 的单步自回归 rollout”。  
   再叠加 `dealias_input` 只在训练前处理输入、评估时不处理，以及 rollout 阶段没有统一 boundary/dealias wrapper，会进一步放大误差累积。

3. **当前所谓 online/data-free 还不是真正严格意义上的 data-free**  
   默认 GRF 生成器会从离线数据统计 `field_scales`，模型选择还直接看测试集 MHD rollout，GRF 测试集本身也是按数据统计生成的。

4. **当前最直接的正面证据来自“真实状态锚定”的场景**  
   也就是 `offline physics-only` 的单轨 overfit 确实能学。  
   但 `combined/supervised` 是否在当前代码树下稳定更强，暂时不能写死，因为现有脚本存在监督时间跨度可能没有和 `dt_data` 对齐的风险。

### 1.3 一句话概括当前阶段

**你现在已经把问题推进到了两个必须并行回答的层面：一是当前实现里哪些协议/实现混杂因素在伤害结果，二是 strict data-free 到底需要什么样的物理先验 / 状态生成器 / 稳定性约束。**

---

## 2. 你真正要解决的科学问题

我建议把目标精确表述为：

> 在不依赖预生成仿真轨迹监督的前提下，仅利用 PDE、边界条件、物理结构和可接受的先验，训练出一个对 5-field MHD 长时 rollout 稳定的 surrogate model。

这里必须把 “data-free” 分层说清楚，否则容易在讨论里混淆。

### 2.1 我建议使用的数据自由度分级

#### Level A: 有监督数据驱动

- 使用真实 `(x_t, x_{t+1})` pairs。
- 当前 `supervised_loss_weight > 0` 的实验属于这一类。

#### Level B: 无 future label，但使用真实状态样本

- 不用真实下一时刻标签，只用真实 `x_t` 状态作为锚点，配合 PDE loss。
- 当前 `offline + physics loss only` 属于这一类。
- 这不是严格 data-free，但它是 **label-free / trajectory-target-free**。

#### Level C: 无离线标签，但仍然使用数据统计

- 例如当前 online GRF 默认 `grf_scale_from_data=1`。
- 这说明虽然训练时不直接喂 trajectory pair，但 GRF 分布仍然是用真实数据标定的。
- 这属于 **weakly data-free / data-informed prior**。

#### Level D: 真正严格的 data-free

- 不用任何真实 trajectory；
- 不用真实数据统计来设定 GRF 振幅；
- 不用测试集选模型；
- 只能用 PDE、边界条件、理论先验、解析模式、物理约束或在线求解器辅助。

**你真正想冲击的是 Level D。**

当前代码整体仍处在 **Level B/C 之间**，还没有到严格的 Level D。

---

## 3. 代码实际上实现了什么

## 3.1 入口与整体训练框架

入口：`nse2d/pretrain/main.py`

主流程：

1. 解析配置。
2. 创建 run 目录和日志。
3. 构建 `OmniFluids2D`。
4. 进入 `train.py::train()`。
5. 根据 `data_mode` 决定数据来源：
   - `offline`: 仿真数据切片
   - `online`: GRF
   - `staged`: 先 GRF 再 offline
   - `alternating`: 在线/离线轮换
6. 训练时可叠加：
   - physics loss
   - supervised loss
   - soft-Linf
   - multi-step PDE loss
   - self-training
   - learnable GRF
7. 周期性在 `eval_data_path` 上做 rollout evaluation。
8. 按 held-out `eval_data_path` 上的 MHD rollout 误差选择 best checkpoint。

### 一个非常重要的事实

当前代码里确实有一个 held-out 的 `eval_data_path`。  
但它实际上同时承担了：

- 选 best checkpoint 的 validation 角色
- 最终汇报结果时的 test 角色

也就是说，**现在缺的不是“任何 eval 集”，而是独立的 final test set。**

这对科研阶段可以接受，但对最终论文结论是不干净的。

---

## 3.2 数据到底是什么

### 仿真数据

从 `batch_config.yaml` 和 `tools.py` 可确认：

- 网格：`Nx=512, Ny=256`
- 模拟方程：`mhd_sim/numerical/equations/five_field_mhd.py`
- solver `dt = 0.002`
- `stamp_interval = 500`
- 所以 `dt_data = 1.0`
- 每个数据集有 `n_samples = 10` 条轨迹
- 当前训练窗口通常是 `[250, 300] s`

因此：

- 每条轨迹约有 `51` 帧
- offline physics 训练快照数：`10 × 51 = 510`
- supervised 连续 pair 数：`10 × 50 = 500`

这意味着当前默认大模型是在 **大约 500 个量级的真实状态/监督 pair** 上训练。

### 一个容易被忽略的静默风险：时间索引假设在不同 loader 里不完全一致

当前代码有两种时间索引逻辑：

- `load_mhd5_snapshots()` / `load_mhd5_trajectories()` 会优先读取 `metadata['time_start']` 或 `t_list[0]`
- `_load_supervised_pairs()` 和 `MHD5FieldGRF.from_data_stats()` 直接按 `time_start / dt_data` 算索引，等价于默认文件从 `0s` 开始

对当前原始数据文件，这大概率没有出错。  
但如果以后换成“已经裁过时间窗后重新保存”的数据，这里会有 **静默时间错位** 的风险。

### GRF 在线数据

来自 `tools.py::MHD5FieldGRF`：

- 5 个场独立生成：`n, U, vpar, psi, Ti`
- x 方向用 Dirichlet window
- y 方向用 periodic rFFT
- 支持 radial mask，默认活跃区硬编码为 `x=[180,330]`
- 默认还会从真实数据里提取 `field_scales`

这说明当前 GRF 不是“纯理论初值”，而是：

- **边界形状**是人工指定的
- **幅值尺度**通常来自真实数据
- **各场之间没有联合协方差，也没有相位耦合**

这正是当前 online data-free 难以成功的核心原因之一。

---

## 3.3 模型到底是什么

模型文件：`nse2d/pretrain/model.py`

### 主干

- 不是 UNet
- 是 OmniFluids 风格谱算子主干
- x 方向用 DST 处理 Dirichlet BC
- y 方向用 FFT 处理 periodic BC
- 每层有一个基于 physics parameter 的 MoE gating

### 但这里有一个关键现实

虽然模型形式上支持 `n_params`，而且有 `default_params`，但训练时几乎所有地方都在调用：

`net(x)` 或 `net(x, inference=True)`

也就是说：

- `params` 基本没有从数据或配置里显式传入
- gating 实际上是固定默认参数驱动的
- 当前并没有真正训练一个“跨参数泛化的 operator”

因此，当前 MoE 在实践里更像是一个 **固定 gating 的大模型**，而不是充分发挥参数条件化能力的 OmniFluids。

### 输出结构

- `output_dim > 1` 时：单次前向输出一段多帧子轨迹，训练和推理仍是双路径
- `output_dim == 1` 时：代码里已经实现了 shared one-step path，head 级别的 train / inference 路径是一致的
- 每个 field 一个独立 `OutputHead`
- 最后输出是 residual form：`x_0 + residual × dt_frac`

这点很重要，因为它说明 “one-step unified path” 不是一个还没落地的想法，而是 **代码已经部分具备支撑，只是实验主线还没有完全切过去**。

### 当前默认大模型参数量

我直接按当前默认配置核算，参数量约为：

- **158.16M**

这和当前训练数据规模形成非常尖锐的对比：

- 大约 500 个真实 pair / 510 个真实状态
- 对应 1.58e8 参数

这也是为什么“纯监督容易漂”“physics-only 容易被错误分布带偏”的背景之一。

---

## 3.4 损失函数到底是什么

文件：`nse2d/pretrain/psm_loss.py`

当前已经不是原始 `nse2d` 的 NS PSM loss，而是 **直接调用 `mhd_sim` 的 5-field MHD RHS**。

也就是说，当前方法的真实定位更准确地说是：

> OmniFluids 架构 + `mhd_sim` 的 five-field RHS + 多种数据模式与训练技巧

而不是“原始 OmniFluids 方法原封不动迁移”。

### 一个必须单独标红的问题：supervised 分支的时间尺度对齐

当前 supervised 分支只有在显式设置了足够大的 `supervised_n_substeps` 时，才会把多次 `rollout_dt` 累积到一个 `dt_data` 上再和 `x_target` 对齐。

这意味着如果出现下面这种设置：

- `rollout_dt = 0.1`
- `dt_data = 1.0`
- `supervised_n_substeps = 1`

那么训练里实际拿去做监督的 `pred_traj[..., -1]` 对应的是 `t + 0.1`，却会被直接和 `x_{t+1.0}` 比。

因此：

- 所有 `supervised / combined` 结果都必须先检查时间跨度是否对齐
- 在这件事重新核实前，不能把当前树下的 `combined` 表现当成强证据
- 这也是为什么本轮文档会主动下调 `combined baseline` 的证据等级

### Physics loss

定义是：

- 预测一段多帧轨迹 `pred_traj`
- 对每一对子帧 `(state_t, state_{t+1})`
- 约束 `(state_{t+1}-state_t)/dt` 接近 PDE RHS target

支持：

- Euler
- Crank-Nicolson
- mixed Euler/CN
- MAE
- soft-Linf

### RHS 计算与物理配置的实现细节

这一块还有两个必须明确写出来的现实：

1. **精度链路是混合的**  
   模型主干主要工作在 `float32`；`rhs_fn` 内部会把状态转成 `float64` 调 `mhd_sim` 的 RHS，再 cast 回模型 dtype。

2. **当前 `FiveFieldMHD` 配置主要依赖默认值**  
   `build_mhd_instance()` 这里只显式设置了 `Nx / Ny / device / precision`；其他物理参数沿用 `FiveFieldMHDConfig` 的默认值。  
   只要当前数据集刚好和默认参数完全一致，这样不会立刻出错；但一旦以后更换数据源、改物理参数或做跨参数实验，这里就存在 **静默失配** 风险。

### 但这里有一个非常关键的实现细节

当前 `compute_physics_loss()` 在构造 target 后做了统一 `detach()`：

- `target = target.detach()`

这意味着：

- 即使选择 `crank_nicolson`
- `rhs(state_{t+1})` 也不会对预测分支反传梯度

所以当前实现并不是严格意义上的“真隐式 CN anchored loss”，而是：

- **time-difference 端有梯度**
- **RHS target 端整体是停梯度的**

这会比真正让梯度穿过 `rhs(pred)` 的 CN loss 更弱。

### Supervised loss

`train.py::_compute_supervised_loss`

- MSE + 可选 MAE
- 可选 `supervised_n_substeps`
- 可选线性插值伪 pair

这块逻辑本身很关键，因为它实际上是在补“physics-only 不足以提供分布锚点”的问题。

---

## 3.5 dt 体系

当前代码里同时有三层 dt：

- solver dt：`0.002`
- data dt：`1.0`
- model rollout dt：默认 `0.1`

再加上：

- `output_dim=10`

所以训练时单个 forward 里的子帧间隔是：

- `train_dt = rollout_dt / output_dim = 0.01`

而评估时：

- 一个 data step = `1.0`
- 一个 model step = `0.1`
- 所以每个 data step 需要 `10` 次 inference

这带来一个重要现象：

- 训练默认优化的是单次 forward 内部的 `0.01` 子帧物理一致性
- 评估却看 `0.1` 单步 repeated rollout 的长期稳定性

这就是当前 train/inference mismatch 的核心之一。

---

## 4. 已有结论中，哪些是我能直接确认的

### 证据等级约定

为了避免把“印象”“历史报告”“当前代码硬事实”混在一起，这里用四档证据等级：

- **A级**：当前代码树里可以直接读代码确认的硬事实
- **B级**：本轮直接读到的当前日志/结果文件支持的经验结论
- **C级**：仓内旧报告、旧实验或间接线索，尚未在本轮原始日志里重核
- **H级**：工作假设，需要后续实验专门证伪/证实

## 4.1 B级直接证据 1：physics-only 在单轨真实状态 overfit 场景下可学

`results/exp2_physics_overfitting/.../eval-*.json`

直接结果：

- `mean_rel_l2 = 0.02669`
- 10-step rollout
- `n_substeps = 10`
- 单条训练轨迹 overfit

这说明：

1. 当前模型容量不是完全不够。
2. 当前 physics loss 不是完全无效。
3. 在 **单轨、强锚定、overfit** 的设置下，当输入状态贴着一条真实轨迹流形走，模型可以把局部动力学学得很准。

**这是当前最关键的正证据之一。**  
但它不能直接外推成：

- `physics-only` 在 held-out 真实轨迹上普遍已经足够
- “future label 不重要”已经被完全证明
- 当前所有失败都只剩分布问题

## 4.2 B级直接证据 2：staged 能回到可用水平，但更像是 offline 把模型救回来

`results/exp6_grf_staged/.../eval-*.json`

直接结果：

- `mean_rel_l2 = 0.33940`
- 10-step rollout
- `n_substeps = 10`

这说明 staged 这条线不是完全没价值，但它不能证明 GRF warmup 真正学到了正确物理；它只证明：

- 在足够强的 offline phase 之后，模型可以被拉回到还不错的水平。

## 4.3 B级直接证据 3：几条 online 强化路线在当前阶段仍然很差

从 `termwise_metrics_*.txt` 可直接看到：

### `exp18_self_training_v2`，`step_30000_mhd`

- 4个观测步 `{1,3,5,10}` 平均 total L2：`6.478`
- 第 10 步 total L2：`17.893`

### `exp20_learnable_grf_10k_soft_linf`，`step_30000_mhd`

- 4步平均 total L2：`10.264`
- 第 10 步 total L2：`29.481`

### `exp21_learnable_grf_10k_soft_linf_multistep`，`step_40000_mhd`

- 4步平均 total L2：`13.633`
- 第 10 步 total L2：`33.784`

### `exp22_self_training_linf_multistep`，`step_30000_mhd`

- 4步平均 total L2：`5.442`
- 第 10 步 total L2：`12.878`

这些数字虽然不是 `evaluate()` 里的“所有步平均”，但已经足够说明：

- self-training
- learnable GRF
- soft-Linf
- multi-step PDE

这些增强项在 **当前协议和当前训练分布设定下**，并没有把问题根本救回来。

## 4.4 C级线索：仓内旧诊断报告里出现过更强的 combined baseline

`results/PRETRAIN_DIAGNOSIS_REPORT_2026_03_08.md` 中记录：

- 历史 `exp3_combined_physics_supervised` 曾达到 `~0.306`

但这条结论我这轮没有直接重读到原始日志，因此应视为：

- **已有内部报告支持**
- **但尚未在本轮用原始日志重新核实**

此外，本轮重新核代码后还发现：  
当前树下 `run_exp3_combined_loss.sh` 这类脚本在 `rollout_dt=0.1` 时，如果没有显式把 `supervised_n_substeps` 设到与 `dt_data / rollout_dt` 一致，就会存在 **监督时间尺度错配** 风险。

所以这条 combined 结论现在最多只能作为：

- **历史线索**
- **待复核 baseline**

而不应当在本轮文档里被当成“strongest practical baseline”的硬依据。

## 4.5 本轮最稳妥的证据矩阵

### A级：代码硬事实

- `eval_data_path` 同时承担 validation 和 final test 角色
- `output_dim > 1` 时训练/推理路径不一致
- rollout 阶段没有统一的 dealias + boundary wrapper
- RHS 计算使用 `fp32 -> fp64 -> fp32`
- `FiveFieldMHDConfig` 主要依赖默认值
- 不同 loader 的时间索引假设不完全一致
- offline 多卡没有 `DistributedSampler`

### B级：当前日志可直接支持的经验结论

- `physics-only` 在单轨真实状态 overfit 场景下可学
- staged 的主要作用更像是被 offline phase 拉回
- 多条 raw online 强化路线当前仍然明显不稳定

### C级：当前只能保留为线索的结论

- `combined loss` 可能更强
- 某些旧实验曾达到更低误差

### H级：当前最值得继续检验的工作假设

- 状态分布错配是主要矛盾之一
- 五场联合结构缺失是 GRF 失败的重要原因
- 长时 rollout 不稳与局部误差传播放大同样有关

---

## 5. 当前阶段最重要的可支持结论与工作假设

## 5.1 当前能稳妥支持的结论

截至目前，我认为可以稳妥写进结论里的，主要是下面几条：

- **局部 PDE 一致性本身不是完全学不会。**  
  单轨真实状态 overfit 已经证明，在强锚定条件下它可以学到很低误差。

- **raw GRF online 目前在长期 rollout 上确实不稳。**  
  这一点已经被多条 online 强化路线反复侧面支持。

- **staged 的收益，当前更可信的解释是 offline phase 在“救模型”。**  
  它还不能被解释成 “GRF warmup 已经学到了足够正确的物理”。

- **`combined/supervised` 的强弱关系当前不能写死。**  
  在监督时间尺度重新核实前，这条线只能被视为待确认。

## 5.2 当前最强的工作假设之一：状态分布错配非常重要，但还不是唯一已证实根因

这是我当前最相信的中间判断，但它必须带限定语：

- raw GRF 很可能远离真实 attractor 邻域
- 在 off-manifold 分布上把局部 residual 压小，并不等价于在真实 rollout 分布上稳定
- 五场联合结构缺失，很可能比“每场谱不够像”更关键

但它还不能直接升级成“唯一根因”，因为当前还同时存在：

- 训练/推理路径不一致
- rollout 阶段预处理不一致
- supervised 时间尺度可能错配
- physics config 默认值依赖
- 多卡数据采样协议不完全干净

因此更稳妥的表述应是：

> **状态分布错配是当前最强工作假设之一，但必须在清理协议后再去证明它是不是“主因”。**

## 5.3 第一性原理视角：长期 rollout 误差由“局部误差 + 误差传播”共同决定

只从“训练分布对不对”看问题还不够。  
从第一性原理看，长期 rollout 的误差至少受两部分共同决定：

1. **局部一步误差有多大**
2. **这一步误差在动力系统中会被怎样传播和放大**

如果记真实一步流映射为 `F`，模型为 `G`，则在轨迹附近可以把 rollout 误差近似写成：

> `e_{k+1} ≈ J_F(x_k) e_k + (G(x_k) - F(x_k))`

- `G(x_k) - F(x_k)` 对应局部建模误差
- `J_F(x_k) e_k` 对应误差在后续动力学中的传播/放大

所以即使：

- one-step loss 已经变小
- 训练分布也比以前更像真实状态

只要 rollout 过程中 `J_F` 的有效放大作用还很强，或者模型输入开始偏离训练分布，长期结果仍然会炸。

这说明后续研究必须同时管三件事：

- on-manifold 的局部精度
- rollout 路径上的输入分布
- 长时误差传播/稳定性控制

## 5.4 当前系统已经非常接近一个明确的研究命题

我认为你现在的真正研究命题已经不是：

> “OmniFluids 能不能做 5-field MHD”

而是：

> “在 five-field MHD 中，怎样构造一个不依赖真实轨迹监督、但足够贴近物理流形的训练状态分布，使得局部 PDE residual 训练能转化成长期 rollout 能力？”

这已经是一个很清楚的科学问题了。

---

## 6. 代码层面的关键问题与风险

## 6.1 训练/推理路径不一致

### 问题 A：`output_dim > 1` 时训练和推理不是同一个任务

- 训练：一次 forward 产生多子帧
- 评估：`inference=True` 单步重复 rollout

这会导致：

- 训练阶段没真正暴露给“长期自回归输入分布”
- 单步 rollout 稳定性不是直接优化目标

### 一个重要补充

代码里其实已经为 `output_dim == 1` 做了 shared one-step path。  
所以这里真正缺的不是“从零设计 one-step 模型”，而是：

- 把实验主线真正切到这条路径上
- 并把训练、评估、脚本、日志口径一起统一

## 6.2 rollout 阶段缺少统一 preprocessing wrapper

在 `_train_step()` 里：

- 如果 `dealias_input=1`，训练前会先把 `x_0` dealias

但 `evaluate()`、`inference.py`、`visualize_model_rollout.py` 的 rollout 过程中：

- 没有在每一步把 `current` 做同样的 dealias
- 也没有显式的 boundary clamp / boundary wrapper

这意味着：

- 模型训练时看到的是“更干净的输入”
- 推理时却直接吃自己输出的原始高频内容和边界附近误差

这很可能直接加剧多步爆炸。

## 6.3 supervised 分支存在时间尺度错配风险

这是本轮新确认的关键问题之一。

如果：

- `rollout_dt = 0.1`
- `dt_data = 1.0`
- `supervised_n_substeps = 1`

那么 supervised 分支里被拿去和 `x_target` 比较的预测，其实只走到了 `t + 0.1`。  
这会让：

- `supervised` 结果变得难解释
- `combined` 结果变得更难解释
- 不同脚本之间的比较口径直接失真

## 6.4 PDE loss 比表面看起来更弱，而且物理配置目前主要靠默认值

虽然表面上叫 CN / mixed integrator，但当前 target 被整体 `detach()`：

- 没有把梯度穿过 `rhs(pred)`
- 不是强隐式约束

所以当前 PDE loss 更像：

- detached target tangent matching

而不是真正的 fully coupled implicit residual matching。

除此之外，还有一个实现层面的风险：

- `build_mhd_instance()` 只显式设置了 `Nx / Ny / device / precision`
- 其余物理参数默认取 `FiveFieldMHDConfig()` 的默认值

这对“当前这一个数据集”大概率没问题，但对以后所有跨参数、换数据、换配置的实验都埋了静默失配风险。

## 6.5 参数条件化几乎没有真正发挥

- 模型形式支持 `n_params`
- 但训练时基本不传 `params`
- gating 只吃 `default_params`

所以当前 MoE 复杂度很高，但没有体现“跨参数泛化”的研究价值。

## 6.6 当前默认模型非常大

- 当前默认参数量：`158.16M`

在只有：

- `510` 个真实快照
- `500` 个监督 pair

的条件下，这个模型规模非常激进。

## 6.7 评估协议不够干净

当前最佳模型选择：

- 直接看 `eval_data_path`
- 默认就是 MHD test set

因此：

- 现在的 “best checkpoint” 实际上已经见过测试集
- 这会放大科研阶段的调参效率
- 但不适合作为最终 paper-level 泛化结论

## 6.8 batch size / 多卡设置使很多实验不严格可比

这是一个非常容易被忽略，但其实很重要的问题。

在当前代码里：

- `batch_size` 是 **每个进程 / 每张卡** 的 batch
- 多卡时全局 batch = `batch_size × num_processes`

但脚本之间：

- GPU 数不同
- `batch_size` 不同
- `num_iterations` 也不同
- 学习率通常没有按 global batch 调整

这意味着不同实验之间：

- 每步看到的样本数不同
- 总样本曝光量不同
- 每个 pair 被重复使用的次数不同

所以“按 step 横向比较”并不总是公平。

还有一个更硬的实现事实：

- 当前 offline 多卡 DataLoader 没有使用 `DistributedSampler`
- 在多进程下，各进程是在各自 shuffle 同一份完整数据集

这会让“全局 batch 的真实语义”和“每步样本覆盖率”都变得不够标准。

## 6.9 时间索引和统计提取在不同入口里不完全一致

- `load_mhd5_snapshots()` / `load_mhd5_trajectories()` 会读取 `metadata['time_start']`
- `_load_supervised_pairs()` 直接按 `time_start / dt_data` 取索引
- `MHD5FieldGRF.from_data_stats()` 也直接按 `time_start / dt_data` 取索引

对当前原始数据问题不大，但只要数据文件换成裁剪版或重存版，就可能出现 **你以为在取 `[250,300]s`，实际没有完全对齐** 的情况。

## 6.10 一些配置和脚本存在漂移或陷阱

### 例子

- `main.py` 里 `--layer_norm` 用的是 `store_true` 且 `default=True`，这使它几乎无法从 CLI 设成 `False`
- `run_exp21.sh` / `run_exp22.sh` 中对 `multi_step_pde_detach` 的注释和 echo 文案有漂移
- `run_exp5_sim_supervised_overfit.sh` 脚本文案写的是 eval test set，但代码里 `is_overfitting_test=1` 会强制用训练轨迹做 eval
- 评估主逻辑同时存在于 `main.py --mode inference`、`inference.py`、`visualize_model_rollout.py`

这些都不是最核心的科学问题，但足以显著干扰结果解释。

---

## 7. 对你当前几个核心困惑的直接回答

## 7.1 `batch_size` 影响什么？

### 代码层面

它影响：

1. 每步梯度噪声大小
2. online 模式下一步里看到多少种 GRF
3. learnable GRF 参数梯度的 Monte Carlo 方差
4. 多卡下 global batch
5. 在固定 `num_iterations` 下的总样本曝光量

### 科学层面

它不是当前第一瓶颈，但如果不控制，会严重影响实验可比性。

### 我的判断

- 对 offline 小数据，`batch_size` 更像是“优化和可比性控制变量”
- 对 online GRF，大 batch 会让模型每步看到更广的 off-manifold 分布，可能更稳定，但也可能更难聚焦
- 如果要做严谨对比，应固定：
  - global batch
  - total seen samples
  - total model NFE
  - total RHS calls

## 7.2 为什么 GRF 训练时 train loss 下降、单步 eval 也变好，但多步 rollout 越来越差？

这是当前系统最核心的现象之一。

我的解释是四层叠加：

1. **局部目标没错，长期目标没被直接优化**
   - loss 主要看局部时间差分与 RHS 一致性
   - 单步可以变好，不代表长期 rollout 稳

2. **训练分布不是 rollout 真正会到达的分布**
   - raw GRF 输入和真实 attractor 上的状态差太远
   - 模型学到的是 “如何在错分布上局部满足 PDE”

3. **推理时吃的是自己的输出**
   - 训练时主要看 teacher-like 输入
   - 推理时每一步都喂回自己的状态

4. **训练和评估的预处理不一致**
   - 训练输入可 dealias
   - 评估 rollout 没有等价处理

这不是一个普通的 exposure bias 小问题，而是：

> **局部 PDE 一致性目标、分布错配与 rollout 自反馈** 是当前最可信的一组共同解释。

## 7.3 为什么“GRF 预训练再 simulation data 继续训练”目前最好理解？

因为这个流程本质上做了两件事：

1. 用 GRF 先让模型学会一个大致的局部算子形状
2. 再用真实状态分布把它拉回物理流形附近

所以它好，不意味着 raw GRF 本身足够好；更可能意味着：

- **真实 simulation state 仍然是关键锚点**

## 7.4 “核心问题是不是构建更符合物理的随机生成数据？”

**是，但不止如此。**

更准确地说，核心问题是：

1. 构建更物理的随机状态分布
2. 让训练目标真正对 rollout 稳定性负责
3. 让训练/推理预处理保持一致
4. 让评估协议足够干净

如果只改第 1 点，后 3 点不改，问题仍然会反复出现。

---

## 8. 我对当前阶段的核心研究假设

## 假设 1：离线真实状态之所以有效，不是因为它提供了很多样本，而是因为它提供了“正确流形上的状态锚点”

证据：

- 真实状态样本数其实很少
- physics-only overfit 仍然能成功

推论：

- data-free 的关键不是更多随机样本
- 而是更好的 **on-manifold prior**

## 假设 2：当前 GRF 生成器的主要问题不是光谱太糙，而是缺少五场联合结构

因为现在 GRF 只控制：

- 每场自己的 alpha/tau/scale

却没有控制：

- `n, U, vpar, psi, Ti` 的联合统计

所以即使 learnable GRF 把 10 个标量参数学得再好，也很难靠这么低维的自由度把状态分布推近真实 attractor。

## 假设 3：如果在修正监督时间尺度后 combined 仍然显著更强，那么更可能是因为“给了轨迹锚点”，而不是“loss 形式更复杂”

这意味着 strict data-free 的最终方向，可能需要替代 supervised pair 的“弱锚点”来源，例如：

- 更物理的生成先验
- 短 solver warmup
- 解析线性模态先验
- 自蒸馏一致性约束

## 假设 4：如果不先把现有 `output_dim == 1` shared path 变成实验主线，就很难干净回答 `model_dt = 0.1` 是否可行

这会影响后续很多实验的解释可信度。

## 假设 5：即使状态先验变得更物理，如果不同时控制误差传播，长时 rollout 仍然可能失稳

也就是说，strict data-free 不是只要找到“更像的初值分布”就够了，还需要：

- 更匹配 rollout 的训练目标
- 更一致的推理路径
- 更明确的稳定性控制

---

## 9. 接下来最值得做的事情

## 9.1 代码模块层面

### P0：先修监督时间尺度，再谈 combined baseline

第一优先级不是继续加 loss 技巧，而是先确认：

- `supervised_n_substeps`
- `rollout_dt`
- `dt_data`
- `x_target` 的物理时间跨度

在所有 `supervised / combined` 脚本里完全对齐。

### P0：把现有 one-step shared path 真正变成实验主线

这不是从零实现，因为 head 级别的 shared path 已经在代码里。  
真正要做的是：

- `output_dim = 1`
- train / inference 同路径
- 每一步 rollout 前可选 dealias
- 同时统一脚本、日志和评估口径

这是后续所有 “strict data-free 0.1s model” 结论的基础。

### P0：把 eval 协议拆成 `val` 和 `test`

至少做到：

- `val`: 用于 early stopping / best checkpoint
- `test`: 只做最终汇报

否则后续所有超参数结论都会掺进 test leakage。

### P0：把物理配置显式从实验配置传到 `FiveFieldMHD`

不要再依赖“当前默认值刚好匹配当前数据”这种隐含假设。

### P1：显式实现 field normalization / loss balancing

建议至少做一个：

1. 输入按每场 RMS/std 标准化
2. physics loss 按每场 RHS 标度归一化
3. supervised loss 做 per-field weighting

这对于 `psi` 尺度远大于其他场的问题非常重要。

### P1：把 inference 里的 dealias / boundary handling 固化

现在训练阶段已经在强调 anti-aliasing，但 rollout 阶段还没有同等级处理。  
建议加一个统一 wrapper：

- `current -> dealias -> model -> boundary clamp -> next`

### P1：把 physics parameter conditioning 做实，或者先去掉

当前 `n_params`/MoE gating 基本没真正用起来。  
要么：

- 真正把 physics param 从数据/config 传入；

要么：

- 先做一个 `K=1` 的简化对照，确认复杂 gating 是否真有价值。

### P1：清理多卡采样协议与时间索引逻辑

至少补齐：

- offline 数据的 `DistributedSampler`
- global batch / total seen samples 的统一统计
- 所有数据入口统一遵循 `metadata['time_start']`

## 9.2 生成器层面

### P0：不要再把“独立 per-field GRF”当成 strict data-free 主方案

它目前最多只能作为 baseline。

### P1：优先尝试下面三类更强先验

#### 方案 A：联合谱协方差 GRF

不是每个 field 各生各的，而是在 Fourier space 里为每个 `k_y` 设一个 `5×5` 联合协方差。

优点：

- 仍然是解析/可控先验
- 比独立 GRF 只多一步联合采样
- 可显式控制 cross-field correlation

#### 方案 B：线性本征模先验

这是我认为最有科研价值的一条线。

做法：

1. 对 5-field MHD 在给定参数下做线性化
2. 求主要不稳定/弱阻尼本征模
3. 随机采样模态系数
4. 组合成初始状态

这个先验比独立 GRF 强得多，因为它天然带有：

- 跨场相位关系
- 正确的相对幅值
- 与真实线性动力学一致的结构

#### 方案 C：solver-assisted short warmup

如果你接受“无离线数据，但允许在线调用 PDE solver”这个定义，那么最强的实用路线是：

- 从解析先验/GRF 出发
- 用真 PDE solver 演化一小段时间
- 再把得到的状态作为训练输入

这不是 strict solver-free，但非常可能是 strict trajectory-dataset-free 的可行桥梁。

## 9.3 实验层面

### 第一优先级实验：锁定可复现 baseline

我建议优先跑下面这 5 组，而且要先保证监督时间跨度已经对齐：

1. `offline + physics-only`
2. `offline + supervised`
3. `offline + combined`
4. `one-step shared path + offline physics-only`
5. `one-step shared path + offline supervised / combined`

目标不是立刻 data-free，而是先把：

- 最稳定 baseline
- 最干净的训练/推理协议

锁住。

### 第二优先级实验：严格回答“更好的随机先验能不能替代真实状态锚点”

建议对比：

1. 独立 GRF
2. 数据尺度标定 GRF
3. 联合谱协方差 GRF
4. 线性本征模先验
5. 上述先验 + 短 solver warmup

### 第三优先级实验：只在 teacher 足够强时再试 self-training

不要再按固定 `step` 激活。建议改成：

- 当 `val MHD rollout L2` 低于阈值时才允许激活
- 激活时同步降低学习率
- 激活后用单独 scheduler

## 9.4 理论层面

我建议你后续认真做三个推导/思考题。

### 理论问题 1：local residual minimization 为什么不能保证 long rollout 正确？

建议你从分布视角写出来：

- 真实 attractor 分布记为 `μ`
- 当前训练分布记为 `q`
- 当前优化的是 `E_{x~q}[local PDE consistency]`

关键结论要证明或论证：

- 当 `q` 远离 `μ` 时，局部 residual 小不代表在 `μ` 上 rollout 好

这会把很多经验现象变成一个干净的理论框架。

### 理论问题 2：严格 data-free 需要约束的不是单个状态，而是“状态分布”

也就是说：

- 不是找一个 random initializer 就够
- 而是找一个能够覆盖正确 attractor 邻域的 prior `q`

这可以把研究问题转化成：

- 如何构造一个 physics-consistent state prior

### 理论问题 3：什么是 five-field MHD 的最低充分统计先验？

这是非常值得和老师讨论的问题。  
比如最小集合可能包括：

- per-field RMS
- per-field spectrum
- cross-field covariance
- cross-phase / coherence
- radial envelope
- 主导线性模态结构

这会直接指导你生成器设计。

---

## 10. 文献调研建议

下面这些文献/方向不是要一口气全吃完，而是建议你围绕“data-free + flow map + manifold”来读。

### A. Physics-informed operator learning

1. **PINO: Physics-Informed Neural Operator for Learning Partial Differential Equations**  
   ICLR 2022  
   价值：看 operator learning 如何把 physics constraint 和 data constraint 结合起来。

2. **Self-Supervised Neural Operator**  
   近期 arXiv 方向  
   价值：看没有显式标注时，如何构造 self-supervised / physics-informed operator 训练。

### B. Train / inference mismatch 与多步一致性

3. **Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks**  
   Bengio et al., NeurIPS 2015  
   价值：虽然不是 SciML，但 exposure bias 的逻辑是完全相通的。

4. 一致性训练 / flow-map consistency 相关工作  
   价值：为“单步训练如何服务长期 rollout”提供新思路。

### C. 更物理的状态生成先验

5. **From Zero to Turbulence: Generative Modeling for 3D Flow Simulation**  
   ICLR 2024  
   价值：从“直接学习湍流流形”角度启发你替代独立 GRF。

6. **CoNFiLD / Conditional Neural Field Latent Diffusion for Turbulence**  
   Nature Communications 2024  
   价值：看如何学习更接近物理 manifold 的 turbulence state distribution。

### 读文献时建议重点看什么

- 它们如何定义“没有数据”或“弱监督”
- 它们如何处理长期 rollout 稳定性
- 它们如何构造 state prior / manifold prior
- 它们如何避免 local objective 和 long-horizon objective 脱节

---

## 11. 建议和老师/专家讨论的具体问题

我建议你下一次讨论不要只问“这个模型为什么不 work”，而是问下面这些更尖锐的问题：

1. 对 five-field MHD，什么样的随机先验才算“物理上合理”？
2. 线性化本征模是否足以作为 strict data-free 初值先验的主骨架？
3. 如果允许在线调用 solver 做短 warmup，这在你的研究叙事里算不算 data-free？
4. 对于 attractor 上的 surrogate 学习，决定成败的是 local tangent 还是 invariant measure？
5. 是否应该把问题表述为“学习流形 + 学习流映射”，而不是只学 operator？
6. 哪些物理统计量可以作为 data-free 训练时的 distribution-level regularizer？
7. 当前 five-field 场之间最关键的联合统计量是什么？ cross-spectrum? coherence? phase lag?

---

## 12. 我建议你现在就采取的行动顺序

### 第 1 步：先统一研究口径

把当前工作明确分成三条线：

- `offline physics-only`: label-free baseline
- `offline supervised / combined`: 需要先确认监督时间尺度对齐后再谈强弱
- `strict data-free online`: 真正要攻克的问题

不要再把它们混成一个结论。

### 第 2 步：先把监督时间尺度、one-step 主线和 eval protocol 清理干净

否则后面所有新实验都不够干净。

### 第 3 步：停止把 raw GRF 当主力方向

raw GRF 可以保留，但它应该是：

- baseline
- ablation

而不是主力。

### 第 4 步：把“更物理的状态生成器”作为主问题

我认为最值得投入的顺序是：

1. 联合谱协方差先验
2. 线性本征模先验
3. 短 solver warmup
4. 之后再考虑 learnable generator / diffusion prior

### 第 5 步：把理论问题写成一个清晰的研究备忘

题目可以类似：

> 为什么在 five-field MHD 中，局部 PDE residual 一致性不足以从 off-manifold 随机态中学出长期稳定的 surrogate dynamics？

这会让你后续所有代码与实验都更聚焦。

---

## 13. 最终判断

如果只用一句话概括我对当前阶段的判断，那就是：

> 你现在已经证明了“在单轨、强锚定的真实轨迹 overfit 场景下，OmniFluids 架构 + MHD RHS loss 是可学习的”，但还没有解决“在清理训练协议后，怎样在严格 data-free 条件下构造足够物理且 rollout 稳定的训练状态分布”这个核心科学难题。

所以接下来最值得做的，不是继续在现有 raw GRF 路线上堆更多训练 trick，而是：

1. 清理训练/推理/评估协议
2. 固化强 baseline
3. 把主要研究火力转向 **物理流形先验 / 状态分布生成器**

这才是把当前阶段推进到真正科学问题的关键一步。
