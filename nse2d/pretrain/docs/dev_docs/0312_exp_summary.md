# 实验与问题汇总报告 (2026-03-12)

> 本报告汇总了 OmniFluids nse2d/pretrain 项目的所有关键问题、观察、结论和实验状态。

---

## 一、项目根问题与目标

### 根问题
针对复杂 PDE 动力过程，能否**尽量不依赖真实仿真数据**，只靠物理约束与随机生成训练态，训练出可用且尽量长期稳定的 data-free surrogate model。

### 当前具体场景
2D 5-field MHD surrogate training

### 验收指标
- **主指标**：MHD 与 GRF 测试集在 step-1/3/5/10 的 per-channel mean relative L2
- **现象指标**：不再长期出现"train loss 与单步变好，但多步 rollout 变坏"
- **协议指标**：train/inference 路径一致、监督时间尺度对齐、val/test 角色拆分

---

## 二、主问题总表（Q01-Q06）

| 主问题ID | 类型 | 主问题 | 当前状态 | 当前结论概要 |
|----------|------|--------|----------|--------------|
| **Q01** | 问题分析 | 相关参考文献、已有方法、主要挑战是什么 | 进行中 | 已形成 DeepResearch prompt，缺系统性外部文献证据 |
| **Q02** | 问题分析 | strict data-free 的价值与可行性；为什么 one-step 不代表 rollout | 进行中 | Level B/C 可行，strict Level D 未证成；主矛盾是长期 rollout 稳定 |
| **Q03** | 方法路线 | 纯用随机产生的物理场（如 GRF），只用 PDE loss 训练 | 进行中 | fixed GRF + PDE-only 不可用；PDE loss 本身不是学不会 |
| **Q04** | 方法路线 | 通过可学习 GRF 端到端优化输入分布 | 进行中 | 低维 per-field learnable GRF 不是主线；exp20/21 未救回长期 rollout |
| **Q05** | 方法路线（主攻） | 通过 bootstrap / self-training 逐渐学好仿真模型 | 进行中 | **当前最强纯 online 路线**；已回答多个关键子问题 |
| **Q06** | 方法路线 | 通过生成模型用少量数据学习分布 | 提出待分析 | **边界路线/fallback**；当前无直接实验支持；不建议直接上 diffusion/flow |

---

## 三、主要结论与观察（跨问题）

| 编号 | 结论/观察 | 关联主问题 |
|------|-----------|------------|
| **OBS01** | nse2d/pretrain 当前真实任务已是 2D 5-field MHD surrogate training | Q01, Q02 |
| **OBS02** | 当前最好经验路线仍是"GRF 预训练 + simulation data 继续训练" | Q02, Q03 |
| **OBS03** | raw online 强化线常见现象：train loss 和单步变好，但多步 rollout 变差 | Q02, Q04, Q05 |
| **OBS04** | learnable GRF、self-training、soft-Linf、multi-step 能改善局部指标，但未根本解决长期 rollout | Q04, Q05 |
| **OBS05** | 当前独立 per-field GRF 缺少五场联合结构，很可能不够物理 | Q01, Q03-Q05 |
| **OBS06** | **Q05 bootstrap 路线关键结论**：soft-Linf 是必要条件；multi-step 有独立净收益；best-teacher 当前不 work；rollout curriculum 有潜力；ensemble smoothing 有潜力但强度不宜过大 | Q05 |

---

## 四、Q05 Bootstrap 路线（当前主攻）详细结论

### 4.1 已回答的关键问题

| 问题 | 结论 | 关键证据 |
|------|------|----------|
| **soft-Linf 是否关键？** | **是 pre-bootstrap 稳定性的必要条件** | exp31 (no-soft-Linf) 灾难性爆炸：MHD `1.66e8`，GRF `1.60e8` |
| **multi-step PDE 是否有独立净收益？** | **是** | exp32 (no-multi-step) MHD `11.58` 略差于 exp22 `11.54` |
| **dynamic vs fixed relative loss？** | **dynamic-scale 已判负，fixed-scale 有潜力** | exp24 (dynamic) MHD `20.54` 差；exp26 (fixed) early window `1.23` 极佳 |
| **best-teacher vs latest-teacher？** | **best-teacher 当前不 work** | exp29 (best) MHD `54.43` 明显差于 exp22 (latest) `11.54` |
| **rollout curriculum 是否有潜力？** | **是** | exp30 MHD `8.89`、GRF `10.30`，GRF 略优于 exp22 |
| **ensemble RHS smoothing 是否有潜力？** | **是，但强度不宜过大** | exp33 (E=5) GRF `8.32` 优于 exp22；exp34 (E=10) MHD `15.85` 说明过强不好 |

### 4.2 实验结果排名（By MHD Mean）

| 排名 | 实验 | MHD Mean | GRF Mean | 状态/备注 |
|------|------|----------|----------|-----------|
| 1 | mhd5_staged_v1 | **0.2903** | 39.16 | 经验最强路线 |
| 2 | exp6 (GRF staged) | **0.3941** | 2.327 | |
| 3 | exp3 (physics+supervised) | **0.3991** | 1.191 | |
| 4 | exp_offline_pde (1GPU) | **0.4097** | 1.060 | strict data-free 参照 |
| 5 | exp1b (sup dt=1s, 1GPU) | **0.4099** | 1.030 | 监督基线 |
| 15 | exp26 (fixed-scale) | **1.23** | 1.44 | early signal (step-2400) |
| 16 | exp30 (curriculum) | **8.89** | 10.30 | 有潜力 |
| 17 | exp33 (ensemble E=5) | **9.66** | **8.32** | 有潜力 |
| 18 | exp23 (later-start) | **10.36** | 10.32 | pre-bootstrap，待激活 |
| 20 | **exp22** (best pure-online) | **11.54** | 11.51 | **当前最强 pure-online** |
| 21 | exp32 (no-multi-step) | **11.58** | 11.22 | multi-step 有净收益 |
| 24 | exp24 (dynamic) | **20.54** | 21.12 | 已判负 |
| 28 | exp29 (best-teacher) | **54.43** | 62.94 | 表现不佳 |
| 30 | exp31 (no-soft-Linf) | **1.66e8** | 1.60e8 | **failed** |

### 4.3 待回答的关键问题

- **exp23**: 晚启动 (start=100000) 是否更稳？待激活后验证
- **exp27**: rollout_dt=1.0 是否比 0.1 更稳？等待首批 eval
- **exp28**: no-detach 是否比 detach 更稳？等待首批 eval
- **batch_size**: 是否污染当前结论？需 clean ablation

---

## 五、Q06 少量数据生成模型路线（边界路线）

### 5.1 定位与状态

| 字段 | 内容 |
|------|------|
| 当前定位 | **边界路线 / fallback**，不是当前主攻线 |
| 当前目标 | 先收敛口径边界、关键 gating 问题与 future probe 设计 |
| 证据状态 | **当前无 Q06 专属实现、训练脚本、结果表；当前无直接实验支持** |

### 5.2 关键结论（OBS-Q06）

| 编号 | 结论 |
|------|------|
| OBS-Q06-1 | Q06 已被定位为"有研究价值，但更像边界路线/fallback" |
| OBS-Q06-2 | 仓库内当前没有 Q06 专属实现、训练脚本、结果表 |
| OBS-Q06-3 | 不能把 Q04/Q05 的结果迁移成 Q06 证据 |
| OBS-Q06-4 | 代码里的 generator 目前仍是 GRF / ModelEvolvedGRFGenerator，不是 small-data generative model |
| OBS-Q06-5 | 如果未来重启 Q06，第一步应先收敛口径、学习对象、数据预算和评估协议 |

### 5.3 首批 Gating 问题（待回答）

| 问题 | 为什么必须先答 |
|------|----------------|
| 这条路线算 strict data-free、weakly data-free，还是 data-informed prior？ | 不先定口径，后续所有讨论都会漂移 |
| 要学习的对象是什么：初始状态分布、短 rollout manifold，还是 latent dynamics prior？ | 直接决定模型选型、数据组织和评估方式 |
| 最小真实数据预算是多少？ | 没有数据预算就无法判断是否值得 |
| 第一版成功标准是什么？ | 如果不能带来 downstream 净收益，样本"看起来更像"不构成价值 |
| 应先做低成本 probe 还是直接上 diffusion/flow？ | 避免在证据极弱时过早投入重模型 |

---

## 六、当前运行实验状态

| 实验 | 所属主问题 | 当前状态 | 关键结果 |
|------|------------|----------|----------|
| exp22 | Q05 | running | MHD 11.54, GRF 11.51；**当前最强 pure-online** |
| exp23 | Q05 | running | MHD 10.36, GRF 10.32；pre-bootstrap，待激活 |
| exp27 | Q05 | running | step_0 已落盘；dt=1.0 对齐测试 |
| exp28 | Q05 | running | step_0 已落盘；no-detach 测试 |
| exp29 | Q05 | running | MHD 54.43, GRF 62.94；best-teacher 表现不佳 |
| exp30 | Q05 | running | MHD 8.89, GRF 10.30；curriculum 有潜力 |
| exp32 | Q05 | running | MHD 11.58, GRF 11.22；略差于 exp22 |
| exp33 | Q05 | running | MHD 9.66, GRF 8.32；ensemble 有潜力 |
| exp24 | Q05 | stopped | dynamic-scale 已判负 |
| exp26 | Q05 | stopped | fixed-scale early signal (MHD 1.23)，手动停止 |
| exp31 | Q05 | stopped | no-soft-Linf 灾难性失败 |

---

## 七、已完成的重要结论

| 条目ID | 结论 |
|--------|------|
| DONE01 | nse2d/pretrain 当前真实任务已是 2D 5-field MHD surrogate training |
| DONE02 | 当前最好经验路线仍是"GRF 预训练 + simulation data 继续训练" |
| DONE03 | 当前主问题应表述为"更符合物理的随机态生成与彻底 data-free 训练" |
| Q06-D1 | Q06 已从旧 ROOT 混写语境中拆出，形成独立问题目录 |
| Q06-D2 | Q06 的项目定位已收敛为边界路线/fallback |
| Q06-D3 | 已完成一轮仓库内直接证据清点，当前无 Q06 专属实现或实验支持 |
| Q05-D4 | **soft-Linf 是 pre-bootstrap 稳定性的必要条件** |
| Q05-D5 | **multi-step PDE 有独立净收益** |
| Q05-D6 | **best-teacher source 当前不 work** |
| Q05-D7 | **dynamic-scale relative loss 已判负** |
| Q05-D8 | **rollout curriculum 有潜力** |
| Q05-D9 | **ensemble RHS smoothing 有潜力** |

---

## 八、下一步优先级

### 最高优先级
1. **盯 exp23 激活窗口** (step 100000 前后)
2. **观察 exp27/exp28 首批 eval** (dt=1.0 和 no-detach)

### 高优先级
3. 补 clean batch_size ablation
4. 确定 optimal ensemble size

### 中优先级
5. Q01 DeepResearch 外部文献
6. Q06 gating 问题收敛

---

*Generated: 2026-03-12*
*Source: main.md, Q05_bootstrap自举路线/sub_main.md, Q06_少量数据生成模型路线/*
