# Q02 第一性原理与可行性判断 sub_main

> 角色：这是 `Q02` 的二级 `main` 控制面。它只管理“Q02 当前要回答什么、回答到哪、下一步推进什么”；问题定义、边界与压缩结论看 `README.md`，长证据继续下沉到 `实验记录.md`、`深度思考.md`、`理论推导.md`。

## 0. 控制面摘要

| 字段 | 内容 |
|------|------|
| 当前主问题 | 从第一性原理回答：“物理随机态生成与彻底 data-free 训练”为什么值得做、在当前项目约束下是否可行、主要挑战是什么，以及后续应优先验证哪些机制与证据 |
| 当前定位 | 给 `Q03-Q06` 提供“为什么值得做 / 当前是否可行 / 当前主瓶颈是什么”的上游判断，不替代各路线自己的实验结论 |
| 当前状态 | 进行中 |
| 当前工作重心 | 锁住单卡 anchored baseline 参照；用 recent online 结果继续解释“为什么真实状态锚点关键”“为什么 one-step 变好不等于 rollout 变好” |
| 当前主阻塞 | strict `data-free` 与 `Level B/C/D` 的正式定义仍未定；随机态先验缺口与五场 joint structure 是否为主因仍缺直接证据；协议污染尚未完全剥离 |
| 细节入口 | `README.md`、`实验记录.md`、`深度思考.md`、`理论推导.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 已有结论和观察

| 编号 | 结论或观察 | 当前归类 | 证据入口 |
|------|------------|----------|----------|
| `OBS-Q02-01` | 这个问题有明确研究价值，因为它直接关系到复杂 PDE surrogate 训练能否显著减少对真实仿真轨迹的依赖 | 已确认 | `README.md`、`深度思考.md` |
| `OBS-Q02-02` | 按当前项目内口径，整体更接近 `Level B/C`，而 strict `Level D` 仍未证成；但这些层级的正式定义仍待补 | 已确认（定义待补） | `README.md`、`深度思考.md`、`理论推导.md` |
| `OBS-Q02-03` | anchored baseline 仍是当前最可信的参照系；`exp_offline_pde (1GPU)` 与 `exp1b (sup dt=1s, 1GPU)` 说明 anchored support 上的 physics-only / supervised 都在可用区间 | 已确认 | `README.md`、`实验记录.md` |
| `OBS-Q02-04` | 多个 raw online 结果共同说明：`train loss` 和 `step-1` 可以继续变好，但多步 rollout 仍可能系统性恶化 | 已确认 | `README.md`、`实验记录.md`、`理论推导.md` |
| `OBS-Q02-05` | 当前主矛盾更像长期 rollout 稳定性与训练分布支持问题，而不只是 `PDE loss` 还不够强 | 部分已确认 | `README.md`、`深度思考.md`、`理论推导.md`、`实验记录.md` |
| `OBS-Q02-06` | 五场 joint structure 缺失、最小充分先验未定义，仍是高价值缺口，但当前还不能写成正式结论 | 待补证 | `README.md`、`深度思考.md`、`理论推导.md` |

## 2. 关键子问题拆解

| 子问题ID | 核心问题 | 主要问题类型 | 当前状态 | 当前判断 | 细节入口 |
|----------|----------|--------------|----------|----------|----------|
| `Q02-S1` | strict `data-free` 为什么值得做，当前可行性层级到底到哪里 | 深度思考 / 理论推导 / Agent讨论 | 待验收 | 已有稳定口径，但正式定义仍缺，暂不能写死边界 | `README.md`、`深度思考.md`、`理论推导.md`、`sub_main.md` |
| `Q02-S2` | 为什么真实状态锚点当前仍关键 | 深度思考 / 实验验证 / 理论推导 | 进行中 | anchored baseline 仍是参照系，但 clean single-GPU 重跑仍在完成中 | `README.md`、`实验记录.md`、`理论推导.md` |
| `Q02-S3` | 为什么 `train loss` 与 one-step 变好不等于 multi-step rollout 变好 | 实验验证 / 理论推导 / 深度思考 | 已完成 | 现象与推导已经形成稳定解释骨架 | `理论推导.md`、`实验记录.md`、`深度思考.md` |
| `Q02-S4` | 当前主矛盾到底更像 `q -> mu` 分布错配、protocol mismatch，还是二者叠加 | 深度思考 / 实验验证 / 理论推导 / Agent讨论 | 待研究 | 已有工作假设，但还缺 clean 分离证据 | `深度思考.md`、`理论推导.md`、`实验记录.md` |
| `Q02-S5` | 最小充分先验应包含什么，五场 joint structure 缺失是否是当前随机态先验的核心缺口 | 调研 / 深度思考 / 理论推导 / 老师/同事讨论 | 提出待分析 | 当前只够写成高价值假设，不够写成正式结论 | `README.md`、`深度思考.md`、`理论推导.md` |
| `Q02-S6` | 如何维护 Q02 的文件角色、阅读路径与回答推进顺序 | Agent讨论 | 进行中 | `README.md` 管摘要入口、`sub_main.md` 管回答推进的边界已经基本稳定 | `README.md`、`sub_main.md` |

## 3. 提出待分析

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 当前说明 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q02-A1` | 辅 | 主 | 辅 | 辅 | 辅 | 辅 | 提出待分析 | 五场 joint structure 缺失、`field-specific` 约束和随机态先验不足目前都只是高价值假设；现有文档只支持“应重点怀疑”，还不支持写成定论 | `深度思考.md`、`理论推导.md` |
| `Q02-A2` | 辅 | 主 | - | 主 | 主 | 辅 | 提出待分析 | strict `data-free` 的口径边界、`Level B/C/D` 的正式定义以及“最小充分先验”仍未定稿；当前只能保留占位，不应提前写死 | `README.md`、`理论推导.md` |

## 4. 待研究

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 验收指标 | 当前准备 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|----------|
| `Q02-R1` | - | 辅 | 主 | 辅 | 辅 | - | 待研究 | `RUN-OFF-PDE-1GPU` 与 `RUN-SUP-1GPU` 跑完后仍与历史单卡 anchored baselines 保持同量级，并足够干净地支持 Q02 作为 reference 使用 | 当前已有历史单卡 anchored baseline 与单卡重跑的最新 checkpoint，可继续做一致性对照 | `实验记录.md`、`README.md` |
| `Q02-R2` | - | 辅 | 主 | 辅 | 辅 | - | 待研究 | 形成一版按 `field / timestep` 归档的 rollout 爆炸 pattern，至少能更干净地支撑“为什么 one-step 不等于 rollout” | `README.md`、`实验记录.md`、`理论推导.md` 都已把这项工作列为后续证据缺口 | `实验记录.md`、`理论推导.md` |

## 5. 进行中

> 主体维护区。这里优先更新正在推进的回答、baseline / 对照实验入口和最近一轮计划；长分析继续下沉到 `实验记录.md`、`深度思考.md`、`理论推导.md`。

### 5.1 当前进行中的关键子问题

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 当前动作 | 当前可见结论 | 验收信号 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|--------------|----------|----------|
| `Q02-I1` | - | 主 | 主 | 辅 | 辅 | - | 进行中 | 继续用单卡 anchored baselines 与 staged historical reference 维持“为什么真实状态锚点关键”的干净参照系 | 当前已经可以写成：anchored support 仍关键，而且 physics-only objective 不是完全学不会 | `RUN-OFF-PDE-1GPU` 与 `RUN-SUP-1GPU` 跑完后仍与历史 single-GPU anchored baseline 接近 | `README.md`、`实验记录.md`、`理论推导.md` |
| `Q02-I2` | - | 主 | 主 | 主 | 辅 | - | 进行中 | 用 `exp20 / exp21 / exp22` 持续补证“局部收益不等于长期 rollout 收益”，并准备按 `field / timestep` 归档爆炸 pattern | 当前已经可见：pure online 路线能改善局部指标，但仍远离 anchored baseline 区间 | 至少形成一版按 `field / timestep` 的 pattern 归档，并能更干净地区分局部改善与 rollout 恶化 | `实验记录.md`、`理论推导.md`、`深度思考.md` |
| `Q02-I3` | 辅 | 主 | 辅 | 主 | 主 | 辅 | 进行中 | 维持 Q02 的统一表述边界：问题有价值、`Level B/C` 可行但 strict `Level D` 未证成；同时约束 `README.md` 与 `sub_main.md` 的分工 | 当前跨 `README.md`、`深度思考.md`、`理论推导.md` 的口径已经基本一致，但正式定义仍缺 | 在不新增未验证口径的前提下，把当前表述稳定到可长期复用 | `README.md`、`深度思考.md`、`理论推导.md`、`sub_main.md` |

### 5.2 Q02 baseline / 对照实验推进表

> 这里只纳入当前已经进入 Q02 证据链的 baseline 与对照。`exp23 / exp26 / exp27 / exp28` 目前仍以 `Q05` 为主，且在 Q02 语境下证据仍不足，后续若形成可复用机制证据再回流。

| run_id | 实验角色 | 对应子问题 | code git commit | 实验目的 | 实现改动 / 脚本 | 实验机器（tmux session） | 结果文件位置 | 预期结果和验收条件 | 当前可见结果 | 当前分析 | 下一步计划 | 当前状态 |
|--------|----------|------------|-----------------|----------|-----------------|--------------------------|--------------|--------------------|--------------|----------|------------|----------|
| `RUN-OFF-PDE-1GPU` | baseline（anchored，physics-only） | `Q02-S2`、`Q02-R1` | 未记录 | 单卡 physics-only anchored baseline 重跑，给 Q02 一个更干净的 label-free 参照 | `run_exp_offline_pde.sh` | `20260308221016 : t1` | `results/exp_offline_pde_only_single_gpu/4ac94d17-03_10_13_46_34-K4-mx128-w80-L12-od10/` | MHD / GRF 都进入稳定可比区间，并可和历史 `exp_offline_pde (1GPU)` 直接对照 | 最新可见 `step_25000`：MHD `0.3806`，GRF `1.0450` | 当前仍处在可用 baseline 区间，趋势合理 | 跑完后与历史 `exp_offline_pde (1GPU)` 做一致性对照 | running |
| `RUN-SUP-1GPU` | baseline（anchored，supervised） | `Q02-S2`、`Q02-R1` | 未记录 | 单卡 supervised anchor 重跑，给 Q02 一个更干净的真实状态参照 | `run_exp1_pure_supervised.sh` | `20260308221016 : t2` | `results/exp1b_pure_supervised_dt1s_signle_gpu/b883fcea-03_10_13_46_34-K4-mx128-w80-L12-od10/` | 结果进入历史 `exp1b (1GPU)` 附近，并和 `RUN-OFF-PDE-1GPU` 直接可比 | 最新可见 `step_32500`：MHD `0.4094`，GRF `1.0306` | 与历史 `exp1b (1GPU)` 很接近，说明监督 baseline 稳定 | 跑完后与 `RUN-OFF-PDE-1GPU` 做直接对照 | running |
| `RUN-EXP20` | 对照（learnable prior） | `Q02-S4`、`Q02-R2` | 未记录 | 检验 learnable GRF + `soft-Linf` 是否至少能把 raw online 拉回更可用的区间 | `learnable GRF` + `soft-Linf` | `a100_dev : t（已切走）` | `results/exp20_learnable_grf_10k_soft_linf/36f2ca4a-03_10_03_13_58-K4-mx128-w80-L12-od10/` | 至少 `step-10` / `Mean` 明显优于 `exp14` | 当前汇总结果：MHD Mean `22.21`，GRF Mean `22.67` | 比 `exp14` 好很多，但离 anchored baselines 仍很远 | 已手动停止，把 `a100_dev : t` 让给 `RUN-EXP28`；后续与 `RUN-EXP21` 配对比较 multi-step 的净作用 | stopped |
| `RUN-EXP21` | 对照（multi-step ablation） | `Q02-S4`、`Q02-R2` | 未记录 | 检验 multi-step PDE 是否能救 learnable GRF 的长期 rollout | `RUN-EXP20` + `multi_step_pde_n=3` | `H800 : t1（已切走）` | `results/exp21_learnable_grf_10k_soft_linf_multistep/3420133e-03_10_09_11_53-K4-mx128-w80-L12-od10/` | `step-10` / `Mean` 优于 `RUN-EXP20` | 当前汇总结果：MHD Mean `25.72`，GRF Mean `27.49` | 目前看 multi-step 没有救这条路线，反而可能更差 | 已在约 `step_69200` 停止，把 `H800 : t1` 让给 `RUN-EXP26`；后续重点做 `exp20` vs `exp21` 的 single / multi-step 对照 | stopped |
| `RUN-EXP22` | 对照（bootstrap / pure online 候选） | `Q02-S4`、`Q02-R2` | 未记录 | 检验 self-training + `soft-Linf` + multi-step 是否把 pure online 路线显著拉回 | `self-training` + `soft-Linf` + `multi_step_pde_n=3` | `H800 : t1-2` | `results/exp22_self_training_linf_multistep/acad9e03-03_10_09_26_12-K4-mx128-w80-L12-od10/` | 相比 `exp18` 明显改善，并尽量接近 anchored baselines | 当前汇总结果：MHD Mean `11.11`，GRF Mean `9.614` | 明显优于旧 pure online 样本，但仍远差于 anchored baselines | 继续判断改善是否只停留在局部或中短期，并整理 rollout 爆炸 pattern | running |

### 5.3 当前非实验推进动作

| 类型 | 当前动作 | 目标 | 指向文档 |
|------|----------|------|----------|
| 调研 | 当前不单开 `Q02` 专属调研文档；若需要外部 literature / benchmark 证据，仍先回上游 `Q01` 收敛后再回流 `Q02` | 保持模板角色清晰，避免在 `Q02` 里混写上游调研 | `README.md` |
| 深度思考 | 持续约束“问题有价值”“为什么 anchor 关键”“为什么不能把局部收益写成 strict data-free 成立”这三层表述不要混写 | 维持一版稳定的第一性原理判断框架 | `深度思考.md` |
| 理论推导 | 持续用 `q -> mu` 分布支持与误差传播框架解释 `one-step != rollout`，并保留失效条件与未证明点 | 给 Q02 提供形式化、可回链的解释骨架 | `理论推导.md` |
| Agent讨论 | 用 `sub_main.md` 管理子问题、状态迁移和当前运行实验入口 | 让 Q02 有稳定的二级控制面，而不是把进度散落在多个过程文档里 | `sub_main.md` |
| 老师 / 同事讨论 | 当前暂未正式启动；如果 clean baseline、pattern 归档和当前解释仍不足以回答“最小充分先验是什么”，再升级为正式讨论问题 | 不在证据不足时过早发散到外部讨论 | `sub_main.md` |

## 6. 待验收

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 验收标准 | 计划来源 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q02-C1` | - | 主 | 辅 | 主 | 主 | 辅 | 待验收 | 当前可以稳定写成统一口径：问题有价值；`Level B/C` 已有一定支撑；strict `Level D` 未证成；Q02 的主矛盾更像训练支持、协议一致性与长期 rollout 稳定性 | `README.md`、`深度思考.md`、`理论推导.md`、`sub_main.md` |

## 7. 已完成

| 条目ID | 结论 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 论据概述 | 细节位置 |
|--------|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q02-D1` | `train loss` 与 one-step 指标改善，不等于 multi-step rollout 稳定 | - | 辅 | 主 | 主 | 辅 | - | 已完成 | 多个 online run 的现象与误差传播推导一致支持这一点 | `理论推导.md`、`实验记录.md`、`深度思考.md` |
| `Q02-D2` | anchored support 上的 physics-only / supervised 都在可用区间，因此 Q02 不应把“physics objective 不可学”当作主结论 | - | 辅 | 主 | 辅 | 辅 | - | 已完成 | 单卡历史 baseline 与当前重跑都在稳定可比区间，支持“问题更像 support / stability 问题” | `实验记录.md`、`README.md` |
| `Q02-D3` | Q02 当前启用 `README.md`、`sub_main.md`、`实验记录.md`、`深度思考.md`、`理论推导.md`；不单开 `调研记录.md` 与 `交流讨论.md` | - | - | - | - | 主 | - | 已完成 | 当前文件角色已按模板触发器收敛：总览、控制面、实验、思考、推导分别归位 | `README.md`、`sub_main.md` |

## 8. 文档分工

| 文件 | 当前职责 |
|------|----------|
| `README.md` | 作为入口页，压缩定义问题、边界、当前结论和优先阅读路径 |
| `sub_main.md` | 作为二级 `main` 控制面，管理子问题拆解、状态桶、baseline / 对照实验入口和回答推进顺序 |
| `实验记录.md` | 记录支撑 Q02 判断的 baseline、对照实验、异常现象与结果影响 |
| `深度思考.md` | 沉淀第一性原理分析、路线比较、反例攻击和当前方法投入判断 |
| `理论推导.md` | 沉淀 `q -> mu` 分布支持、误差传播、边界条件和未证明点 |
| `调研记录.md` | 当前未启用；若需要外部文献、benchmark 或 baseline 证据，先回上游 `Q01` 收敛 |
| `交流讨论.md` | 当前未启用；只有讨论本身改变 Q02 判断时才应创建并回链 |
