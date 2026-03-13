# Q04 可学习 GRF 路线 sub_main

> 作用：这是 `Q04` 的二级 `main` 控制面，只管理“这个主问题当前答到哪里、接下来怎么推进”。路线摘要看 `README.md`；实验细节看 `实验记录.md`；路线解释与反例攻击看 `深度思考.md`。

## 0. 控制面摘要

| 字段 | 内容 |
|------|------|
| 当前主问题 | learnable GRF 能否通过端到端优化输入分布，把随机态逐步推近更有训练价值的区域，并改善 5-field MHD surrogate 的长期 rollout |
| 当前定位 | 现阶段更适合作为 bridge / baseline；是否值得继续升格为主线仍待补证 |
| 当前最小证据链 | `exp14 -> exp20 -> exp21` |
| 当前工作重心 | 先拆清“learnable prior 收益”与“loss trick 收益”，基于 `exp20 / exp21` 现有证据做 paired analysis，再决定是否继续投入更强 prior |
| 当前主阻塞 | 缺 matched fixed-vs-learnable 对照；当前 prior 仍是 low-dimensional per-field 参数化；长期 rollout 还没有被救回 |
| 先读哪些文件 | `README.md` -> `实验记录.md` -> `深度思考.md` -> `../main.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 已有结论和观察

- 已确认：
  - `exp14` 说明 raw learnable GRF 是高价值失败，“可学习”本身不是充分条件。
  - `exp20` 把 Q04 从灾难性失稳拉回可分析区间，说明 learnable prior 配合稳定化手段有局部价值。
  - `exp21` 只改善中短期，不改善 `step-10` / `Mean`，当前证据不支持 multi-step PDE 能救回长期 rollout。
  - anchored baseline 仍显著优于 Q04 当前 best，因此 Q04 还不能被判成 strict data-free 主线。
- 待补证 / 未定论：
  - `exp20` 的改善到底有多少来自 learnable prior，本质上是否主要仍是 `soft-Linf` 在稳训练。
  - 当前 low-dimensional per-field learnable GRF 是否已经触顶，还是只是参数化还不够强。
  - learnable GRF 是否真的在改善 `q -> \mu`，还是只在错误支持集上做局部适配。
- 详细依据：
  - 结果与 run 细节：`实验记录.md`
  - 路线解释、反例攻击与下一步判断：`深度思考.md`
  - Q04 在总问题树中的位置：`../main.md`

## 2. 关键子问题拆解

| 子问题ID | 核心问题 | 这题为什么关键 | 当前状态 | 主落点文档 |
|----------|----------|----------------|----------|------------|
| `Q04-S1` | `exp20` 的收益到底来自 learnable prior 还是 `soft-Linf`？ | 决定 Q04 是否真的有独立方法价值 | 进行中 | `实验记录.md`、`深度思考.md` |
| `Q04-S2` | `multi-step PDE` 是否能救回 learnable GRF 的长期 rollout？ | 决定是否继续围绕训练 trick 投入 | 进行中 | `实验记录.md`、`深度思考.md` |
| `Q04-S3` | 当前 low-dimensional per-field prior 是否已到上限？ | 决定是否值得设计更强 joint prior | 待研究 | `深度思考.md` |
| `Q04-S4` | Q04 是否应固定为 bridge / baseline，而不是主线？ | 影响资源分配与 `../main.md` 口径 | 待验收 | `README.md`、`../main.md` |
| `Q04-S5` | 是否需要为 Q04 单独开启更强 prior 的外部调研？ | 决定是否启用 `调研记录.md` | 提出待分析 | `README.md` |
| `Q04-S6` | 是否需要单独形式化 `q -> \mu` / support mismatch 的理论解释？ | 决定是否启用 `理论推导.md` | 提出待分析 | `深度思考.md` |

## 3. 进行中

### 3.1 当前进行中的问题

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 当前动作 | 预期产物 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|----------|
| `Q04-ING-01` | - | 主 | 主 | 辅 | 辅 | - | 进行中 | 用 `exp14 / exp20 / exp21` 这条最小证据链先回答“learnable prior 是否真有独立收益”；在 matched 对照缺失前，不把 `exp20` 的改善直接记成 Q04 已成立 | 一版收益归因的阶段判断，以及最小 matched 对照需求清单 | `实验记录.md`、`深度思考.md` |
| `Q04-ING-02` | - | 主 | 主 | - | 辅 | - | 进行中 | 盯住 `exp21` 相对 `exp20` 的 `step-3 / step-5` 改善与 `step-10 / Mean` 恶化，判断 multi-step 到底在救什么、伤什么 | `exp20 vs exp21` 的 paired comparison 结论 | `实验记录.md`、`深度思考.md` |
| `Q04-ING-03` | 辅 | 主 | 辅 | - | 主 | - | 进行中 | 先把 Q04 暂定位为 bridge / baseline，并等待 matched 对照后再决定是否把这个判断同步回写到 `../main.md` 的路线优先级 | 一版可验收的路线定位结论 | `README.md`、`深度思考.md`、`../main.md` |

### 3.2 相关实验状态

| run_id | code git commit | 实验目的 | 实现改动 / 脚本 | 实验机器（tmux session） | 结果文件位置 | 预期结果和验收条件 | 当前结果 | 当前分析 | 下一步计划 | 当前状态 |
|--------|-----------------|----------|-----------------|--------------------------|--------------|--------------------|----------|----------|------------|----------|
| `RUN-EXP20` | 未记录 | 判断 learnable GRF + `soft-Linf` 是否至少能把 raw online learnable prior 从灾难性区间拉回可分析区间 | `learnable_grf_start=10000`、`soft_linf_weight=0.1`；`run_exp20_learnable_grf_10k_soft_linf.sh` | `a100_dev : t（已切走）` | `results/exp20_learnable_grf_10k_soft_linf/36f2ca4a-03_10_03_13_58-K4-mx128-w80-L12-od10/` | 至少相对 `exp14` 明显改善，且 `step-10` / `Mean` 不再处于灾难性区间 | 当前汇总结果：MHD Mean `22.21`，GRF Mean `22.67` | 已证明“比 raw learnable GRF 好很多”，但离 anchored baseline 仍很远；收益是否来自 prior 本身仍待补证 | 该 run 已手动停止，把 `a100_dev : t` 让给 `RUN-EXP28`；后续与 `RUN-EXP21` 做 paired comparison，并补 matched fixed-vs-learnable 对照 | stopped |
| `RUN-EXP21` | 未记录 | 判断在 `exp20` 基础上增加 multi-step PDE，是否能真正改善 learnable GRF 的多步稳定性 | `RUN-EXP20` + `multi_step_pde_n=3`；`run_exp21_learnable_grf_10k_soft_linf_multistep.sh` | `H800 : t1（已切走）` | `results/exp21_learnable_grf_10k_soft_linf_multistep/3420133e-03_10_09_11_53-K4-mx128-w80-L12-od10/` | `step-10` / `Mean` 优于 `RUN-EXP20`，否则不支持“multi-step 能救回 Q04” | 当前汇总结果：MHD `step-3 / step-5 / step-10 / Mean = 2.424 / 14.71 / 85.21 / 25.72`，GRF `4.736 / 21.50 / 83.29 / 27.49` | 当前只看到中程改善，远程反而更差；现阶段不支持“multi-step 救回 learnable GRF” | 该 run 已在约 `step_69200` 主动停止，把 H800 `t1` 让给 `RUN-EXP26`；后续直接基于现有结果做 paired analysis | stopped |

### 3.3 当前实验推进缺口

| 计划ID | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 最小动作 | 验收条件 | 细节位置 |
|--------|------|----------|----------|----------|-----------|---------------|------|----------|----------|----------|
| `Q04-PLAN-01` | - | 辅 | 主 | - | 辅 | - | 进行中 | 设计 matched fixed-vs-learnable 对照：相同 schedule、相同 loss、只改变 prior 是否可学习 | 若 learnable prior 在 `step-10` / `Mean` 上无明确净收益，则不再把当前 Q04 视为可升格路线 | `实验记录.md`、`深度思考.md` |
| `Q04-PLAN-02` | - | 主 | 主 | - | 辅 | - | 进行中 | 拆出 `soft-Linf` 的独立贡献，避免把稳定化收益误记到 prior 上 | 至少能回答 `exp20` 的改善主要来自哪一部分 | `实验记录.md`、`深度思考.md` |
| `Q04-PLAN-03` | - | 主 | 主 | - | - | - | 进行中 | 做 `exp20` vs `exp21` 的按 field / 按时间步 / 按频谱诊断，确认 multi-step 在当前支持集上到底改变了什么 | 能解释“中程变好但远程变差”的主要 pattern；否则维持未定论 | `实验记录.md`、`深度思考.md` |

## 4. 待研究

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 验收指标 | 当前准备 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|----------|
| `Q04-R-01` | 辅 | 主 | 主 | 辅 | 辅 | - | 待研究 | 若继续 Q04，应验证更强 joint prior 或 field-specific 约束能否在 matched 设置下明确改善 `step-10` / `Mean` | 当前只有方向性判断，暂无独立 run 或设计文档；待主代理确认是否继续投入 | `深度思考.md` |
| `Q04-R-02` | - | 辅 | 主 | - | 辅 | - | 待研究 | 单独判断 `batch_size` 对 prior 覆盖范围与梯度方差的影响是否足以改变 Q04 结论 | 当前只在证据缺口中被提出，未形成实验方案 | `深度思考.md` |

## 5. 提出待分析

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 当前说明 | 细节位置 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q04-A-01` | 主 | 辅 | - | - | 辅 | - | 提出待分析 | 是否需要为 Q04 单独启用 `调研记录.md`，专门整理更强 joint prior / structured prior / learnable spectral prior 的可选设计；当前这部分还主要借道 `Q01` | `README.md`、`../main.md` |
| `Q04-A-02` | - | 主 | - | 主 | 辅 | - | 提出待分析 | 是否需要启用 `理论推导.md`，把 `q -> \mu`、support mismatch 与长期误差传播的关系单独形式化；当前只够做定性判断 | `深度思考.md` |
| `Q04-A-03` | - | 辅 | - | - | 主 | 主 | 提出待分析 | 是否需要把“Q04 暂作 bridge / baseline”的判断形成正式讨论提纲，拿去和主代理 / 老师 / 同事讨论资源优先级；当前暂不宜提前定案 | `README.md`、`../main.md` |

## 6. 待验收

| 条目 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 验收标准 | 计划来源 |
|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q04-C-01` | - | 主 | 主 | 辅 | 主 | 辅 | 待验收 | “Q04 当前更适合作为 bridge / baseline，而不是主线”这条判断，需要在 `exp14 / exp20 / exp21` 证据链稳定、且 matched 对照缺口被明确写清后，由主代理决定是否同步回写 `../main.md` | `README.md`、`实验记录.md`、`深度思考.md` |

## 7. 已完成

| 条目ID | 结论 | 调研 | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态 | 论据概述 | 细节位置 |
|--------|------|------|----------|----------|----------|-----------|---------------|------|----------|----------|
| `Q04-DONE-01` | raw learnable GRF（`exp14`）不足以支撑这条路线 | - | 主 | 主 | - | 辅 | - | 已完成 | `exp14` 的 MHD / GRF Mean 达到 `301.3 / 346.0`，已经足以构成高价值失败证据 | `实验记录.md`、`README.md` |
| `Q04-DONE-02` | learnable GRF + `soft-Linf`（`exp20`）有局部价值，但不等于 Q04 已成立 | - | 主 | 主 | - | 辅 | - | 已完成 | `exp20` 相比 `exp14` 有数量级改善，但仍远落后于 anchored baseline | `实验记录.md`、`README.md` |
| `Q04-DONE-03` | 当前证据不支持“multi-step PDE 能救回 learnable GRF 的长期 rollout” | - | 主 | 主 | - | 辅 | - | 已完成 | `exp21` 的 `step-3 / step-5` 虽更好，但 `step-10` / `Mean` 更差 | `实验记录.md`、`深度思考.md` |
