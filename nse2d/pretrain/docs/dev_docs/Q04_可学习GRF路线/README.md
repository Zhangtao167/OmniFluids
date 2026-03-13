# Q04 可学习 GRF 路线

> 说明：本页负责路线摘要与当前判断；Q04 的回答推进控制面见 `sub_main.md`。

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 判断可学习 GRF 是否能通过端到端优化输入分布，把随机态推近更有训练价值的区域，并在不依赖真实状态锚点的前提下改善 5-field MHD surrogate 的 rollout |
| 状态 | active |
| 当前结论 | `exp20` 相比 `exp14` 显著降低了灾难性失稳，说明“可学习先验 + 稳定化手段”有局部价值；但 `exp21` 没有把长期 rollout 救回来，且当前低维 per-field learnable GRF 仍远弱于 anchored baseline，因此这条线更适合作为 bridge / baseline，而不是当前主线 |
| 最大阻塞 | 缺 matched fixed-vs-learnable 对照，且当前参数化缺少五场联合结构；现有改善是否来自 learnable prior 本身，还是更多来自 `soft-Linf` 等局部稳定化手段，仍未定论 |
| 先读哪些文件 | `sub_main.md`、`实验记录.md`、`深度思考.md`、`../main.md`、`../../../results/EVAL_SUMMARY_TABLE.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 不是只问“把 GRF 参数设成可学习能不能降 loss”，而是问：能否通过联合优化随机态生成器和 surrogate，把训练分布逐步推近更物理、更有训练价值的区域。
- 为什么重要：
  - 这是 fixed GRF 与更强生成模型之间成本最低、解释性最强的一条中间路线。
  - 如果它成立，说明 strict data-free 不一定非要先上重型生成模型，也可能先从可学习先验打开缺口。
- 当前不处理什么：
  - 不把 anchored / staged 的成功误写成 learnable GRF 已经解决 strict data-free。
  - 不把 `soft-Linf`、multi-step PDE 之类 loss trick 单独当成这条路线的最终答案。
  - 不直接覆盖 `Q05` 的 bootstrap / self-training 路线判断。

## 2. 当前结论

- 已确认：
  - `exp14 (learnable GRF)` 是高价值失败：MHD Mean `301.3`，GRF Mean `346.0`，说明“只把 GRF 设成可学习”远远不够。
  - `exp20 (learn-GRF + soft-Linf)` 相比 `exp14` 显著改善：MHD / GRF Mean 从 `301.3 / 346.0` 降到 `22.21 / 22.67`，说明当前 learnable GRF 至少能配合稳定化手段把路线从“灾难性失效”拉回“可分析区间”。
  - `exp21 (learn-GRF + soft-Linf + multi-step)` 在 `step-3` / `step-5` 上优于 `exp20`，但 `step-10` 与 `Mean` 更差；当前证据不支持“multi-step PDE 能救回 learnable GRF 的长期 rollout”。
  - 与 anchored baseline 相比，这条路线仍很弱：`exp_offline_pde (1GPU)` 为 `0.4097 / 1.060`，`exp1b (1GPU)` 为 `0.4099 / 1.030`，说明 learnable GRF 还没有把训练分布推进到可用物理流形附近。
- 仍不确定：
  - `exp20` 的改善到底有多少来自 learnable prior，本质上是否只是 `soft-Linf` 在帮忙稳局部目标。
  - 当前 low-dimensional per-field learnable GRF 是否已经到上限，还是只是参数化还不够强。
  - learnable GRF 是否真的在改善 `q -> \mu`，还是只在错误支持集上做了更好的局部适配。
  - `batch_size`、field-specific 约束、五场联合结构缺失，对这条路线各自贡献多大，仍待补证。

## 3. 当前阻塞

- 阻塞类型：对照不干净、参数化太弱、长期指标未被救回
- 影响：
  - 现在可以确认“比 `exp14` 好很多”，但还不能确认“是否值得继续作为主线投入”。
  - 如果没有 matched ablation，很容易把 `soft-Linf` 或训练协议的收益误判成 learnable GRF 的收益。
- 当前处理方式：
  - 先把 `exp14 / exp20 / exp21` 作为最小证据链整理清楚。
  - 暂把 learnable GRF 定位为 bridge / baseline，等待 matched 对照和更强 prior 证据再决定是否继续升格。

## 4. 优先阅读路径

1. `sub_main.md`
2. `实验记录.md`
3. `深度思考.md`
4. `../main.md`
5. `../../../results/EVAL_SUMMARY_TABLE.md`

## 5. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `sub_main.md` | 是 | Q04 的二级 `main` 控制面；维护子问题拆解、状态流转、运行实验与待验收判断 |
| `调研记录.md` | 否 | 外部文献与更强 joint prior 的专门调研，当前先留给 `Q01` 或后续需要时再启用 |
| `实验记录.md` | 是 | 记录 `exp14 / exp20 / exp21` 的目的、配置、结果、风险与下一步 |
| `深度思考.md` | 是 | 回答 learnable GRF 是否真在改善训练分布，以及它为何暂时不适合升格为主线 |
| `理论推导.md` | 否 | 如后续要单独形式化 `q -> \mu` 与 learnable prior 的关系，再启用 |
| `交流讨论.md` | 否 | 当前还没有必须独立沉淀的 Q04 讨论记录 |
