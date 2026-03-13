# Q03 固定随机先验与 PDE-only 路线

> 说明：本页负责 Q03 的问题摘要、边界、当前判断与入口；当前回答推进控制面见 `sub_main.md`。

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 把 fixed random prior / pure GRF / PDE-only 这条 strict data-free 最干净的路线单独沉淀，回答它当前能做到哪里、不能做到哪里，以及应拿什么作参照 |
| 状态 | active |
| 当前结论 | 当前项目内证据更支持：这条路线必须保留为 strict data-free 的 clean baseline，但独立 fixed `GRF + PDE-only` 目前还不能被视为可用仿真模型；`PDE loss` 本身并非不可学，关键瓶颈更像是 fixed prior 离真实物理流形太远且缺少五场联合结构 |
| 最大阻塞 | 缺 clean fixed-prior historical run 元数据与系统失稳归档；更物理 fixed prior、`soft-Linf` / multi-step 在 fixed prior 下的净收益，以及爆炸的 field / 频谱 / 径向 pattern 都还没被单独钉住 |
| 先读哪些文件 | `sub_main.md`、`实验记录.md`、`深度思考.md`、`../shared_context/evaluation_protocol.md`、`../ROOT_物理随机态生成与彻底data_free训练/实验记录.md`、`../main.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 限定在 fixed random prior / pure GRF / PDE-only 路线：用固定随机物理场作为训练输入，不依赖 learnable prior、bootstrap teacher 或 simulation continuation，问能否训练出较准确且可 rollout 的 surrogate。
- 为什么重要：
  - 它是 strict data-free 最直接、最干净的路线，也是判断“固定随机态先验本身够不够”的 lower bound 和 clean control。
  - 若它失败，能直接暴露 fixed prior 与真实物理流形错位、五场联合结构缺失，以及局部 objective 与长期 rollout 脱钩的问题。
- 当前不处理什么：
  - 不把 `Q04` 的 learnable prior / learnable `GRF` 混进来。
  - 不把 `Q05` 的 bootstrap / self-training 混进来。
  - 不把 anchored baseline 或 staged rescue 写成这条路线已经成功。
  - 不在缺少 clean route-specific run 元数据时强写过细定量结论；缺失处统一保留占位。

## 2. 当前结论

- 已确认：
  - 这条路线必须保留为 strict data-free 的 clean baseline / lower bound。
  - `exp_offline_pde (PDE loss, 1GPU)` 在 anchored real-state support 上达到 MHD / GRF Mean `0.4097 / 1.060`，说明 `PDE loss` 本身不是学不会；障碍更像在 fixed random support，而不是 physics objective 本身。
  - 源材料已经给出 route-level 判断：当前独立 fixed `GRF + PDE-only` 不可用，raw `GRF` warmup 价值很低。
  - 当前独立 per-field `GRF` 很可能缺少五场联合结构，因此即使 `train loss` 和 single-step 指标下降，也未必能转化成 multi-step rollout 稳定。
- 仍不确定：
  - 更物理的 fixed prior 是否能把这条路线推进到可用区间。
  - `soft-Linf` / multi-step PDE 在 fixed prior 下是净收益、暂时 stabilizer，还是只是把优化做得更努力。
  - 爆炸 pattern 主要来自哪些 field、频段或径向结构，当前还缺系统归档。
  - `batch_size` 在 `GRF` 覆盖、梯度方差和长期稳定性里占多大比重。

## 3. 当前阻塞

- 阻塞类型：fixed prior 物理性不足、route-specific 证据回填不足、长期失稳诊断不足
- 影响：
  - 目前能下方向性判断，但还缺一张干净的 Q03 route table 来回答“plain fixed prior 到底差到什么程度、被什么 pattern 卡住”。
  - 也因此很难单独判断更物理 fixed prior 或 loss trick 是否真的有独立收益。
- 当前处理方式：
  - 先把与 Q03 直接相关的 baseline、failure 和 open question 从 ROOT 文档中抽出来。
  - 对缺少 `run_id` / `path` / 评分表的历史 fixed-prior 证据统一保留占位，不编造数值。
  - 所有 route judgment 默认同时对照 MHD / GRF 与 `step-10` / `Mean`。

## 4. 优先阅读路径

1. `sub_main.md`
2. `实验记录.md`
3. `深度思考.md`
4. `../shared_context/evaluation_protocol.md`
5. `../ROOT_物理随机态生成与彻底data_free训练/实验记录.md`
6. `../ROOT_物理随机态生成与彻底data_free训练/深度思考.md`
7. `../main.md`

## 5. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `sub_main.md` | 是 | Q03 的二级 `main` 控制面；管理子问题拆解、状态迁移和当前推进 |
| `调研记录.md` | 否 | 当前没有新增外部资料；route-level facts 先只用项目内来源 |
| `实验记录.md` | 是 | 固定 prior 路线的 baseline、adjacent references、historical failure 和待回填 run |
| `深度思考.md` | 是 | 能力边界、主要假设、loss trick 角色、下一步判别条件 |
| `理论推导.md` | 否 | `q` vs `\mu` 与 one-step vs rollout 的通用推导已在 `Q02`；Q03 暂无新增 route-specific 符号 |
| `交流讨论.md` | 否 | 当前没有需要单独保留的 Q03 独立讨论记录 |
