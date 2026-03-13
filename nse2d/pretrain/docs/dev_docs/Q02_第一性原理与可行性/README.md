# Q02 第一性原理与可行性判断

> 说明：本页负责问题摘要、边界、当前判断与入口；Q02 的回答推进控制面见 `sub_main.md`。

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 从第一性原理回答：“物理随机态生成与彻底 data-free 训练”为什么值得做、在当前项目约束下是否可行、主要挑战是什么，以及后续应优先验证哪些方法与证据 |
| 状态 | active |
| 当前结论 | 这个问题有明确研究价值；按当前项目内口径，当前整体更接近 `Level B/C`，而 strict `Level D` 仍未证成。现有证据更支持：`PDE loss` 本身不是学不会，主矛盾在随机训练态与真实 rollout 分布错配，以及局部一步目标不能自动带来长期稳定 |
| 最大阻塞 | 随机态先验过弱、五场联合结构是否缺失仍未证实、train/inference 与 val/test 协议尚未完全干净，导致“为什么 anchor 重要”“为什么 one-step 不等于 rollout”仍需持续补证 |
| 先读哪些文件 | `sub_main.md`、`深度思考.md`、`理论推导.md`、`实验记录.md`、`../shared_context/evaluation_protocol.md`、`../shared_context/common_context.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 不是再问“某个 loss trick 是否有效”，而是回答：在缺少真实状态锚点时，怎样的随机训练态、训练协议与稳定性控制，才足以支撑 2D 5-field MHD surrogate 的长期 rollout。
  - 关注四件事：这个根问题为什么有价值；当前条件下是否可行；最主要的瓶颈是什么；哪些方法值得继续投入。
- 为什么重要：
  - 根目标本身就是尽量减少对真实仿真数据的依赖，只靠物理约束与随机生成训练态训练出可用 surrogate。
  - 如果不先回答“是否可行、难点在哪、该拿什么做参照”，后续很容易继续在 raw online 路线上堆 trick，而无法判断进展是否真正接近 strict data-free。
  - 这也是后续 `Q03-Q06` 路线判断的上游决策面。
- 当前不处理什么：
  - 不把 staged / offline anchor 的成功写成 strict data-free 已经成立。
  - 不替代 `Q03-Q06` 的 route-specific 实验结论。
  - 不在缺少正式定义时强写 `Level B/C/D` 或 strict `data-free` 的最终边界；相关口径先保留占位。

## 2. 当前结论

- 已确认：
  - 当前最可信的经验参照仍是带真实状态锚点的路线，包括 `exp_offline_pde (1GPU)`、`exp1b (sup dt=1s, 1GPU)` 以及 staged / offline rescue 类方法。
  - 多个 raw online 结果共同说明：`train loss` 和单步指标可以继续下降，但多步 rollout 仍可能系统性恶化。
  - 这意味着当前主矛盾更像是长期 rollout 稳定性与训练分布支持问题，而不只是 `PDE loss` 还不够强。
  - 按当前项目内口径，整体更接近 `Level B/C` 而不是真正的 `Level D`；其中 `Level B/C/D` 的正式定义仍待补。
- 仍不确定：
  - 训练分布错配是否是第一主因，还是与 train/inference mismatch、preprocessing 差异、误差传播共同构成主瓶颈。
  - 五场联合结构缺失是否是当前随机态先验最关键的缺口。
  - 什么才是 strict data-free 下的“最小充分先验”，目前还没有正式定义。
  - `data-free` 口径边界与路线资源优先级仍需继续确认。

## 3. 当前阻塞

- 阻塞类型：随机态先验不足、协议一致性不足、长期稳定性解释不足、口径边界未定
- 影响：
  - 很容易把“单步变好”“局部 loss 更低”误当成根问题已接近解决。
  - baseline 与 online 变体的比较容易被协议差异污染，导致路线排序不稳。
- 当前处理方式：
  - 用 `RUN-OFF-PDE-1GPU` 与 `RUN-SUP-1GPU` 继续做干净单卡 baseline 校准。
  - 用 `exp20 / exp21 / exp22` 这类结果持续检验“why one-step != rollout”。
  - 用 `sub_main.md` 维护 Q02 的子问题、状态桶与当前运行实验入口。
  - 把第一性原理判断、形式化推导和证据性 run 记录分开沉淀到本目录。

## 4. 优先阅读路径

1. `sub_main.md`
2. `深度思考.md`
3. `理论推导.md`
4. `实验记录.md`
5. `../shared_context/evaluation_protocol.md`
6. `../shared_context/common_context.md`
7. `../main.md`

## 5. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `sub_main.md` | 是 | Q02 的二级 `main` 控制面，管理子问题拆解、状态桶、当前 baseline / 对照实验入口和回答推进顺序 |
| `调研记录.md` | 否 | Q02 当前不单独承载外部文献；相关上游证据先留在 `Q01_lit_method_map/` |
| `实验记录.md` | 是 | 保存支撑 Q02 判断的 baseline、对照实验与异常现象 |
| `深度思考.md` | 是 | 回答“为什么值得做 / 为什么难 / 哪些方法值得继续投入” |
| `理论推导.md` | 是 | 形式化“one-step 变好为何不保证 rollout 稳定”以及训练分布与真实 rollout 分布的错配问题 |
| `交流讨论.md` | 否 | 当前未见必须单独保留的 Q02 讨论记录 |
