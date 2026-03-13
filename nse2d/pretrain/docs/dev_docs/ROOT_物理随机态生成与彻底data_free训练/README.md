# 物理随机态生成与彻底 data-free 训练

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 针对复杂 PDE 动力过程，尽量不依赖真实仿真数据，只靠物理约束与随机生成训练态，训练出可用且尽量长期稳定的 data-free 5-field MHD surrogate model |
| 状态 | active |
| 当前结论 | 目前效果最好的仍然是“GRF 预训练，再接 simulation data 继续训练”；这说明真实状态锚点仍然关键，也说明 strict data-free 主问题还没有被解决 |
| 最大阻塞 | 当前随机态先验仍然太弱，且 one-step 变好并不会自动转化成 multi-step rollout 稳定；当前主攻方向仍是 bootstrap / self-training，但 `exp24` 这版 dynamic relative loss 已停，`exp26` 当前只保留为 fixed-scale early-signal 归档，后续主要分叉已扩展到 `exp23 / exp27 / exp28 / exp29 / exp30 / exp31 / exp32`；其中 `exp29` 当前先暴露 launch / GPU 可见性问题，`exp30 / exp31 / exp32` 本地结果目录待落盘，细节以 `Q05` 文档维护为准 |
| 先读哪些文件 | `../shared_context/common_context.md`、`../shared_context/evaluation_protocol.md`、`实验记录.md`、`深度思考.md`、`../../../CURRENT_STAGE_DATA_FREE_RESEARCH_REPORT.md`、`../../../results/EVAL_SUMMARY_TABLE.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 主问题不是“继续调一个 loss”，而是：
  - 对复杂 PDE 动力过程，怎样构建出更符合物理的随机生成训练态，并只靠这些随机态与物理约束去训练模型，最终尽量实现彻底的 data-free 5-field MHD surrogate。
- 为什么重要：
  - 当前 strict data-free 线还没有成功。
  - 已有结果说明：如果没有真实状态锚点，模型虽然能把训练 loss 和单步误差压下去，但多步 rollout 依然容易爆掉。
  - 因此真正要解决的是“随机训练态分布如何逼近真实物理流形”，而不只是“loss 如何继续下降”。
- 当前不处理什么：
  - 不把当前最优 staged / offline 结果误写成 strict data-free 已经成立。
  - 不把所有问题都归结成单个 trick 是否有效。
  - 不把所有 field 的生成约束一刀切处理。

## 2. 需要回答的主问题

| 主问题ID | 类型 | 主问题 | 当前理解 | 当前状态 | 文档入口 |
|----------|------|--------|----------|----------|----------|
| `Q01` | 问题分析 | 相关参考文献、已有方法、主要挑战是什么，后续应拆出哪些关键子问题 | 先把已有方法和挑战摸清，后续子问题才不会漂移 | 进行中 | `../Q01_lit_method_map/README.md` |
| `Q02` | 问题分析 | 从第一性原理分析：这个根问题为什么有价值、是否可行、主要挑战和可行动的方法是什么 | 重点解释 strict data-free 为什么难、为什么真实状态锚点目前仍关键 | 进行中 | `../Q02_第一性原理与可行性/README.md` |
| `Q03` | 方法路线 | 纯用随机产生的物理场作为输入（如 GRF），只用 PDE loss 训练，能否学到较准确的仿真模型 | 这是 strict data-free 最直接的路线，也最容易暴露输入分布错位 | 进行中 | `../Q03_固定随机先验与PDE_only路线/README.md` |
| `Q04` | 方法路线 | 通过可学习 GRF 端到端优化输入分布，能否逐渐构建更好的训练态并学好模型 | 关键在于输入分布是否能被 jointly optimize 到更物理的区域 | 进行中 | `../Q04_可学习GRF路线/README.md` |
| `Q05` | 方法路线 | 通过 bootstrap / self-training 的方式，能否逐渐学好仿真模型 | 当前主攻方向；关键是 teacher 是否真能把训练态推向更稳定、更物理的区域 | 进行中（主攻） | `../Q05_bootstrap自举路线/README.md` |
| `Q06` | 方法路线 | 通过生成模型用少量数据学习到分布，再用生成数据训练 | 有研究价值，但相对 strict data-free 稍微偏题，可作为边界路线保留 | 提出待分析 | `../Q06_少量数据生成模型路线/README.md` |

## 3. 当前结论

- 已确认：
  - 当前经验上效果最好的路线仍是“GRF 预训练 + simulation data 继续训练”。
  - 纯 GRF / raw online 路线里，经常出现“训练 loss 持续下降、单步 eval 变好，但多步 eval 反而变差”的现象。
  - 这说明当前 objective 更像是在优化局部一步拟合，而不是长期 rollout 稳定性。
  - 已有结果的 `exp18 / exp20 / exp21 / exp22` 这类 online 强化线目前都还没有把 strict data-free 真正跑通。
- 运行中观察：
  - `exp23` 仍在继续追踪 bootstrap teacher 上线时机。
  - `exp24` 的 dynamic relative loss 分支已在 early-to-mid 观察窗口判定不 work 并停止。
  - `exp26` 已保留为 fixed-scale relative loss 的 early-signal 归档；`exp27` 仍在继续验证 `model dt=1.0` 这条 `Q05` 新分叉。
  - `exp28` 仍在继续验证 no-detach multi-step 路线。
  - `exp29` 是 latest-vs-best teacher source 新分支，但当前本地 run 在初始评估窗口报 `No CUDA GPUs are available`，还不能算有效方法结果。
  - `exp30` 是 rollout curriculum 新分支，用户已同步 launch，但当前本地尚未看到结果目录与 log。
  - `exp31` 与 `exp32` 分别是 clean no-soft-Linf 与 clean no-multi-step ablation；当前都还处在“已 launch / 待本地落盘”阶段。
- 仍不确定：
  - `batch_size` 到底在多大程度上影响 GRF 覆盖范围、梯度方差和 rollout 稳定性。
  - 当前 rollout 爆掉的主要 pattern，到底更像是输入分布错位、误差传播失稳，还是某几个 field / 模态被系统性放大。
  - `n` 场和 `Ti` 场是否确实不需要绝对值限制，还是只是在当前可视化样本中看起来没必要。

## 4. 当前阻塞

- 阻塞类型：随机态先验、协议一致性、长期稳定性解释
- 影响：
  - 现在可以快速构建很多 online 变体，但这些变体的成功或失败常常不够可解释。
  - 如果不先把主问题收敛到“随机态物理性 + rollout 稳定性”，后续会继续陷入试参堆 trick。
- 当前处理方式：
  - 已把当前阶段的大报告、评估总表和 recent experiment 结论回写到本问题文档。
  - 后续所有开发都默认先看 `实验记录.md` 与 `深度思考.md`。

## 5. 优先阅读路径

1. `实验记录.md`
2. `深度思考.md`
3. `理论推导.md`
4. `调研记录.md`
5. `交流讨论.md`
6. `../shared_context/evaluation_protocol.md`

## 6. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `调研记录.md` | 是 | 汇总内部报告、结果表与外部相关方向，避免主问题表述漂移 |
| `实验记录.md` | 是 | 记录 baseline、random prior 变体、脚本、异常与结果 |
| `深度思考.md` | 是 | 分析为什么 single-step 变好但 rollout 变差，比较生成模型 / learnable GRF / self-training bootstrap |
| `理论推导.md` | 是 | 形式化“局部误差下降为何不保证多步正确”以及 `q` vs `mu` 分布问题 |
| `交流讨论.md` | 是 | 记录用户、Agent、老师等改变路线判断的讨论 |
