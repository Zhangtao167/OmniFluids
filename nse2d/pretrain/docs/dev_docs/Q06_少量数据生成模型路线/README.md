# Q06 少量数据生成模型路线

> 说明：本页负责路线摘要、边界和当前判断；Q06 的回答推进控制面见 `sub_main.md`。

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 明确“用少量真实数据学习状态分布或先验，再用生成态训练 surrogate”这条路线在当前项目里的定位、边界和首批 gating 问题，避免在当前证据极弱时过早启动 `diffusion/VAE/flow` 实现 |
| 状态 | proposed |
| 当前结论 | 这条路线有研究价值，但在当前项目里更像边界路线 / fallback；仓库内当前无直接实验支持，也没有专门实现、脚本或结果可以证明它已值得优先投入 |
| 最大阻塞 | 缺直接实现与实验、缺最小数据预算定义、缺 clean evaluation protocol，而且这条路线相对 strict data-free 已经偏向 `data-informed prior` |
| 先读哪些文件 | `sub_main.md`、`调研记录.md`、`深度思考.md`、`../main.md`、`../../../results/EVAL_SUMMARY_TABLE.md`、`../../../CURRENT_STAGE_DATA_FREE_RESEARCH_REPORT.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 允许使用少量真实状态或短轨迹，学习一个比独立 `GRF` 更强的状态分布、潜变量先验或 manifold prior，再从该分布生成训练态，用于后续 surrogate 训练、warmup 或对照实验。
- 为什么重要：
  - 当前项目的主矛盾之一就是随机训练态离真实可达物理流形太远；如果 small-data generative route 真能学到更强的五场联合结构，它有可能比独立 `GRF` 更接近“物理随机态生成”这个核心目标。
- 当前不处理什么：
  - 不把这条路线写成当前主攻线。
  - 不把 `Q04/Q05` 的 learnable `GRF`、self-training bootstrap 结果误算成 `Q06` 证据。
  - 不在“当前无直接实验支持”的前提下假装 `diffusion/VAE/flow` 已经被验证有效。
  - 不把“允许少量真实数据”与 strict data-free 混为一谈。

## 2. 当前结论

- 已确认：
  - `../main.md` 已将 `Q06` 定位为“有研究价值，但相对当前 strict data-free 主题稍微偏题；可作为边界路线或 fallback 方向保留”。
  - 本轮仓库关键词检索显示，更具体的 `diffusion / VAE / flow / latent / manifold prior` 内容只直接出现在 `../main.md` 和 `../../../CURRENT_STAGE_DATA_FREE_RESEARCH_REPORT.md`；当前无直接实验支持。
  - `../../../results/EVAL_SUMMARY_TABLE.md` 中没有 `Q06` 专属实验条目；现有结果覆盖的是 anchored baseline、`Q04` learnable `GRF`、`Q05` self-training 等路线。
  - 代码里的 `generator` 目前主要指 `GRF` 或 `ModelEvolvedGRFGenerator`，属于随机先验 / bootstrap 机制，不是 small-data generative model。
  - 因此这条路线当前只能保留为边界路线 / fallback；当前无直接实验支持。
- 仍不确定：
  - 如果未来真做 `Q06`，首试对象更应该是低成本 latent prior、`VAE` 风格路线，还是更重的 `diffusion` / `flow`。
  - 需要多少真实数据才足以形成有意义的先验，而不是只学到很窄的 anchor 模仿器。
  - 生成态的好坏应主要由样本可视化、统计量匹配，还是 downstream rollout 改善来定义。
  - 这条路线是否值得排在更强结构先验、`Q04`、`Q05` 之前投入。

## 3. 当前阻塞

- 阻塞类型：证据缺口、口径边界、评估协议缺失
- 影响：
  - 很容易在没有 clean success criterion 的情况下启动一个算力开销很大的方向。
  - 也很容易把“small-data prior”误写成 strict data-free 的主线，导致研究叙事漂移。
- 当前处理方式：
  - 先把它从旧 `ROOT_...` 的混写内容中独立出来，只保留为边界路线。
  - 对所有没有落到实现、实验或结果表的点，统一明确写成“当前无直接实验支持”。
  - 先收敛 gating 问题，再决定未来是否立项。

## 4. 优先阅读路径

1. `sub_main.md`
2. `调研记录.md`
3. `深度思考.md`
4. `../main.md`
5. `../../../results/EVAL_SUMMARY_TABLE.md`
6. `../../../CURRENT_STAGE_DATA_FREE_RESEARCH_REPORT.md`

## 5. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `sub_main.md` | 是 | `Q06` 的二级 `main` 控制面，管理子问题拆解、状态与回答推进顺序 |
| `调研记录.md` | 是 | 汇总 Q06 当前定位、内部证据缺口、关键词检索结果与 future gating questions |
| `实验记录.md` | 否 | 当前无 Q06 专属实验，暂不启用 |
| `深度思考.md` | 是 | 回答为什么它是 boundary / fallback，而不是当前主攻线 |
| `理论推导.md` | 否 | 当前无独立推导需要沉淀 |
| `交流讨论.md` | 否 | 当前先不单独记录 |

## 6. 首批 gating 问题

| gating 问题 | 为什么必须先答 | 当前状态 |
|-------------|----------------|----------|
| 这条路线在项目叙事里算 strict data-free、weakly data-free，还是 `data-informed prior`？ | 不先定口径，后续所有“是否值得做”的讨论都会漂移 | 当前更接近边界路线 / fallback |
| 未来要学习的对象到底是什么：初始状态分布、短 rollout manifold，还是 latent dynamics prior？ | 这会直接决定模型选型、数据组织和评估方式 | 当前未定 |
| 最小真实数据预算是多少？ | 没有数据预算就无法判断这条路线是否真的“少量数据”且是否值得 | 当前无直接实验支持 |
| 第一版成功标准是什么？ | 如果不能在 downstream surrogate 上带来净收益，样本“看起来更像”并不构成项目价值 | 当前未定 |
| 应该先做低成本 probe 还是直接上高容量生成模型？ | 可避免在证据极弱时过早投入 `diffusion` 级别成本 | 当前更倾向先做低成本 probe，而不是直接上重模型 |
| 哪些 distribution-level 指标最能表示“更接近物理流形”？ | 没有统计量与 rollout 双重口径，就很难解释成败 | 当前未定，需与主代理后续统一 |
