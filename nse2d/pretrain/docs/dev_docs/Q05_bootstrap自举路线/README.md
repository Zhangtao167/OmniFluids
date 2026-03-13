# Q05 bootstrap 自举路线

> 说明：本页负责路线摘要与当前判断；Q05 的回答推进控制面见 `sub_main.md`。

## 摘要

| 字段 | 内容 |
|------|------|
| 当前目标 | 回答：只靠 `bootstrap / self-training` 逐步生成更有训练价值的状态，是否能把当前纯 online 路线推进到更稳定、更物理的 5-field MHD surrogate |
| 状态 | active |
| 当前结论 | `self-training / bootstrap` 是当前最强的纯 online 路线；`exp22` 明显优于 `exp13 / exp18 / exp20 / exp21`，但 step-10 仍严重爆炸，离可用长期稳定 surrogate 还很远；`exp24` 这版 dynamic relative loss 已在 early-to-mid 窗口判定不 work 并停止，`exp26` 留下了最强 early positive signal 但已手动停止给 `exp32` 让位，`exp27 / exp28` 仍是 very-early 分支，`exp29` 当前是 launch 失败而非方法结论，`exp30 / exp31` 仍是本地结果待落盘分支，而 `exp32` 已作为 clean no-multi-step ablation 接管 `H800 : t1` |
| 最大阻塞 | 还不能证明 teacher 真把训练态推近物理流形；`teacher` 上线时机、teacher source 与 rollout curriculum 的证据都还不干净；`soft-Linf / multi-step / batch_size` 还没拆出独立贡献；`exp27` 的 `model dt=1.0` 分支、`exp28` 的 no-detach 分支、`exp30` 的 rollout curriculum 分支、`exp31` 的 clean no-soft-Linf 分支与 `exp32` 的 clean no-multi-step 分支都还没有跑到足以下结论的阶段；ensemble / perturb-target 分支还没实现 |
| 先读哪些文件 | `sub_main.md`、`实验记录.md`、`深度思考.md`、`../../../results/EVAL_SUMMARY_TABLE.md`、`../../../run_exp22_self_training_linf_multistep.sh`、`../../../run_exp24_self_training_linf_multistep_normed_MXE_loss.sh`、`../../../run_exp26_self_training_linf_multistep_fixed_normed_MXE_loss.sh`、`../../../run_exp27_self_training_linf_multistep_model_dt1.0.sh`、`../../../run_exp28_self_training_linf_multistep_no_detach.sh`、`../../../run_exp29_self_training_linf_multistep_best_teacher.sh`、`../../../run_exp30_self_training_linf_multistep_rollout_curriculum.sh`、`../../../run_exp31_self_training_multistep.sh`、`../../../run_exp32_self_training_linf.sh`、`../main.md` |
| 最近更新时间 | 2026-03-11 |

## 1. 问题定义

- 这个问题具体是什么：
  - 不再泛泛地问“self-training 有没有一点提升”，而是问：
  - 在没有真实状态锚点的前提下，teacher 能否逐步把随机输入推进到更稳定、更物理的区域，从而真正学好 surrogate。
- 为什么重要：
  - 这是当前 strict online / strict data-free 方向里最有希望的主攻线。
  - 如果这条路成立，它比固定 `GRF` 更接近“训练态可自我改进”的设想，也比单纯继续堆 loss trick 更直接地攻击主问题。
- 当前不处理什么：
  - 不把 `Q05` 写成 strict data-free 已经被解决。
  - 不替代 `Q02` 对“为什么 one-step 变好不等于 rollout 变好”的上游解释。
  - 不把旧 `ROOT_物理随机态生成与彻底data_free训练/` 目录删除、重命名或重写成只剩 `Q05`。

### 1.1 当前子问题树（按实验技巧/疑问清单）

- 局部训练目标：
  - `soft-Linf`、multi-step PDE loss、relative loss 到底分别在帮什么，还是只是组合配置族整体有效。
- 自举协议与反传路径：
  - `self_training_start / update_every / rollout_steps / teacher_source / rollout curriculum / batch_size / rollout_dt / multi_step_pde_detach` 哪些是真有效变量，哪些只是混杂项。
- 新分支候选：
  - 输入扰动、`input_noise_scale`、ensemble RHS averaging、windowed perturb target 是否值得单开实验。
- 现象诊断：
  - 为什么 `train loss / step-1` 能持续变好，但多步 rollout 仍越来越差；为什么当前 best pure-online 仍远差于 anchored baseline。

## 2. 当前结论

- 已回答：
  - 在当前纯 online 候选线里，`self-training / bootstrap` 明显强于当前 `learnable GRF` 路线；`exp22` 的当前汇总结果是 MHD `11.11`、GRF `9.614`，优于 `exp13` 的 `28.12 / 25.01`、`exp18` 的 `17.37 / 16.13`、`exp20` 的 `22.21 / 22.67`、`exp21` 的 `25.72 / 27.49`。
  - `exp13 -> exp18 -> exp22` 这条链说明：更积极的 bootstrap 配置，再叠加 `soft-Linf + multi-step PDE`，确实能把单步到中短期 rollout 拉回一些。
  - 但 `exp22` 在 step-10 仍明显失稳，说明 bootstrap 目前只是在“最差纯 online 路线”之上做了显著改进，还没有接近 anchored baseline 的稳定区间。
  - 从当前可见的按 field 结果看，`exp22` 的主要爆炸仍集中在 `U`，其次是 MHD 测试上的 `n` 与 GRF 测试上的 `vpar`；这说明“field balance”是当前真实问题，而不是一个抽象口号。
- 部分回答：
  - `teacher` 上线时机很可能重要，但现有证据还不干净。`exp13` 与 `exp18` 同时改了 `self_training_start` 和 rollout 步数；`exp23` 不仅把 start 从 `30000` 改成 `100000`，也把 `update_every` 从 `20000` 改成了 `10000`，所以它不是纯粹的“只改上线时机”对照。
  - `exp23` 已经落盘到 `step_18000`，但它的 bootstrap 设计要到 `step_100000` 才激活，因此当前只能说明“晚启动实验已经进入 pre-bootstrap 阶段”，还不能说明晚启动更稳。
  - `exp24` 的 dynamic relative PDE loss 已经实际跑到 `step_8000` 左右；虽然 `step_6000` 时 MHD `1.875`、GRF `1.199` 一度看起来还行，但 `step_8000` 已恶化到 MHD `8.486`、GRF `2.905`，且 `U` 爆炸明显，因此当前已停止，并可暂时判为“不 work”的 dynamic-scale 分支。
  - `exp26` 的 fixed-scale relative PDE loss 已经落盘 `step_2000`，当前为 MHD `1.1631`、GRF `1.0618`；相对 `exp24 step_2000` 的 MHD `7.4550`、GRF `2.4649`，这是当前最强 early positive signal；但该 run 已在约 `step_2400` 手动停止并给 `exp32` 让位，因此 fixed-scale 是否能跨过后续窗口仍待补证。
  - `exp27` 的 `model dt=1.0` 分支已经落盘 `step_0`，当前为 MHD `1.0110`、GRF `1.0458`；但因为它同时把评估协议改成 `10 NFE`，所以还不能回答 `rollout_dt=1.0` 是否真的更稳。
  - `exp28` 的 no-detach 分支已经落盘 `step_0`，当前为 MHD `1.8126`、GRF `1.1483`；但 no-detach 只影响训练反传路径，因此还不能回答 no-detach 是否更稳。
  - `exp29` 的 best-teacher 分支已经在 `20260308215522 : t1` 生成本地 run_tag，但当前这次尝试在初始评估窗口报了 `No CUDA GPUs are available`；因此这不是“负结果”，而是 launch / GPU 可见性问题。
  - `exp30` 的 rollout curriculum 分支已由用户口头同步在 `a100_dev` launch，但当前本地尚未看到结果目录或 log；因此也还不能写任何效果判断。
  - `exp31` 的 clean no-soft-Linf 分支已由用户口头同步在 `20260308215522 : t2` launch；这是以 `exp22` 为底、只去掉 `soft-Linf` 的 clean ablation，但当前本地尚未看到 run 目录或 log，因此还不能回答 `soft-Linf` 是否有独立净收益。
  - `soft-Linf + multi-step PDE` 这个组合配置族当前是有净改进的，但两者的独立贡献仍没拆清。
- 未回答：
  - teacher 生成的训练态是否真的更接近物理流形，而不只是复用当前模型偏差。
  - bootstrap 的收益能否穿过多次 `teacher refresh` 后继续保持，而不是晚一点再爆。
  - fixed-scale relative PDE loss 能否把“训练时的尺度平衡”真正转化成“最终 step-10 的 rollout 稳定性”；dynamic target-RMS 版本当前已基本判负。
  - `rollout_dt=1.0` 的 `exp27` 是否比 `exp22` 的 `rollout_dt=0.1` 更利于 bootstrap 稳定性，还是只是换了一种局部误差结构。
  - `multi_step_pde_detach=0` 的 `exp28` 是否比 `exp22` 的 detach 版本更利于 bootstrap 稳定性，还是只是把反传链变长并放大训练不稳。
  - `self_training_teacher_source=best` 的 `exp29` 是否比 latest teacher 更稳，还是只是减少了 refresh 时的 teacher 漂移。
  - rollout curriculum 的 `exp30` 是否比固定 5-step teacher rollout 更稳，还是只是把错误更晚地推迟。
  - `exp31` 这组 clean no-soft-Linf ablation 是否说明 `soft-Linf` 确有独立净收益，还是 `exp22` 的收益主要来自 multi-step PDE 与更积极 bootstrap 协议。
  - `exp32` 这组 clean no-multi-step ablation 是否说明 `multi-step PDE` 确有独立净收益，还是 `exp22` 的收益主要来自 `soft-Linf` 与更积极 bootstrap 协议。
  - `batch_size` 到底有多大影响，当前还没有 clean ablation。
  - ensemble RHS averaging / windowed perturb target 这类 target-smoothing 分支还没有任何实现或实验。
  - `teacher` 上线时机到底该更早还是更晚，目前还没有 clean ablation 能直接回答。

## 3. 当前阻塞

- 阻塞类型：teacher 质量不可直接观测、时机 ablation 不干净、field-wise 失稳还没系统归档
- 影响：
  - 现在可以确认 `Q05` 是当前最强纯 online 路线，但还不能把“相对更强”误写成“已经学会”。
  - 如果不区分“teacher 时机”“refresh 周期”“relative loss”“batch size”，后续很容易继续把多个变量绑在一起改，导致结论不干净。
- 当前处理方式：
  - 把 `exp13 / exp18 / exp22 / exp23 / exp24 / exp26 / exp27 / exp28 / exp29 / exp30 / exp31 / exp32` 单独归到 `Q05`，不再散落在旧 ROOT 文档里。
  - 对正在运行或刚切换的 `exp22 / exp23 / exp27 / exp28 / exp32` 统一写成“当前可见结果”或“运行中观察”；`exp24` 改写为已停止的 dynamic-scale 负结果；`exp26` 改写为手动停止的 fixed-scale early-signal 分支；`exp29` 当前记录为 launch 失败；`exp30 / exp31` 记录为已 launch 但本地结果待落盘。
  - 把“已经可回答”“运行中准备回答”“待新实验 / 待新实现”的问题轴，统一放到 `sub_main.md` 与 `实验记录.md` 显式维护。

### 3.1 当前实验优先级

- 当前最值得占卡：
  - `exp22` 第一优先，理由是它仍是所有分支的 pure-online 主对照，而且距离下一次 refresh 窗口最近。
  - `exp23` 第二优先，理由是它是唯一已经在路上的 later-start 长线问题，现在停掉会直接浪费等待成本。
  - `exp32` 第三优先，理由是它已经接手 `H800 : t1`，并且是当前最关键的 clean no-multi-step ablation。
- 二级占卡优先级：
  - `exp29` 当前不是“继续跑”的问题，而是“先修复 launch / GPU 可见性，再决定是否保留”为 best-teacher clean 分支；它一旦修好，优先级高于 `exp27`。
  - `exp31` 当前高于 `exp28 / exp30 / exp27`，因为它是已经 launch 的 clean no-soft-Linf ablation，正对 `soft-Linf` 独立贡献这个关键未答问题；但在本地结果目录落盘前，只能先按 launch_announced 维护。
  - `exp28` 仍高于 `exp27`，因为它是更 clean 的 no-detach 对照，而且主要占 `A100`，不直接挤压主 `H800` 队列。
  - `exp30` 当前暂列在 `exp28` 之后、`exp27` 之前，前提是本地结果目录正常落盘并进入首轮 eval 窗口。
  - `exp27` 当前排在最后，主要因为 `10 NFE` 协议变化让对照不够干净；如果要腾 `H800`，优先让它停。
- 下一张空卡先开什么：
  - 先等 `exp32 / exp31` 都进入首轮 `step_0 / 2000`，把 `multi-step` 与 `soft-Linf` 两个 clean ablation 都推进到可比较窗口。
  - 下一张空卡优先开 clean `batch_size` ablation。
  - 然后补 clean start-step ablation。
  - `input_noise_scale -> ensemble / window target` 放在这些 clean ablation 之后。

## 4. 优先阅读路径

1. `sub_main.md`
2. `实验记录.md`
3. `深度思考.md`
4. `../../../results/EVAL_SUMMARY_TABLE.md`
5. `../../../run_exp22_self_training_linf_multistep.sh`
6. `../../../run_exp24_self_training_linf_multistep_normed_MXE_loss.sh`
7. `../../../run_exp26_self_training_linf_multistep_fixed_normed_MXE_loss.sh`
8. `../../../run_exp27_self_training_linf_multistep_model_dt1.0.sh`
9. `../../../run_exp28_self_training_linf_multistep_no_detach.sh`
10. `../../../run_exp31_self_training_multistep.sh`
11. `../../../run_exp32_self_training_linf.sh`
12. `../main.md`

## 5. 文件索引

| 文件 | 是否启用 | 用途 |
|------|----------|------|
| `sub_main.md` | 是 | `Q05` 的二级 `main` 控制面，用来管理回答推进、子问题状态迁移与运行中实验，不重复长正文证据 |
| `调研记录.md` | 否 | `Q05` 当前不单独承载外部文献；上游资料继续放在 `Q01_lit_method_map/` |
| `实验记录.md` | 是 | 记录 `exp13 / exp18 / exp22 / exp23 / exp24 / exp26 / exp27 / exp28 / exp29 / exp30 / exp31 / exp32` 的脚本差异、运行状态、结果与风险 |
| `深度思考.md` | 是 | 分析 teacher 为什么可能有用、哪里可能自举放大错误、relative loss 为什么未必自动变成 rollout 改善 |
| `理论推导.md` | 否 | 形式化推导暂继续留在 `Q02_第一性原理与可行性/`，这里先不重复建设 |
| `交流讨论.md` | 否 | 当前还没有必须单独保存的 `Q05` 讨论记录 |
