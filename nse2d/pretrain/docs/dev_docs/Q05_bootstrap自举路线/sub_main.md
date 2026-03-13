# Q05 bootstrap 自举路线 sub_main

> 用途：这是 `Q05` 的二级 `main` 控制面，只管理“这个主问题当前回答推进到哪里、下一步盯什么、哪些结论已可写、哪些还不能写”。长证据与细节统一下沉到 `README.md`、`实验记录.md`、`深度思考.md`。

## 0. 当前快照


| 字段     | 内容                                                                                           |
| ------ | -------------------------------------------------------------------------------------------- |
| 当前主问题  | `bootstrap / self-training` 能否逐步生成更有训练价值的状态，并把当前纯 online 路线推进到更稳定、更物理的 5-field MHD surrogate |
| 当前主判断  | `self-training / bootstrap` 是当前最强的纯 online 路线，但还不能证明 teacher 真把训练态推近物理流形                     |
| 当前主体工作 | 以 `exp22 > exp23 > exp32` 维护当前 `Q05` 主 `H800` 队列：`exp26` 已保留为 fixed-scale early-signal 归档并给 `exp32` 让位；同时把 `exp29` 作为“先修 launch / GPU 可见性”的并行任务，把 `exp31` 作为 clean no-soft-Linf ablation 记到 `20260308215522 : t2`，再把 `exp28 / exp30` 放在 `A100` 二级分支上继续观察，并保留 `exp27` 作为当前最先可让位的 `H800` 次级分支；下一张主卡优先留给 clean `batch_size` ablation |
| 当前最大阻塞 | 时机 ablation 不干净；`soft-Linf / multi-step / batch_size` 还没拆出独立贡献；teacher source、rollout curriculum、`model dt=1.0`、no-detach 与 clean no-multi-step 还没分出净收益；`exp29` 当前卡在 launch 失败，`exp31` 与 `exp32` 当前都还没在本地落出首轮可比较结果；ensemble / perturb-target 分支尚未实现；teacher 训练态“更物理”缺直接证据          |
| 当前入口   | `README.md`、`实验记录.md`、`深度思考.md`                                                              |
| 最近更新时间 | 2026-03-11                                                                                   |


## 1. 已有结论和观察


| 编号           | 结论或观察                                                                                                  | 当前归类 | 证据入口                  |
| ------------ | ------------------------------------------------------------------------------------------------------ | ---- | --------------------- |
| `OBS-Q05-01` | `exp22` 当前是最强的 `Q05` 配置族，也是当前最强的纯 online 路线；当前汇总结果优于 `exp13 / exp18 / exp20 / exp21`                   | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-02` | `exp13 -> exp18 -> exp22` 说明更积极的 bootstrap 配置族，再叠加 `soft-Linf + multi-step PDE`，确实把单步到中短期 rollout 拉回一些 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-03` | `exp22` 在 step-10 仍明显失稳，主要爆炸集中在 `U`，其次涉及 `n / vpar`，说明当前主矛盾仍是 field balance 与长期 rollout 稳定性            | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-04` | `exp23` 目前只进入 pre-bootstrap 阶段，不能拿来回答 later-start 是否更稳                                                 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-05` | `exp24` 已跑到 `step_8000` 左右并在 early-to-mid 窗口暴露明显失稳，当前已停止；dynamic relative PDE loss 可暂判不 work                  | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-06` | `exp26` 已在 `H800 : t1` 跑过 `step_2000`：MHD `1.1631`、GRF `1.0618`；相对 `exp24 step_2000` 的 MHD `7.4550`、GRF `2.4649`，这是当前最强 early positive signal；但该 run 已在约 `step_2400` 手动停止并给 `exp32` 让位，因此 fixed-scale 是否能跨过后续窗口仍待补证 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-07` | `exp27` 已在 `H800 : t2-2` 落盘 run 目录与 `step_0`；当前 `step_0` 为 MHD `1.0110`、GRF `1.0458`，但因 `rollout_dt=1.0` 带来 `10 NFE` 协议变化，不能直接当正结果 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-08` | “当前结果更强”不等于“teacher 已经学会生成更物理的训练态”；这一点目前仍未定论                                                           | 待补证  | `深度思考.md`、`README.md` |
| `OBS-Q05-09` | `exp28` 已在 `a100_dev : t` 落盘 run 目录与 `step_0`；这是基于 `exp22` 的 clean no-detach ablation，但 no-detach 的作用要到训练后才可能体现 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-10` | `exp29` 已在 `20260308215522 : t1` 生成本地 run_tag `7ada96df-03_11_12_48_41-K4-mx128-w80-L12-od10`，但当前尝试在初始评估窗口报 `No CUDA GPUs are available`；这更像 launch / GPU 可见性问题，而不是 best-teacher 分支的负结果 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-11` | `exp30` 已由用户口头同步在 `a100_dev` launch；当前本地尚未看到结果目录或 log，因此只能记录为 launch announced，不能写成已正常运行或已有结果 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-12` | `exp31` 已由用户口头同步在 `20260308215522 : t2` launch；这是以 `exp22` 为底、只去掉 `soft-Linf` 的 clean ablation，但当前本地尚未看到 run 目录或 log，因此只能记录为 launch announced，不能写成已正常运行或已有结果 | 已确认  | `README.md`、`实验记录.md` |
| `OBS-Q05-13` | `exp32` 已由用户同步在 `H800 : t1` 所在 session 手动启动；脚本是以 `exp22` 为底、只关掉 `multi-step PDE` 的 clean no-multi-step ablation。当前 terminal 头部已打印，但本地 run 目录与首轮 eval 仍待落盘，因此只能先记为 launch_announced | 已确认  | `README.md`、`实验记录.md` |


## 2. 关键子问题拆解


| 子问题ID    | 子问题                                                     | 主要回答抓手                | 当前状态  |
| -------- | ------------------------------------------------------- | --------------------- | ----- |
| `Q05-S1` | teacher 生成态是否真的比原始 `GRF` 更接近物理流形                        | 深度思考 + 实验验证 + 必要时讨论   | 提出待分析 |
| `Q05-S2` | `self_training_start_step` 应该更早还是更晚                     | 实验验证 + 深度思考           | 进行中   |
| `Q05-S3` | `teacher refresh / teacher source` 收益能否跨过多次刷新继续保持                        | 实验验证 + 深度思考           | 进行中   |
| `Q05-S4` | `relative PDE loss` 改善的是训练尺度平衡，还是最终 rollout 稳定性；dynamic vs fixed scale 哪种更稳 | 实验验证 + 深度思考           | 进行中   |
| `Q05-S5` | timing / refresh / rollout steps / rollout curriculum / `batch_size` / `rollout_dt` / `multi_step_pde_detach` 是否需要 clean ablation 拆开 | Agent讨论 + 实验计划        | 待研究   |
| `Q05-S6` | 哪些问题需要升级为理论推导或老师/同事讨论                                   | Agent讨论 + 理论推导 + 交流讨论 | 提出待分析 |
| `Q05-S7` | `soft-Linf` 与 multi-step PDE 到底是只有组合收益，还是各自都有净贡献 | 实验验证 + 深度思考 | 进行中 |
| `Q05-S8` | `batch_size` 是否是当前结论里的关键干扰变量 | 实验验证 + Agent讨论 | 待研究 |
| `Q05-S9` | 输入扰动 / ensemble RHS smoothing 是否值得做 | 实现设计 + 实验验证 | 待研究 |
| `Q05-S10` | 为什么 one-step 变好不等于 rollout 稳定，以及 Q05 相对 anchored baseline 还差在哪里 | 深度思考 + 实验验证 + 理论推导 | 进行中 |

### 2.1 当前技巧/疑问清单映射

| 条目 | 子问题 | 相关实验 | 当前状态 | 当前判断 / 当前缺口 |
| ---- | ---- | ---- | ---- | ---- |
| `Q05-L1` | rollout 多步后对中间态算 PDE loss，这类 `multi-step PDE` 目标有没有净收益 | `exp18`、`exp22`、`exp32`；侧向证据 `exp20 / exp21` | 进行中 | `exp22` 配置族明显强于 `exp18`，说明“更强局部 rollout 目标”有价值；`exp32` 已作为 clean no-multi-step ablation 启动，但在本地结果落盘前还不能把净收益单独归给 multi-step |
| `Q05-L2` | 除 `MSE / MAE` 外加 `soft-Linf` 是否有独立收益 | `exp18`、`exp22`、`exp31` | 进行中 | 当前只能说 `soft-Linf + multi-step` 这个组合配置族有净改进；`exp31` 已作为 clean no-soft-Linf ablation 启动，但本地结果仍待落盘 |
| `Q05-L3` | bootstrap 超参数如 `start / update_every / rollout_steps / teacher_source / rollout curriculum` 到底哪些有用 | `exp13`、`exp18`、`exp22`、`exp23`、`exp29`、`exp30` | 进行中 | 旧的“晚启动 + 短 rollout”很差，更积极 family 更强；但 clean 早晚启动还不能回答，`exp23` 仍在 pre-bootstrap；`exp29` 目前先暴露的是 launch 问题，`exp30` 还在等本地结果落盘 |
| `Q05-L4` | relative loss 能否避免大数值 field dominate；dynamic vs fixed 哪个更稳 | `exp22`、`exp24`、`exp26` | 部分回答 + 进行中 | `exp24` 已给出 dynamic-scale 负例；`exp26 step_2000` 已给出明显更强 early signal，但 fixed-scale 是否能稳定穿过后续窗口仍待观察 |
| `Q05-L5` | multi-step rollout 时中间态应 `detach` 还是 no-detach | `exp22`、`exp28` | 进行中 | `exp28` clean no-detach ablation 已上线并落 `step_0`；但 no-detach 只会在训练后体现，当前还不能判断更稳或更差 |
| `Q05-L6` | `rollout_dt` 应更小还是直接对齐 `dt_data=1.0` | `exp22`、`exp27` | 进行中 | `exp27` 已落 `step_0`，且初始 eval 更低；但因为 `10 NFE vs 100 NFE`、协议已变，不能把 `step_0` 直接当正结果 |
| `Q05-L7` | ensemble prediction / 输入扰动 / 平均 RHS target 是否有用 | 暂无；相邻代码钩子只有 `input_noise_scale` | 待新实现 + 新实验 | 当前代码只有“输入加噪、target 不变”的简单钩子，还没有 ensemble RHS averaging / window 聚合实现 |
| `Q05-L8` | `batch_size` 是否显著影响当前结论 | `exp13`、`exp18`、`exp22`、`exp23`、`exp24`、`exp26`、`exp27`、`exp28`、`exp29`、`exp30`、`exp31`、`exp32` | 待新实验 | `exp13 / exp18` 是 `bs=10`，`exp22+` 基本是 `bs=2`；当前几乎所有关键比较都被 batch size 污染，不能直接归因 |
| `Q05-L9` | 当前 best practical 路线是否仍是“GRF 预训练 + simulation data 继续训练” | `exp22` 与 anchored baselines | 已回答 | 是。`exp22` 是 best pure-online，但仍远差于 `exp_offline_pde (1GPU)` 与 `exp1b (1GPU)` |
| `Q05-L10` | 为什么 `train loss / step-1` 变好，但 multi-step rollout 变坏；为什么 one-step 可接近 anchored 但 rollout 仍爆 | `exp18`、`exp20`、`exp21`、`exp22` + `Q02` baselines | 现象已回答，机理部分回答 | 现象已稳定复现；当前更像支持集错配 + 长期误差传播问题，而不是单纯 one-step objective 不够好；但 teacher 是否真改善输入分布仍未证成 |


## 3. 提出待分析


| 条目       | 调研  | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态    | 当前说明                                                                                | 细节位置                  |
| -------- | --- | ---- | ---- | ---- | ------- | ------- | ----- | ----------------------------------------------------------------------------------- | --------------------- |
| `Q05-A1` | 辅   | 主    | 辅    | 辅    | 主       | 辅       | 提出待分析 | “teacher 更物理”目前只有 downstream rollout 的间接证据，缺直接 proxy；先要定义判据，再决定是否单开 `Q05` 专属调研或理论条目 | `深度思考.md`、`README.md` |
| `Q05-A2` | -   | 主    | 辅    | 辅    | 主       | 主       | 提出待分析 | 现有时机问题混着 `start / update_every / rollout_steps`，需要先把问题拆干净，再决定哪些必须拉老师或同事讨论           | `实验记录.md`、`深度思考.md`   |


## 4. 待研究


| 条目       | 调研  | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态  | 验收指标                                                                                          | 当前准备                            | 细节位置                     |
| -------- | --- | ---- | ---- | ---- | ------- | ------- | --- | --------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------ |
| `Q05-R1` | -   | 辅    | 主    | -    | 辅       | -       | 待研究 | 做一组只改 `self_training_start_step`、其余保持 `exp22` 协议一致的 clean ablation，才能单独回答上线时机问题               | 需求已经在 `深度思考.md` 明确，但当前还没有对应 run | `深度思考.md`、`实验记录.md`      |
| `Q05-R2` | -   | 辅    | 主    | -    | 主       | -       | 待研究 | 把 `exp22 / exp23 / exp24 / exp26 / exp27 / exp28 / exp29 / exp30 / exp31 / exp32` 按 `field / timestep / refresh` 系统归档，至少能回答“先爆哪个 field、refresh 后是否变稳” | 目前只有散落观察，还没有统一台账                | `实验记录.md`                |
| `Q05-R3` | 主   | 辅    | -    | -    | 辅       | -       | 待研究 | 如果 `exp23 / exp26 / exp27 / exp28 / exp29 / exp30 / exp31 / exp32` 仍不能区分“状态分布收益”“loss trick 收益”“teacher source 收益”“rollout curriculum 收益”“dt 对齐收益”“no-detach 反传收益”和“no-multi-step 目标收益”，就需要回上游问题补自举相关 failure mode 调研入口              | `Q05` 当前不单开调研文档，现阶段只保留升级条件      | `README.md`              |
| `Q05-R4` | -   | 辅    | 辅    | 主    | 辅       | 辅       | 待研究 | 若后续要正式回答“teacher 偏差回灌为什么在 rollout 放大”，需要决定是否从 `Q02` 独立拆出 `Q05` 专属推导                           | 当前理论解释仍借用 `Q02`，这里先不重复建设        | `README.md`、`../main.md` |
| `Q05-R5` | -   | 辅    | 主    | -    | 辅       | -       | 待研究 | 做一组只改 `soft_linf_weight`、其余保持 `exp22` 协议一致的 clean ablation，才能回答 `soft-Linf` 的独立净收益 | `exp31` 已在 `20260308215522 : t2` launch，当前等待本地 run 目录与首轮 log 落盘 | `实验记录.md`、`深度思考.md` |
| `Q05-R6` | -   | 辅    | 主    | -    | 辅       | -       | 待研究 | 做一组只改 `batch_size`、其余协议尽量一致的 clean ablation，才能判断当前结论里有多少是 batch-size effect | 当前 `exp13 / exp18` 与 `exp22+` 的关键比较都混着 `bs=10 vs 2` | `实验记录.md` |
| `Q05-R7` | -   | 辅    | 主    | -    | 主       | -       | 待研究 | 实现并比较“输入加噪”“ensemble RHS averaging”“windowed perturb target”等分支，至少回答哪种值得继续投入 | 当前只有 `input_noise_scale` 简单钩子，尚无专门 run | `实验记录.md`、`../main.md` |


## 5. 进行中

> 主体维护区。这里优先更新正在推进的回答、运行中的实验和最近一轮计划；长分析继续下沉到 `实验记录.md` 与 `深度思考.md`。

### 5.1 当前进行中的关键子问题


| 条目       | 调研  | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态  | 当前动作                                                                               | 当前可见结论                                                        | 验收信号                                                         | 细节位置                      |
| -------- | --- | ---- | ---- | ---- | ------- | ------- | --- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------- |
| `Q05-I1` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp23` 的 `step_100000` 激活窗口，以及激活后的第一批 eval / refresh                            | 当前只能确认 `exp23` 已正常进入 pre-bootstrap 阶段，不能提前写 later-start 有效或无效 | 至少拿到 `step_100000` 激活前后同口径对照，再和 `exp22` 的 `30000-40000` 窗口比较 | `实验记录.md`、`深度思考.md`       |
| `Q05-I2` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 把 `exp24` 归档为 dynamic-scale 失败分支，并把 `exp26` 归档为 fixed-scale early-signal 分支，保留其与 `exp24` 的首轮对照价值 | 当前 `exp24` 已在 `step_8000` 左右暴露明显失稳并停掉；`exp26` 已拿到 `step_2000: MHD 1.1631, GRF 1.0618`，首轮明显优于 `exp24 step_2000`，但 run 已手动停在约 `step_2400` | `exp26` 当前只能支撑“fixed-scale 在 early window 强于 dynamic-scale”的局部结论；是否能跨过后续窗口仍待未来补证           | `实验记录.md`、`深度思考.md`       |
| `Q05-I3` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 继续跟 `exp22` 后续 checkpoint 与 refresh，整理 `U / n / vpar` 的爆炸 pattern                  | 当前可以写成“最强纯 online”，但不能写成“已接近可用长期稳定 surrogate”                 | 在后续 checkpoint 中仍能保持相对优势，并补齐按 field / timestep 证据            | `README.md`、`实验记录.md`     |
| `Q05-I4` | -   | 辅    | 辅    | -    | 主       | 辅       | 进行中 | 用本控制面持续约束表述边界，把“已确认”和“待补证”分开，不让运行中实验被提前写成定论                                        | 当前边界已明确：`exp23 / exp27 / exp28` 仍只能写当前可见结果，`exp30 / exp31 / exp32` 当前只能写 launch_announced 或 startup-stage，`exp24` 已归档为 stopped negative branch，`exp26` 已归档为 early-signal stopped branch，`exp29` 当前是 launch 失败而非方法结论                          | 当 `exp23 / exp27 / exp28 / exp30 / exp31 / exp32` 出现可解释窗口后，再决定是否升级到待验收或回写 `main.md`          | `sub_main.md`、`README.md` |
| `Q05-I5` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp27` 的 `step_0 / 2000`，看 `rollout_dt=1.0` 是否比 `exp22` 的 `0.1 x 10 substeps` 更稳 | 当前已落 `step_0: MHD 1.0110, GRF 1.0458`；但因为这是 `10 NFE` 协议，不能把首轮数值直接当训练收益 | 至少拿到首轮 `step_2000` 对照，并能和 `exp22` 做同口径 `dt` 对齐比较 | `实验记录.md`、`深度思考.md` |
| `Q05-I6` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp28` 的 `step_0 / 2000`，看 `multi_step_pde_detach=0` 是否比 `exp22` 的 detach 版本更稳 | 当前已落 `step_0: MHD 1.8126, GRF 1.1483`；和 `exp26` 一样，这一步只说明 run 正常，no-detach 作用要到训练后才可能体现 | 至少拿到首轮 `step_2000` 对照，并能和 `exp22` 做同口径 detach vs no-detach 比较 | `实验记录.md`、`深度思考.md` |
| `Q05-I7` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 把 `exp29` 先从“launch 失败”修到“能正常完成初始评估并进入训练”，再看 best-teacher source 是否值得保留 | 当前本地已有 run_tag，但 `step_0` 前报 `No CUDA GPUs are available`；还没有任何可用方法结论 | 至少重新启动并拿到 `step_0 / 2000`，再和 `exp22` 做 latest-vs-best teacher 对照 | `实验记录.md`、`深度思考.md` |
| `Q05-I8` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp30` 的本地 run 目录与首轮 log，确认 rollout curriculum 分支是否真的启动成功 | 当前只有用户口头同步 launch；本地尚未看到结果目录或 log | 至少落盘 run 目录并拿到 `step_0 / 2000`，再和 `exp22` 做固定 5-step vs curriculum 对照 | `实验记录.md`、`深度思考.md` |
| `Q05-I9` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp31` 的本地 run 目录与首轮 log，确认 clean no-soft-Linf ablation 是否真的启动成功 | 用户已口头同步在 `20260308215522 : t2` launch；当前本地尚未看到结果目录或 log | 至少落盘 run 目录并拿到 `step_0 / 2000`，再和 `exp22` 做 soft-Linf vs no-soft-Linf 对照 | `实验记录.md`、`深度思考.md` |
| `Q05-I10` | -   | 主    | 主    | -    | 辅       | -       | 进行中 | 盯 `exp32` 的本地 run 目录与首轮 log，确认 clean no-multi-step ablation 是否真的启动成功 | 用户已同步 `H800 : t1` 所在 session 已切到 `exp32`；当前 terminal 头部已打印，但本地 run 目录与首轮 eval 尚未落盘 | 至少落盘 run 目录并拿到 `step_0 / 2000`，再和 `exp22` 做 multi-step vs no-multi-step 对照 | `实验记录.md`、`深度思考.md` |


### 5.2 Q05 运行实验 / 实验计划表


| run_id      | 对应子问题    | code git commit | 实验目的                                                            | 实现改动 / 脚本                                                                  | 实验机器（tmux session） | 结果文件位置                                                                                                                     | 预期结果和验收条件                                                 | 当前可见结果                                                                   | 当前分析                                         | 下一步计划                                       | 当前状态    |
| ----------- | -------- | --------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------- | ------- |
| `RUN-EXP22` | `Q05-S3` | 未记录             | 看 `exp18` 上叠加 `soft-Linf + multi-step PDE` 后，bootstrap 收益能否持续维持 | `run_exp22_self_training_linf_multistep.sh`                                | `H800 : t1-2`      | `results/exp22_self_training_linf_multistep/acad9e03-03_10_09_26_12-K4-mx128-w80-L12-od10/`                                | 后续 checkpoint 仍保持相对优势，并补齐 `field / timestep / refresh` 证据 | 当前已落盘到 `step_62000`；最新可见 eval `step_62000` 为 MHD `6.0249`、GRF `2.8387`，但长期 rollout 仍未彻底稳住           | 现阶段仍是最强纯 online anchor，且距离下一次 refresh 窗口较近               | 继续跟 `step_70000+` refresh 窗口，并做按 field 归档                  | running |
| `RUN-EXP23` | `Q05-S2` | 未记录             | 看 teacher 更晚上线是否更稳                                              | `run_exp23_self_training_linf_multistep_later100000step_start_boostrap.sh` | `H800 : t2`        | `results/exp23_self_training_linf_multistep_later100000step_start_boostrap/3c6a51c4-03_11_05_54_30-K4-mx128-w80-L12-od10/` | 至少比较 `step_100000` 激活前后，以及第一次 refresh 后是否比 `exp22` 更稳     | 当前已落盘到 `step_18000`；仍在 pre-bootstrap 阶段。最新可见 eval `step_18000` 为 MHD `10.9756`、GRF `5.9992` | 当前只能说明 pre-bootstrap 训练正常推进，不能回答 later-start 是否有效             | 重点盯 `100000-110000` 激活窗口，并和 `exp22` 激活窗口做对照 | running |
| `RUN-EXP24` | `Q05-S4` | 未记录             | 在 `exp22` 基础上加 dynamic relative PDE loss，看是否改善 field balance 与 rollout  | `run_exp24_self_training_linf_multistep_normed_MXE_loss.sh`                | `H800 : t2-2（已停）` | `results/exp24_self_training_linf_multistep_normed_mse/b0d48701-03_11_08_39_25-K4-mx128-w80-L12-od10/`                     | 至少拿到 `step_2000 / 10000` 的首轮同阶段对照，并确认改进不只停留在 early eval   | 已跑到 `step_8000`；`step_6000` 时 MHD `1.875`、GRF `1.199`，但 `step_8000` 已恶化到 MHD `8.486`、GRF `2.905`，且 `U` 爆炸明显 | 动态 target-RMS relative loss 没把 field balance 转成稳定 rollout，当前可暂判不 work | 停止实验，保留为 dynamic-scale 负例；后续主要看 `exp26` 是否能避开这类早期失稳                  | stopped |
| `RUN-EXP26` | `Q05-S4` | 未记录             | 在 `exp22` 基础上改成 fixed-scale relative PDE loss，直接比较 fixed vs dynamic 归一化 | `run_exp26_self_training_linf_multistep_fixed_normed_MXE_loss.sh`          | `H800 : t1（已停）`        | `results/exp26_self_training_linf_multistep_fixed_normed_mse/fd3e2a38-03_11_11_45_40-K4-mx128-w80-L12-od10/`               | 至少成功启动并拿到 `step_0 / 2000`，再与 `exp24` 做 fixed-vs-dynamic 首轮对照      | 已确认 run_tag `fd3e2a38-03_11_11_45_40-K4-mx128-w80-L12-od10`；已落 `step_2000`：MHD `1.1631`、GRF `1.0618`；训练在约 `step_2400` 后手动停止并给 `exp32` 让位 | 这仍是当前最强 early positive signal，但只覆盖 early window；不能提前外推成稳态主线结论 | 保留为 fixed-scale early-signal 归档，后续主要作为 `exp24` 的首轮反例对照 | stopped |
| `RUN-EXP27` | `Q05-S5` | 未记录             | 在 `exp22` 基础上改成 `rollout_dt=1.0`，直接测试模型步长与数据步长对齐是否更利于 bootstrap 稳定性 | `run_exp27_self_training_linf_multistep_model_dt1.0.sh`                    | `H800 : t2-2`      | `results/exp27_self_training_linf_multistep_model_dt1.0/1bc8517a-03_11_12_11_13-K4-mx128-w80-L12-od10/`                                                               | 至少落盘 run 目录并拿到 `step_0 / 2000`，再与 `exp22` 做 `rollout_dt=0.1 vs 1.0` 对照 | 已确认 run_tag `1bc8517a-03_11_12_11_13-K4-mx128-w80-L12-od10`；`step_0` 为 MHD `1.0110`、GRF `1.0458`（`10 NFE`）；当前训练约到 `step_1300` | `step_0` 数值更低，但因为评估协议已变成 `10 NFE`，不能直接写成更稳；当前优先级低于 `exp32 / exp22 / exp23 / exp28` | 先等 `step_2000`，再和 `exp22` 做 `dt` 对齐对照 | running |
| `RUN-EXP28` | `Q05-S5` | 未记录             | 在 `exp22` 基础上只改 `multi_step_pde_detach=0`，直接测试 no-detach 是否更利于 bootstrap 稳定性 | `run_exp28_self_training_linf_multistep_no_detach.sh`                      | `a100_dev : t`     | `results/exp28_self_training_linf_multistep_no_detach/80803b10-03_11_12_17_38-K4-mx128-w80-L12-od10/`                                                               | 至少落盘 run 目录并拿到 `step_0 / 2000`，再与 `exp22` 做 detach vs no-detach 对照 | 已确认 run_tag `80803b10-03_11_12_17_38-K4-mx128-w80-L12-od10`；`step_0` 为 MHD `1.8126`、GRF `1.1483`；当前训练约到 `step_600` | 这是 clean no-detach ablation，比较口径比 `exp27` 更干净；若 A100 不紧张，值得继续保留到首轮 eval | 先等 `step_2000`，再和 `exp22` 做 detach vs no-detach 对照 | running |
| `RUN-EXP29` | `Q05-S3` | 未记录             | 在 `exp22` 基础上改成 `self_training_teacher_source=best`，直接测试 latest-vs-best teacher source | `run_exp29_self_training_linf_multistep_best_teacher.sh`                    | `20260308215522 : t1`      | `results/exp29_self_training_linf_multistep_best_teacher/7ada96df-03_11_12_48_41-K4-mx128-w80-L12-od10/`                                                               | 至少完成初始评估并进入训练，再与 `exp22` 做 latest-vs-best teacher 对照 | 已确认本地 run_tag，但初始评估窗口报 `RuntimeError: No CUDA GPUs are available` | 当前不能算有效运行中的实验；先修 launch，再讨论方法结论 | 先确认 GPU 可见性或 `GPU_IDS` 设置，再决定是否原样重启 | launch_failed |
| `RUN-EXP30` | `Q05-S5` | 未记录             | 在 `exp22` 基础上给 teacher rollout depth 加 curriculum，测试固定 5-step vs curriculum 是否更稳 | `run_exp30_self_training_linf_multistep_rollout_curriculum.sh`                      | `a100_dev : session 未记录`     | `results/exp30_self_training_linf_multistep_rollout_curriculum/`                                                               | 至少落盘 run 目录并拿到 `step_0 / 2000`，再与 `exp22` 做固定 5-step vs curriculum 对照 | 用户已口头同步 launch；当前本地尚未看到结果目录或 log | 当前只能记录为 launch announced，不能写任何效果判断 | 先等本地 run 目录与第一批 log 落盘，再补 run_tag 和首轮状态 | launch_announced |


| `RUN-EXP31` | `Q05-S7` | 未记录             | 在 `exp22` 基础上只去掉 `soft-Linf`，直接测试 `soft-Linf` 是否有独立净收益 | `run_exp31_self_training_multistep.sh`                      | `20260308215522 : t2`     | `results/exp31_self_training_multistep/`                                                               | 至少落盘 run 目录并拿到 `step_0 / 2000`，再与 `exp22` 做 soft-Linf vs no-soft-Linf 对照 | 用户已口头同步 launch；当前本地尚未看到结果目录或 log | 这是当前最关键的 clean `soft-Linf` ablation；在本地结果落盘前不能写任何效果判断 | 先等本地 run 目录与首轮 log 落盘，再与 `exp22` 做 clean 对照 | launch_announced |
| `RUN-EXP32` | `Q05-S7` | 未记录             | 在 `exp22` 基础上只关掉 `multi-step PDE`，直接测试 `multi-step PDE` 是否有独立净收益 | `run_exp32_self_training_linf.sh`                      | `H800 : t1`     | `results/exp32_self_training_linf/`                                                               | 至少落盘 run 目录并拿到 `step_0 / 2000`，再与 `exp22` 做 multi-step vs no-multi-step 对照 | 用户已同步 `H800 : t1` 所在 session 已切到 `exp32`；当前 terminal 头部已打印，但本地 run 目录与首轮 eval 尚未落盘 | 这是当前最关键的 clean `multi-step` ablation；在本地结果落盘前不能写任何效果判断 | 先等本地 run 目录与首轮 log 落盘，再与 `exp22` 做 clean 对照 | launch_announced |

### 5.3 Q05 实验优先级队列

#### 5.3.1 当前占卡优先级

| 优先级 | 实验 | 主要回答子问题 | 为什么排这里 | 占卡建议 | 降级 / 停止条件 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| `P0-0` | `exp29`（先修 launch） | `Q05-S3` | 这条 best-teacher 分支当前最大问题不是方法本身，而是还没真正跑起来；如果不先修掉，`20260308215522` 这张卡就是空转 | 优先修复并重启；不占用主 `H800` 队列 | 至少修到能完成 `step_0` 并进入训练；若仍反复卡在环境问题，再临时降级 |
| `P0-1` | `exp22` | `Q05-S3` | 它仍是当前最强 pure-online anchor，且距离下一次 refresh 窗口比 `exp23` 更近；所有新分支都需要拿它做主对照 | 高优先级；`2x H800` 尽量保留到下一次 refresh 后 | 至少盯到 `step_70000+`；若后续窗口彻底失守，再考虑让位 |
| `P0-2` | `exp23` | `Q05-S2` | 虽然回报慢，但这是唯一已经在路上的 later-start 分支；现在停掉会把长等待成本直接作废 | 高优先级；`2x H800` 除非极端缺卡不建议停 | 至少保留到 `step_100000-110000` 激活窗口 |
| `P0-3` | `exp32` | `Q05-S7` | 这是刚接手 `H800 : t1` 的 clean no-multi-step ablation，正对 `multi-step PDE` 独立贡献这个关键未答问题 | 先等本地 run 目录与首轮 log 落盘；若正常启动，优先保留到 `step_2000` | 至少拿到 `step_0 / 2000`；若长时间仍不落盘，则先不把它算作有效占卡 |
| `P1-4` | `exp31` | `Q05-S7` | 这是已经 launch 的 clean no-soft-Linf ablation，正对 `soft-Linf` 独立贡献这个关键未答问题；而且占的是 `A100`，不直接挤压主 `H800` 队列 | 先等本地 run 目录与首轮 log 落盘；若正常启动，优先保留到 `step_2000` | 至少拿到 `step_0 / 2000`；若长时间仍不落盘，则先不把它算作有效占卡 |
| `P1-5` | `exp28` | `Q05-S5` | 这是 clean no-detach ablation，比较口径比 `exp27` 更干净；而且它占的是 `A100`，不直接挤压主 `H800` 队列 | 有空 `A100` 就继续；若 `A100` 出现更高优先任务可让位 | 至少拿到 `step_2000`，最好看到 `step_10000` |
| `P1-6` | `exp30` | `Q05-S5` | rollout curriculum 是协议层新分支，而且也走 `A100`；但当前本地结果还没落盘，所以优先级先放在 `exp31 / exp28` 之后 | 先确认本地落盘，再决定是否长期保留 | 至少拿到 run 目录与 `step_0 / 2000`；若长时间不落盘，则先不把它算作有效占卡 |
| `P2-7` | `exp27` | `Q05-S5` | 有潜力回答 `dt` 对齐问题，但 `10 NFE` 协议变化让当前结论不够干净；在 `H800` 紧张时它是当前第一停牌候选 | 只在 `H800` 富余时继续；若要腾 `H800`，优先停它 | 至少拿到 `step_2000`；若后续仍无法和 `exp22` 同口径解释，则归档为次级分支 |

#### 5.3.2 新实验启动顺序

| 顺位 | 建议实验 | 只改什么 | 主要回答子问题 | 为什么先后这样排 | 推荐卡位 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| `N1` | clean `batch_size` ablation | 以 `exp22` 或 `exp26` 主协议为底，只改 `batch_size` | `Q05-S8` | `exp31 / exp32` 已经分别承接了 clean `soft-Linf` 与 clean `multi-step` ablation；下一步最该补的是长期污染结论的 `batch_size` | `H800`；如果显存受限，可优先围绕当前主线做最小改动版本 |
| `N2` | clean start-step ablation | 以 `exp22` 为底，只改 `self_training_start_step`，不要同时改 `update_every` | `Q05-S2` | `exp23` 是必要的长线，但不是 clean 对照；要真正回答“早开还是晚开”，还得补这一组 | `H800` |
| `N3` | `input_noise_scale` 轻量分支 | 用现有输入加噪钩子，先做 target 不变的 cheap test | `Q05-S9` | 这是进入 perturb/ensemble 路线前成本最低的探针；如果连它都没有正信号，就不急着上更复杂实现 | `A100` 优先 |
| `N4` | ensemble RHS averaging / windowed target | 新实现后再开 | `Q05-S9` | 工程成本最高，也最容易把“更平滑 target”与“更多实现改动”绑在一起；应放在前面几类 clean ablation 之后 | 最后再占卡 |

### 5.4 当前非实验推进动作


| 类型        | 当前动作                                            | 目标                   | 指向文档          |
| --------- | ----------------------------------------------- | -------------------- | ------------- |
| 深度思考      | 持续约束“teacher 更物理”和“loss 更平衡”这两个问题不要混写           | 避免把局部训练收益误写成状态分布收益   | `深度思考.md`     |
| Agent讨论   | 用 `sub_main.md` 管理 `Q05` 的回答推进和状态迁移             | 让 `Q05` 的进行中问题有统一控制面 | `sub_main.md` |
| 老师 / 同事讨论 | 暂未正式启动，等 clean ablation 或 direct proxy 仍不清楚时再升级 | 不在证据不足时过早发散到外部讨论     | `sub_main.md` |


## 6. 待验收


| 条目       | 调研  | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态  | 验收标准                                                             | 计划来源                |
| -------- | --- | ---- | ---- | ---- | ------- | ------- | --- | ---------------------------------------------------------------- | ------------------- |
| `Q05-C0` | -   | 辅    | 辅    | -    | 主       | -       | 待验收 | 当前暂无可独立验收的新增条目；待 `exp23` 过 `step_100000`，或 `exp27 / exp28 / exp30 / exp31 / exp32` 至少出现 `step_2000` 同阶段对照，或 `exp29` 至少修到完成初始评估后再回填 | `实验记录.md`、`深度思考.md` |


## 7. 已完成


| 条目ID     | 结论                                                                                   | 调研  | 深度思考 | 实验验证 | 理论推导 | Agent讨论 | 老师/同事讨论 | 状态  | 论据概述                                               | 细节位置                  |
| -------- | ------------------------------------------------------------------------------------ | --- | ---- | ---- | ---- | ------- | ------- | --- | -------------------------------------------------- | --------------------- |
| `Q05-D1` | `self-training / bootstrap` 已可确认是当前最强的纯 online 路线                                    | -   | 辅    | 主    | -    | 辅       | -       | 已完成 | `exp22` 当前汇总结果同时优于 `exp13 / exp18 / exp20 / exp21` | `README.md`、`实验记录.md` |
| `Q05-D2` | `exp13 -> exp18 -> exp22` 说明更积极的 bootstrap 配置族叠加 `soft-Linf + multi-step PDE` 确实有净改进 | -   | 辅    | 主    | -    | 辅       | -       | 已完成 | 旧 baseline 很差，`exp18` 明显拉回，`exp22` 再进一步            | `README.md`、`实验记录.md` |
| `Q05-D3` | 当前不能把 `Q05` 的改善写成“teacher 已学会生成物理态”                                                  | -   | 主    | 辅    | 辅    | 主       | 辅       | 已完成 | 现有证据仍无法区分“状态分布收益”和“loss trick 收益”                  | `深度思考.md`、`README.md` |


