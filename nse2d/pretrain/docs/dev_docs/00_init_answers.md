# 00 Init Answers

复制到新项目后，先回答这份问答，再开始填 `main.md`。

## 基础信息

- 项目名：`OmniFluids/nse2d/pretrain` 2D 5-field MHD surrogate training
- 仓库路径：`/zhangtao/project2026/OmniFluids/nse2d/pretrain`
- 当前阶段：协议清理 + baseline 固化 + strict data-free 主问题聚焦
- 负责人：用户本人
- 主要协作 Agent：Cursor / Agent mode

## 目标

- 根问题：在尽量不依赖真实未来轨迹监督的前提下，训练出对 5-field MHD 长时 rollout 稳定的 surrogate model。
- 预期交付：
  - 一套可解释的 baseline 体系
  - 一套能持续回写的 `dev_docs`
  - 一条更物理的 strict data-free 状态先验主线
- 验收指标：
  - MHD / GRF 测试集 step-1/3/5/10 的 per-channel mean relative L2
  - train / inference / eval 协议一致
  - val / test 分离
- 当前不做什么：
  - 不把当前结果直接当论文最终结论
  - 不在协议未清前继续无限堆 online trick
  - 不把跨参数泛化当成本阶段主目标

## 资源和约束

- 算力：单卡和多卡都可用，近期主线常见 1-4 GPU
- 人力：用户主导，Agent 协作
- 主要工具：`/zhangtao/envs/rae/bin/python`、`accelerate`、`tmux`、Markdown 报告
- 时间窗口：当前以阶段性研究推进和实验迭代为主
- 预算或资源上限：数据体量很大，必须优先 `mmap`，多步实验显存压力高
- 必须升级给人类的情况：
  - 研究口径需要重新定义
  - 文档主问题要新增或拆分
  - 实验资源 / session / 机器分配需要协调

## 当前默认约束

- 主问题目录命名：`Qxx_问题短名/`
- `main.md` 只写当前活跃面：是
- `Qxx/README.md` 为问题入口：是
- 其余问题文件按需创建：是
- `实验记录.md` 默认单文件：是
- `runs/` 按需启用：是
- `shared_context/` 使用索引和适用范围约束：是

## 初始主问题

| QID | 问题 | 类型 | 为什么现在做 |
|-----|------|------|--------------|
| Q01 | 物理随机态生成与彻底 data-free 训练 | 调研 / 实验 / 理论 / 协议清理 | 这是当前项目最核心、也最容易因为信息散落而失控的问题 |

## 备注

- 与当前项目最相近的旧项目或旧实验：
  - `exp_offline_pde`
  - `exp1b`
  - `mhd5_staged_v1`
  - `exp18 / exp20 / exp21 / exp22`
- 最容易失控的点：
  - 把协议问题和科学问题混在一起
  - 只看单张图或单个数值，不回看完整日志与评估口径
  - 更新了实验与判断，但没有回写 `Q01/README.md` 和 `main.md`
- 希望 Agent 特别注意的协作习惯：
  - 先写清“本轮是在修协议、做 baseline，还是推进 strict data-free 主线”
  - 所有高价值实验都要有结果路径和结论落点

