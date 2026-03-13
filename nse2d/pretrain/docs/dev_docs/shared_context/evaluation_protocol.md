# Evaluation Protocol and Baselines

## 1. 当前默认评估口径

- 主指标：Per-channel Mean Relative L2 Error
- 统计方式：
  - 先对每个 field 计算 relative L2
  - 再对 5 个 field 取平均
  - 再对测试轨迹 batch 取平均
- 常用汇报步：
  - rollout step `1 / 3 / 5 / 10`
  - `Mean` 默认是这 4 个步的平均
- 常用测试集：
  - MHD test：真实仿真测试集
  - GRF test：GRF 初始化测试集

## 2. 当前协议 caveat

- 当前 `eval_data_path` 还同时承担 validation 和 final test 角色。
- 这意味着当前 best checkpoint 选择并不完全干净。
- 研发阶段可以接受，但论文级别结论前必须拆分成 `val` 和 `test`。

## 3. 当前最应优先信任的 baseline

这里不单按数值最低排序，而是按“协议干净度 + 可解释性 + 结果表现”综合判断。

同时要特别记住：

- 当前“效果最好”的经验路线并不等于 strict data-free 主问题已经被解决。
- `GRF 预训练 + simulation data 继续训练` 的成功，更像是在说明真实状态锚点仍然关键。

### 第一梯队：当前最可信的锚定 baseline

- `exp_offline_pde (PDE loss, 1GPU)`
  - MHD Mean：`0.4097`
  - GRF Mean：`1.060`
  - 价值：label-free / physics-only / 真实状态锚定，适合做 strict data-free 的近邻参照

- `exp1b (sup dt=1s, 1GPU)`
  - MHD Mean：`0.4099`
  - GRF Mean：`1.030`
  - 价值：监督基线，训练/评估时间尺度相对更干净，适合作为上界参照之一

### 第二梯队：结果强但解释需谨慎

- `exp3 (physics+supervised)`
  - MHD Mean：`0.3991`
  - GRF Mean：`1.191`
  - caveat：监督时间尺度需要重新核实，未清理前不应当直接当最强硬结论

- `mhd5_staged_v1`
  - MHD Mean：`0.2903`
  - GRF Mean：`39.16`
  - caveat：MHD 很强，但 GRF 极差，说明它更像“offline rescue”而不是对 strict data-free 有直接帮助

## 4. 当前 online 强化线的现实表现

以下实验共同说明：在当前协议和当前状态生成器下，仅靠 raw online 增强项还没有把问题根本救回来。

| 实验 | MHD Mean | GRF Mean | 状态 | 关键结论 |
|------|----------|----------|------|----------|
| `exp14` learnable GRF | `301.3` | `346.0` | stopped | 典型高价值失败样本 |
| `exp18` self-training v2 | `17.37` | `16.13` | stopped | 比更早自训练略好，但仍明显失稳 |
| `exp20` learnable GRF + soft-Linf | `22.21` | `22.67` | stopped | 比 `exp14` 好很多，但仍远差于 offline anchor |
| `exp21` learnable GRF + soft-Linf + multi-step | `25.72` | `27.49` | stopped | multi-step 没有把 learnable GRF 路线救回来 |
| **`exp22`** self-training + soft-Linf + multi-step | **`11.54`** | **`11.51`** | running | **当前最强 pure-online**，但 step-10 仍失稳 |
| `exp23` later-start bootstrap | `10.36` | `10.32` | running | pre-bootstrap 阶段，待激活后评估 |
| `exp24` dynamic relative loss | `20.54` | `21.12` | stopped | dynamic-scale 不 work，已判负 |
| `exp26` fixed-scale relative | `1.23` | `1.44` | stopped | **最强 early signal**，但仅到 step-2400 |
| `exp29` best-teacher source | `54.43` | `62.94` | running | 当前表现不佳，待观察 |
| `exp30` rollout curriculum | `8.89` | `10.30` | running | 有潜力，GRF 略优于 exp22 |
| `exp31` no-soft-Linf ablation | `1.66e8` | `1.60e8` | stopped | **soft-Linf 关键证据**，无则灾难性失稳 |
| `exp32` no-multi-step ablation | `11.58` | `11.22` | running | 略差于 exp22，multi-step 有净收益 |
| `exp33` ensemble RHS smoothing | `9.66` | `8.32` | running | GRF 优于 exp22，有潜力 |
| `exp34` ensemble E=10 | `15.85` | `11.52` | running | 强度增加未改善，可能过强 |

### 关键观察

1. **soft-Linf 是关键组件**：`exp31` (no-soft-Linf) 在 step-20000 前即灾难性爆炸，证明 soft-Linf 对 pre-bootstrap 稳定性至关重要。
2. **multi-step 有独立净收益**：`exp32` (no-multi-step) 略差于 `exp22`，说明 multi-step PDE 有独立贡献。
3. **fixed-scale relative loss 有潜力**：`exp26` early window 表现极佳，但稳定性待验证。
4. **ensemble smoothing 值得继续探索**：`exp33` GRF 指标优于 `exp22`，且 vpar 场稳定性明显改善。

## 5. 使用这份口径时的注意事项

- 先看协议是否一致，再看数值高低。
- 单卡与多卡实验不总是严格可比，因为：
  - `batch_size` 是每卡 batch
  - global batch 不同
  - offline 多卡历史上还存在没有 `DistributedSampler` 的问题
- 不要只看 MHD test，还要同时看 GRF test。
- 不要只看 step-1，要重点看 step-10 和 mean。

## 6. 推荐的后续使用方式

- 做任何新开发前，先在 `main.md` 和对应 `Qxx/实验记录.md` 中明确：
  - 该实验对比哪个 baseline
  - 预期提升的是 MHD、GRF，还是两者兼顾
  - 该实验是在“协议清理”还是“状态先验”哪条线
- 任何新结果进入总结前，先和这份协议文件比对口径是否一致。
