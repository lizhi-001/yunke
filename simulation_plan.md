# 高维与低秩 VAR 已知断点检验 — 新版仿真实验方案

## 一、方案目标

本轮仿真实验不再将第一类错误与功效混合在同一块中汇报，而是改为两条独立主线：

1. **Size 主线**：单独汇报不同场景、不同模型、不同 Monte Carlo 重复次数 `M` 下的 size（即 H0 为真时的拒绝率）。
2. **Power 主线**：单独汇报不同场景、不同模型、不同变化幅度下的 power（即 H1 为真时的拒绝率），并明确回答：**随着时间序列结构变化幅度增大，power 是否提升。** Power 固定在最大的 `M_max` 上播报，不再随 `M` 变化。

同时保留 baseline 的 p 值比较模块，作为有限样本下渐近推断与 bootstrap 推断差异的辅助说明。

---

## 二、统一假设框架

### 2.1 已知断点检验问题

给定一个 VAR(p) 时间序列 `Y_t`，以及一个**已知**候选断点位置 `t`，检验该时点前后系数矩阵是否发生结构变化。

统一假设写为：

```text
H0: Φ1 = Φ2
H1: Φ1 ≠ Φ2
```

其中：
- `H0` 表示全样本结构稳定；
- `H1` 表示在给定断点 `t` 处出现结构变化；
- 本方案只讨论**已知断点检验**，不包含断点搜索。

### 2.2 baseline 的 Chow / LR 口径

baseline 场景使用同一有效样本上的 Chow / LR 形式：

```text
受限模型:    y_t = B x_t + u_t
非受限模型:  y_t = B x_t + Γ(D_t x_t) + u_t
```

其中 `D_t = 1(t >= t0)`。

在该口径下并行输出：
- `asym-F`
- `asym-χ²(LR)`
- `bootstrap-F`
- `bootstrap-LR`

baseline 的定位仍然是：
- 给出低维基准；
- 对比渐近推断与 bootstrap 推断；
- 为高维结果提供参照。

---

## 三、场景设置

本轮实验统一使用 `p = 3`，并保留三类场景：

| 场景 | 模型 | 结构假设 | 定位 |
|---|---|---|---|
| baseline | OLS + Chow/LR | 无约束低维 VAR | 基准对照 |
| sparse | Lasso + bootstrap LR | 稀疏高维 VAR | 核心高维场景一 |
| lowrank | SVD / 低秩 bootstrap LR | 低秩高维 VAR | 核心高维场景二 |

建议默认维度如下：

| 场景 | N | T | p | 断点 t |
|---|---:|---:|---:|---:|
| baseline | 2 | 150 | 3 | 75 |
| sparse | 3 | 150 | 3 | 75 |
| lowrank | 4 | 150 | 3 | 75 |

说明：
- 三个场景统一使用 `T=150` 与断点 `t=75`，以保证横向比较口径一致；
- baseline 维度较小，用于稳定比较 Chow/F/LR 各种推断方式；
- sparse 与 lowrank 保持相同 `p=3` 口径，但用不同结构约束完成高维推断。

---

## 四、数据生成原则

### 4.1 H0 数据生成

在 size 主线中，所有数据都从无断点 VAR 生成：

```text
Y_t ~ VAR(p; Φ1, Σ)
```

此时理论上不应拒绝 `H0`，因此拒绝率应接近显著性水平 `α`。

### 4.2 H1 数据生成

在 power 主线中，数据从含已知断点的 VAR 生成：

```text
断点前: Y_t ~ VAR(p; Φ1, Σ)
断点后: Y_t ~ VAR(p; Φ2, Σ)
```

其中 `Φ2` 由 `Φ1` 加上统一变化幅度构造：

```text
Φ2 = ensure_stationary(Φ1 + Δ · 1)
```

这里：
- `Δ` 表示变化幅度；
- `ensure_stationary()` 表示若新矩阵不平稳，则按统一缩放规则收缩至平稳区；
- 最终同时记录名义变化幅度 `change_scale` 与实际变化强度（如 `delta_fro`、`delta_max_abs`）。

### 4.3 至少 5 档变化幅度

本轮 power 试验至少固定 5 档变化幅度，默认建议：

```text
Δ-grid = {0.04, 0.08, 0.12, 0.16, 0.20}
```

后续若正式版算力允许，可额外扩展到：

```text
Δ-grid = {0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24}
```

本轮正式最少要求：
- 不少于 5 档；
- 从弱变化到强变化单调排列；
- 能够回答“power 是否随着变化幅度增大而提升”。

---

## 五、试验主线 A：Size 单独汇报

### 5.1 目标

单独估计不同场景、不同模型在 `H0` 下的 size：

```text
Size = P(reject H0 | H0 true)
```

### 5.2 设计

对于每个场景、每个模型、每个 `M`：

1. 固定一个 seed；
2. 生成 `M` 次无断点序列；
3. 每次计算检验 p 值；
4. 统计 `p <= α` 的比例；
5. 得到该 seed 下的单次 size 估计；
6. 跨 seed 仅汇总 size 的均值，不再报告 95%CI。

### 5.3 汇报口径

size 结果按下列维度单独列表：

| 场景 | 模型 | M | Size mean |
|---|---|---:|---:|

必须能直接回答：
- baseline 的 size 是否接近显著性水平；
- sparse / lowrank 的 size 是否偏保守或偏激进；
- 当 `M` 增大时，size 估计是否更稳定。

### 5.4 建议 M-grid

建议使用：

```text
M-grid = {50, 100, 200}
```

若为正式大规模版，可使用：

```text
M-grid = {300, 500, 800}
```

原则：
- size 主线的 `M` 用于控制 Monte Carlo 误差；
- `M` 越大，size 估计越稳定；
- `M` 不改变检验本身，只改变估计精度。

---

## 六、试验主线 B：Power 单独汇报

### 6.1 目标

单独估计不同变化幅度下的 power：

```text
Power(Δ) = P(reject H0 | H1 true under change scale Δ)
```

重点不是只给一个单点功效，而是回答：

```text
当变化幅度 Δ 变大时，Power 是否上升？
```

### 6.2 设计

对于每个场景、每个模型、每个变化幅度 `Δ`：

1. 固定一个 seed；
2. 构造 `Φ1`；
3. 用给定变化幅度构造 `Φ2 = ensure_stationary(Φ1 + Δ·1)`；
4. 固定使用 `M-grid` 中最大的 `M_max` 生成 `M_max` 次含断点序列；
5. 每次计算 p 值；
6. 统计 `p <= α` 的比例；
7. 得到该 seed 下该幅度对应的 power；
8. 跨 seed 再做均值与区间汇总。

### 6.3 汇报口径

power 结果按下列维度单独列表：

| 场景 | 模型 | change scale | 固定 M_max | Power mean [95%CI] |
|---|---|---:|---:|---:|

并需额外给出文字判断：
- 在固定 `M_max` 下，变化幅度从最小档增至最大档时，power 是否整体提升；
- 哪些模型对弱变化更敏感；
- 哪些模型只有在中高变化幅度下才出现明显 power。

### 6.4 核心判读标准

若某模型在固定 `M_max` 下满足：

```text
Power(Δ1) <= Power(Δ2) <= ... <= Power(Δ5)
```

则说明该模型的检验能力随结构变化增强而提升，结果符合预期。

若出现明显非单调情形，则需检查：
- `M` 是否太小；
- seed 数是否太少；
- 变化幅度虽然名义上变大，但 `ensure_stationary` 收缩后实际增量是否被压缩；
- 检验器在该场景下是否本身较保守。

---

## 七、跨 seed 汇总规则

本轮统一采用“先按单个 seed 计算，再跨 seed 汇总”的两层结构。

### 7.1 单个 seed 内

- size：

```text
size_seed = 拒绝 H0 的次数 / H0 下成功模拟次数
```

- power：

```text
power_seed(Δ) = 拒绝 H0 的次数 / H1(Δ) 下成功模拟次数
```

### 7.2 跨 seed 层

对于相同的：
- `(scenario, model, M)`，汇总 size；
- `(scenario, model, change_scale)`，汇总 power，其中 `M` 固定为 `M_max = max(M-grid)`。

本轮仅尝试 `3` 个 seed，因此汇报口径调整为：
- `size`：只播报 `mean across seeds`，不再报告 `95%CI`；
- `power`：重点看不同变化幅度下的均值变化趋势，必要时再补充区间。

### 7.3 seed 数建议

本轮固定：
- `n_seed = 3`

原因：
- 当前主要目标是先看 size 水平与 power 随变化幅度的单调趋势；
- 只有 3 个 seed 时，`95%CI` 的解释价值有限，尤其不适合放在 size 主表中；
- 因此本轮优先播报均值，后续如需正式统计区间，再扩展到更多 seed。

---

## 八、baseline p 值比较模块

该模块保留，但定位调整为**辅助分析**，不再与 size / power 混在同一部分汇报。

### 8.1 目的

在同一条 baseline 无断点序列上比较：
- 渐近 `χ²(LR)` p 值
- `bootstrap-LR` p 值
- `bootstrap-F` p 值

### 8.2 关注点

重点观察：
- 随 `B` 增大，bootstrap p 值是否稳定；
- bootstrap-LR 与 asym-χ²(LR) 是否接近；
- 有限样本下 bootstrap 修正是否明显。

### 8.3 推荐 B-grid

```text
B-grid = {100, 300, 500}
```

正式版可用：

```text
B-grid = {200, 500, 1000}
```

---

## 九、输出文件与报告结构

### 9.1 原始与聚合输出

本轮输出应至少包括：

- `baseline_raw_csv`
- `baseline_agg_csv`
- `size_raw_csv`
- `size_agg_csv`
- `power_raw_csv`
- `power_agg_csv`
- `baseline_png`
- `size_png`
- `power_png`
- `summary_json`
- 中文主报告
- 中文详细报告

### 9.2 中文报告结构

新的中文报告结构建议固定为：

1. 配置
2. baseline p 值比较
3. **不同情况下的 Size mean**
4. **固定最大 M 下不同变化幅度的 Power**
5. 数据背景说明
6. 输出文件

其中：
- 第 3 节只放 size mean；
- 第 4 节只放固定最大 `M` 下的 power；
- 第 4 节必须明确讨论“power 是否随变化幅度上升”。

---

## 十、实现映射

本轮方案对应的实现文件建议如下：

| 功能 | 文件 |
|---|---|
| 主实验脚本 | `experiments/run_next_stage_large_scale_p3.py` |
| 中文详细报告生成 | `experiments/write_detailed_next_stage_report.py` |
| baseline Chow 检验 | `simulation/chow_test.py` |
| baseline bootstrap 推断 | `simulation/chow_bootstrap.py` |
| 稀疏 bootstrap 推断 | `sparse_var/sparse_bootstrap.py` |
| 低秩 bootstrap 推断 | `lowrank_var/lowrank_bootstrap.py` |

主脚本中应明确包含：
- `size` 单独计算函数；
- `power` 单独计算函数；
- `power-change-grid` 参数；
- `size` / `power` 独立 CSV 与图表输出。

---

## 十一、正式版建议参数

若进入正式运行，本轮统一采用 3 个 seed：

```text
seeds = {42, 2026, 7}
M-grid = {300, 500, 800}
Δ-grid = {0.04, 0.08, 0.12, 0.16, 0.20}
B_mc_baseline = 80
B_mc_highdim = 100
B-grid = {200, 500, 1000}
```

如果算力受限，可先使用：

```text
seeds = {42, 2026, 7}
M-grid = {100, 200, 300}
Δ-grid = {0.04, 0.08, 0.12, 0.16, 0.20}
B_mc_baseline = 50
B_mc_highdim = 50
B-grid = {100, 300, 500}
```

---

## 十二、验收标准

本轮方案修改完成后，应满足以下验收条件：

### 12.1 结构要求

- [ ] size 与 power 完全分开汇报；
- [ ] power 至少包含 5 档变化幅度；
- [ ] baseline / sparse / lowrank 三类场景均有独立 size 结果；
- [ ] baseline / sparse / lowrank 三类场景均有独立 power 结果；
- [ ] 报告中能直接看出不同 `M` 下的 size mean；
- [ ] size 主表不再强制展示 95%CI；
- [ ] 报告中能直接看出固定最大 `M` 下不同变化幅度的 power。

### 12.2 分析要求

- [ ] 明确讨论 size mean 是否接近 `α`；
- [ ] 明确讨论 power 是否随变化幅度增大而提升；
- [ ] 若未提升，要在报告中解释可能原因；
- [ ] 不再使用“单一 power 数值”代表整个 H1 表现。

### 12.3 输出要求

- [ ] `summary_json` 中分别保存 `size_agg_rows` 与 `power_agg_rows`；
- [ ] 图文件分别对应 `size vs M` 与 `power vs change scale (fixed max M)`；
- [ ] 中文主报告中 size 主表默认只播报 mean；
- [ ] 中文主报告与中文详细报告都采用新结构。

---

## 十三、一句话总结

**新版方案的核心变化是：把“是否控 size”与“是否有 power”拆开看，并把 power 从单点检验改成“随变化幅度变化的整条响应曲线”来评估。**
