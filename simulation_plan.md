# 已知断点 VAR 检验 — 当前代码版本仿真方案

## 一、方案定位

本方案以当前代码实现为准，围绕 `experiments/run_large_scale_current_plan.py` 组织一次**已知断点、单脚本、单轮输出**的仿真实验。

当前版本的目标不是做多阶段论文级大汇总，而是：

1. 在统一脚本下同时评估 `baseline_ols`、`sparse_lasso`、`lowrank_svd` 三类模型；
2. 分别报告每个模型在 `H0` 下的 `size / type I error`；
3. 在一组给定 `delta` 网格上报告 `power(delta)`；
4. 检查 `power` 是否随 `delta` 增大而整体上升；
5. 利用新的并行与向量化实现，把实验控制在可接受时间内。

---

## 二、当前实现对应的主脚本

当前试验方案直接对应：

- 主脚本：`experiments/run_large_scale_current_plan.py`
- baseline 蒙特卡洛：`simulation/monte_carlo.py`
- 稀疏蒙特卡洛：`sparse_var/sparse_monte_carlo.py`
- 低秩蒙特卡洛：`lowrank_var/lowrank_monte_carlo.py`
- 向量化设计矩阵：`simulation/design_matrix.py`
- 并行执行助手：`simulation/parallel.py`

当前版本已经支持：

- 外层 Monte Carlo 并行；
- 设计矩阵向量化构造；
- 稀疏 Lasso 路径缓存复用；
- 命令行参数 `--jobs` 控制并行核数。

---

## 三、统一检验框架

### 3.1 检验问题

对一个 **已知断点** `t` 的 VAR(p) 序列，检验断点前后系数是否变化：

```text
H0: Φ1 = Φ2
H1: Φ1 ≠ Φ2
```

当前代码版本同时包含两类已知断点口径：

- `baseline_ols`：低维 OLS + bootstrap LR at point；
- `sparse_lasso`：稀疏 Lasso + bootstrap LR at point；
- `lowrank_svd`：低秩 SVD + bootstrap LR at point。

注意：

- 当前主脚本的 `baseline_ols` 与高维模型一样，都是走**已知断点处的 bootstrap 检验**；
- 当前脚本**不包含** baseline 的 `asym-F / asym-χ² / bootstrap-F / bootstrap-LR` 并列输出；
- 当前脚本的重点是统一比较三类模型的 `size` 与 `power curve`。

---

## 四、参数含义

当前代码中统一使用以下记号：

- `M`：Monte Carlo 重复次数
- `B`：每次检验中的 bootstrap 重复次数
- `alpha`：显著性水平
- `deltas`：power 曲线上的变化幅度网格
- `seed`：随机种子
- `jobs`：Monte Carlo 外层并行工作数

特别说明：

- 当前代码里的 `M` **不是最大秩、不是截断维数**；
- 当前代码里的 `M` 专指“每个设定下重复模拟多少次”；
- `jobs` 只并行 **Monte Carlo 外层**，不并行 bootstrap 内层。

---

## 五、三类模型与默认场景

当前脚本内置了三类固定场景。

### 5.1 baseline_ols

参数为：

```text
N = 2
T = 100
p = 1
t = 50
Sigma = 0.5 * I
```

系数矩阵 `phi` 通过平稳 VAR 随机生成。

### 5.2 sparse_lasso

参数为：

```text
N = 5
T = 200
p = 1
t = 100
Sigma = 0.5 * I
sparsity = 0.2
lasso_alpha = 0.02
```

系数矩阵 `phi` 通过带稀疏约束的平稳 VAR 随机生成。

### 5.3 lowrank_svd

参数为：

```text
N = 10
T = 200
p = 1
t = 100
Sigma = 0.5 * I
rank = 2
```

系数矩阵 `phi` 通过低秩平稳 VAR 随机生成。

### 5.4 统一说明

当前版本中：

- 三个模型并没有强制使用相同的 `N`、`T`；
- 这是有意设计：baseline 更低维、更快；sparse 与 lowrank 保持各自典型场景；
- 因此当前结果应理解为“模型在各自代表性设定下的表现”，而不是完全同维度同样本下的逐项公平比较。

---

## 六、数据生成机制

### 6.1 H0：size / type I error

在 `H0` 下，数据来自无断点 VAR：

```text
Y_t ~ VAR(p; Φ, Σ)
```

每次 Monte Carlo：

1. 生成一条无断点时间序列；
2. 在固定检验点 `t` 做 bootstrap 检验；
3. 记录是否拒绝 `H0`；
4. 用拒绝比例估计 `type1_error`。

### 6.2 H1：power

在 `H1` 下，数据来自含已知断点的 VAR：

```text
断点前: VAR(p; Φ1, Σ)
断点后: VAR(p; Φ2, Σ)
```

其中：

```text
Φ2_raw = Φ1 + delta * 1
Φ2 = ensure_stationary(Φ2_raw)
```

也就是说：

- 每个 `delta` 对应一组新的断点后系数；
- 如果 `Φ2_raw` 不平稳，则按统一规则收缩：
  - shrink factor = `0.9`
  - max_attempts = `30`
- 若收缩后仍不平稳，则该 `delta` 点记为 `skipped`。

### 6.3 对 delta 的解释

当前代码中的 `delta` 表示：

- 对 `Φ1` 的所有元素同时施加同幅度加法扰动；
- 其作用是控制断点前后结构差异强度；
- 理论上 `delta` 越大，power 越应提高；
- 但如果平稳化收缩较强，名义 `delta` 与实际有效变化强度可能不完全一致。

因此解释结果时，必须同时关注：

- `power(delta)` 是否上升；
- `stationarity_shrinks` 是否变大；
- 若高 `delta` 下 shrink 很多，则不能把名义 `delta` 直接等同于实际信号强度。

---

## 七、当前输出指标

### 7.1 H0 主线

每个模型输出：

- `type1_error`
- `size_distortion = type1_error - alpha`
- `rejections`
- `M_effective`
- `runtime_sec`
- `pvalue_summary`

这里：

```text
type1_error = rejections / successful_iterations
```

### 7.2 H1 主线

每个模型对每个 `delta` 输出：

- `power`
- `rejections`
- `M_effective`
- `runtime_sec`
- `stationarity_shrinks`
- `pvalue_summary`
- `skipped`

其中：

```text
power(delta) = rejections / successful_iterations
```

当前版本重点不在置信区间，而在：

- 曲线趋势；
- 是否单调；
- 不稳定是否来自 Monte Carlo 波动、bootstrap 波动或平稳性收缩。

---

## 八、当前默认命令行参数

主脚本默认参数为：

```text
M = 200
B = 150
alpha = 0.05
deltas = [0.10, 0.20, 0.30, 0.40]
seed = 42
jobs = 1
```

可用命令行参数：

```bash
python3 experiments/run_large_scale_current_plan.py \
  --M 300 \
  --B 299 \
  --alpha 0.05 \
  --deltas 0.04 0.08 0.16 \
  --seed 42 \
  --jobs 8 \
  --tag fast_screen
```

---

## 九、建议的运行层次

### 9.1 第一轮：快速筛查版

目的：先看整体趋势，不追求最终高精度。

推荐参数：

```text
M = 300
B = 299
deltas = {0.04, 0.08, 0.16}
jobs = 可用 CPU 核数
```

用途：

- 判断 `power` 是否总体随 `delta` 上升；
- 判断哪个模型最不稳定；
- 粗略比较三类模型的 `size` 水平。

### 9.2 第二轮：确认版

目的：只对可疑趋势补精度。

推荐参数：

```text
M = 800 ~ 1000
B = 799 ~ 999
deltas = 只补跑可疑区域
jobs = 可用 CPU 核数
```

用途：

- 验证某个非单调点是否只是模拟误差；
- 验证高 `delta` 下的 shrink 是否破坏单调性；
- 对最关键模型补高精度结果。

### 9.3 不建议的做法

当前版本不建议：

- 一上来就把所有 `delta`、所有 `M`、所有 seed 一次拉满；
- 同时扫太多参数维度；
- 在结果明显不稳定之前就做复杂区间推断。

---

## 十、并行与加速后的实现变化

### 10.1 新增加速点

当前代码修改后，主要优化包括：

1. **向量化设计矩阵构造**
   - 避免在 OLS / Lasso / 低秩路径中反复使用 Python 双层循环构造滞后矩阵；
2. **外层 Monte Carlo 并行**
   - 每次 Monte Carlo 重复独立，支持 `jobs > 1`；
3. **线程回退机制**
   - 若环境不允许多进程信号量，则自动退回线程池；
4. **稀疏 Lasso 估计器缓存复用**
   - bootstrap 中减少重复创建 Lasso 对象的成本。

### 10.2 对试验方案的影响

因此当前仿真方案应修改为：

- 时间预算优先通过 `jobs` 控制；
- `M` 与 `B` 不再必须一味压低才能跑得动；
- 第一轮可在多核下直接跑筛查版；
- 第二轮再对关键模型、关键 `delta` 精化。

---

## 十一、当前输出文件结构

脚本每次运行会在 `results/` 下输出三类文件：

- JSON：完整结果对象
- CSV：整理好的绘图表格
- Markdown：中文简要分析报告

对应文件名模式：

```text
results/large_scale_experiment_<timestamp>_<tag>.json
results/large_scale_plot_data_<timestamp>_<tag>.csv
results/大规模试验分析报告_<timestamp>_<tag>.md
```

这意味着当前版本是：

- **单轮运行 → 单组 JSON / CSV / MD**
- 而不是旧方案中的多份 raw / agg / detailed report 并列结构。

---

## 十二、结果解释规则

### 12.1 size

当前优先回答：

- 哪个模型更接近 `alpha = 0.05`；
- 哪个模型偏保守（size 明显低于 0.05）；
- 哪个模型偏激进（size 明显高于 0.05）。

### 12.2 power

当前优先回答：

- `power(delta)` 是否整体上升；
- 非单调是否只发生在相邻点小波动；
- 如果高 `delta` 反而回落，是否伴随 `stationarity_shrinks` 上升；
- 哪个模型对弱变化更敏感。

### 12.3 不稳定的解释顺序

若出现不稳定，优先按下列顺序解释：

1. `M` 太小，Monte Carlo 误差大；
2. `B` 太小，bootstrap 临界值不稳；
3. 高 `delta` 下平稳化 shrink 太强；
4. 稀疏/低秩估计本身不稳定；
5. 样本长度不足。

---

## 十三、当前版本的验收标准

当前代码版本下，仿真方案验收应改为：

- [ ] 能成功运行 `baseline_ols`、`sparse_lasso`、`lowrank_svd` 三类模型；
- [ ] 每类模型都输出一条 `type1_error` 结果；
- [ ] 每类模型都输出一条 `power_curve`；
- [ ] `power_curve` 中每个 `delta` 都包含 `power`、`rejections`、`M_effective`；
- [ ] 若某 `delta` 因不平稳被跳过，结果中能明确标识 `skipped`；
- [ ] 能通过 `--jobs` 控制并行度；
- [ ] 输出 JSON / CSV / Markdown 三类文件；
- [ ] 报告中能直接看出 `size` 与 `power curve`。

---

## 十四、一句话总结

**当前代码版本的仿真方案，已经从“多阶段、多文件、多口径”的计划稿，收敛为“单脚本驱动、三模型并列、size 与 power curve 同时输出、支持并行加速”的可执行版本。**
