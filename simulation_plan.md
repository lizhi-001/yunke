# 已知断点 VAR 检验 — 当前代码版本仿真方案

## 1. 方案目标

当前方案对应一个**多 seed、已知点结构断裂、单脚本输出**的仿真实验。目标分为两部分：

1. **第一类错误（size）**：在多个 `M` 下考察三类模型的 `type1_error` 是否稳定、是否接近名义显著性水平；
2. **检验功效（power）**：固定使用 `M_max = max(M_grid)`，比较三类模型在不同断裂强度下的 `power(delta)`。

因此，当前版本不再是“同一个 `M` 同时看 size 和 power”，而是：

- **size 看 `M_grid`**；
- **power 固定看 `M_max`**；
- **结果对多个 seed 做并行重复并聚合**。

---

## 2. 对应代码

### 2.1 当前版本已修复的问题

相较于前一版实验脚本，当前版本已明确修复或补全以下机制：

- **结构断裂口径统一**：三类模型统一采用“已知点结构断裂检验”口径，`H0/H1` 使用相同的有效样本量 `T-p`；
- **第一类错误与功效拆分**：`size` 在 `M_grid` 下评估，`power` 固定在 `M_max = max(M_grid)` 下评估；
- **多 seed 并行**：支持多个 seed 并行运行，并输出逐 seed 结果与跨 seed 聚合结果；
- **效应量网格修正**：`delta` 解释为统一的基准单元素变化尺度，各模型内部按 `target_fro = delta * sqrt(#coefficients)` 换算为对应的 Frobenius 强度，避免高维模型信号过弱；
- **baseline p 值口径可选**：`baseline_ols` 支持 `bootstrap_lr`、`asymptotic_chi2`、`asymptotic_f` 三种 p 值计算方式；
- **进度日志持久化**：每个实验目录下单独提供 `progress/` 子目录；
- **异常/中断可追踪**：实验若异常退出、被中断或收到终止信号，会在进度日志中显式记录 `failed` 事件，避免“无感退出”。

核心脚本与模块：

- 主脚本：`experiments/run_large_scale_mgrid_multiseed.py`
- baseline 蒙特卡洛：`simulation/monte_carlo.py`
- 稀疏蒙特卡洛：`sparse_var/sparse_monte_carlo.py`
- 低秩蒙特卡洛：`lowrank_var/lowrank_monte_carlo.py`
- 设计矩阵：`simulation/design_matrix.py`
- 并行执行：`simulation/parallel.py`

当前实现已支持：

- 多 seed 并行；
- 三模型并行；
- Monte Carlo 外层并行（loky 真多进程）；
- 向量化设计矩阵构造；
- 向量化伪序列滞后向量组装；
- `--jobs` 控制总并行预算；
- `--seed-workers` 控制并发 seed 数。

### 2.2 并行后端：loky 替代 ProcessPoolExecutor

`simulation/parallel.py` 的 Monte Carlo 外层并行使用 **loky**（通过 `joblib.externals.loky`）作为默认进程池后端，替代标准库 `ProcessPoolExecutor`。

**背景**：标准库 `ProcessPoolExecutor` 在受限环境（容器、部分云 VM）中会因 POSIX semaphore 权限问题抛出 `PermissionError`/`OSError`，导致自动回退到 `ThreadPoolExecutor`。Python 线程受 GIL 限制，对 CPU 密集型 Monte Carlo + Bootstrap 任务无法实现真正的多核并行，实测 CPU 利用率仅 ~130%。

**改动**：

- 优先使用 loky 的 `get_reusable_executor`（绕过 POSIX semaphore 问题，复用 worker 进程减少 fork 开销）；
- 若 loky 不可用，回退到标准库 `ProcessPoolExecutor`；
- 若仍失败，最终回退到 `ThreadPoolExecutor`。

**验证结果**：loky 验证运行（M_grid=[20,50], B=50, seeds=[42], jobs=4）12/12 stages 全部完成，三模型并行正常，总耗时 62.62 秒。

---

## 3. 检验问题与三类模型

研究问题：对已知断点 `t` 的 VAR(p) 序列，检验该点前后参数是否相同：

```text
H0: Φ1 = Φ2
H1: Φ1 ≠ Φ2
```

三类模型分别为：

- `baseline_ols`：低维 OLS + 已知点结构断裂检验；默认使用 bootstrap LR p 值
- `baseline_ols_f`：与 `baseline_ols` 参数完全相同的对照组，始终使用渐近 F 检验 p 值，用于对比验证 bootstrap 与渐近方法的 size 差异
- `sparse_lasso`：稀疏 Lasso + 已知点结构断裂检验 + bootstrap LR at point
- `lowrank_svd`：低秩 SVD + 已知点结构断裂检验 + bootstrap LR at point

统一理解：

- 三类模型都在**已知断点处**做 bootstrap 检验；
- 三类模型共享相同的 `H0/H1` 定义；
- 差异只在参数估计器不同。

---

## 4. 当前统一检验口径

三类模型统一采用“**已知点结构断裂检验**”的设定：

- `H0`：已知点处不存在结构断裂，整条时间序列共用一套参数；
- `H1`：已知点处存在结构断裂，断点前后使用两套参数；
- 第二段从断点后第一个响应开始生效，但允许其滞后项借用断点前的 `p` 个观测。

因此，非受限模型按如下方式拟合：

- 第一段使用 `Y[:t]`；
- 第二段使用 `Y[t-p:]`。

对应的有效样本量为：

- `H0`：`T-p`
- `H1`：`(t-p) + (T-t) = T-p`

也就是说，`H0` 与 `H1` 在比较时使用相同的有效样本量口径。

### 4.1 统一检验流程

对三类模型，当前已知断点检验流程统一为：

1. 在 `H0` 下，用整条序列拟合单一参数模型；
2. 在 `H1` 下，用 `Y[:t]` 拟合第一段，用 `Y[t-p:]` 拟合第二段；
3. 由 `H0` 与 `H1` 的对数似然构造 LR 统计量；
4. 在 `H0` 下提取估计参数与残差；
5. 对残差居中后有放回重抽样，递归生成 bootstrap 伪样本；
6. 在每个伪样本上重复同样的 `H0` / `H1` 拟合；
7. 用 `p_value = P(LR* >= LR_obs)` 计算 p 值；
8. 若 `p_value <= alpha`，则拒绝 `H0`。

### 4.2 baseline 的理论解释

在低维 `baseline_ols` 下，这一设定对应的是标准的已知点结构断裂 LR / Chow 检验框架：

- 同一条序列的一套参数 vs 两套参数比较；
- `H0` 与 `H1` 具有相同的样本量口径；
- 因而其 LR 统计量更便于和标准渐近理论对应。

---

## 5. 参数含义

统一记号如下：

- `M_grid`：用于考察第一类错误的 Monte Carlo 网格
- `M_max`：`max(M_grid)`，用于计算功效
- `B`：每次检验中的 bootstrap 重复次数
- `alpha`：显著性水平
- `deltas`：统一基准单元素变化尺度网格；各模型内部按 `target_fro = delta * sqrt(#coefficients)` 换算为对应的 Frobenius 目标强度
- `seeds`：多 seed 重复列表
- `jobs`：总并行预算
- `seed_workers`：并发 seed 数
- `baseline_pvalue_method`：baseline_ols 的 p 值口径，默认 `bootstrap_lr`；`baseline_ols_f` 始终使用 `asymptotic_f`，不受此参数影响

特别说明：

- 当前代码中的 **size 只对 `M_grid` 逐个评估**；
- 当前代码中的 **power 只在 `M_max` 下评估**；
- `jobs` 不并行 bootstrap 内层，而是分配给“seed 并行 + 模型并行 + Monte Carlo 外层并行”。

---

## 6. 三类模型的默认场景

### 6.1 `baseline_ols` / `baseline_ols_f`

```text
N = 2
T = 200
p = 1
t = 100
Sigma = 0.5 * I
```

`baseline_ols` 默认使用 `bootstrap_lr` p 值；`baseline_ols_f` 始终使用 `asymptotic_f` p 值，作为对照。两者共享相同的数据生成参数。

### 6.2 `sparse_lasso`

```text
N = 5
T = 200
p = 1
t = 100
Sigma = 0.5 * I
sparsity = 0.2
lasso_alpha = 0.02
```

### 6.3 `lowrank_svd`

```text
N = 10
T = 200
p = 1
t = 100
Sigma = 0.5 * I
rank = 2
```

### 6.4 解释口径

四个模型统一使用 `T = 200`、`t = 100`，保证每段有效样本量一致（均为 `T - t = 99`），使 size 和 power 的比较具有公平性。各模型的维度 `N` 不同（2 / 5 / 10），反映不同复杂度场景下的检验表现。

---

## 7. 数据生成机制

### 7.1 `H0`：size / type I error

在 `H0` 下，数据来自无断点 VAR：

```text
Y_t ~ VAR(p; Φ, Σ)
```

对每个 seed、每个模型、每个 `M ∈ M_grid`：

1. 生成无断点序列；
2. 在固定检验点 `t` 做 bootstrap 检验；
3. 记录是否拒绝 `H0`；
4. 用拒绝比例估计 `type1_error(M)`。

### 7.2 `H1`：power

在 `H1` 下，数据来自含已知断点的 VAR：

```text
断点前: VAR(p; Φ1, Σ)
断点后: VAR(p; Φ2, Σ)
```

断点后参数通过下式生成：

```text
Φ2_raw = Φ1 + delta * 1
Φ2 = ensure_stationary(Φ2_raw)
```

若 `Φ2_raw` 不平稳，则按统一规则收缩：

- `shrink factor = 0.9`
- `max_attempts = 30`

若仍不平稳，则该 `delta` 记为 `skipped`。

### 7.3 `delta` 的解释

当前 `delta` 表示统一的**基准单元素变化尺度**。

在实际构造断点后参数时，各模型会先按

```text
target_fro = delta * sqrt(#coefficients)
```

将其换算为模型特定的 Frobenius 目标强度，再沿归一化全 1 方向施加扰动。这样做的目的是让不同维度模型具有更接近的平均单系数变化幅度，而不至于在高维模型中因固定总 Frobenius 强度过小而导致信号过弱。

解释结果时要同时看：

- `power(delta)` 是否上升；
- `stationarity_shrinks` 是否变大；
- 若高 `delta` 下 shrink 很多，则名义 `delta` 不等于实际有效信号强度。

---

## 8. 输出指标

### 8.1 `H0` 输出：不同 `M` 下的第一类错误

每个模型对每个 `M ∈ M_grid` 输出：

- `type1_error`
- `size_distortion = type1_error - alpha`
- `rejections`
- `M_effective`
- `runtime_sec`
- `pvalue_summary`

其中：

```text
type1_error(M) = rejections / successful_iterations
```

### 8.2 `H1` 输出：固定 `M_max` 下的功效

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
power(delta; M_max) = rejections / successful_iterations
```

### 8.3 多 seed 聚合

当前版本会同时保留：

- **每个 seed 的原始结果**；
- **跨 seed 聚合结果**（均值、标准差等）。

因此当前输出重点是：

- size 随 `M` 的稳定性；
- power 在 `M_max` 下的变化趋势；
- seed 间波动大小。

---

## 9. 默认参数与运行建议

### 9.1 默认参数

```text
M_grid = [30, 50, 100, 150, 200, 300]
B = 200
alpha = 0.05
deltas = [0.04, 0.08, 0.12, 0.16]
seeds = [42, 2026, 7]
jobs = 4
seed_workers = 0 (自动)
baseline_pvalue_method = bootstrap_lr
```

### 9.2 命令行示例

```bash
python3 experiments/run_large_scale_mgrid_multiseed.py \
  --M-grid 30 50 100 150 200 300 \
  --B 500 \
  --alpha 0.05 \
  --deltas 0.04 0.08 0.12 0.16 \
  --seeds 42 2026 7 \
  --jobs 8 \
  --seed-workers 2 \
  --tag formal_mgrid_multiseed
```

### 9.3 推荐理解方式

- 若关注 **size 是否稳定**，看同一模型在不同 `M` 下的 `type1_error`；
- 若关注 **power 是否足够**，看 `M_max` 下的 `power(delta)`；
- 若关注 **结果是否稳健**，看 seed 间均值与标准差。

---

## 10. 输出文件

每次运行在 `results/large_scale_runs/<timestamp>_<tag>/` 下输出：

- `large_scale_experiment_*.json`：完整结果（含每个 seed 与聚合结果）
- `large_scale_raw_*.csv`：逐 seed 原始结果
- `large_scale_agg_*.csv`：跨 seed 聚合结果
- `大规模试验分析报告_*.md`：中文简要分析报告
- `seed_results/seed_<seed>.json`：各 seed 单独结果
- `run_meta.json`：路径元信息
- `progress/`：进度目录，专门存放实验过程日志，不与结果文件混放

---

### 10.1 `progress/` 目录说明

当前版本中，每个实验目录下都会单独创建 `progress/` 子目录，包含：

- `progress.log`：全实验的人类可读进度日志；
- `progress.jsonl`：全实验的结构化事件流；
- `summary.json`：全实验总进度摘要；
- `seed_<seed>_summary.json`：单个 seed 的独立进度摘要。

说明：

- `summary.json` 反映整个实验 run 的总进度；
- `seed_<seed>_summary.json` 只反映单个 seed 自己的完成情况；
- 当前版本已移除易混淆的 `seed_<seed>_events.log`，避免把事件流水误当成单 seed 进度看板。

### 10.2 失败与中断日志

若实验异常退出、被中断或收到终止信号，当前版本会在：

- `progress.log`
- `progress.jsonl`
- `summary.json`

中显式记录 `failed` 事件。也就是说，当前版本不允许长实验“无感退出”。

## 11. 结果解释规则

优先回答以下问题：

### 11.1 size

- 随 `M` 增大，第一类错误是否更稳定；
- 哪个模型更接近 `alpha = 0.05`；
- 哪个模型偏保守；
- 哪个模型偏激进。

### 11.2 power

- 在 `M_max` 下，`power(delta)` 是否整体上升；
- 高 `delta` 下是否仍然保持单调；
- 哪个模型对弱断裂更敏感。

### 11.3 若结果不稳定，优先按以下顺序解释

1. `M` 太小，Monte Carlo 误差大；
2. `B` 太小，bootstrap 临界值不稳；
3. seed 数太少，跨 seed 波动大；
4. 高 `delta` 下平稳化 shrink 太强；
5. 稀疏/低秩估计本身不稳定。

---

## 12. 验收标准

当前版本的仿真方案应满足：

- [ ] 能成功运行 `baseline_ols`、`baseline_ols_f`、`sparse_lasso`、`lowrank_svd` 四类模型；
- [ ] `baseline_ols` 使用 `bootstrap_lr`，`baseline_ols_f` 使用 `asymptotic_f` 作为对照；
- [ ] 每类模型都输出不同 `M` 下的 `type1_error`；
- [ ] 每类模型都输出固定 `M_max` 下的 `power_curve`；
- [ ] 三类模型都使用相同的结构断裂检验口径；
- [ ] 支持多个 seed 并行运行；
- [ ] 同时输出逐 seed 原始结果与跨 seed 聚合结果；
- [ ] 输出 JSON / raw CSV / agg CSV / Markdown 四类结果；
- [ ] 报告中能直接看出 size 随 `M` 的变化与 `M_max` 下的 power 曲线。

---

## 13. 一句话总结

**当前代码版本的仿真方案，是一个”多 seed 并行、不同 `M` 看第一类错误、固定 `M_max` 看功效、统一比较四模型（含 F 检验对照）结构断裂检验表现”的可执行版本。**
