# 第一类错误与统计功效（Power）计算方法说明

## 一、统计检验框架回顾

### 1.1 假设设定

```
H0: Φ₁ = Φ₂ = Φ    （无结构变化）
H1: Φ₁ ≠ Φ₂         （在已知时间点t存在结构变化）
```

### 1.2 两类错误的定义

| 错误类型 | 定义 | 代码中的度量 |
|---------|------|------------|
| 第一类错误（Type I Error / Size） | H0为真时错误地拒绝H0的概率 | `type1_error` = 拒绝次数 / 总次数 |
| 第二类错误（Type II Error） | H1为真时错误地接受H0的概率 | 代码中不直接计算，通过 `β = 1 - power` 得到 |
| 统计功效（Power） | H1为真时正确地拒绝H0的概率 | `power` = 拒绝次数 / 总次数 |

关系：`Power = 1 - Type II Error`，即 `功效 = 1 - 第二类错误率`。

---

## 二、计算方法的统一结构

三条流程（基准OLS、稀疏Lasso、低秩）共享相同的Monte Carlo评估框架，仅在Bootstrap检验环节使用不同的估计方法。

### 2.1 通用流程

```
重复M次：
  1. 按照DGP生成一条时间序列Y
  2. 对Y执行Bootstrap LR检验，得到p值
  3. 若 p值 ≤ α，则记为"拒绝H0"
最终指标 = 拒绝次数 / 成功迭代次数
```

**区别仅在于第1步的DGP**：
- 评估第一类错误时：DGP在H0下生成（无断点）
- 评估功效时：DGP在H1下生成（有断点）

---

## 三、第一类错误（Size）的计算

### 3.1 数据生成（H0为真）

使用**单一系数矩阵Φ**在整个样本期内生成序列，不存在结构变化。

对应代码（以低秩为例，`lowrank_monte_carlo.py:78-79`）：
```python
# H0为真：使用同一个Φ生成整条序列
Y = generator.generate_var_series(T, N, p, Phi, Sigma)
```

数据生成过程（`data_generator.py:169-176`）：
```python
for t in range(p, total_length):
    Y_lag_ordered = np.zeros(N * p)
    for lag in range(p):
        Y_lag_ordered[lag*N:(lag+1)*N] = Y[t-lag-1, :]
    Y[t, :] = c + Phi @ Y_lag_ordered + epsilon[t, :]
```

### 3.2 Bootstrap检验

对每条生成的序列Y，执行完整的Bootstrap LR检验：

```python
bootstrap = LowRankBootstrapInference(B=self.B, method=self.method, rank=self.rank)
result = bootstrap.test(Y, p, t, alpha=test_alpha)
```

Bootstrap检验内部流程（`lowrank_bootstrap.py:61-138`）：

1. 在H0下拟合全样本，得到 `Φ̂_R, ĉ_R, ε̂_R`
2. 计算原始LR统计量：`LR_obs = 2(lnL_U - lnL_R)`
3. 重复B次：
   - 残差居中并有放回重抽样
   - 用 `Φ̂_R, ĉ_R` 和重抽样残差生成伪序列 `Y*`
   - 计算 `LR*`
4. `p值 = mean(LR* ≥ LR_obs)`
5. 若 `p值 ≤ α`，则拒绝H0

### 3.3 汇总计算

```python
# lowrank_monte_carlo.py:96
type1_error = rejections / successful_iterations
```

**含义**：在M次独立实验中（每次都是H0为真的数据），有多少比例错误地拒绝了H0。

**理想值**：`type1_error ≈ α`（名义显著性水平，通常0.05）。

**Size Distortion**（尺寸扭曲）：
```python
size_distortion = type1_error - alpha
```
衡量实际拒绝率与名义水平的偏差。正值表示过度拒绝（检验偏激进），负值表示拒绝不足（检验偏保守）。

### 3.4 三条流程的实现对比

| 流程 | 文件 | 函数 | Bootstrap类 |
|------|------|------|------------|
| 基准OLS | `simulation/monte_carlo.py:32` | `evaluate_type1_error_at_point()` | `BootstrapInference` |
| 稀疏Lasso | `sparse_var/sparse_monte_carlo.py` | `evaluate_type1_error()` | `SparseBootstrapInference` |
| 低秩 | `lowrank_var/lowrank_monte_carlo.py:46` | `evaluate_type1_error()` | `LowRankBootstrapInference` |

三者的Monte Carlo外层循环结构完全一致，区别仅在于调用的Bootstrap类不同。

---

## 四、统计功效（Power）的计算

### 4.1 数据生成（H1为真）

使用**两个不同的系数矩阵** `Φ₁` 和 `Φ₂`，在断点 `break_point` 处切换，生成含结构变化的序列。

对应代码（`lowrank_monte_carlo.py:141-144`）：
```python
# H1为真：断点前用Φ₁，断点后用Φ₂
Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)
```

数据生成过程（`data_generator.py:224-231`）：
```python
for t in range(p, total_length):
    Y_lag_ordered = np.zeros(N * p)
    for lag in range(p):
        Y_lag_ordered[lag*N:(lag+1)*N] = Y[t-lag-1, :]
    # 根据断点选择系数矩阵
    Phi = Phi1 if t < actual_break else Phi2
    Y[t, :] = c + Phi @ Y_lag_ordered + epsilon[t, :]
```

### 4.2 Bootstrap检验

与第一类错误评估完全相同的检验流程：
```python
bootstrap = LowRankBootstrapInference(B=self.B, method=self.method, rank=self.rank)
result = bootstrap.test(Y, p, t, alpha=test_alpha)
```

**关键点**：Bootstrap检验本身不知道数据是在H0还是H1下生成的。它始终在H0假设下生成伪序列，然后比较原始LR与Bootstrap LR分布。当数据确实含有结构变化时，原始LR会较大，更容易落在Bootstrap分布的尾部，从而被拒绝。

### 4.3 汇总计算

```python
# lowrank_monte_carlo.py:161
power = rejections / successful_iterations
```

**含义**：在M次独立实验中（每次都是H1为真的数据），有多少比例正确地拒绝了H0。

**理想值**：`power` 越接近1越好，表示检验能有效检测到结构变化。

**第二类错误率**：`β = 1 - power`。代码中不直接输出，但可由power推算。

### 4.4 功效曲线（Power Curve）

基准OLS的Monte Carlo模块还提供了功效曲线计算（`simulation/monte_carlo.py:350-420`）：

```python
def power_curve(self, ..., delta_values, ...):
    for delta in delta_values:
        Phi2 = Phi_base + delta * np.ones_like(Phi_base)
        # 检查平稳性
        if not VARDataGenerator.check_stationarity(Phi2):
            powers.append(np.nan)
            continue
        result = self.evaluate_power(...)
        powers.append(result['power'])
```

**Δ的构造方式**：`Φ₂ = Φ₁ + Δ·𝟙`，即对系数矩阵的所有元素施加均匀的加性扰动。

**预期结果**：
- Δ = 0 时，power ≈ α（退化为第一类错误）
- Δ 增大时，power 单调递增趋向1
- 功效曲线的上升速度反映检验的灵敏度

---

## 五、完整计算流程图

```
┌──────────────────────────────────────────────────────────┐
│                     Monte Carlo 外层循环                    │
│                      重复 M = 1000 次                      │
├─────────────────────────┬────────────────────────────────┤
│   第一类错误评估（Size）   │       统计功效评估（Power）       │
├─────────────────────────┼────────────────────────────────┤
│ DGP: Y ~ VAR(Φ)         │ DGP: Y ~ VAR(Φ₁→Φ₂, 断点t)   │
│ （H0为真，无断点）         │ （H1为真，有断点）               │
├─────────────────────────┴────────────────────────────────┤
│             对当前样本 Y 执行一次 Bootstrap LR 检验          │
│                                                          │
│  [Bootstrap准备阶段：仅 1 次]                               │
│  1. H0下全样本拟合 → Φ̂_R, ĉ_R, ε̂_R                      │
│  2. 计算原始 LR_obs = 2(lnL_U - lnL_R)                    │
│                                                          │
│  [Bootstrap重抽样循环：重复 B = 500 次]                     │
│  3. 残差居中 + 有放回重抽样                                 │
│  4. 用 Φ̂_R, ĉ_R 和重抽样残差生成伪序列 Y*                  │
│  5. 计算 LR*_b（第 b 次Bootstrap统计量）                   │
│                                                          │
│  [循环结束后]                                               │
│  6. p值 = P(LR*_b ≥ LR_obs)                               │
│  7. 若 p值 ≤ α → 拒绝H0                                   │
├─────────────────────────┬────────────────────────────────┤
│ type1_error             │ power                          │
│ = 拒绝次数 / M_effective │ = 拒绝次数 / M_effective        │
│ 理想值 ≈ α (0.05)       │ 理想值 → 1                      │
│                         │ Type II Error = 1 - power      │
└─────────────────────────┴────────────────────────────────┘
```

---

## 六、关键参数说明

| 参数 | 符号 | 含义 | 典型值 |
|------|------|------|--------|
| Monte Carlo次数 | M | 外层重复次数，决定Size/Power估计精度 | 演示30-50，正式1000-5000 |
| Bootstrap次数 | B | 内层重复次数，决定p值精度 | 演示50-100，正式500-1000 |
| 显著性水平 | α | 拒绝H0的阈值 | 0.05 |
| 变化强度 | Δ | Φ₂ = Φ₁ + Δ·𝟙 中的扰动大小 | 0.05 ~ 0.5 |
| 检验时间点 | t | 已知的候选断点位置 | 通常取 T/2 |

### 计算量

每次完整的Size或Power评估需要：
- M × (1 + B) 次模型拟合
- 例如 M=50, B=100 时：50 × 101 = 5,050 次拟合
- 正式实验 M=1000, B=500 时：1000 × 501 = 501,000 次拟合

---

## 七、异常处理机制

所有Monte Carlo循环都包含异常捕获：

```python
try:
    Y = generator.generate_var_series(...)
    result = bootstrap.test(...)
    p_values.append(result['p_value'])
    successful_iterations += 1
    if result['reject_h0']:
        rejections += 1
except Exception as e:
    continue  # 跳过失败的迭代
```

最终指标使用 `successful_iterations`（而非 `M`）作为分母：
```python
type1_error = rejections / successful_iterations
```

这意味着失败的迭代被排除在外，不计入拒绝率的分母。`M_effective` 记录了实际成功的次数，用于评估结果的可靠性。


---

## 八、已知断点检验统计量分布说明（新增）

### 8.1 默认“高斯假设”出现在哪一步

当前实现中，高斯假设同时出现在**数据生成**与**似然计算**两个环节：

1. **数据生成（DGP）**：误差项使用多元正态分布抽样，
   `ε_t ~ N(0, Σ)`（见 `simulation/data_generator.py:166` 与 `simulation/data_generator.py:221`）。
2. **LR对数似然**：使用高斯对数似然公式，
   `lnL = -T_eff/2 × [N·ln(2π) + ln|Σ̂| + N]`（见 `simulation/var_estimator.py:146`）。

因此，“默认高斯”并不表示观测序列 `Y_t` 本身必须正态，而是表示驱动噪声 `ε_t` 采用高斯分布，并据此构造似然评分。

### 8.2 baseline / 稀疏 / 低秩三种场景下，LR是否服从卡方

- **baseline（OLS）**：在常规正则条件、样本足够大、并且统计量构造标准时，LR在 `H0` 下可用渐近 `χ²` 近似。
- **稀疏（Lasso）**：存在正则化与变量选择不确定性，零假设分布通常是非标准分布，不能直接套用标准 `χ²`。
- **低秩（秩约束）**：参数空间受约束，通常也属于非标准情形，直接套用标准 `χ²` 不稳健。

结论：三条流程中，只有baseline在严格条件下可考虑渐近 `χ²`；稀疏与低秩仍应以Bootstrap为主。

### 8.3 旧写法中“直接用卡方”会偏移的原因（适用范围说明）

本节结论仅适用于旧的baseline写法（“全样本拟合 vs 两段分别拟合”）：

- 约束模型有效样本量：`T_eff,R = T - p`
- 非约束模型有效样本量：`T_eff,1 + T_eff,2 = (t-p) + (T-t-p) = T - 2p`

即非约束模型总有效样本量比约束模型少 `p`。展开后，LR会出现一个与数据无关的常数偏移：

`p × [N·ln(2π) + N]`

这会影响LR绝对值与标准 `χ²` 临界值的直接比较。该点在 `results/bootstrap_implementation_analysis.md` 中已有推导说明。

对当前采用的“同一有效样本口径”Chow/LR并行实现（见8.5节），该偏移问题已被规避。

### 8.4 为什么Bootstrap p值仍然可用

在旧写法中，`LR_obs` 与每个 `LR*`（Bootstrap统计量）由同一套算法构造，常数偏移会“同口径”进入原始统计量与重抽样统计量，
因此用 `p = P(LR* ≥ LR_obs)` 计算的Bootstrap p值通常仍具有可比性与稳健性。

### 8.5 若希望直接使用卡方/F临界值，建议的baseline改造

可将baseline改为“**同一有效样本口径**”的条件似然/Chow实现：

1. 在统一样本 `s = p, ..., T-1` 上建模（`n = T-p`），不再对两段各自二次损失 `p` 个样本。
2. 旧版交互项口径中，可设断点指示变量 `D_s = 1(s >= t)`，构造：

`H0: y_s = c + Φx_s + u_s`

`H1: y_s = c + Φx_s + D_s(Δc + ΔΦx_s) + u_s`（该写法现仅保留作等价参考）

3. 在该口径下：
   - 可构造LR：`LR = n[ln|Σ̂_R| - ln|Σ̂_U|]`
   - 或构造Chow型F统计量：
     `F = ((SSR_R - SSR_U)/q) / (SSR_U/(n-k_U))`

在经典OLS条件下，`F`可用F分布判定，LR/Wald可用渐近 `χ²` 判定。

### 8.6 与Bootstrap对比的建议

若目标是“验证Bootstrap有效性”，建议：

- 在baseline（同一有效样本口径）中并行计算 `p_χ² / p_F` 与 `p_bootstrap`，比较Size与Power；
- 在稀疏/低秩场景继续以Bootstrap为主，再展示“直接渐近法”与Bootstrap的差异。

这类对比能更清晰地说明：Bootstrap不仅是替代方案，而是在非标准和有限样本场景下更稳健的推断工具。


### 8.7 与文献框架的一致性说明（Lütkepohl, 2005）

在已知断点场景下，经典VAR文献通常将“参数稳定性”写为线性约束（如 `B1 = B2`），并构造Wald/LR（或quasi-LR）统计量进行检验；
在常规正则条件下使用渐近 `χ²` 近似。在有限样本、非标准估计或高维约束情形下，常结合Bootstrap改善推断。

本文档第8.5节的“同一有效样本口径”改造与该文献思路一致，可作为baseline中 `χ²/F` 与Bootstrap并行对照的实现基础。
