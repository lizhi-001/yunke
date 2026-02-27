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
