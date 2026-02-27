# 低秩VAR Bootstrap实现正确性分析报告

## 一、整体架构概述

低秩VAR的Bootstrap检验流程涉及以下模块的协作：

| 模块 | 文件 | 核心职责 |
|------|------|----------|
| `NuclearNormVAR` | `lowrank_var/nuclear_norm.py` | 低秩VAR参数估计（SVD/CVXPY） |
| `RankSelector` | `lowrank_var/rank_selection.py` | 自动秩选择 |
| `LowRankLRTest` | `lowrank_var/lowrank_lr_test.py` | LR统计量计算 |
| `LowRankBootstrapInference` | `lowrank_var/lowrank_bootstrap.py` | Bootstrap推断主流程 |
| `LowRankMonteCarloSimulation` | `lowrank_var/lowrank_monte_carlo.py` | Monte Carlo评估 |

Bootstrap检验的完整流程为：
```
原始数据Y → H0下全样本估计(Φ̂_R, ĉ_R, ε̂_R) → 计算原始LR(t)
                                              → B次循环：残差重抽样 → 生成伪序列Y* → 计算LR*(t)
                                              → p值 = P(LR* ≥ LR_observed)
```

---

## 二、逐步骤详细分析

### 2.1 设计矩阵构建（`nuclear_norm.py:56-63`）

```python
X = np.zeros((T_eff, N * p))
for t in range(T_eff):
    for lag in range(p):
        X[t, lag*N:(lag+1)*N] = Y[t + p - lag - 1, :]
```

**滞后排列方式**：`X[t, :]` = `[Y_{t+p-1}, Y_{t+p-2}, ..., Y_t]`，即第一个N列对应最近的滞后（lag=1），第二个N列对应次近的滞后（lag=2），以此类推。

**验证**：当 `t=0, lag=0` 时，`X[0, 0:N] = Y[p-1, :]`（最近滞后）；当 `t=0, lag=1` 时，`X[0, N:2N] = Y[p-2, :]`（次近滞后）。这与VAR(p)模型 `Y_t = c + Φ₁Y_{t-1} + Φ₂Y_{t-2} + ... + ε_t` 的标准排列一致。

**结论**：✅ 正确

---

### 2.2 参数估计

#### 2.2.1 截断SVD方法（`nuclear_norm.py:158-232`）

```python
# OLS估计
X_full = np.column_stack([np.ones(T_eff), X])
B_ols = np.linalg.lstsq(X_full, Y_response, rcond=None)[0]
c_hat = B_ols[0, :]
Phi_ols = B_ols[1:, :].T  # 形状 (N, N*p)

# 截断SVD
U, s, Vt = svd(Phi_ols, full_matrices=False)
s_truncated[:rank] = s[:rank]
Phi_hat = U @ np.diag(s_truncated) @ Vt
```

**流程**：先做OLS得到无约束估计 `Φ̂_OLS`，再对其做截断SVD，只保留前r个奇异值，将系数矩阵投影到秩-r空间。

**注意**：`B_ols` 的形状为 `(N*p+1, N)`，其中第0行是常数项。`B_ols[1:, :].T` 得到 `(N, N*p)` 的系数矩阵。

**结论**：✅ 正确。这是标准的"先OLS后截断"两步估计法。

#### 2.2.2 核范数正则化方法（`nuclear_norm.py:65-156`）

```python
Phi = cp.Variable((N, N * p))
loss = cp.sum_squares(Y_response - Y_pred) / (2 * T_eff)
reg = self.lambda_nuc * cp.normNuc(Phi)
objective = cp.Minimize(loss + reg)
```

**目标函数**：`min ||Y - XΦ'||²_F / (2T_eff) + λ||Φ||_*`

**结论**：✅ 正确。标准的核范数正则化目标函数。

---

### 2.3 对数似然计算（`nuclear_norm.py:142-145, 217-221`）

```python
Sigma_hat = (self.residuals.T @ self.residuals) / T_eff
det_Sigma = det(Sigma_hat)
if det_Sigma <= 0:
    det_Sigma = 1e-10
log_likelihood = -0.5 * T_eff * (N * np.log(2 * np.pi) + np.log(det_Sigma) + N)
```

**公式推导**：

多元正态分布的对数似然为：
```
lnL = -T_eff/2 × [N·ln(2π) + ln|Σ̂| + tr(Σ̂⁻¹·S)]
```
其中 `S = ε'ε/T_eff` 是样本协方差。当 `Σ̂ = S`（MLE估计）时，`tr(Σ̂⁻¹·S) = tr(I_N) = N`，因此：
```
lnL = -T_eff/2 × [N·ln(2π) + ln|Σ̂| + N]
```

**结论**：✅ 公式正确。使用MLE口径的协方差矩阵（除以T_eff而非T_eff-1），代入后简化为上述形式。

**⚠️ 注意事项**：`det_Sigma <= 0` 时的兜底处理（设为1e-10）在数值上是合理的安全措施，但可能掩盖协方差矩阵奇异的根本问题（如样本量不足或维度过高）。

---

### 2.4 LR统计量计算（`lowrank_lr_test.py:49-105`）

```python
# H0: 全样本拟合
result_r = self._fit_model(Y, p, include_const)
log_lik_r = result_r['log_likelihood']

# H1: 分段拟合
Y1 = Y[:t, :]
Y2 = Y[t:, :]
result1 = self._fit_model(Y1, p, include_const)
result2 = self._fit_model(Y2, p, include_const)
log_lik_u = log_lik_1 + log_lik_2

lr_statistic = 2 * (log_lik_u - log_lik_r)
```

**公式**：`LR(t) = 2 × [lnL_U(t) - lnL_R]`

**⚠️ 关键问题：有效样本量不一致**

| 模型 | 有效样本量 |
|------|-----------|
| 约束模型（全样本） | T_eff_R = T - p |
| 非约束段1（Y[:t]） | T_eff_1 = t - p |
| 非约束段2（Y[t:]） | T_eff_2 = (T-t) - p |
| 非约束总计 | T_eff_1 + T_eff_2 = T - 2p |

**问题**：非约束模型的总有效样本量 `T - 2p` 少于约束模型的 `T - p`，差值为 `p`。这是因为每个子段都独立损失了 `p` 个观测值用于构建滞后结构。

**影响分析**：

展开LR统计量：
```
LR = T_eff_R·[N·ln(2π) + ln|Σ̂_R| + N]
   - T_eff_1·[N·ln(2π) + ln|Σ̂_1| + N]
   - T_eff_2·[N·ln(2π) + ln|Σ̂_2| + N]

   = p·[N·ln(2π) + N] + T_eff_R·ln|Σ̂_R| - T_eff_1·ln|Σ̂_1| - T_eff_2·ln|Σ̂_2|
```

第一项 `p·[N·ln(2π) + N]` 是一个不依赖于数据的常数偏移。由于Bootstrap检验中原始LR和Bootstrap LR*都包含相同的常数偏移，**p值计算不受影响**。

**结论**：✅ 对Bootstrap推断而言正确。LR统计量的绝对值包含常数偏移，但不影响Bootstrap p值。

**⚠️ 但需注意**：这意味着LR统计量的绝对值不能直接与卡方分布的临界值比较（这也是使用Bootstrap而非渐近分布的原因之一）。

---

### 2.5 Bootstrap伪序列生成（`lowrank_bootstrap.py:37-59`）

```python
def generate_pseudo_series(self, Y, p, Phi, c, residuals):
    T, N = Y.shape
    T_eff = len(residuals)

    # 残差居中
    centered_residuals = residuals - np.mean(residuals, axis=0)
    # 有放回重抽样
    indices = np.random.choice(T_eff, size=T_eff, replace=True)
    resampled_residuals = centered_residuals[indices, :]

    # 初始值
    Y_star = np.zeros((T, N))
    Y_star[:p, :] = Y[:p, :]

    # 递归生成
    for t in range(p, T):
        Y_lag_ordered = np.zeros(N * p)
        for lag in range(p):
            Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]
        epsilon_t = resampled_residuals[t - p, :]
        Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t
```

**逐项分析**：

1. **残差居中**（第44行）：`centered_residuals = residuals - np.mean(residuals, axis=0)`
   - ✅ 正确。标准做法，确保Bootstrap DGP的新息均值为零。

2. **有放回重抽样**（第45-46行）：从T_eff个居中残差中有放回抽取T_eff个。
   - ✅ 正确。标准的残差Bootstrap（residual bootstrap）。

3. **初始值**（第48-49行）：使用原始数据的前p个观测值。
   - ✅ 正确。标准做法，保持初始条件一致。

4. **滞后排列**（第52-54行）：`Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]`
   - ✅ 与设计矩阵构建（2.1节）的排列方式一致。

5. **递归公式**（第57行）：`Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t`
   - ✅ 正确。使用H0下的参数生成数据，确保伪序列服从H0。

6. **残差索引**（第56行）：`epsilon_t = resampled_residuals[t - p, :] if t - p < T_eff else np.zeros(N)`
   - ✅ 当 `t` 从 `p` 到 `T-1` 时，`t-p` 从 `0` 到 `T-p-1 = T_eff-1`，条件始终为真，兜底分支不会执行。

**结论**：✅ 伪序列生成完全正确。

---

### 2.6 Bootstrap主流程（`lowrank_bootstrap.py:61-138`）

```python
def test(self, Y, p, t, alpha=0.05, verbose=False):
    # Step 1: 原始LR统计量
    lr_test = LowRankLRTest(method=self.method, rank=self.rank, lambda_nuc=self.lambda_nuc)
    original_result = lr_test.compute_lr_at_point(Y, p, t)
    original_lr = original_result['lr_statistic']

    # 获取H0下参数
    Phi_r = restricted_result['Phi']
    c_r = restricted_result['c']
    residuals_r = restricted_result['residuals']

    # Step 2: Bootstrap循环
    for b in range(self.B):
        Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)
        result_b = lr_test_b.compute_lr_at_point(Y_star, p, t)
        bootstrap_lr_values.append(result_b['lr_statistic'])

    # Step 3: p值
    p_value = np.mean(bootstrap_statistics >= original_lr)
```

**逐步验证**：

1. **H0下参数提取**：从约束模型（全样本拟合）中提取 `Φ̂_R, ĉ_R, ε̂_R`。
   - ✅ 正确。Bootstrap应在H0下生成数据。

2. **Bootstrap循环中的估计方法**：每次Bootstrap样本使用相同的低秩估计方法。
   - ✅ 正确。Bootstrap应模拟原始检验的完整流程。

3. **p值计算**：`p_value = P(LR* ≥ LR_observed)`
   - ✅ 正确。单侧检验，LR统计量越大越倾向于拒绝H0。

4. **临界值计算**：
   ```python
   critical_values = {
       0.10: np.percentile(bootstrap_statistics, 90),
       0.05: np.percentile(bootstrap_statistics, 95),
       0.01: np.percentile(bootstrap_statistics, 99)
   }
   ```
   - ✅ 正确。α=0.05对应第95百分位数。

5. **拒绝规则**：`reject_h0 = p_value <= alpha`
   - ✅ 正确。

**结论**：✅ Bootstrap主流程正确。

---

### 2.7 与基准Bootstrap的一致性对比

将低秩Bootstrap（`lowrank_bootstrap.py`）与基准OLS Bootstrap（`simulation/bootstrap.py`）逐项对比：

| 环节 | 基准Bootstrap | 低秩Bootstrap | 一致性 |
|------|-------------|-------------|--------|
| 残差居中 | `residuals - mean(residuals, axis=0)` | 相同 | ✅ |
| 重抽样 | `np.random.choice(T_eff, size=T_eff, replace=True)` | 相同 | ✅ |
| 初始值 | `Y_star[:p, :] = Y[:p, :]` | 相同 | ✅ |
| 滞后排列 | `Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]` | 相同 | ✅ |
| 递归公式 | `c + Phi @ Y_lag_ordered + epsilon_t` | 相同 | ✅ |
| p值 | `mean(bootstrap_stats >= original_lr)` | 相同 | ✅ |

**结论**：✅ 低秩Bootstrap与基准Bootstrap在框架层面完全一致，唯一区别在于参数估计方法（OLS vs 低秩），符合设计意图。

---

## 三、潜在问题与风险点

### 3.1 ⚠️ LR统计量可能为负值

由于低秩估计不是无约束MLE，非约束模型（分段拟合）的对数似然不一定大于约束模型（全样本拟合）。这意味着 `LR = 2(lnL_U - lnL_R)` 可能为负。

**影响**：负的LR统计量在经典理论中不应出现。当LR < 0时，意味着分段拟合反而不如全样本拟合，这可能是由于：
- 低秩约束在小样本段上估计不稳定
- 秩选择在子段上不准确

**当前处理**：代码未对负LR做特殊处理。在Bootstrap框架下，如果原始LR为负，p值会接近1（不拒绝H0），结果仍然合理。

**建议**：可以考虑对负LR取max(LR, 0)，但在Bootstrap框架下不是必须的。

### 3.2 ⚠️ 秩选择的一致性问题

当 `rank=None` 且 `method='svd'` 时，每次拟合都会通过BIC自动选择秩：

```python
if rank is None:
    selector = RankSelector()
    rank_result = selector.select_by_information_criterion(Y, p, max_rank=..., criterion='bic')
    rank = rank_result['selected_rank']
```

**问题**：
1. 原始数据的全样本拟合、段1拟合、段2拟合可能选择不同的秩
2. Bootstrap样本的拟合也可能选择不同的秩
3. 这增加了额外的随机性，可能影响检验的size控制

**建议**：在实验中建议指定固定的 `rank` 参数，避免自动选择带来的不确定性。

### 3.3 ⚠️ 子段样本量不足的风险

分段拟合时，每段的有效样本量为：
- 段1：`T_eff_1 = t - p`
- 段2：`T_eff_2 = (T-t) - p`

当 `t ≈ T/2` 时，每段约有 `T/2 - p` 个有效观测。对于高维情况（N较大），如果 `T/2 - p` 不远大于 `N×p`，低秩估计可能不稳定。

**当前保护**：`min_segment_size = p + 2`（`lowrank_lr_test.py:72`），这个阈值较低。

**建议**：考虑提高最小段长度要求，例如 `min_segment_size = max(p + 2, 2*N*p)`。

### 3.4 ⚠️ 段2的初始条件处理

```python
Y2 = Y[t:, :]
result2 = self._fit_model(Y2, p, include_const)
```

段2从 `Y[t:]` 开始拟合，其前p个观测值 `Y[t:t+p]` 作为初始条件，有效样本从 `Y[t+p:]` 开始。这意味着：
- 断点附近的p个观测值（`Y[t:t+p]`）仅用作初始条件，不参与段2的参数估计
- 这些观测值实际上是在新参数 `Φ₂` 下生成的，但其滞后值来自旧参数 `Φ₁` 下的数据

**影响**：这是结构断点检验中的标准处理方式，损失少量信息但避免了跨断点的复杂处理。在样本量足够时影响可忽略。

**结论**：✅ 可接受的标准做法。

### 3.5 ✅ 异常处理

```python
try:
    Y_star = self.generate_pseudo_series(...)
    result_b = lr_test_b.compute_lr_at_point(Y_star, p, t)
    bootstrap_lr_values.append(result_b['lr_statistic'])
except Exception:
    continue
```

Bootstrap循环中的异常被静默跳过，`B_effective` 记录了实际成功的次数。

**评价**：这是合理的做法。Bootstrap样本可能因数值问题（如协方差矩阵奇异）而失败，跳过这些样本不影响推断的有效性，前提是失败率不太高。

---

## 四、与数据生成器的一致性验证

### 4.1 数据生成器中的滞后排列（`data_generator.py:172-174`）

```python
for lag in range(p):
    Y_lag_ordered[lag*N:(lag+1)*N] = Y[t-lag-1, :]
```

**与Bootstrap伪序列生成的排列完全一致**。✅

### 4.2 数据生成器中的递归公式（`data_generator.py:175`）

```python
Y[t, :] = c + Phi @ Y_lag_ordered + epsilon[t, :]
```

**与Bootstrap伪序列生成的递归公式完全一致**。✅

---

## 五、总结评估

### 正确性判定

| 环节 | 状态 | 说明 |
|------|------|------|
| 设计矩阵构建 | ✅ 正确 | 滞后排列一致 |
| 低秩参数估计（SVD） | ✅ 正确 | 标准两步法 |
| 低秩参数估计（CVXPY） | ✅ 正确 | 标准核范数正则化 |
| 对数似然计算 | ✅ 正确 | MLE口径Σ̂下的标准公式 |
| LR统计量 | ✅ 正确 | 含常数偏移但不影响Bootstrap推断 |
| 残差居中 | ✅ 正确 | 标准做法 |
| 残差重抽样 | ✅ 正确 | 标准残差Bootstrap |
| 伪序列生成 | ✅ 正确 | 与DGP一致 |
| p值计算 | ✅ 正确 | 单侧检验 |
| 与基准Bootstrap一致性 | ✅ 一致 | 仅估计方法不同 |

### 需关注的风险点

1. LR统计量可能为负（低秩约束导致），但不影响Bootstrap推断的有效性
2. 自动秩选择可能引入额外随机性，建议实验中使用固定秩
3. 子段最小样本量阈值较低（p+2），高维场景下可能不足
4. 段2的初始条件处理是标准做法，但损失断点附近少量信息

### 总体结论

**低秩VAR的Bootstrap实现在算法层面是正确的**。代码忠实地实现了"残差Bootstrap + LR检验"的标准框架，与基准OLS Bootstrap保持了结构一致性，唯一的差异在于参数估计方法（低秩 vs OLS），符合设计文档的意图。上述风险点属于方法论层面的固有特性，不构成代码bug。
