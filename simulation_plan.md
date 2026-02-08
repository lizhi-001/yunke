# VAR模型结构性变化检验 - 仿真实现方案

## 项目概述

基于开题报告《高维向量自回归模型的结构性变化检验》，本项目实现了完整的结构性变化检验仿真系统。**整体实现包含两套完整的检验流程**：

1. **正常VAR模型检验**：基于传统OLS估计的VAR模型，适用于低维场景（N较小）
2. **高维稀疏VAR模型检验**：基于Lasso/去偏Lasso估计的VAR模型，适用于高维稀疏场景（N较大，系数矩阵稀疏）

两套流程均包含：参数估计 → LR统计量计算 → Bootstrap推断 → 蒙特卡洛评估（第一类错误/功效）。

---

## 代码与开题报告对应关系

### 对应关系总览表

| 开题报告章节 | 对应代码模块 | 文件路径 | 说明 |
|-------------|-------------|----------|------|
| 研究方法 1. VAR(P)模型 | `VAREstimator` | `simulation/var_estimator.py` | OLS参数估计、残差协方差、对数似然 |
| 研究方法 2. 似然比检验LR | `SupLRTest` | `simulation/sup_lr_test.py` | LR统计量计算 |
| 研究方法 3. Bootstrap | `BootstrapInference` | `simulation/bootstrap.py` | 残差重抽样、伪序列生成 |
| 研究方案 1. Sup-LR检验流程 | `SupLRTest` + `BootstrapInference` | `sup_lr_test.py` + `bootstrap.py` | 完整检验流程 |
| 研究方案 2.1 第一类错误 | `MonteCarloSimulation.evaluate_type1_error` | `simulation/monte_carlo.py` | H0为真时的拒绝率 |
| 研究方案 2.2 第二类错误 | `MonteCarloSimulation.evaluate_power` | `simulation/monte_carlo.py` | H1为真时的功效 |
| 高维稀疏VAR | `LassoVAREstimator`, `DebiasedLassoVAR` | `sparse_var/` | Lasso估计、去偏推断 |
| 低秩VAR | `NuclearNormVAR`, `RankSelector` | `lowrank_var/` | 核范数正则化、秩选择 |

---

### 详细对应说明

#### 1. VAR(P)模型参数估计 → `var_estimator.py`

**开题报告原文（研究方法 1）**：
```
(1) 参数估计：拟合含有N个变量的滞后p期的VAR(p)模型。对每个模型单独使用最小二乘法(OLS)进行参数估计。
(2) 求解残差协方差矩阵
(3) 计算对数似然值
```

**代码实现**：
```python
# var_estimator.py 中的 VAREstimator.fit_ols() 方法

def fit_ols(self, Y, p, include_const=True):
    # (1) OLS参数估计：使用lstsq替代显式求逆，提升数值稳定性
    # B = (X'X)^{-1} X'Y 等价于求解 X @ B = Y 的最小二乘解
    B_hat, _, _, _ = lstsq(X, Y_response, rcond=None)

    # (2) 残差协方差矩阵（MLE口径，分母为T_eff）
    self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff

    # (3) 对数似然值：lnL ∝ -T_eff/2 * ln(|Σ̂|)
    self.log_likelihood = self._compute_log_likelihood(T_eff, N, self.Sigma_hat)
```

---

#### 2. 似然比检验LR → `sup_lr_test.py`

**开题报告原文（研究方法 2）**：
```
Step1：明确原假设和备择假设
  - H0：序列在整个观测期内结构稳定，不存在结构性变化
  - H1：序列在某个未知的时间点t发生了结构性变化
Step2：在原假设和备择假设下进行分别参数估计，计算各自的最大对数似然值 lnL_R 和 lnL_U
Step3：构造对数似然比统计量 G = -2ln(λ) = 2[lnL_U - lnL_R]
```

**代码实现**：
```python
# sup_lr_test.py 中的 SupLRTest.compute_lr_at_point() 方法

def compute_lr_at_point(self, Y, p, t):
    """
    针对特定时间点t检验是否发生结构性变化

    参数:
        Y: 观测序列
        p: VAR滞后阶数
        t: 待检验的变点位置
    """
    # 边界验证：确保分段样本量足够
    min_segment_size = p + 2
    if t < min_segment_size or t > len(Y) - min_segment_size:
        raise ValueError(f"变点位置t={t}无效，需满足分段样本量要求")

    # Step1: H0 - 无结构变化（约束模型）
    result_r = estimator_r.fit_ols(Y, p)
    log_lik_r = result_r['log_likelihood']

    # Step2: H1 - 在时间点t发生结构变化（分段拟合）
    log_lik_1 = estimator1.fit_ols(Y[:t], p)['log_likelihood']
    log_lik_2 = estimator2.fit_ols(Y[t:], p)['log_likelihood']
    log_lik_u = log_lik_1 + log_lik_2

    # Step3: LR统计量
    lr = 2 * (log_lik_u - log_lik_r)

    return lr
```

**说明**：与传统Sup-LR检验遍历所有可能断点不同，本实现针对特定的时间点t进行检验，判断该点是否发生了结构性变化。这种方法适用于已知或怀疑某个特定时间点可能存在结构变化的场景。

**断点口径说明（实现与开题报告的差异）**：
当前实现对断点两侧分别**独立估计**VAR(p)模型，第二段样本从 `τ` 开始，因此需要使用该段自身的 `p` 个滞后观测作为初始值，导致第二段有效样本长度实际为 `T - τ - p`。这与开题报告中将第二段有效样本记为 `T - τ` 的口径不完全一致。若需严格对齐报告推导，应考虑复用断点前的最后 `p` 个观测作为第二段初始值，并在设计矩阵构造中显式处理该口径差异（对应 `improvement_todo.md` 的待办项）。

---

#### 3. Bootstrap方法 → `bootstrap.py`

**开题报告原文（研究方法 3）**：
```
Step1：在原有的样本中通过重抽样抽取一定数量的新样本
Step2：基于产生的新样本，计算我们需要估计的统计量
Step3：重复上述步骤n次（一般是n>1000次），利用得到的n个值，拟合目标统计量的总体分布
```

**代码实现**：
```python
# bootstrap.py 中的 BootstrapInference 类

def generate_pseudo_series(self, Y, p, Phi, c, residuals):
    # Step1: 残差重抽样（有放回）并居中处理
    # 居中处理确保Bootstrap残差均值为0，减少小样本偏移
    centered_residuals = residuals - np.mean(residuals, axis=0)
    indices = np.random.choice(T_eff, size=T_eff, replace=True)
    resampled_residuals = centered_residuals[indices, :]

    # 生成伪序列：Y*_t = c + Φ Y*_{t-1} + ε*_t
    for t in range(p, T):
        Y_star[t] = c + Phi @ Y_lag + resampled_residuals[t-p]
    return Y_star

def bootstrap_lr(self, Y, p, t):
    for b in range(self.B):  # Step3: 重复B次
        Y_star = self.generate_pseudo_series(...)  # Step1
        lr_star = compute_lr_at_point(Y_star, t)   # Step2: 针对特定点t计算LR
        bootstrap_lr_values.append(lr_star)
```

---

#### 4. 结构变化检验流程 → `sup_lr_test.py` + `bootstrap.py`

**开题报告原文（研究方案 1）**：
```
1.1 确定原假设和备择假设
    - H0：序列在整个观测期内结构稳定
    - H1：序列在某个未知的时间点t发生了结构性变化
1.2 计算H0和H1下的最大似然值
1.3 计算似然比检验统计量 LR(t) = 2[lnL_U(t) - lnL_R]
1.4 Bootstrap估计p值或拒绝域
1.5 做出决策
```

**代码实现**：
```python
# 完整检验流程
from simulation import BootstrapInference

bootstrap = BootstrapInference(B=500)
# 针对特定时间点t进行检验
result = bootstrap.test(Y, p=1, t=100, alpha=0.05)

# 内部执行流程：
# 1.1-1.3: sup_lr_test.compute_lr_at_point() 计算针对时间点t的LR统计量
# 1.4: bootstrap.bootstrap_lr() 构建经验分布，计算p值
# 1.5: 比较p值与α，做出决策
print(result['decision'])  # "拒绝H0：在时间点t存在结构性变化" 或 "接受H0"
```

---

#### 5. 第一类错误评估 → `monte_carlo.py`

**开题报告原文（研究方案 2.1）**：
```
Step1: 理论模型设定（H0为真）- 预先设定一个结构稳定且平稳的VAR(p)模型
Step2: 蒙特卡洛循环 - 重复M次生成仿真序列并执行检验
Step3: 估计第一类错误 P(拒绝H0|H0为真) = #{Sup-LR > C_α} / M
```

**代码实现**：
```python
# monte_carlo.py 中的 MonteCarloSimulation.evaluate_type1_error()

def evaluate_type1_error(self, N, T, p, Phi, Sigma, alpha=0.05):
    successful_iterations = 0  # 跟踪成功迭代次数

    for m in range(self.M):  # Step2: 蒙特卡洛循环
        # Step1: 生成无结构变化的序列（H0为真）
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)

        # 执行Bootstrap Sup-LR检验
        result = bootstrap.test(Y, p, alpha=alpha)
        successful_iterations += 1
        if result['reject_h0']:
            rejections += 1

    # Step3: 计算第一类错误率（使用成功迭代次数作为分母）
    type1_error = rejections / successful_iterations
    return {'type1_error': type1_error, 'M_effective': successful_iterations}
```

---

#### 6. 第二类错误与功效评估 → `monte_carlo.py`

**开题报告原文（研究方案 2.2）**：
```
Step1: 理论模型设定（H1为真）- 设定存在结构变化的VAR(p)模型
Step2: 蒙特卡洛循环 - 生成含断点序列并执行检验
Step3: 估计第二类错误和统计功效 Power = 1 - P(接受H0|H1为真)
Step4: 观察功效随Δ增加的变化
```

**代码实现**：
```python
# monte_carlo.py 中的 MonteCarloSimulation.evaluate_power()

def evaluate_power(self, N, T, p, Phi1, Phi2, Sigma, break_point, alpha=0.05):
    successful_iterations = 0  # 跟踪成功迭代次数

    for m in range(self.M):
        # Step1: 生成含断点的序列（H1为真）
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)

        # 执行检验
        result = bootstrap.test(Y, p, alpha=alpha)
        successful_iterations += 1
        if result['reject_h0']:
            rejections += 1

    # Step3: 计算功效（使用成功迭代次数作为分母）
    power = rejections / successful_iterations
    return {'power': power, 'M_effective': successful_iterations}

# Step4: 功效曲线
def power_curve(self, ..., delta_values):
    for delta in delta_values:
        Phi2 = Phi_base + delta  # 不同强度的结构变化
        power = self.evaluate_power(...)['power']
        powers.append(power)
```

---

#### 7. 高维稀疏VAR完整检验流程 → `sparse_var/`

**开题报告原文（国内外研究进展 3.1）**：
```
稀疏VAR模型假设系数矩阵中只有少数非零元素，通过Lasso估计量进行变量选择和稀疏化。
Krampe等人将去偏Lasso估计量推广到高维VAR模型中，实现有效的参数推断和假设检验。
```

**完整检验流程**：

高维稀疏VAR的结构变化检验流程与正常VAR类似，但在参数估计环节使用Lasso/去偏Lasso替代OLS：

```
1. 参数估计：使用Lasso-VAR或去偏Lasso-VAR进行稀疏估计
2. LR统计量计算：基于稀疏估计结果计算似然比统计量
3. Bootstrap推断：残差重抽样构建经验分布
4. 蒙特卡洛评估：评估第一类错误和功效
```

**代码实现**：

```python
# sparse_var/lasso_var.py - Lasso-VAR估计
class LassoVAREstimator:
    def fit(self, Y, p):
        for i in range(N):
            model = LassoCV(cv=self.cv)  # 交叉验证选择λ
            model.fit(X, y_i)
            # 实现稀疏化
        return {
            'Phi': self.Phi_hat,
            'residuals': self.residuals,
            'log_likelihood': self._compute_log_likelihood()
        }

# sparse_var/debiased_lasso.py - 去偏Lasso（假设检验）
class DebiasedLassoVAR:
    def fit(self, Y, p):
        # β_debiased = β_lasso + (X'X)^{-1} X' (Y - X β_lasso)
        B_debiased = B_lasso + XtX_inv @ X.T @ residuals_lasso
        return {
            'Phi': B_debiased,
            'residuals': self.residuals,
            'log_likelihood': self._compute_log_likelihood()
        }

# sparse_var/sparse_lr_test.py - 高维稀疏VAR的LR检验
class SparseLRTest:
    def compute_lr_at_point(self, Y, p, t, estimator_type='lasso'):
        """
        使用稀疏估计方法计算LR统计量

        参数:
            estimator_type: 'lasso' 或 'debiased_lasso'
        """
        if estimator_type == 'lasso':
            estimator = LassoVAREstimator()
        else:
            estimator = DebiasedLassoVAR()

        # H0: 无结构变化
        result_r = estimator.fit(Y, p)
        log_lik_r = result_r['log_likelihood']

        # H1: 在时间点t发生结构变化
        log_lik_1 = estimator.fit(Y[:t], p)['log_likelihood']
        log_lik_2 = estimator.fit(Y[t:], p)['log_likelihood']
        log_lik_u = log_lik_1 + log_lik_2

        return 2 * (log_lik_u - log_lik_r)

# sparse_var/sparse_bootstrap.py - 高维稀疏VAR的Bootstrap推断
class SparseBootstrapInference:
    def test(self, Y, p, t, alpha=0.05, estimator_type='lasso'):
        """高维稀疏VAR的完整检验流程"""
        # 1. 计算原始LR统计量
        lr_test = SparseLRTest()
        lr_observed = lr_test.compute_lr_at_point(Y, p, t, estimator_type)

        # 2. Bootstrap构建经验分布
        bootstrap_lr_values = []
        for b in range(self.B):
            Y_star = self.generate_pseudo_series(Y, p, ...)
            lr_star = lr_test.compute_lr_at_point(Y_star, p, t, estimator_type)
            bootstrap_lr_values.append(lr_star)

        # 3. 计算p值并做出决策
        p_value = np.mean(np.array(bootstrap_lr_values) >= lr_observed)
        reject_h0 = p_value < alpha

        return {
            'lr_statistic': lr_observed,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'decision': "拒绝H0：在时间点t存在结构性变化" if reject_h0 else "接受H0"
        }

# sparse_var/sparse_monte_carlo.py - 高维稀疏VAR的蒙特卡洛评估
class SparseMonteCarloSimulation:
    def evaluate_type1_error(self, N, T, p, Phi, Sigma, t, alpha=0.05):
        """评估高维稀疏VAR检验的第一类错误"""
        bootstrap = SparseBootstrapInference(B=self.B)
        rejections = 0

        for m in range(self.M):
            Y = generator.generate_sparse_var_series(T, N, p, Phi, Sigma)
            result = bootstrap.test(Y, p, t, alpha)
            if result['reject_h0']:
                rejections += 1

        return {'type1_error': rejections / self.M}

    def evaluate_power(self, N, T, p, Phi1, Phi2, Sigma, break_point, alpha=0.05):
        """评估高维稀疏VAR检验的功效"""
        bootstrap = SparseBootstrapInference(B=self.B)
        rejections = 0

        for m in range(self.M):
            Y, _ = generator.generate_sparse_var_with_break(
                T, N, p, Phi1, Phi2, Sigma, break_point
            )
            result = bootstrap.test(Y, p, break_point, alpha)
            if result['reject_h0']:
                rejections += 1

        return {'power': rejections / self.M}
```

**使用示例**：
```python
# 高维稀疏VAR的完整检验流程
from sparse_var import SparseBootstrapInference

bootstrap = SparseBootstrapInference(B=500)
result = bootstrap.test(Y, p=1, t=100, alpha=0.05, estimator_type='debiased_lasso')
print(result['decision'])
```

---

#### 8. 低秩VAR → `lowrank_var/`

**开题报告原文（国内外研究进展 3.1）**：
```
低秩VAR模型的系数矩阵具有低秩结构。Hou and Zhang (2019) 提出了基于核范数正则化的低秩VAR模型估计方法。
由于Low-Rank模型的参数空间是非凸的，传统渐近理论完全失效，需要Bootstrap进行推断。
```

**代码实现**：
```python
# lowrank_var/nuclear_norm.py - 核范数正则化
class NuclearNormVAR:
    def fit_cvxpy(self, Y, p):
        # 目标函数：min ||Y - XΦ'||_F^2 / (2T) + λ ||Φ||_*
        loss = cp.sum_squares(Y_response - Y_pred) / (2 * T_eff)
        reg = self.lambda_nuc * cp.normNuc(Phi)  # 核范数惩罚
        problem = cp.Problem(cp.Minimize(loss + reg))

# lowrank_var/rank_selection.py - 秩选择
class RankSelector:
    def select_by_information_criterion(self, Y, p, criterion='bic'):
        # 使用BIC选择最优秩
```

---

## 数据集生成与复用策略

### 数据集复用分析

在蒙特卡洛仿真中，数据集生成是**计算密集型**操作。以下分析各场景下的数据复用可行性：

#### 场景1：第一类错误评估（H0为真）

**是否可复用**：❌ **不可复用**

**原因**：
- 每次蒙特卡洛迭代需要**独立的随机序列**
- 复用数据会导致统计量的方差估计偏差
- 第一类错误率的准确估计依赖于独立重复

**代码逻辑**：
```python
def evaluate_type1_error(self, ...):
    for m in range(self.M):
        # 每次迭代生成新的独立序列
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)
        # 不能复用Y
```

---

#### 场景2：第二类错误/功效评估（H1为真）

**是否可复用**：❌ **不可复用**

**原因**：
- 同样需要独立的随机序列来估计功效
- 断点位置固定，但残差需要独立生成

---

#### 场景3：Bootstrap内部循环

**是否可复用**：⚠️ **部分可复用**

**可复用部分**：
- H0下的估计结果（Φ̂_R, ĉ_R, 残差）
- 这些在Bootstrap循环开始前计算一次即可

**不可复用部分**：
- 每次Bootstrap迭代的伪序列Y*必须独立生成

**代码逻辑**：
```python
def bootstrap_sup_lr(self, Y, p):
    # ✅ 可复用：H0估计结果（只计算一次）
    result_r = sup_lr_test.compute_sup_lr(Y, p)
    Phi_r = result_r['restricted_result']['Phi']
    residuals_r = result_r['restricted_result']['residuals']

    for b in range(self.B):
        # ❌ 不可复用：每次生成新的伪序列
        Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)
```

---

#### 场景4：功效曲线（不同Δ值）

**是否可复用**：⚠️ **部分可复用**

**可复用部分**：
- 基准系数矩阵 Φ_base
- 残差协方差矩阵 Σ
- 随机种子（用于结果可复现）

**不可复用部分**：
- 不同Δ值对应不同的Φ2，需要重新生成序列

**优化策略**：
```python
def power_curve(self, ..., delta_values):
    # ✅ 复用基准参数
    Phi_base = ...  # 只生成一次
    Sigma = ...     # 只生成一次

    for delta in delta_values:
        # ❌ 每个delta需要新的序列
        Phi2 = Phi_base + delta
        # 生成新序列并评估功效
```

---

### 数据复用优化建议

#### 1. 预生成随机种子序列
```python
# 预生成M个随机种子，确保可复现性
seeds = np.random.randint(0, 2**31, size=M)

for m in range(M):
    np.random.seed(seeds[m])  # 使用预定义种子
    Y = generator.generate_var_series(...)
```

#### 2. 缓存H0估计结果
```python
class BootstrapInference:
    def __init__(self):
        self._cached_h0_result = None

    def bootstrap_sup_lr(self, Y, p):
        # 缓存H0结果，避免重复计算
        if self._cached_h0_result is None:
            self._cached_h0_result = sup_lr_test.compute_sup_lr(Y, p)

        Phi_r = self._cached_h0_result['restricted_result']['Phi']
        # ...
```

#### 3. 并行化数据生成
```python
from multiprocessing import Pool

def generate_and_test(seed):
    np.random.seed(seed)
    Y = generator.generate_var_series(...)
    result = bootstrap.test(Y, p)
    return result['reject_h0']

# 并行执行M次仿真
with Pool(processes=4) as pool:
    results = pool.map(generate_and_test, seeds)
```

#### 4. 批量预生成数据（仅用于调试）
```python
# 仅用于调试和开发阶段，正式实验不建议使用
def pregenerate_datasets(M, T, N, p, Phi, Sigma, seed=42):
    np.random.seed(seed)
    datasets = []
    for m in range(M):
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)
        datasets.append(Y)
    return datasets

# 警告：正式实验中不应复用这些数据集
```

---

### 数据复用总结表

| 场景 | 可复用 | 说明 |
|------|--------|------|
| 蒙特卡洛循环中的Y | ❌ | 每次迭代需独立序列 |
| Bootstrap循环中的Y* | ❌ | 每次迭代需独立伪序列 |
| H0估计结果（Φ̂_R, 残差） | ✅ | Bootstrap内可复用 |
| 基准参数（Φ_base, Σ） | ✅ | 功效曲线中可复用 |
| 随机种子 | ✅ | 用于结果可复现 |
| 设计矩阵X的结构 | ✅ | 相同T,N,p时可复用 |

---

## 目录结构

```
yunke/
├── simulation/                    # 核心仿真模块
│   ├── __init__.py
│   ├── data_generator.py         # VAR数据生成器
│   ├── var_estimator.py          # VAR模型OLS估计
│   ├── sup_lr_test.py            # Sup-LR检验统计量
│   ├── bootstrap.py              # Bootstrap推断
│   └── monte_carlo.py            # 蒙特卡洛仿真
├── sparse_var/                    # 高维稀疏VAR模块
│   ├── __init__.py
│   ├── lasso_var.py              # Lasso-VAR估计
│   ├── debiased_lasso.py         # 去偏Lasso（假设检验）
│   └── cv_tuning.py              # 交叉验证调参
├── lowrank_var/                   # 低秩VAR模块
│   ├── __init__.py
│   ├── nuclear_norm.py           # 核范数正则化估计
│   └── rank_selection.py         # 秩选择
├── experiments/                   # 实验脚本目录
├── results/                       # 实验结果目录
├── main.py                        # 主程序入口
└── simulation_plan.md            # 本文档
```

## 依赖包

### 必需依赖
```bash
pip install numpy scipy
```

### 高维稀疏估计（推荐）
```bash
pip install scikit-learn    # Lasso-VAR
pip install cvxpy            # 核范数正则化
```

## 实验参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| M (蒙特卡洛次数) | 1000-5000 | 评估Size/Power |
| B (Bootstrap次数) | 500-1000 | 计算p值 |
| trim | 0.15 | 断点搜索范围 |
| α | 0.05 | 显著性水平 |
| T | 200-500 | 样本长度 |
| N | 2-20 | 变量数量 |
| p | 1-2 | 滞后阶数 |

## 参考文献

1. Andrews (1993) - Sup-LR检验理论
2. Bai-Perron (1998) - 多重断裂点检验
3. Tibshirani (1996) - Lasso估计
4. Krampe et al. (2021) - 高维VAR的Bootstrap推断
5. Hou and Zhang (2019) - 低秩VAR估计

---

## 代码改进记录（2026-02-05）

### 已完成改进

| 改进项 | 文件 | 修改内容 | 改进原因 |
|--------|------|----------|----------|
| **OLS数值稳定性** | `var_estimator.py:91` | 使用`lstsq`替代`inv(X'X)`显式求逆 | 提升高维或共线场景的数值鲁棒性 |
| **Bootstrap残差居中** | `bootstrap.py:59` | 添加`centered_residuals = residuals - np.mean(residuals, axis=0)` | 确保Bootstrap残差均值为0，减少小样本偏移 |
| **异常样本统计** | `monte_carlo.py:71,175` | 使用`successful_iterations`作为分母计算错误率/功效 | 避免失败迭代影响统计量估计 |
| **断点搜索边界** | `sup_lr_test.py:50` | 添加边界验证，确保搜索范围有效 | 防止trim与p组合导致空LR序列 |
| **似然协方差口径** | `var_estimator.py:107` | 统一使用MLE口径（分母为`T_eff`） | 与对数似然公式假设一致 |

### 待实施改进

| 改进项 | 类别 | 暂未实施原因 | 建议时机 |
|--------|------|--------------|----------|
| 断点样本切分口径统一 | 理论待确认 | 涉及VAR分段估计理论细节，需确认是否复用第一段末尾观测 | 理论推导完成后 |
| 稀疏/低秩嵌入Sup-LR | 理论待确认 | 需解决高维似然定义、Bootstrap有效性、渐近分布等理论问题 | 理论推导完成后 |
| 高维检验统计量口径 | 理论待确认 | 惩罚似然比/去偏Wald/预测误差检验等方案需根据研究方向确定 | 理论推导完成后 |
| 提升仿真规模 | 参数调整 | 当前演示规模用于验证流程，正式实验时直接修改参数即可 | 正式实验阶段 |
| 扩展Δ与T网格 | 参数调整 | 框架已支持任意组合，根据计算资源规划网格密度 | 正式实验阶段 |
| 固定随机种子策略 | 参数调整 | 当前已支持seed参数，更精细管理对统计有效性影响较小 | 后续优化 |

### 改进详情

#### 1. OLS数值稳定性改进

**改进前**：
```python
XtX_inv = inv(X.T @ X)
B_hat = XtX_inv @ X.T @ Y_response
```

**改进后**：
```python
B_hat, _, _, _ = lstsq(X, Y_response, rcond=None)
```

**说明**：`lstsq`使用SVD分解求解最小二乘问题，避免了显式计算矩阵逆，在设计矩阵接近奇异或条件数较大时更加稳定。

#### 2. Bootstrap残差居中改进

**改进前**：
```python
indices = np.random.choice(T_eff, size=T_eff, replace=True)
resampled_residuals = residuals[indices, :]
```

**改进后**：
```python
centered_residuals = residuals - np.mean(residuals, axis=0)
indices = np.random.choice(T_eff, size=T_eff, replace=True)
resampled_residuals = centered_residuals[indices, :]
```

**说明**：残差居中是Bootstrap的标准做法，确保重抽样残差的期望为零，特别是在小样本或无常数项模型中可减少偏移。

#### 3. 异常样本统计改进

**改进前**：
```python
type1_error = rejections / self.M
```

**改进后**：
```python
type1_error = rejections / successful_iterations if successful_iterations > 0 else np.nan
```

**说明**：当部分蒙特卡洛迭代因数值问题失败时，使用成功迭代次数作为分母可得到更准确的统计量估计。

#### 4. 断点搜索边界验证

**新增代码**：
```python
min_segment_size = p + 2
if tau_min < min_segment_size:
    tau_min = min_segment_size
if tau_max > T - min_segment_size:
    tau_max = T - min_segment_size
if tau_min >= tau_max:
    raise ValueError(f"断点搜索范围无效: tau_min={tau_min}, tau_max={tau_max}...")
```

**说明**：确保每个分段至少有`p+2`个观测用于VAR估计，避免因样本过小导致估计失败。

#### 5. 似然协方差口径统一

**改进前**：
```python
L = N * p + (1 if include_const else 0)
self.Sigma_hat = (self.residuals.T @ self.residuals) / (T_eff - L)  # 无偏估计
```

**改进后**：
```python
self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff  # MLE估计
```

**说明**：对数似然公式假设协方差矩阵为MLE估计（分母为T_eff），统一口径后似然比统计量的计算更加一致。

---

详细改进清单请参见 `improvement_todo.md`。
