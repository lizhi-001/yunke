# 高维与低秩VAR模型已知断点检验 — 仿真实验方案

## 一、研究目标与假设框架

### 1.1 核心研究问题

给定一个高维时间序列 Y_t 服从 VAR(p) 模型，以及一个**已知**的候选断点时间 t，检验系数矩阵是否在该时间点发生了结构性变化。

本研究聚焦于**已知断点**的检验场景（如政策变化、市场事件等外生冲击的时间点已知），不涉及断点搜索（Sup-LR）。

### 1.2 假设框架（统一适用于所有模型类型）

```
H0: Φ₁ = Φ₂ = Φ    序列在整个观测期内结构稳定，不存在结构性变化
H1: Φ₁ ≠ Φ₂         序列在已知时间点t发生了结构性变化
```

### 1.3 三类系数矩阵结构假设

本研究考虑三种系数矩阵结构，对应三条检验流程：

| 结构假设 | 估计方法 | 适用场景 | 研究定位 |
|---------|---------|---------|---------|
| **无约束**（一般矩阵） | OLS | 低维（N较小） | 基准对照 |
| **稀疏**（多数元素为零） | Lasso / 去偏Lasso | 高维稀疏（N大，Φ稀疏） | **核心贡献一** |
| **低秩**（秩 r ≪ N） | 核范数正则化 / 截断SVD | 高维低秩（N大，少数潜在因子驱动） | **核心贡献二** |

**核心思想**：三种模型共享同一个检验框架（LR统计量 + Bootstrap推断 + 蒙特卡洛评估），唯一的区别在于**参数估计方法**。

### 1.4 系数矩阵设计原理

#### 稀疏VAR：随机稀疏矩阵

稀疏矩阵是指大部分元素为零的矩阵。Φ[i,j]≠0 表示变量 j 的滞后值对变量 i 有直接Granger因果影响；稀疏意味着大多数变量对之间没有直接因果关系。

生成方式：
```
Φ = randn(N, N) × 0.3              # 随机高斯元素
mask ~ Bernoulli(0.2)               # 每个位置20%概率保留
Φ_sparse = Φ ⊙ mask                 # 约80%元素置零
```

设计理由：
- 随机稀疏模拟"哪些变量之间有联系是未知的"这一现实情况
- 与Lasso估计的理论假设一致：Lasso在稀疏条件下具有变量选择一致性

#### 低秩VAR：随机低秩矩阵

矩阵的秩（rank）是做初等行变换（消元）后剩余的非零行数，即线性无关的行的数量。

在VAR模型中，Φ 的每一行对应一个变量的方程。rank(Φ) = r 意味着 N 个方程中只有 r 个是真正独立的"基础方程"，其余 N-r 个都是这 r 个基础方程的线性组合。整个系统的动态行为由 r 个独立方向决定。

低秩矩阵可分解为 Φ = U·V'：V' 的每一行提供一个"基础行模式"（行阶梯形中的非零行），U 的每一行提供"组合权重"（第 i 个变量的方程 = Σ U[i,k]×基础行k）。

生成方式：
```
U ∈ R^{N×r}, V ∈ R^{N×r}           # r=2, U为组合权重, V'为基础行模式
U[i,j], V[i,j] ~ N(0, 0.3²)       # 随机高斯元素
Φ = U · V'                          # rank(Φ) ≤ r, 参数量从 N² 降至 2Nr
```

设计理由：
- 参数降维：N=10, r=2 时，行变换后仅2个非零行，有效参数从 100 降至 40
- 截断SVD估计利用此结构：SVD本质上是找最优的行列分解，只保留最大的 r 个奇异值，将 Φ̂ 投影到最接近的秩-r 矩阵上，过滤高维噪声

#### 平稳性保证

三类矩阵均通过伴随矩阵特征值检验（max|λ(C)| < 1）确保平稳性，采用拒绝抽样反复生成直到满足条件。

#### 结构变化（Δ）的施加

功效评估中，断点后系数矩阵通过均匀加性扰动构造：Φ₂ = Φ₁ + Δ·𝟙。Δ 越大，结构变化信号越强，检验功效越高。若 Φ₂ 不满足平稳性则跳过该 Δ 值。

---

## 二、统一检验方法论

### 2.1 已知点LR统计量

```
LR(t) = 2 × [lnL_U(t) - lnL_R]
```

其中：
- `lnL_R`：全样本估计的对数似然（H0：无结构变化）
- `lnL_U(t) = lnL₁(t) + lnL₂(t)`：在已知点t处分段估计的对数似然之和（H1）
- 对数似然公式：`lnL = -T_eff/2 × [N·ln(2π) + ln|Σ̂| + N]`
- 协方差矩阵使用MLE口径：`Σ̂ = (ε'ε) / T_eff`

该LR公式与估计方法无关——无论使用OLS、Lasso还是核范数正则化，对数似然均从残差以相同方式计算。

### 2.2 Bootstrap推断

```
1. 在H0下拟合全样本，得到 Φ̂_R, ĉ_R, 残差 ε̂_R
2. 残差居中：centered_ε = ε̂ - mean(ε̂)
3. 重复 B 次：
   a. 有放回重抽样残差
   b. 用 Φ̂_R, ĉ_R 和重抽样残差生成伪序列 Y*
   c. 计算伪序列的 LR*(t)
4. p值 = P(LR* ≥ LR_observed)
```

### 2.3 蒙特卡洛评估

- **第一类错误（Size）**：H0为真时，重复M次生成序列并检验，计算拒绝率。目标：接近名义水平α
- **统计功效（Power）**：H1为真时，重复M次生成含断点序列并检验，计算拒绝率。目标：随Δ增大而增大

### 2.4 统一流程图

```
数据生成 → 参数估计(OLS/Lasso/NuclearNorm) → LR(t)计算 → Bootstrap p值 → Monte Carlo Size/Power
```

三条流程仅在**参数估计**环节不同，其余环节结构完全一致。

---

## 三、三条检验流程

### 3.1 流程A：基准VAR（OLS估计）— 完整

**定位**：低维基准，验证检验方法的正确性。

| 环节 | 模块 | 文件 | 状态 |
|------|------|------|------|
| 数据生成 | `VARDataGenerator` | `simulation/data_generator.py` | ✅ |
| OLS估计 | `VAREstimator.fit_ols()` | `simulation/var_estimator.py` | ✅ |
| LR检验 | `LRTest.compute_lr_at_point()` | `simulation/sup_lr_test.py` | ✅ |
| Bootstrap | `BootstrapInference.test_at_point()` | `simulation/bootstrap.py` | ✅ |
| 蒙特卡洛 | `MonteCarloSimulation` | `simulation/monte_carlo.py` | ✅ |

适用范围：N = 2~5，T = 100~500

### 3.2 流程B：高维稀疏VAR（Lasso估计）— 完整

**定位**：核心贡献一。系数矩阵稀疏时的高维结构变化检验。

| 环节 | 模块 | 文件 | 状态 |
|------|------|------|------|
| 数据生成 | `VARDataGenerator(sparsity=...)` | `simulation/data_generator.py` | ✅ |
| Lasso估计 | `LassoVAREstimator.fit()` | `sparse_var/lasso_var.py` | ✅ |
| 去偏Lasso | `DebiasedLassoVAR.fit()` | `sparse_var/debiased_lasso.py` | ✅ |
| LR检验 | `SparseLRTest.compute_lr_at_point()` | `sparse_var/sparse_lr_test.py` | ✅ |
| Bootstrap | `SparseBootstrapInference.test()` | `sparse_var/sparse_bootstrap.py` | ✅ |
| 蒙特卡洛 | `SparseMonteCarloSimulation` | `sparse_var/sparse_monte_carlo.py` | ✅ |

适用范围：N = 10~50，T = 200~1000，Φ稀疏

### 3.3 流程C：高维低秩VAR（核范数/截断SVD估计）— 已补全

**定位**：核心贡献二。系数矩阵低秩时的高维结构变化检验。

| 环节 | 模块 | 文件 | 状态 |
|------|------|------|------|
| 数据生成 | `VARDataGenerator.generate_lowrank_phi()` | `simulation/data_generator.py` | ✅ |
| 核范数估计 | `NuclearNormVAR.fit_cvxpy()` | `lowrank_var/nuclear_norm.py` | ✅ |
| 截断SVD | `NuclearNormVAR.fit_svd()` | `lowrank_var/nuclear_norm.py` | ✅ |
| 秩选择 | `RankSelector` | `lowrank_var/rank_selection.py` | ✅ |
| LR检验 | `LowRankLRTest.compute_lr_at_point()` | `lowrank_var/lowrank_lr_test.py` | ✅ |
| Bootstrap | `LowRankBootstrapInference.test()` | `lowrank_var/lowrank_bootstrap.py` | ✅ |
| 蒙特卡洛 | `LowRankMonteCarloSimulation` | `lowrank_var/lowrank_monte_carlo.py` | ✅ |

适用范围：N = 8~50，T = 200~1000，Φ低秩（秩 r ≪ N）

---

## 四、仿真实验设计

### 4.1 实验组I：基准VAR检验（验证方法正确性）

| 实验 | 内容 | 参数 |
|------|------|------|
| I-1 | Type I Error（H0为真） | N=2, T=100, t=50, M=50, B=100 |
| I-2 | Power曲线（H1为真） | N=2, T=100, t=50, Δ=[0.1,0.2,0.3] |

### 4.2 实验组II：高维稀疏VAR检验（核心贡献一）

| 实验 | 内容 | 参数 |
|------|------|------|
| II-1 | Lasso估计精度 | N=10, T=300, sparsity=0.2 |
| II-2 | LR检验演示（H0为真） | N=10, T=300, t=150, B=50 |
| II-3 | Type I Error | N=10, T=300, t=150, M=30, B=50 |
| II-4 | Power评估 | N=10, T=300, t=150, Δ=[0.1,0.2,0.3] |

### 4.3 实验组III：高维低秩VAR检验（核心贡献二）

| 实验 | 内容 | 参数 |
|------|------|------|
| III-1 | 低秩估计 + 秩选择 | N=8, T=250, rank=2 |
| III-2 | LR检验演示（H0为真） | N=8, T=250, t=125, B=50 |
| III-3 | Type I Error | N=8, T=250, t=125, M=30, B=50 |
| III-4 | Power评估 | N=8, T=250, t=125, Δ=[0.1,0.2,0.3] |

### 4.4 实验组IV：跨模型对比分析

| 实验 | 内容 |
|------|------|
| IV-1 | Size对比：三种模型在相同(N,T)下的第一类错误率 |
| IV-2 | Power对比：三种模型在相同(N,T,Δ)下的功效曲线 |
| IV-3 | 维度扩展：N增大时各模型性能变化 |

---

## 五、实验参数设计

### 5.1 仿真控制参数

| 参数 | 演示值 | 正式值 | 说明 |
|------|--------|--------|------|
| M（蒙特卡洛次数） | 30-50 | 1000-5000 | 评估Size/Power |
| B（Bootstrap次数） | 50-100 | 500-1000 | 计算p值 |
| α（显著性水平） | 0.05 | 0.05 | 标准 |
| seed | 42 | 42 | 可复现 |

### 5.2 模型参数

| 参数 | 测试值 | 说明 |
|------|--------|------|
| N（维度） | 2, 5, 8, 10, 20 | N=2,5为基准；N=8,10,20为高维 |
| T（样本长度） | 100, 200, 300, 500 | 需满足 T ≫ N×p |
| p（滞后阶数） | 1, 2 | p=1为主 |
| t（已知断点） | T/2 | 样本中点 |

### 5.3 结构变化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Δ（变化强度） | 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 | 功效曲线 |
| sparsity（稀疏度） | 0.1, 0.2, 0.3 | 非零元素比例 |
| rank（秩） | 1, 2, 3 | 低秩模型的真实秩 |

---

## 六、代码结构

```
yunke/
├── simulation/                    # 流程A：基准VAR（完整）
│   ├── data_generator.py         # 数据生成（三条流程共用）
│   ├── var_estimator.py          # OLS估计
│   ├── sup_lr_test.py            # LR检验（LRTest类）
│   ├── bootstrap.py              # Bootstrap推断
│   └── monte_carlo.py            # 蒙特卡洛仿真
├── sparse_var/                    # 流程B：高维稀疏VAR（完整）
│   ├── lasso_var.py              # Lasso估计
│   ├── debiased_lasso.py         # 去偏Lasso
│   ├── cv_tuning.py              # 交叉验证调参
│   ├── sparse_lr_test.py         # 稀疏LR检验
│   ├── sparse_bootstrap.py       # 稀疏Bootstrap
│   └── sparse_monte_carlo.py     # 稀疏蒙特卡洛
├── lowrank_var/                   # 流程C：高维低秩VAR（完整）
│   ├── nuclear_norm.py           # 核范数正则化估计
│   ├── rank_selection.py         # 秩选择
│   ├── lowrank_lr_test.py        # 低秩LR检验
│   ├── lowrank_bootstrap.py      # 低秩Bootstrap
│   └── lowrank_monte_carlo.py    # 低秩蒙特卡洛
├── experiments/                   # 实验脚本
├── results/                       # 实验结果
├── main.py                        # 主程序入口
└── simulation_plan.md            # 本文档
```

---

## 七、参考文献

1. Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856.
2. Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78.
3. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267-288.
4. Krampe, J., Kreiss, J. P., & Paparoditis, E. (2021). Bootstrap based inference for sparse high-dimensional time series models. *Bernoulli*, 27(3), 1441-1466.
5. Hou, J. & Zhang, Z. (2019). Low-rank VAR estimation via nuclear norm regularization.
