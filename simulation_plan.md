# 高维与低秩VAR模型已知断点检验 — 仿真实验方案

## 一、研究目标与假设框架

### 1.1 核心研究问题

给定一个高维时间序列 Y_t 服从 VAR(p) 模型，以及一个**已知**的候选断点时间 t，检验系数矩阵是否在该时间点发生了结构性变化。

本研究聚焦于**已知断点**的检验场景（如政策变化、市场事件等外生冲击的时间点已知），不涉及断点搜索（Sup-LR）。

### 1.2 假设框架（统一定义 + baseline的Chow口径）

统一的结构变化假设为：

```
H0: Φ₁ = Φ₂ = Φ    序列在整个观测期内结构稳定，不存在结构性变化
H1: Φ₁ ≠ Φ₂         序列在已知时间点t发生了结构性变化
```

在baseline（OLS）中，为使用Chow检验，进一步采用“同一有效样本 + 哑变量交互”的等价写法：

```
H0_baseline: Γ = 0   （等价于断点前后参数相同，B1 = B2）
H1_baseline: Γ ≠ 0   （断点后参数发生变化）

受限模型:    y_t = B x_t + u_t
非受限模型:  y_t = B x_t + Γ(D_t x_t) + u_t
```

其中 `D_t = 1(t >= t0)`，`x_t = [1, y_{t-1}', ..., y_{t-p}']'`，两模型都在统一样本 `t = p, ..., T-1` 上估计（有效样本 `n = T-p`）。

### 1.3 三类系数矩阵结构假设

本研究考虑三种系数矩阵结构，对应三条检验流程：

| 结构假设 | 估计方法 | 适用场景 | 研究定位 |
|---------|---------|---------|---------|
| **无约束**（一般矩阵） | OLS | 低维（N较小） | 基准对照 |
| **稀疏**（多数元素为零） | Lasso / 去偏Lasso | 高维稀疏（N大，Φ稀疏） | **核心贡献一** |
| **低秩**（秩 r ≪ N） | 核范数正则化 / 截断SVD | 高维低秩（N大，少数潜在因子驱动） | **核心贡献二** |

**核心思想**：三种模型共享同一个“结构变化检验 + Bootstrap + 蒙特卡洛”框架；其中baseline以Chow/F为主，高维稀疏与低秩以LR统计量为主，差异主要在于参数估计与统计量构造。

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

### 2.1 已知断点统计量构造（baseline并行F与LR）

#### baseline（OLS）：同一有效样本口径下并行构造 F 与 LR

在统一样本 `t = p, ..., T-1`（`n = T-p`）上构造：

- `X = [x_t]`
- `Z = [D_t x_t]`（断点哑变量与回归量交互）
- 受限模型：`Y = X B + U`
- 非受限模型：`Y = X B + Z Γ + U`

记受限/非受限残差平方和为 `SSR_R` 与 `SSR_U`，则Chow型统计量为：

```
F = ((SSR_R - SSR_U) / q) / (SSR_U / (n - k_U))
```

其中：
- `q` 为约束个数（检验“截距+滞后系数均稳定”时，`q = K(1+Kp)`；仅检验滞后系数时 `q = K^2 p`）
- `k_U` 为非受限模型参数数

在经典OLS条件下，`F` 可直接使用F分布临界值计算p值。

同时，在同一口径（同样本 `n=T-p`）下并行构造LR统计量：

```
LR = n [ln|Σ̂_R| - ln|Σ̂_U|]
```

并给出卡方近似 p 值（`χ²` 渐近）用于与 `F` 检验和Bootstrap结果对照。

#### 稀疏/低秩：保持LR + Bootstrap构造

高维稀疏与低秩流程仍采用LR统计量：

```
LR(t) = 2 × [lnL_U(t) - lnL_R]
```

并通过Bootstrap获取经验零分布与p值，避免非标准渐近分布带来的误判风险。

### 2.2 Bootstrap推断（baseline同时输出F*与LR*）

```
1. 在H0下拟合全样本，得到 Φ̂_R, ĉ_R, 残差 ε̂_R
2. 残差居中：centered_ε = ε̂ - mean(ε̂)
3. 重复 B 次：
   a. 有放回重抽样残差
   b. 用 Φ̂_R, ĉ_R 和重抽样残差生成伪序列 Y*
   c. 计算伪序列统计量：
      - baseline：同时计算 `F*_Chow` 与 `LR*`
      - 高维稀疏/低秩：计算 `LR*(t)`
4. p值 = P(统计量* ≥ 统计量_observed)
```

### 2.3 蒙特卡洛评估

- **第一类错误（Size）**：H0为真时，重复M次生成序列并检验，计算拒绝率。目标：接近名义水平α
- **统计功效（Power）**：H1为真时，重复M次生成含断点序列并检验，计算拒绝率。目标：随Δ增大而增大

### 2.4 统一流程图

```
数据生成 → 参数估计 → baseline: F与LR(同口径并行)；高维: LR(t) → 渐近或Bootstrap p值 → Monte Carlo Size/Power
```

三条流程仅在**参数估计**环节不同，其余环节结构完全一致。


### 2.5 检验统计量分布与高斯假设说明（新增）

当前框架中的“默认高斯”来自两个层面：

1. **DGP误差项**：`ε_t ~ N(0, Σ)`，在代码中通过多元正态抽样实现；
2. **LR对数似然**：`lnL = -T_eff/2 × [N·ln(2π) + ln|Σ̂| + N]`，属于高斯似然写法。

这表示“噪声与评分规则采用高斯口径”，并不要求观测序列 `Y_t` 本身严格正态。

### 2.6 baseline / 稀疏 / 低秩场景下的分布判定

- **baseline（OLS，同口径Chow）**：经典条件下可直接使用F分布；对应LR/Wald可用渐近 `χ²` 近似；
- **稀疏（Lasso）**：存在正则化与变量选择，零假设分布通常非标准，不宜直接使用标准 `χ²`；
- **低秩（秩约束）**：参数受约束，常见非标准渐近，同样不宜直接套用标准 `χ²`。

因此，本文统一以Bootstrap推断为主，渐近分布判定仅作为baseline对照。

### 2.7 旧写法中直接用χ²的偏移来源（适用范围说明）

本节结论**仅适用于旧的baseline实现**（“全样本拟合 vs 两段分别拟合”）：

- `T_eff,R = T - p`
- `T_eff,U = (t-p) + (T-t-p) = T - 2p`

由于非约束模型较约束模型少 `p` 个有效观测，LR中会引入常数偏移项，影响LR绝对值与标准 `χ²` 临界值的直接比较。

对当前采用的“同一有效样本口径”Chow/LR并行实现（见 2.8 节），该偏移问题已被规避。

### 2.8 baseline-Chow实施细则（同口径条件似然）

建议将baseline改为“**同一有效样本口径**”的条件似然/Chow实现：

- 在统一样本 `s = p, ..., T-1` 上建模（`n = T-p`）；
- 设断点指示变量 `D_s = 1(s >= t)`，构造

`H0: y_s = c + Φx_s + u_s`

`H1: y_s = c + Φx_s + D_s(Δc + ΔΦx_s) + u_s`

在该口径下可并行构造：

- LR：`LR = n[ln|Σ̂_R| - ln|Σ̂_U|]`（渐近 `χ²`）
- Chow型F：`F = ((SSR_R - SSR_U)/q) / (SSR_U/(n-k_U))`（经典条件下服从F）

### 2.9 与文献框架一致性（Lütkepohl, 2005）

上述改造与VAR结构稳定性检验的经典写法一致：将已知断点检验转化为线性约束检验（`B1=B2`），构造Wald/LR（或quasi-LR）统计量；
在有限样本或非标准估计场景下，推荐结合Bootstrap提高推断稳健性。

---

## 三、三条检验流程

### 3.1 流程A：基准VAR（OLS估计）— 完整

**定位**：低维基准，验证检验方法的正确性。

| 环节 | 模块 | 文件 | 状态 |
|------|------|------|------|
| 数据生成 | `VARDataGenerator` | `simulation/data_generator.py` | ✅ |
| OLS估计 | `VAREstimator.fit_ols()` | `simulation/var_estimator.py` | ✅ |
| Chow+LR并行检验（同口径） | `ChowTest.compute_at_point()` | `simulation/chow_test.py` | ✅ |
| baseline并行Bootstrap（F*与LR*） | `ChowBootstrapInference.test_at_point()` | `simulation/chow_bootstrap.py` | ✅ |
| LR检验（原有实现，保留对照） | `LRTest.compute_lr_at_point()` | `simulation/sup_lr_test.py` | ✅ |
| Bootstrap | `BootstrapInference.test_at_point()` | `simulation/bootstrap.py` | ✅ |
| 蒙特卡洛 | `MonteCarloSimulation` | `simulation/monte_carlo.py` | ✅ |

适用范围：N = 2~5，T = 100~500

baseline正式推断采用同口径 **F 与 LR 并行报告**：
- 渐近：`asym-F` 与 `asym-χ²(LR)`
- Bootstrap：`boot-F` 与 `boot-LR`
以便系统对比LR检验在不同推断口径下的表现。

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
| I-3 | `χ²/F` vs Bootstrap对照（baseline） | N=2, T=100~300, t=T/2, M≥500, B≥500 |

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

### 4.5 本轮实施框架（与代码脚本一致）

本轮按两部分实施：

1. **检验流程与p值计算对照（baseline）**
   - 明确 `H0/H1` 与统计量构造（同口径下 F 与 LR 并行）
   - 在常规时间序列下并行计算：
     - 渐近 p 值（`F` 分布 / `χ²` 分布）
     - Bootstrap p 值（`F*` / `LR*`）
     - 形成四组对照：`asym-F`、`asym-χ²`、`boot-F`、`boot-LR`
   - 设置 `B` 网格（如 `B={50,100,200,400}`），绘制
     - `p值-B` 关系图（`asym-F`、`asym-χ²`、`boot-F`、`boot-LR`）
     `|p_bootstrap(LR)-p_chi2(LR)|` 随 `B` 的变化图

2. **检验合理性实证验证（三类序列）**
   - 场景：常规（baseline）、低秩、稀疏
   - 指标：第一类错误（Type I Error）与统计功效（Power）
   - 设置 `M` 网格（如 `M={30,60,120}`），绘制不同 `M` 下 Type I Error / Power 曲线
     - 重点输出：`M-Type I Error` 关系图
   - 目标：观察 Monte Carlo 采样次数对检验稳定性的影响

对应实现脚本：`experiments/run_revised_validation_framework.py`

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
│   ├── chow_test.py              # baseline同口径Chow/F检验
│   ├── chow_bootstrap.py         # baseline-Chow的Bootstrap推断
│   ├── bootstrap.py              # LR/Sup-LR Bootstrap推断
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
