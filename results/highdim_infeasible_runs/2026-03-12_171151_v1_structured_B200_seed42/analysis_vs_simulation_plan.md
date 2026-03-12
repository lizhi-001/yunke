# 真高维实验对照分析（结合 `simulation_plan.md`）

## 1. 实验上下文

- 运行目录：`results/highdim_infeasible_runs/2026-03-12_171151_v1_structured_B200_seed42`
- 当前运行已完成：`72/72`，见 `progress/summary.json`
- 本次配置：`seed=42`、`B=200`、`power_M=500`
- 说明：这不是 `simulation_plan.md` Section 11 中的正式配置。正式配置是 `B=500` 且双 seed；因此本分析应视为**探索性/阶段性结论**，不能直接替代论文主结果。

## 2. 文档中的核心预期

根据 `simulation_plan.md` Section 11.10–11.11，本实验的关键预期为：

1. `baseline_ols_f_n20/n30` 在欠定场景下应出现 **size 爆炸**，即 `size@M=2000 ≈ 1.0`
2. `sparse_lasso_*` 与 `lowrank_svd_*` 应保持 **良好 size 控制**，即 `size@M=2000 ∈ [0.03, 0.07]`
3. `Lasso` 与 `SVD` 的 power 应随 `delta` **单调增加**
4. `actual_fro` 应接近 `target_fro`
5. `perturbation_type` 应正确记录为 `uniform_ones / sparse / lowrank`

## 3. 本次观测结果

### 3.1 Size（重点看 `M=2000`）

| 模型 | 文档预期 | 本次结果 | 判断 |
|---|---|---:|---|
| `baseline_ols_f_n20` | ≈ 1.0 | 0.0480 | **不符合预期** |
| `baseline_ols_f_n30` | ≈ 1.0 | 0.0490 | **不符合预期** |
| `sparse_lasso_n20` | [0.03, 0.07] | 0.0580 | **符合预期** |
| `sparse_lasso_n30` | [0.03, 0.07] | 0.1020 | **不符合预期** |
| `lowrank_svd_n20` | [0.03, 0.07] | 0.1675 | **明显不符合预期** |
| `lowrank_svd_n30` | [0.03, 0.07] | 0.2105 | **明显不符合预期** |

初步结论：

- 只有 `sparse_lasso_n20` 的 size 控制达到了文档阈值。
- `sparse_lasso_n30` 已出现明显 size 膨胀。
- 两个 `lowrank_svd` 模型的 size 明显偏高，且偏离幅度较大。
- 最关键的偏差在于：`baseline_ols_f_*` **没有**表现出文档中预期的“size → 1”崩溃。

### 3.2 Power 单调性

| 模型 | 是否单调 | 备注 |
|---|---|---|
| `baseline_ols_f_n20` | 否 | 波动明显 |
| `baseline_ols_f_n30` | 是 | 本次唯一严格单调 |
| `sparse_lasso_n20` | 否 | 小 `delta` 区间波动 |
| `sparse_lasso_n30` | 否 | 小 `delta` 区间波动 |
| `lowrank_svd_n20` | 否 | 整体上升，但非严格单调 |
| `lowrank_svd_n30` | 否 | 整体上升，但非严格单调 |

解释：

- 在 `seed=42`、`B=200` 的单次探索配置下，power 曲线出现波动是可能的。
- 但从论文论证角度，当前结果**还不足以支撑**“结构化方法 power 随 `delta` 单调增加”的强结论。
- 尤其当 size 已经显著偏高时，power 的抬升可能部分反映的是过拒绝，而不是有效检验能力提升。

### 3.3 扰动幅度与扰动类型

- `actual_fro` 基本与 `target_fro` 一致，当前结果支持“扰动幅度实现正确”。
- `perturbation_type` 记录也符合预期：
  - `baseline_ols_f_*` → `uniform_ones`
  - `sparse_lasso_*` → `sparse`
  - `lowrank_svd_*` → `lowrank`

因此，**扰动路由本身看起来不是主要问题**。

## 4. 与 `simulation_plan.md` 的一致性判断

### 4.1 符合的部分

- 实验设计层面与文档一致：`T=300`、`t=150`、`N∈{20,30}` 的真高维欠定设定是成立的。
- `sparse_lasso_n20` 在 `M=2000` 时 size 仍接近 0.05，这与“正则化方法可在高维下保持控制”的叙事部分一致。
- `actual_fro` 与 `perturbation_type` 两项验证通过，说明扰动实现基本可信。

### 4.2 不符合的部分

- 文档最关键的叙事是：`OLS(F)` 在欠定场景下应“size 爆炸”；但本次结果显示两个 `baseline_ols_f` 的 size 都接近 0.05，而不是接近 1.0。
- 文档预期 `Lasso/SVD` 在 `M=2000` 时应落在 `[0.03, 0.07]`；本次只有 `sparse_lasso_n20` 达标，其余 3 个结构化模型均未达标。
- 文档预期 power 随 `delta` 单调增加；本次只有 `baseline_ols_f_n30` 满足严格单调。

## 5. 可能含义与优先怀疑点

### 5.1 `baseline_ols_f` 结果与文档叙事冲突

这是本次最重要的问题，优先级最高。可能含义包括：

1. 文档中的“OLS(F) 欠定后 size → 1”判断已经不再适用于当前实现
2. 当前 `baseline_ols_f` 路径实际上并未触发预期中的数值崩溃/退化行为
3. `simulation_plan.md` 的预期写得过强，需要根据当前实现修订

无论哪一种，都意味着：**在正式写论文或做最终汇报前，必须重新核查 `baseline_ols_f` 的实现逻辑与理论预期是否一致。**

### 5.2 `lowrank_svd` 的 size 明显偏高

这表明当前低秩 Bootstrap 路线在真高维设定下并没有达到文档中宣称的“正常控制”。如果 `B=500`、双 seed 后仍保持这一结论，那么 Section 11 的核心论证需要重写，而不是只做措辞微调。

### 5.3 `B=200` 可能放大波动，但不足以解释全部偏差

- `B=200` 相比正式配置 `B=500` 会引入更粗的 bootstrap p 值与更大的 Monte Carlo 波动。
- 但像 `lowrank_svd_n30 size@M=2000 = 0.2105` 这样的偏差已经远大于“轻微随机波动”可解释的量级。
- 因此，本次结果虽然是探索性的，但已经足以提示：**问题不太可能只靠把 `B` 调回 500 就完全消失。**

## 6. 对自动生成报告的一个说明

自动报告中的 `size_at_Mmax=...` 字样有歧义。代码实际优先取的是 `M = power_M = 500`，而不是严格意义上的 `M_grid` 最大值 `2000`。因此如果要和 `simulation_plan.md` 的验证清单对照，应优先看 `M=2000` 那一行，而不是摘要里的 `size_at_Mmax`。

## 7. 当前结论

如果问题是“**本次 `B=200` 单 seed 结果是否符合 `simulation_plan.md` 的预期？**”，答案是：

**整体上不符合，而且是关键性不符合。**

更具体地说：

- `sparse_lasso_n20`：基本符合预期
- `sparse_lasso_n30`：部分不符合（size 偏高）
- `lowrank_svd_n20/n30`：明显不符合（size 明显偏高）
- `baseline_ols_f_n20/n30`：与文档核心叙事直接冲突（未出现预期中的 size 爆炸）

## 8. 建议的下一步

建议按优先级执行：

1. **先核查 `baseline_ols_f` 实现**：确认它为何没有出现文档预期的 size 爆炸
2. **再跑正式参数的最小复核版**：优先单 seed、`B=500`，只针对 `baseline_ols_f_n30`、`sparse_lasso_n30`、`lowrank_svd_n30`
3. 如果 `B=500` 下仍复现当前 pattern，再决定是修代码还是修 `simulation_plan.md` 的叙事

