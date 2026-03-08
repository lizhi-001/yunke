# 大规模试验分析报告（多 seed + M_grid）

- 生成时间: 2026-03-08 00:42:28
- seeds: (42, 2026)
- M_grid: (4, 6)
- power 使用的 : 6
- B: 4; alpha: 0.05; base deltas: (0.04, 0.08); baseline_pvalue_method: asymptotic_chi2
- 断点构造: 对所有系数施加统一基准单元素变化尺度 `delta`，再按 `target_fro = delta * sqrt(#coefficients)` 换算为模型特定的 Frobenius 目标强度。
- 总耗时(秒): 7.21

## 1. Size（不同 M 下的第一类错误，跨 seed 聚合）

| 模型 | M | Type I Error Mean | Type I Error Std | Size Distortion Mean | Seeds |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 4 | 0.1250 | 0.1250 | +0.0750 | 2 |
| baseline_ols | 6 | 0.3333 | 0.1667 | +0.2833 | 2 |
| sparse_lasso | 4 | 0.1250 | 0.1250 | +0.0750 | 2 |
| sparse_lasso | 6 | 0.1667 | 0.0000 | +0.1167 | 2 |
| lowrank_svd | 4 | 0.3750 | 0.1250 | +0.3250 | 2 |
| lowrank_svd | 6 | 0.1667 | 0.0000 | +0.1167 | 2 |

## 2. Power（固定 M_max，跨 seed 聚合）

| 模型 | M | Base δ | Target `||ΔΦ||_F` Mean | Power Mean | Power Std | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| baseline_ols | 6 | 0.04 | 0.08 | 0.0833 | 0.0833 | 2 |
| baseline_ols | 6 | 0.08 | 0.16 | 0.1667 | 0.1667 | 2 |
| sparse_lasso | 6 | 0.04 | 0.20 | 0.2500 | 0.2500 | 2 |
| sparse_lasso | 6 | 0.08 | 0.40 | 0.5000 | 0.3333 | 2 |
| lowrank_svd | 6 | 0.04 | 0.40 | 0.4167 | 0.0833 | 2 |
| lowrank_svd | 6 | 0.08 | 0.80 | 1.0000 | 0.0000 | 2 |

## 3. 结论摘要

- baseline_ols: size_at_Mmax=0.3333; power_at_Mmax_and_max_delta(base_delta=0.08; target_fro_mean=0.16)=0.1667; power_gain_over_size=-0.1667; power_monotone=True
- sparse_lasso: size_at_Mmax=0.1667; power_at_Mmax_and_max_delta(base_delta=0.08; target_fro_mean=0.40)=0.5000; power_gain_over_size=0.3333; power_monotone=True
- lowrank_svd: size_at_Mmax=0.1667; power_at_Mmax_and_max_delta(base_delta=0.08; target_fro_mean=0.80)=1.0000; power_gain_over_size=0.8333; power_monotone=True
