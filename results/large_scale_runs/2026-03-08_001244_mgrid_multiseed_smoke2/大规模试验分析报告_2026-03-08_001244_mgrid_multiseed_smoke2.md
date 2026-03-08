# 大规模试验分析报告（多 seed + M_grid）

- 生成时间: 2026-03-08 00:12:52
- seeds: (42, 2026)
- M_grid: (4, 6)
- power 使用的 M: 6
- B: 4; alpha: 0.05; deltas: (0.04, 0.08)
- 总耗时(秒): 7.49

## 1. Size（不同 M 下的第一类错误，跨 seed 聚合）

| 模型 | M | Type I Error Mean | Type I Error Std | Size Distortion Mean | Seeds |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 4 | 0.3750 | 0.1250 | +0.3250 | 2 |
| baseline_ols | 6 | 0.2500 | 0.2500 | +0.2000 | 2 |
| sparse_lasso | 4 | 0.1250 | 0.1250 | +0.0750 | 2 |
| sparse_lasso | 6 | 0.2500 | 0.0833 | +0.2000 | 2 |
| lowrank_svd | 4 | 0.1250 | 0.1250 | +0.0750 | 2 |
| lowrank_svd | 6 | 0.2500 | 0.0833 | +0.2000 | 2 |

## 2. Power（固定 M_max，跨 seed 聚合）

| 模型 | M | Target `||ΔΦ||_F` | Power Mean | Power Std | Seeds |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 6 | 0.04 | 0.1667 | 0.0000 | 2 |
| baseline_ols | 6 | 0.08 | 0.0833 | 0.0833 | 2 |
| sparse_lasso | 6 | 0.04 | 0.2500 | 0.0833 | 2 |
| sparse_lasso | 6 | 0.08 | 0.2500 | 0.0833 | 2 |
| lowrank_svd | 6 | 0.04 | 0.2500 | 0.0833 | 2 |
| lowrank_svd | 6 | 0.08 | 0.2500 | 0.0833 | 2 |

## 3. 结论摘要

- baseline_ols: size_at_Mmax=0.2500; power_at_Mmax_and_max_delta(0.08)=0.0833; power_gain_over_size=-0.1667; power_monotone=False
- sparse_lasso: size_at_Mmax=0.2500; power_at_Mmax_and_max_delta(0.08)=0.2500; power_gain_over_size=0.0000; power_monotone=True
- lowrank_svd: size_at_Mmax=0.2500; power_at_Mmax_and_max_delta(0.08)=0.2500; power_gain_over_size=0.0000; power_monotone=True
