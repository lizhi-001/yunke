# 大规模试验分析报告

- 生成时间: 2026-03-07 17:31:03
- 方案配置: M=20, B=30, alpha=0.05, deltas=(0.04, 0.08, 0.16)
- 总耗时(秒): 25.05

## 1. Size（第一类错误）

| 模型 | Type I Error | Size Distortion | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 0.1500 | +0.1000 | 3 | 20 | 0.64 |
| sparse_lasso | 0.0000 | -0.0500 | 0 | 20 | 4.43 |
| lowrank_svd | 0.1500 | +0.1000 | 3 | 20 | 1.23 |

## 2. Power 曲线数据

| 模型 | Δ | Power | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 0.04 | 0.1000 | 2 | 20 | 0.65 |
| baseline_ols | 0.08 | 0.1500 | 3 | 20 | 0.59 |
| baseline_ols | 0.16 | 0.3000 | 6 | 20 | 0.59 |
| sparse_lasso | 0.04 | 0.0000 | 0 | 20 | 4.41 |
| sparse_lasso | 0.08 | 0.1000 | 2 | 20 | 4.41 |
| sparse_lasso | 0.16 | 0.9500 | 19 | 20 | 4.43 |
| lowrank_svd | 0.04 | 0.2000 | 4 | 20 | 1.22 |
| lowrank_svd | 0.08 | 0.7000 | 14 | 20 | 1.21 |
| lowrank_svd | 0.16 | 1.0000 | 20 | 20 | 1.25 |

## 3. 结论摘要

- baseline_ols: type1_error=0.1500; max_power=0.3000; power_monotone=True
- sparse_lasso: type1_error=0.0000; max_power=0.9500; power_monotone=True
- lowrank_svd: type1_error=0.1500; max_power=1.0000; power_monotone=True
