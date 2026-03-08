# 大规模试验分析报告

- 生成时间: 2026-03-07 23:51:46
- 方案配置: M=120, B=200, alpha=0.05, Frobenius targets=(0.04, 0.08, 0.12, 0.16)
- 断点构造: 沿归一化全1方向施加扰动，并固定目标 `||Phi2-Phi1||_F`；若不平稳则仅缩小扰动幅度。

- 总耗时(秒): 3385.92

## 1. Size（第一类错误）

| 模型 | Type I Error | Size Distortion | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 0.0583 | +0.0083 | 7 | 120 | 507.14 |
| sparse_lasso | 0.0500 | +0.0000 | 6 | 120 | 738.15 |
| lowrank_svd | 0.0917 | +0.0417 | 11 | 120 | 1192.13 |

## 2. Power 曲线数据

| 模型 | Target `||ΔΦ||_F` | Actual `||ΔΦ||_F` | Power | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_ols | 0.04 | 0.0400 | 0.0500 | 6 | 120 | 505.40 |
| baseline_ols | 0.08 | 0.0800 | 0.0500 | 6 | 120 | 496.31 |
| baseline_ols | 0.12 | 0.1200 | 0.0583 | 7 | 120 | 514.51 |
| baseline_ols | 0.16 | 0.1600 | 0.0667 | 8 | 120 | 501.27 |
| sparse_lasso | 0.04 | 0.0400 | 0.0417 | 5 | 120 | 742.55 |
| sparse_lasso | 0.08 | 0.0800 | 0.0583 | 7 | 120 | 736.84 |
| sparse_lasso | 0.12 | 0.1200 | 0.0833 | 10 | 120 | 590.27 |
| sparse_lasso | 0.16 | 0.1600 | 0.0500 | 6 | 120 | 489.87 |
| lowrank_svd | 0.04 | 0.0400 | 0.0917 | 11 | 120 | 1191.19 |
| lowrank_svd | 0.08 | 0.0800 | 0.1333 | 16 | 120 | 824.68 |
| lowrank_svd | 0.12 | 0.1200 | 0.0750 | 9 | 120 | 131.34 |
| lowrank_svd | 0.16 | 0.1600 | 0.0750 | 9 | 120 | 46.58 |

## 3. 结论摘要

- baseline_ols: type1_error=0.0583; power_at_max_target_fro(0.16)=0.0667; power_gain_over_size=0.0083; power_monotone=True
- sparse_lasso: type1_error=0.0500; power_at_max_target_fro(0.16)=0.0500; power_gain_over_size=0.0000; power_monotone=False
- lowrank_svd: type1_error=0.0917; power_at_max_target_fro(0.16)=0.0750; power_gain_over_size=-0.0167; power_monotone=False
