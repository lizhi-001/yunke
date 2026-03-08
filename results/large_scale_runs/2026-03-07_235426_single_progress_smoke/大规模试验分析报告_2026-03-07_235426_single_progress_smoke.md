# 大规模试验分析报告

- 生成时间: 2026-03-07 23:54:26
- 方案配置: M=1, B=1, alpha=0.05, Frobenius targets=(0.1,)
- 断点构造: 沿归一化全1方向施加扰动，并固定目标 `||Phi2-Phi1||_F`；若不平稳则仅缩小扰动幅度。

- 总耗时(秒): 0.07
- 进度日志: results/large_scale_runs/2026-03-07_235426_single_progress_smoke/progress.log
- 输出目录: results/large_scale_runs/2026-03-07_235426_single_progress_smoke

## 1. Size（第一类错误）

| 模型 | Type I Error | Size Distortion | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|
| baseline_ols | 1.0000 | +0.9500 | 1 | 1 | 0.00 |
| sparse_lasso | 0.0000 | -0.0500 | 0 | 1 | 0.02 |
| lowrank_svd | 0.0000 | -0.0500 | 0 | 1 | 0.01 |

## 2. Power 曲线数据

| 模型 | Target `||ΔΦ||_F` | Actual `||ΔΦ||_F` | Power | Rejections | M_effective | Runtime(s) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_ols | 0.10 | 0.1000 | 0.0000 | 0 | 1 | 0.00 |
| sparse_lasso | 0.10 | 0.1000 | 1.0000 | 1 | 1 | 0.02 |
| lowrank_svd | 0.10 | 0.1000 | 0.0000 | 0 | 1 | 0.01 |

## 3. 结论摘要

- baseline_ols: type1_error=1.0000; power_at_max_target_fro(0.10)=0.0000; power_gain_over_size=-1.0000; power_monotone=False
- sparse_lasso: type1_error=0.0000; power_at_max_target_fro(0.10)=1.0000; power_gain_over_size=1.0000; power_monotone=False
- lowrank_svd: type1_error=0.0000; power_at_max_target_fro(0.10)=0.0000; power_gain_over_size=0.0000; power_monotone=False
