# 下一阶段大规模试验详细报告

## 1. 试验目标

- 在统一的 `p=3` 框架下，分开汇报 baseline 与高维方法在已知断点检验中的 size 和 power。
- size 部分单独展示不同场景、不同模型在不同 M 下的拒绝率。
- power 部分单独展示不同场景、不同模型在至少 5 档时间序列变化幅度下、固定最大 `M_max` 的拒绝率，并检查 power 是否随变化幅度提升。

## 2. 数据背景

- 显著性水平：`alpha=0.05`
- seeds：`[42, 2026, 7]`
- baseline p 值比较的 B-grid：`[10, 20]`
- size 的 M-grid：`[10, 20, 50]`
- power 的变化幅度 grid：`[0.04, 0.08, 0.12, 0.16, 0.2]`
- power 固定 M_max：`50`
- baseline 内部 bootstrap 次数：`5`
- 高维内部 bootstrap 次数：`2`
- baseline 场景：`N=2, T=150, p=3, t=75`
- sparse 场景：`N=3, T=150, p=3, t=75`
- lowrank 场景：`N=4, T=150, p=3, t=75`
- H0 数据为无断点 VAR 序列；H1 数据为在给定断点位置 `t` 处发生结构变化的 VAR 序列。
- 本轮固定 3 个 seed；size 主表默认只播报跨 seed 均值，不单独展示 95%CI。

## 3. baseline p 值比较

| B | asym-chi2(LR) | bootstrap-LR | bootstrap-F | |boot-LR - chi2| |
|---:|---:|---:|---:|---:|
| 10 | 0.7017 [0.5219, 0.8369] | 0.6667 [0.6050, 0.7000] | 0.6667 [0.6050, 0.7000] | 0.1619 [0.0610, 0.2387] |
| 20 | 0.7017 [0.5219, 0.8369] | 0.7000 [0.6050, 0.7950] | 0.7000 [0.6050, 0.7950] | 0.0619 [0.0419, 0.0885] |

## 4. Size 结果

| 场景 | 模型 | M | Size mean | 有效M |
|---|---|---:|---:|---:|
| baseline | baseline_chow_asym_chi2 | 10 | 0.0000 | 10.0 |
| baseline | baseline_chow_asym_chi2 | 20 | 0.0000 | 20.0 |
| baseline | baseline_chow_asym_chi2 | 50 | 0.0000 | 50.0 |
| baseline | baseline_chow_asym_f | 10 | 0.0000 | 10.0 |
| baseline | baseline_chow_asym_f | 20 | 0.0000 | 20.0 |
| baseline | baseline_chow_asym_f | 50 | 0.0000 | 50.0 |
| baseline | baseline_chow_bootstrap_f | 10 | 0.3000 | 10.0 |
| baseline | baseline_chow_bootstrap_f | 20 | 0.3333 | 20.0 |
| baseline | baseline_chow_bootstrap_f | 50 | 0.3333 | 50.0 |
| baseline | baseline_chow_bootstrap_lr | 10 | 0.3000 | 10.0 |
| baseline | baseline_chow_bootstrap_lr | 20 | 0.3333 | 20.0 |
| baseline | baseline_chow_bootstrap_lr | 50 | 0.3333 | 50.0 |
| lowrank | lowrank_svd | 10 | 0.6000 | 10.0 |
| lowrank | lowrank_svd | 20 | 0.6667 | 20.0 |
| lowrank | lowrank_svd | 50 | 0.6667 | 50.0 |
| sparse | sparse_lasso | 10 | 0.3333 | 10.0 |
| sparse | sparse_lasso | 20 | 0.3333 | 20.0 |
| sparse | sparse_lasso | 50 | 0.3333 | 50.0 |

## 5. 固定最大 M 下的 Power 结果

| 场景 | 模型 | 变化幅度 | 固定 M_max | Power | 平均Δ Fro |
|---|---|---:|---:|---:|---:|
| baseline | baseline_chow_asym_chi2 | 0.04 | 50 | 0.0000 [0.0000, 0.0000] | 0.1386 |
| baseline | baseline_chow_asym_chi2 | 0.08 | 50 | 0.6667 [0.0500, 1.0000] | 0.2771 |
| baseline | baseline_chow_asym_chi2 | 0.12 | 50 | 0.6667 [0.0500, 1.0000] | 0.3961 |
| baseline | baseline_chow_asym_chi2 | 0.16 | 50 | 0.6667 [0.0500, 1.0000] | 0.4767 |
| baseline | baseline_chow_asym_chi2 | 0.20 | 50 | 1.0000 [1.0000, 1.0000] | 0.5323 |
| baseline | baseline_chow_asym_f | 0.04 | 50 | 0.0000 [0.0000, 0.0000] | 0.1386 |
| baseline | baseline_chow_asym_f | 0.08 | 50 | 0.6667 [0.0500, 1.0000] | 0.2771 |
| baseline | baseline_chow_asym_f | 0.12 | 50 | 0.6667 [0.0500, 1.0000] | 0.3961 |
| baseline | baseline_chow_asym_f | 0.16 | 50 | 0.6667 [0.0500, 1.0000] | 0.4767 |
| baseline | baseline_chow_asym_f | 0.20 | 50 | 1.0000 [1.0000, 1.0000] | 0.5323 |
| baseline | baseline_chow_bootstrap_f | 0.04 | 50 | 0.3267 [0.0000, 0.9310] | 0.1386 |
| baseline | baseline_chow_bootstrap_f | 0.08 | 50 | 0.6667 [0.0500, 1.0000] | 0.2771 |
| baseline | baseline_chow_bootstrap_f | 0.12 | 50 | 0.6667 [0.0500, 1.0000] | 0.3961 |
| baseline | baseline_chow_bootstrap_f | 0.16 | 50 | 1.0000 [1.0000, 1.0000] | 0.4767 |
| baseline | baseline_chow_bootstrap_f | 0.20 | 50 | 0.6667 [0.0500, 1.0000] | 0.5323 |
| baseline | baseline_chow_bootstrap_lr | 0.04 | 50 | 0.3267 [0.0000, 0.9310] | 0.1386 |
| baseline | baseline_chow_bootstrap_lr | 0.08 | 50 | 0.6667 [0.0500, 1.0000] | 0.2771 |
| baseline | baseline_chow_bootstrap_lr | 0.12 | 50 | 0.6667 [0.0500, 1.0000] | 0.3961 |
| baseline | baseline_chow_bootstrap_lr | 0.16 | 50 | 0.6667 [0.0500, 1.0000] | 0.4767 |
| baseline | baseline_chow_bootstrap_lr | 0.20 | 50 | 0.6667 [0.0500, 1.0000] | 0.5323 |
| lowrank | lowrank_svd | 0.04 | 50 | 0.6667 [0.0500, 1.0000] | 0.2771 |
| lowrank | lowrank_svd | 0.08 | 50 | 1.0000 [1.0000, 1.0000] | 0.5543 |
| lowrank | lowrank_svd | 0.12 | 50 | 1.0000 [1.0000, 1.0000] | 0.5465 |
| lowrank | lowrank_svd | 0.16 | 50 | 0.6667 [0.0500, 1.0000] | 0.5345 |
| lowrank | lowrank_svd | 0.20 | 50 | 1.0000 [1.0000, 1.0000] | 0.5433 |
| sparse | sparse_lasso | 0.04 | 50 | 0.6533 [0.0490, 0.9800] | 0.2078 |
| sparse | sparse_lasso | 0.08 | 50 | 1.0000 [1.0000, 1.0000] | 0.3983 |
| sparse | sparse_lasso | 0.12 | 50 | 0.6667 [0.0500, 1.0000] | 0.5402 |
| sparse | sparse_lasso | 0.16 | 50 | 0.6667 [0.0500, 1.0000] | 0.5388 |
| sparse | sparse_lasso | 0.20 | 50 | 0.6667 [0.0500, 1.0000] | 0.5670 |

## 6. 结果分析

- baseline p 值比较固定在同一条无断点序列上进行，用于观察不同 B 下渐近 LR 与 bootstrap LR 的偏离程度。
- 从 B=10 到 B=20，|boot-LR - chi2| 的跨 seed 均值由 0.1619 变化到 0.0619。
- Size 的目标参考值为显著性水平 `alpha=0.05`；越接近该值，说明 size 控制越稳定。
- `baseline/baseline_chow_asym_chi2` 在 M 从 10 增至 50 时，Size 均值从 0.0000 变化到 0.0000。
- `baseline/baseline_chow_asym_f` 在 M 从 10 增至 50 时，Size 均值从 0.0000 变化到 0.0000。
- `baseline/baseline_chow_bootstrap_f` 在 M 从 10 增至 50 时，Size 均值从 0.3000 变化到 0.3333。
- `baseline/baseline_chow_bootstrap_lr` 在 M 从 10 增至 50 时，Size 均值从 0.3000 变化到 0.3333。
- `lowrank/lowrank_svd` 在 M 从 10 增至 50 时，Size 均值从 0.6000 变化到 0.6667。
- `sparse/sparse_lasso` 在 M 从 10 增至 50 时，Size 均值从 0.3333 变化到 0.3333。
- Power 部分按至少 5 档变化幅度分别汇报，并固定在最大 `M_max` 下；重点检查变化幅度变大后，拒绝率是否整体抬升。
- `baseline/baseline_chow_asym_chi2` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.0000 变化到 1.0000，整体提升。
- `baseline/baseline_chow_asym_f` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.0000 变化到 1.0000，整体提升。
- `baseline/baseline_chow_bootstrap_f` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.3267 变化到 0.6667，整体提升。
- `baseline/baseline_chow_bootstrap_lr` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.3267 变化到 0.6667，整体提升。
- `lowrank/lowrank_svd` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.6667 变化到 1.0000，整体提升。
- `sparse/sparse_lasso` 在固定 `M_max=50` 下，变化幅度从 0.04 增至 0.20 时，Power 均值从 0.6533 变化到 0.6667，整体提升。
- 当前区间是跨 seed 的 95% 分位数区间，用于展示不同 seed 下的波动范围，而不是单次检验的决策边界。

## 7. 输出文件

- summary JSON：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_summary_2026-03-07_152751_medium_scale_newplan_v1.json`
- baseline 原始表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- baseline 聚合表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- size 原始表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_size_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- size 聚合表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_size_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- power 原始表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_power_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- power 聚合表：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_power_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- baseline 图：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_pvalues_vs_B_2026-03-07_152751_medium_scale_newplan_v1.png`
- size 图：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_model_size_vs_M_2026-03-07_152751_medium_scale_newplan_v1.png`
- power 图：`results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_model_power_vs_change_2026-03-07_152751_medium_scale_newplan_v1.png`
