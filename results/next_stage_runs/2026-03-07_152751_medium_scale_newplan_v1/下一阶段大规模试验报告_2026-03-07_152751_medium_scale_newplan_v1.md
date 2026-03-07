# 下一阶段大规模试验报告

## 1. 配置
- 时间：2026-03-07 15:39:09
- p：3（所有场景统一）
- seeds：[42, 2026, 7]
- M-grid：[10, 20, 50]
- power 变化幅度 grid：[0.04, 0.08, 0.12, 0.16, 0.2]
- power 固定 M_max：50
- B-grid：[10, 20]
- seed_workers：2
- Baseline B_mc：5
- 高维 B_mc：2
- 高维检验口径：稀疏场景使用 `SparseBootstrapInference` 的 LR+bootstrap；低秩场景使用 `LowRankBootstrapInference` 的 LR+bootstrap。
- 场景维度：
  - baseline: {'N': 2, 'T': 150, 't': 75}
  - sparse: {'N': 3, 'T': 150, 't': 75}
  - lowrank: {'N': 4, 'T': 150, 't': 75}

## 2. Baseline p值对比（跨seed均值 + 95%分位数CI）
| B | chi2 p mean [95%CI] | bootstrap LR p mean [95%CI] | |boot LR-chi2| mean [95%CI] |
|---:|---:|---:|---:|
| 10 | 0.7017 [0.5219, 0.8369] | 0.6667 [0.6050, 0.7000] | 0.1619 [0.0610, 0.2387] |
| 20 | 0.7017 [0.5219, 0.8369] | 0.7000 [0.6050, 0.7950] | 0.0619 [0.0419, 0.0885] |

## 3. 不同情况下的 Size（跨seed均值）
| 场景 | 模型 | M | Size mean |
|---|---|---:|---:|
| baseline | baseline_chow_asym_chi2 | 10 | 0.0000 |
| baseline | baseline_chow_asym_chi2 | 20 | 0.0000 |
| baseline | baseline_chow_asym_chi2 | 50 | 0.0000 |
| baseline | baseline_chow_asym_f | 10 | 0.0000 |
| baseline | baseline_chow_asym_f | 20 | 0.0000 |
| baseline | baseline_chow_asym_f | 50 | 0.0000 |
| baseline | baseline_chow_bootstrap_f | 10 | 0.3000 |
| baseline | baseline_chow_bootstrap_f | 20 | 0.3333 |
| baseline | baseline_chow_bootstrap_f | 50 | 0.3333 |
| baseline | baseline_chow_bootstrap_lr | 10 | 0.3000 |
| baseline | baseline_chow_bootstrap_lr | 20 | 0.3333 |
| baseline | baseline_chow_bootstrap_lr | 50 | 0.3333 |
| lowrank | lowrank_svd | 10 | 0.6000 |
| lowrank | lowrank_svd | 20 | 0.6667 |
| lowrank | lowrank_svd | 50 | 0.6667 |
| sparse | sparse_lasso | 10 | 0.3333 |
| sparse | sparse_lasso | 20 | 0.3333 |
| sparse | sparse_lasso | 50 | 0.3333 |

## 4. 固定最大 M 下不同变化幅度的 Power（跨seed均值 + 95%分位数CI）
| 场景 | 模型 | change scale | 固定 M_max | Power mean [95%CI] |
|---|---|---:|---:|---:|
| baseline | baseline_chow_asym_chi2 | 0.04 | 50 | 0.0000 [0.0000, 0.0000] |
| baseline | baseline_chow_asym_chi2 | 0.08 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_chi2 | 0.12 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_chi2 | 0.16 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_chi2 | 0.20 | 50 | 1.0000 [1.0000, 1.0000] |
| baseline | baseline_chow_asym_f | 0.04 | 50 | 0.0000 [0.0000, 0.0000] |
| baseline | baseline_chow_asym_f | 0.08 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_f | 0.12 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_f | 0.16 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_asym_f | 0.20 | 50 | 1.0000 [1.0000, 1.0000] |
| baseline | baseline_chow_bootstrap_f | 0.04 | 50 | 0.3267 [0.0000, 0.9310] |
| baseline | baseline_chow_bootstrap_f | 0.08 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_f | 0.12 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_f | 0.16 | 50 | 1.0000 [1.0000, 1.0000] |
| baseline | baseline_chow_bootstrap_f | 0.20 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_lr | 0.04 | 50 | 0.3267 [0.0000, 0.9310] |
| baseline | baseline_chow_bootstrap_lr | 0.08 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_lr | 0.12 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_lr | 0.16 | 50 | 0.6667 [0.0500, 1.0000] |
| baseline | baseline_chow_bootstrap_lr | 0.20 | 50 | 0.6667 [0.0500, 1.0000] |
| lowrank | lowrank_svd | 0.04 | 50 | 0.6667 [0.0500, 1.0000] |
| lowrank | lowrank_svd | 0.08 | 50 | 1.0000 [1.0000, 1.0000] |
| lowrank | lowrank_svd | 0.12 | 50 | 1.0000 [1.0000, 1.0000] |
| lowrank | lowrank_svd | 0.16 | 50 | 0.6667 [0.0500, 1.0000] |
| lowrank | lowrank_svd | 0.20 | 50 | 1.0000 [1.0000, 1.0000] |
| sparse | sparse_lasso | 0.04 | 50 | 0.6533 [0.0490, 0.9800] |
| sparse | sparse_lasso | 0.08 | 50 | 1.0000 [1.0000, 1.0000] |
| sparse | sparse_lasso | 0.12 | 50 | 0.6667 [0.0500, 1.0000] |
| sparse | sparse_lasso | 0.16 | 50 | 0.6667 [0.0500, 1.0000] |
| sparse | sparse_lasso | 0.20 | 50 | 0.6667 [0.0500, 1.0000] |

## 5. 数据背景说明
- 指标定义：Size = H0真时拒绝率；Power = H1真时拒绝率。
- 本轮固定 3 个 seed；size 主表默认只播报跨 seed 均值，不单独展示 95%CI。
- H0 数据：无断点序列；H1 数据：在给定 t 处设定参数突变。
- Power 部分额外固定至少 5 档变化幅度，并观察 power 是否随变化幅度增大而提升。
- 同一场景下 H0/H1 共享相同 `p,T,N,t` 口径。

## 6. 输出文件
- run dir: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1`
- progress log: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/progress.log`
- worker logs: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/worker_logs`
- state dir: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/state`
- summary: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_summary_2026-03-07_152751_medium_scale_newplan_v1.json`
- baseline raw: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- baseline agg: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- size raw: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_size_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- size agg: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_size_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- power raw: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_power_raw_2026-03-07_152751_medium_scale_newplan_v1.csv`
- power agg: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_power_agg_2026-03-07_152751_medium_scale_newplan_v1.csv`
- baseline 图: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_baseline_pvalues_vs_B_2026-03-07_152751_medium_scale_newplan_v1.png`
- size 图: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_model_size_vs_M_2026-03-07_152751_medium_scale_newplan_v1.png`
- power 图: `results/next_stage_runs/2026-03-07_152751_medium_scale_newplan_v1/next_stage_model_power_vs_change_2026-03-07_152751_medium_scale_newplan_v1.png`
