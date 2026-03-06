# 下一阶段大规模试验报告

## 1. 配置
- 时间：2026-03-06 01:47:20
- p：3（所有场景统一）
- seeds：[42, 2026]
- M-grid：[300, 500, 800]
- B-grid：[200, 500, 1000]
- Baseline B_mc：80
- 高维 B_mc：100
- 场景维度：
  - baseline: {'N': 2, 'T': 220, 't': 110}
  - sparse: {'N': 3, 'T': 220, 't': 110}
  - lowrank: {'N': 4, 'T': 220, 't': 110}

## 2. Baseline p值对比（跨seed均值±95%CI）
| B | chi2 p mean [95%CI] | bootstrap LR p mean [95%CI] | |boot LR-chi2| mean [95%CI] |
|---:|---:|---:|---:|
| 200 | 0.7758 [0.6948, 0.8567] | 0.8350 [0.7860, 0.8840] | 0.0592 [0.0273, 0.0912] |
| 500 | 0.7758 [0.6948, 0.8567] | 0.8350 [0.7782, 0.8918] | 0.0592 [0.0351, 0.0834] |
| 1000 | 0.7758 [0.6948, 0.8567] | 0.8180 [0.7474, 0.8886] | 0.0422 [0.0318, 0.0527] |

## 3. 第一类错误与功效（跨seed均值±95%CI）
| 模型 | M | Type I mean [95%CI] | Power mean [95%CI] |
|---|---:|---:|---:|
| baseline_chow_asym_chi2 | 300 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_asym_chi2 | 500 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_asym_chi2 | 800 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_asym_f | 300 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_asym_f | 500 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_asym_f | 800 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_bootstrap_f | 300 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_bootstrap_f | 500 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_bootstrap_f | 800 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| baseline_chow_bootstrap_lr | 300 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| baseline_chow_bootstrap_lr | 500 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| baseline_chow_bootstrap_lr | 800 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| lowrank_bootstrap_lr | 300 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| lowrank_bootstrap_lr | 500 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| lowrank_bootstrap_lr | 800 | 0.0000 [0.0000, 0.0000] | 0.5000 [-0.4800, 1.4800] |
| sparse_bootstrap_lr | 300 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| sparse_bootstrap_lr | 500 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| sparse_bootstrap_lr | 800 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |

## 4. 数据背景说明
- 指标定义：第一类错误 = H0真时拒绝率；功效 = H1真时拒绝率。
- 置信区间：跨seed统计，按 `mean ± 1.96*sd/sqrt(n_seed)`。
- H0 数据：无断点序列；H1 数据：在给定 t 处设定参数突变。
- 同一场景下 H0/H1 共享相同 `p,T,N,t` 口径。

## 5. 输出文件
- summary: `results/next_stage_summary_2026-03-06_014719_next_stage_p3_multiseed_v4.json`
- baseline raw: `results/next_stage_baseline_raw_2026-03-06_014719_next_stage_p3_multiseed_v4.csv`
- baseline agg: `results/next_stage_baseline_agg_2026-03-06_014719_next_stage_p3_multiseed_v4.csv`
- validation raw: `results/next_stage_validation_raw_2026-03-06_014719_next_stage_p3_multiseed_v4.csv`
- validation agg: `results/next_stage_validation_agg_2026-03-06_014719_next_stage_p3_multiseed_v4.csv`
- baseline 图: `results/next_stage_baseline_pvalues_vs_B_2026-03-06_014719_next_stage_p3_multiseed_v4.png`
- type1 图: `results/next_stage_model_type1_vs_M_2026-03-06_014719_next_stage_p3_multiseed_v4.png`
- power 图: `results/next_stage_model_power_vs_M_2026-03-06_014719_next_stage_p3_multiseed_v4.png`

## 6. 方法与口径补充说明
- 本阶段严格满足：`p=3`、`M-grid={300,500,800}`、高维 `B_mc=100`、多 seed（2个）。
- 时间序列长度较上一阶段提升：baseline 从 `T=120` 提升到 `T=220`；高维场景使用 `T=220`。
- 高维部分采用“**稀疏/低秩数据生成 + bootstrap LR 推断**”统一口径（模型名：`sparse_bootstrap_lr`、`lowrank_bootstrap_lr`），用于在大规模多 seed 设定下稳定完成计算。

## 7. 结果解读注意事项
- 置信区间按正态近似 `mean ± 1.96*sd/sqrt(n_seed)`，当前 `n_seed=2`，区间可能超出 `[0,1]`（如 `-0.48, 1.48`），这是小 seed 数下的统计现象。
- 建议下一轮将 seed 扩展到 `5~10`，并可使用分位数区间或对区间裁剪到 `[0,1]`，提高可解释性。
