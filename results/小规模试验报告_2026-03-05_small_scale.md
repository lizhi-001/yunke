# 小规模试验报告（small_scale）

## 1. 试验信息
- 时间：2026-03-05 18:39（本地运行）
- 运行脚本：`experiments/run_revised_validation_framework.py`
- 运行命令：
  - `python3 experiments/run_revised_validation_framework.py --B-grid 20 50 100 --M-grid 10 20 --B-mc 20 --tag small_scale`
- 关键参数：
  - 显著性水平 `alpha = 0.05`
  - 随机种子 `seed = 42`
  - Bootstrap 次数网格 `B ∈ {20, 50, 100}`
  - 蒙特卡洛次数网格 `M ∈ {10, 20}`
  - Monte Carlo 内部 bootstrap 次数 `B_mc = 20`

## 2. 输出文件
- 汇总 JSON：`results/revised_framework_summary_2026-03-05_183911_small_scale.json`
- baseline 对比 CSV：`results/baseline_bootstrap_vs_chi2_2026-03-05_183911_small_scale.csv`
- baseline 对比图：`results/baseline_bootstrap_vs_chi2_2026-03-05_183911_small_scale.png`
- baseline p值-B关系图：`results/baseline_pvalues_vs_B_2026-03-05_183911_small_scale.png`
- 模型验证 CSV：`results/model_validation_vs_M_2026-03-05_183911_small_scale.csv`
- 第一类错误-M关系图：`results/model_type1_vs_M_2026-03-05_183911_small_scale.png`
- 功效-M关系图：`results/model_power_vs_M_2026-03-05_183911_small_scale.png`

## 3. Baseline：渐近 p 值 vs Bootstrap p 值（随 B 变化）

**数据背景（本节口径）**：
- 本节表格基于 **H0（无断点）** 的同一条 baseline VAR 序列计算：`N=2, T=120, p=1, t=60`。
- 误差项设定为高斯白噪声：`u_t ~ N(0, 0.5I)`。
- 对该固定样本先得到 `LR_obs`，再在不同 `B` 下计算 `χ² p值(LR)` 与 `bootstrap LR p值` 并比较差异。

### 3.1 H0（无断点）对比结果（随 B 变化）
| B | χ² p值 (LR) | F渐近p值 | bootstrap LR p值 | bootstrap F p值 | |bootstrap LR - χ²| |
|---:|---:|---:|---:|---:|---:|
| 20  | 0.37299 | 0.35923 | 0.40000 | 0.40000 | 0.02701 |
| 50  | 0.37299 | 0.35923 | 0.34000 | 0.36000 | 0.03299 |
| 100 | 0.37299 | 0.35923 | 0.38000 | 0.36000 | 0.00701 |

### 3.2 H1（有断点）补充对比结果（随 B 变化）
**数据背景（本节口径）**：
- 本节基于单条 **H1（有断点）** baseline VAR 序列，断点位置为 `t=60`。
- 其余设定与 H0 对比一致：`N=2, T=120, p=1`，误差项 `u_t ~ N(0, 0.5I)`。
- 结果来源文件：`results/baseline_bootstrap_vs_chi2_2026-03-05_191656_H1_small.csv`。

| B | χ² p值 (LR) | F渐近p值 | bootstrap LR p值 | bootstrap F p值 | |bootstrap LR - χ²| |
|---:|---:|---:|---:|---:|---:|
| 20  | 0.11328 | 0.09080 | 0.15000 | 0.10000 | 0.03672 |
| 50  | 0.11328 | 0.09080 | 0.14000 | 0.12000 | 0.02672 |
| 100 | 0.11328 | 0.09080 | 0.13000 | 0.09000 | 0.01672 |

**解读（小规模下）**：
- H0 与 H1 下都能看到：`B` 增加时，`bootstrap LR p值` 与 `χ² p值` 的差距趋于缩小。
- 在小 `B`（20/50）下，bootstrap p值波动相对更明显；`B=100` 时差距明显收敛。
- H1 这组为单条样本结果，数值会受样本随机性影响，应结合更大 `M` 的 Monte Carlo 结果解读。

## 4. Monte Carlo 稳定性：第一类错误与统计功效（随 M 变化）
| 模型 | M | 第一类错误 | 功效 |
|---|---:|---:|---:|
| baseline_chow_asym_f | 10 | 0.00 | 0.90 |
| baseline_chow_asym_chi2 | 10 | 0.00 | 0.90 |
| baseline_chow_bootstrap_f | 10 | 0.10 | 0.90 |
| baseline_chow_bootstrap_lr | 10 | 0.10 | 0.80 |
| baseline_chow_asym_f | 20 | 0.05 | 0.80 |
| baseline_chow_asym_chi2 | 20 | 0.05 | 0.80 |
| baseline_chow_bootstrap_f | 20 | 0.10 | 0.85 |
| baseline_chow_bootstrap_lr | 20 | 0.10 | 0.90 |
| sparse_lasso | 10 | 0.10 | 0.90 |
| sparse_lasso | 20 | 0.10 | 0.95 |
| lowrank_svd | 10 | 0.00 | 1.00 |
| lowrank_svd | 20 | 0.00 | 1.00 |

**解读（小规模下）**：
- `M=10` 与 `M=20` 的结果波动较大，说明 Monte Carlo 误差仍主导。
- baseline 渐近方法在 `M=20` 时第一类错误达到目标值附近（0.05）。
- 高维场景（sparse/lowrank）结果目前偏“理想化”，需更大 `M` 与更多重复验证稳定性。

## 5. 初步结论
1. 当前“同口径样本 + Chow(F)/LR并行 + bootstrap对比”的流程可运行、可产出完整图表。
2. 在小规模参数下，bootstrap p值与渐近 p值方向一致，但数值仍有可见噪声。
3. 若用于方法学结论，建议将规模提升到：
   - `B-grid` 至少包含 `200, 500, 1000`
   - `M-grid` 至少包含 `100, 300, 500`
   - 重点比较 baseline 四条线（asym-F / asym-χ² / boot-F / boot-LR）的收敛关系。
