# 结构化 VAR 断裂检验仿真报告

- 生成时间：2026-03-16 16:05:08
- seeds：[42, 2026]
- B=100  alpha=0.05  M_grid=[50, 100, 300, 500, 1000]  power_M=300
- deltas：[0.1, 0.3, 0.5, 0.75, 1.0]
- 总耗时：213.58s

## 实验设计

核心叙事：在稀疏和低秩背景下使用 LR+Bootstrap 进行断裂检验，不要求性能高于 OLS。

| 层次 | 模型 | DGP | N | 估计 | 推断 | 扰动类型 | 角色 |
|---|---|---|---:|---|---|---|---|
| 第一层 | baseline_ols_f | 稠密 | 10 | OLS   | 渐近 F       | uniform | 普通多元时间序列基准 |
| 第一层 | baseline_ols   | 稠密 | 10 | OLS   | LR+Bootstrap | uniform | 提出方法，与 F 检验一致 |
| 第二层 | sparse_lasso   | 稀疏(0.15) | 20 | Lasso | LR+Bootstrap | sparse  | 高维稀疏断裂检验 |
| 第三层 | lowrank_rrr    | 低秩(r=2)  | 20 | RRR   | LR+Bootstrap | lowrank | 高维低秩断裂检验 |

预期：第一层两方法 size/power 一致；第二/三层 size→0.05（随 M 增大），power 随 δ 单调递增。

## 1. Size（第一类错误，M=2000）

| 层次 | 模型 | size mean | size std | size distortion | 95% CI | seeds |
|---|---|---:|---:|---:|---|---:|
| 第三层 | lowrank_rrr_fv | 0.0265 | 0.0025 | -0.0235 | [0.0195, 0.0335] | 2 |

## 2. Power（随 δ 的功效曲线）

| 层次 | 模型 | 扰动类型 | δ | power mean | power std | seeds |
|---|---|---|---:|---:|---:|---:|
| 第三层 | lowrank_rrr_fv | lowrank_fixedV | 0.10 | 0.0367 | 0.0033 | 2 |
| 第三层 | lowrank_rrr_fv | lowrank_fixedV | 0.30 | 0.0867 | 0.0200 | 2 |
| 第三层 | lowrank_rrr_fv | lowrank_fixedV | 0.50 | 0.2783 | 0.0383 | 2 |
| 第三层 | lowrank_rrr_fv | lowrank_fixedV | 0.75 | 0.7550 | 0.0050 | 2 |
| 第三层 | lowrank_rrr_fv | lowrank_fixedV | 1.00 | 0.9950 | 0.0050 | 2 |

## 3. 结论摘要

- **lowrank_rrr_fv** [高维低秩 RRR+LR+Bootstrap（固定空间，随机载荷扰动）]: size@M=1000=0.0265; power@δ=1.00=0.9950; power单调=✓
