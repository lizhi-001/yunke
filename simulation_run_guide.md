# 仿真实验运行文档

## 1. 主实验脚本

`experiments/run_large_scale_mgrid_multiseed.py` — 大规模结构断裂实验脚本。

### 实验逻辑

- 第一类错误（size）：在 `M_grid` 下逐个评估；
- 功效（power）：固定使用 `power_M`（默认 `max(M_grid)`，可通过 `--power-M` 独立指定）评估；
- 支持单 seed 或多 seed；
- 支持 seed 并行、模型并行、Monte Carlo 外层并行；
- `baseline_ols` 默认使用 `bootstrap_lr` p 值；`baseline_ols_f` 始终使用 `asymptotic_f` 作为对照；
- 四个模型统一 `T = 500`、`t = 250`，保证每段有效样本量一致，同时避免高维模型（lowrank_svd, N=10）在短样本下 SVD 截断导致的 size distortion。

### 四类模型

| 模型 | N | T | t | p | p 值方法 | 说明 |
|---|---|---|---|---|---|---|
| baseline_ols | 2 | 500 | 250 | 1 | bootstrap_lr | 低维 OLS，主检验 |
| baseline_ols_f | 2 | 500 | 250 | 1 | asymptotic_f | 渐近 F 对照组 |
| sparse_lasso | 5 | 500 | 250 | 1 | bootstrap_lr | 稀疏 Lasso (sparsity=0.2) |
| lowrank_svd | 10 | 500 | 250 | 1 | bootstrap_lr | 低秩 SVD (rank=2) |

### 启动命令示例

```bash
python3 -u experiments/run_large_scale_mgrid_multiseed.py \
  --M-grid 50 100 300 500 1000 2000 \
  --power-M 300 \
  --B 500 \
  --alpha 0.05 \
  --deltas 0.05 0.1 0.15 0.2 0.3 0.5 \
  --seeds 42 \
  --jobs 4 \
  --seed-workers 1 \
  --tag v5_t500_fro
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--M-grid` | [50, 100, 300, 500, 1000, 2000] | 第一类错误评估的 Monte Carlo 重复次数网格 |
| `--power-M` | 0 (= max(M_grid)) | 功效评估的 MC 重复次数，独立于 M_grid |
| `--B` | 200 | 每次检验的 bootstrap 重复次数 |
| `--alpha` | 0.05 | 显著性水平 |
| `--deltas` | [0.05, 0.1, 0.15, 0.2, 0.3, 0.5] | Frobenius 范数目标网格（`||ΔΦ||_F = delta`） |
| `--seeds` | [42, 2026, 7] | 随机种子列表 |
| `--jobs` | 4 | 总并行预算 |
| `--seed-workers` | 0 (自动) | 并发 seed 数 |
| `--baseline-pvalue-method` | bootstrap_lr | baseline_ols 的 p 值方法 |
| `--skip-type1` | false | 跳过 Type I error 评估，只跑 power |
| `--tag` | (空) | 运行标签 |

---

## 2. 输出目录结构

每次运行在 `results/large_scale_runs/<timestamp>_<tag>/` 下输出：

```text
results/large_scale_runs/
├── <run_name>/
│   ├── large_scale_experiment_*.json
│   ├── large_scale_raw_*.csv
│   ├── large_scale_agg_*.csv
│   ├── 大规模试验分析报告_*.md
│   ├── run_meta.json
│   ├── seed_results/
│   │   └── seed_<seed>.json
│   └── progress/
│       ├── progress.log
│       ├── progress.jsonl
│       ├── summary.json
│       └── seed_<seed>_summary.json
```

### 进度日志

- `progress/progress.log`：人类可读进度日志，记录 started/completed/failed 事件。
- `progress/progress.jsonl`：结构化事件流，适合程序化监控。
- `progress/summary.json`：总进度摘要（completed_stage_count / total_stage_count / progress_ratio）。
- `progress/seed_<seed>_summary.json`：单个 seed 的独立进度摘要。

若实验异常退出或收到终止信号，日志中会显式记录 `failed` 事件。

---

## 3. 模型调度与并行策略

### 两阶段调度

1. **Phase 1**：baseline_ols 和 baseline_ols_f 顺序执行，每个模型独占全部 `--jobs` 个 worker；
2. **Phase 2**：sparse_lasso 和 lowrank_svd 顺序执行，每个模型独占全部 `--jobs` 个 worker。

所有模型均顺序执行、独占全部 worker 的原因：loky 在同一进程内维护全局共享 worker pool，多线程并行时无法为每个模型创建独立 pool，实际并行度 = max(各请求 n_jobs) 而非求和。顺序独占让每个模型使用全部 worker，CPU 利用率从 ~25% 提升到 ~100%。

| jobs | 执行方式 |
|---|---|
| 4 | baseline_ols(4w) → baseline_ols_f(4w) → sparse(4w) → lowrank(4w) |
| 8 | baseline_ols(8w) → baseline_ols_f(8w) → sparse(8w) → lowrank(8w) |

### 并行后端

Monte Carlo 外层并行使用 **loky**（`joblib.externals.loky`），回退链：loky → ProcessPoolExecutor → ThreadPoolExecutor。

---

## 4. 长实验运行建议

对于 1 小时以上的实验，推荐使用 `tmux` 避免进程被回收：

```bash
tmux new -s var_exp
python3 -u experiments/run_large_scale_mgrid_multiseed.py \
  --M-grid 50 100 300 500 1000 2000 \
  --power-M 300 \
  --B 500 \
  --alpha 0.05 \
  --deltas 0.05 0.1 0.15 0.2 0.3 0.5 \
  --seeds 42 \
  --jobs 4 \
  --seed-workers 1 \
  --tag v5_t500_fro
```

```bash
# 退出 tmux（不终止实验）
Ctrl-b d

# 重新进入会话
tmux attach -t var_exp
```

### 确认实验在运行

```bash
# 1) 进程是否存在
ps -ef | grep run_large_scale_mgrid_multiseed.py | grep -v grep

# 2) 总进度是否在更新
cat results/large_scale_runs/<run_name>/progress/summary.json

# 3) 进度日志是否持续追加
tail -f results/large_scale_runs/<run_name>/progress/progress.log
```

---

## 5. 计算量瓶颈

各模型耗时差异大：

| 模型 | 单次 MC 迭代耗时(B=500) | 说明 |
|---|---|---|
| baseline_ols | ~2s | Bootstrap LR，N=2 矩阵运算快 |
| baseline_ols_f | ~0.09s | 渐近 F，无 bootstrap |
| sparse_lasso | ~10s | Lasso 求解慢，N=5 |
| lowrank_svd | ~16s | SVD 求解慢，N=10 |

核心瓶颈是 sparse/lowrank 的 Lasso/SVD 优化求解。

---

## 6. 第二层实验：真高维（OLS 不可行）

脚本：`experiments/run_highdim_v2_infeasible_ols.py`

### 在论文中的角色

本实验是论文两层递进证明的第二层（第一层为 Section 1 的低维基准实验），回答"为什么需要正则化方法"。通过制造 OLS 不可行的场景，展示 Lasso/SVD 的不可替代性。详见 `simulation_plan.md` Section 11。

### 实验设计

在 OLS 真正不可行（参数量 >> 每段观测数）的场景下，用结构匹配扰动验证 Lasso 和 SVD 的 Bootstrap Sup-LR 检验。结构匹配扰动（稀疏支撑集 / 列空间低秩）的设计基于实证观察：真实世界的结构断裂是保结构的（见 `simulation_plan.md` Section 11.2）。

- $T=300$，$p=1$，$t^*=150$，每段有效观测约 149
- $N=20$：OLS 参数量 420，参数/观测比 2.82 → OLS 不可行
- $N=30$：OLS 参数量 930，参数/观测比 6.24 → OLS 严重不可行

### 6 个模型

| 模型 | N | 方法 | p 值方法 | 扰动类型 | 说明 |
|---|---:|---|---|---|---|
| `baseline_ols_f_n20` | 20 | OLS | 渐近 F | 均匀全 1 | 对照：预期 size → 1 |
| `baseline_ols_f_n30` | 30 | OLS | 渐近 F | 均匀全 1 | 对照：预期 size → 1 |
| `sparse_lasso_n20` | 20 | Lasso | Bootstrap LR | 稀疏支撑集 | 结构匹配扰动 |
| `sparse_lasso_n30` | 30 | Lasso | Bootstrap LR | 稀疏支撑集 | 结构匹配扰动 |
| `lowrank_svd_n20` | 20 | SVD | Bootstrap LR | 列空间低秩 | 结构匹配扰动 |
| `lowrank_svd_n30` | 30 | SVD | Bootstrap LR | 列空间低秩 | 结构匹配扰动 |

> **设计说明**：OLS 对照组仅使用渐近 F 检验，不设 Bootstrap OLS 对照。原因：OLS 在欠定场景（$X^\top X$ 奇异）下的失败发生在参数估计阶段，与 p 值计算方式无关——无论用渐近 F 还是 Bootstrap LR，都在同一步骤崩溃。F-test 对照组已能清晰展示 size 爆炸现象，增加 Bootstrap OLS 只会重复相同的数值崩溃，增加大量计算开销而不提供额外信息。

### 启动命令

```bash
# 快速验证（B=100, 单 seed）
python3 -u experiments/run_highdim_v2_infeasible_ols.py \
  --B 100 --seeds 42 --jobs 8 --tag smoke_test

# 正式实验（B=500, 双 seed）
python3 -u experiments/run_highdim_v2_infeasible_ols.py \
  --B 500 --seeds 42 2026 --jobs 8 --tag v1_structured
```

### 命令行参数

与主实验脚本参数一致，差异：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--B` | 500 | Bootstrap 重复次数 |
| `--power-M` | 500 | Power 评估 MC 次数 |
| `--models` | 全部 6 个 | 可指定子集，如 `--models sparse_lasso_n20 lowrank_svd_n20` |

固定参数（脚本内定义）：`T=300`, `p=1`, `t=150`。

### 输出目录

```text
results/highdim_infeasible_runs/
├── <run_name>/
│   ├── highdim_infeasible_*.json
│   ├── highdim_infeasible_raw_*.csv
│   ├── highdim_infeasible_agg_*.csv
│   ├── 真高维仿真报告_*.md
│   ├── run_meta.json
│   ├── seed_results/
│   │   └── seed_<seed>.json
│   └── progress/
│       ├── progress.log
│       ├── progress.jsonl
│       ├── summary.json
│       └── seed_<seed>_summary.json
```

### 关键输出字段

Power 结果额外记录 `perturbation_type`（`sparse` / `lowrank` / `uniform_ones`），可用于验证扰动方向是否正确路由。

聚合 CSV 包含跨 seed 的 `mc_se_pooled` 和 `ci95_low/ci95_high`，可直接用于绘制误差带。

### 验证清单

1. smoke test 确认 6 个模型正常完成
2. Lasso/SVD 的 size@M=2000 在 [0.03, 0.07] 内
3. OLS(F) 的 size 接近 1.0（size 爆炸）
4. power 随 $\delta$ 单调增加
5. actual_fro 接近 target_fro
6. 报告中"扰动类型"字段记录正确
