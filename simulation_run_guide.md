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
