# 仿真实验运行文档

## 1. 主实验脚本

`experiments/run_structured_scenarios.py` — 结构化 VAR 断裂检验三层四模型实验。

### 实验逻辑

- 三层实验框架：第一层（OLS 基准）→ 第二层（稀疏 Lasso）→ 第三层（低秩 RRR）
- 第一类错误（size）：在 `M_grid` 下逐个评估
- 功效（power）：固定使用 `power_M`（默认 `max(M_grid)`，可通过 `--power-M` 独立指定）评估
- 支持单 seed 或多 seed，支持 seed 并行
- `baseline_ols` 使用 `bootstrap_lr` p 值；`baseline_ols_f` 使用 `asymptotic_f` 作为对照
- 四个模型统一 `T = 500`、`t = 250`，保证每段有效样本量一致

### 四类模型

| 模型 | N | T | t | p | 估计器 | p 值方法 | DGP | 扰动类型 |
|---|---:|---:|---:|---:|---|---|---|---|
| baseline_ols_f | 10 | 500 | 250 | 1 | OLS | asymptotic_f | 稠密 | uniform |
| baseline_ols | 10 | 500 | 250 | 1 | OLS | bootstrap_lr | 稠密 | uniform |
| sparse_lasso | 20 | 500 | 250 | 1 | Lasso(α=0.02) | bootstrap_lr | 稀疏(0.15) | sparse |
| lowrank_rrr | 20 | 500 | 250 | 1 | RRR(rank=2) | bootstrap_lr | 低秩(rank=2) | lowrank |

### 启动命令

```bash
# 正式实验（推荐参数）
python3 -u experiments/run_structured_scenarios.py \
  --B 500 --seeds 42 2026 --jobs 10 --power-M 500 \
  --deltas 0.1 0.3 0.5 1.0 --tag final

# 仅跑指定模型
python3 -u experiments/run_structured_scenarios.py \
  --B 500 --seeds 42 2026 --jobs 10 --power-M 500 \
  --models lowrank_rrr --deltas 0.1 0.3 0.5 1.0 --tag lowrank_only

# 跳过 type1，仅跑 power
python3 -u experiments/run_structured_scenarios.py \
  --B 500 --seeds 42 2026 --jobs 10 --power-M 500 \
  --skip-type1 --deltas 0.1 0.3 0.5 1.0 --tag power_only

# 快速 smoke test
python3 -u experiments/run_structured_scenarios.py \
  --B 20 --seeds 42 --jobs 8 --M-grid 50 --deltas 0.1 0.3 --tag smoke
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--M-grid` | [50, 100, 300, 500, 1000, 2000] | 第一类错误评估的 Monte Carlo 重复次数网格 |
| `--power-M` | 0 (= max(M_grid)) | 功效评估的 MC 重复次数，独立于 M_grid |
| `--B` | 500 | 每次检验的 bootstrap 重复次数 |
| `--alpha` | 0.05 | 显著性水平 |
| `--deltas` | [0.05, 0.1, 0.15, 0.2, 0.3, 0.5] | Frobenius 效应量网格 |
| `--seeds` | [42, 2026] | 随机种子列表 |
| `--jobs` | 4 | 总并行 worker 数 |
| `--seed-workers` | 1 | 并发 seed 数（默认顺序跑 seed，独占全部 jobs） |
| `--skip-type1` | false | 跳过 Type I error 评估 |
| `--skip-power` | false | 跳过 Power 评估 |
| `--models` | 全部 4 个 | 指定运行的模型子集 |
| `--tag` | (空) | 运行标签，附加到输出目录名 |

可选模型名：`baseline_ols_f`, `baseline_ols`, `sparse_lasso`, `lowrank_rrr`

### 固定参数（脚本内定义）

| 参数 | 值 | 说明 |
|---|---|---|
| T | 500 | 总样本长度 |
| t* | 250 | 已知断点位置 |
| p | 1 | VAR 滞后阶数 |
| N_baseline | 10 | 第一层维度 |
| N_sparse | 20 | 第二层维度 |
| N_lowrank | 20 | 第三层维度 |
| sparsity | 0.15 | 稀疏 DGP 非零元素比例 |
| lowrank_rank | 2 | 低秩 DGP 真实秩 |
| lasso_alpha | 0.02 | Lasso 正则化参数 |
| rrr_rank | 2 | RRR 目标秩 |
| lowrank_target_sr | 0.40 | 低秩 DGP 目标谱半径 |

---

## 2. 输出目录结构

每次运行在 `results/structured_scenario_runs/<timestamp>_<tag>/` 下输出：

```text
results/structured_scenario_runs/
├── <run_name>/
│   ├── structured_<run_name>.json
│   ├── structured_raw_<run_name>.csv
│   ├── structured_agg_<run_name>.csv
│   ├── 结构化场景仿真报告_<run_name>.md
│   ├── seed_results/
│   │   └── seed_<seed>.json
│   └── progress/
│       ├── progress.log
│       ├── progress.jsonl
│       └── summary.json
```

### 进度日志

- `progress/progress.log`：人类可读进度日志，记录 started/completed/failed 事件。
- `progress/progress.jsonl`：结构化事件流，适合程序化监控。
- `progress/summary.json`：总进度摘要（completed_stage_count / total_stage_count / progress_ratio）。

若实验异常退出或收到终止信号，日志中会显式记录 `failed` 事件。

---

## 3. 模型调度与并行策略

### 顺序执行

四个模型按固定顺序依次执行，每个模型独占全部 `--jobs` 个 worker：

```
baseline_ols_f(全部w) → baseline_ols(全部w) → sparse_lasso(全部w) → lowrank_rrr(全部w)
```

顺序独占的原因：同一进程内多模型并行时 worker pool 共享，实际并行度 = max(各请求 n_jobs) 而非求和。顺序独占使每个模型使用全部 worker，CPU 利用率最大化。

### 并行后端

Monte Carlo 外层并行使用 `ProcessPoolExecutor` 进程池。

---

## 4. 长实验运行建议

对于 1 小时以上的实验，推荐使用 `tmux` 避免进程被回收：

```bash
tmux new -s var_exp
python3 -u experiments/run_structured_scenarios.py \
  --B 500 --seeds 42 2026 --jobs 10 --power-M 500 \
  --deltas 0.1 0.3 0.5 1.0 --tag final
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
ps -ef | grep run_structured_scenarios.py | grep -v grep

# 2) 总进度是否在更新
cat results/structured_scenario_runs/<run_name>/progress/summary.json

# 3) 进度日志是否持续追加
tail -f results/structured_scenario_runs/<run_name>/progress/progress.log
```

---

## 5. 计算量参考

各模型单次 MC 迭代耗时（B=500，参考值）：

| 模型 | 单次迭代耗时 | 说明 |
|---|---|---|
| baseline_ols_f | ~0.1s | 渐近 F，无 bootstrap |
| baseline_ols | ~2s | Bootstrap LR，N=10 |
| sparse_lasso | ~600s / delta | Lasso 求解慢，N=20 |
| lowrank_rrr | ~125s / delta | RRR 求解，N=20 |

计算量瓶颈在 sparse_lasso（每个 delta 点约 10 分钟）和 lowrank_rrr（每个 delta 点约 2 分钟）。

---

## 6. 实证应用脚本

### 稀疏模型实证

```bash
python3 applications/cross_asset_sparse_test.py --B 500 --verbose
```

- N=5 跨资产 ETF（SPY, AGG, TLT, GLD, VNQ）
- 估计：Lasso + Post-Lasso OLS（CV 选 alpha）
- 输出：`results/empirical/cross_asset_sparse_*.json`

### 低秩模型实证

```bash
python3 applications/sector_lowrank_test.py --B 500 --verbose
```

- N=11 SPDR 行业 ETF
- 估计：SVD 截断（自动选秩）
- 输出：`results/empirical/sector_lowrank_*.json`
