# 仿真实验运行文档

## 1. 实验脚本

`experiments/run_next_stage_large_scale_p3.py` — 下一阶段大规模实验（p=3，多seed，均值+95%CI）。

### 实验内容

| 阶段 | 说明 |
|---|---|
| baseline_pvalue | 比较渐近 p 值（chi2/F）与 Bootstrap p 值，随 B 变化 |
| baseline_validation | baseline Chow 检验的 Type I Error 与 Power，随 M 变化 |
| sparse_validation | 稀疏 Lasso bootstrap 检验的 Type I Error 与 Power |
| lowrank_validation | 低秩 SVD bootstrap 检验的 Type I Error 与 Power |

每个 seed 依次完成以上 4 个阶段，多 seed 并行（由 `--seed-workers` 控制）。

### 启动命令

```bash
nohup python3 -u experiments/run_next_stage_large_scale_p3.py \
  --seeds 42 2026 7 \
  --seed-workers 2 \
  --B-grid 100 300 500 \
  --M-grid 50 100 200 \
  --B-mc-baseline 50 \
  --B-mc-highdim 30 \
  --tag <tag_name> \
  > results/next_stage_runs/<log_name>.log 2>&1 &
```

### 监控命令

```bash
# 实时进度
tail -f results/next_stage_runs/<log_name>.log

# 检查进程
ps -p <PID> -o pid,stat,etime,rss

# 查看 worker 日志
tail -f results/next_stage_runs/<run_dir>/worker_logs/seed_*.log
```

---

## 2. 命令行参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--alpha` | 0.05 | 显著性水平 |
| `--seeds` | [42, 2026, 7] | 随机种子列表，决定重复次数 |
| `--B-grid` | [100, 300, 500] | baseline p值对比中的 bootstrap 重复次数网格 |
| `--M-grid` | [50, 100, 200] | validation 中的 Monte Carlo 重复次数网格 |
| `--B-mc-baseline` | 50 | baseline validation 中每次 MC 迭代的 bootstrap 重复数 |
| `--B-mc-highdim` | 30 | sparse/lowrank validation 中每次 MC 迭代的 bootstrap 重复数 |
| `--seed-workers` | 0 (自动) | 并行 worker 数，0 表示按 CPU 核数自动选择 |
| `--tag` | next_stage_p3 | 运行标签，体现在输出目录名中 |

### 场景维度配置（硬编码在脚本中）

| 场景 | N (变量数) | T (序列长) | t (断点位置) | 说明 |
|---|---|---|---|---|
| baseline | 2 | 150 | 75 | 低维，传统 Chow 检验 |
| sparse | 3 | 120 | 60 | 中维，Lasso 稀疏估计 |
| lowrank | 4 | 120 | 60 | 较高维，SVD 低秩估计 |

所有场景统一 p=3（VAR 阶数）。

---

## 3. 输出目录结构

每次运行在 `results/next_stage_runs/` 下创建一个带时间戳的目录：

```
results/next_stage_runs/
├── <YYYY-MM-DD_HHMMSS>_<tag>/              # 运行目录
│   ├── progress.log                         # 主进度日志
│   ├── run_meta.json                        # 运行元信息（路径等）
│   ├── next_stage_summary_*.json            # 完整配置+聚合结果
│   ├── 下一阶段大规模试验报告_*.md           # Markdown 实验报告
│   │
│   ├── next_stage_baseline_raw_*.csv        # 各seed各B的原始p值
│   ├── next_stage_baseline_agg_*.csv        # 跨seed聚合p值（均值+CI）
│   ├── next_stage_validation_raw_*.csv      # 各seed各M的type1/power原始值
│   ├── next_stage_validation_agg_*.csv      # 跨seed聚合type1/power（均值+CI）
│   │
│   ├── next_stage_baseline_pvalues_vs_B_*.png   # p值随B变化图
│   ├── next_stage_model_type1_vs_M_*.png        # Type I Error随M变化图
│   ├── next_stage_model_power_vs_M_*.png        # Power随M变化图
│   │
│   ├── worker_logs/                         # 子进程日志
│   │   ├── seed_42.log
│   │   ├── seed_2026.log
│   │   └── seed_7.log
│   │
│   └── state/                               # 运行中间状态
│       ├── seed_42.json                     # seed 42 完整结果 bundle
│       ├── seed_42.progress                 # seed 42 阶段完成事件
│       ├── seed_2026.json
│       ├── seed_2026.progress
│       ├── seed_7.json
│       ├── seed_7.progress
│       └── mpl_*/fontlist-*.json            # matplotlib 字体缓存（可忽略）
│
└── <log_name>.log                           # nohup 主进程日志（与 progress.log 内容相同）
```

---

## 4. 实验规模与运行时间对应关系

以下为在 2 核 CPU（seed_workers=2）上的实测/估算数据：

### 已完成的运行记录

| 运行 | seeds | M_grid | B_mc_baseline | B_mc_highdim | T(baseline/sparse,lowrank) | 耗时 | 状态 |
|---|---|---|---|---|---|---|---|
| formal_5seed_v2 | 5个 | [300,500,800] | 80 | 100 | 220/220 | 预估16-24h | 中断（过慢） |
| formal_fast_3seed | 3个 | [100,200,300] | 50 | 50 | 150/150 | >2.5h 仍在 4/12 | 中断（仍过慢） |
| formal_fast_v3 | 3个 | [50,100,200] | 50 | 30 | 150/120 | **4小时8分** | **已完成** |

### 计算量瓶颈分析

各阶段耗时占比极不均匀：

| 阶段 | 单 seed 计算量 | 耗时占比 | 说明 |
|---|---|---|---|
| baseline_pvalue | len(B_grid) 次 bootstrap | <1% | 仅做 1 次数据生成 + 几次 bootstrap |
| baseline_validation | sum(M_grid) × 2 × B_mc_baseline | ~10% | 低维(N=2)，矩阵运算快 |
| sparse_validation | sum(M_grid) × 2 × B_mc_highdim | **~45%** | Lasso 求解慢，N=3 |
| lowrank_validation | sum(M_grid) × 2 × B_mc_highdim | **~45%** | SVD 求解慢，N=4 |

**核心瓶颈**：sparse/lowrank 的单次迭代包含 Lasso/SVD 优化求解，比 baseline 的 OLS 慢 1-2 个数量级。

### 并行后端优化（loky）

Monte Carlo 外层并行已从标准库 `ProcessPoolExecutor` 切换为 **loky**（`joblib.externals.loky`）。

| 项目 | 改动前 | 改动后 |
|---|---|---|
| 进程池后端 | `ProcessPoolExecutor`，受限环境自动回退 `ThreadPoolExecutor` | `loky.get_reusable_executor`（优先），回退链：loky → ProcessPool → ThreadPool |
| 受限环境表现 | 回退线程池，CPU ~130%，GIL 限制真并行 | loky 绕过 POSIX semaphore 限制，真多进程并行 |
| Worker 进程复用 | 每次 `run_task_map` 创建新进程池 | loky 复用已有 worker 进程，减少 fork/spawn 开销 |
| 改动文件 | — | `simulation/parallel.py` |
| 依赖 | 无额外依赖 | `joblib`（已作为 scikit-learn 的依赖存在） |

验证运行（M_grid=[20,50], B=50, seeds=[42], jobs=4）：12/12 stages 完成，三模型并行正常，62.62 秒。

### Worker 动态回收（baseline 先跑完）

`run_all_models_for_seed` 的模型调度已从"三模型同时启动、预分配 worker"改为**两阶段调度**：

1. **Phase 1**：baseline_ols 串行跑完（~0.5s，无需多进程）；
2. **Phase 2**：全部 jobs 预算分给 sparse_lasso 和 lowrank_svd 并行执行，按 2/3 : 1/3 分配（sparse 是瓶颈，优先拿更多 worker）。

| jobs | 改前分配 (baseline/sparse/lowrank) | 改后分配 (baseline → sparse/lowrank) |
|---|---|---|
| 4 | 1 / 2 / 1 同时启动 | 1 串行 → 3 / 1 并行 |
| 6 | 1 / 3 / 2 同时启动 | 1 串行 → 4 / 2 并行 |
| 8 | 1 / 4 / 3 同时启动 | 1 串行 → 6 / 2 并行 |

验证结果（同参数 jobs=4）：sparse 从 62.6s 降至 56.4s，总 wall time 从 62.6s 降至 **56.9s（~9% 加速）**。

改动文件：`experiments/run_large_scale_mgrid_multiseed.py`（`run_all_models_for_seed` 函数）。

### 向量化伪序列生成

Bootstrap 伪序列生成函数 `generate_pseudo_series` 中的内层滞后向量组装循环已向量化：

```python
# 改前：Python 双层循环
for t in range(p, T):
    Y_lag_ordered = np.zeros(N * p)
    for lag in range(p):
        Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]

# 改后：numpy 切片一行替代
for t in range(p, T):
    Y_lag_ordered = Y_star[t-p:t, :][::-1].ravel()
```

- 数值结果 **bit-for-bit 相同**（已验证 sum、元素值完全一致）；
- 单次伪序列生成加速 ~22%（500 次计时：0.971s → 0.760s）；
- 统计方案零影响。

改动文件（共 6 处，逻辑相同）：

| 文件 | 函数 |
|---|---|
| `simulation/bootstrap.py` | `generate_pseudo_series` |
| `sparse_var/sparse_bootstrap.py` | `generate_pseudo_series` |
| `lowrank_var/lowrank_bootstrap.py` | `generate_pseudo_series` |
| `simulation/chow_bootstrap.py` | `generate_pseudo_series` |
| `simulation/data_generator.py` | `generate_var_series` |
| `simulation/data_generator.py` | `generate_var_with_break` |

### 三项优化累计效果

| 版本 | wall time | 相对初始加速 |
|---|---|---|
| 初始（ThreadPool 回退） | 62.6s | — |
| +loky 真多进程 | 62.6s | (小规模测试，大 M 时收益更显著) |
| +worker 动态回收 | 56.9s | 9% |
| +向量化伪序列 | **54.5s** | **13%** |

### 参数调整对运行时间的影响

| 调整 | 时间缩减倍数 | 对结果影响 |
|---|---|---|
| M_grid 总量减半 | ~2x | CI 更宽，趋势仍可见（需 M≥30） |
| B_mc_highdim 减半 | ~2x（仅影响 sparse/lowrank） | bootstrap 分布略粗糙（需 B≥20） |
| T 减小 | ~1.3-1.5x | 每次迭代更快，但 T 不宜低于 100 |
| seeds 减少 | 线性（受 worker 批次影响） | CI 更宽（需 seeds≥3） |
| N 增大 | 超线性增长 | N=4→5 约 2-3x 变慢 |

### 推荐参数组合

| 目标时间 | seeds | M_grid | B_mc_highdim | T(sparse/lowrank) | 预计耗时 |
|---|---|---|---|---|---|
| 快速验证 | 3个 | [20,50] | 20 | 100 | ~30min |
| 标准实验 | 3个 | [50,100,200] | 30 | 120 | ~4h |
| 完整实验 | 5个 | [100,200,500] | 50 | 150 | ~16-20h |
| 论文级别 | 5个 | [300,500,800] | 100 | 220 | ~24-48h |

---

## 5. 已完成正式实验结果

**运行目录**: `results/next_stage_runs/2026-03-06_235232_formal_fast_v3/`

- 开始时间: 2026-03-06 23:52:32
- 完成时间: 2026-03-07 04:00:02
- 总耗时: **4 小时 8 分钟**
- 参数: 3 seeds, M_grid=[50,100,200], B_mc_baseline=50, B_mc_highdim=30, T=150/120


---

## 6. 当前主实验脚本（M_grid + 多 seed / 单 seed）

`experiments/run_large_scale_mgrid_multiseed.py` — 当前主用的大规模结构断裂实验脚本。

### 已修复的问题

当前版本相较于前一版，已经补全以下机制：

- 结构断裂检验口径统一；
- `size` 与 `power` 的 `M` 使用逻辑分离：`size` 看 `M_grid`，`power` 固定看 `M_max`；
- 多 seed 并行；
- 按维度缩放的效应量网格；
- `baseline_ols` 的渐近 / bootstrap p 值口径切换；
- `progress/` 目录下的持久化进度日志；
- 失败 / 中断必须写日志，避免无感退出。

### 实验逻辑

- 第一类错误（size）：在 `M_grid` 下逐个评估；
- 功效（power）：固定使用 `M_max = max(M_grid)` 评估；
- 支持单 seed 或多 seed；
- 支持 seed 并行、模型并行、Monte Carlo 外层并行；
- `baseline_ols` 支持 `bootstrap_lr`、`asymptotic_chi2`、`asymptotic_f` 三种 p 值口径。

### 启动命令示例

```bash
python3 -u experiments/run_large_scale_mgrid_multiseed.py \
  --M-grid 50 100 300 \
  --B 500 \
  --alpha 0.05 \
  --deltas 0.04 0.08 0.12 0.16 \
  --seeds 42 \
  --jobs 4 \
  --seed-workers 1 \
  --baseline-pvalue-method asymptotic_chi2 \
  --tag single_seed_m300_b500
```

### 输出目录结构

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

### 进度日志解释

- `progress/progress.log`
  - 全实验的人类可读进度日志；
  - 记录 `run / seed / model / stage` 的 started/completed/failed 事件。
  - 若收到终止信号或出现异常，日志中会明确出现 `failed` 事件。

- `progress/progress.jsonl`
  - 全实验的结构化事件流；
  - 适合程序化监控。

- `progress/summary.json`
  - 全实验总进度摘要；
  - 包含 `completed_stage_count / total_stage_count / progress_ratio / active_stages`。
  - 若实验失败或被中断，也会同步更新状态。

- `progress/seed_<seed>_summary.json`
  - 单个 seed 的独立进度摘要；
  - 只反映该 seed 自己的完成情况，不混入其他 seed。


### 长实验的正确后台运行方式（避免无感退出）

对于 1 小时以上的大实验，**不建议只靠一次性 `nohup ... &` 从临时代理命令中启动**。在当前 Codex / CLI 使用方式下，这种启动方式可能会随着外层会话结束而被回收，表现为：

- 启动日志只写了开头几行；
- `progress/summary.json` 长时间不再更新；
- 进程消失，但目录里没有明确结果文件。

更稳妥的做法有两种：

#### 方式 A：使用 `tmux`（推荐）

```bash
tmux new -s var_exp
python3 -u experiments/run_large_scale_mgrid_multiseed.py \
  --M-grid 50 100 300 \
  --B 500 \
  --alpha 0.05 \
  --deltas 0.04 0.08 0.12 0.16 \
  --seeds 42 \
  --jobs 4 \
  --seed-workers 1 \
  --baseline-pvalue-method asymptotic_chi2 \
  --tag single_seed_m300_b500
```

启动后：

```bash
# 退出 tmux，但不终止实验
Ctrl-b d

# 重新进入会话
tmux attach -t var_exp
```

#### 方式 B：在一个持久的交互式终端里直接运行

如果不使用 `tmux`，也应保证启动实验的终端会话本身持续存在，不要在启动后立即销毁该 shell 会话。

### 启动后如何确认实验真的在跑

至少同时检查以下三项：

```bash
# 1) 进程是否存在
ps -ef | grep run_large_scale_mgrid_multiseed.py | grep -v grep

# 2) 总进度是否在更新
cat results/large_scale_runs/<run_name>/progress/summary.json

# 3) 进度日志是否持续追加
tail -f results/large_scale_runs/<run_name>/progress/progress.log
```

如果出现以下现象之一，通常说明实验已经退出：

- 进程不存在；
- `progress/summary.json` 的 `updated_at` 长时间不变；
- `progress.log` 长时间没有新记录；
- 目录里没有最终 JSON / CSV / MD 输出。

### 监控命令

```bash
# 看整个实验总进度
cat results/large_scale_runs/<run_name>/progress/summary.json

# 看单个 seed 进度
cat results/large_scale_runs/<run_name>/progress/seed_42_summary.json

# 实时看全局进度日志
tail -f results/large_scale_runs/<run_name>/progress/progress.log

# 看后台启动日志（如果用 nohup）
tail -f results/large_scale_runs/<run_name>/launch.log
```
