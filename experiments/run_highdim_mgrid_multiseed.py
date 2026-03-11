"""高维 VAR 结构断裂检验仿真实验（多维度 × 多方法对称设计）

设计：
  4 种方法 × 3 个 N 维度 = 12 个模型
    N ∈ {5, 10, 20}；T = 1000（已知断点 t = 500）
  Type I error：M_grid = (50, 100, 300, 500, 1000, 2000)，每个 M 都跑
  Power       ：固定 M = 500（3 seed 聚合后精度足够），跑所有 delta
  B = 500 bootstrap 迭代，seeds = (42, 2026, 7)

OLS 可行性（T=1000, p=1, t=500，每段有效观测约 499-500）：
  N=5 : params=30,  obs=499, ratio=0.06 ✓
  N=10: params=110, obs=499, ratio=0.22 ✓
  N=20: params=420, obs=499, ratio=0.84 ✓（可行但紧）

核心叙事：
  OLS 在三个 N 水平下均可行，但高维时估计效率下降、size 控制变差；
  Lasso（稀疏先验）和 SVD（低秩先验）在高维下依然保持良好的 size 控制和检验功效。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation import MonteCarloSimulation, VARDataGenerator  # noqa: E402
from sparse_var import SparseMonteCarloSimulation               # noqa: E402
from lowrank_var import LowRankMonteCarloSimulation             # noqa: E402


# ---------------------------------------------------------------------------
# 实验固定参数
# ---------------------------------------------------------------------------
_T: int = 1000   # 时间序列长度
_p: int = 1      # VAR 阶数
_t: int = 500    # 已知断点位置

# 模型执行顺序：OLS bootstrap → OLS F-test → Lasso → SVD，各 N 水平
MODEL_EXECUTION_ORDER: Tuple[str, ...] = (
    "baseline_ols_n5",   "baseline_ols_n10",   "baseline_ols_n20",
    "baseline_ols_f_n5", "baseline_ols_f_n10", "baseline_ols_f_n20",
    "sparse_lasso_n5",   "sparse_lasso_n10",   "sparse_lasso_n20",
    "lowrank_svd_n5",    "lowrank_svd_n10",    "lowrank_svd_n20",
)

# 作业优先级：慢模型（大 N、bootstrap）优先分配资源
MODEL_JOB_PRIORITY: Tuple[str, ...] = (
    "sparse_lasso_n20",  "lowrank_svd_n20",
    "sparse_lasso_n10",  "lowrank_svd_n10",
    "sparse_lasso_n5",   "lowrank_svd_n5",
    "baseline_ols_n20",  "baseline_ols_n10",  "baseline_ols_n5",
    "baseline_ols_f_n20","baseline_ols_f_n10","baseline_ols_f_n5",
)


_ACTIVE_TRACKER = None
_RUN_START_TIME = None


def _install_signal_handlers(tracker):
    def _handler(signum, _frame):
        global _ACTIVE_TRACKER, _RUN_START_TIME
        runtime = 0.0 if _RUN_START_TIME is None else max(0.0, time.time() - _RUN_START_TIME)
        if _ACTIVE_TRACKER is tracker:
            try:
                tracker.log_run_failed(runtime, f"terminated_by_signal_{signum}")
            except Exception:
                pass
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ===========================================================================
# 配置
# ===========================================================================

@dataclass
class ExperimentConfig:
    M_grid: Tuple[int, ...] = (50, 100, 300, 500, 1000, 2000)
    B: int = 500
    alpha: float = 0.05
    deltas: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
    seeds: Tuple[int, ...] = (42, 2026, 7)
    jobs: int = 4
    seed_workers: int = 0
    baseline_pvalue_method: str = "bootstrap_lr"
    _power_M: int = 500
    skip_type1: bool = False

    @property
    def power_M(self) -> int:
        return self._power_M if self._power_M > 0 else max(self.M_grid)


# ===========================================================================
# 进度追踪（与原脚本一致）
# ===========================================================================

@dataclass
class ProgressTracker:
    run_dir: str
    total_stage_count: int
    per_seed_stage_count: int
    progress_dir: str = field(init=False)
    progress_log_path: str = field(init=False)
    progress_jsonl_path: str = field(init=False)
    summary_path: str = field(init=False)
    completed_stage_count: int = field(default=0, init=False)
    active_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    completed_stages: List[str] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.progress_dir = os.path.join(self.run_dir, "progress")
        os.makedirs(self.progress_dir, exist_ok=True)
        self.progress_log_path = os.path.join(self.progress_dir, "progress.log")
        self.progress_jsonl_path = os.path.join(self.progress_dir, "progress.jsonl")
        self.summary_path = os.path.join(self.progress_dir, "summary.json")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _seed_summary_path(self, seed: int) -> str:
        return os.path.join(self.progress_dir, f"seed_{seed}_summary.json")

    def _stage_id(self, seed: int, model: str, stage: str) -> str:
        return f"seed_{seed}/{model}/{stage}"

    def _write_summary(self) -> None:
        payload = {
            "updated_at": self._timestamp(),
            "completed_stage_count": self.completed_stage_count,
            "total_stage_count": self.total_stage_count,
            "progress_ratio": 0.0 if self.total_stage_count <= 0 else self.completed_stage_count / self.total_stage_count,
            "active_stages": sorted(self.active_stages.keys()),
            "completed_stages": list(self.completed_stages[-20:]),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        seeds = sorted({details["seed"] for details in self.active_stages.values() if details.get("seed") is not None})
        for stage_id in self.completed_stages:
            if stage_id.startswith("seed_"):
                seed_str = stage_id.split("/", 1)[0].replace("seed_", "")
                if seed_str.isdigit():
                    seeds.append(int(seed_str))
        for seed in sorted(set(seeds)):
            self._write_seed_summary(seed)

    def _write_seed_summary(self, seed: int) -> None:
        seed_prefix = f"seed_{seed}/"
        completed = 0
        active_stage_ids: List[str] = []
        completed_stage_ids: List[str] = []
        for stage_id in self.completed_stages:
            if stage_id.startswith(seed_prefix):
                completed += 1
                completed_stage_ids.append(stage_id)
        for stage_id in self.active_stages.keys():
            if stage_id.startswith(seed_prefix):
                active_stage_ids.append(stage_id)
        payload = {
            "updated_at": self._timestamp(),
            "seed": seed,
            "completed_stage_count": completed,
            "total_stage_count": self.per_seed_stage_count,
            "progress_ratio": 0.0 if self.per_seed_stage_count <= 0 else completed / self.per_seed_stage_count,
            "active_stages": active_stage_ids,
            "completed_stages": completed_stage_ids,
        }
        with open(self._seed_summary_path(seed), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _append(self, event: Dict[str, Any]) -> None:
        parts = [
            f"[{event['timestamp']}]",
            f"event={event['event_type']}",
            f"status={event['status']}",
            f"progress={self.completed_stage_count}/{self.total_stage_count}",
        ]
        if event.get("seed") is not None:
            parts.append(f"seed={event['seed']}")
        if event.get("model"):
            parts.append(f"model={event['model']}")
        if event.get("stage"):
            parts.append(f"stage={event['stage']}")
        if event.get("runtime_sec") is not None:
            parts.append(f"runtime_sec={event['runtime_sec']:.2f}")
        extra = event.get("extra") or {}
        for key in sorted(extra):
            parts.append(f"{key}={extra[key]}")
        line = " | ".join(parts)
        with open(self.progress_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        with open(self.progress_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._write_summary()

    def record_event(self, event_type: str, status: str, seed: int | None = None,
                     model: str = "", stage: str = "", runtime_sec: float | None = None,
                     count_complete: bool = False, extra: Dict[str, Any] | None = None) -> None:
        extra = extra or {}
        with self._lock:
            if event_type == "seed" and seed is not None and status == "started":
                self._write_seed_summary(seed)
            stage_id = self._stage_id(seed, model, stage) if seed is not None and model and stage else ""
            if status == "started" and stage_id:
                self.active_stages[stage_id] = {"seed": seed, "model": model, "stage": stage, **extra}
            elif stage_id and stage_id in self.active_stages:
                self.active_stages.pop(stage_id, None)
                if count_complete:
                    self.completed_stage_count += 1
                    self.completed_stages.append(stage_id)

            event = {
                "timestamp": self._timestamp(),
                "event_type": event_type,
                "status": status,
                "seed": seed,
                "model": model or None,
                "stage": stage or None,
                "runtime_sec": runtime_sec,
                "extra": extra or None,
            }
            self._append(event)

    def log_run_started(self, cfg: ExperimentConfig) -> None:
        self.record_event("run", "started", extra={
            "M_grid": list(cfg.M_grid), "power_M": cfg.power_M, "B": cfg.B,
            "alpha": cfg.alpha, "deltas": list(cfg.deltas), "seeds": list(cfg.seeds),
            "jobs": cfg.jobs, "seed_workers": cfg.seed_workers,
            "baseline_pvalue_method": cfg.baseline_pvalue_method,
        })

    def log_run_finished(self, runtime_sec: float) -> None:
        self.record_event("run", "completed", runtime_sec=runtime_sec, extra={"completed_stage_count": self.completed_stage_count})

    def log_run_failed(self, runtime_sec: float, error: str) -> None:
        self.record_event("run", "failed", runtime_sec=runtime_sec, extra={"error": error, "completed_stage_count": self.completed_stage_count})

    def log_seed_started(self, seed: int, jobs: int) -> None:
        self.record_event("seed", "started", seed=seed, extra={"jobs": jobs})

    def log_seed_finished(self, seed: int, runtime_sec: float) -> None:
        self.record_event("seed", "completed", seed=seed, runtime_sec=runtime_sec)

    def log_seed_failed(self, seed: int, error: str, runtime_sec: float) -> None:
        self.record_event("seed", "failed", seed=seed, runtime_sec=runtime_sec, extra={"error": error})

    def log_model_started(self, seed: int, model: str, jobs: int) -> None:
        self.record_event("model", "started", seed=seed, model=model, extra={"jobs": jobs})

    def log_model_finished(self, seed: int, model: str, runtime_sec: float) -> None:
        self.record_event("model", "completed", seed=seed, model=model, runtime_sec=runtime_sec)

    def log_model_failed(self, seed: int, model: str, error: str, runtime_sec: float) -> None:
        self.record_event("model", "failed", seed=seed, model=model, runtime_sec=runtime_sec, extra={"error": error})

    def start_stage(self, seed: int, model: str, stage: str, **extra: Any) -> None:
        self.record_event("stage", "started", seed=seed, model=model, stage=stage, extra=extra)

    def finish_stage(self, seed: int, model: str, stage: str, runtime_sec: float, **extra: Any) -> None:
        self.record_event("stage", "completed", seed=seed, model=model, stage=stage, runtime_sec=runtime_sec, count_complete=True, extra=extra)

    def fail_stage(self, seed: int, model: str, stage: str, error: str, runtime_sec: float, **extra: Any) -> None:
        payload = {"error": error, **extra}
        self.record_event("stage", "failed", seed=seed, model=model, stage=stage, runtime_sec=runtime_sec, extra=payload)


# ===========================================================================
# 工具函数
# ===========================================================================

def make_run_dir(tag: str) -> Tuple[str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    base_dir = os.path.join("results", "highdim_runs")
    run_name = f"{stamp}{suffix}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "seed_results"), exist_ok=True)
    return stamp, run_dir


def allocate_job_budgets(total_jobs: int, slot_count: int) -> List[int]:
    total_jobs = max(1, int(total_jobs))
    slot_count = max(1, int(slot_count))
    budgets = [1] * slot_count
    if total_jobs <= slot_count:
        return budgets
    extra = total_jobs - slot_count
    idx = 0
    while extra > 0:
        budgets[idx % slot_count] += 1
        extra -= 1
        idx += 1
    return budgets


def ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int]:
    attempts = 0
    current = phi.copy()
    while not VARDataGenerator.check_stationarity(current) and attempts < max_attempts:
        current = current * shrink
        attempts += 1
    return current, VARDataGenerator.check_stationarity(current), attempts


def build_phi2_with_target_frobenius(
    phi: np.ndarray,
    target_fro: float,
    shrink: float = 0.9,
    max_attempts: int = 30,
) -> Tuple[np.ndarray, bool, int, float]:
    direction = np.ones_like(phi, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0:
        raise ValueError("无法构造有效的断点扰动方向")
    direction /= direction_norm

    scale = float(target_fro)
    attempts = 0
    candidate = phi + scale * direction
    while not VARDataGenerator.check_stationarity(candidate) and attempts < max_attempts:
        scale *= shrink
        candidate = phi + scale * direction
        attempts += 1

    return candidate, VARDataGenerator.check_stationarity(candidate), attempts, float(np.linalg.norm(candidate - phi))


def scale_base_delta_to_target_fro(phi: np.ndarray, base_delta: float) -> float:
    return float(base_delta)


def summarize_pvalues(pvalues: np.ndarray) -> Dict[str, float]:
    if pvalues is None or len(pvalues) == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan,
                "q25": math.nan, "q50": math.nan, "q75": math.nan}
    return {
        "count": int(len(pvalues)),
        "mean": float(np.mean(pvalues)),
        "std": float(np.std(pvalues)),
        "q25": float(np.quantile(pvalues, 0.25)),
        "q50": float(np.quantile(pvalues, 0.50)),
        "q75": float(np.quantile(pvalues, 0.75)),
    }


def _aggregate_numeric(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "count": 0}
    arr = np.array(clean, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(len(arr)),
    }


def _compute_pooled_mc_band(rejections: Sequence[float], effective: Sequence[float]) -> Dict[str, float]:
    total_rejections = int(sum(int(v) for v in rejections if not math.isnan(v)))
    total_effective = int(sum(int(v) for v in effective if not math.isnan(v)))
    if total_effective <= 0:
        return {
            "rejections_total": 0,
            "M_effective_total": 0,
            "mc_se_pooled": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }

    p_hat = total_rejections / total_effective
    mc_se = math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / total_effective))
    ci95_low = max(0.0, p_hat - 1.96 * mc_se)
    ci95_high = min(1.0, p_hat + 1.96 * mc_se)
    return {
        "rejections_total": total_rejections,
        "M_effective_total": total_effective,
        "mc_se_pooled": float(mc_se),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
    }


# ===========================================================================
# 模型配置工厂
# ===========================================================================

def _dense_scale(N: int) -> float:
    """稠密 N×N VAR 系数矩阵的合适 scale。

    稠密随机矩阵的谱半径约为 scale*sqrt(N)，
    为保证平稳性需 scale*sqrt(N) < 1，取安全边界 0.85。
    """
    return min(0.3, 0.85 / math.sqrt(N))


def _lowrank_scale(N: int) -> float:
    """低秩矩阵中因子的合适 scale。

    Phi 的谱范数约为 scale^2 * N，
    为保证平稳性需 scale^2 * N < 0.7。
    """
    return min(0.3, math.sqrt(0.7 / N))


def _make_ols_setup(model_name: str, N: int, use_f_test: bool,
                    generator: VARDataGenerator) -> Dict[str, Any]:
    """构造 OLS 模型（bootstrap_lr 或 asymptotic_f）的实验配置。"""
    T, p, t = _T, _p, _t
    Sigma = np.eye(N) * 0.5
    phi = generator.generate_stationary_phi(N, p, scale=_dense_scale(N))
    phi, ok, shrinks = ensure_stationary(phi)
    pv_method = "asymptotic_f" if use_f_test else "bootstrap_lr"

    def mc_factory(M, B, seed_, jobs, _baseline_pvalue_method, _pv=pv_method):
        return MonteCarloSimulation(M=M, B=B, seed=seed_, n_jobs=jobs,
                                    baseline_pvalue_method=_pv)

    def type1_fn(mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_type1_error_at_point(
            _N, _T, _p, phi_mat, sigma, t=_t, alpha=alpha, verbose=False)

    def power_fn(mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_power_at_point(
            _N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, alpha=alpha, verbose=False)

    extra = {"pvalue_method": pv_method} if use_f_test else {}
    return {
        "model": model_name, "N": N, "T": T, "p": p, "t": t,
        "Sigma": Sigma, "phi": phi, "extra_parameters": extra,
        "mc_factory": mc_factory, "type1_fn": type1_fn, "power_fn": power_fn,
    }


def _make_lasso_setup(model_name: str, N: int,
                      generator: VARDataGenerator) -> Dict[str, Any]:
    """构造 Lasso 稀疏模型的实验配置。"""
    T, p, t = _T, _p, _t
    Sigma = np.eye(N) * 0.5
    phi = generator.generate_stationary_phi(N, p, sparsity=0.2, scale=_dense_scale(N))
    phi, ok, shrinks = ensure_stationary(phi)
    lasso_alpha = 0.02

    def mc_factory(M, B, seed_, jobs, _baseline_pvalue_method, _la=lasso_alpha):
        return SparseMonteCarloSimulation(
            M=M, B=B, seed=seed_, estimator_type="lasso", alpha=_la, n_jobs=jobs)

    def type1_fn(mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_type1_error(
            _N, _T, _p, phi_mat, sigma, t=_t, test_alpha=alpha, verbose=False)

    def power_fn(mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_power(
            _N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, test_alpha=alpha, verbose=False)

    return {
        "model": model_name, "N": N, "T": T, "p": p, "t": t,
        "Sigma": Sigma, "phi": phi,
        "extra_parameters": {"sparsity": 0.2, "lasso_alpha": lasso_alpha},
        "mc_factory": mc_factory, "type1_fn": type1_fn, "power_fn": power_fn,
    }


def _make_svd_setup(model_name: str, N: int,
                    generator: VARDataGenerator) -> Dict[str, Any]:
    """构造 SVD 低秩模型的实验配置。"""
    T, p, t = _T, _p, _t
    Sigma = np.eye(N) * 0.5
    phi = generator.generate_lowrank_phi(N, p, rank=2, scale=_lowrank_scale(N))
    phi, ok, shrinks = ensure_stationary(phi)
    svd_rank = 2

    def mc_factory(M, B, seed_, jobs, _baseline_pvalue_method, _rank=svd_rank):
        return LowRankMonteCarloSimulation(
            M=M, B=B, seed=seed_, method="svd", rank=_rank, n_jobs=jobs)

    def type1_fn(mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_type1_error(
            _N, _T, _p, phi_mat, sigma, t=_t, test_alpha=alpha, verbose=False)

    def power_fn(mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t):
        return mc.evaluate_power(
            _N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, test_alpha=alpha, verbose=False)

    return {
        "model": model_name, "N": N, "T": T, "p": p, "t": t,
        "Sigma": Sigma, "phi": phi,
        "extra_parameters": {"rank": svd_rank},
        "mc_factory": mc_factory, "type1_fn": type1_fn, "power_fn": power_fn,
    }


def get_model_setup(model_name: str, seed: int) -> Dict[str, Any]:
    """根据模型名称解析方法类型和 N 维度，返回实验配置。"""
    generator = VARDataGenerator(seed=seed)

    # 注意：baseline_ols_f_n* 必须在 baseline_ols_n* 之前匹配
    if model_name.startswith("baseline_ols_f_n"):
        N = int(model_name.split("_n")[-1])
        return _make_ols_setup(model_name, N, use_f_test=True, generator=generator)
    if model_name.startswith("baseline_ols_n"):
        N = int(model_name.split("_n")[-1])
        return _make_ols_setup(model_name, N, use_f_test=False, generator=generator)
    if model_name.startswith("sparse_lasso_n"):
        N = int(model_name.split("_n")[-1])
        return _make_lasso_setup(model_name, N, generator=generator)
    if model_name.startswith("lowrank_svd_n"):
        N = int(model_name.split("_n")[-1])
        return _make_svd_setup(model_name, N, generator=generator)

    raise ValueError(f"Unknown model: {model_name!r}, available: {MODEL_EXECUTION_ORDER}")


# ===========================================================================
# 执行逻辑
# ===========================================================================

def run_model_for_seed(model_name: str, cfg: ExperimentConfig, seed: int,
                       model_jobs: int, tracker: ProgressTracker | None = None) -> Dict[str, Any]:
    setup = get_model_setup(model_name, seed)
    model_start = time.time()
    if tracker is not None:
        tracker.log_model_started(seed, model_name, model_jobs)
    phi = setup["phi"]
    sigma = setup["Sigma"]

    # ---- Type I error 评估 ----
    type1_by_M = []
    if not cfg.skip_type1:
        for M in cfg.M_grid:
            stage_name = f"type1_M_{M}"
            if tracker is not None:
                tracker.start_stage(seed, model_name, stage_name, task="type1_error", M=int(M))
            mc = setup["mc_factory"](M, cfg.B, seed, model_jobs, cfg.baseline_pvalue_method)
            start = time.time()
            try:
                type1 = setup["type1_fn"](mc, phi, sigma, cfg.alpha)
            except Exception as exc:
                if tracker is not None:
                    tracker.fail_stage(seed, model_name, stage_name, str(exc), time.time() - start, task="type1_error", M=int(M))
                raise
            runtime = time.time() - start
            type1_payload = {
                "M": int(M),
                "value": float(type1["type1_error"]),
                "size_distortion": float(type1["size_distortion"]),
                "rejections": int(type1["rejections"]),
                "M_effective": int(type1["M_effective"]),
                "runtime_sec": float(runtime),
                "pvalue_summary": summarize_pvalues(type1["p_values"]),
            }
            type1_by_M.append(type1_payload)
            if tracker is not None:
                tracker.finish_stage(seed, model_name, stage_name, runtime,
                                     task="type1_error", M=int(M), metric_value=float(type1_payload["value"]))

    # ---- Power 评估 ----
    power_curve: List[Dict[str, Any]] = []
    power_M = cfg.power_M
    mc = setup["mc_factory"](power_M, cfg.B, seed, model_jobs, cfg.baseline_pvalue_method)
    for base_delta in cfg.deltas:
        stage_name = f"power_delta_{base_delta:.2f}"
        target_fro = scale_base_delta_to_target_fro(phi, base_delta)
        if tracker is not None:
            tracker.start_stage(seed, model_name, stage_name, task="power",
                                M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro))
        phi2, ok, shrinks, actual_fro = build_phi2_with_target_frobenius(phi, target_fro)
        if not ok:
            skipped_payload = {
                "M": int(power_M), "delta": float(base_delta),
                "target_fro": float(target_fro), "actual_fro": float("nan"),
                "power": math.nan, "M_effective": 0, "rejections": 0,
                "runtime_sec": 0.0, "stationarity_shrinks": int(shrinks),
                "skipped": True, "reason": "nonstationary_after_fro_shrink",
            }
            power_curve.append(skipped_payload)
            if tracker is not None:
                tracker.finish_stage(seed, model_name, stage_name, 0.0,
                                     task="power", M=int(power_M), base_delta=float(base_delta),
                                     target_fro=float(target_fro), skipped=True, reason="nonstationary_after_fro_shrink")
            continue

        start = time.time()
        try:
            power = setup["power_fn"](mc, phi, phi2, sigma, cfg.alpha)
        except Exception as exc:
            if tracker is not None:
                tracker.fail_stage(seed, model_name, stage_name, str(exc), time.time() - start,
                                   task="power", M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro))
            raise
        runtime = time.time() - start
        power_payload = {
            "M": int(power_M), "delta": float(base_delta),
            "target_fro": float(target_fro), "actual_fro": float(actual_fro),
            "power": float(power["power"]),
            "M_effective": int(power["M_effective"]),
            "rejections": int(power["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(power["p_values"]),
        }
        power_curve.append(power_payload)
        if tracker is not None:
            tracker.finish_stage(seed, model_name, stage_name, runtime,
                                 task="power", M=int(power_M), base_delta=float(base_delta),
                                 target_fro=float(target_fro), metric_value=float(power_payload["power"]))

    model_runtime = time.time() - model_start
    if tracker is not None:
        tracker.log_model_finished(seed, model_name, model_runtime)

    return {
        "model": model_name,
        "parameters": {
            "N": setup["N"], "T": setup["T"], "p": setup["p"], "t": setup["t"],
            "M_grid": [int(m) for m in cfg.M_grid], "power_M": int(power_M),
            "B": int(cfg.B), "alpha": float(cfg.alpha),
            "baseline_pvalue_method": cfg.baseline_pvalue_method,
            **setup["extra_parameters"],
        },
        "phi": phi.tolist(),
        "type1_error_by_M": type1_by_M,
        "power_curve": power_curve,
    }


def run_all_models_for_seed(cfg: ExperimentConfig, seed: int, seed_jobs: int,
                            tracker: ProgressTracker | None = None) -> Dict[str, Dict[str, Any]]:
    """顺序执行全部 12 个模型，每个模型独占全部 job 预算。"""
    results: Dict[str, Dict[str, Any]] = {}
    for model_name in MODEL_EXECUTION_ORDER:
        results[model_name] = run_model_for_seed(model_name, cfg, seed, seed_jobs, tracker)
    return results


def write_seed_result(run_dir: str, seed: int, seed_result: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "seed_results", f"seed_{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seed_result, f, ensure_ascii=False, indent=2)


def run_seed(seed: int, cfg: ExperimentConfig, seed_jobs: int,
             run_dir: str, tracker: ProgressTracker | None = None) -> Dict[str, Any]:
    print(f"[seed {seed}] start (jobs={seed_jobs}, power_M={cfg.power_M}, M_grid={list(cfg.M_grid)})")
    if tracker is not None:
        tracker.log_seed_started(seed, seed_jobs)
    start = time.time()
    try:
        models = run_all_models_for_seed(cfg, seed, seed_jobs, tracker)
    except Exception as exc:
        if tracker is not None:
            tracker.log_seed_failed(seed, str(exc), time.time() - start)
        raise
    runtime = time.time() - start
    seed_result = {
        "seed": int(seed),
        "jobs": int(seed_jobs),
        "runtime_sec": float(runtime),
        "models": models,
    }
    write_seed_result(run_dir, seed, seed_result)
    if tracker is not None:
        tracker.log_seed_finished(seed, runtime)
    print(f"[seed {seed}] done in {runtime:.2f}s")
    return seed_result


def run_all_seeds(cfg: ExperimentConfig, run_dir: str,
                  tracker: ProgressTracker | None = None) -> Dict[str, Dict[str, Any]]:
    seed_count = len(cfg.seeds)
    seed_workers = cfg.seed_workers if cfg.seed_workers > 0 else min(seed_count, max(1, cfg.jobs))
    seed_workers = max(1, min(seed_count, seed_workers, max(1, cfg.jobs)))
    slot_budgets = allocate_job_budgets(cfg.jobs, seed_workers)

    print(f"Running seeds with total_jobs={cfg.jobs}; seed_workers={seed_workers}; seed_job_split={slot_budgets}")

    seeds_iter = iter(cfg.seeds)
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=seed_workers) as executor:
        future_to_slot: Dict[Any, int] = {}
        future_to_seed: Dict[Any, int] = {}

        for slot_idx, budget in enumerate(slot_budgets):
            try:
                seed = next(seeds_iter)
            except StopIteration:
                break
            future = executor.submit(run_seed, seed, cfg, budget, run_dir, tracker)
            future_to_slot[future] = slot_idx
            future_to_seed[future] = seed

        while future_to_slot:
            future = next(as_completed(list(future_to_slot.keys())))
            slot_idx = future_to_slot.pop(future)
            seed = future_to_seed.pop(future)
            results[str(seed)] = future.result()

            try:
                next_seed = next(seeds_iter)
            except StopIteration:
                continue
            next_future = executor.submit(run_seed, next_seed, cfg, slot_budgets[slot_idx], run_dir, tracker)
            future_to_slot[next_future] = slot_idx
            future_to_seed[next_future] = next_seed

    return {str(seed): results[str(seed)] for seed in cfg.seeds}


# ===========================================================================
# 结果聚合
# ===========================================================================

def aggregate_results(seed_results: Dict[str, Dict[str, Any]], cfg: ExperimentConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for model_name in MODEL_EXECUTION_ORDER:
        type1_by_M = []
        if not cfg.skip_type1:
            for M in cfg.M_grid:
                rows = [
                    next(item for item in seed_results[str(seed)]["models"][model_name]["type1_error_by_M"] if item["M"] == M)
                    for seed in cfg.seeds
                ]
                values = [row["value"] for row in rows]
                distortions = [row["size_distortion"] for row in rows]
                rejections = [row["rejections"] for row in rows]
                effective = [row["M_effective"] for row in rows]
                runtime = [row["runtime_sec"] for row in rows]
                stats = _aggregate_numeric(values)
                dist_stats = _aggregate_numeric(distortions)
                rej_stats = _aggregate_numeric(rejections)
                eff_stats = _aggregate_numeric(effective)
                run_stats = _aggregate_numeric(runtime)
                pooled_band = _compute_pooled_mc_band(rejections, effective)
                type1_by_M.append({
                    "M": int(M),
                    "value_mean": stats["mean"], "value_std": stats["std"],
                    "value_min": stats["min"], "value_max": stats["max"],
                    "size_distortion_mean": dist_stats["mean"],
                    "rejections_mean": rej_stats["mean"],
                    "M_effective_mean": eff_stats["mean"],
                    "runtime_sec_mean": run_stats["mean"],
                    "seed_count": stats["count"],
                    **pooled_band,
                })

        power_curve = []
        for delta in cfg.deltas:
            rows = [
                next(item for item in seed_results[str(seed)]["models"][model_name]["power_curve"]
                     if abs(item["delta"] - delta) < 1e-12)
                for seed in cfg.seeds
            ]
            values = [row["power"] for row in rows]
            actual_fro = [row.get("actual_fro", math.nan) for row in rows]
            rejections = [row.get("rejections", math.nan) for row in rows]
            effective = [row.get("M_effective", math.nan) for row in rows]
            runtime = [row.get("runtime_sec", math.nan) for row in rows]
            skipped = sum(1 for row in rows if row.get("skipped", False))
            stats = _aggregate_numeric(values)
            fro_stats = _aggregate_numeric(actual_fro)
            rej_stats = _aggregate_numeric(rejections)
            eff_stats = _aggregate_numeric(effective)
            run_stats = _aggregate_numeric(runtime)
            target_fro_values = [row.get("target_fro", math.nan) for row in rows]
            target_stats = _aggregate_numeric(target_fro_values)
            power_curve.append({
                "M": int(cfg.power_M), "delta": float(delta),
                "target_fro_mean": float(target_stats["mean"]),
                "actual_fro_mean": fro_stats["mean"],
                "power_mean": stats["mean"], "power_std": stats["std"],
                "power_min": stats["min"], "power_max": stats["max"],
                "rejections_mean": rej_stats["mean"],
                "M_effective_mean": eff_stats["mean"],
                "runtime_sec_mean": run_stats["mean"],
                "seed_count": stats["count"], "skipped_count": int(skipped),
            })

        parameters = seed_results[str(cfg.seeds[0])]["models"][model_name]["parameters"]
        models[model_name] = {
            "parameters": parameters,
            "type1_error_by_M": type1_by_M,
            "power_curve": power_curve,
        }
    return {"models": models}


# ===========================================================================
# CSV / Markdown 输出
# ===========================================================================

def build_raw_rows(seed_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed, seed_block in seed_results.items():
        for model_name in MODEL_EXECUTION_ORDER:
            block = seed_block["models"][model_name]
            for row in block["type1_error_by_M"]:
                rows.append({
                    "scope": "seed", "seed": int(seed), "model": model_name,
                    "metric": "type1_error", "M": int(row["M"]),
                    "delta": "", "target_fro": "", "actual_fro": "",
                    "value": float(row["value"]),
                    "size_distortion": float(row["size_distortion"]),
                    "rejections": int(row["rejections"]),
                    "effective_iterations": int(row["M_effective"]),
                    "runtime_sec": float(row["runtime_sec"]),
                })
            for row in block["power_curve"]:
                rows.append({
                    "scope": "seed", "seed": int(seed), "model": model_name,
                    "metric": "power", "M": int(row["M"]),
                    "delta": float(row["delta"]),
                    "target_fro": float(row["target_fro"]),
                    "actual_fro": float(row.get("actual_fro", math.nan)),
                    "value": float(row["power"]) if not math.isnan(row["power"]) else math.nan,
                    "size_distortion": "",
                    "rejections": int(row.get("rejections", 0)),
                    "effective_iterations": int(row.get("M_effective", 0)),
                    "runtime_sec": float(row.get("runtime_sec", math.nan)),
                })
    return rows


def build_aggregate_rows(aggregate_results_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in MODEL_EXECUTION_ORDER:
        block = aggregate_results_block["models"][model_name]
        for row in block["type1_error_by_M"]:
            rows.append({
                "scope": "aggregate", "seed": "", "model": model_name,
                "metric": "type1_error", "M": int(row["M"]),
                "delta": "", "target_fro": "", "actual_fro": "",
                "value_mean": float(row["value_mean"]),
                "value_std": float(row["value_std"]),
                "size_distortion_mean": float(row["size_distortion_mean"]),
                "rejections_mean": float(row["rejections_mean"]),
                "effective_iterations_mean": float(row["M_effective_mean"]),
                "rejections_total": int(row["rejections_total"]),
                "effective_iterations_total": int(row["M_effective_total"]),
                "mc_se_pooled": float(row["mc_se_pooled"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "runtime_sec_mean": float(row["runtime_sec_mean"]),
                "seed_count": int(row["seed_count"]),
            })
        for row in block["power_curve"]:
            rows.append({
                "scope": "aggregate", "seed": "", "model": model_name,
                "metric": "power", "M": int(row["M"]),
                "delta": float(row["delta"]),
                "target_fro": float(row["target_fro_mean"]),
                "target_fro_mean": float(row["target_fro_mean"]),
                "actual_fro_mean": float(row["actual_fro_mean"]),
                "value_mean": float(row["power_mean"]),
                "value_std": float(row["power_std"]),
                "size_distortion_mean": "",
                "rejections_mean": float(row["rejections_mean"]),
                "effective_iterations_mean": float(row["M_effective_mean"]),
                "runtime_sec_mean": float(row["runtime_sec_mean"]),
                "seed_count": int(row["seed_count"]),
            })
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(results: Dict[str, Any], report_path: str) -> None:
    info = results["experiment_info"]
    cfg = info["config"]
    agg = results["aggregates"]["models"]

    lines: List[str] = []
    lines.append("# 高维 VAR 结构断裂检验仿真报告（4方法 × 3维度）")
    lines.append("")
    lines.append(f"- 生成时间: {info['timestamp']}")
    lines.append(f"- seeds: {cfg['seeds']}")
    lines.append(f"- M_grid: {cfg['M_grid']}")
    lines.append(f"- power 使用的 M: {cfg['power_M']}")
    lines.append(f"- B: {cfg['B']}; alpha: {cfg['alpha']}; base deltas: {cfg['deltas']}")
    lines.append(f"- T: {_T}; p: {_p}; 已知断点 t: {_t}")
    lines.append(f"- 维度 N: 5, 10, 20")
    lines.append("- 断点构造: delta 直接作为 Frobenius 范数目标 `||ΔΦ||_F = delta`")
    lines.append(f"- 总耗时(秒): {info['total_runtime_sec']:.2f}")
    lines.append("")

    # OLS 可行性表
    lines.append("## 0. OLS 可行性（每段有效观测 vs 参数量）")
    lines.append("")
    lines.append("| N | 参数量 (N²+N) | seg 有效观测 (~499) | 参数/观测比 | 可行? |")
    lines.append("|---:|---:|---:|---:|---|")
    for N in (5, 10, 20):
        params = N * N * _p + N
        seg_obs = _t - _p  # 499
        ratio = params / seg_obs
        feasible = "✓ 充裕" if ratio < 0.5 else ("✓ 可行" if ratio < 1.0 else "✗ 欠定")
        lines.append(f"| {N} | {params} | {seg_obs} | {ratio:.2f} | {feasible} |")
    lines.append("")

    if agg[MODEL_EXECUTION_ORDER[0]]["type1_error_by_M"]:
        lines.append("## 1. Size（Type I Error，跨 seed 聚合）")
        lines.append("")
        lines.append("| 模型 | N | 方法 | M | Type I Error Mean | MC SE | 95% CI | Std | Size Distortion | Seeds |")
        lines.append("|---|---:|---|---:|---:|---:|---|---:|---:|---:|")
        for model_name in MODEL_EXECUTION_ORDER:
            params = agg[model_name]["parameters"]
            N = params["N"]
            if "lasso" in model_name:
                method = "Lasso"
            elif "svd" in model_name:
                method = "SVD"
            elif "_f_" in model_name:
                method = "OLS(F)"
            else:
                method = "OLS(LR)"
            for row in agg[model_name]["type1_error_by_M"]:
                lines.append(
                    f"| {model_name} | {N} | {method} | {row['M']} | "
                    f"{row['value_mean']:.4f} | {row['mc_se_pooled']:.4f} | "
                    f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['value_std']:.4f} | "
                    f"{row['size_distortion_mean']:+.4f} | {row['seed_count']} |"
                )
        lines.append("")

    lines.append("## 2. Power（跨 seed 聚合）")
    lines.append("")
    lines.append("| 模型 | N | 方法 | M | δ | Target Fro | Power Mean | Std | Seeds |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for model_name in MODEL_EXECUTION_ORDER:
        params = agg[model_name]["parameters"]
        N = params["N"]
        if "lasso" in model_name:
            method = "Lasso"
        elif "svd" in model_name:
            method = "SVD"
        elif "_f_" in model_name:
            method = "OLS(F)"
        else:
            method = "OLS(LR)"
        for row in agg[model_name]["power_curve"]:
            lines.append(
                f"| {model_name} | {N} | {method} | {row['M']} | "
                f"{row['delta']:.2f} | {row['target_fro_mean']:.2f} | "
                f"{row['power_mean']:.4f} | {row['power_std']:.4f} | {row['seed_count']} |"
            )
    lines.append("")

    lines.append("## 3. 结论摘要")
    lines.append("")
    for model_name in MODEL_EXECUTION_ORDER:
        power_points = agg[model_name]["power_curve"]
        power_values = [row["power_mean"] for row in power_points if not math.isnan(row["power_mean"])]
        monotone = len(power_values) >= 2 and all(power_values[i] <= power_values[i + 1] for i in range(len(power_values) - 1))
        final_point = power_points[-1]
        type1_rows = agg[model_name]["type1_error_by_M"]
        size_str = ""
        if type1_rows:
            size_row = next((row for row in type1_rows if row["M"] == cfg["power_M"]), type1_rows[-1])
            size_str = f"size_at_Mmax={size_row['value_mean']:.4f}; "
        lines.append(
            f"- {model_name}: {size_str}"
            f"power_at_max_delta(δ={final_point['delta']:.2f})={final_point['power_mean']:.4f}; "
            f"power_monotone={monotone}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ===========================================================================
# 入口
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="高维 VAR 结构断裂检验仿真（4方法 × 3维度 = 12模型）")
    parser.add_argument("--M-grid", type=int, nargs="+", default=[50, 100, 300, 500, 1000, 2000])
    parser.add_argument("--power-M", type=int, default=500, help="Power 评估使用的 M（默认 500）")
    parser.add_argument("--B", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 7])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed-workers", type=int, default=0)
    parser.add_argument("--baseline-pvalue-method", type=str, default="bootstrap_lr",
                        choices=["bootstrap_lr", "asymptotic_chi2", "asymptotic_f"])
    parser.add_argument("--skip-type1", action="store_true", help="跳过 Type I error 评估，仅跑 Power")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="指定要运行的模型子集（默认全部 12 个）")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    # 如果指定了模型子集，验证并覆盖
    if args.models is not None:
        global MODEL_EXECUTION_ORDER
        for m in args.models:
            if m not in MODEL_EXECUTION_ORDER:
                parser.error(f"Unknown model: {m!r}. Available: {MODEL_EXECUTION_ORDER}")
        MODEL_EXECUTION_ORDER = tuple(args.models)

    cfg = ExperimentConfig(
        M_grid=tuple(sorted(set(int(m) for m in args.M_grid))),
        B=int(args.B),
        alpha=float(args.alpha),
        deltas=tuple(float(d) for d in args.deltas),
        seeds=tuple(int(seed) for seed in args.seeds),
        jobs=max(1, int(args.jobs)),
        seed_workers=max(0, int(args.seed_workers)),
        baseline_pvalue_method=args.baseline_pvalue_method,
        _power_M=max(0, int(args.power_M)),
        skip_type1=args.skip_type1,
    )

    stamp, run_dir = make_run_dir(args.tag)
    tag_suffix = f"_{args.tag}" if args.tag else ""

    print("=" * 72)
    print("高维 VAR 结构断裂检验仿真实验")
    print(f"  模型数: {len(MODEL_EXECUTION_ORDER)}")
    print(f"  T={_T}, p={_p}, t={_t}, N∈{{5,10,20}}")
    print(f"  M_grid={list(cfg.M_grid)}, power_M={cfg.power_M}, B={cfg.B}")
    print(f"  alpha={cfg.alpha}, deltas={list(cfg.deltas)}")
    print(f"  seeds={list(cfg.seeds)}, jobs={cfg.jobs}, seed_workers={cfg.seed_workers}")
    print(f"  baseline_pvalue_method={cfg.baseline_pvalue_method}")
    print(f"  models: {list(MODEL_EXECUTION_ORDER)}")
    print("=" * 72)

    per_seed_stage_count = len(MODEL_EXECUTION_ORDER) * (
        (len(cfg.M_grid) if not cfg.skip_type1 else 0) + len(cfg.deltas)
    )
    total_stage_count = len(cfg.seeds) * per_seed_stage_count
    tracker = ProgressTracker(run_dir=run_dir, total_stage_count=total_stage_count,
                              per_seed_stage_count=per_seed_stage_count)
    tracker.log_run_started(cfg)

    global _ACTIVE_TRACKER, _RUN_START_TIME
    _ACTIVE_TRACKER = tracker
    all_start = time.time()
    _RUN_START_TIME = all_start
    _install_signal_handlers(tracker)

    try:
        seed_results = run_all_seeds(cfg, run_dir, tracker)
    except BaseException as exc:
        total_runtime = time.time() - all_start
        tracker.log_run_failed(total_runtime, f"{type(exc).__name__}: {exc}")
        raise
    total_runtime = time.time() - all_start
    tracker.log_run_finished(total_runtime)
    aggregates = aggregate_results(seed_results, cfg)

    results = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "高维 VAR 结构断裂检验仿真（4方法×3维度=12模型，T=1000）",
            "config": {
                **asdict(cfg),
                "power_M": cfg.power_M,
                "effect_size_mode": "direct_frobenius",
                "effect_direction": "normalized_all_ones",
                "delta_interpretation": "frobenius_norm_of_break",
                "T": _T, "p": _p, "t": _t,
            },
            "total_runtime_sec": float(total_runtime),
        },
        "seed_results": seed_results,
        "aggregates": aggregates,
    }

    json_path = os.path.join(run_dir, f"highdim_experiment_{stamp}{tag_suffix}.json")
    raw_csv_path = os.path.join(run_dir, f"highdim_raw_{stamp}{tag_suffix}.csv")
    agg_csv_path = os.path.join(run_dir, f"highdim_agg_{stamp}{tag_suffix}.csv")
    md_path = os.path.join(run_dir, f"高维仿真报告_{stamp}{tag_suffix}.md")
    run_meta_path = os.path.join(run_dir, "run_meta.json")

    results["experiment_info"]["output_paths"] = {
        "run_dir": run_dir,
        "json": json_path,
        "raw_csv": raw_csv_path,
        "agg_csv": agg_csv_path,
        "markdown": md_path,
        "run_meta": run_meta_path,
        "seed_results_dir": os.path.join(run_dir, "seed_results"),
        "progress_dir": tracker.progress_dir,
        "progress_log": tracker.progress_log_path,
        "progress_jsonl": tracker.progress_jsonl_path,
        "progress_summary": tracker.summary_path,
        "seed_summary_pattern": os.path.join(tracker.progress_dir, "seed_<seed>_summary.json"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_csv(raw_csv_path, build_raw_rows(seed_results))
    write_csv(agg_csv_path, build_aggregate_rows(aggregates))
    write_markdown_report(results, md_path)

    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(results["experiment_info"]["output_paths"], f, ensure_ascii=False, indent=2)

    print(f"\nOutputs:")
    print(f"  RUN     : {run_dir}")
    print(f"  JSON    : {json_path}")
    print(f"  RAW CSV : {raw_csv_path}")
    print(f"  AGG CSV : {agg_csv_path}")
    print(f"  MD      : {md_path}")
    print(f"  PROGRESS: {tracker.progress_dir}")
    print(f"  META    : {run_meta_path}")
    print(f"Total runtime: {total_runtime:.2f}s")


if __name__ == "__main__":
    main()
