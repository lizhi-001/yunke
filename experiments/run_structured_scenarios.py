"""结构化 VAR 断裂检验仿真实验

叙事框架（三层）：
  第一层 — 普通多元时间序列基准（N=10，T=500，稠密 DGP）
    baseline_ols_f: OLS + 渐近 F 检验   → 常规断裂检验（理论锚点）
    baseline_ols:   OLS + LR+Bootstrap  → 提出的 LR+Bootstrap 方法

  第二层 — 高维稀疏多元时间序列（N=20，T=500，稀疏 DGP，稀疏度 0.15）
    sparse_lasso:   Lasso + LR+Bootstrap → 稀疏背景下使用 LR+Bootstrap 进行断裂检验

  第三层 — 高维低秩多元时间序列（N=20，T=500，低秩 DGP，rank=2）
    lowrank_rrr:    RRR   + LR+Bootstrap → 低秩背景下使用 LR+Bootstrap 进行断裂检验

核心叙事逻辑：
  在稀疏和低秩背景下，可以用提出的 LR+Bootstrap 方法进行断裂检验，
  不要求性能高于 OLS。

预期结果：
  - 第一层：F 检验与 LR+Bootstrap 的 size 和 power 表现一致
  - 第二/三层：LR+Bootstrap 的 size 随 M 增大逐渐稳定在 0.05
  - 第二/三层：M 固定时，power 随 δ 增大越来越大
  - 所有模型 power 随 δ 单调递增（检验有功效）
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

from simulation import MonteCarloSimulation, VARDataGenerator   # noqa: E402
from sparse_var import SparseMonteCarloSimulation               # noqa: E402
from lowrank_var import LowRankMonteCarloSimulation             # noqa: E402


# ---------------------------------------------------------------------------
# 固定实验参数
# ---------------------------------------------------------------------------
_T: int = 500
_p: int = 1
_t: int = 250

_N_BASELINE: int = 10   # 第一层：OLS 基准
_N_SPARSE:   int = 20   # 第二层：稀疏场景
_N_LOWRANK:  int = 20   # 第三层：低秩场景

_SPARSE_SPARSITY: float = 0.15   # 15% 非零元素
_LOWRANK_RANK:    int   = 2      # 真实 rank
_LASSO_ALPHA:     float = 0.02   # Lasso 正则化参数
_RRR_RANK:        int   = 2      # RRR 目标秩（匹配真实 rank）
_LOWRANK_TARGET_SR: float = 0.40 # 低秩 DGP 目标谱半径（直接控制，消除 seed 间方差）

# 系数矩阵 scale（随 N 自适应）
def _dense_scale(N: int) -> float: return min(0.3, 0.85 / N ** 0.5)

# 模型执行顺序
MODEL_EXECUTION_ORDER: Tuple[str, ...] = (
    "baseline_ols_f",  # 第一层：F 检验
    "baseline_ols",    # 第一层：LR+Bootstrap
    "sparse_lasso",    # 第二层：稀疏 LR+Bootstrap
    "lowrank_rrr",     # 第三层：低秩 LR+Bootstrap
)

# 作业优先级（慢模型优先分配资源）
MODEL_JOB_PRIORITY: Tuple[str, ...] = (
    "sparse_lasso", "lowrank_rrr",
    "baseline_ols", "baseline_ols_f",
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
    M_grid:   Tuple[int, ...]   = (50, 100, 300, 500, 1000, 2000)
    B:        int               = 500
    alpha:    float             = 0.05
    deltas:   Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
    seeds:    Tuple[int, ...]   = (42, 2026)
    jobs:     int               = 4
    seed_workers: int           = 1   # 默认顺序跑seed，单seed独占全部jobs
    _power_M: int               = 0
    skip_type1: bool            = False
    skip_power: bool            = False

    @property
    def power_M(self) -> int:
        return self._power_M if self._power_M > 0 else max(self.M_grid)


# ===========================================================================
# 进度追踪
# ===========================================================================

@dataclass
class ProgressTracker:
    run_dir:             str
    total_stage_count:   int
    per_seed_stage_count: int
    progress_dir:        str = field(init=False)
    progress_log_path:   str = field(init=False)
    progress_jsonl_path: str = field(init=False)
    summary_path:        str = field(init=False)
    completed_stage_count: int = field(default=0, init=False)
    active_stages:  Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    completed_stages: List[str]               = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.progress_dir       = os.path.join(self.run_dir, "progress")
        os.makedirs(self.progress_dir, exist_ok=True)
        self.progress_log_path  = os.path.join(self.progress_dir, "progress.log")
        self.progress_jsonl_path = os.path.join(self.progress_dir, "progress.jsonl")
        self.summary_path       = os.path.join(self.progress_dir, "summary.json")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _stage_id(self, seed: int, model: str, stage: str) -> str:
        return f"seed_{seed}/{model}/{stage}"

    def _write_summary(self) -> None:
        payload = {
            "updated_at": self._timestamp(),
            "completed_stage_count": self.completed_stage_count,
            "total_stage_count": self.total_stage_count,
            "progress_ratio": (
                0.0 if self.total_stage_count <= 0
                else self.completed_stage_count / self.total_stage_count
            ),
            "active_stages": sorted(self.active_stages.keys()),
            "completed_stages": list(self.completed_stages[-20:]),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _log(self, msg: str, payload: Dict[str, Any] | None = None) -> None:
        ts = self._timestamp()
        with self._lock:
            with open(self.progress_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
            if payload is not None:
                entry = {"timestamp": ts, **payload}
                with open(self.progress_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_run_started(self, cfg: ExperimentConfig) -> None:
        self._log(
            f"run started | seeds={list(cfg.seeds)} B={cfg.B} M_grid={list(cfg.M_grid)}",
            {"event": "run_started", "config": {
                "seeds": list(cfg.seeds), "B": cfg.B,
                "M_grid": list(cfg.M_grid), "power_M": cfg.power_M,
                "alpha": cfg.alpha, "deltas": list(cfg.deltas),
            }},
        )

    def log_run_finished(self, runtime: float) -> None:
        with self._lock:
            self._write_summary()
        self._log(f"run finished | runtime={runtime:.2f}s", {"event": "run_finished", "runtime_sec": runtime})

    def log_run_failed(self, runtime: float, reason: str) -> None:
        self._log(f"run FAILED | {reason}", {"event": "run_failed", "runtime_sec": runtime, "reason": reason})

    def log_seed_started(self, seed: int, jobs: int) -> None:
        self._log(f"seed {seed} started | jobs={jobs}", {"event": "seed_started", "seed": seed, "jobs": jobs})

    def log_seed_finished(self, seed: int, runtime: float) -> None:
        self._log(f"seed {seed} done | {runtime:.2f}s", {"event": "seed_finished", "seed": seed, "runtime_sec": runtime})

    def log_seed_failed(self, seed: int, error: str, runtime: float) -> None:
        self._log(f"seed {seed} FAILED | {error}", {"event": "seed_failed", "seed": seed, "error": error, "runtime_sec": runtime})

    def log_model_started(self, seed: int, model: str, jobs: int) -> None:
        self._log(f"  [{seed}] {model} started", {"event": "model_started", "seed": seed, "model": model, "jobs": jobs})

    def log_model_finished(self, seed: int, model: str, runtime: float) -> None:
        self._log(f"  [{seed}] {model} done | {runtime:.2f}s", {"event": "model_finished", "seed": seed, "model": model, "runtime_sec": runtime})

    def start_stage(self, seed: int, model: str, stage: str, **extra) -> None:
        sid = self._stage_id(seed, model, stage)
        with self._lock:
            self.active_stages[sid] = {"started_at": self._timestamp(), **extra}
            self._write_summary()
        self._log(f"  [{seed}] {model}/{stage} start", {"event": "stage_started", "seed": seed, "model": model, "stage": stage, **extra})

    def finish_stage(self, seed: int, model: str, stage: str, runtime_sec: float, **extra) -> None:
        sid = self._stage_id(seed, model, stage)
        with self._lock:
            self.active_stages.pop(sid, None)
            self.completed_stage_count += 1
            self.completed_stages.append(sid)
            self._write_summary()
        val = extra.get("metric_value", "")
        val_str = f" val={val:.4f}" if isinstance(val, float) else ""
        self._log(
            f"  [{seed}] {model}/{stage} done | {runtime_sec:.2f}s{val_str}  "
            f"[{self.completed_stage_count}/{self.total_stage_count}]",
            {"event": "stage_finished", "seed": seed, "model": model, "stage": stage,
             "runtime_sec": runtime_sec, **extra},
        )

    def fail_stage(self, seed: int, model: str, stage: str, error: str, runtime_sec: float, **extra) -> None:
        sid = self._stage_id(seed, model, stage)
        with self._lock:
            self.active_stages.pop(sid, None)
            self._write_summary()
        self._log(f"  [{seed}] {model}/{stage} FAILED | {error}", {"event": "stage_failed", "seed": seed, "model": model, "stage": stage, "error": error, "runtime_sec": runtime_sec, **extra})


# ===========================================================================
# 扰动函数
# ===========================================================================

def _ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int]:
    current = phi.copy()
    for i in range(max_attempts):
        if VARDataGenerator.check_stationarity(current):
            return current, True, i
        current = current * shrink
    return current, VARDataGenerator.check_stationarity(current), max_attempts


def build_phi2_uniform(phi: np.ndarray, target_fro: float, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int, float]:
    """均匀全 1 扰动（基准层使用）。"""
    direction = np.ones_like(phi, dtype=float)
    direction /= float(np.linalg.norm(direction))
    scale = float(target_fro)
    candidate = phi + scale * direction
    attempts = 0
    while not VARDataGenerator.check_stationarity(candidate) and attempts < max_attempts:
        scale *= shrink
        candidate = phi + scale * direction
        attempts += 1
    actual_fro = float(np.linalg.norm(candidate - phi))
    return candidate, VARDataGenerator.check_stationarity(candidate), attempts, actual_fro


def build_phi2_sparse(phi: np.ndarray, target_fro: float, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int, float]:
    """稀疏扰动：仅在 φ 的非零支撑集上施加扰动，匹配稀疏场景。"""
    support = (np.abs(phi) > 1e-10).astype(float)
    support_norm = float(np.linalg.norm(support))
    if support_norm <= 0:
        # 退化：全零矩阵，改用均匀方向
        return build_phi2_uniform(phi, target_fro, shrink, max_attempts)
    direction = support / support_norm
    scale = float(target_fro)
    candidate = phi + scale * direction
    attempts = 0
    while not VARDataGenerator.check_stationarity(candidate) and attempts < max_attempts:
        scale *= shrink
        candidate = phi + scale * direction
        attempts += 1
    actual_fro = float(np.linalg.norm(candidate - phi))
    return candidate, VARDataGenerator.check_stationarity(candidate), attempts, actual_fro


def build_phi2_lowrank(phi: np.ndarray, target_fro: float, rank: int = 2, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int, float]:
    """低秩扰动：在 φ 的前 rank 个奇异向量张成的子空间内施加扰动，匹配低秩场景。"""
    U, _, Vt = np.linalg.svd(phi, full_matrices=False)
    r = min(rank, U.shape[1], Vt.shape[0])
    Ur, Vr = U[:, :r], Vt[:r, :]
    # 扰动方向：U_r @ 1_{r×r} @ V_r，然后归一化
    direction = Ur @ np.ones((r, r)) @ Vr
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0:
        return build_phi2_uniform(phi, target_fro, shrink, max_attempts)
    direction /= direction_norm
    scale = float(target_fro)
    candidate = phi + scale * direction
    attempts = 0
    while not VARDataGenerator.check_stationarity(candidate) and attempts < max_attempts:
        scale *= shrink
        candidate = phi + scale * direction
        attempts += 1
    actual_fro = float(np.linalg.norm(candidate - phi))
    return candidate, VARDataGenerator.check_stationarity(candidate), attempts, actual_fro


def scale_base_delta_to_target_fro(phi: np.ndarray, base_delta: float) -> float:
    return float(base_delta)


def summarize_pvalues(pvalues: np.ndarray) -> Dict[str, float]:
    if pvalues is None or len(pvalues) == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "q25": math.nan, "q50": math.nan, "q75": math.nan}
    return {
        "count": int(len(pvalues)),
        "mean":  float(np.mean(pvalues)),
        "std":   float(np.std(pvalues)),
        "q25":   float(np.quantile(pvalues, 0.25)),
        "q50":   float(np.quantile(pvalues, 0.50)),
        "q75":   float(np.quantile(pvalues, 0.75)),
    }


# ===========================================================================
# MC 工厂
# ===========================================================================

def _ols_bootstrap_mc(M, B, seed, jobs, _ignored):
    return MonteCarloSimulation(M=M, B=B, seed=seed, n_jobs=jobs, baseline_pvalue_method="bootstrap_lr")

def _ols_f_mc(M, B, seed, jobs, _ignored):
    return MonteCarloSimulation(M=M, B=B, seed=seed, n_jobs=jobs, baseline_pvalue_method="asymptotic_f")

def _lasso_mc(M, B, seed, jobs, _ignored):
    return SparseMonteCarloSimulation(M=M, B=B, seed=seed, estimator_type="lasso", alpha=_LASSO_ALPHA,
                                      post_lasso_ols=False, n_jobs=jobs)

def _rrr_mc(M, B, seed, jobs, _ignored):
    return LowRankMonteCarloSimulation(M=M, B=B, seed=seed, method="rrr", rank=_RRR_RANK, n_jobs=jobs)


# ===========================================================================
# 模型配置
# ===========================================================================

def get_model_setup(model_name: str, seed: int) -> Dict[str, Any]:
    """返回模型运行所需的全部配置。

    场景内两个模型（如 sparse_lasso / sparse_ols_f）共享相同的 φ：
    两者均用同一 seed 初始化 VARDataGenerator，并以相同参数调用生成函数，
    因此得到相同的 φ 实现。
    """
    # ------------------------------------------------------------------
    # 第一层：OLS 基准（N=10，稠密 DGP）
    # ------------------------------------------------------------------
    if model_name == "baseline_ols":
        N, T, p, t = _N_BASELINE, _T, _p, _t
        gen = VARDataGenerator(seed=seed)
        phi = gen.generate_stationary_phi(N, p, scale=_dense_scale(N))
        Sigma = np.eye(N) * 0.5
        return {
            "model": model_name, "N": N, "T": T, "p": p, "t": t,
            "Sigma": Sigma, "phi": phi,
            "extra_parameters": {"pvalue_method": "bootstrap_lr"},
            "mc_factory": _ols_bootstrap_mc,
            "type1_fn": lambda mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_type1_error_at_point(_N, _T, _p, phi_mat, sigma, t=_t, alpha=alpha, verbose=False),
            "power_fn": lambda mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_power_at_point(_N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, alpha=alpha, verbose=False),
        }

    if model_name == "baseline_ols_f":
        N, T, p, t = _N_BASELINE, _T, _p, _t
        gen = VARDataGenerator(seed=seed)          # 与 baseline_ols 相同 seed + 相同调用 → 相同 φ
        phi = gen.generate_stationary_phi(N, p, scale=_dense_scale(N))
        Sigma = np.eye(N) * 0.5
        return {
            "model": model_name, "N": N, "T": T, "p": p, "t": t,
            "Sigma": Sigma, "phi": phi,
            "extra_parameters": {"pvalue_method": "asymptotic_f"},
            "mc_factory": _ols_f_mc,
            "type1_fn": lambda mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_type1_error_at_point(_N, _T, _p, phi_mat, sigma, t=_t, alpha=alpha, verbose=False),
            "power_fn": lambda mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_power_at_point(_N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, alpha=alpha, verbose=False),
        }

    # ------------------------------------------------------------------
    # 第二层：稀疏场景（N=20，稀疏 DGP，sparsity=0.15）
    # ------------------------------------------------------------------
    if model_name == "sparse_lasso":
        N, T, p, t = _N_SPARSE, _T, _p, _t
        gen = VARDataGenerator(seed=seed)
        phi = gen.generate_stationary_phi(N, p, sparsity=_SPARSE_SPARSITY, scale=_dense_scale(N))
        Sigma = np.eye(N) * 0.5
        return {
            "model": model_name, "N": N, "T": T, "p": p, "t": t,
            "Sigma": Sigma, "phi": phi,
            "extra_parameters": {"pvalue_method": "bootstrap_lr", "sparsity": _SPARSE_SPARSITY, "lasso_alpha": _LASSO_ALPHA},
            "mc_factory": _lasso_mc,
            "type1_fn": lambda mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_type1_error(_N, _T, _p, phi_mat, sigma, t=_t, test_alpha=alpha, verbose=False),
            "power_fn": lambda mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_power(_N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, test_alpha=alpha, verbose=False),
        }

    # ------------------------------------------------------------------
    # 第三层：低秩场景（N=20，低秩 DGP，rank=2）
    # ------------------------------------------------------------------
    if model_name == "lowrank_rrr":
        N, T, p, t = _N_LOWRANK, _T, _p, _t
        gen = VARDataGenerator(seed=seed)
        phi = gen.generate_lowrank_phi(N, p, rank=_LOWRANK_RANK, scale=0.3,
                                       target_spectral_radius=_LOWRANK_TARGET_SR)
        Sigma = np.eye(N) * 0.5
        return {
            "model": model_name, "N": N, "T": T, "p": p, "t": t,
            "Sigma": Sigma, "phi": phi,
            "extra_parameters": {"pvalue_method": "bootstrap_lr", "rank": _RRR_RANK},
            "mc_factory": _rrr_mc,
            "type1_fn": lambda mc, phi_mat, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_type1_error(_N, _T, _p, phi_mat, sigma, t=_t, test_alpha=alpha, verbose=False),
            "power_fn": lambda mc, phi1, phi2, sigma, alpha, _N=N, _T=T, _p=p, _t=t:
                mc.evaluate_power(_N, _T, _p, phi1, phi2, sigma, break_point=_t, t=_t, test_alpha=alpha, verbose=False),
        }

    raise ValueError(f"未知模型：{model_name}")


def _get_perturbation_type(model_name: str) -> str:
    if "sparse" in model_name:
        return "sparse"
    if "lowrank" in model_name:
        return "lowrank"
    return "uniform"


def _build_phi2(phi: np.ndarray, target_fro: float, perturbation_type: str) -> Tuple[np.ndarray, bool, int, float]:
    if perturbation_type == "sparse":
        return build_phi2_sparse(phi, target_fro)
    if perturbation_type == "lowrank":
        return build_phi2_lowrank(phi, target_fro, rank=_RRR_RANK)
    return build_phi2_uniform(phi, target_fro)


# ===========================================================================
# 作业分配
# ===========================================================================

def allocate_job_budgets(total_jobs: int, slot_count: int) -> List[int]:
    total_jobs = max(1, int(total_jobs))
    slot_count = max(1, int(slot_count))
    budgets = [1] * slot_count
    extra = total_jobs - slot_count
    idx = 0
    while extra > 0:
        budgets[idx % slot_count] += 1
        extra -= 1
        idx += 1
    return budgets


# ===========================================================================
# 核心执行
# ===========================================================================

def run_model_for_seed(
    model_name: str,
    cfg: ExperimentConfig,
    seed: int,
    model_jobs: int,
    tracker: ProgressTracker | None = None,
) -> Dict[str, Any]:
    setup = get_model_setup(model_name, seed)
    model_start = time.time()
    if tracker is not None:
        tracker.log_model_started(seed, model_name, model_jobs)

    phi = setup["phi"]
    sigma = setup["Sigma"]
    perturbation_type = _get_perturbation_type(model_name)

    # ---- Type I error ----
    type1_by_M: List[Dict[str, Any]] = []
    if not cfg.skip_type1:
        for M in cfg.M_grid:
            stage_name = f"type1_M_{M}"
            if tracker is not None:
                tracker.start_stage(seed, model_name, stage_name, task="type1_error", M=int(M))
            mc = setup["mc_factory"](M, cfg.B, seed, model_jobs, None)
            t0 = time.time()
            try:
                type1 = setup["type1_fn"](mc, phi, sigma, cfg.alpha)
            except Exception as exc:
                if tracker is not None:
                    tracker.fail_stage(seed, model_name, stage_name, str(exc), time.time() - t0, task="type1_error", M=int(M))
                raise
            rt = time.time() - t0
            payload = {
                "M": int(M),
                "value": float(type1["type1_error"]),
                "size_distortion": float(type1["size_distortion"]),
                "rejections": int(type1["rejections"]),
                "M_effective": int(type1["M_effective"]),
                "runtime_sec": float(rt),
                "pvalue_summary": summarize_pvalues(type1["p_values"]),
            }
            type1_by_M.append(payload)
            if tracker is not None:
                tracker.finish_stage(seed, model_name, stage_name, rt, task="type1_error", M=int(M), metric_value=payload["value"])

    # ---- Power ----
    power_curve: List[Dict[str, Any]] = []
    power_M = cfg.power_M
    if not cfg.skip_power:
        mc = setup["mc_factory"](power_M, cfg.B, seed, model_jobs, None)
        for base_delta in cfg.deltas:
            stage_name = f"power_delta_{base_delta:.2f}"
            target_fro = scale_base_delta_to_target_fro(phi, base_delta)
            if tracker is not None:
                tracker.start_stage(seed, model_name, stage_name, task="power", M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro))

            phi2, ok, shrinks, actual_fro = _build_phi2(phi, target_fro, perturbation_type)
            if not ok:
                skipped = {
                    "M": int(power_M), "delta": float(base_delta),
                    "target_fro": float(target_fro), "actual_fro": float("nan"),
                    "power": math.nan, "M_effective": 0, "rejections": 0,
                    "runtime_sec": 0.0, "stationarity_shrinks": int(shrinks),
                    "skipped": True, "reason": "nonstationary_after_shrink",
                    "perturbation_type": perturbation_type,
                }
                power_curve.append(skipped)
                if tracker is not None:
                    tracker.finish_stage(seed, model_name, stage_name, 0.0, task="power", M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro), skipped=True)
                continue

            t0 = time.time()
            try:
                power = setup["power_fn"](mc, phi, phi2, sigma, cfg.alpha)
            except Exception as exc:
                if tracker is not None:
                    tracker.fail_stage(seed, model_name, stage_name, str(exc), time.time() - t0, task="power", M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro))
                raise
            rt = time.time() - t0
            payload = {
                "M": int(power_M), "delta": float(base_delta),
                "target_fro": float(target_fro), "actual_fro": float(actual_fro),
                "power": float(power["power"]),
                "M_effective": int(power["M_effective"]),
                "rejections": int(power["rejections"]),
                "runtime_sec": float(rt),
                "stationarity_shrinks": int(shrinks),
                "skipped": False,
                "perturbation_type": perturbation_type,
                "pvalue_summary": summarize_pvalues(power["p_values"]),
            }
            power_curve.append(payload)
            if tracker is not None:
                tracker.finish_stage(seed, model_name, stage_name, rt, task="power", M=int(power_M), base_delta=float(base_delta), target_fro=float(target_fro), metric_value=payload["power"])

    model_runtime = time.time() - model_start
    if tracker is not None:
        tracker.log_model_finished(seed, model_name, model_runtime)

    return {
        "model": model_name,
        "parameters": {
            "N": setup["N"], "T": setup["T"], "p": setup["p"], "t": setup["t"],
            "M_grid": [int(m) for m in cfg.M_grid],
            "power_M": int(power_M),
            "B": int(cfg.B), "alpha": float(cfg.alpha),
            **setup["extra_parameters"],
        },
        "phi": phi.tolist(),
        "type1_error_by_M": type1_by_M,
        "power_curve": power_curve,
    }


def run_all_models_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    seed_jobs: int,
    tracker: ProgressTracker | None = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for m in MODEL_EXECUTION_ORDER:
        results[m] = run_model_for_seed(m, cfg, seed, seed_jobs, tracker)
    return results


def write_seed_result(run_dir: str, seed: int, data: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "seed_results", f"seed_{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_seed(
    seed: int,
    cfg: ExperimentConfig,
    seed_jobs: int,
    run_dir: str,
    tracker: ProgressTracker | None = None,
) -> Dict[str, Any]:
    print(f"[seed {seed}] start (jobs={seed_jobs}, B={cfg.B}, power_M={cfg.power_M})")
    if tracker is not None:
        tracker.log_seed_started(seed, seed_jobs)
    t0 = time.time()
    try:
        models = run_all_models_for_seed(cfg, seed, seed_jobs, tracker)
    except Exception as exc:
        if tracker is not None:
            tracker.log_seed_failed(seed, str(exc), time.time() - t0)
        raise
    rt = time.time() - t0
    result = {"seed": int(seed), "jobs": int(seed_jobs), "runtime_sec": float(rt), "models": models}
    write_seed_result(run_dir, seed, result)
    if tracker is not None:
        tracker.log_seed_finished(seed, rt)
    print(f"[seed {seed}] done in {rt:.2f}s")
    return result


def run_all_seeds(
    cfg: ExperimentConfig,
    run_dir: str,
    tracker: ProgressTracker | None = None,
) -> Dict[str, Dict[str, Any]]:
    seed_count = len(cfg.seeds)
    seed_workers = cfg.seed_workers if cfg.seed_workers > 0 else min(seed_count, max(1, cfg.jobs))
    seed_workers = max(1, min(seed_count, seed_workers, max(1, cfg.jobs)))
    slot_budgets = allocate_job_budgets(cfg.jobs, seed_workers)
    print(f"seeds={list(cfg.seeds)} | seed_workers={seed_workers} | job_split={slot_budgets}")

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
            f = executor.submit(run_seed, seed, cfg, budget, run_dir, tracker)
            future_to_slot[f] = slot_idx
            future_to_seed[f] = seed

        while future_to_slot:
            done = next(as_completed(list(future_to_slot.keys())))
            slot_idx = future_to_slot.pop(done)
            seed = future_to_seed.pop(done)
            results[str(seed)] = done.result()
            try:
                next_seed = next(seeds_iter)
                nf = executor.submit(run_seed, next_seed, cfg, slot_budgets[slot_idx], run_dir, tracker)
                future_to_slot[nf] = slot_idx
                future_to_seed[nf] = next_seed
            except StopIteration:
                pass

    return {str(s): results[str(s)] for s in cfg.seeds}


# ===========================================================================
# 聚合与输出
# ===========================================================================

def _agg_numeric(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "count": 0}
    arr = np.array(clean)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr)), "count": len(arr)}


def aggregate_results(seed_results: Dict[str, Dict[str, Any]], cfg: ExperimentConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for model_name in MODEL_EXECUTION_ORDER:
        type1_by_M = []
        if not cfg.skip_type1:
            for M in cfg.M_grid:
                rows = [
                    next(r for r in seed_results[str(s)]["models"][model_name]["type1_error_by_M"] if r["M"] == M)
                    for s in cfg.seeds
                ]
                values = [r["value"] for r in rows]
                dist   = [r["size_distortion"] for r in rows]
                rej    = [r["rejections"] for r in rows]
                eff    = [r["M_effective"] for r in rows]
                rt     = [r["runtime_sec"] for r in rows]
                vs = _agg_numeric(values); ds = _agg_numeric(dist)
                rs = _agg_numeric(rej); es = _agg_numeric(eff); rts = _agg_numeric(rt)
                n_seeds = len(cfg.seeds)
                mc_se = math.sqrt(vs["mean"] * (1 - vs["mean"]) / (M * n_seeds)) if vs["mean"] is not math.nan else math.nan
                type1_by_M.append({
                    "M": int(M),
                    "value_mean": vs["mean"], "value_std": vs["std"],
                    "size_distortion_mean": ds["mean"],
                    "rejections_mean": rs["mean"],
                    "M_effective_mean": es["mean"],
                    "runtime_sec_mean": rts["mean"],
                    "seed_count": vs["count"],
                    "mc_se": mc_se,
                    "ci95_low":  vs["mean"] - 1.96 * mc_se if not math.isnan(mc_se) else math.nan,
                    "ci95_high": vs["mean"] + 1.96 * mc_se if not math.isnan(mc_se) else math.nan,
                })

        power_curve = []
        for delta in cfg.deltas:
            rows = [
                next(r for r in seed_results[str(s)]["models"][model_name]["power_curve"] if abs(r["delta"] - delta) < 1e-12)
                for s in cfg.seeds
            ]
            values    = [r["power"] for r in rows]
            fro       = [r.get("actual_fro", math.nan) for r in rows]
            target    = [r.get("target_fro", math.nan) for r in rows]
            rej       = [r.get("rejections", math.nan) for r in rows]
            eff       = [r.get("M_effective", math.nan) for r in rows]
            rt        = [r.get("runtime_sec", math.nan) for r in rows]
            pert_type = rows[0].get("perturbation_type", "")
            skipped   = sum(1 for r in rows if r.get("skipped", False))
            vs = _agg_numeric(values); fs = _agg_numeric(fro)
            ts = _agg_numeric(target); rs = _agg_numeric(rej)
            es = _agg_numeric(eff); rts = _agg_numeric(rt)
            power_curve.append({
                "M": int(cfg.power_M), "delta": float(delta),
                "target_fro_mean": ts["mean"], "actual_fro_mean": fs["mean"],
                "power_mean": vs["mean"], "power_std": vs["std"],
                "power_min": vs["min"], "power_max": vs["max"],
                "rejections_mean": rs["mean"],
                "M_effective_mean": es["mean"],
                "runtime_sec_mean": rts["mean"],
                "seed_count": vs["count"],
                "skipped_count": int(skipped),
                "perturbation_type": pert_type,
            })

        params = seed_results[str(cfg.seeds[0])]["models"][model_name]["parameters"]
        models[model_name] = {"parameters": params, "type1_error_by_M": type1_by_M, "power_curve": power_curve}
    return {"models": models}


def build_raw_rows(seed_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for seed, sb in seed_results.items():
        for model_name in MODEL_EXECUTION_ORDER:
            block = sb["models"][model_name]
            for r in block["type1_error_by_M"]:
                rows.append({
                    "scope": "seed", "seed": int(seed), "model": model_name,
                    "metric": "type1_error", "M": int(r["M"]),
                    "delta": "", "target_fro": "", "actual_fro": "",
                    "value": float(r["value"]),
                    "size_distortion": float(r["size_distortion"]),
                    "rejections": int(r["rejections"]),
                    "effective_iterations": int(r["M_effective"]),
                    "runtime_sec": float(r["runtime_sec"]),
                    "perturbation_type": "",
                })
            for r in block["power_curve"]:
                rows.append({
                    "scope": "seed", "seed": int(seed), "model": model_name,
                    "metric": "power", "M": int(r["M"]),
                    "delta": float(r["delta"]),
                    "target_fro": float(r["target_fro"]),
                    "actual_fro": float(r.get("actual_fro", math.nan)),
                    "value": float(r["power"]) if not math.isnan(r["power"]) else math.nan,
                    "size_distortion": "",
                    "rejections": int(r.get("rejections", 0)),
                    "effective_iterations": int(r.get("M_effective", 0)),
                    "runtime_sec": float(r.get("runtime_sec", math.nan)),
                    "perturbation_type": r.get("perturbation_type", ""),
                })
    return rows


def build_aggregate_rows(agg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for model_name in MODEL_EXECUTION_ORDER:
        block = agg["models"][model_name]
        for r in block["type1_error_by_M"]:
            rows.append({
                "scope": "aggregate", "seed": "", "model": model_name,
                "metric": "type1_error", "M": int(r["M"]),
                "delta": "", "target_fro": "", "actual_fro": "",
                "value_mean": float(r["value_mean"]),
                "value_std": float(r["value_std"]),
                "size_distortion_mean": float(r["size_distortion_mean"]),
                "rejections_mean": float(r["rejections_mean"]),
                "effective_iterations_mean": float(r["M_effective_mean"]),
                "runtime_sec_mean": float(r["runtime_sec_mean"]),
                "mc_se": float(r["mc_se"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
                "seed_count": int(r["seed_count"]),
                "perturbation_type": "",
            })
        for r in block["power_curve"]:
            rows.append({
                "scope": "aggregate", "seed": "", "model": model_name,
                "metric": "power", "M": int(r["M"]),
                "delta": float(r["delta"]),
                "target_fro_mean": float(r["target_fro_mean"]),
                "actual_fro_mean": float(r["actual_fro_mean"]),
                "value_mean": float(r["power_mean"]),
                "value_std": float(r["power_std"]),
                "size_distortion_mean": "",
                "rejections_mean": float(r["rejections_mean"]),
                "effective_iterations_mean": float(r["M_effective_mean"]),
                "runtime_sec_mean": float(r["runtime_sec_mean"]),
                "seed_count": int(r["seed_count"]),
                "perturbation_type": r.get("perturbation_type", ""),
            })
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(results: Dict[str, Any], path: str) -> None:
    info = results["experiment_info"]
    cfg_dict = info["config"]
    agg = results["aggregates"]["models"]

    lines: List[str] = []
    lines += [
        "# 结构化 VAR 断裂检验仿真报告", "",
        f"- 生成时间：{info['timestamp']}",
        f"- seeds：{cfg_dict['seeds']}",
        f"- B={cfg_dict['B']}  alpha={cfg_dict['alpha']}  M_grid={cfg_dict['M_grid']}  power_M={cfg_dict['power_M']}",
        f"- deltas：{cfg_dict['deltas']}",
        f"- 总耗时：{info['total_runtime_sec']:.2f}s", "",
        "## 实验设计", "",
        "核心叙事：在稀疏和低秩背景下使用 LR+Bootstrap 进行断裂检验，不要求性能高于 OLS。",
        "",
        "| 层次 | 模型 | DGP | N | 估计 | 推断 | 扰动类型 | 角色 |",
        "|---|---|---|---:|---|---|---|---|",
        "| 第一层 | baseline_ols_f | 稠密 | 10 | OLS   | 渐近 F       | uniform | 普通多元时间序列基准 |",
        "| 第一层 | baseline_ols   | 稠密 | 10 | OLS   | LR+Bootstrap | uniform | 提出方法，与 F 检验一致 |",
        "| 第二层 | sparse_lasso   | 稀疏(0.15) | 20 | Lasso | LR+Bootstrap | sparse  | 高维稀疏断裂检验 |",
        "| 第三层 | lowrank_rrr    | 低秩(r=2)  | 20 | RRR   | LR+Bootstrap | lowrank | 高维低秩断裂检验 |",
        "",
        "预期：第一层两方法 size/power 一致；第二/三层 size→0.05（随 M 增大），power 随 δ 单调递增。",
        "",
    ]

    # Size table
    lines += ["## 1. Size（第一类错误，M=2000）", "",
              "| 层次 | 模型 | size mean | size std | size distortion | 95% CI | seeds |",
              "|---|---|---:|---:|---:|---|---:|"]
    size_M = max(cfg_dict["M_grid"])
    layer_map = {
        "baseline_ols_f": "第一层",
        "baseline_ols": "第一层",
        "sparse_lasso": "第二层",
        "lowrank_rrr":  "第三层",
    }
    for m in MODEL_EXECUTION_ORDER:
        row = next((r for r in agg[m]["type1_error_by_M"] if r["M"] == size_M), None)
        if row is None:
            continue
        ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        lines.append(
            f"| {layer_map[m]} | {m} | {row['value_mean']:.4f} | {row['value_std']:.4f} "
            f"| {row['size_distortion_mean']:+.4f} | {ci} | {row['seed_count']} |"
        )
    lines.append("")

    # Power table
    lines += ["## 2. Power（随 δ 的功效曲线）", "",
              "| 层次 | 模型 | 扰动类型 | δ | power mean | power std | seeds |",
              "|---|---|---|---:|---:|---:|---:|"]
    for m in MODEL_EXECUTION_ORDER:
        for r in agg[m]["power_curve"]:
            lines.append(
                f"| {layer_map[m]} | {m} | {r['perturbation_type']} "
                f"| {r['delta']:.2f} | {r['power_mean']:.4f} | {r['power_std']:.4f} | {r['seed_count']} |"
            )
    lines.append("")

    # Summary
    lines += ["## 3. 结论摘要", ""]
    role_map = {
        "baseline_ols_f": "普通多元时间序列 F 检验基准",
        "baseline_ols":   "LR+Bootstrap（与 F 检验对比）",
        "sparse_lasso":   "高维稀疏 LR+Bootstrap",
        "lowrank_rrr":    "高维低秩 RRR+LR+Bootstrap",
    }
    for m in MODEL_EXECUTION_ORDER:
        pc = agg[m]["power_curve"]
        pvals = [r["power_mean"] for r in pc if not math.isnan(r["power_mean"])]
        mono = len(pvals) >= 2 and all(pvals[i] <= pvals[i+1] for i in range(len(pvals)-1))
        t1_rows = agg[m]["type1_error_by_M"]
        size_str = ""
        if t1_rows:
            sr = next((r for r in t1_rows if r["M"] == size_M), t1_rows[-1])
            size_str = f"size@M={size_M}={sr['value_mean']:.4f}; "
        final = pc[-1] if pc else {}
        lines.append(
            f"- **{m}** [{role_map.get(m, layer_map.get(m, '?'))}]: {size_str}"
            f"power@δ={final.get('delta', '?'):.2f}={final.get('power_mean', math.nan):.4f}; "
            f"power单调={'✓' if mono else '✗'}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ===========================================================================
# 主程序
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="结构化 VAR 断裂检验：普通/稀疏/低秩三层场景实验")
    parser.add_argument("--M-grid",    type=int,   nargs="+", default=[50, 100, 300, 500, 1000, 2000])
    parser.add_argument("--power-M",   type=int,   default=0,   help="0 = use max(M_grid)")
    parser.add_argument("--B",         type=int,   default=500)
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--deltas",    type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--seeds",     type=int,   nargs="+", default=[42, 2026])
    parser.add_argument("--jobs",      type=int,   default=4)
    parser.add_argument("--seed-workers", type=int, default=1,
                        help="并发seed数（默认1：顺序跑seed，单seed独占全部jobs）")
    parser.add_argument("--skip-type1",   action="store_true")
    parser.add_argument("--skip-power",   action="store_true")
    parser.add_argument("--models",    type=str,   nargs="+", default=None,
                        help="仅运行指定模型（默认全部）：" + " ".join(MODEL_EXECUTION_ORDER))
    parser.add_argument("--tag",       type=str,   default="")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        M_grid=tuple(sorted(set(int(m) for m in args.M_grid))),
        B=int(args.B),
        alpha=float(args.alpha),
        deltas=tuple(float(d) for d in args.deltas),
        seeds=tuple(int(s) for s in args.seeds),
        jobs=max(1, int(args.jobs)),
        seed_workers=max(0, int(args.seed_workers)),
        _power_M=max(0, int(args.power_M)),
        skip_type1=args.skip_type1,
        skip_power=args.skip_power,
    )

    # 支持只跑部分模型（覆盖全局 MODEL_EXECUTION_ORDER）
    _all_models = MODEL_EXECUTION_ORDER
    active_models = tuple(args.models) if args.models else _all_models
    if args.models:
        unknown = [m for m in args.models if m not in _all_models]
        if unknown:
            parser.error(f"未知模型：{unknown}。可选：{list(_all_models)}")
    globals()["MODEL_EXECUTION_ORDER"] = active_models

    # 输出目录
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    run_name = f"{ts}{tag_suffix}"
    run_dir = os.path.join(PROJECT_ROOT, "results", "structured_scenario_runs", run_name)
    os.makedirs(os.path.join(run_dir, "seed_results"), exist_ok=True)

    # 计算总阶段数
    n_type1 = 0 if cfg.skip_type1 else len(cfg.M_grid)
    n_power = 0 if cfg.skip_power else len(cfg.deltas)
    stages_per_model = n_type1 + n_power
    total_stages = len(cfg.seeds) * len(MODEL_EXECUTION_ORDER) * stages_per_model

    tracker = ProgressTracker(run_dir=run_dir, total_stage_count=total_stages, per_seed_stage_count=len(MODEL_EXECUTION_ORDER) * stages_per_model)
    global _ACTIVE_TRACKER, _RUN_START_TIME
    _ACTIVE_TRACKER = tracker
    _RUN_START_TIME = time.time()
    _install_signal_handlers(tracker)

    print(f"输出目录：{run_dir}")
    print(f"模型：{list(MODEL_EXECUTION_ORDER)}")
    print(f"seeds={list(cfg.seeds)}  B={cfg.B}  power_M={cfg.power_M}  jobs={cfg.jobs}")
    tracker.log_run_started(cfg)

    run_start = time.time()
    seed_results = run_all_seeds(cfg, run_dir, tracker)
    total_runtime = time.time() - run_start

    # 聚合
    agg = aggregate_results(seed_results, cfg)

    # 完整结果 JSON
    full_results = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": run_name,
            "total_runtime_sec": total_runtime,
            "config": {
                "seeds": list(cfg.seeds), "B": cfg.B, "alpha": cfg.alpha,
                "M_grid": list(cfg.M_grid), "power_M": cfg.power_M,
                "deltas": list(cfg.deltas),
            },
            "models": list(MODEL_EXECUTION_ORDER),
            "scenario": {
                "T": _T, "p": _p, "t_star": _t,
                "N_baseline": _N_BASELINE, "N_sparse": _N_SPARSE, "N_lowrank": _N_LOWRANK,
                "sparse_sparsity": _SPARSE_SPARSITY, "lowrank_rank": _LOWRANK_RANK,
                "lasso_alpha": _LASSO_ALPHA, "rrr_rank": _RRR_RANK,
            },
        },
        "aggregates": agg,
        "seed_results": {str(s): seed_results[str(s)] for s in cfg.seeds},
    }

    json_path = os.path.join(run_dir, f"structured_{run_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    # CSV
    raw_path = os.path.join(run_dir, f"structured_raw_{run_name}.csv")
    agg_path = os.path.join(run_dir, f"structured_agg_{run_name}.csv")
    write_csv(raw_path, build_raw_rows(seed_results))
    write_csv(agg_path, build_aggregate_rows(agg))

    # Markdown 报告
    report_path = os.path.join(run_dir, f"结构化场景仿真报告_{run_name}.md")
    write_markdown_report(full_results, report_path)

    tracker.log_run_finished(total_runtime)

    print(f"\n实验完成！耗时 {total_runtime:.1f}s")
    print(f"  JSON  ：{json_path}")
    print(f"  CSV   ：{agg_path}")
    print(f"  报告  ：{report_path}")

    # 快速结果预览
    print("\n=== Size@M=2000 / Power@δ=0.5 ===")
    size_M = max(cfg.M_grid)
    for m in MODEL_EXECUTION_ORDER:
        block = agg["models"][m]
        sr = next((r for r in block["type1_error_by_M"] if r["M"] == size_M), None)
        pr = next((r for r in block["power_curve"] if abs(r["delta"] - 0.5) < 1e-9), None)
        size_str  = f"{sr['value_mean']:.4f}" if sr else "N/A"
        power_str = f"{pr['power_mean']:.4f}" if pr else "N/A"
        print(f"  {m:<20} size={size_str}  power={power_str}")


if __name__ == "__main__":
    main()
