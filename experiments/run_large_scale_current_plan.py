"""Run larger-scale Monte Carlo experiments under the current known-breakpoint plan.

Outputs:
- JSON: full experiment outputs and metadata
- CSV: tidy plotting data (size + power curve)
- Markdown: analysis report with core findings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

# Ensure project root is importable when script is launched via relative path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation import MonteCarloSimulation, VARDataGenerator
from sparse_var import SparseMonteCarloSimulation
from lowrank_var import LowRankMonteCarloSimulation


@dataclass
class ExperimentConfig:
    M: int = 200
    B: int = 150
    alpha: float = 0.05
    deltas: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    seed: int = 42
    jobs: int = 1
    baseline_pvalue_method: str = "bootstrap_lr"


MODEL_EXECUTION_ORDER = ("baseline_ols", "sparse_lasso", "lowrank_svd")
MODEL_JOB_PRIORITY = ("sparse_lasso", "lowrank_svd", "baseline_ols")


@dataclass
class ProgressTracker:
    run_dir: str
    seed: int
    total_stage_count: int
    progress_log_path: str = field(init=False)
    completed_stage_count: int = field(default=0, init=False)
    completed_stages: List[str] = field(default_factory=list, init=False)
    active_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.progress_log_path = os.path.join(self.run_dir, "00_progress.log")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _stage_id(self, model_name: str, stage_name: str) -> str:
        return f"seed_{self.seed}/{model_name}/{stage_name}"

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.6f}"
        if isinstance(value, list):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return str(value)

    def _append(self, event: Dict[str, Any]) -> None:
        active_stage_ids = sorted(self.active_stages.keys())
        completed_stage_ids = list(self.completed_stages)
        parts = [
            f"[{event['timestamp']}]",
            f"event={event['event_type']}",
            f"status={event['status']}",
            f"seed={self.seed}",
            f"total_progress={event['total_progress']['completed']}/{event['total_progress']['total']}",
        ]
        if event.get("model"):
            parts.append(f"model={event['model']}")
        if event.get("stage"):
            parts.append(f"stage={event['stage']}")
        if event.get("runtime_sec") is not None:
            parts.append(f"runtime_sec={event['runtime_sec']:.2f}")
        extra = event.get("extra") or {}
        for key in sorted(extra):
            parts.append(f"{key}={self._format_value(extra[key])}")
        parts.append(f"active_stages={self._format_value(active_stage_ids)}")
        parts.append(f"completed_stages={self._format_value(completed_stage_ids)}")
        with open(self.progress_log_path, "a", encoding="utf-8") as f:
            f.write(" | ".join(parts) + "\n")
        self.history.append(event)

    def record_event(
        self,
        *,
        event_type: str,
        status: str,
        model_name: str = "",
        stage_name: str = "",
        count_complete: bool = False,
        runtime_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra = extra or {}
        with self._lock:
            stage_id = self._stage_id(model_name, stage_name) if model_name and stage_name else ""
            if status == "started" and stage_id:
                self.active_stages[stage_id] = {
                    "stage_id": stage_id,
                    "seed": self.seed,
                    "model": model_name,
                    "stage": stage_name,
                    "status": status,
                    "started_at": self._timestamp(),
                    **extra,
                }
            elif stage_id and stage_id in self.active_stages:
                active = self.active_stages.pop(stage_id)
                if count_complete:
                    self.completed_stage_count += 1
                    self.completed_stages.append(stage_id)
                for key in ("seed", "model", "stage", "status"):
                    active.pop(key, None)
                extra = {**active, **extra}

            timestamp = self._timestamp()
            event = {
                "timestamp": timestamp,
                "seed": self.seed,
                "event_type": event_type,
                "status": status,
                "model": model_name or None,
                "stage": stage_name or None,
                "stage_id": stage_id or None,
                "total_progress": {
                    "completed": self.completed_stage_count,
                    "total": self.total_stage_count,
                    "ratio": 0.0 if self.total_stage_count <= 0 else self.completed_stage_count / self.total_stage_count,
                },
                "runtime_sec": runtime_sec,
                "extra": extra or None,
            }
            self._append(event)

    def log_run_started(self, tag: str, jobs: int, deltas: Tuple[float, ...]) -> None:
        self.record_event(
            event_type="run",
            status="started",
            extra={"tag": tag, "jobs": jobs, "deltas": list(deltas)},
        )

    def log_model_submitted(self, model_name: str, mc_jobs: int, queue_index: int) -> None:
        self.record_event(
            event_type="model",
            status="submitted",
            model_name=model_name,
            extra={"mc_jobs": mc_jobs, "queue_index": queue_index},
        )

    def start_stage(self, model_name: str, stage_name: str, **extra: Any) -> None:
        self.record_event(
            event_type="stage",
            status="started",
            model_name=model_name,
            stage_name=stage_name,
            extra=extra,
        )

    def finish_stage(self, model_name: str, stage_name: str, runtime_sec: float, **extra: Any) -> None:
        self.record_event(
            event_type="stage",
            status="completed",
            model_name=model_name,
            stage_name=stage_name,
            count_complete=True,
            runtime_sec=runtime_sec,
            extra=extra,
        )

    def fail_stage(self, model_name: str, stage_name: str, error: str, runtime_sec: Optional[float] = None, **extra: Any) -> None:
        self.record_event(
            event_type="stage",
            status="failed",
            model_name=model_name,
            stage_name=stage_name,
            runtime_sec=runtime_sec,
            extra={**extra, "error": error},
        )

    def log_model_finished(self, model_name: str, runtime_sec: float) -> None:
        self.record_event(
            event_type="model",
            status="completed",
            model_name=model_name,
            runtime_sec=runtime_sec,
        )

    def log_model_failed(self, model_name: str, error: str, runtime_sec: Optional[float] = None) -> None:
        self.record_event(
            event_type="model",
            status="failed",
            model_name=model_name,
            runtime_sec=runtime_sec,
            extra={"error": error},
        )

    def log_run_finished(self, runtime_sec: float) -> None:
        self.record_event(
            event_type="run",
            status="completed",
            runtime_sec=runtime_sec,
            extra={"completed_stage_count": self.completed_stage_count},
        )


def make_run_dir(tag: str) -> Tuple[str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    base_dir = os.path.join("results", "large_scale_runs")
    run_name = f"{stamp}{suffix}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return stamp, run_dir


def allocate_model_jobs(total_jobs: int) -> Dict[str, int]:
    """Allocate the total worker budget across models.

    The experiment-level `jobs` flag is treated as a total CPU budget.
    We keep at least one worker per model when possible, and assign extra
    workers to the more expensive sparse / low-rank models first.
    """
    total_jobs = max(1, int(total_jobs))
    budgets = {name: 1 for name in MODEL_EXECUTION_ORDER}
    if total_jobs <= 1:
        return {"baseline_ols": 1, "sparse_lasso": 1, "lowrank_svd": 1}

    extra = max(0, total_jobs - len(MODEL_EXECUTION_ORDER))
    idx = 0
    while extra > 0:
        budgets[MODEL_JOB_PRIORITY[idx % len(MODEL_JOB_PRIORITY)]] += 1
        extra -= 1
        idx += 1
    return budgets


def run_all_models(cfg: ExperimentConfig, tracker: Optional[ProgressTracker] = None) -> Dict[str, Dict]:
    runners = {
        "baseline_ols": run_baseline,
        "sparse_lasso": run_sparse,
        "lowrank_svd": run_lowrank,
    }
    model_jobs = allocate_model_jobs(cfg.jobs)
    max_workers = min(len(MODEL_EXECUTION_ORDER), max(1, cfg.jobs))

    if max_workers <= 1:
        results: Dict[str, Dict] = {}
        for idx, model_name in enumerate(MODEL_EXECUTION_ORDER, start=1):
            model_cfg = replace(cfg, jobs=model_jobs[model_name])
            if tracker is not None:
                tracker.log_model_submitted(model_name, model_cfg.jobs, idx)
            print(f"[{idx}/3] {model_name} ... (mc_jobs={model_cfg.jobs})")
            start = time.time()
            try:
                results[model_name] = runners[model_name](model_cfg, tracker)
            except Exception as exc:
                if tracker is not None:
                    tracker.log_model_failed(model_name, str(exc), time.time() - start)
                raise
            runtime = time.time() - start
            if tracker is not None:
                tracker.log_model_finished(model_name, runtime)
            print(f"  done in {runtime:.2f}s")
        return results

    print(f"Running models in parallel with total_jobs={cfg.jobs}; model_job_split={model_jobs}")
    results: Dict[str, Dict] = {}
    started_at: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {}
        for idx, model_name in enumerate(MODEL_EXECUTION_ORDER, start=1):
            model_cfg = replace(cfg, jobs=model_jobs[model_name])
            if tracker is not None:
                tracker.log_model_submitted(model_name, model_cfg.jobs, idx)
            print(f"[{idx}/3] submit {model_name} ... (mc_jobs={model_cfg.jobs})")
            started_at[model_name] = time.time()
            future = executor.submit(runners[model_name], model_cfg, tracker)
            future_to_model[future] = model_name

        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as exc:
                if tracker is not None:
                    tracker.log_model_failed(model_name, str(exc), time.time() - started_at[model_name])
                raise
            runtime = time.time() - started_at[model_name]
            if tracker is not None:
                tracker.log_model_finished(model_name, runtime)
            print(f"  {model_name} done in {runtime:.2f}s")

    return {name: results[name] for name in MODEL_EXECUTION_ORDER}


def ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int]:
    """Shrink coefficient matrix until stationary or attempts exhausted."""
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
    """Build a stationary post-break coefficient matrix with target Frobenius shift.

    The perturbation direction is the normalized all-ones matrix, so all models use
    the same direction family while matching the same requested total effect size
    `||Phi2 - Phi1||_F` before any stationarity shrinkage.
    """
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
    """Convert a common per-entry change scale into a model-specific Frobenius target.

    When the perturbation direction is the normalized all-ones matrix, a common
    per-entry shift `base_delta` corresponds to `target_fro = base_delta * sqrt(phi.size)`.
    This keeps the average coefficient perturbation comparable across models of
    different dimensionality.
    """
    return float(base_delta * np.sqrt(phi.size))
def summarize_pvalues(pvalues: np.ndarray) -> Dict[str, float]:
    if pvalues is None or len(pvalues) == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
        }
    return {
        "count": int(len(pvalues)),
        "mean": float(np.mean(pvalues)),
        "std": float(np.std(pvalues)),
        "q25": float(np.quantile(pvalues, 0.25)),
        "q50": float(np.quantile(pvalues, 0.50)),
        "q75": float(np.quantile(pvalues, 0.75)),
    }


def run_baseline(cfg: ExperimentConfig, tracker: Optional[ProgressTracker] = None) -> Dict:
    model_name = "baseline_ols"
    generator = VARDataGenerator(seed=cfg.seed)

    N, T, p, t = 2, 100, 1, 50
    Sigma = np.eye(N) * 0.5
    phi = generator.generate_stationary_phi(N, p, scale=0.3)

    mc = MonteCarloSimulation(M=cfg.M, B=cfg.B, seed=cfg.seed, n_jobs=cfg.jobs, baseline_pvalue_method=cfg.baseline_pvalue_method)

    type1_stage = "type1_error"
    if tracker is not None:
        tracker.start_stage(model_name, type1_stage, task="type1_error", N=N, T=T, p=p, t=t)
    t0 = time.time()
    try:
        type1 = mc.evaluate_type1_error_at_point(N, T, p, phi, Sigma, t=t, alpha=cfg.alpha, verbose=False)
    except Exception as exc:
        if tracker is not None:
            tracker.fail_stage(model_name, type1_stage, str(exc), time.time() - t0)
        raise
    type1_runtime = time.time() - t0
    if tracker is not None:
        tracker.finish_stage(
            model_name,
            type1_stage,
            type1_runtime,
            task="type1_error",
            rejections=int(type1["rejections"]),
            effective_iterations=int(type1["M_effective"]),
            metric_value=float(type1["type1_error"]),
        )

    power_curve = []
    for base_delta in cfg.deltas:
        stage_name = f"power_delta_{base_delta:.2f}"
        target_fro = scale_base_delta_to_target_fro(phi, base_delta)
        if tracker is not None:
            tracker.start_stage(model_name, stage_name, task="power", base_delta=float(base_delta), target_fro=float(target_fro))
        phi2, ok, shrinks, actual_fro = build_phi2_with_target_frobenius(phi, target_fro)
        if not ok:
            skipped_payload = {
                "delta": float(base_delta),
                "target_fro": float(target_fro),
                "actual_fro": float("nan"),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_fro_shrink",
            }
            power_curve.append(skipped_payload)
            if tracker is not None:
                tracker.finish_stage(
                    model_name,
                    stage_name,
                    0.0,
                    task="power",
                    target_fro=float(target_fro),
                    base_delta=float(base_delta),
                    actual_fro=float("nan"),
                    skipped=True,
                    reason="nonstationary_after_fro_shrink",
                )
            continue

        t1 = time.time()
        try:
            result = mc.evaluate_power_at_point(
                N, T, p, phi, phi2, Sigma, break_point=t, t=t, alpha=cfg.alpha, verbose=False
            )
        except Exception as exc:
            if tracker is not None:
                tracker.fail_stage(model_name, stage_name, str(exc), time.time() - t1, target_fro=float(target_fro), base_delta=float(base_delta))
            raise
        runtime = time.time() - t1
        power_payload = {
            "delta": float(base_delta),
            "target_fro": float(target_fro),
            "actual_fro": float(actual_fro),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        }
        power_curve.append(power_payload)
        if tracker is not None:
            tracker.finish_stage(
                model_name,
                stage_name,
                runtime,
                task="power",
                target_fro=float(target_fro),
                base_delta=float(base_delta),
                actual_fro=float(actual_fro),
                rejections=int(result["rejections"]),
                effective_iterations=int(result["M_effective"]),
                metric_value=float(result["power"]),
                skipped=False,
            )

    return {
        "model": "baseline_ols",
        "parameters": {"N": N, "T": T, "p": p, "t": t, "M": cfg.M, "B": cfg.B, "alpha": cfg.alpha, "baseline_pvalue_method": cfg.baseline_pvalue_method},
        "phi": phi.tolist(),
        "type1_error": {
            "value": float(type1["type1_error"]),
            "size_distortion": float(type1["size_distortion"]),
            "rejections": int(type1["rejections"]),
            "M_effective": int(type1["M_effective"]),
            "runtime_sec": float(type1_runtime),
            "pvalue_summary": summarize_pvalues(type1["p_values"]),
        },
        "power_curve": power_curve,
    }


def run_sparse(cfg: ExperimentConfig, tracker: Optional[ProgressTracker] = None) -> Dict:
    model_name = "sparse_lasso"
    generator = VARDataGenerator(seed=cfg.seed)

    N, T, p, t = 5, 200, 1, 100
    Sigma = np.eye(N) * 0.5
    sparsity = 0.2
    lasso_alpha = 0.02

    phi = generator.generate_stationary_phi(N, p, sparsity=sparsity, scale=0.3)

    mc = SparseMonteCarloSimulation(
        M=cfg.M,
        B=cfg.B,
        seed=cfg.seed,
        estimator_type="lasso",
        alpha=lasso_alpha,
        n_jobs=cfg.jobs,
    )

    type1_stage = "type1_error"
    if tracker is not None:
        tracker.start_stage(model_name, type1_stage, task="type1_error", N=N, T=T, p=p, t=t)
    t0 = time.time()
    try:
        type1 = mc.evaluate_type1_error(N, T, p, phi, Sigma, t=t, test_alpha=cfg.alpha, verbose=False)
    except Exception as exc:
        if tracker is not None:
            tracker.fail_stage(model_name, type1_stage, str(exc), time.time() - t0)
        raise
    type1_runtime = time.time() - t0
    if tracker is not None:
        tracker.finish_stage(
            model_name,
            type1_stage,
            type1_runtime,
            task="type1_error",
            rejections=int(type1["rejections"]),
            effective_iterations=int(type1["M_effective"]),
            metric_value=float(type1["type1_error"]),
        )

    power_curve = []
    for base_delta in cfg.deltas:
        stage_name = f"power_delta_{base_delta:.2f}"
        target_fro = scale_base_delta_to_target_fro(phi, base_delta)
        if tracker is not None:
            tracker.start_stage(model_name, stage_name, task="power", base_delta=float(base_delta), target_fro=float(target_fro))
        phi2, ok, shrinks, actual_fro = build_phi2_with_target_frobenius(phi, target_fro)
        if not ok:
            skipped_payload = {
                "delta": float(base_delta),
                "target_fro": float(target_fro),
                "actual_fro": float("nan"),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_fro_shrink",
            }
            power_curve.append(skipped_payload)
            if tracker is not None:
                tracker.finish_stage(
                    model_name,
                    stage_name,
                    0.0,
                    task="power",
                    target_fro=float(target_fro),
                    base_delta=float(base_delta),
                    actual_fro=float("nan"),
                    skipped=True,
                    reason="nonstationary_after_fro_shrink",
                )
            continue

        t1 = time.time()
        try:
            result = mc.evaluate_power(
                N, T, p, phi, phi2, Sigma, break_point=t, t=t, test_alpha=cfg.alpha, verbose=False
            )
        except Exception as exc:
            if tracker is not None:
                tracker.fail_stage(model_name, stage_name, str(exc), time.time() - t1, target_fro=float(target_fro), base_delta=float(base_delta))
            raise
        runtime = time.time() - t1
        power_payload = {
            "delta": float(base_delta),
            "target_fro": float(target_fro),
            "actual_fro": float(actual_fro),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        }
        power_curve.append(power_payload)
        if tracker is not None:
            tracker.finish_stage(
                model_name,
                stage_name,
                runtime,
                task="power",
                target_fro=float(target_fro),
                base_delta=float(base_delta),
                actual_fro=float(actual_fro),
                rejections=int(result["rejections"]),
                effective_iterations=int(result["M_effective"]),
                metric_value=float(result["power"]),
                skipped=False,
            )

    return {
        "model": "sparse_lasso",
        "parameters": {
            "N": N,
            "T": T,
            "p": p,
            "t": t,
            "M": cfg.M,
            "B": cfg.B,
            "alpha": cfg.alpha,
            "sparsity": sparsity,
            "lasso_alpha": lasso_alpha,
        },
        "phi": phi.tolist(),
        "type1_error": {
            "value": float(type1["type1_error"]),
            "size_distortion": float(type1["size_distortion"]),
            "rejections": int(type1["rejections"]),
            "M_effective": int(type1["M_effective"]),
            "runtime_sec": float(type1_runtime),
            "pvalue_summary": summarize_pvalues(type1["p_values"]),
        },
        "power_curve": power_curve,
    }


def run_lowrank(cfg: ExperimentConfig, tracker: Optional[ProgressTracker] = None) -> Dict:
    model_name = "lowrank_svd"
    generator = VARDataGenerator(seed=cfg.seed)

    N, T, p, t = 10, 200, 1, 100
    Sigma = np.eye(N) * 0.5
    rank = 2

    phi = generator.generate_lowrank_phi(N, p, rank=rank, scale=0.3)

    mc = LowRankMonteCarloSimulation(
        M=cfg.M,
        B=cfg.B,
        seed=cfg.seed,
        method="svd",
        rank=rank,
        n_jobs=cfg.jobs,
    )

    type1_stage = "type1_error"
    if tracker is not None:
        tracker.start_stage(model_name, type1_stage, task="type1_error", N=N, T=T, p=p, t=t)
    t0 = time.time()
    try:
        type1 = mc.evaluate_type1_error(N, T, p, phi, Sigma, t=t, test_alpha=cfg.alpha, verbose=False)
    except Exception as exc:
        if tracker is not None:
            tracker.fail_stage(model_name, type1_stage, str(exc), time.time() - t0)
        raise
    type1_runtime = time.time() - t0
    if tracker is not None:
        tracker.finish_stage(
            model_name,
            type1_stage,
            type1_runtime,
            task="type1_error",
            rejections=int(type1["rejections"]),
            effective_iterations=int(type1["M_effective"]),
            metric_value=float(type1["type1_error"]),
        )

    power_curve = []
    for base_delta in cfg.deltas:
        stage_name = f"power_delta_{base_delta:.2f}"
        target_fro = scale_base_delta_to_target_fro(phi, base_delta)
        if tracker is not None:
            tracker.start_stage(model_name, stage_name, task="power", base_delta=float(base_delta), target_fro=float(target_fro))
        phi2, ok, shrinks, actual_fro = build_phi2_with_target_frobenius(phi, target_fro)
        if not ok:
            skipped_payload = {
                "delta": float(base_delta),
                "target_fro": float(target_fro),
                "actual_fro": float("nan"),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_fro_shrink",
            }
            power_curve.append(skipped_payload)
            if tracker is not None:
                tracker.finish_stage(
                    model_name,
                    stage_name,
                    0.0,
                    task="power",
                    target_fro=float(target_fro),
                    base_delta=float(base_delta),
                    actual_fro=float("nan"),
                    skipped=True,
                    reason="nonstationary_after_fro_shrink",
                )
            continue

        t1 = time.time()
        try:
            result = mc.evaluate_power(
                N, T, p, phi, phi2, Sigma, break_point=t, t=t, test_alpha=cfg.alpha, verbose=False
            )
        except Exception as exc:
            if tracker is not None:
                tracker.fail_stage(model_name, stage_name, str(exc), time.time() - t1, target_fro=float(target_fro), base_delta=float(base_delta))
            raise
        runtime = time.time() - t1
        power_payload = {
            "delta": float(base_delta),
            "target_fro": float(target_fro),
            "actual_fro": float(actual_fro),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        }
        power_curve.append(power_payload)
        if tracker is not None:
            tracker.finish_stage(
                model_name,
                stage_name,
                runtime,
                task="power",
                target_fro=float(target_fro),
                base_delta=float(base_delta),
                actual_fro=float(actual_fro),
                rejections=int(result["rejections"]),
                effective_iterations=int(result["M_effective"]),
                metric_value=float(result["power"]),
                skipped=False,
            )

    return {
        "model": "lowrank_svd",
        "parameters": {
            "N": N,
            "T": T,
            "p": p,
            "t": t,
            "M": cfg.M,
            "B": cfg.B,
            "alpha": cfg.alpha,
            "rank": rank,
        },
        "phi": phi.tolist(),
        "type1_error": {
            "value": float(type1["type1_error"]),
            "size_distortion": float(type1["size_distortion"]),
            "rejections": int(type1["rejections"]),
            "M_effective": int(type1["M_effective"]),
            "runtime_sec": float(type1_runtime),
            "pvalue_summary": summarize_pvalues(type1["p_values"]),
        },
        "power_curve": power_curve,
    }


def build_plot_rows(results: Dict) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        block = results["models"][model_key]
        rows.append(
            {
                "model": model_key,
                "metric": "type1_error",
                "delta": "",
                "target_fro": "",
                "actual_fro": "",
                "value": f"{block['type1_error']['value']:.6f}",
                "size_distortion": f"{block['type1_error']['size_distortion']:.6f}",
                "rejections": str(block["type1_error"]["rejections"]),
                "effective_iterations": str(block["type1_error"]["M_effective"]),
                "runtime_sec": f"{block['type1_error']['runtime_sec']:.3f}",
            }
        )
        for pt in block["power_curve"]:
            rows.append(
                {
                    "model": model_key,
                    "metric": "power",
                    "delta": f"{pt['delta']:.2f}",
                    "target_fro": "" if "target_fro" not in pt else f"{pt['target_fro']:.6f}",
                    "actual_fro": "" if np.isnan(pt.get("actual_fro", np.nan)) else f"{pt['actual_fro']:.6f}",
                    "value": "" if np.isnan(pt["power"]) else f"{pt['power']:.6f}",
                    "size_distortion": "",
                    "rejections": str(pt.get("rejections", "")),
                    "effective_iterations": str(pt.get("M_effective", "")),
                    "runtime_sec": f"{pt.get('runtime_sec', float('nan')):.3f}" if "runtime_sec" in pt else "",
                }
            )
    return rows


def write_markdown_report(results: Dict, report_path: str) -> None:
    ts = results["experiment_info"]["timestamp"]
    cfg = results["experiment_info"]["config"]
    output_paths = results["experiment_info"].get("output_paths", {})

    lines: List[str] = []
    lines.append("# 大规模试验分析报告")
    lines.append("")
    lines.append(f"- 生成时间: {ts}")
    lines.append(f"- 方案配置: M={cfg['M']}, B={cfg['B']}, alpha={cfg['alpha']}, base deltas={cfg['deltas']}, baseline_pvalue_method={cfg.get('baseline_pvalue_method', 'bootstrap_lr')}")
    lines.append("- 断点构造: 先给定统一基准单元素变化尺度 `delta`，再按 `target_fro = delta * sqrt(#coefficients)` 换算为模型特定的 Frobenius 目标强度；若不平稳则仅缩小扰动幅度。\n")
    lines.append(f"- 总耗时(秒): {results['experiment_info']['total_runtime_sec']:.2f}")
    if output_paths:
        lines.append(f"- 进度日志: {output_paths.get('progress_log', '')}")
        lines.append(f"- 输出目录: {output_paths.get('run_dir', '')}")
    lines.append("")

    lines.append("## 1. Size（第一类错误）")
    lines.append("")
    lines.append("| 模型 | Type I Error | Size Distortion | Rejections | M_effective | Runtime(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        block = results["models"][model_key]
        t1 = block["type1_error"]
        lines.append(
            f"| {model_key} | {t1['value']:.4f} | {t1['size_distortion']:+.4f} | "
            f"{t1['rejections']} | {t1['M_effective']} | {t1['runtime_sec']:.2f} |"
        )
    lines.append("")

    lines.append("## 2. Power 曲线数据")
    lines.append("")
    lines.append("| 模型 | Base δ / Target `||ΔΦ||_F` | Actual `||ΔΦ||_F` | Power | Rejections | M_effective | Runtime(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        for pt in results["models"][model_key]["power_curve"]:
            power_val = "nan" if np.isnan(pt["power"]) else f"{pt['power']:.4f}"
            actual_fro = "nan" if np.isnan(pt.get("actual_fro", math.nan)) else f"{pt['actual_fro']:.4f}"
            lines.append(
                f"| {model_key} | {pt['delta']:.2f} | {pt.get('target_fro', math.nan):.2f} | {actual_fro} | {power_val} | "
                f"{pt.get('rejections', 0)} | {pt.get('M_effective', 0)} | {pt.get('runtime_sec', math.nan):.2f} |"
            )
    lines.append("")

    lines.append("## 3. 结论摘要")
    lines.append("")
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        block = results["models"][model_key]
        t1 = block["type1_error"]["value"]
        valid_points = [pt for pt in block["power_curve"] if not np.isnan(pt["power"])]
        valid_power = [pt["power"] for pt in valid_points]
        if len(valid_power) >= 2:
            monotone = all(valid_power[i] <= valid_power[i + 1] for i in range(len(valid_power) - 1))
        else:
            monotone = False

        if valid_points:
            final_point = valid_points[-1]
            final_delta = final_point.get("target_fro", final_point["delta"])
            final_power = final_point["power"]
            power_gain = final_power - t1
        else:
            final_delta = math.nan
            final_power = math.nan
            power_gain = math.nan

        lines.append(
            f"- {model_key}: type1_error={t1:.4f}; power_at_max_base_delta({final_delta:.2f})={final_power:.4f}; "
            f"power_gain_over_size={power_gain:.4f}; power_monotone={monotone}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run larger-scale experiments and export report/data.")
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--B", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--baseline-pvalue-method", type=str, default="bootstrap_lr", choices=["bootstrap_lr", "asymptotic_chi2", "asymptotic_f"])
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        M=args.M,
        B=args.B,
        alpha=args.alpha,
        deltas=tuple(args.deltas),
        seed=args.seed,
        jobs=max(1, args.jobs),
        baseline_pvalue_method=args.baseline_pvalue_method,
    )

    stamp, run_dir = make_run_dir(args.tag)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    total_stage_count = len(MODEL_EXECUTION_ORDER) * (1 + len(cfg.deltas))
    tracker = ProgressTracker(run_dir=run_dir, seed=cfg.seed, total_stage_count=total_stage_count)
    tracker.log_run_started(args.tag, cfg.jobs, cfg.deltas)

    print("=" * 72)
    print("Running large-scale experiment")
    print(
        f"Config: M={cfg.M}, B={cfg.B}, alpha={cfg.alpha}, "
        f"base_deltas={list(cfg.deltas)}, seed={cfg.seed}, jobs={cfg.jobs}, baseline_pvalue_method={cfg.baseline_pvalue_method}"
    )
    print("=" * 72)

    all_start = time.time()

    model_results = run_all_models(cfg, tracker)

    total_runtime = time.time() - all_start
    tracker.log_run_finished(total_runtime)

    results = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Large-scale known-breakpoint LR+Bootstrap experiments",
            "config": {
                **asdict(cfg),
                "effect_size_mode": "dimension_scaled_frobenius_from_base_delta",
                "effect_direction": "normalized_all_ones",
                "delta_interpretation": "common per-entry shift scale",
            },
            "total_runtime_sec": float(total_runtime),
        },
        "models": model_results,
    }

    json_path = os.path.join(run_dir, f"large_scale_experiment_{stamp}{tag_suffix}.json")
    csv_path = os.path.join(run_dir, f"large_scale_plot_data_{stamp}{tag_suffix}.csv")
    md_path = os.path.join(run_dir, f"大规模试验分析报告_{stamp}{tag_suffix}.md")
    run_meta_path = os.path.join(run_dir, "run_meta.json")

    results["experiment_info"]["output_paths"] = {
        "run_dir": run_dir,
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "run_meta": run_meta_path,
        "progress_log": tracker.progress_log_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_rows = build_plot_rows(results)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "metric",
                "delta",
                "target_fro",
                "actual_fro",
                "value",
                "size_distortion",
                "rejections",
                "effective_iterations",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(plot_rows)

    write_markdown_report(results, md_path)

    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(results["experiment_info"]["output_paths"], f, ensure_ascii=False, indent=2)

    print("\nOutputs:")
    print(f"- PROGRESS: {tracker.progress_log_path}")
    print(f"- RUN     : {run_dir}")
    print(f"- JSON    : {json_path}")
    print(f"- CSV     : {csv_path}")
    print(f"- MD      : {md_path}")
    print(f"- META    : {run_meta_path}")
    print(f"Total runtime: {total_runtime:.2f}s")


if __name__ == "__main__":
    main()
