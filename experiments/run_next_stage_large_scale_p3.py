"""
下一阶段大规模实验（p=3，多seed，均值+95%CI）。

目标：
1) Baseline：比较渐近 p 值与 Bootstrap p 值（随 B 变化）。
2) Size：在 baseline / sparse_lasso / lowrank_svd 场景下分别估计不同 M 下的 size。
3) Power：固定在最大 M 下比较不同时间序列变化幅度对应的 power，并检查 power 是否随变化幅度提升。
4) 对同一配置做多 seed 重复，输出均值与 95% 分位数区间。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

from simulation import VARDataGenerator, ChowTest, ChowBootstrapInference
from simulation.parallel import run_task_map
from sparse_var import SparseBootstrapInference
from lowrank_var import LowRankBootstrapInference

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 80) -> np.ndarray:
    current = phi.copy()
    attempts = 0
    while not VARDataGenerator.check_stationarity(current) and attempts < max_attempts:
        current = current * shrink
        attempts += 1
    if not VARDataGenerator.check_stationarity(current):
        raise ValueError("无法构造平稳的 Phi2")
    return current


def mean_ci95(values: List[float]) -> Tuple[float, float, float, float, int]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_low = float(np.quantile(arr, 0.025))
    ci_high = float(np.quantile(arr, 0.975))
    return mean, ci_low, ci_high, std, n


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


STAGE_LABELS = {
    "baseline_pvalue": "baseline p-value comparison",
    "baseline_validation": "baseline size + power sweep",
    "sparse_validation": "sparse_lasso size + power sweep",
    "lowrank_validation": "lowrank_svd size + power sweep",
}
STAGE_SEQUENCE = [
    "baseline_pvalue",
    "baseline_validation",
    "sparse_validation",
    "lowrank_validation",
]

DEFAULT_POWER_CHANGE_GRID = [0.04, 0.08, 0.12, 0.16, 0.20]


def _iteration_seed(base_seed: Optional[int], stream_id: int, iteration: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed + stream_id * 1_000_003 + iteration)


def _bootstrap_seed(iteration_seed: Optional[int]) -> Optional[int]:
    if iteration_seed is None:
        return None
    return int(iteration_seed + 97_531)


def _run_mc_tasks(worker, tasks: List[Tuple[Any, ...]], mc_workers: int) -> List[Dict[str, Any]]:
    return run_task_map(
        worker,
        tasks,
        n_jobs=max(1, mc_workers),
        verbose=False,
        progress_every=10,
        progress_label="Monte Carlo iteration",
    )


def _baseline_h0_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Sigma, t, B_bootstrap_baseline, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = ChowBootstrapInference(B=B_bootstrap_baseline, seed=_bootstrap_seed(iteration_seed))
    try:
        Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha)
        return {
            "success": True,
            "p_values": {
                "baseline_chow_asym_f": float(result["f_asymptotic_p_value"]),
                "baseline_chow_asym_chi2": float(result["chi2_asymptotic_p_value"]),
                "baseline_chow_bootstrap_f": float(result["bootstrap_f_p_value"]),
                "baseline_chow_bootstrap_lr": float(result["bootstrap_lr_p_value"]),
            },
        }
    except Exception:
        return {"success": False}


def _baseline_h1_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_baseline, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = ChowBootstrapInference(B=B_bootstrap_baseline, seed=_bootstrap_seed(iteration_seed))
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha)
        return {
            "success": True,
            "p_values": {
                "baseline_chow_asym_f": float(result["f_asymptotic_p_value"]),
                "baseline_chow_asym_chi2": float(result["chi2_asymptotic_p_value"]),
                "baseline_chow_bootstrap_f": float(result["bootstrap_f_p_value"]),
                "baseline_chow_bootstrap_lr": float(result["bootstrap_lr_p_value"]),
            },
        }
    except Exception:
        return {"success": False}


def _sparse_h0_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Sigma, t, B_bootstrap_hd, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = SparseBootstrapInference(B=B_bootstrap_hd, seed=_bootstrap_seed(iteration_seed), estimator_type="lasso")
    try:
        Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
        result = bootstrap.test(Y, p, t, alpha=alpha)
        return {"success": True, "p_value": float(result["p_value"])}
    except Exception:
        return {"success": False}


def _sparse_h1_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_hd, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = SparseBootstrapInference(B=B_bootstrap_hd, seed=_bootstrap_seed(iteration_seed), estimator_type="lasso")
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
        result = bootstrap.test(Y, p, t, alpha=alpha)
        return {"success": True, "p_value": float(result["p_value"])}
    except Exception:
        return {"success": False}


def _lowrank_h0_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Sigma, t, B_bootstrap_hd, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = LowRankBootstrapInference(B=B_bootstrap_hd, seed=_bootstrap_seed(iteration_seed), method="svd", rank=2)
    try:
        Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
        result = bootstrap.test(Y, p, t, alpha=alpha)
        return {"success": True, "p_value": float(result["p_value"])}
    except Exception:
        return {"success": False}


def _lowrank_h1_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    iteration_seed, T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_hd, alpha = task
    generator = VARDataGenerator(seed=iteration_seed)
    bootstrap = LowRankBootstrapInference(B=B_bootstrap_hd, seed=_bootstrap_seed(iteration_seed), method="svd", rank=2)
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
        result = bootstrap.test(Y, p, t, alpha=alpha)
        return {"success": True, "p_value": float(result["p_value"])}
    except Exception:
        return {"success": False}


def build_phi2_with_change(Phi1: np.ndarray, change_scale: float) -> Tuple[np.ndarray, Dict[str, float]]:
    Phi2 = ensure_stationary(Phi1 + change_scale * np.ones_like(Phi1))
    delta = Phi2 - Phi1
    return Phi2, {
        "change_scale": float(change_scale),
        "delta_mean_abs": float(np.mean(np.abs(delta))),
        "delta_max_abs": float(np.max(np.abs(delta))),
        "delta_fro": float(np.linalg.norm(delta)),
    }


def append_progress(progress_output: Optional[str], seed: int, stage: str) -> None:
    if not progress_output:
        return
    with open(progress_output, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seed": seed, "stage": stage}, ensure_ascii=False) + "\n")
        f.flush()


def read_progress_events(progress_output: str, offset: int) -> Tuple[List[Dict[str, Any]], int]:
    if not os.path.exists(progress_output):
        return [], offset
    with open(progress_output, "r", encoding="utf-8") as f:
        f.seek(offset)
        lines = f.readlines()
        new_offset = f.tell()
    events = [json.loads(line) for line in lines if line.strip()]
    return events, new_offset


def log_message(message: str, log_path: Optional[str] = None) -> None:
    print(message, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()


def make_run_dir(tag: str) -> Tuple[str, str, str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    base_dir = os.path.join("results", "next_stage_runs")
    run_name = f"{stamp}{suffix}"
    run_dir = os.path.join(base_dir, run_name)
    worker_log_dir = os.path.join(run_dir, "worker_logs")
    state_dir = os.path.join(run_dir, "state")
    os.makedirs(worker_log_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    return stamp, run_dir, worker_log_dir, state_dir


def run_baseline_pvalue_comparison(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    alpha: float,
    B_values: List[int],
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi = generator.generate_stationary_phi(N, p, scale=0.2)
    Y = generator.generate_var_series(T, N, p, Phi, Sigma)

    chow = ChowTest()
    asym = chow.compute_at_point(Y, p, t)
    chi2_p = asym["chi2_p_value"]

    rows = []
    for B in B_values:
        bootstrap = ChowBootstrapInference(B=B, seed=seed)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha, verbose=False)
        p_boot_lr = result["bootstrap_lr_p_value"]
        rows.append(
            {
                "seed": seed,
                "B": B,
                "chi2_p_value": float(chi2_p),
                "f_asymptotic_p_value": float(result["f_asymptotic_p_value"]),
                "bootstrap_lr_p_value": float(p_boot_lr),
                "bootstrap_f_p_value": float(result["bootstrap_f_p_value"]),
                "abs_diff_boot_lr_vs_chi2": float(abs(p_boot_lr - chi2_p)),
                "B_effective": int(result["B_effective"]),
            }
        )
    return rows


def run_baseline_size(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_baseline: int,
    alpha: float,
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.2)
    rows = []
    inference_keys = [
        ("baseline_chow_asym_f", "f_asymptotic_p_value"),
        ("baseline_chow_asym_chi2", "chi2_asymptotic_p_value"),
        ("baseline_chow_bootstrap_f", "bootstrap_f_p_value"),
        ("baseline_chow_bootstrap_lr", "bootstrap_lr_p_value"),
    ]
    for stream_id, M in enumerate(M_values, start=1):
        h0_counts = {name: 0 for name, _ in inference_keys}
        tasks = [
            (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Sigma, t, B_bootstrap_baseline, alpha)
            for iteration in range(M)
        ]
        results = _run_mc_tasks(_baseline_h0_worker, tasks, mc_workers)
        succ_h0 = 0
        for result in results:
            if not result.get("success"):
                continue
            succ_h0 += 1
            p_values = result["p_values"]
            for name, _ in inference_keys:
                p_value = p_values[name]
                if not np.isnan(p_value) and p_value <= alpha:
                    h0_counts[name] += 1

        for name, _ in inference_keys:
            rows.append(
                {
                    "seed": seed,
                    "scenario": "baseline",
                    "model": name,
                    "M": M,
                    "size": h0_counts[name] / succ_h0 if succ_h0 > 0 else np.nan,
                    "M_effective": succ_h0,
                }
            )

    return rows


def run_baseline_power(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_baseline: int,
    alpha: float,
    change_scales: List[float],
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.2)

    rows = []
    inference_keys = [
        ("baseline_chow_asym_f", "f_asymptotic_p_value"),
        ("baseline_chow_asym_chi2", "chi2_asymptotic_p_value"),
        ("baseline_chow_bootstrap_f", "bootstrap_f_p_value"),
        ("baseline_chow_bootstrap_lr", "bootstrap_lr_p_value"),
    ]
    for change_idx, change_scale in enumerate(change_scales, start=1):
        Phi2, delta_info = build_phi2_with_change(Phi1, change_scale)
        for m_idx, M in enumerate(M_values, start=1):
            h1_counts = {name: 0 for name, _ in inference_keys}
            stream_id = 100 + change_idx * 100 + m_idx
            tasks = [
                (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_baseline, alpha)
                for iteration in range(M)
            ]
            results = _run_mc_tasks(_baseline_h1_worker, tasks, mc_workers)
            succ_h1 = 0
            for result in results:
                if not result.get("success"):
                    continue
                succ_h1 += 1
                p_values = result["p_values"]
                for name, _ in inference_keys:
                    p_value = p_values[name]
                    if not np.isnan(p_value) and p_value <= alpha:
                        h1_counts[name] += 1

            for name, _ in inference_keys:
                rows.append(
                    {
                        "seed": seed,
                        "scenario": "baseline",
                        "model": name,
                        "M": M,
                        "change_scale": delta_info["change_scale"],
                        "delta_mean_abs": delta_info["delta_mean_abs"],
                        "delta_max_abs": delta_info["delta_max_abs"],
                        "delta_fro": delta_info["delta_fro"],
                        "power": h1_counts[name] / succ_h1 if succ_h1 > 0 else np.nan,
                        "M_effective": succ_h1,
                    }
                )

    return rows


def run_sparse_size(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, sparsity=0.2, scale=0.15)

    rows = []
    for stream_id, M in enumerate(M_values, start=201):
        tasks = [
            (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Sigma, t, B_bootstrap_hd, alpha)
            for iteration in range(M)
        ]
        results = _run_mc_tasks(_sparse_h0_worker, tasks, mc_workers)
        reject_h0 = 0
        succ_h0 = 0
        for result in results:
            if not result.get("success"):
                continue
            succ_h0 += 1
            pval = result["p_value"]
            if not np.isnan(pval) and pval <= alpha:
                reject_h0 += 1

        rows.append(
            {
                "seed": seed,
                "scenario": "sparse",
                "model": "sparse_lasso",
                "M": M,
                "size": reject_h0 / succ_h0 if succ_h0 > 0 else np.nan,
                "M_effective": succ_h0,
            }
        )
    return rows


def run_sparse_power(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
    change_scales: List[float],
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, sparsity=0.2, scale=0.15)

    rows = []
    for change_idx, change_scale in enumerate(change_scales, start=1):
        Phi2, delta_info = build_phi2_with_change(Phi1, change_scale)
        for m_idx, M in enumerate(M_values, start=1):
            stream_id = 300 + change_idx * 100 + m_idx
            tasks = [
                (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_hd, alpha)
                for iteration in range(M)
            ]
            results = _run_mc_tasks(_sparse_h1_worker, tasks, mc_workers)
            reject_h1 = 0
            succ_h1 = 0
            for result in results:
                if not result.get("success"):
                    continue
                succ_h1 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h1 += 1

            rows.append(
                {
                    "seed": seed,
                    "scenario": "sparse",
                    "model": "sparse_lasso",
                    "M": M,
                    "change_scale": delta_info["change_scale"],
                    "delta_mean_abs": delta_info["delta_mean_abs"],
                    "delta_max_abs": delta_info["delta_max_abs"],
                    "delta_fro": delta_info["delta_fro"],
                    "power": reject_h1 / succ_h1 if succ_h1 > 0 else np.nan,
                    "M_effective": succ_h1,
                }
            )
    return rows


def run_lowrank_size(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_lowrank_phi(N, p, rank=2, scale=0.15)

    rows = []
    for stream_id, M in enumerate(M_values, start=401):
        tasks = [
            (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Sigma, t, B_bootstrap_hd, alpha)
            for iteration in range(M)
        ]
        results = _run_mc_tasks(_lowrank_h0_worker, tasks, mc_workers)
        reject_h0 = 0
        succ_h0 = 0
        for result in results:
            if not result.get("success"):
                continue
            succ_h0 += 1
            pval = result["p_value"]
            if not np.isnan(pval) and pval <= alpha:
                reject_h0 += 1

        rows.append(
            {
                "seed": seed,
                "scenario": "lowrank",
                "model": "lowrank_svd",
                "M": M,
                "size": reject_h0 / succ_h0 if succ_h0 > 0 else np.nan,
                "M_effective": succ_h0,
            }
        )
    return rows


def run_lowrank_power(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
    change_scales: List[float],
    mc_workers: int = 1,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_lowrank_phi(N, p, rank=2, scale=0.15)

    rows = []
    for change_idx, change_scale in enumerate(change_scales, start=1):
        Phi2, delta_info = build_phi2_with_change(Phi1, change_scale)
        for m_idx, M in enumerate(M_values, start=1):
            stream_id = 500 + change_idx * 100 + m_idx
            tasks = [
                (_iteration_seed(seed, stream_id, iteration), T, N, p, Phi1, Phi2, Sigma, t, B_bootstrap_hd, alpha)
                for iteration in range(M)
            ]
            results = _run_mc_tasks(_lowrank_h1_worker, tasks, mc_workers)
            reject_h1 = 0
            succ_h1 = 0
            for result in results:
                if not result.get("success"):
                    continue
                succ_h1 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h1 += 1

            rows.append(
                {
                    "seed": seed,
                    "scenario": "lowrank",
                    "model": "lowrank_svd",
                    "M": M,
                    "change_scale": delta_info["change_scale"],
                    "delta_mean_abs": delta_info["delta_mean_abs"],
                    "delta_max_abs": delta_info["delta_max_abs"],
                    "delta_fro": delta_info["delta_fro"],
                    "power": reject_h1 / succ_h1 if succ_h1 > 0 else np.nan,
                    "M_effective": succ_h1,
                }
            )
    return rows


def aggregate_baseline_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["B"]), []).append(row)

    out = []
    for B in sorted(grouped.keys()):
        g = grouped[B]
        record = {"B": B, "n_seed": len(g)}
        for key in [
            "chi2_p_value",
            "f_asymptotic_p_value",
            "bootstrap_lr_p_value",
            "bootstrap_f_p_value",
            "abs_diff_boot_lr_vs_chi2",
        ]:
            m, lo, hi, sd, n = mean_ci95([float(x[key]) for x in g])
            record[f"{key}_mean"] = m
            record[f"{key}_ci95_low"] = lo
            record[f"{key}_ci95_high"] = hi
            record[f"{key}_sd"] = sd
            record[f"{key}_n"] = n
        out.append(record)
    return out


def aggregate_size_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["model"]), int(row["M"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (scenario, model, M) in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        g = grouped[(scenario, model, M)]
        size_vals = [float(x["size"]) for x in g]
        mean, low, high, sd, n = mean_ci95(size_vals)
        out.append(
            {
                "scenario": scenario,
                "model": model,
                "M": M,
                "n_seed": len(g),
                "size_mean": mean,
                "size_ci95_low": low,
                "size_ci95_high": high,
                "size_sd": sd,
                "size_n": n,
                "M_effective_avg": float(np.mean([x["M_effective"] for x in g])),
            }
        )
    return out


def aggregate_power_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float, int], List[Dict[str, Any]]] = {}
    for row in rows:
        change_scale = round(float(row["change_scale"]), 10)
        key = (str(row["scenario"]), str(row["model"]), change_scale, int(row["M"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (scenario, model, change_scale, M) in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2], x[3])):
        g = grouped[(scenario, model, change_scale, M)]
        power_vals = [float(x["power"]) for x in g]
        mean, low, high, sd, n = mean_ci95(power_vals)
        delta_fro_mean = float(np.mean([float(x["delta_fro"]) for x in g]))
        delta_max_abs_mean = float(np.mean([float(x["delta_max_abs"]) for x in g]))
        out.append(
            {
                "scenario": scenario,
                "model": model,
                "change_scale": change_scale,
                "M": M,
                "n_seed": len(g),
                "power_mean": mean,
                "power_ci95_low": low,
                "power_ci95_high": high,
                "power_sd": sd,
                "power_n": n,
                "delta_fro_mean": delta_fro_mean,
                "delta_max_abs_mean": delta_max_abs_mean,
                "M_effective_avg": float(np.mean([x["M_effective"] for x in g])),
            }
        )
    return out


def plot_baseline_ci(rows_agg: List[Dict[str, Any]], output_path: str) -> None:
    if not HAS_MATPLOTLIB or not rows_agg:
        return
    B = [r["B"] for r in rows_agg]
    chi2 = [r["chi2_p_value_mean"] for r in rows_agg]
    boot_lr = [r["bootstrap_lr_p_value_mean"] for r in rows_agg]

    plt.figure(figsize=(8, 5))
    plt.plot(B, chi2, marker="o", linewidth=2, label="chi2 p (LR) mean")
    plt.plot(B, boot_lr, marker="s", linewidth=2, label="bootstrap LR p mean")
    plt.xlabel("Bootstrap repetitions (B)")
    plt.ylabel("p-value")
    plt.title("Baseline p-values vs B (mean across seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_size_ci(rows_agg: List[Dict[str, Any]], output_path: str) -> None:
    if not HAS_MATPLOTLIB or not rows_agg:
        return

    labels = sorted(set(f"{r['scenario']}::{r['model']}" for r in rows_agg))
    plt.figure(figsize=(9, 5))

    for label in labels:
        scenario, model = label.split("::", 1)
        sub = sorted([r for r in rows_agg if r["scenario"] == scenario and r["model"] == model], key=lambda x: x["M"])
        M = [r["M"] for r in sub]
        y = [r["size_mean"] for r in sub]
        ylo = [r["size_ci95_low"] for r in sub]
        yhi = [r["size_ci95_high"] for r in sub]
        err_lower = np.array(y) - np.array(ylo)
        err_upper = np.array(yhi) - np.array(y)
        plt.errorbar(M, y, yerr=[err_lower, err_upper], marker="o", capsize=4, linewidth=1.5, label=label)

    plt.xlabel("Monte Carlo repetitions (M)")
    plt.ylabel("size")
    plt.title("Size vs M (mean with 95% quantile CI across seeds)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_power_vs_change_ci(rows_agg: List[Dict[str, Any]], output_path: str) -> None:
    if not HAS_MATPLOTLIB or not rows_agg:
        return

    labels = sorted(set(f"{r['scenario']}::{r['model']}" for r in rows_agg))
    m_value = int(rows_agg[0]["M"])

    plt.figure(figsize=(9, 5))
    for label in labels:
        scenario, model = label.split("::", 1)
        sub = sorted(
            [r for r in rows_agg if r["scenario"] == scenario and r["model"] == model],
            key=lambda x: float(x["change_scale"]),
        )
        if not sub:
            continue
        x = [r["change_scale"] for r in sub]
        y = [r["power_mean"] for r in sub]
        ylo = [r["power_ci95_low"] for r in sub]
        yhi = [r["power_ci95_high"] for r in sub]
        err_lower = np.array(y) - np.array(ylo)
        err_upper = np.array(yhi) - np.array(y)
        plt.errorbar(x, y, yerr=[err_lower, err_upper], marker="o", capsize=3, linewidth=1.5, label=label)

    plt.xlabel("change scale")
    plt.ylabel("power")
    plt.title(f"Power vs change scale (fixed M={m_value})")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_seed_bundle(
    seed: int,
    alpha: float,
    B_grid: List[int],
    M_grid: List[int],
    power_change_grid: List[float],
    power_M: int,
    B_mc_baseline: int,
    B_mc_highdim: int,
    mc_workers: int,
    p: int,
    config_dims: Dict[str, Dict[str, int]],
    progress_output: Optional[str] = None,
    print_progress: bool = False,
) -> Dict[str, Any]:
    generator = VARDataGenerator(seed=seed)

    baseline_rows = run_baseline_pvalue_comparison(
        generator=generator,
        seed=seed,
        p=p,
        T=config_dims["baseline"]["T"],
        N=config_dims["baseline"]["N"],
        t=config_dims["baseline"]["t"],
        alpha=alpha,
        B_values=B_grid,
    )
    append_progress(progress_output, seed, "baseline_pvalue")
    if print_progress:
        print(f"[Seed {seed}] completed {STAGE_LABELS['baseline_pvalue']}", flush=True)

    size_rows: List[Dict[str, Any]] = []
    power_rows: List[Dict[str, Any]] = []

    size_rows.extend(
        run_baseline_size(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["baseline"]["T"],
            N=config_dims["baseline"]["N"],
            t=config_dims["baseline"]["t"],
            M_values=M_grid,
            B_bootstrap_baseline=B_mc_baseline,
            alpha=alpha,
            mc_workers=mc_workers,
        )
    )
    append_progress(progress_output, seed, "baseline_validation")
    if print_progress:
        print(f"[Seed {seed}] completed {STAGE_LABELS['baseline_validation']}", flush=True)

    power_rows.extend(
        run_baseline_power(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["baseline"]["T"],
            N=config_dims["baseline"]["N"],
            t=config_dims["baseline"]["t"],
            M_values=[power_M],
            B_bootstrap_baseline=B_mc_baseline,
            alpha=alpha,
            change_scales=power_change_grid,
            mc_workers=mc_workers,
        )
    )

    size_rows.extend(
        run_sparse_size(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["sparse"]["T"],
            N=config_dims["sparse"]["N"],
            t=config_dims["sparse"]["t"],
            M_values=M_grid,
            B_bootstrap_hd=B_mc_highdim,
            alpha=alpha,
            mc_workers=mc_workers,
        )
    )
    append_progress(progress_output, seed, "sparse_validation")
    if print_progress:
        print(f"[Seed {seed}] completed {STAGE_LABELS['sparse_validation']}", flush=True)

    power_rows.extend(
        run_sparse_power(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["sparse"]["T"],
            N=config_dims["sparse"]["N"],
            t=config_dims["sparse"]["t"],
            M_values=[power_M],
            B_bootstrap_hd=B_mc_highdim,
            alpha=alpha,
            change_scales=power_change_grid,
            mc_workers=mc_workers,
        )
    )

    size_rows.extend(
        run_lowrank_size(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["lowrank"]["T"],
            N=config_dims["lowrank"]["N"],
            t=config_dims["lowrank"]["t"],
            M_values=M_grid,
            B_bootstrap_hd=B_mc_highdim,
            alpha=alpha,
            mc_workers=mc_workers,
        )
    )
    append_progress(progress_output, seed, "lowrank_validation")
    if print_progress:
        print(f"[Seed {seed}] completed {STAGE_LABELS['lowrank_validation']}", flush=True)

    power_rows.extend(
        run_lowrank_power(
            generator=generator,
            seed=seed,
            p=p,
            T=config_dims["lowrank"]["T"],
            N=config_dims["lowrank"]["N"],
            t=config_dims["lowrank"]["t"],
            M_values=[power_M],
            B_bootstrap_hd=B_mc_highdim,
            alpha=alpha,
            change_scales=power_change_grid,
            mc_workers=mc_workers,
        )
    )

    return {
        "seed": seed,
        "baseline_rows": baseline_rows,
        "size_rows": size_rows,
        "power_rows": power_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-stage large-scale experiment with p=3 and multi-seed CI")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 7])
    parser.add_argument("--B-grid", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--M-grid", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--power-change-grid", type=float, nargs="+", default=DEFAULT_POWER_CHANGE_GRID)
    parser.add_argument("--B-mc-baseline", type=int, default=50)
    parser.add_argument("--B-mc-highdim", type=int, default=30)
    parser.add_argument("--seed-workers", type=int, default=0, help="并行运行的 seed worker 数；0 表示按 CPU 核数自动选择")
    parser.add_argument("--mc-workers", type=int, default=0, help="每个 seed 内部 Monte Carlo worker 数；0 表示按 CPU/seed_workers 自动分配")
    parser.add_argument("--single-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bundle-output", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--progress-output", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--tag", type=str, default="next_stage_p3")
    args = parser.parse_args()

    p = 3
    power_M = max(args.M_grid)
    config_dims = {
        "baseline": {"N": 2, "T": 150, "t": 75},
        "sparse": {"N": 3, "T": 150, "t": 75},
        "lowrank": {"N": 4, "T": 150, "t": 75},
    }

    if args.single_seed is not None:
        mc_workers = args.mc_workers if args.mc_workers > 0 else max(1, os.cpu_count() or 1)
        seed_result = run_seed_bundle(
            seed=args.single_seed,
            alpha=args.alpha,
            B_grid=args.B_grid,
            M_grid=args.M_grid,
            power_change_grid=args.power_change_grid,
            power_M=power_M,
            B_mc_baseline=args.B_mc_baseline,
            B_mc_highdim=args.B_mc_highdim,
            mc_workers=mc_workers,
            p=p,
            config_dims=config_dims,
            progress_output=args.progress_output or None,
            print_progress=True,
        )
        if not args.bundle_output:
            raise ValueError("--bundle-output is required when --single-seed is set")
        with open(args.bundle_output, "w", encoding="utf-8") as f:
            json.dump(seed_result, f, ensure_ascii=False)
        return

    stamp, run_dir, worker_log_dir, state_dir = make_run_dir(args.tag)
    progress_log = os.path.join(run_dir, "progress.log")
    error_log = os.path.join(run_dir, "errors.log")
    run_meta_json = os.path.join(run_dir, "run_meta.json")

    seed_workers = args.seed_workers if args.seed_workers > 0 else min(len(args.seeds), os.cpu_count() or 1)
    seed_workers = max(1, min(seed_workers, len(args.seeds)))
    mc_workers = args.mc_workers if args.mc_workers > 0 else max(1, (os.cpu_count() or 1) // seed_workers)

    log_message(f"Running {len(args.seeds)} seeds with seed_workers={seed_workers}, mc_workers={mc_workers}...", progress_log)

    seed_results: List[Dict[str, Any]] = []
    if seed_workers == 1:
        for seed in args.seeds:
            log_message(f"[Seed {seed}] running...", progress_log)
            seed_results.append(
                run_seed_bundle(
                    seed=seed,
                    alpha=args.alpha,
                    B_grid=args.B_grid,
                    M_grid=args.M_grid,
                    power_change_grid=args.power_change_grid,
                    power_M=power_M,
                    B_mc_baseline=args.B_mc_baseline,
                    B_mc_highdim=args.B_mc_highdim,
                    mc_workers=mc_workers,
                    p=p,
                    config_dims=config_dims,
                    print_progress=True,
                )
            )
            log_message(f"[Seed {seed}] done. [Seed progress 4/4, Total stage progress {len(seed_results) * len(STAGE_SEQUENCE)}/{len(args.seeds) * len(STAGE_SEQUENCE)}]", progress_log)
    else:
        pending_seeds = list(args.seeds)
        running_jobs: List[Tuple[int, subprocess.Popen[str], str, str, str]] = []
        progress_offsets: Dict[int, int] = {}
        stage_done_count = 0
        seed_done_count = 0
        total_stage_count = len(args.seeds) * len(STAGE_SEQUENCE)

        while pending_seeds or running_jobs:
            while pending_seeds and len(running_jobs) < seed_workers:
                seed = pending_seeds.pop(0)
                bundle_path = os.path.join(state_dir, f"seed_{seed}.json")
                log_path = os.path.join(worker_log_dir, f"seed_{seed}.log")
                progress_path = os.path.join(state_dir, f"seed_{seed}.progress")
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--alpha",
                    str(args.alpha),
                    "--B-grid",
                    *[str(x) for x in args.B_grid],
                    "--M-grid",
                    *[str(x) for x in args.M_grid],
                    "--power-change-grid",
                    *[str(x) for x in args.power_change_grid],
                    "--B-mc-baseline",
                    str(args.B_mc_baseline),
                    "--B-mc-highdim",
                    str(args.B_mc_highdim),
                    "--mc-workers",
                    str(mc_workers),
                    "--seed-workers",
                    "1",
                    "--single-seed",
                    str(seed),
                    "--bundle-output",
                    bundle_path,
                    "--progress-output",
                    progress_path,
                    "--tag",
                    args.tag,
                ]
                env = os.environ.copy()
                env.setdefault("MPLCONFIGDIR", os.path.join(state_dir, f"mpl_{seed}"))
                env.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")
                log_file = open(log_path, "w", encoding="utf-8")
                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True)
                running_jobs.append((seed, process, bundle_path, log_path, progress_path))
                progress_offsets[seed] = 0
                log_message(f"[Seed {seed}] submitted.", progress_log)

            time.sleep(1.0)
            next_running: List[Tuple[int, subprocess.Popen[str], str, str, str]] = []
            for seed, process, bundle_path, log_path, progress_path in running_jobs:
                events, progress_offsets[seed] = read_progress_events(progress_path, progress_offsets.get(seed, 0))
                for event in events:
                    stage_done_count += 1
                    stage_name = STAGE_LABELS.get(event["stage"], event["stage"])
                    log_message(f"[Progress {stage_done_count}/{total_stage_count}] [Seed {seed}] completed {stage_name}", progress_log)

                return_code = process.poll()
                if return_code is None:
                    next_running.append((seed, process, bundle_path, log_path, progress_path))
                    continue
                if return_code != 0:
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_text = f.read()
                    failure_message = f"[Seed {seed}] FAILED with exit code {return_code}"
                    log_message(failure_message, progress_log)
                    with open(error_log, "a", encoding="utf-8") as f:
                        f.write(failure_message + "\n")
                        f.write(log_text + "\n")
                    log_message(f"[Seed {seed}] skipped due to failure. Continuing with remaining seeds.", progress_log)
                    continue
                final_events, progress_offsets[seed] = read_progress_events(progress_path, progress_offsets.get(seed, 0))
                for event in final_events:
                    stage_done_count += 1
                    stage_name = STAGE_LABELS.get(event["stage"], event["stage"])
                    log_message(f"[Progress {stage_done_count}/{total_stage_count}] [Seed {seed}] completed {stage_name}", progress_log)
                with open(bundle_path, "r", encoding="utf-8") as f:
                    seed_results.append(json.load(f))
                seed_done_count += 1
                log_message(f"[Seed {seed}] done. [Seed progress 4/4, Total seed progress {seed_done_count}/{len(args.seeds)}]", progress_log)
            running_jobs = next_running


    seed_results.sort(key=lambda item: item["seed"])
    baseline_rows_raw = [row for item in seed_results for row in item["baseline_rows"]]
    size_rows_raw = [row for item in seed_results for row in item["size_rows"]]
    power_rows_raw = [row for item in seed_results for row in item["power_rows"]]

    baseline_rows_agg = aggregate_baseline_rows(baseline_rows_raw)
    size_rows_agg = aggregate_size_rows(size_rows_raw)
    power_rows_agg = aggregate_power_rows(power_rows_raw)

    suffix = f"_{args.tag}" if args.tag else ""

    baseline_raw_csv = os.path.join(run_dir, f"next_stage_baseline_raw_{stamp}{suffix}.csv")
    baseline_agg_csv = os.path.join(run_dir, f"next_stage_baseline_agg_{stamp}{suffix}.csv")
    size_raw_csv = os.path.join(run_dir, f"next_stage_size_raw_{stamp}{suffix}.csv")
    size_agg_csv = os.path.join(run_dir, f"next_stage_size_agg_{stamp}{suffix}.csv")
    power_raw_csv = os.path.join(run_dir, f"next_stage_power_raw_{stamp}{suffix}.csv")
    power_agg_csv = os.path.join(run_dir, f"next_stage_power_agg_{stamp}{suffix}.csv")

    write_csv(baseline_raw_csv, baseline_rows_raw)
    write_csv(baseline_agg_csv, baseline_rows_agg)
    write_csv(size_raw_csv, size_rows_raw)
    write_csv(size_agg_csv, size_rows_agg)
    write_csv(power_raw_csv, power_rows_raw)
    write_csv(power_agg_csv, power_rows_agg)

    baseline_png = os.path.join(run_dir, f"next_stage_baseline_pvalues_vs_B_{stamp}{suffix}.png")
    size_png = os.path.join(run_dir, f"next_stage_model_size_vs_M_{stamp}{suffix}.png")
    power_png = os.path.join(run_dir, f"next_stage_model_power_vs_change_{stamp}{suffix}.png")

    if HAS_MATPLOTLIB:
        plot_baseline_ci(baseline_rows_agg, baseline_png)
        plot_size_ci(size_rows_agg, output_path=size_png)
        plot_power_vs_change_ci(power_rows_agg, output_path=power_png)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "alpha": args.alpha,
            "seeds": args.seeds,
            "B_grid": args.B_grid,
            "M_grid": args.M_grid,
            "power_change_grid": args.power_change_grid,
            "power_M": power_M,
            "B_mc_baseline": args.B_mc_baseline,
            "B_mc_highdim": args.B_mc_highdim,
            "seed_workers": seed_workers,
            "mc_workers": mc_workers,
            "run_dir": run_dir,
            "state_dir": state_dir,
            "p": p,
            "dimensions": config_dims,
        },
        "baseline_agg_rows": baseline_rows_agg,
        "size_agg_rows": size_rows_agg,
        "power_agg_rows": power_rows_agg,
        "outputs": {
            "baseline_raw_csv": baseline_raw_csv,
            "baseline_agg_csv": baseline_agg_csv,
            "size_raw_csv": size_raw_csv,
            "size_agg_csv": size_agg_csv,
            "power_raw_csv": power_raw_csv,
            "power_agg_csv": power_agg_csv,
            "baseline_png": baseline_png if HAS_MATPLOTLIB else None,
            "size_png": size_png if HAS_MATPLOTLIB else None,
            "power_png": power_png if HAS_MATPLOTLIB else None,
            "progress_log": progress_log,
            "error_log": error_log,
            "worker_log_dir": worker_log_dir,
            "state_dir": state_dir,
        },
    }

    with open(run_meta_json, "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "progress_log": progress_log, "error_log": error_log, "worker_log_dir": worker_log_dir, "state_dir": state_dir}, f, ensure_ascii=False, indent=2)

    summary_json = os.path.join(run_dir, f"next_stage_summary_{stamp}{suffix}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_md = os.path.join(run_dir, f"下一阶段大规模试验报告_{stamp}{suffix}.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# 下一阶段大规模试验报告\n\n")
        f.write("## 1. 配置\n")
        f.write(f"- 时间：{summary['timestamp']}\n")
        f.write(f"- p：{p}（所有场景统一）\n")
        f.write(f"- seeds：{args.seeds}\n")
        f.write(f"- M-grid：{args.M_grid}\n")
        f.write(f"- power 变化幅度 grid：{args.power_change_grid}\n")
        f.write(f"- power 固定 M_max：{power_M}\n")
        f.write(f"- B-grid：{args.B_grid}\n")
        f.write(f"- seed_workers：{seed_workers}\n")
        f.write(f"- mc_workers：{mc_workers}\n")
        f.write(f"- Baseline B_mc：{args.B_mc_baseline}\n")
        f.write(f"- 高维 B_mc：{args.B_mc_highdim}\n")
        f.write("- 高维检验口径：稀疏场景使用 `SparseBootstrapInference` 的 LR+bootstrap；低秩场景使用 `LowRankBootstrapInference` 的 LR+bootstrap。\n")
        f.write("- 场景维度：\n")
        f.write(f"  - baseline: {config_dims['baseline']}\n")
        f.write(f"  - sparse: {config_dims['sparse']}\n")
        f.write(f"  - lowrank: {config_dims['lowrank']}\n\n")

        f.write("## 2. Baseline p值对比（跨seed均值 + 95%分位数CI）\n")
        f.write("| B | chi2 p mean [95%CI] | bootstrap LR p mean [95%CI] | |boot LR-chi2| mean [95%CI] |\n")
        f.write("|---:|---:|---:|---:|\n")
        for r in baseline_rows_agg:
            f.write(
                f"| {r['B']} | "
                f"{r['chi2_p_value_mean']:.4f} [{r['chi2_p_value_ci95_low']:.4f}, {r['chi2_p_value_ci95_high']:.4f}] | "
                f"{r['bootstrap_lr_p_value_mean']:.4f} [{r['bootstrap_lr_p_value_ci95_low']:.4f}, {r['bootstrap_lr_p_value_ci95_high']:.4f}] | "
                f"{r['abs_diff_boot_lr_vs_chi2_mean']:.4f} [{r['abs_diff_boot_lr_vs_chi2_ci95_low']:.4f}, {r['abs_diff_boot_lr_vs_chi2_ci95_high']:.4f}] |\n"
            )
        f.write("\n")

        f.write("## 3. 不同情况下的 Size（跨seed均值）\n")
        f.write("| 场景 | 模型 | M | Size mean |\n")
        f.write("|---|---|---:|---:|\n")
        for r in sorted(size_rows_agg, key=lambda x: (x['scenario'], x['model'], x['M'])):
            f.write(
                f"| {r['scenario']} | {r['model']} | {r['M']} | "
                f"{r['size_mean']:.4f} |\n"
            )
        f.write("\n")

        f.write("## 4. 固定最大 M 下不同变化幅度的 Power（跨seed均值 + 95%分位数CI）\n")
        f.write("| 场景 | 模型 | change scale | 固定 M_max | Power mean [95%CI] |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in sorted(power_rows_agg, key=lambda x: (x['scenario'], x['model'], x['change_scale'], x['M'])):
            f.write(
                f"| {r['scenario']} | {r['model']} | {r['change_scale']:.2f} | {power_M} | "
                f"{r['power_mean']:.4f} [{r['power_ci95_low']:.4f}, {r['power_ci95_high']:.4f}] |\n"
            )
        f.write("\n")

        f.write("## 5. 数据背景说明\n")
        f.write("- 指标定义：Size = H0真时拒绝率；Power = H1真时拒绝率。\n")
        f.write("- 本轮固定 3 个 seed；size 主表默认只播报跨 seed 均值，不单独展示 95%CI。\n")
        f.write("- H0 数据：无断点序列；H1 数据：在给定 t 处设定参数突变。\n")
        f.write("- Power 部分额外固定至少 5 档变化幅度，并观察 power 是否随变化幅度增大而提升。\n")
        f.write("- 同一场景下 H0/H1 共享相同 `p,T,N,t` 口径。\n\n")

        f.write("## 6. 输出文件\n")
        f.write(f"- run dir: `{run_dir}`\n")
        f.write(f"- progress log: `{progress_log}`\n")
        f.write(f"- worker logs: `{worker_log_dir}`\n")
        f.write(f"- state dir: `{state_dir}`\n")
        f.write(f"- summary: `{summary_json}`\n")
        f.write(f"- baseline raw: `{baseline_raw_csv}`\n")
        f.write(f"- baseline agg: `{baseline_agg_csv}`\n")
        f.write(f"- size raw: `{size_raw_csv}`\n")
        f.write(f"- size agg: `{size_agg_csv}`\n")
        f.write(f"- power raw: `{power_raw_csv}`\n")
        f.write(f"- power agg: `{power_agg_csv}`\n")
        if HAS_MATPLOTLIB:
            f.write(f"- baseline 图: `{baseline_png}`\n")
            f.write(f"- size 图: `{size_png}`\n")
            f.write(f"- power 图: `{power_png}`\n")

    log_message("Experiment completed.", progress_log)
    log_message(f"Run directory: {run_dir}", progress_log)
    log_message(f"Progress log: {progress_log}", progress_log)
    log_message("Outputs:", progress_log)
    log_message(f"- {summary_json}", progress_log)
    log_message(f"- {report_md}", progress_log)
    log_message(f"- {baseline_raw_csv}", progress_log)
    log_message(f"- {baseline_agg_csv}", progress_log)
    log_message(f"- {size_raw_csv}", progress_log)
    log_message(f"- {size_agg_csv}", progress_log)
    log_message(f"- {power_raw_csv}", progress_log)
    log_message(f"- {power_agg_csv}", progress_log)
    if HAS_MATPLOTLIB:
        log_message(f"- {baseline_png}", progress_log)
        log_message(f"- {size_png}", progress_log)
        log_message(f"- {power_png}", progress_log)


if __name__ == "__main__":
    main()
