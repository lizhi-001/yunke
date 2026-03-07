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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

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


def ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 30) -> Tuple[np.ndarray, bool, int]:
    """Shrink coefficient matrix until stationary or attempts exhausted."""
    attempts = 0
    current = phi.copy()
    while not VARDataGenerator.check_stationarity(current) and attempts < max_attempts:
        current = current * shrink
        attempts += 1
    return current, VARDataGenerator.check_stationarity(current), attempts


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


def run_baseline(cfg: ExperimentConfig) -> Dict:
    generator = VARDataGenerator(seed=cfg.seed)

    N, T, p, t = 2, 100, 1, 50
    Sigma = np.eye(N) * 0.5
    phi = generator.generate_stationary_phi(N, p, scale=0.3)

    mc = MonteCarloSimulation(M=cfg.M, B=cfg.B, seed=cfg.seed, n_jobs=cfg.jobs)

    t0 = time.time()
    type1 = mc.evaluate_type1_error_at_point(N, T, p, phi, Sigma, t=t, alpha=cfg.alpha, verbose=False)
    type1_runtime = time.time() - t0

    power_curve = []
    for delta in cfg.deltas:
        phi2_raw = phi + delta * np.ones_like(phi)
        phi2, ok, shrinks = ensure_stationary(phi2_raw)
        if not ok:
            power_curve.append({
                "delta": float(delta),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_shrink",
            })
            continue

        t1 = time.time()
        result = mc.evaluate_power_at_point(
            N, T, p, phi, phi2, Sigma, break_point=t, t=t, alpha=cfg.alpha, verbose=False
        )
        runtime = time.time() - t1
        power_curve.append({
            "delta": float(delta),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        })

    return {
        "model": "baseline_ols",
        "parameters": {"N": N, "T": T, "p": p, "t": t, "M": cfg.M, "B": cfg.B, "alpha": cfg.alpha},
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


def run_sparse(cfg: ExperimentConfig) -> Dict:
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

    t0 = time.time()
    type1 = mc.evaluate_type1_error(N, T, p, phi, Sigma, t=t, test_alpha=cfg.alpha, verbose=False)
    type1_runtime = time.time() - t0

    power_curve = []
    for delta in cfg.deltas:
        phi2_raw = phi + delta * np.ones_like(phi)
        phi2, ok, shrinks = ensure_stationary(phi2_raw)
        if not ok:
            power_curve.append({
                "delta": float(delta),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_shrink",
            })
            continue

        t1 = time.time()
        result = mc.evaluate_power(
            N, T, p, phi, phi2, Sigma, break_point=t, t=t, test_alpha=cfg.alpha, verbose=False
        )
        runtime = time.time() - t1
        power_curve.append({
            "delta": float(delta),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        })

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


def run_lowrank(cfg: ExperimentConfig) -> Dict:
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

    t0 = time.time()
    type1 = mc.evaluate_type1_error(N, T, p, phi, Sigma, t=t, test_alpha=cfg.alpha, verbose=False)
    type1_runtime = time.time() - t0

    power_curve = []
    for delta in cfg.deltas:
        phi2_raw = phi + delta * np.ones_like(phi)
        phi2, ok, shrinks = ensure_stationary(phi2_raw)
        if not ok:
            power_curve.append({
                "delta": float(delta),
                "power": math.nan,
                "M_effective": 0,
                "rejections": 0,
                "skipped": True,
                "reason": "nonstationary_after_shrink",
            })
            continue

        t1 = time.time()
        result = mc.evaluate_power(
            N, T, p, phi, phi2, Sigma, break_point=t, t=t, test_alpha=cfg.alpha, verbose=False
        )
        runtime = time.time() - t1
        power_curve.append({
            "delta": float(delta),
            "power": float(result["power"]),
            "M_effective": int(result["M_effective"]),
            "rejections": int(result["rejections"]),
            "runtime_sec": float(runtime),
            "stationarity_shrinks": int(shrinks),
            "skipped": False,
            "pvalue_summary": summarize_pvalues(result["p_values"]),
        })

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

    lines: List[str] = []
    lines.append("# 大规模试验分析报告")
    lines.append("")
    lines.append(f"- 生成时间: {ts}")
    lines.append(f"- 方案配置: M={cfg['M']}, B={cfg['B']}, alpha={cfg['alpha']}, deltas={cfg['deltas']}")
    lines.append(f"- 总耗时(秒): {results['experiment_info']['total_runtime_sec']:.2f}")
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
    lines.append("| 模型 | Δ | Power | Rejections | M_effective | Runtime(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        for pt in results["models"][model_key]["power_curve"]:
            power_val = "nan" if np.isnan(pt["power"]) else f"{pt['power']:.4f}"
            lines.append(
                f"| {model_key} | {pt['delta']:.2f} | {power_val} | "
                f"{pt.get('rejections', 0)} | {pt.get('M_effective', 0)} | {pt.get('runtime_sec', math.nan):.2f} |"
            )
    lines.append("")

    lines.append("## 3. 结论摘要")
    lines.append("")
    for model_key in ("baseline_ols", "sparse_lasso", "lowrank_svd"):
        block = results["models"][model_key]
        t1 = block["type1_error"]["value"]
        valid_power = [pt["power"] for pt in block["power_curve"] if not np.isnan(pt["power"])]
        if len(valid_power) >= 2:
            monotone = all(valid_power[i] <= valid_power[i + 1] for i in range(len(valid_power) - 1))
        else:
            monotone = False
        lines.append(
            f"- {model_key}: type1_error={t1:.4f}; max_power={np.nanmax(valid_power) if valid_power else math.nan:.4f}; "
            f"power_monotone={monotone}"
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
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        M=args.M,
        B=args.B,
        alpha=args.alpha,
        deltas=tuple(args.deltas),
        seed=args.seed,
        jobs=max(1, args.jobs),
    )

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""

    print("=" * 72)
    print("Running large-scale experiment")
    print(
        f"Config: M={cfg.M}, B={cfg.B}, alpha={cfg.alpha}, "
        f"deltas={list(cfg.deltas)}, seed={cfg.seed}, jobs={cfg.jobs}"
    )
    print("=" * 72)

    all_start = time.time()

    model_results: Dict[str, Dict] = {}

    print("[1/3] baseline_ols ...")
    s1 = time.time()
    model_results["baseline_ols"] = run_baseline(cfg)
    print(f"  done in {time.time() - s1:.2f}s")

    print("[2/3] sparse_lasso ...")
    s2 = time.time()
    model_results["sparse_lasso"] = run_sparse(cfg)
    print(f"  done in {time.time() - s2:.2f}s")

    print("[3/3] lowrank_svd ...")
    s3 = time.time()
    model_results["lowrank_svd"] = run_lowrank(cfg)
    print(f"  done in {time.time() - s3:.2f}s")

    total_runtime = time.time() - all_start

    results = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Large-scale known-breakpoint LR+Bootstrap experiments",
            "config": asdict(cfg),
            "total_runtime_sec": float(total_runtime),
        },
        "models": model_results,
    }

    json_path = f"results/large_scale_experiment_{stamp}{tag_suffix}.json"
    csv_path = f"results/large_scale_plot_data_{stamp}{tag_suffix}.csv"
    md_path = f"results/大规模试验分析报告_{stamp}{tag_suffix}.md"

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

    print("\nOutputs:")
    print(f"- JSON: {json_path}")
    print(f"- CSV : {csv_path}")
    print(f"- MD  : {md_path}")
    print(f"Total runtime: {total_runtime:.2f}s")


if __name__ == "__main__":
    main()
