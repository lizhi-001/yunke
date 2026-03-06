"""
下一阶段大规模实验（p=3，多seed，均值+95%CI）。

目标：
1) Baseline：比较渐近 p 值与 Bootstrap p 值（随 B 变化）。
2) 验证：在 baseline / sparse_lasso / lowrank_svd 场景下估计 Type I Error 与 Power（随 M 变化）。
3) 对同一配置做多 seed 重复，输出均值与 95% 置信区间。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation import VARDataGenerator, ChowTest, ChowBootstrapInference
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


def run_baseline_validation(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_baseline: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    np.random.seed(seed)

    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.2)
    Phi2 = ensure_stationary(Phi1 + 0.10 * np.ones_like(Phi1))

    rows = []
    inference_keys = [
        ("baseline_chow_asym_f", "f_asymptotic_p_value"),
        ("baseline_chow_asym_chi2", "chi2_asymptotic_p_value"),
        ("baseline_chow_bootstrap_f", "bootstrap_f_p_value"),
        ("baseline_chow_bootstrap_lr", "bootstrap_lr_p_value"),
    ]

    for M in M_values:
        h0_counts = {name: 0 for name, _ in inference_keys}
        h1_counts = {name: 0 for name, _ in inference_keys}
        succ_h0 = 0
        succ_h1 = 0

        for _ in range(M):
            try:
                Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
                result = ChowBootstrapInference(B=B_bootstrap_baseline, seed=seed).test_at_point(
                    Y, p, t, alpha=alpha
                )
                succ_h0 += 1
                for name, key in inference_keys:
                    p_value = result[key]
                    if not np.isnan(p_value) and p_value <= alpha:
                        h0_counts[name] += 1
            except Exception:
                continue

        for _ in range(M):
            try:
                Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
                result = ChowBootstrapInference(B=B_bootstrap_baseline, seed=seed).test_at_point(
                    Y, p, t, alpha=alpha
                )
                succ_h1 += 1
                for name, key in inference_keys:
                    p_value = result[key]
                    if not np.isnan(p_value) and p_value <= alpha:
                        h1_counts[name] += 1
            except Exception:
                continue

        for name, _ in inference_keys:
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "M": M,
                    "type1_error": h0_counts[name] / succ_h0 if succ_h0 > 0 else np.nan,
                    "power": h1_counts[name] / succ_h1 if succ_h1 > 0 else np.nan,
                    "M_effective_type1": succ_h0,
                    "M_effective_power": succ_h1,
                }
            )

    return rows


def run_sparse_validation(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, sparsity=0.2, scale=0.15)
    Phi2 = ensure_stationary(Phi1 + 0.10 * np.ones_like(Phi1))

    rows = []
    for M in M_values:
        reject_h0 = 0
        reject_h1 = 0
        succ_h0 = 0
        succ_h1 = 0

        bootstrap = SparseBootstrapInference(B=B_bootstrap_hd, seed=seed, estimator_type="lasso")

        for _ in range(M):
            try:
                Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
                result = bootstrap.test(Y, p, t, alpha=alpha)
                succ_h0 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h0 += 1
            except Exception:
                continue

        for _ in range(M):
            try:
                Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
                result = bootstrap.test(Y, p, t, alpha=alpha)
                succ_h1 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h1 += 1
            except Exception:
                continue

        rows.append(
            {
                "seed": seed,
                "model": "sparse_lasso",
                "M": M,
                "type1_error": reject_h0 / succ_h0 if succ_h0 > 0 else np.nan,
                "power": reject_h1 / succ_h1 if succ_h1 > 0 else np.nan,
                "M_effective_type1": succ_h0,
                "M_effective_power": succ_h1,
            }
        )
    return rows


def run_lowrank_validation(
    generator: VARDataGenerator,
    seed: int,
    p: int,
    T: int,
    N: int,
    t: int,
    M_values: List[int],
    B_bootstrap_hd: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_lowrank_phi(N, p, rank=2, scale=0.15)
    Phi2 = ensure_stationary(Phi1 + 0.08 * np.ones_like(Phi1))

    rows = []
    for M in M_values:
        reject_h0 = 0
        reject_h1 = 0
        succ_h0 = 0
        succ_h1 = 0

        bootstrap = LowRankBootstrapInference(B=B_bootstrap_hd, seed=seed, method="svd", rank=2)

        for _ in range(M):
            try:
                Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
                result = bootstrap.test(Y, p, t, alpha=alpha)
                succ_h0 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h0 += 1
            except Exception:
                continue

        for _ in range(M):
            try:
                Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, t)
                result = bootstrap.test(Y, p, t, alpha=alpha)
                succ_h1 += 1
                pval = result["p_value"]
                if not np.isnan(pval) and pval <= alpha:
                    reject_h1 += 1
            except Exception:
                continue

        rows.append(
            {
                "seed": seed,
                "model": "lowrank_svd",
                "M": M,
                "type1_error": reject_h0 / succ_h0 if succ_h0 > 0 else np.nan,
                "power": reject_h1 / succ_h1 if succ_h1 > 0 else np.nan,
                "M_effective_type1": succ_h0,
                "M_effective_power": succ_h1,
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


def aggregate_validation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["M"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (model, M) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        g = grouped[(model, M)]
        type1_vals = [float(x["type1_error"]) for x in g]
        power_vals = [float(x["power"]) for x in g]

        m1, lo1, hi1, sd1, n1 = mean_ci95(type1_vals)
        m2, lo2, hi2, sd2, n2 = mean_ci95(power_vals)

        out.append(
            {
                "model": model,
                "M": M,
                "n_seed": len(g),
                "type1_error_mean": m1,
                "type1_error_ci95_low": lo1,
                "type1_error_ci95_high": hi1,
                "type1_error_sd": sd1,
                "type1_error_n": n1,
                "power_mean": m2,
                "power_ci95_low": lo2,
                "power_ci95_high": hi2,
                "power_sd": sd2,
                "power_n": n2,
                "M_effective_type1_avg": float(np.mean([x["M_effective_type1"] for x in g])),
                "M_effective_power_avg": float(np.mean([x["M_effective_power"] for x in g])),
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


def plot_model_metric_ci(rows_agg: List[Dict[str, Any]], metric: str, output_path: str) -> None:
    if not HAS_MATPLOTLIB or not rows_agg:
        return

    models = sorted(set(r["model"] for r in rows_agg))
    plt.figure(figsize=(9, 5))

    for model in models:
        sub = sorted([r for r in rows_agg if r["model"] == model], key=lambda x: x["M"])
        M = [r["M"] for r in sub]
        y = [r[f"{metric}_mean"] for r in sub]
        ylo = [r[f"{metric}_ci95_low"] for r in sub]
        yhi = [r[f"{metric}_ci95_high"] for r in sub]
        err_lower = np.array(y) - np.array(ylo)
        err_upper = np.array(yhi) - np.array(y)
        plt.errorbar(M, y, yerr=[err_lower, err_upper], marker="o", capsize=4, linewidth=1.5, label=model)

    plt.xlabel("Monte Carlo repetitions (M)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs M (mean with 95% quantile CI across seeds)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-stage large-scale experiment with p=3 and multi-seed CI")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--B-grid", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--M-grid", type=int, nargs="+", default=[300, 500, 800])
    parser.add_argument("--B-mc-baseline", type=int, default=80)
    parser.add_argument("--B-mc-highdim", type=int, default=100)
    parser.add_argument("--tag", type=str, default="next_stage_p3")
    args = parser.parse_args()

    p = 3
    config_dims = {
        "baseline": {"N": 2, "T": 220, "t": 110},
        "sparse": {"N": 3, "T": 220, "t": 110},
        "lowrank": {"N": 4, "T": 220, "t": 110},
    }

    baseline_rows_raw: List[Dict[str, Any]] = []
    validation_rows_raw: List[Dict[str, Any]] = []

    for seed in args.seeds:
        print(f"[Seed {seed}] running...")
        generator = VARDataGenerator(seed=seed)

        print(f"[Seed {seed}] baseline p-value comparison...")
        baseline_rows_raw.extend(
            run_baseline_pvalue_comparison(
                generator=generator,
                seed=seed,
                p=p,
                T=config_dims["baseline"]["T"],
                N=config_dims["baseline"]["N"],
                t=config_dims["baseline"]["t"],
                alpha=args.alpha,
                B_values=args.B_grid,
            )
        )

        print(f"[Seed {seed}] baseline type1/power...")
        validation_rows_raw.extend(
            run_baseline_validation(
                generator=generator,
                seed=seed,
                p=p,
                T=config_dims["baseline"]["T"],
                N=config_dims["baseline"]["N"],
                t=config_dims["baseline"]["t"],
                M_values=args.M_grid,
                B_bootstrap_baseline=args.B_mc_baseline,
                alpha=args.alpha,
            )
        )

        print(f"[Seed {seed}] sparse type1/power...")
        validation_rows_raw.extend(
            run_sparse_validation(
                generator=generator,
                seed=seed,
                p=p,
                T=config_dims["sparse"]["T"],
                N=config_dims["sparse"]["N"],
                t=config_dims["sparse"]["t"],
                M_values=args.M_grid,
                B_bootstrap_hd=args.B_mc_highdim,
                alpha=args.alpha,
            )
        )

        print(f"[Seed {seed}] lowrank type1/power...")
        validation_rows_raw.extend(
            run_lowrank_validation(
                generator=generator,
                seed=seed,
                p=p,
                T=config_dims["lowrank"]["T"],
                N=config_dims["lowrank"]["N"],
                t=config_dims["lowrank"]["t"],
                M_values=args.M_grid,
                B_bootstrap_hd=args.B_mc_highdim,
                alpha=args.alpha,
            )
        )

        print(f"[Seed {seed}] done.")

    baseline_rows_agg = aggregate_baseline_rows(baseline_rows_raw)
    validation_rows_agg = aggregate_validation_rows(validation_rows_raw)

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""

    baseline_raw_csv = f"results/next_stage_baseline_raw_{stamp}{suffix}.csv"
    baseline_agg_csv = f"results/next_stage_baseline_agg_{stamp}{suffix}.csv"
    validation_raw_csv = f"results/next_stage_validation_raw_{stamp}{suffix}.csv"
    validation_agg_csv = f"results/next_stage_validation_agg_{stamp}{suffix}.csv"

    write_csv(baseline_raw_csv, baseline_rows_raw)
    write_csv(baseline_agg_csv, baseline_rows_agg)
    write_csv(validation_raw_csv, validation_rows_raw)
    write_csv(validation_agg_csv, validation_rows_agg)

    baseline_png = f"results/next_stage_baseline_pvalues_vs_B_{stamp}{suffix}.png"
    type1_png = f"results/next_stage_model_type1_vs_M_{stamp}{suffix}.png"
    power_png = f"results/next_stage_model_power_vs_M_{stamp}{suffix}.png"

    if HAS_MATPLOTLIB:
        plot_baseline_ci(baseline_rows_agg, baseline_png)
        plot_model_metric_ci(validation_rows_agg, metric="type1_error", output_path=type1_png)
        plot_model_metric_ci(validation_rows_agg, metric="power", output_path=power_png)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "alpha": args.alpha,
            "seeds": args.seeds,
            "B_grid": args.B_grid,
            "M_grid": args.M_grid,
            "B_mc_baseline": args.B_mc_baseline,
            "B_mc_highdim": args.B_mc_highdim,
            "p": p,
            "dimensions": config_dims,
        },
        "baseline_agg_rows": baseline_rows_agg,
        "validation_agg_rows": validation_rows_agg,
        "outputs": {
            "baseline_raw_csv": baseline_raw_csv,
            "baseline_agg_csv": baseline_agg_csv,
            "validation_raw_csv": validation_raw_csv,
            "validation_agg_csv": validation_agg_csv,
            "baseline_png": baseline_png if HAS_MATPLOTLIB else None,
            "type1_png": type1_png if HAS_MATPLOTLIB else None,
            "power_png": power_png if HAS_MATPLOTLIB else None,
        },
    }

    summary_json = f"results/next_stage_summary_{stamp}{suffix}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_md = f"results/下一阶段大规模试验报告_{stamp}{suffix}.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# 下一阶段大规模试验报告\n\n")
        f.write("## 1. 配置\n")
        f.write(f"- 时间：{summary['timestamp']}\n")
        f.write(f"- p：{p}（所有场景统一）\n")
        f.write(f"- seeds：{args.seeds}\n")
        f.write(f"- M-grid：{args.M_grid}\n")
        f.write(f"- B-grid：{args.B_grid}\n")
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

        f.write("## 3. 第一类错误与功效（跨seed均值 + 95%分位数CI）\n")
        f.write("| 模型 | M | Type I mean [95%CI] | Power mean [95%CI] |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in sorted(validation_rows_agg, key=lambda x: (x['model'], x['M'])):
            f.write(
                f"| {r['model']} | {r['M']} | "
                f"{r['type1_error_mean']:.4f} [{r['type1_error_ci95_low']:.4f}, {r['type1_error_ci95_high']:.4f}] | "
                f"{r['power_mean']:.4f} [{r['power_ci95_low']:.4f}, {r['power_ci95_high']:.4f}] |\n"
            )
        f.write("\n")

        f.write("## 4. 数据背景说明\n")
        f.write("- 指标定义：第一类错误 = H0真时拒绝率；功效 = H1真时拒绝率。\n")
        f.write("- 置信区间：跨seed统计，区间端点取各指标在 seed 维度上的 2.5% / 97.5% 分位数。\n")
        f.write("- H0 数据：无断点序列；H1 数据：在给定 t 处设定参数突变。\n")
        f.write("- 同一场景下 H0/H1 共享相同 `p,T,N,t` 口径。\n\n")

        f.write("## 5. 输出文件\n")
        f.write(f"- summary: `{summary_json}`\n")
        f.write(f"- baseline raw: `{baseline_raw_csv}`\n")
        f.write(f"- baseline agg: `{baseline_agg_csv}`\n")
        f.write(f"- validation raw: `{validation_raw_csv}`\n")
        f.write(f"- validation agg: `{validation_agg_csv}`\n")
        if HAS_MATPLOTLIB:
            f.write(f"- baseline 图: `{baseline_png}`\n")
            f.write(f"- type1 图: `{type1_png}`\n")
            f.write(f"- power 图: `{power_png}`\n")

    print("Experiment completed.")
    print("Outputs:")
    print(f"- {summary_json}")
    print(f"- {report_md}")
    print(f"- {baseline_raw_csv}")
    print(f"- {baseline_agg_csv}")
    print(f"- {validation_raw_csv}")
    print(f"- {validation_agg_csv}")
    if HAS_MATPLOTLIB:
        print(f"- {baseline_png}")
        print(f"- {type1_png}")
        print(f"- {power_png}")


if __name__ == "__main__":
    main()
