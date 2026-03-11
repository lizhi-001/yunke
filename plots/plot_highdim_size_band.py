"""绘制高维实验的 Type I error 均值 + 误差带图。

用法示例：
  python3 plots/plot_highdim_size_band.py \
    --run-dir results/highdim_runs/2026-03-11_095431_b500_seed2_band

  python3 plots/plot_highdim_size_band.py \
    --agg-csv results/highdim_runs/<run>/highdim_agg_*.csv \
    --output results/highdim_runs/<run>/size_band_plot.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib-codex-cache")

import matplotlib.pyplot as plt


MODEL_ORDER = ["baseline_ols", "baseline_ols_f", "sparse_lasso", "lowrank_svd"]
MODEL_LABELS = {
    "baseline_ols": "OLS(LR)",
    "baseline_ols_f": "OLS(F)",
    "sparse_lasso": "Lasso",
    "lowrank_svd": "SVD",
}
MODEL_COLORS = {
    "baseline_ols": "#1f77b4",
    "baseline_ols_f": "#ff7f0e",
    "sparse_lasso": "#2ca02c",
    "lowrank_svd": "#d62728",
}
N_ORDER = [5, 10, 20]


def infer_agg_csv(run_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(run_dir, "highdim_agg_*.csv")))
    if not matches:
        raise FileNotFoundError(f"未找到聚合结果 CSV: {run_dir}")
    return matches[-1]


def infer_output_path(run_dir: str) -> str:
    return os.path.join(run_dir, "高维_size_mean_band.png")


def parse_model_name(model_name: str) -> tuple[str, int]:
    for prefix in MODEL_ORDER:
        if model_name.startswith(prefix):
            suffix = model_name.removeprefix(prefix)
            if suffix.startswith("_n"):
                return prefix, int(suffix[2:])
    raise ValueError(f"无法解析模型名: {model_name}")


def load_type1_rows(agg_csv_path: str) -> Dict[int, Dict[str, List[dict]]]:
    grouped: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(agg_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "aggregate" or row.get("metric") != "type1_error":
                continue
            base_model, n_dim = parse_model_name(row["model"])
            grouped[n_dim][base_model].append({
                "M": int(row["M"]),
                "mean": float(row["value_mean"]),
                "ci95_low": float(row["ci95_low"]) if row.get("ci95_low") else math.nan,
                "ci95_high": float(row["ci95_high"]) if row.get("ci95_high") else math.nan,
                "mc_se_pooled": float(row["mc_se_pooled"]) if row.get("mc_se_pooled") else math.nan,
                "seed_count": int(row["seed_count"]) if row.get("seed_count") else 0,
            })
    for n_dim in grouped:
        for model in grouped[n_dim]:
            grouped[n_dim][model].sort(key=lambda item: item["M"])
    return grouped


def plot_size_band(agg_csv_path: str, output_path: str, title: str | None = None) -> None:
    grouped = load_type1_rows(agg_csv_path)
    fig, axes = plt.subplots(1, len(N_ORDER), figsize=(18, 5), sharey=True)
    if len(N_ORDER) == 1:
        axes = [axes]

    for ax, n_dim in zip(axes, N_ORDER):
        for base_model in MODEL_ORDER:
            rows = grouped.get(n_dim, {}).get(base_model, [])
            if not rows:
                continue
            xs = [row["M"] for row in rows]
            ys = [row["mean"] for row in rows]
            y_low = [row["ci95_low"] for row in rows]
            y_high = [row["ci95_high"] for row in rows]
            color = MODEL_COLORS[base_model]
            label = MODEL_LABELS[base_model]
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
            if not any(math.isnan(v) for v in y_low + y_high):
                ax.fill_between(xs, y_low, y_high, color=color, alpha=0.15)

        ax.axhline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(f"N = {n_dim}")
        ax.set_xlabel("M")
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 300, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_ylim(0.0, 0.14)

    axes[0].set_ylabel("Type I Error Mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title or "High-Dimensional VAR: Type I Error Mean with 95% MC Bands", y=1.08, fontsize=14)
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制高维实验的 M-size 均值 + 误差带图像")
    parser.add_argument("--run-dir", type=str, default="", help="实验运行目录（优先自动定位 highdim_agg_*.csv）")
    parser.add_argument("--agg-csv", type=str, default="", help="聚合 CSV 路径")
    parser.add_argument("--output", type=str, default="", help="输出图片路径")
    parser.add_argument("--title", type=str, default="", help="图像标题")
    args = parser.parse_args()

    if not args.run_dir and not args.agg_csv:
        parser.error("必须提供 --run-dir 或 --agg-csv 之一")

    agg_csv_path = args.agg_csv or infer_agg_csv(args.run_dir)
    run_dir = args.run_dir or str(Path(agg_csv_path).parent)
    output_path = args.output or infer_output_path(run_dir)
    title = args.title or f"Type I Error Mean + 95% MC Band ({Path(run_dir).name})"

    plot_size_band(agg_csv_path, output_path, title)
    print(f"AGG CSV : {agg_csv_path}")
    print(f"OUTPUT  : {output_path}")


if __name__ == "__main__":
    main()
