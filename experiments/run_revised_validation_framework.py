"""
按“baseline-Chow + 高维Bootstrap”框架运行实验并导出结果。

实验结构：
1) baseline（常规时间序列）下，比较 Bootstrap p 值与卡方近似 p 值，
   并评估不同 Bootstrap 次数 B 下差异程度。
2) 在常规/稀疏/低秩场景下评估 Type I Error 与 Power，
   并比较不同 Monte Carlo 次数 M 下估计结果变化。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation import VARDataGenerator, ChowTest, ChowBootstrapInference
from sparse_var import SparseMonteCarloSimulation
from lowrank_var import LowRankMonteCarloSimulation

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def ensure_stationary(phi: np.ndarray, shrink: float = 0.9, max_attempts: int = 50) -> np.ndarray:
    current = phi.copy()
    attempts = 0
    while not VARDataGenerator.check_stationarity(current) and attempts < max_attempts:
        current = current * shrink
        attempts += 1
    if not VARDataGenerator.check_stationarity(current):
        raise ValueError("无法构造平稳的 Phi2")
    return current


def baseline_pvalue_comparison(generator: VARDataGenerator,
                               N: int,
                               T: int,
                               p: int,
                               t: int,
                               alpha: float,
                               B_values: List[int],
                               seed: int) -> Dict[str, Any]:
    """固定同一条常规序列，比较不同 B 下 bootstrap 与卡方 p 值差异。"""
    Sigma = np.eye(N) * 0.5
    Phi = generator.generate_stationary_phi(N, p, scale=0.3)
    Y = generator.generate_var_series(T, N, p, Phi, Sigma)

    chow = ChowTest()
    asym = chow.compute_at_point(Y, p, t)
    chi2_p = asym['chi2_p_value']

    rows = []
    for B in B_values:
        bootstrap = ChowBootstrapInference(B=B, seed=seed)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha, verbose=False)
        p_boot_lr = result['bootstrap_lr_p_value']
        p_boot_f = result['bootstrap_f_p_value']
        rows.append({
            'B': B,
            'chi2_p_value': chi2_p,
            'f_asymptotic_p_value': result['f_asymptotic_p_value'],
            'bootstrap_lr_p_value': p_boot_lr,
            'bootstrap_f_p_value': p_boot_f,
            'abs_diff_boot_lr_vs_chi2': abs(p_boot_lr - chi2_p),
            'B_effective': result['B_effective'],
        })

    return {
        'parameters': {
            'N': N,
            'T': T,
            'p': p,
            't': t,
            'alpha': alpha,
            'B_values': B_values,
            'seed': seed,
        },
        'chi2_p_value_fixed': chi2_p,
        'rows': rows,
    }


def baseline_type1_power_vs_M(generator: VARDataGenerator,
                              M_values: List[int],
                              B_bootstrap: int,
                              alpha: float,
                              seed: int) -> List[Dict[str, Any]]:
    """baseline 在不同 M 下比较四种推断：渐近 F、渐近χ²、Bootstrap-F、Bootstrap-LR。"""
    np.random.seed(seed)
    N, T, p, t = 2, 120, 1, 60
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.3)
    Phi2 = ensure_stationary(Phi1 + 0.25 * np.ones_like(Phi1))

    rows = []
    inference_keys = [
        ('baseline_chow_asym_f', 'f_asymptotic_p_value'),
        ('baseline_chow_asym_chi2', 'chi2_asymptotic_p_value'),
        ('baseline_chow_bootstrap_f', 'bootstrap_f_p_value'),
        ('baseline_chow_bootstrap_lr', 'bootstrap_lr_p_value'),
    ]

    for M in M_values:
        h0_counts = {name: 0 for name, _ in inference_keys}
        h1_counts = {name: 0 for name, _ in inference_keys}
        succ_h0 = 0
        succ_h1 = 0

        for _ in range(M):
            try:
                Y = generator.generate_var_series(T, N, p, Phi1, Sigma)
                result = ChowBootstrapInference(B=B_bootstrap).test_at_point(Y, p, t, alpha=alpha)
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
                result = ChowBootstrapInference(B=B_bootstrap).test_at_point(Y, p, t, alpha=alpha)
                succ_h1 += 1
                for name, key in inference_keys:
                    p_value = result[key]
                    if not np.isnan(p_value) and p_value <= alpha:
                        h1_counts[name] += 1
            except Exception:
                continue

        for name, _ in inference_keys:
            rows.append({
                'model': name,
                'M': M,
                'type1_error': h0_counts[name] / succ_h0 if succ_h0 > 0 else np.nan,
                'power': h1_counts[name] / succ_h1 if succ_h1 > 0 else np.nan,
                'M_effective_type1': succ_h0,
                'M_effective_power': succ_h1,
            })

    return rows


def sparse_type1_power_vs_M(generator: VARDataGenerator,
                            M_values: List[int],
                            B_bootstrap: int,
                            alpha: float,
                            seed: int) -> List[Dict[str, Any]]:
    """稀疏场景在不同 M 下的 size/power。"""
    N, T, p, t = 5, 200, 1, 100
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_stationary_phi(N, p, sparsity=0.2, scale=0.3)
    Phi2 = ensure_stationary(Phi1 + 0.25 * np.ones_like(Phi1))

    rows = []
    for M in M_values:
        mc = SparseMonteCarloSimulation(
            M=M,
            B=B_bootstrap,
            seed=seed,
            estimator_type='lasso',
            alpha=0.02,
        )
        type1 = mc.evaluate_type1_error(N, T, p, Phi1, Sigma, t, test_alpha=alpha, verbose=False)
        power = mc.evaluate_power(N, T, p, Phi1, Phi2, Sigma, t, t, test_alpha=alpha, verbose=False)
        rows.append({
            'model': 'sparse_lasso',
            'M': M,
            'type1_error': float(type1['type1_error']),
            'power': float(power['power']),
            'M_effective_type1': int(type1['M_effective']),
            'M_effective_power': int(power['M_effective']),
        })

    return rows


def lowrank_type1_power_vs_M(generator: VARDataGenerator,
                             M_values: List[int],
                             B_bootstrap: int,
                             alpha: float,
                             seed: int) -> List[Dict[str, Any]]:
    """低秩场景在不同 M 下的 size/power。"""
    N, T, p, t = 8, 200, 1, 100
    Sigma = np.eye(N) * 0.5
    Phi1 = generator.generate_lowrank_phi(N, p, rank=2, scale=0.3)
    Phi2 = ensure_stationary(Phi1 + 0.20 * np.ones_like(Phi1))

    rows = []
    for M in M_values:
        mc = LowRankMonteCarloSimulation(
            M=M,
            B=B_bootstrap,
            seed=seed,
            method='svd',
            rank=2,
        )
        type1 = mc.evaluate_type1_error(N, T, p, Phi1, Sigma, t, test_alpha=alpha, verbose=False)
        power = mc.evaluate_power(N, T, p, Phi1, Phi2, Sigma, t, t, test_alpha=alpha, verbose=False)
        rows.append({
            'model': 'lowrank_svd',
            'M': M,
            'type1_error': float(type1['type1_error']),
            'power': float(power['power']),
            'M_effective_type1': int(type1['M_effective']),
            'M_effective_power': int(power['M_effective']),
        })

    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_baseline_diff(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return
    B = [r['B'] for r in rows]
    diff = [r['abs_diff_boot_lr_vs_chi2'] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(B, diff, marker='o', linewidth=2)
    plt.xlabel('Bootstrap repetitions (B)')
    plt.ylabel('|p_bootstrap(LR) - p_chi2(LR)|')
    plt.title('Baseline: bootstrap vs chi-square p-value difference')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_baseline_pvalues_vs_B(rows: List[Dict[str, Any]], output_path: str) -> None:
    """绘制 baseline 中 p 值与 Bootstrap 次数 B 的关系图。"""
    if not HAS_MATPLOTLIB or not rows:
        return

    rows_sorted = sorted(rows, key=lambda x: x['B'])
    B = [r['B'] for r in rows_sorted]
    p_chi2 = [r['chi2_p_value'] for r in rows_sorted]
    p_f_asym = [r['f_asymptotic_p_value'] for r in rows_sorted]
    p_boot_lr = [r['bootstrap_lr_p_value'] for r in rows_sorted]
    p_boot_f = [r['bootstrap_f_p_value'] for r in rows_sorted]

    plt.figure(figsize=(9, 5.5))
    plt.plot(B, p_chi2, marker='o', linewidth=2, label='asym-chi2(LR)')
    plt.plot(B, p_f_asym, marker='o', linewidth=2, label='asym-F')
    plt.plot(B, p_boot_lr, marker='o', linewidth=2, label='bootstrap-LR')
    plt.plot(B, p_boot_f, marker='o', linewidth=2, label='bootstrap-F')
    plt.xlabel('Bootstrap repetitions (B)')
    plt.ylabel('p-value')
    plt.title('Baseline: p-value vs Bootstrap repetitions')
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_m_sensitivity(rows: List[Dict[str, Any]], output_type1: str, output_power: str) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return
    models = sorted(set(r['model'] for r in rows))

    plt.figure(figsize=(8, 5))
    for model in models:
        model_rows = sorted([r for r in rows if r['model'] == model], key=lambda x: x['M'])
        plt.plot([r['M'] for r in model_rows], [r['type1_error'] for r in model_rows], marker='o', label=model)
    plt.axhline(0.05, color='k', linestyle='--', linewidth=1, label='alpha=0.05')
    plt.xlabel('Monte Carlo repetitions (M)')
    plt.ylabel('Type I Error')
    plt.title('Type I Error sensitivity to M')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_type1, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model in models:
        model_rows = sorted([r for r in rows if r['model'] == model], key=lambda x: x['M'])
        plt.plot([r['M'] for r in model_rows], [r['power'] for r in model_rows], marker='o', label=model)
    plt.xlabel('Monte Carlo repetitions (M)')
    plt.ylabel('Power')
    plt.title('Power sensitivity to M')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_power, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Run revised modeling/validation framework experiments.')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--B-grid', type=int, nargs='+', default=[50, 100, 200, 400])
    parser.add_argument('--M-grid', type=int, nargs='+', default=[30, 60, 120])
    parser.add_argument('--B-mc', type=int, default=80,
                        help='Bootstrap repetitions used inside MC loops for Part 2')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    generator = VARDataGenerator(seed=args.seed)
    os.makedirs('results', exist_ok=True)

    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    suffix = f"_{args.tag}" if args.tag else ''

    # Part 1: baseline p-value comparison across B
    baseline_cmp = baseline_pvalue_comparison(
        generator=generator,
        N=2,
        T=120,
        p=1,
        t=60,
        alpha=args.alpha,
        B_values=args.B_grid,
        seed=args.seed,
    )

    baseline_cmp_csv = f"results/baseline_bootstrap_vs_chi2_{stamp}{suffix}.csv"
    write_csv(baseline_cmp_csv, baseline_cmp['rows'])
    baseline_cmp_png = f"results/baseline_bootstrap_vs_chi2_{stamp}{suffix}.png"
    plot_baseline_diff(baseline_cmp['rows'], baseline_cmp_png)
    baseline_pvalues_png = f"results/baseline_pvalues_vs_B_{stamp}{suffix}.png"
    plot_baseline_pvalues_vs_B(baseline_cmp['rows'], baseline_pvalues_png)

    # Part 2: Type I / Power vs M across three model classes
    rows_all = []
    rows_all.extend(baseline_type1_power_vs_M(generator, args.M_grid, args.B_mc, args.alpha, args.seed))
    rows_all.extend(sparse_type1_power_vs_M(generator, args.M_grid, args.B_mc, args.alpha, args.seed))
    rows_all.extend(lowrank_type1_power_vs_M(generator, args.M_grid, args.B_mc, args.alpha, args.seed))

    mc_sensitivity_csv = f"results/model_validation_vs_M_{stamp}{suffix}.csv"
    write_csv(mc_sensitivity_csv, rows_all)

    type1_png = f"results/model_type1_vs_M_{stamp}{suffix}.png"
    power_png = f"results/model_power_vs_M_{stamp}{suffix}.png"
    plot_m_sensitivity(rows_all, type1_png, power_png)

    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'alpha': args.alpha,
            'seed': args.seed,
            'B_grid': args.B_grid,
            'M_grid': args.M_grid,
            'B_mc': args.B_mc,
        },
        'part1_baseline_comparison': {
            'parameters': baseline_cmp['parameters'],
            'rows': baseline_cmp['rows'],
        },
        'part2_validation_vs_M': rows_all,
        'outputs': {
            'baseline_comparison_csv': baseline_cmp_csv,
            'baseline_comparison_png': baseline_cmp_png if HAS_MATPLOTLIB else None,
            'baseline_pvalues_vs_B_png': baseline_pvalues_png if HAS_MATPLOTLIB else None,
            'mc_sensitivity_csv': mc_sensitivity_csv,
            'type1_vs_M_png': type1_png if HAS_MATPLOTLIB else None,
            'power_vs_M_png': power_png if HAS_MATPLOTLIB else None,
        },
    }

    summary_json = f"results/revised_framework_summary_{stamp}{suffix}.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Experiment completed.')
    print('Outputs:')
    print(f'- {summary_json}')
    print(f'- {baseline_cmp_csv}')
    print(f'- {mc_sensitivity_csv}')
    if HAS_MATPLOTLIB:
        print(f'- {baseline_cmp_png}')
        print(f'- {baseline_pvalues_png}')
        print(f'- {type1_png}')
        print(f'- {power_png}')
    else:
        print('- matplotlib not available, PNG plots skipped.')


if __name__ == '__main__':
    main()
