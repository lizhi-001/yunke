from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fmt(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def format_ci(mean: Any, low: Any, high: Any) -> str:
    return f"{fmt(mean)} [{fmt(low)}, {fmt(high)}]"


def build_baseline_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| B | asym-chi2(LR) | bootstrap-LR | bootstrap-F | |boot-LR - chi2| |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['B']} | "
            f"{format_ci(row['chi2_p_value_mean'], row['chi2_p_value_ci95_low'], row['chi2_p_value_ci95_high'])} | "
            f"{format_ci(row['bootstrap_lr_p_value_mean'], row['bootstrap_lr_p_value_ci95_low'], row['bootstrap_lr_p_value_ci95_high'])} | "
            f"{format_ci(row['bootstrap_f_p_value_mean'], row['bootstrap_f_p_value_ci95_low'], row['bootstrap_f_p_value_ci95_high'])} | "
            f"{format_ci(row['abs_diff_boot_lr_vs_chi2_mean'], row['abs_diff_boot_lr_vs_chi2_ci95_low'], row['abs_diff_boot_lr_vs_chi2_ci95_high'])} |"
        )
    return "\n".join(lines)


def build_validation_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 模型 | M | Type I Error | Power | 有效M(Type I) | 有效M(Power) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x['model'], int(x['M']))):
        lines.append(
            f"| {row['model']} | {row['M']} | "
            f"{format_ci(row['type1_error_mean'], row['type1_error_ci95_low'], row['type1_error_ci95_high'])} | "
            f"{format_ci(row['power_mean'], row['power_ci95_low'], row['power_ci95_high'])} | "
            f"{fmt(row['M_effective_type1_avg'], 1)} | {fmt(row['M_effective_power_avg'], 1)} |"
        )
    return "\n".join(lines)


def analyze_baseline(rows: List[Dict[str, Any]]) -> List[str]:
    analyses: List[str] = []
    if not rows:
        return analyses
    rows_sorted = sorted(rows, key=lambda x: int(x['B']))
    first = rows_sorted[0]
    last = rows_sorted[-1]
    analyses.append(
        f"- baseline p 值比较固定在同一条无断点序列上进行，目的是检验不同 bootstrap 次数 B 下渐近 LR p 值与 bootstrap LR p 值的偏离程度。"
    )
    analyses.append(
        f"- 从 B={first['B']} 到 B={last['B']}，|boot-LR - chi2| 的跨 seed 均值由 {fmt(first['abs_diff_boot_lr_vs_chi2_mean'])} 变化到 {fmt(last['abs_diff_boot_lr_vs_chi2_mean'])}；若数值下降，说明增大 B 后 bootstrap LR 与渐近 chi2 更接近。"
    )
    analyses.append(
        f"- 若 bootstrap-LR 与 asym-chi2(LR) 的区间明显重叠，则说明当前样本规模下两者差异有限；若差异不重叠，则提示有限样本修正更重要。"
    )
    return analyses


def analyze_validation(rows: List[Dict[str, Any]], alpha: float) -> List[str]:
    analyses: List[str] = []
    if not rows:
        return analyses
    analyses.append(
        f"- Type I Error 的目标参考值为显著性水平 alpha={alpha:.2f}；越接近该值，说明 size 控制越稳定。"
    )
    for model in sorted({row['model'] for row in rows}):
        sub = sorted([row for row in rows if row['model'] == model], key=lambda x: int(x['M']))
        start = sub[0]
        end = sub[-1]
        analyses.append(
            f"- `{model}` 在 M 从 {start['M']} 增至 {end['M']} 时，Type I Error 均值从 {fmt(start['type1_error_mean'])} 变化到 {fmt(end['type1_error_mean'])}，Power 均值从 {fmt(start['power_mean'])} 变化到 {fmt(end['power_mean'])}。"
        )
    analyses.append(
        "- 由于当前汇总区间是跨 seed 的 95% 分位数区间，它主要用于展示不同 seed 下结果的波动范围，而不是单次假设检验的接受/拒绝规则。"
    )
    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description='Write detailed next-stage experiment report from summary JSON.')
    parser.add_argument('summary_json', type=str)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    with open(args.summary_json, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    config = summary['config']
    baseline_rows = summary['baseline_agg_rows']
    validation_rows = summary['validation_agg_rows']
    outputs = summary['outputs']

    output_path = args.output or os.path.join(
        os.path.dirname(args.summary_json),
        os.path.basename(args.summary_json).replace('next_stage_summary_', '下一阶段大规模试验详细报告_').replace('.json', '.md'),
    )

    lines: List[str] = []
    lines.append('# 下一阶段大规模试验详细报告')
    lines.append('')
    lines.append('## 1. 试验目标')
    lines.append('')
    lines.append('- 在统一的 `p=3` 框架下，同时评估 baseline 与高维方法在已知断点检验中的 size / power 表现。')
    lines.append('- baseline 继续使用同一有效样本 + 哑变量交互的 Chow/LR 方案，并同时输出渐近 F、渐近 chi2(LR)、bootstrap-F、bootstrap-LR。')
    lines.append('- 高维部分直接回答原专用方法的表现：稀疏场景使用 `SparseBootstrapInference`，低秩场景使用 `LowRankBootstrapInference`。')
    lines.append('')
    lines.append('## 2. 数据背景')
    lines.append('')
    lines.append(f"- 显著性水平：`alpha={config['alpha']}`")
    lines.append(f"- seeds：`{config['seeds']}`")
    lines.append(f"- baseline p 值比较的 B-grid：`{config['B_grid']}`")
    lines.append(f"- Monte Carlo 的 M-grid：`{config['M_grid']}`")
    lines.append(f"- baseline 内部 bootstrap 次数：`{config['B_mc_baseline']}`")
    lines.append(f"- 高维内部 bootstrap 次数：`{config['B_mc_highdim']}`")
    dims = config['dimensions']
    lines.append(f"- baseline 场景：`N={dims['baseline']['N']}, T={dims['baseline']['T']}, p={config['p']}, t={dims['baseline']['t']}`")
    lines.append(f"- sparse 场景：`N={dims['sparse']['N']}, T={dims['sparse']['T']}, p={config['p']}, t={dims['sparse']['t']}`")
    lines.append(f"- lowrank 场景：`N={dims['lowrank']['N']}, T={dims['lowrank']['T']}, p={config['p']}, t={dims['lowrank']['t']}`")
    lines.append('- H0 数据为无断点 VAR 序列；H1 数据为在给定断点位置 `t` 处发生结构变化的 VAR 序列。')
    lines.append('- 所有汇总区间采用跨 seed 的 95% 分位数区间，即端点为 2.5% / 97.5% 分位数。')
    lines.append('')
    lines.append('## 3. 检验流程')
    lines.append('')
    lines.append('### 3.1 baseline')
    lines.append('')
    lines.append('- 对同一条 baseline 无断点序列，先计算渐近 LR/chi2 p 值，再对不同 B 计算 bootstrap-F 与 bootstrap-LR p 值。')
    lines.append('- 在 baseline size/power 评估中，对每个 M 分别重复 H0 与 H1 模拟，并统计四种 baseline 推断口径的拒绝率。')
    lines.append('')
    lines.append('### 3.2 sparse_lasso')
    lines.append('')
    lines.append('- 数据由稀疏 VAR 结构生成。')
    lines.append('- 检验器使用 `SparseBootstrapInference`：先以稀疏 LR 统计量比较 H0 与 H1，再在 H0 下重抽样残差构造 bootstrap LR 分布，最终以 bootstrap p 值做拒绝决策。')
    lines.append('')
    lines.append('### 3.3 lowrank_svd')
    lines.append('')
    lines.append('- 数据由低秩 VAR 结构生成。')
    lines.append('- 检验器使用 `LowRankBootstrapInference`：先构造低秩 LR 统计量，再在 H0 下执行 bootstrap 得到经验 p 值。')
    lines.append('')
    lines.append('## 4. 结果展示')
    lines.append('')
    lines.append('### 4.1 baseline p 值比较')
    lines.append('')
    lines.append(build_baseline_table(baseline_rows))
    lines.append('')
    lines.append('### 4.2 size / power 汇总')
    lines.append('')
    lines.append(build_validation_table(validation_rows))
    lines.append('')
    lines.append('## 5. 结果分析')
    lines.append('')
    lines.extend(analyze_baseline(baseline_rows))
    lines.extend(analyze_validation(validation_rows, float(config['alpha'])))
    lines.append('')
    lines.append('## 6. 输出文件')
    lines.append('')
    lines.append(f"- summary JSON：`{args.summary_json}`")
    lines.append(f"- baseline 原始表：`{outputs['baseline_raw_csv']}`")
    lines.append(f"- baseline 聚合表：`{outputs['baseline_agg_csv']}`")
    lines.append(f"- validation 原始表：`{outputs['validation_raw_csv']}`")
    lines.append(f"- validation 聚合表：`{outputs['validation_agg_csv']}`")
    if outputs.get('baseline_png'):
        lines.append(f"- baseline 图：`{outputs['baseline_png']}`")
    if outputs.get('type1_png'):
        lines.append(f"- type I 图：`{outputs['type1_png']}`")
    if outputs.get('power_png'):
        lines.append(f"- power 图：`{outputs['power_png']}`")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(output_path)


if __name__ == '__main__':
    main()
