from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


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


def build_size_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 场景 | 模型 | M | Size mean | 有效M |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x['scenario'], x['model'], int(x['M']))):
        lines.append(
            f"| {row['scenario']} | {row['model']} | {row['M']} | "
            f"{fmt(row['size_mean'])} | "
            f"{fmt(row['M_effective_avg'], 1)} |"
        )
    return "\n".join(lines)


def build_power_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 场景 | 模型 | 变化幅度 | 固定 M_max | Power | 平均Δ Fro |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x['scenario'], x['model'], float(x['change_scale']), int(x['M']))):
        lines.append(
            f"| {row['scenario']} | {row['model']} | {fmt(row['change_scale'], 2)} | {row['M']} | "
            f"{format_ci(row['power_mean'], row['power_ci95_low'], row['power_ci95_high'])} | "
            f"{fmt(row['delta_fro_mean'], 4)} |"
        )
    return "\n".join(lines)


def analyze_baseline(rows: List[Dict[str, Any]]) -> List[str]:
    analyses: List[str] = []
    if not rows:
        return analyses
    rows_sorted = sorted(rows, key=lambda x: int(x['B']))
    first = rows_sorted[0]
    last = rows_sorted[-1]
    analyses.append("- baseline p 值比较固定在同一条无断点序列上进行，用于观察不同 B 下渐近 LR 与 bootstrap LR 的偏离程度。")
    analyses.append(
        f"- 从 B={first['B']} 到 B={last['B']}，|boot-LR - chi2| 的跨 seed 均值由 {fmt(first['abs_diff_boot_lr_vs_chi2_mean'])} 变化到 {fmt(last['abs_diff_boot_lr_vs_chi2_mean'])}。"
    )
    return analyses


def analyze_size(rows: List[Dict[str, Any]], alpha: float) -> List[str]:
    analyses: List[str] = []
    if not rows:
        return analyses
    analyses.append(f"- Size 的目标参考值为显著性水平 `alpha={alpha:.2f}`；越接近该值，说明 size 控制越稳定。")
    for label in sorted({(row['scenario'], row['model']) for row in rows}):
        scenario, model = label
        sub = sorted([row for row in rows if row['scenario'] == scenario and row['model'] == model], key=lambda x: int(x['M']))
        start = sub[0]
        end = sub[-1]
        analyses.append(
            f"- `{scenario}/{model}` 在 M 从 {start['M']} 增至 {end['M']} 时，Size 均值从 {fmt(start['size_mean'])} 变化到 {fmt(end['size_mean'])}。"
        )
    return analyses


def analyze_power(rows: List[Dict[str, Any]]) -> List[str]:
    analyses: List[str] = []
    if not rows:
        return analyses
    analyses.append("- Power 部分按至少 5 档变化幅度分别汇报，并固定在最大 `M_max` 下；重点检查变化幅度变大后，拒绝率是否整体抬升。")
    labels = sorted({(row['scenario'], row['model']) for row in rows})
    for scenario, model in labels:
        sub = sorted(
            [row for row in rows if row['scenario'] == scenario and row['model'] == model],
            key=lambda x: float(x['change_scale']),
        )
        start = sub[0]
        end = sub[-1]
        direction = "提升" if float(end['power_mean']) >= float(start['power_mean']) else "未提升"
        analyses.append(
            f"- `{scenario}/{model}` 在固定 `M_max={start['M']}` 下，变化幅度从 {fmt(start['change_scale'], 2)} 增至 {fmt(end['change_scale'], 2)} 时，Power 均值从 {fmt(start['power_mean'])} 变化到 {fmt(end['power_mean'])}，整体{direction}。"
        )
    analyses.append("- 当前区间是跨 seed 的 95% 分位数区间，用于展示不同 seed 下的波动范围，而不是单次检验的决策边界。")
    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description='Write detailed next-stage experiment report from summary JSON.')
    parser.add_argument('summary_json', type=str)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    with open(args.summary_json, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    config = summary['config']
    baseline_rows = summary.get('baseline_agg_rows', [])
    size_rows = summary.get('size_agg_rows', [])
    power_rows = summary.get('power_agg_rows', [])
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
    lines.append('- 在统一的 `p=3` 框架下，分开汇报 baseline 与高维方法在已知断点检验中的 size 和 power。')
    lines.append('- size 部分单独展示不同场景、不同模型在不同 M 下的拒绝率。')
    lines.append('- power 部分单独展示不同场景、不同模型在至少 5 档时间序列变化幅度下、固定最大 `M_max` 的拒绝率，并检查 power 是否随变化幅度提升。')
    lines.append('')
    lines.append('## 2. 数据背景')
    lines.append('')
    lines.append(f"- 显著性水平：`alpha={config['alpha']}`")
    lines.append(f"- seeds：`{config['seeds']}`")
    lines.append(f"- baseline p 值比较的 B-grid：`{config['B_grid']}`")
    lines.append(f"- size 的 M-grid：`{config['M_grid']}`")
    lines.append(f"- power 的变化幅度 grid：`{config['power_change_grid']}`")
    lines.append(f"- power 固定 M_max：`{config['power_M']}`")
    lines.append(f"- baseline 内部 bootstrap 次数：`{config['B_mc_baseline']}`")
    lines.append(f"- 高维内部 bootstrap 次数：`{config['B_mc_highdim']}`")
    dims = config['dimensions']
    lines.append(f"- baseline 场景：`N={dims['baseline']['N']}, T={dims['baseline']['T']}, p={config['p']}, t={dims['baseline']['t']}`")
    lines.append(f"- sparse 场景：`N={dims['sparse']['N']}, T={dims['sparse']['T']}, p={config['p']}, t={dims['sparse']['t']}`")
    lines.append(f"- lowrank 场景：`N={dims['lowrank']['N']}, T={dims['lowrank']['T']}, p={config['p']}, t={dims['lowrank']['t']}`")
    lines.append('- H0 数据为无断点 VAR 序列；H1 数据为在给定断点位置 `t` 处发生结构变化的 VAR 序列。')
    lines.append('- 本轮固定 3 个 seed；size 主表默认只播报跨 seed 均值，不单独展示 95%CI。')
    lines.append('')
    lines.append('## 3. baseline p 值比较')
    lines.append('')
    lines.append(build_baseline_table(baseline_rows))
    lines.append('')
    lines.append('## 4. Size 结果')
    lines.append('')
    lines.append(build_size_table(size_rows))
    lines.append('')
    lines.append('## 5. 固定最大 M 下的 Power 结果')
    lines.append('')
    lines.append(build_power_table(power_rows))
    lines.append('')
    lines.append('## 6. 结果分析')
    lines.append('')
    lines.extend(analyze_baseline(baseline_rows))
    lines.extend(analyze_size(size_rows, float(config['alpha'])))
    lines.extend(analyze_power(power_rows))
    lines.append('')
    lines.append('## 7. 输出文件')
    lines.append('')
    lines.append(f"- summary JSON：`{args.summary_json}`")
    lines.append(f"- baseline 原始表：`{outputs['baseline_raw_csv']}`")
    lines.append(f"- baseline 聚合表：`{outputs['baseline_agg_csv']}`")
    lines.append(f"- size 原始表：`{outputs['size_raw_csv']}`")
    lines.append(f"- size 聚合表：`{outputs['size_agg_csv']}`")
    lines.append(f"- power 原始表：`{outputs['power_raw_csv']}`")
    lines.append(f"- power 聚合表：`{outputs['power_agg_csv']}`")
    if outputs.get('baseline_png'):
        lines.append(f"- baseline 图：`{outputs['baseline_png']}`")
    if outputs.get('size_png'):
        lines.append(f"- size 图：`{outputs['size_png']}`")
    if outputs.get('power_png'):
        lines.append(f"- power 图：`{outputs['power_png']}`")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(output_path)


if __name__ == '__main__':
    main()
