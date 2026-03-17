# -*- coding: utf-8 -*-
"""
实证应用：iShares国家权益ETF在COVID-19前后的结构性断裂检验（高维低秩模型）

数据说明：
- 22只iShares国家权益ETF日对数收益率
  EWA(澳大利亚), EWC(加拿大), EWG(德国), EWH(香港), EWI(意大利),
  EWJ(日本), EWK(比利时), EWL(瑞士), EWM(马来西亚), EWN(荷兰),
  EWO(奥地利), EWP(西班牙), EWQ(法国), EWS(新加坡), EWT(台湾),
  EWU(英国), EWW(墨西哥), EWY(韩国), EWZ(巴西), FXI(中国),
  INDA(印度), EFA(国际发达市场综合)
- 各国股市收益率由少数全球/区域公共因子驱动 → 低秩VAR系数结构
- 断点：2020-03-11（WHO宣布COVID-19为全球大流行）

检验方案：
- lowrank_rrr (N=22)：RRR + Bootstrap LR，匹配数据的低秩结构
- lowrank_rrr_fv (N=22)：固定行空间RRR + Bootstrap LR
- 1个真实断点 + 4个安慰剂对照
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lowrank_var import LowRankBootstrapInference, RankSelector


# ============================================================
# 配置
# ============================================================

COUNTRY_ETFS = {
    'EWA': '澳大利亚', 'EWC': '加拿大', 'EWG': '德国', 'EWH': '香港',
    'EWI': '意大利', 'EWJ': '日本', 'EWK': '比利时', 'EWL': '瑞士',
    'EWM': '马来西亚', 'EWN': '荷兰', 'EWO': '奥地利', 'EWP': '西班牙',
    'EWQ': '法国', 'EWS': '新加坡', 'EWT': '台湾', 'EWU': '英国',
    'EWW': '墨西哥', 'EWY': '韩国', 'EWZ': '巴西', 'FXI': '中国',
    'INDA': '印度', 'EFA': '国际发达综合',
}

TICKERS = list(COUNTRY_ETFS.keys())

CANDIDATE_BREAKS = {
    'COVID-19 断点': {
        'date': '2020-03-11',
        'data_start': '2019-01-02',
        'data_end': '2021-06-30',
        'desc': 'WHO宣布COVID-19全球大流行，预期拒绝H0',
    },
    '安慰剂1: 2019-07 (断点前远端)': {
        'date': '2019-07-01',
        'data_start': '2019-02-01',
        'data_end': '2019-12-31',
        'desc': '2019年中平稳期，预期不拒绝H0',
    },
    '安慰剂2: 2019-06 (断点前近端)': {
        'date': '2019-06-01',
        'data_start': '2019-02-01',
        'data_end': '2019-09-30',
        'desc': '2019年上半年平稳期（8月关税升级前），预期不拒绝H0',
    },
    '安慰剂3: 2021-08 (断点后近端)': {
        'date': '2021-08-01',
        'data_start': '2021-04-01',
        'data_end': '2021-10-31',
        'desc': '2021年Q3后疫情恢复稳定期（omicron前），预期不拒绝H0',
    },
    '安慰剂4: 2021-09 (断点后远端)': {
        'date': '2021-09-01',
        'data_start': '2021-05-01',
        'data_end': '2021-11-15',
        'desc': '2021年Q3-Q4稳定期（omicron前），预期不拒绝H0',
    },
}

DATA_START = '2017-06-01'
DATA_END = '2022-06-30'


# ============================================================
# 数据获取
# ============================================================

def download_data(cache_path: str = None) -> pd.DataFrame:
    """下载国家ETF价格数据"""
    if cache_path and os.path.exists(cache_path):
        print(f"[数据] 从缓存加载: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    d1 = DATA_START.replace('-', '')
    d2 = DATA_END.replace('-', '')
    print(f"[数据] 从Stooq下载 {len(TICKERS)} 只国家权益ETF...")

    all_prices = {}
    for ticker in TICKERS:
        stooq_ticker = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&d1={d1}&d2={d2}&i=d"
        for attempt in range(3):
            try:
                df = pd.read_csv(url)
                if len(df) > 0 and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    all_prices[ticker] = df['Close']
                    print(f"  {ticker} ({COUNTRY_ETFS[ticker]}): {len(df)} days")
                    break
            except Exception as e:
                print(f"  {ticker}: attempt {attempt+1} failed ({e})")
                time.sleep(3)
            time.sleep(0.5)

    prices = pd.DataFrame(all_prices).dropna()
    print(f"[数据] 共 {prices.shape[0]} 个交易日, {prices.shape[1]} 只ETF")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
        print(f"[数据] 已缓存至 {cache_path}")
    return prices


def prepare_window(log_returns, tickers, break_date,
                   data_start=None, data_end=None):
    """围绕断点截取数据窗口"""
    df = log_returns[tickers].copy()
    if data_start or data_end:
        start = pd.Timestamp(data_start) if data_start else df.index[0]
        end = pd.Timestamp(data_end) if data_end else df.index[-1]
        df = df.loc[start:end]

    break_dt = pd.Timestamp(break_date)
    if break_dt not in df.index:
        mask = df.index >= break_dt
        if mask.any():
            break_dt = df.index[mask][0]
        else:
            raise ValueError(f"断点 {break_date} 超出数据范围")

    break_idx = df.index.get_loc(break_dt)
    return df.values, break_idx, df.index


# ============================================================
# 诊断
# ============================================================

def data_diagnostics(Y, t_break, dates, tickers, label):
    """数据结构诊断"""
    T, N = Y.shape
    print(f"\n{'='*65}")
    print(f"数据诊断: {label}")
    print(f"{'='*65}")
    print(f"  N={N}, T={T}, t={t_break} ({dates[t_break].strftime('%Y-%m-%d')})")
    print(f"  日期范围: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  第一段: t=0..{t_break-1} ({t_break}个观测)")
    print(f"  第二段: t={t_break}..{T-1} ({T-t_break}个观测)")

    # 低秩诊断：协方差矩阵特征值
    cov_mat = np.cov(Y.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]
    cum_var = np.cumsum(eigvals) / np.sum(eigvals)
    print(f"\n  协方差矩阵特征值 (低秩诊断):")
    for k in range(min(N, 6)):
        print(f"    eig[{k+1}] = {eigvals[k]:.6f}  cum = {cum_var[k]:.3f}")
    # 找出到达90%的秩
    rank_90 = np.searchsorted(cum_var, 0.9) + 1
    print(f"  → 前 {rank_90} 个特征值解释 ≥90% 方差 → 低秩结构确认")

    # VAR(1) OLS系数矩阵的奇异值
    if T > N + 2:
        Y_resp = Y[1:]
        Y_lag = Y[:-1]
        X = np.column_stack([np.ones(T-1), Y_lag])
        B = np.linalg.lstsq(X, Y_resp, rcond=None)[0]
        Phi = B[1:, :].T
        sv = np.linalg.svd(Phi, compute_uv=False)
        cum_sv = np.cumsum(sv**2) / np.sum(sv**2)
        print(f"\n  VAR(1) OLS系数矩阵奇异值:")
        for k in range(min(6, len(sv))):
            print(f"    sv[{k+1}] = {sv[k]:.4f}  cum = {cum_sv[k]:.3f}")

    # 断点前后均值差异 (top 5)
    print(f"\n  断点前后均值差异 (top 5):")
    diffs = []
    for i, ticker in enumerate(tickers):
        diff = np.mean(Y[t_break:, i]) - np.mean(Y[:t_break, i])
        diffs.append((ticker, COUNTRY_ETFS.get(ticker, ''), diff))
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    for ticker, name, diff in diffs[:5]:
        print(f"    {ticker:<5} {name:<12} diff = {diff:>+.6f}")


# ============================================================
# 检验
# ============================================================

def test_single_breakpoint(log_returns, break_date, break_label, info,
                           B=500, seed=42, p=1, alpha=0.05, verbose=False,
                           use_fixed_space=False):
    """对单个检验点执行低秩Bootstrap LR检验"""
    print(f"\n{'#'*70}")
    print(f"# {break_label}")
    print(f"# {info['desc']}")
    print(f"{'#'*70}")

    result_row = {'break_label': break_label, 'break_date': break_date}

    try:
        Y, t, dates = prepare_window(
            log_returns, TICKERS, break_date,
            data_start=info.get('data_start'),
            data_end=info.get('data_end')
        )
        data_diagnostics(Y, t, dates, TICKERS, break_label)

        T, N = Y.shape

        # 秩选择
        selector = RankSelector()
        rank_res = selector.select_by_eigenvalue_ratio(Y, p, threshold=0.9)
        rank = rank_res['selected_rank']
        sv = rank_res['singular_values']
        cvr = rank_res['cumulative_variance_ratio']
        print(f"\n  自动选秩: r = {rank} (90%阈值)")
        for k in range(min(len(sv), 5)):
            print(f"    sv[{k+1}] = {sv[k]:.4f}  cum = {cvr[k]:.3f}")

        method_label = 'rrr_fv' if use_fixed_space else 'rrr'
        print(f"\n  [Low-Rank {method_label.upper()}] N={N}, T={T}, t={t}, rank={rank}")

        boot = LowRankBootstrapInference(
            B=B, seed=seed, method='rrr', rank=rank,
            fixed_space=use_fixed_space
        )
        res = boot.test(Y, p, t, alpha=alpha, verbose=verbose)

        print(f"    LR = {res['original_lr']:.2f}")
        print(f"    p值 = {res['p_value']:.4f}")
        print(f"    结论: {'拒绝H0 ***' if res['reject_h0'] else '不拒绝H0'}")
        print(f"    B_effective = {res['B_effective']}/{B}")

        result_row.update({
            'method': f'lowrank_{method_label}',
            'N': N, 'T': T, 't': t, 'rank': rank,
            'lr_stat': res['original_lr'],
            'p_value': res['p_value'],
            'reject_h0': res['reject_h0'],
            'B_effective': res['B_effective'],
            'critical_05': res['critical_values'].get(0.05, np.nan),
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        result_row.update({
            'method': f'lowrank_{"rrr_fv" if use_fixed_space else "rrr"}',
            'lr_stat': np.nan, 'p_value': np.nan, 'reject_h0': None
        })

    return result_row


def print_summary(all_results, method_name):
    """打印汇总表"""
    print(f"\n{'='*80}")
    print(f"汇总: 国家ETF高维低秩断点检验 ({method_name}, N=22)")
    print(f"{'='*80}")
    print(f"数据: 22只iShares国家权益ETF日对数收益率, p=1, B=500, alpha=0.05")
    print()

    header = f"{'检验点':<32} {'T':>4} {'t':>4} {'r':>2} {'LR统计量':>10} {'p值':>8} {'结论':>14}"
    print(header)
    print('-' * 80)

    for r in all_results:
        T = r.get('T', '?')
        t = r.get('t', '?')
        rank = r.get('rank', '?')
        lr = r.get('lr_stat', np.nan)
        pv = r.get('p_value', np.nan)
        reject = r.get('reject_h0')

        lr_s = f"{lr:.2f}" if isinstance(lr, (int, float)) and not np.isnan(lr) else "N/A"
        p_s = f"{pv:.4f}" if isinstance(pv, (int, float)) and not np.isnan(pv) else "N/A"
        if reject is None:
            conclusion = "N/A"
        elif reject:
            conclusion = "拒绝H0 ***"
        else:
            conclusion = "不拒绝H0"

        print(f"{r['break_label']:<32} {T:>4} {t:>4} {rank:>2} {lr_s:>10} {p_s:>8} {conclusion:>14}")

    print('-' * 80)
    print("*** p <= 0.05")
    print()


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='高维低秩断点检验: N=22 iShares国家权益ETF'
    )
    parser.add_argument('--B', type=int, default=500,
                        help='Bootstrap重复次数 (default: 500)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--fixed-space', action='store_true',
                        help='使用固定行空间RRR (lowrank_rrr_fv)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / 'results' / 'empirical')

    cache_path = str(PROJECT_ROOT / 'applications' / 'data_cache'
                     / 'country_etf_prices.csv')

    method_name = 'Low-Rank RRR (fixed V)' if args.fixed_space else 'Low-Rank RRR'

    print("=" * 70)
    print(f"高维低秩断点检验: N=22 国家权益ETF ({method_name})")
    print("=" * 70)

    prices = download_data(cache_path=cache_path)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"[数据] 对数收益率: {log_returns.shape[0]} 天, {log_returns.shape[1]} 只ETF")

    # 全样本结构诊断
    print(f"\n{'='*65}")
    print("全样本低秩结构诊断")
    print(f"{'='*65}")
    Y_all = log_returns.values
    cov_all = np.cov(Y_all.T)
    eigvals_all = np.sort(np.linalg.eigvalsh(cov_all))[::-1]
    cum_all = np.cumsum(eigvals_all) / np.sum(eigvals_all)
    for k in range(min(6, len(eigvals_all))):
        print(f"  eig[{k+1}] = {eigvals_all[k]:.6f}  cum = {cum_all[k]:.3f}")

    # 逐个检验点
    all_results = []
    for label, info in CANDIDATE_BREAKS.items():
        row = test_single_breakpoint(
            log_returns, info['date'], label, info,
            B=args.B, seed=args.seed, verbose=args.verbose,
            use_fixed_space=args.fixed_space
        )
        all_results.append(row)

    print_summary(all_results, method_name)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'fv' if args.fixed_space else 'rrr'
    out_file = os.path.join(args.output_dir,
                            f'country_lowrank_{tag}_{timestamp}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Country ETF High-Dim Low-Rank Structural Break Test',
            'model': f'lowrank_{tag}',
            'N': len(TICKERS),
            'tickers': TICKERS,
            'country_names': COUNTRY_ETFS,
            'fixed_space': args.fixed_space,
            'B': args.B,
            'seed': args.seed,
            'timestamp': timestamp,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[结果已保存] {out_file}")


if __name__ == '__main__':
    main()
