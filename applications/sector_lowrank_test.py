# -*- coding: utf-8 -*-
"""
实证应用：美国行业ETF在COVID-19前后的结构性断裂检验（低秩模型）

数据说明：
- 11只SPDR行业ETF的日对数收益率
- 行业收益率由少数市场公共因子驱动 → 低秩VAR系数结构
- 断点：2020-03-11（WHO宣布COVID-19为全球大流行）

检验方案：
- lowrank_svd (N=11)：核心方法，匹配数据的低秩结构
- 断点/安慰剂对照
"""

import sys
import os
import json
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

SECTOR_ETFS = {
    'XLB': 'Materials', 'XLC': 'Communication', 'XLE': 'Energy',
    'XLF': 'Financials', 'XLI': 'Industrials', 'XLK': 'Technology',
    'XLP': 'Cons. Staples', 'XLRE': 'Real Estate', 'XLU': 'Utilities',
    'XLV': 'Health Care', 'XLY': 'Cons. Discret.',
}

TICKERS = list(SECTOR_ETFS.keys())

CANDIDATE_BREAKS = {
    'COVID-19 (WHO)': {
        'date': '2020-03-11',
        'data_start': '2019-01-02',
        'data_end': '2021-06-30',
        'desc': 'WHO declares pandemic - should reject',
    },
    'Placebo: 2019-07 (calm)': {
        'date': '2019-07-01',
        'data_start': '2019-02-01',
        'data_end': '2019-12-31',
        'desc': 'Mid-2019, purely calm period',
    },
    'Placebo: 2021-07 (calm)': {
        'date': '2021-07-01',
        'data_start': '2021-02-01',
        'data_end': '2021-12-31',
        'desc': 'Mid-2021, post-COVID stabilized',
    },
}

PRE_WINDOW = 250
POST_WINDOW = 250


# ============================================================
# 数据获取
# ============================================================

def download_sector_data(cache_path: str = None) -> pd.DataFrame:
    """下载行业ETF数据"""
    if cache_path and os.path.exists(cache_path):
        print(f"[数据] 从缓存加载: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import time
    print(f"[数据] 从Stooq下载 {len(TICKERS)} 只行业ETF...")

    all_prices = {}
    for ticker in TICKERS:
        stooq_ticker = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&d1=20170601&d2=20211231&i=d"
        for attempt in range(3):
            try:
                df = pd.read_csv(url)
                if len(df) > 0 and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    all_prices[ticker] = df['Close']
                    print(f"  {ticker}: {len(df)} days")
                    break
            except Exception as e:
                print(f"  {ticker}: attempt {attempt+1} failed ({e})")
                time.sleep(3)
            time.sleep(0.5)

    prices = pd.DataFrame(all_prices).dropna()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def prepare_window(log_returns, tickers, break_date, pre_window, post_window,
                   data_start=None, data_end=None):
    """围绕断点截取窗口"""
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
    start_idx = max(0, break_idx - pre_window)
    end_idx = min(len(df), break_idx + post_window)
    df_win = df.iloc[start_idx:end_idx]
    t_break = break_idx - start_idx
    return df_win.values, t_break, df_win.index


# ============================================================
# 诊断
# ============================================================

def data_diagnostics(Y, t_break, dates, tickers, label):
    T, N = Y.shape
    print(f"\n{'='*60}")
    print(f"数据诊断: {label}")
    print(f"{'='*60}")
    print(f"  N={N}, T={T}, t={t_break} ({dates[t_break].strftime('%Y-%m-%d')})")
    print(f"  日期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

    # 低秩诊断
    cov_mat = np.cov(Y.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    print(f"\n  协方差矩阵特征值 (低秩诊断):")
    for k in range(min(N, 5)):
        print(f"    第{k+1}特征值: {eigenvalues[k]:.6f}  累积方差: {cum_var[k]:.3f}")

    # 断点前后均值
    print(f"\n  断点前后均值差异 (top 5):")
    diffs = []
    for i, ticker in enumerate(tickers):
        diff = np.mean(Y[t_break:, i]) - np.mean(Y[:t_break, i])
        diffs.append((ticker, diff))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for ticker, diff in diffs[:5]:
        print(f"    {ticker:<6} {SECTOR_ETFS.get(ticker, ''):<16} diff={diff:>+.6f}")


# ============================================================
# 检验
# ============================================================

def test_single_breakpoint(log_returns, break_date, break_label, info,
                           B=500, seed=42, p=1, alpha=0.05, verbose=False):
    print(f"\n{'#'*70}")
    print(f"# {break_label} ({break_date})")
    print(f"{'#'*70}")

    result_row = {'break_label': break_label, 'break_date': break_date}

    try:
        Y, t, dates = prepare_window(
            log_returns, TICKERS, break_date,
            PRE_WINDOW, POST_WINDOW,
            data_start=info.get('data_start'),
            data_end=info.get('data_end')
        )
        data_diagnostics(Y, t, dates, TICKERS, break_label)

        # 秩选择
        selector = RankSelector()
        rank_res = selector.select_by_eigenvalue_ratio(Y, p, threshold=0.9)
        rank = rank_res['selected_rank']
        sv = rank_res['singular_values']
        cvr = rank_res['cumulative_variance_ratio']
        print(f"\n  自动选秩 r={rank} (90%阈值)")
        for k in range(min(len(sv), 4)):
            print(f"    sv[{k+1}]={sv[k]:.4f}  累积={cvr[k]:.3f}")

        # 检验
        T, N = Y.shape
        print(f"\n  [Low-Rank SVD] N={N}, T={T}, t={t}, rank={rank}")
        boot = LowRankBootstrapInference(B=B, seed=seed, method='svd', rank=rank)
        res = boot.test(Y, p, t, alpha=alpha, verbose=verbose)

        print(f"    LR={res['original_lr']:.2f}, p={res['p_value']:.4f}, "
              f"reject={res['reject_h0']}, B_eff={res['B_effective']}/{B}")

        result_row['N'] = N
        result_row['T'] = T
        result_row['t'] = t
        result_row['rank'] = rank
        result_row['lr_stat'] = res['original_lr']
        result_row['p_value'] = res['p_value']
        result_row['reject_h0'] = res['reject_h0']
        result_row['B_effective'] = res['B_effective']

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        result_row['lr_stat'] = np.nan
        result_row['p_value'] = np.nan
        result_row['reject_h0'] = None

    return result_row


def print_summary(all_results):
    print(f"\n{'='*75}")
    print("汇总: 行业ETF结构性断裂检验 (Low-Rank SVD, N=11)")
    print(f"{'='*75}")
    print(f"数据: 11只SPDR行业ETF日对数收益率, p=1, B=500, alpha=0.05")
    print()

    print(f"{'断点':<28} {'T':>4} {'t':>4} {'r':>2} {'LR':>10} {'p值':>8} {'结论':>12}")
    print('-' * 75)
    for r in all_results:
        T = r.get('T', '?')
        t = r.get('t', '?')
        rank = r.get('rank', '?')
        lr = r.get('lr_stat', np.nan)
        p = r.get('p_value', np.nan)
        reject = r.get('reject_h0')

        lr_s = f"{lr:.2f}" if not (isinstance(lr, float) and np.isnan(lr)) else "N/A"
        p_s = f"{p:.4f}" if not (isinstance(p, float) and np.isnan(p)) else "N/A"
        conclusion = "Reject H0 *" if reject is True else ("Not reject" if reject is False else "N/A")

        print(f"{r['break_label']:<28} {T:>4} {t:>4} {rank:>2} {lr_s:>10} {p_s:>8} {conclusion:>12}")

    print('-' * 75)
    print("* p <= 0.05")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / 'results' / 'empirical')

    # Use existing cache if available
    cache_path = str(PROJECT_ROOT / 'applications' / 'data_cache' / 'sector_etf_extended.csv')

    print("=" * 70)
    print("行业ETF COVID-19结构性断裂检验 (Low-Rank SVD)")
    print("=" * 70)

    prices = download_sector_data(cache_path=cache_path)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"[数据] 对数收益率: {log_returns.shape[0]} 天")

    all_results = []
    for label, info in CANDIDATE_BREAKS.items():
        row = test_single_breakpoint(
            log_returns, info['date'], label, info,
            B=args.B, seed=args.seed, verbose=args.verbose
        )
        all_results.append(row)

    print_summary(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(args.output_dir, f'sector_lowrank_{timestamp}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Sector ETF Low-Rank Structural Break Test',
            'model': 'lowrank_svd',
            'N': len(TICKERS),
            'tickers': TICKERS,
            'timestamp': timestamp,
            'results': all_results
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[结果已保存] {out_file}")


if __name__ == '__main__':
    main()
