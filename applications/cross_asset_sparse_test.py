# -*- coding: utf-8 -*-
"""
实证应用：跨资产类别ETF在COVID-19前后的结构性断裂检验（稀疏模型）

数据说明：
- 9只覆盖不同大类资产的ETF日对数收益率
  SPY(美股), EFA(国际股), EEM(新兴市场), AGG(债券), TLT(长期国债),
  TIP(通胀保护), GLD(黄金), VNQ(REIT), USO(原油)
- 不同资产类别间的VAR系数天然稀疏（驱动因素不同）
- 协方差第一特征值仅解释55%，非低秩结构
- 断点：2020-03-11（WHO宣布COVID-19为全球大流行）

检验方案：
- sparse_lasso (N=9)：核心方法，匹配数据的稀疏结构
- 断点/安慰剂对照（安慰剂选在2019年纯平静期内）
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

from sparse_var import SparseBootstrapInference


# ============================================================
# 配置
# ============================================================

CROSS_ASSET_ETFS = {
    'SPY': 'US Equity',
    'AGG': 'US Agg Bond',
    'TLT': 'Long-Term Treasury',
    'GLD': 'Gold',
    'VNQ': 'US REIT',
}

TICKERS = list(CROSS_ASSET_ETFS.keys())

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
        'desc': 'Mid-2019, purely calm pre-COVID period',
    },
    'Placebo: 2019-04 (calm)': {
        'date': '2019-04-15',
        'data_start': '2019-02-01',
        'data_end': '2019-12-31',
        'desc': 'Mid-Apr 2019, same calm window different break',
    },
}

DATA_START = '2017-06-01'
DATA_END = '2021-12-31'
PRE_WINDOW = 250
POST_WINDOW = 250


# ============================================================
# 数据获取
# ============================================================

def download_data(cache_path: str = None) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        print(f"[数据] 从缓存加载: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import time
    d1 = DATA_START.replace('-', '')
    d2 = DATA_END.replace('-', '')
    print(f"[数据] 从Stooq下载 {len(TICKERS)} 只跨资产ETF...")

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
                    print(f"  {ticker}: {len(df)} days")
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
    return prices


def prepare_window(log_returns, tickers, break_date, pre_window, post_window,
                   data_start=None, data_end=None):
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

    # 稀疏性诊断
    if T > N + 2:
        Y_resp = Y[1:]
        Y_lag = Y[:-1]
        X = np.column_stack([np.ones(T-1), Y_lag])
        B = np.linalg.lstsq(X, Y_resp, rcond=None)[0]
        Phi = B[1:, :].T
        for thr in [0.05, 0.1]:
            nz = np.sum(np.abs(Phi) > thr)
            print(f"  OLS |coeff|>{thr}: {nz}/{N*N} ({100*nz/(N*N):.0f}%)")

    # 协方差特征值（确认非低秩）
    cov_mat = np.cov(Y.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]
    cum_var = np.cumsum(eigvals) / np.sum(eigvals)
    print(f"  协方差特征值:")
    for k in range(min(4, N)):
        print(f"    eig[{k+1}]={eigvals[k]:.6f}  cum={cum_var[k]:.3f}")

    # 断点前后均值差异 (top 5)
    print(f"  断点前后均值差异 (top 5):")
    diffs = []
    for i, ticker in enumerate(tickers):
        diff = np.mean(Y[t_break:, i]) - np.mean(Y[:t_break, i])
        diffs.append((ticker, CROSS_ASSET_ETFS.get(ticker, ''), diff))
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    for ticker, name, diff in diffs[:5]:
        print(f"    {ticker:<5} {name:<16} diff={diff:>+.6f}")


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

        N = Y.shape[1]
        T = Y.shape[0]

        # 两阶段策略：先CV选alpha，再固定alpha做bootstrap
        from sparse_var.lasso_var import LassoVAREstimator
        est_cv = LassoVAREstimator(alpha=None)
        cv_result = est_cv.fit(Y, p)
        # 取各方程CV alpha的中位数作为统一alpha
        cv_alpha = float(np.median(cv_result['alphas_used']))
        print(f"\n  [Sparse Lasso] N={N}, T={T}, t={t}")
        print(f"    CV alpha (median): {cv_alpha:.6f}, sparsity: {cv_result['sparsity']:.3f}")

        boot = SparseBootstrapInference(B=B, seed=seed,
                                         estimator_type='lasso', alpha=cv_alpha,
                                         post_lasso_ols=True)
        res = boot.test(Y, p, t, alpha=alpha, verbose=verbose)

        print(f"    LR={res['original_lr']:.2f}, p={res['p_value']:.4f}, "
              f"reject={res['reject_h0']}, B_eff={res['B_effective']}/{B}")

        result_row.update({
            'N': N, 'T': T, 't': t,
            'lr_stat': res['original_lr'],
            'p_value': res['p_value'],
            'reject_h0': res['reject_h0'],
            'B_effective': res['B_effective']
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        result_row.update({'lr_stat': np.nan, 'p_value': np.nan, 'reject_h0': None})

    return result_row


def print_summary(all_results):
    print(f"\n{'='*75}")
    print("汇总: 跨资产类别ETF结构性断裂检验 (Sparse Lasso, N=5)")
    print(f"{'='*75}")
    print(f"数据: 5只跨资产ETF日对数收益率, p=1, B=500, alpha=0.05")
    print()

    print(f"{'断点':<28} {'T':>4} {'t':>4} {'LR':>10} {'p值':>8} {'结论':>12}")
    print('-' * 75)
    for r in all_results:
        T = r.get('T', '?')
        t = r.get('t', '?')
        lr = r.get('lr_stat', np.nan)
        pv = r.get('p_value', np.nan)
        reject = r.get('reject_h0')

        lr_s = f"{lr:.2f}" if not (isinstance(lr, float) and np.isnan(lr)) else "N/A"
        p_s = f"{pv:.4f}" if not (isinstance(pv, float) and np.isnan(pv)) else "N/A"
        conclusion = "Reject H0 *" if reject else ("Not reject" if reject is not None else "N/A")

        print(f"{r['break_label']:<28} {T:>4} {t:>4} {lr_s:>10} {p_s:>8} {conclusion:>12}")

    print('-' * 75)
    print("* p <= 0.05")


# ============================================================
# 主流程
# ============================================================

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

    cache_path = str(PROJECT_ROOT / 'applications' / 'data_cache' / 'cross_asset_etf_prices.csv')

    print("=" * 70)
    print("跨资产类别ETF COVID-19结构性断裂检验 (Sparse Lasso)")
    print("=" * 70)

    prices = download_data(cache_path=cache_path)
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
    out_file = os.path.join(args.output_dir, f'cross_asset_sparse_{timestamp}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Cross-Asset ETF Sparse Structural Break Test',
            'model': 'sparse_lasso', 'N': len(TICKERS),
            'tickers': TICKERS, 'timestamp': timestamp,
            'results': all_results
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[结果已保存] {out_file}")


if __name__ == '__main__':
    main()
