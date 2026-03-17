# -*- coding: utf-8 -*-
"""
实证应用：跨大类资产全球ETF在COVID-19前后的结构性断裂检验（高维稀疏模型）

数据说明：
- 20只覆盖不同大类资产的ETF日对数收益率
  权益: SPY(美国大盘), IWM(美国小盘), QQQ(纳斯达克100), EFA(国际发达), EEM(新兴市场)
  固收: AGG(综合债券), TLT(长期国债), TIP(通胀保护), LQD(投资级公司债), HYG(高收益债)
  商品: GLD(黄金), SLV(白银), USO(原油), DBA(农产品), DBC(商品综合)
  地产: VNQ(REIT), IYR(房地产)
  汇率: UUP(美元指数), FXE(欧元), FXY(日元)
- 不同大类资产的收益驱动因素差异极大 → VAR系数矩阵天然稀疏
- 断点：2020-03-11（WHO宣布COVID-19为全球大流行）

检验方案：
- sparse_lasso (N=20)：LassoCV + Post-Lasso OLS + Bootstrap LR
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

from sparse_var import SparseBootstrapInference
from sparse_var.lasso_var import LassoVAREstimator


# ============================================================
# 配置
# ============================================================

CROSS_ASSET_ETFS = {
    # 权益 (5)
    'SPY': '美国大盘股', 'IWM': '美国小盘股', 'QQQ': '纳斯达克100',
    'EFA': '国际发达市场', 'EEM': '新兴市场',
    # 固收 (5)
    'AGG': '综合债券', 'TLT': '长期国债', 'TIP': '通胀保护债券',
    'LQD': '投资级公司债', 'HYG': '高收益债',
    # 商品 (5)
    'GLD': '黄金', 'SLV': '白银', 'USO': '原油',
    'DBA': '农产品', 'DBC': '商品综合',
    # 房地产 (2)
    'VNQ': '美国REIT', 'IYR': '美国房地产',
    # 汇率 (3)
    'UUP': '美元指数', 'FXE': '欧元', 'FXY': '日元',
}

TICKERS = list(CROSS_ASSET_ETFS.keys())

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
    """下载跨大类资产ETF价格数据"""
    if cache_path and os.path.exists(cache_path):
        print(f"[数据] 从缓存加载: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    d1 = DATA_START.replace('-', '')
    d2 = DATA_END.replace('-', '')
    print(f"[数据] 从Stooq下载 {len(TICKERS)} 只跨大类资产ETF...")

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
                    print(f"  {ticker} ({CROSS_ASSET_ETFS[ticker]}): {len(df)} days")
                    break
            except Exception as e:
                print(f"  {ticker}: attempt {attempt+1} failed ({e})")
                time.sleep(3)
            time.sleep(0.5)

    prices = pd.DataFrame(all_prices).dropna()
    print(f"[数据] 共 {prices.shape[0]} 个交易日, {prices.shape[1]} 只ETF")

    # 检查是否有缺失的ticker
    missing = set(TICKERS) - set(prices.columns)
    if missing:
        print(f"[警告] 以下ETF下载失败，将从列表中移除: {missing}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
        print(f"[数据] 已缓存至 {cache_path}")
    return prices


def prepare_window(log_returns, tickers, break_date,
                   data_start=None, data_end=None):
    """围绕断点截取数据窗口"""
    available = [t for t in tickers if t in log_returns.columns]
    df = log_returns[available].copy()

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
    return df.values, break_idx, df.index, available


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

    # 稀疏性诊断：OLS系数矩阵的非零比例
    if T > N + 2:
        Y_resp = Y[1:]
        Y_lag = Y[:-1]
        X = np.column_stack([np.ones(T-1), Y_lag])
        B = np.linalg.lstsq(X, Y_resp, rcond=None)[0]
        Phi = B[1:, :].T
        for thr in [0.01, 0.05, 0.1]:
            nz = np.sum(np.abs(Phi) > thr)
            print(f"  OLS |coeff| > {thr}: {nz}/{N*N} ({100*nz/(N*N):.0f}%)")

    # 协方差特征值（确认非低秩）
    cov_mat = np.cov(Y.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]
    cum_var = np.cumsum(eigvals) / np.sum(eigvals)
    print(f"\n  协方差矩阵特征值 (确认非低秩):")
    for k in range(min(6, N)):
        print(f"    eig[{k+1}] = {eigvals[k]:.6f}  cum = {cum_var[k]:.3f}")

    # 断点前后均值差异 (top 5)
    print(f"\n  断点前后均值差异 (top 5):")
    diffs = []
    for i, ticker in enumerate(tickers):
        diff = np.mean(Y[t_break:, i]) - np.mean(Y[:t_break, i])
        diffs.append((ticker, CROSS_ASSET_ETFS.get(ticker, ''), diff))
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    for ticker, name, diff in diffs[:5]:
        print(f"    {ticker:<5} {name:<14} diff = {diff:>+.6f}")


# ============================================================
# 检验
# ============================================================

def test_single_breakpoint(log_returns, tickers, break_date, break_label, info,
                           B=500, seed=42, p=1, alpha=0.05, verbose=False):
    """对单个检验点执行稀疏Bootstrap LR检验"""
    print(f"\n{'#'*70}")
    print(f"# {break_label}")
    print(f"# {info['desc']}")
    print(f"{'#'*70}")

    result_row = {'break_label': break_label, 'break_date': break_date}

    try:
        Y, t, dates, available_tickers = prepare_window(
            log_returns, tickers, break_date,
            data_start=info.get('data_start'),
            data_end=info.get('data_end')
        )
        data_diagnostics(Y, t, dates, available_tickers, break_label)

        N = Y.shape[1]
        T = Y.shape[0]

        # 两阶段固定支撑策略：先CV选alpha，再用该alpha固定支撑集做所有拟合
        # fixed_support=True 对应 simulation_plan 中 sparse_lasso 方法：
        # 全样本一次确定支撑集 S，H0/H1/Bootstrap 统一使用 OLS on S，
        # 消除自适应选择自由度带来的 type I error 膨胀
        est_cv = LassoVAREstimator(alpha=None)
        cv_result = est_cv.fit(Y, p)
        cv_alpha = float(np.median(cv_result['alphas_used']))
        sparsity = cv_result['sparsity']
        print(f"\n  [Sparse Lasso, fixed_support] N={N}, T={T}, t={t}")
        print(f"    CV alpha (median): {cv_alpha:.2e}")
        print(f"    Lasso sparsity: {sparsity:.3f} ({100*sparsity:.0f}% 零系数)")

        boot = SparseBootstrapInference(
            B=B, seed=seed,
            estimator_type='lasso', alpha=cv_alpha,
            fixed_support=True   # 固定支撑集 Post-Lasso OLS，对应 simulation_plan sparse_lasso
        )
        res = boot.test(Y, p, t, alpha=alpha, verbose=verbose)

        print(f"    LR = {res['original_lr']:.2f}")
        print(f"    p值 = {res['p_value']:.4f}")
        print(f"    结论: {'拒绝H0 ***' if res['reject_h0'] else '不拒绝H0'}")
        print(f"    B_effective = {res['B_effective']}/{B}")

        result_row.update({
            'N': N, 'T': T, 't': t,
            'cv_alpha': cv_alpha,
            'sparsity': sparsity,
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
            'lr_stat': np.nan, 'p_value': np.nan, 'reject_h0': None
        })

    return result_row


def print_summary(all_results):
    """打印汇总表"""
    print(f"\n{'='*85}")
    print(f"汇总: 跨大类资产高维稀疏断点检验 (Sparse Lasso + Post-Lasso OLS, N=20)")
    print(f"{'='*85}")
    print(f"数据: 20只跨大类资产全球ETF日对数收益率, p=1, B=500, alpha=0.05")
    print()

    header = (f"{'检验点':<32} {'T':>4} {'t':>4} {'CV_α':>10} "
              f"{'稀疏度':>6} {'LR统计量':>10} {'p值':>8} {'结论':>14}")
    print(header)
    print('-' * 85)

    for r in all_results:
        T = r.get('T', '?')
        t = r.get('t', '?')
        cv_a = r.get('cv_alpha', np.nan)
        sp = r.get('sparsity', np.nan)
        lr = r.get('lr_stat', np.nan)
        pv = r.get('p_value', np.nan)
        reject = r.get('reject_h0')

        cv_s = f"{cv_a:.1e}" if isinstance(cv_a, (int, float)) and not np.isnan(cv_a) else "N/A"
        sp_s = f"{sp:.3f}" if isinstance(sp, (int, float)) and not np.isnan(sp) else "N/A"
        lr_s = f"{lr:.2f}" if isinstance(lr, (int, float)) and not np.isnan(lr) else "N/A"
        p_s = f"{pv:.4f}" if isinstance(pv, (int, float)) and not np.isnan(pv) else "N/A"
        if reject is None:
            conclusion = "N/A"
        elif reject:
            conclusion = "拒绝H0 ***"
        else:
            conclusion = "不拒绝H0"

        print(f"{r['break_label']:<32} {T:>4} {t:>4} {cv_s:>10} "
              f"{sp_s:>6} {lr_s:>10} {p_s:>8} {conclusion:>14}")

    print('-' * 85)
    print("*** p <= 0.05")
    print()


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='高维稀疏断点检验: N=20 跨大类资产全球ETF'
    )
    parser.add_argument('--B', type=int, default=500,
                        help='Bootstrap重复次数 (default: 500)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / 'results' / 'empirical')

    cache_path = str(PROJECT_ROOT / 'applications' / 'data_cache'
                     / 'cross_asset_highdim_prices.csv')

    print("=" * 70)
    print("高维稀疏断点检验: N=20 跨大类资产全球ETF")
    print("(Sparse Lasso + Post-Lasso OLS + Bootstrap LR)")
    print("=" * 70)

    prices = download_data(cache_path=cache_path)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    available_tickers = [t for t in TICKERS if t in log_returns.columns]
    print(f"[数据] 对数收益率: {log_returns.shape[0]} 天, "
          f"{len(available_tickers)}/{len(TICKERS)} 只ETF可用")

    if len(available_tickers) < len(TICKERS):
        missing = set(TICKERS) - set(available_tickers)
        print(f"[警告] 缺失: {missing}")

    # 全样本结构诊断
    print(f"\n{'='*65}")
    print("全样本稀疏结构诊断")
    print(f"{'='*65}")
    Y_all = log_returns[available_tickers].values
    cov_all = np.cov(Y_all.T)
    eigvals_all = np.sort(np.linalg.eigvalsh(cov_all))[::-1]
    cum_all = np.cumsum(eigvals_all) / np.sum(eigvals_all)
    N_all = len(available_tickers)
    print(f"  N = {N_all}")
    for k in range(min(6, N_all)):
        print(f"  eig[{k+1}] = {eigvals_all[k]:.6f}  cum = {cum_all[k]:.3f}")

    # 逐个检验点
    all_results = []
    for label, info in CANDIDATE_BREAKS.items():
        row = test_single_breakpoint(
            log_returns, available_tickers, break_date=info['date'],
            break_label=label, info=info,
            B=args.B, seed=args.seed, verbose=args.verbose
        )
        all_results.append(row)

    print_summary(all_results)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(args.output_dir,
                            f'cross_asset_highdim_sparse_{timestamp}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Cross-Asset High-Dim Sparse Structural Break Test',
            'model': 'sparse_lasso',
            'N': len(available_tickers),
            'tickers': available_tickers,
            'asset_names': {t: CROSS_ASSET_ETFS[t] for t in available_tickers},
            'B': args.B,
            'seed': args.seed,
            'timestamp': timestamp,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[结果已保存] {out_file}")


if __name__ == '__main__':
    main()
