"""
低秩VAR的Bootstrap推断模块
用于低秩VAR模型的已知断点结构变化检验
"""

import numpy as np
from typing import Dict, Any, Optional
from .lowrank_lr_test import LowRankLRTest


class LowRankBootstrapInference:
    """低秩VAR的Bootstrap推断"""

    def __init__(self, B: int = 500, seed: Optional[int] = None,
                 method: str = 'svd', rank: Optional[int] = None,
                 lambda_nuc: Optional[float] = None):
        """
        Parameters
        ----------
        B : int
            Bootstrap重复次数
        seed : int, optional
            随机种子
        method : str
            估计方法：'svd' 或 'cvxpy'
        rank : int, optional
            指定秩（仅用于svd方法）
        lambda_nuc : float, optional
            核范数正则化参数
        """
        self.B = B
        self.seed = seed
        self.method = method
        self.rank = rank
        self.lambda_nuc = lambda_nuc
        self.rng = np.random.default_rng(seed)

    def generate_pseudo_series(self, Y: np.ndarray, p: int,
                                Phi: np.ndarray, c: np.ndarray,
                                residuals: np.ndarray) -> np.ndarray:
        """生成Bootstrap伪序列（残差重抽样）"""
        T, N = Y.shape
        T_eff = len(residuals)

        centered_residuals = residuals - np.mean(residuals, axis=0)
        indices = self.rng.choice(T_eff, size=T_eff, replace=True)
        resampled_residuals = centered_residuals[indices, :]

        Y_star = np.zeros((T, N))
        Y_star[:p, :] = Y[:p, :]

        for t in range(p, T):
            Y_lag_ordered = np.zeros(N * p)
            for lag in range(p):
                Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]

            epsilon_t = resampled_residuals[t - p, :] if t - p < T_eff else np.zeros(N)
            Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t

        return Y_star

    def test(self, Y: np.ndarray, p: int, t: int,
             alpha: float = 0.05, verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的Bootstrap LR检验（低秩版本）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        t : int
            已知的变点位置
        alpha : float
            显著性水平
        verbose : bool
            是否显示进度
        """
        # Step 1: 计算原始数据的LR统计量
        lr_test = LowRankLRTest(method=self.method, rank=self.rank,
                                 lambda_nuc=self.lambda_nuc)
        original_result = lr_test.compute_lr_at_point(Y, p, t)
        original_lr = original_result['lr_statistic']

        # 获取H0下的估计结果
        restricted_result = original_result['restricted_result']
        Phi_r = restricted_result['Phi']
        c_r = restricted_result['c']
        residuals_r = restricted_result['residuals']

        # Step 2: Bootstrap循环
        bootstrap_lr_values = []

        for b in range(self.B):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{self.B}")

            try:
                Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)
                lr_test_b = LowRankLRTest(method=self.method, rank=self.rank,
                                           lambda_nuc=self.lambda_nuc)
                result_b = lr_test_b.compute_lr_at_point(Y_star, p, t)
                bootstrap_lr_values.append(result_b['lr_statistic'])
            except Exception:
                continue

        bootstrap_statistics = np.array(bootstrap_lr_values)

        # Step 3: 计算p值和临界值
        p_value = np.mean(bootstrap_statistics >= original_lr) if len(bootstrap_statistics) > 0 else np.nan

        if len(bootstrap_statistics) > 0:
            critical_values = {
                0.10: np.percentile(bootstrap_statistics, 90),
                0.05: np.percentile(bootstrap_statistics, 95),
                0.01: np.percentile(bootstrap_statistics, 99)
            }
        else:
            critical_values = {0.10: np.nan, 0.05: np.nan, 0.01: np.nan}

        reject_h0 = p_value <= alpha

        return {
            'original_lr': original_lr,
            'test_point': t,
            'p_value': p_value,
            'critical_values': critical_values,
            'bootstrap_statistics': bootstrap_statistics,
            'B_effective': len(bootstrap_statistics),
            'alpha': alpha,
            'reject_h0': reject_h0,
            'decision': (f"拒绝H0：在时间点{t}存在结构性变化"
                        if reject_h0 else "接受H0：不存在结构性变化"),
            'method': self.method
        }
