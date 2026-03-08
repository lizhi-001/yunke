"""
低秩VAR的LR检验模块
使用核范数正则化/截断SVD估计进行已知断点下的结构断裂检验。

口径说明：
- H0：已知点处不存在结构断裂，整条时间序列共用一套参数；
- H1：已知点处存在结构断裂，断点前后使用两套参数；
- 第二段从断点后第一个响应开始生效，但允许其滞后项借用断点前的 p 个观测。
"""

import numpy as np
from typing import Dict, Any, Optional
from .nuclear_norm import NuclearNormVAR
from .rank_selection import RankSelector


class LowRankLRTest:
    """低秩VAR的LR结构变化检验（针对已知时间点t）"""

    def __init__(self, method: str = 'svd', rank: Optional[int] = None,
                 lambda_nuc: Optional[float] = None):
        """
        Parameters
        ----------
        method : str
            估计方法：'svd'（截断SVD）或 'cvxpy'（核范数正则化）
        rank : int, optional
            指定秩（仅用于svd方法），None则自动选择
        lambda_nuc : float, optional
            核范数正则化参数（仅用于cvxpy方法）
        """
        self.method = method
        self.rank = rank
        self.lambda_nuc = lambda_nuc
        self.lr_statistic = None

    def _fit_model(self, Y: np.ndarray, p: int,
                   include_const: bool = True) -> Dict[str, Any]:
        """使用低秩方法拟合VAR模型"""
        estimator = NuclearNormVAR(lambda_nuc=self.lambda_nuc)

        rank = self.rank
        if self.method == 'svd':
            if rank is None:
                selector = RankSelector()
                rank_result = selector.select_by_information_criterion(
                    Y, p, max_rank=min(Y.shape[1], 10), criterion='bic'
                )
                rank = rank_result['selected_rank']
            return estimator.fit_svd(Y, p, rank=rank, include_const=include_const)
        else:
            return estimator.fit_cvxpy(Y, p, include_const=include_const)

    def compute_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                            include_const: bool = True) -> Dict[str, Any]:
        """
        针对已知时间点t计算LR统计量（使用低秩估计）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状为 (T, N)
        p : int
            滞后阶数
        t : int
            已知的变点位置
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            包含LR统计量和相关信息的字典
        """
        T, N = Y.shape

        min_segment_size = p + 2
        if t < min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第一段样本量不足")
        if t > T - min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第二段样本量不足")

        # H0: 无结构变化（约束模型）
        result_r = self._fit_model(Y, p, include_const)
        log_lik_r = result_r['log_likelihood']

        # H1: 在已知时间点t发生结构变化（结构断裂拟合）
        Y1 = Y[:t, :]
        Y2 = Y[t - p:, :]

        result1 = self._fit_model(Y1, p, include_const)
        log_lik_1 = result1['log_likelihood']

        result2 = self._fit_model(Y2, p, include_const)
        log_lik_2 = result2['log_likelihood']

        if result_r['T_eff'] != result1['T_eff'] + result2['T_eff']:
            raise ValueError('H0 与 H1 的有效样本量不一致')

        log_lik_u = log_lik_1 + log_lik_2

        self.lr_statistic = 2 * (log_lik_u - log_lik_r)

        return {
            'lr_statistic': self.lr_statistic,
            'test_point': t,
            'log_lik_restricted': log_lik_r,
            'log_lik_unrestricted': log_lik_u,
            'restricted_result': result_r,
            'segment1_result': result1,
            'segment2_result': result2,
            'method': self.method
        }
