"""
Sup-LR检验模块
实现Andrews-Quandt框架的Sup-LR结构变化检验
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .var_estimator import VAREstimator


class LRTest:
    """LR结构变化检验（针对特定时间点）"""

    def __init__(self):
        """初始化LR检验"""
        self.lr_statistic = None

    def compute_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                            include_const: bool = True) -> Dict[str, Any]:
        """
        针对特定时间点t计算LR统计量

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状为 (T, N)
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            包含LR统计量和相关信息的字典
        """
        T, N = Y.shape

        # 边界验证：确保分段样本量足够
        min_segment_size = p + 2
        if t < min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第一段样本量不足（需要至少{min_segment_size}个观测）")
        if t > T - min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第二段样本量不足（需要至少{min_segment_size}个观测）")

        # H0: 无结构变化（约束模型）
        estimator_r = VAREstimator(method='ols')
        result_r = estimator_r.fit_ols(Y, p, include_const)
        log_lik_r = result_r['log_likelihood']

        # H1: 在时间点t发生结构变化（分段拟合）
        Y1 = Y[:t, :]
        Y2 = Y[t:, :]

        estimator1 = VAREstimator(method='ols')
        result1 = estimator1.fit_ols(Y1, p, include_const)
        log_lik_1 = result1['log_likelihood']

        estimator2 = VAREstimator(method='ols')
        result2 = estimator2.fit_ols(Y2, p, include_const)
        log_lik_2 = result2['log_likelihood']

        # 非约束模型的总对数似然
        log_lik_u = log_lik_1 + log_lik_2

        # LR统计量
        self.lr_statistic = 2 * (log_lik_u - log_lik_r)

        return {
            'lr_statistic': self.lr_statistic,
            'test_point': t,
            'log_lik_restricted': log_lik_r,
            'log_lik_unrestricted': log_lik_u,
            'restricted_result': result_r,
            'segment1_result': result1,
            'segment2_result': result2
        }


class SupLRTest:
    """Sup-LR结构变化检验（遍历所有可能断点）"""

    def __init__(self, trim: float = 0.15):
        """
        初始化Sup-LR检验

        Parameters
        ----------
        trim : float
            修剪比例，用于确定断点搜索范围 [trim*T, (1-trim)*T]
        """
        self.trim = trim
        self.sup_lr = None
        self.estimated_break = None
        self.lr_sequence = None

    def compute_sup_lr(self, Y: np.ndarray, p: int,
                       include_const: bool = True) -> Dict[str, Any]:
        """
        计算Sup-LR统计量

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状为 (T, N)
        p : int
            滞后阶数
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            包含Sup-LR统计量和相关信息的字典
        """
        T, N = Y.shape
        T_eff = T - p

        # 确定断点搜索范围
        tau_min = int(np.ceil(self.trim * T_eff)) + p
        tau_max = int(np.floor((1 - self.trim) * T_eff)) + p

        # 边界验证：确保trim与p组合后搜索范围有效
        min_segment_size = p + 2  # 每段至少需要p+2个观测才能进行有效估计
        if tau_min < min_segment_size:
            tau_min = min_segment_size
        if tau_max > T - min_segment_size:
            tau_max = T - min_segment_size
        if tau_min >= tau_max:
            raise ValueError(
                f"断点搜索范围无效: tau_min={tau_min}, tau_max={tau_max}. "
                f"请增加样本量T或减小trim参数。当前T={T}, p={p}, trim={self.trim}"
            )

        # 计算约束模型（H0：无结构变化）的对数似然
        estimator_r = VAREstimator(method='ols')
        result_r = estimator_r.fit_ols(Y, p, include_const)
        log_lik_r = result_r['log_likelihood']

        # 遍历所有可能的断点
        lr_values = []
        tau_values = []

        for tau in range(tau_min, tau_max + 1):
            # 分段拟合
            # 第一段：[0, tau)
            Y1 = Y[:tau, :]
            if len(Y1) <= p + 1:
                continue

            # 第二段：[tau, T)
            Y2 = Y[tau:, :]
            if len(Y2) <= p + 1:
                continue

            try:
                estimator1 = VAREstimator(method='ols')
                result1 = estimator1.fit_ols(Y1, p, include_const)
                log_lik_1 = result1['log_likelihood']

                estimator2 = VAREstimator(method='ols')
                result2 = estimator2.fit_ols(Y2, p, include_const)
                log_lik_2 = result2['log_likelihood']

                # 非约束模型的总对数似然
                log_lik_u = log_lik_1 + log_lik_2

                # 似然比统计量
                lr = 2 * (log_lik_u - log_lik_r)
                lr_values.append(lr)
                tau_values.append(tau)
            except Exception:
                continue

        if not lr_values:
            raise ValueError("无法计算任何有效的LR统计量")

        # Sup-LR统计量
        lr_values = np.array(lr_values)
        tau_values = np.array(tau_values)

        max_idx = np.argmax(lr_values)
        self.sup_lr = lr_values[max_idx]
        self.estimated_break = tau_values[max_idx]
        self.lr_sequence = (tau_values, lr_values)

        return {
            'sup_lr': self.sup_lr,
            'estimated_break': self.estimated_break,
            'lr_sequence': self.lr_sequence,
            'log_lik_restricted': log_lik_r,
            'restricted_result': result_r
        }

    def test(self, Y: np.ndarray, p: int, alpha: float = 0.05,
             critical_value: Optional[float] = None) -> Dict[str, Any]:
        """
        执行Sup-LR检验

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        alpha : float
            显著性水平
        critical_value : float, optional
            临界值（如果提供则使用，否则需要Bootstrap）

        Returns
        -------
        Dict[str, Any]
            检验结果
        """
        result = self.compute_sup_lr(Y, p)

        if critical_value is not None:
            reject_h0 = result['sup_lr'] > critical_value
            result['reject_h0'] = reject_h0
            result['critical_value'] = critical_value
            result['alpha'] = alpha

        return result
