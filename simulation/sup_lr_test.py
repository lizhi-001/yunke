"""
Sup-LR检验模块
实现已知 / 未知断点下的结构断裂检验。

口径说明：
- H0：已知点处不存在结构断裂，整条时间序列共用一套参数；
- H1：已知点处存在结构断裂，断点前后使用两套参数；
- 第二段从断点后第一个响应开始生效，但允许其滞后项借用断点前的 p 个观测。
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

from .var_estimator import VAREstimator


def _validate_break(T: int, p: int, t: int) -> None:
    min_segment_size = p + 2
    if t < min_segment_size:
        raise ValueError(f"变点位置t={t}无效，第一段样本量不足（需要至少{min_segment_size}个观测）")
    if t > T - min_segment_size:
        raise ValueError(f"变点位置t={t}无效，第二段样本量不足（需要至少{min_segment_size}个观测）")


def _split_known_break_series(Y: np.ndarray, p: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    return Y[:t, :], Y[t - p:, :]


class LRTest:
    """LR结构变化检验（针对特定时间点）"""

    def __init__(self):
        self.lr_statistic = None

    @staticmethod
    def _fit_model(Y: np.ndarray, p: int, include_const: bool = True) -> Dict[str, Any]:
        estimator = VAREstimator(method='ols')
        return estimator.fit_ols(Y, p, include_const)

    def compute_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                            include_const: bool = True) -> Dict[str, Any]:
        T, _ = Y.shape
        _validate_break(T, p, t)

        result_r = self._fit_model(Y, p, include_const)
        Y1, Y2 = _split_known_break_series(Y, p, t)
        result1 = self._fit_model(Y1, p, include_const)
        result2 = self._fit_model(Y2, p, include_const)

        log_lik_r = result_r['log_likelihood']
        log_lik_u = result1['log_likelihood'] + result2['log_likelihood']
        if result_r['T_eff'] != result1['T_eff'] + result2['T_eff']:
            raise ValueError('H0 与 H1 的有效样本量不一致')

        self.lr_statistic = 2 * (log_lik_u - log_lik_r)
        return {
            'lr_statistic': float(self.lr_statistic),
            'test_point': t,
            'log_lik_restricted': float(log_lik_r),
            'log_lik_unrestricted': float(log_lik_u),
            'restricted_result': result_r,
            'segment1_result': result1,
            'segment2_result': result2,
        }


class SupLRTest:
    """Sup-LR结构变化检验（遍历所有可能断点）"""

    def __init__(self, trim: float = 0.15):
        self.trim = trim
        self.sup_lr = None
        self.estimated_break = None
        self.lr_sequence = None

    @staticmethod
    def _fit_model(Y: np.ndarray, p: int, include_const: bool = True) -> Dict[str, Any]:
        estimator = VAREstimator(method='ols')
        return estimator.fit_ols(Y, p, include_const)

    def compute_sup_lr(self, Y: np.ndarray, p: int,
                       include_const: bool = True) -> Dict[str, Any]:
        T, _ = Y.shape
        T_eff = T - p

        tau_min = int(np.ceil(self.trim * T_eff)) + p
        tau_max = int(np.floor((1 - self.trim) * T_eff)) + p

        min_segment_size = p + 2
        if tau_min < min_segment_size:
            tau_min = min_segment_size
        if tau_max > T - min_segment_size:
            tau_max = T - min_segment_size
        if tau_min >= tau_max:
            raise ValueError(
                f"断点搜索范围无效: tau_min={tau_min}, tau_max={tau_max}. "
                f"请增加样本量T或减小trim参数。当前T={T}, p={p}, trim={self.trim}"
            )

        result_r = self._fit_model(Y, p, include_const)
        log_lik_r = result_r['log_likelihood']

        lr_values = []
        tau_values = []
        for tau in range(tau_min, tau_max + 1):
            try:
                Y1, Y2 = _split_known_break_series(Y, p, tau)
                result1 = self._fit_model(Y1, p, include_const)
                result2 = self._fit_model(Y2, p, include_const)
                if result_r['T_eff'] != result1['T_eff'] + result2['T_eff']:
                    continue
                log_lik_u = result1['log_likelihood'] + result2['log_likelihood']
                lr_values.append(float(2 * (log_lik_u - log_lik_r)))
                tau_values.append(int(tau))
            except Exception:
                continue

        if not lr_values:
            raise ValueError('无法计算任何有效的LR统计量')

        lr_values = np.array(lr_values)
        tau_values = np.array(tau_values)
        max_idx = int(np.argmax(lr_values))
        self.sup_lr = float(lr_values[max_idx])
        self.estimated_break = int(tau_values[max_idx])
        self.lr_sequence = (tau_values, lr_values)

        return {
            'sup_lr': self.sup_lr,
            'estimated_break': self.estimated_break,
            'lr_sequence': self.lr_sequence,
            'log_lik_restricted': float(log_lik_r),
            'restricted_result': result_r,
        }

    def test(self, Y: np.ndarray, p: int, alpha: float = 0.05,
             critical_value: Optional[float] = None) -> Dict[str, Any]:
        result = self.compute_sup_lr(Y, p)
        if critical_value is not None:
            reject_h0 = result['sup_lr'] > critical_value
            result['reject_h0'] = reject_h0
            result['critical_value'] = critical_value
            result['alpha'] = alpha
        return result
