"""
Chow检验模块（已知断点、已知点结构断裂口径）

核心思想：
- H0：已知点处不存在结构断裂，整条时间序列共用一套参数；
- H1：已知点处存在结构断裂，断点前后使用两套参数；
- 第二段从断点后第一个响应开始生效，但允许其滞后项使用断点前的 p 个观测。
"""

import numpy as np
from numpy.linalg import det, lstsq
from typing import Dict, Any, Tuple

from scipy.stats import f as f_dist
from scipy.stats import chi2

from .var_estimator import VAREstimator


class ChowTest:
    """已知断点的 Chow / LR 检验（baseline OLS）"""

    def __init__(self):
        self.f_statistic = None
        self.lr_statistic = None

    @staticmethod
    def _validate_break(T: int, p: int, t: int) -> None:
        min_segment_size = p + 2
        if t < min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第一段样本量不足")
        if t > T - min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第二段样本量不足")

    @staticmethod
    def _split_known_break_series(Y: np.ndarray, p: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """构造已知断点下的两段样本。

        第一段使用 `Y[:t]`。
        第二段使用 `Y[t-p:]`，这样断点后第一个响应点仍可借用断点前的 p 个滞后值。
        """
        Y1 = Y[:t, :]
        Y2 = Y[t - p:, :]
        return Y1, Y2

    @staticmethod
    def _safe_logdet(Sigma: np.ndarray) -> float:
        det_sigma = det(Sigma)
        if det_sigma <= 0:
            det_sigma = 1e-10
        return np.log(det_sigma)

    @staticmethod
    def _fit_segment_ols(Y: np.ndarray, p: int, include_const: bool = True) -> Dict[str, Any]:
        estimator = VAREstimator(method='ols')
        return estimator.fit_ols(Y, p, include_const)

    @staticmethod
    def _system_ssr(result: Dict[str, Any]) -> float:
        return float(np.sum(result['residuals'] ** 2))

    def compute_at_point(self, Y: np.ndarray, p: int, t: int,
                         include_const: bool = True) -> Dict[str, Any]:
        """在已知断点 t 处计算 Chow 型 F 与 LR 统计量。"""
        T, N = Y.shape
        self._validate_break(T, p, t)

        result_r = self._fit_segment_ols(Y, p, include_const)
        Y1, Y2 = self._split_known_break_series(Y, p, t)
        result1 = self._fit_segment_ols(Y1, p, include_const)
        result2 = self._fit_segment_ols(Y2, p, include_const)

        n_r = result_r['T_eff']
        n1 = result1['T_eff']
        n2 = result2['T_eff']
        n_u = n1 + n2
        if n_r != n_u:
            raise ValueError(f"样本量口径不一致: H0={n_r}, H1={n_u}")

        ssr_r = self._system_ssr(result_r)
        ssr_1 = self._system_ssr(result1)
        ssr_2 = self._system_ssr(result2)
        ssr_u = ssr_1 + ssr_2

        k_r_eq = (N * p) + (1 if include_const else 0)
        k_u_eq = 2 * k_r_eq
        q_eq = k_u_eq - k_r_eq
        q = q_eq * N
        df2 = N * ((n1 - k_r_eq) + (n2 - k_r_eq))

        if ssr_u <= 0:
            ssr_u = 1e-10
        numerator = (ssr_r - ssr_u) / q if q > 0 else np.nan
        denominator = ssr_u / df2 if df2 > 0 else np.nan
        self.f_statistic = numerator / denominator if denominator > 0 else np.nan
        p_value_f = f_dist.sf(self.f_statistic, q, df2) if np.isfinite(self.f_statistic) else np.nan

        logdet_r = self._safe_logdet(result_r['Sigma'])
        logdet_1 = self._safe_logdet(result1['Sigma'])
        logdet_2 = self._safe_logdet(result2['Sigma'])
        self.lr_statistic = 2.0 * (
            (-0.5 * n1 * logdet_1 - 0.5 * n2 * logdet_2)
            - (-0.5 * n_r * logdet_r)
        )
        p_value_chi2 = chi2.sf(self.lr_statistic, q)

        return {
            'test_point': t,
            'f_statistic': float(self.f_statistic),
            'f_p_value': float(p_value_f),
            'f_df1': int(q),
            'f_df2': int(df2),
            'lr_statistic': float(self.lr_statistic),
            'chi2_p_value': float(p_value_chi2),
            'chi2_df': int(q),
            'restricted_result': result_r,
            'segment1_result': result1,
            'segment2_result': result2,
            'design_info': {
                'n_total': int(n_r),
                'n_segment1': int(n1),
                'n_segment2': int(n2),
                'k_r_per_equation': int(k_r_eq),
                'k_u_per_equation': int(k_u_eq),
                'q_per_equation': int(q_eq),
                'q_system': int(q),
            },
        }
