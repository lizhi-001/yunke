"""
Chow检验模块（已知断点、同一有效样本口径）

核心思想：
- 在统一样本 t=p,...,T-1 上构造受限/非受限模型
- 使用哑变量交互项 D_t * x_t 表示断点后参数增量
- 同时给出 Chow 型 F 统计量与同口径 LR 统计量
"""

import numpy as np
from numpy.linalg import det, lstsq
from typing import Dict, Any

from scipy.stats import f as f_dist
from scipy.stats import chi2


class ChowTest:
    """已知断点的同口径 Chow 检验（baseline OLS）"""

    def __init__(self):
        self.f_statistic = None
        self.lr_statistic = None

    @staticmethod
    def _build_base_design_matrix(Y: np.ndarray, p: int,
                                  include_const: bool = True) -> Dict[str, np.ndarray]:
        """构建统一样本口径下的 VAR 回归设计矩阵。"""
        T, N = Y.shape
        T_eff = T - p

        X = np.zeros((T_eff, N * p))
        for row in range(T_eff):
            t = row + p
            for lag in range(p):
                X[row, lag * N:(lag + 1) * N] = Y[t - lag - 1, :]

        if include_const:
            X = np.column_stack([np.ones(T_eff), X])

        Y_response = Y[p:, :]
        time_index = np.arange(p, T)

        return {
            'X': X,
            'Y_response': Y_response,
            'time_index': time_index,
            'T_eff': T_eff,
            'N': N,
        }

    @staticmethod
    def _fit_multivariate_ols(X: np.ndarray, Y_response: np.ndarray) -> Dict[str, np.ndarray]:
        """多元 OLS 拟合。"""
        B_hat, _, _, _ = lstsq(X, Y_response, rcond=None)
        residuals = Y_response - X @ B_hat
        return {'B_hat': B_hat, 'residuals': residuals}

    @staticmethod
    def _safe_logdet(Sigma: np.ndarray) -> float:
        """稳定计算 log|Sigma|。"""
        det_sigma = det(Sigma)
        if det_sigma <= 0:
            det_sigma = 1e-10
        return np.log(det_sigma)

    def compute_at_point(self, Y: np.ndarray, p: int, t: int,
                         include_const: bool = True) -> Dict[str, Any]:
        """
        在已知断点 t 处计算同口径 Chow/F 与 LR 统计量。

        Parameters
        ----------
        Y : np.ndarray
            时间序列，形状 (T, N)
        p : int
            滞后阶数
        t : int
            已知断点位置（原始样本坐标）
        include_const : bool
            是否包含截距
        """
        T, N = Y.shape
        min_segment_size = p + 2
        if t < min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第一段样本量不足")
        if t > T - min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第二段样本量不足")

        base = self._build_base_design_matrix(Y, p, include_const)
        X_r = base['X']
        Y_response = base['Y_response']
        time_index = base['time_index']
        n = base['T_eff']

        # D_t = 1(t >= t0)
        D = (time_index >= t).astype(float).reshape(-1, 1)
        X_int = X_r * D
        X_u = np.column_stack([X_r, X_int])

        k_r = X_r.shape[1]
        k_u = X_u.shape[1]

        fit_r = self._fit_multivariate_ols(X_r, Y_response)
        fit_u = self._fit_multivariate_ols(X_u, Y_response)

        residuals_r = fit_r['residuals']
        residuals_u = fit_u['residuals']

        # 系统 SSR（所有方程残差平方和）
        ssr_r = float(np.sum(residuals_r ** 2))
        ssr_u = float(np.sum(residuals_u ** 2))

        # Chow 型 F（系统 pooled 版本）
        q_eq = k_u - k_r
        q = q_eq * N
        df2 = N * (n - k_u)
        if ssr_u <= 0:
            ssr_u = 1e-10
        numerator = (ssr_r - ssr_u) / q if q > 0 else np.nan
        denominator = ssr_u / df2 if df2 > 0 else np.nan
        self.f_statistic = numerator / denominator if denominator > 0 else np.nan
        p_value_f = f_dist.sf(self.f_statistic, q, df2) if np.isfinite(self.f_statistic) else np.nan

        # 同口径 LR
        Sigma_r = (residuals_r.T @ residuals_r) / n
        Sigma_u = (residuals_u.T @ residuals_u) / n
        logdet_r = self._safe_logdet(Sigma_r)
        logdet_u = self._safe_logdet(Sigma_u)
        self.lr_statistic = n * (logdet_r - logdet_u)

        # 渐近卡方自由度：约束参数个数
        df_chi2 = q
        p_value_chi2 = chi2.sf(self.lr_statistic, df_chi2)

        # 受限模型参数（供 bootstrap 伪样本生成）
        B_hat_r = fit_r['B_hat']
        if include_const:
            c_r = B_hat_r[0, :]
            Phi_r = B_hat_r[1:, :].T
        else:
            c_r = np.zeros(N)
            Phi_r = B_hat_r.T

        return {
            'test_point': t,
            'f_statistic': self.f_statistic,
            'f_p_value': float(p_value_f),
            'f_df1': int(q),
            'f_df2': int(df2),
            'lr_statistic': float(self.lr_statistic),
            'chi2_p_value': float(p_value_chi2),
            'chi2_df': int(df_chi2),
            'restricted_result': {
                'Phi': Phi_r,
                'c': c_r,
                'Sigma': Sigma_r,
                'residuals': residuals_r,
                'T_eff': n,
            },
            'unrestricted_result': {
                'Sigma': Sigma_u,
                'residuals': residuals_u,
                'T_eff': n,
            },
            'design_info': {
                'n': n,
                'k_r': int(k_r),
                'k_u': int(k_u),
                'q_per_equation': int(q_eq),
                'q_system': int(q),
            },
        }

