"""
baseline 同口径 Chow 检验的 Bootstrap 推断模块。

输出同时包含：
- 渐近 p 值（F 分布与卡方分布）
- Bootstrap p 值（F* 与 LR*）
"""

import numpy as np
from typing import Dict, Any, Optional

from .chow_test import ChowTest


class ChowBootstrapInference:
    """同口径 Chow 检验的 Bootstrap 推断"""

    def __init__(self, B: int = 500, seed: Optional[int] = None):
        self.B = B
        self.seed = seed

    @staticmethod
    def generate_pseudo_series(Y: np.ndarray, p: int,
                               Phi: np.ndarray, c: np.ndarray,
                               residuals: np.ndarray) -> np.ndarray:
        """在 H0 受限估计下通过残差重抽样生成伪序列。"""
        T, N = Y.shape
        T_eff = len(residuals)

        centered_residuals = residuals - np.mean(residuals, axis=0)
        indices = np.random.choice(T_eff, size=T_eff, replace=True)
        resampled_residuals = centered_residuals[indices, :]

        Y_star = np.zeros((T, N))
        Y_star[:p, :] = Y[:p, :]

        for t in range(p, T):
            Y_lag_ordered = np.zeros(N * p)
            for lag in range(p):
                Y_lag_ordered[lag * N:(lag + 1) * N] = Y_star[t - lag - 1, :]

            epsilon_t = resampled_residuals[t - p, :] if t - p < T_eff else np.zeros(N)
            Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t

        return Y_star

    def test_at_point(self, Y: np.ndarray, p: int, t: int,
                      alpha: float = 0.05, verbose: bool = False) -> Dict[str, Any]:
        """
        对已知断点 t 执行同口径 Chow Bootstrap 检验。
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        chow = ChowTest()
        original = chow.compute_at_point(Y, p, t)

        original_f = original['f_statistic']
        original_lr = original['lr_statistic']

        restricted = original['restricted_result']
        Phi_r = restricted['Phi']
        c_r = restricted['c']
        residuals_r = restricted['residuals']

        f_stars = []
        lr_stars = []

        for b in range(self.B):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{self.B}")

            try:
                Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)
                stat_b = ChowTest().compute_at_point(Y_star, p, t)
                f_stars.append(stat_b['f_statistic'])
                lr_stars.append(stat_b['lr_statistic'])
            except Exception:
                continue

        f_stars = np.array(f_stars)
        lr_stars = np.array(lr_stars)

        p_boot_f = np.mean(f_stars >= original_f) if len(f_stars) > 0 else np.nan
        p_boot_lr = np.mean(lr_stars >= original_lr) if len(lr_stars) > 0 else np.nan

        reject_h0_boot_lr = p_boot_lr <= alpha if not np.isnan(p_boot_lr) else False
        reject_h0_boot_f = p_boot_f <= alpha if not np.isnan(p_boot_f) else False

        return {
            'test_point': t,
            'alpha': alpha,
            'original_f': float(original_f),
            'original_lr': float(original_lr),
            'f_asymptotic_p_value': original['f_p_value'],
            'chi2_asymptotic_p_value': original['chi2_p_value'],
            'f_df1': original['f_df1'],
            'f_df2': original['f_df2'],
            'chi2_df': original['chi2_df'],
            'bootstrap_f_statistics': f_stars,
            'bootstrap_lr_statistics': lr_stars,
            'bootstrap_f_p_value': float(p_boot_f) if not np.isnan(p_boot_f) else np.nan,
            'bootstrap_lr_p_value': float(p_boot_lr) if not np.isnan(p_boot_lr) else np.nan,
            'reject_h0_bootstrap_f': reject_h0_boot_f,
            'reject_h0_bootstrap_lr': reject_h0_boot_lr,
            'B_effective': int(len(lr_stars)),
            'design_info': original['design_info'],
        }

