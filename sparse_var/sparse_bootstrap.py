"""
高维稀疏VAR的Bootstrap推断模块
用于稀疏VAR模型的结构变化检验
"""

import numpy as np
from typing import Dict, Any, Optional
from .sparse_lr_test import SparseLRTest
from .lasso_var import LassoVAREstimator
from .debiased_lasso import DebiasedLassoVAR


class SparseBootstrapInference:
    """高维稀疏VAR的Bootstrap推断"""

    def __init__(self, B: int = 500, seed: Optional[int] = None,
                 estimator_type: str = 'lasso', alpha: Optional[float] = None,
                 post_lasso_ols: bool = False):
        """
        初始化Bootstrap推断

        Parameters
        ----------
        B : int
            Bootstrap重复次数
        seed : int, optional
            随机种子
        estimator_type : str
            估计方法：'lasso' 或 'debiased_lasso'
        alpha : float, optional
            正则化参数
        post_lasso_ols : bool
            是否使用Post-Lasso OLS重拟合
        """
        self.B = B
        self.seed = seed
        self.estimator_type = estimator_type
        self.alpha = alpha
        self.post_lasso_ols = post_lasso_ols
        self.bootstrap_statistics = None
        self.p_value = None
        self.critical_values = None
        self.rng = np.random.default_rng(seed)

    def _get_phi_and_c(self, result: Dict[str, Any]) -> tuple:
        """从估计结果中提取Phi和c"""
        if 'Phi_debiased' in result:
            return result['Phi_debiased'], result['c']
        else:
            return result['Phi'], result['c']

    def generate_pseudo_series(self, Y: np.ndarray, p: int,
                                Phi: np.ndarray, c: np.ndarray,
                                residuals: np.ndarray) -> np.ndarray:
        """
        生成Bootstrap伪序列

        Parameters
        ----------
        Y : np.ndarray
            原始时间序列
        p : int
            滞后阶数
        Phi : np.ndarray
            H0下估计的系数矩阵
        c : np.ndarray
            H0下估计的常数项
        residuals : np.ndarray
            H0下的残差

        Returns
        -------
        np.ndarray
            Bootstrap伪序列
        """
        T, N = Y.shape
        T_eff = len(residuals)

        # 残差重抽样（有放回）并居中处理
        centered_residuals = residuals - np.mean(residuals, axis=0)
        indices = self.rng.choice(T_eff, size=T_eff, replace=True)
        resampled_residuals = centered_residuals[indices, :]

        # 生成伪序列
        Y_star = np.zeros((T, N))
        Y_star[:p, :] = Y[:p, :]

        for t in range(p, T):
            Y_lag_ordered = Y_star[t-p:t, :][::-1].ravel()
            epsilon_t = resampled_residuals[t - p, :] if t - p < T_eff else np.zeros(N)
            Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t

        return Y_star

    def bootstrap_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                               verbose: bool = False) -> Dict[str, Any]:
        """
        执行针对特定时间点t的Bootstrap LR检验（稀疏版本）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            Bootstrap检验结果
        """
        # Step 1: 计算原始数据的LR统计量
        lr_test = SparseLRTest(estimator_type=self.estimator_type, alpha=self.alpha,
                               post_lasso_ols=self.post_lasso_ols)
        original_result = lr_test.compute_lr_at_point(Y, p, t)
        original_lr = original_result['lr_statistic']

        # 获取H0下的估计结果
        restricted_result = original_result['restricted_result']
        Phi_r, c_r = self._get_phi_and_c(restricted_result)
        residuals_r = restricted_result['residuals']

        # Step 2: Bootstrap循环
        bootstrap_lr_values = []

        for b in range(self.B):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{self.B}")

            try:
                # 生成伪序列
                Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)

                # 计算伪序列的LR统计量
                result_b = lr_test.compute_lr_at_point(Y_star, p, t)
                bootstrap_lr_values.append(result_b['lr_statistic'])
            except Exception:
                continue

        self.bootstrap_statistics = np.array(bootstrap_lr_values)

        # Step 3: 计算p值
        self.p_value = np.mean(self.bootstrap_statistics >= original_lr) if len(self.bootstrap_statistics) > 0 else np.nan

        # Step 4: 计算临界值
        if len(self.bootstrap_statistics) > 0:
            self.critical_values = {
                0.10: np.percentile(self.bootstrap_statistics, 90),
                0.05: np.percentile(self.bootstrap_statistics, 95),
                0.01: np.percentile(self.bootstrap_statistics, 99)
            }
        else:
            self.critical_values = {0.10: np.nan, 0.05: np.nan, 0.01: np.nan}

        return {
            'original_lr': original_lr,
            'test_point': t,
            'p_value': self.p_value,
            'critical_values': self.critical_values,
            'bootstrap_statistics': self.bootstrap_statistics,
            'B_effective': len(self.bootstrap_statistics),
            'estimator_type': self.estimator_type
        }

    def test(self, Y: np.ndarray, p: int, t: int,
             alpha: float = 0.05, verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的Bootstrap LR检验（稀疏版本）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            检验结果
        """
        result = self.bootstrap_lr_at_point(Y, p, t, verbose)

        # 做出决策
        reject_h0 = result['p_value'] <= alpha

        result['alpha'] = alpha
        result['reject_h0'] = reject_h0
        result['decision'] = (f"拒绝H0：在时间点{t}存在结构性变化"
                              if reject_h0 else "接受H0：不存在结构性变化")

        return result
