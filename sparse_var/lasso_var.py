"""
Lasso-VAR估计器
使用L1正则化进行高维稀疏VAR模型估计
"""

import os

import numpy as np
from numpy.linalg import det
from typing import Tuple, Optional, Dict, Any

from simulation.design_matrix import build_var_design_matrix

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

try:
    from sklearn.linear_model import LassoCV, Lasso
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class LassoVAREstimator:
    """Lasso-VAR估计器（高维稀疏场景）"""

    def __init__(self, alpha: Optional[float] = None, cv: int = 5,
                 max_iter: int = 10000, post_lasso_ols: bool = False):
        """
        初始化Lasso-VAR估计器

        Parameters
        ----------
        alpha : float, optional
            正则化参数，如果为None则使用交叉验证选择
        cv : int
            交叉验证折数
        max_iter : int
            最大迭代次数
        post_lasso_ols : bool
            是否使用Post-Lasso OLS：先用Lasso做变量选择，
            再用OLS在选出的变量上无偏重拟合。
            可消除正则化偏差对似然函数的影响。
        """
        if not HAS_SKLEARN:
            raise ImportError("需要安装sklearn: pip install scikit-learn")

        self.alpha = alpha
        self.cv = cv
        self.max_iter = max_iter
        self.post_lasso_ols = post_lasso_ols
        self.Phi_hat = None
        self.c_hat = None
        self.Sigma_hat = None
        self.residuals = None
        self.alphas_used = None
        self._cached_models = {}

    def build_design_matrix(self, Y: np.ndarray, p: int,
                            include_const: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建VAR模型的设计矩阵

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
        Tuple[np.ndarray, np.ndarray]
            (设计矩阵X, 响应矩阵Y_response)
        """
        return build_var_design_matrix(Y, p, include_const)

    def _get_cached_lasso(self, equation_idx: int):
        if self.alpha is None:
            return LassoCV(cv=self.cv, max_iter=self.max_iter)

        cache_key = (equation_idx, self.alpha)
        model = self._cached_models.get(cache_key)
        if model is None:
            model = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                warm_start=True,
                selection='random',
            )
            self._cached_models[cache_key] = model
        return model

    def fit(self, Y: np.ndarray, p: int,
            include_const: bool = True) -> Dict[str, Any]:
        """
        使用Lasso估计VAR模型

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
            估计结果
        """
        T, N = Y.shape
        X, Y_response = self.build_design_matrix(Y, p, include_const)
        T_eff = T - p

        # 对每个方程分别进行Lasso估计
        n_features = X.shape[1]
        B_hat = np.zeros((n_features, N))
        self.alphas_used = []

        for i in range(N):
            y_i = Y_response[:, i]

            model = self._get_cached_lasso(i)
            model.fit(X, y_i)
            if self.alpha is None:
                self.alphas_used.append(model.alpha_)
            else:
                self.alphas_used.append(self.alpha)

            B_hat[:, i] = model.coef_
            if include_const:
                B_hat[0, i] = model.intercept_

        # 提取系数
        if include_const:
            self.c_hat = B_hat[0, :]
            self.Phi_hat = B_hat[1:, :].T  # 转置为 (N, N*p)
        else:
            self.c_hat = np.zeros(N)
            self.Phi_hat = B_hat.T

        # 计算残差
        Y_fitted = X @ B_hat
        self.residuals = Y_response - Y_fitted

        # 计算稀疏度
        sparsity = np.mean(np.abs(self.Phi_hat) < 1e-10)

        # Post-Lasso OLS: 用Lasso选出的非零变量做OLS重拟合
        if self.post_lasso_ols:
            B_hat, self.residuals = self._post_lasso_refit(
                X, Y_response, B_hat, N, include_const
            )
            if include_const:
                self.c_hat = B_hat[0, :]
                self.Phi_hat = B_hat[1:, :].T
            else:
                self.c_hat = np.zeros(N)
                self.Phi_hat = B_hat.T

        # 计算残差协方差矩阵
        self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff

        # 计算对数似然值
        det_Sigma = det(self.Sigma_hat)
        if det_Sigma <= 0:
            det_Sigma = 1e-10
        log_likelihood = -0.5 * T_eff * (N * np.log(2 * np.pi) + np.log(det_Sigma) + N)

        return {
            'Phi': self.Phi_hat,
            'c': self.c_hat,
            'Sigma': self.Sigma_hat,
            'residuals': self.residuals,
            'log_likelihood': log_likelihood,
            'T_eff': T_eff,
            'alphas_used': self.alphas_used,
            'sparsity': sparsity
        }

    def _post_lasso_refit(self, X: np.ndarray, Y_response: np.ndarray,
                          B_lasso: np.ndarray, N: int,
                          include_const: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-Lasso OLS重拟合：在Lasso选出的非零变量上用OLS重拟合。

        对每个响应方程，找出Lasso系数非零的特征，用OLS重新估计，
        消除正则化偏差。

        Returns
        -------
        B_refit : np.ndarray, 与B_lasso同形状
        residuals : np.ndarray
        """
        n_features = X.shape[1]
        B_refit = np.zeros_like(B_lasso)

        for i in range(N):
            y_i = Y_response[:, i]
            coef_i = B_lasso[:, i].copy()

            # 确定非零特征（常数项始终保留）
            if include_const:
                nonzero = np.abs(coef_i[1:]) > 1e-10
                selected = np.concatenate([[True], nonzero])  # 保留常数项
            else:
                selected = np.abs(coef_i) > 1e-10

            if np.sum(selected) == 0:
                # 无特征被选中，只用常数项
                if include_const:
                    selected[0] = True
                else:
                    B_refit[:, i] = 0
                    continue

            # OLS重拟合
            X_sel = X[:, selected]
            beta_ols = np.linalg.lstsq(X_sel, y_i, rcond=None)[0]
            B_refit[selected, i] = beta_ols

        residuals = Y_response - X @ B_refit
        return B_refit, residuals

    def get_nonzero_coefficients(self) -> Dict[str, Any]:
        """
        获取非零系数的位置和值

        Returns
        -------
        Dict[str, Any]
            非零系数信息
        """
        if self.Phi_hat is None:
            raise ValueError("模型尚未拟合")

        nonzero_mask = np.abs(self.Phi_hat) > 1e-10
        nonzero_indices = np.argwhere(nonzero_mask)
        nonzero_values = self.Phi_hat[nonzero_mask]

        return {
            'indices': nonzero_indices,
            'values': nonzero_values,
            'count': len(nonzero_values),
            'total': self.Phi_hat.size,
            'sparsity': 1 - len(nonzero_values) / self.Phi_hat.size
        }
