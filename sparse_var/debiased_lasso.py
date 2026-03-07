"""
去偏Lasso估计器
用于高维稀疏VAR模型的假设检验
"""

import numpy as np
from numpy.linalg import inv, det
from typing import Tuple, Optional, Dict, Any

from simulation.design_matrix import build_var_design_matrix

try:
    from sklearn.linear_model import LassoCV, Lasso
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DebiasedLassoVAR:
    """去偏Lasso-VAR估计器（用于假设检验）"""

    def __init__(self, alpha: Optional[float] = None, cv: int = 5,
                 max_iter: int = 10000):
        """
        初始化去偏Lasso-VAR估计器

        Parameters
        ----------
        alpha : float, optional
            正则化参数
        cv : int
            交叉验证折数
        max_iter : int
            最大迭代次数
        """
        if not HAS_SKLEARN:
            raise ImportError("需要安装sklearn: pip install scikit-learn")

        self.alpha = alpha
        self.cv = cv
        self.max_iter = max_iter
        self.Phi_hat = None
        self.Phi_debiased = None
        self.standard_errors = None

    def fit(self, Y: np.ndarray, p: int,
            include_const: bool = True) -> Dict[str, Any]:
        """
        使用去偏Lasso估计VAR模型

        去偏Lasso的核心思想：
        β_debiased = β_lasso + (X'X)^{-1} X' (Y - X β_lasso)

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
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
        X, Y_response = build_var_design_matrix(Y, p, include_const)
        T_eff = T - p
        n_features = X.shape[1]

        # Step 1: Lasso估计
        B_lasso = np.zeros((n_features, N))
        for i in range(N):
            y_i = Y_response[:, i]
            if self.alpha is None:
                model = LassoCV(cv=self.cv, max_iter=self.max_iter)
            else:
                model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            model.fit(X, y_i)
            B_lasso[:, i] = model.coef_
            if include_const:
                B_lasso[0, i] = model.intercept_

        # Step 2: 去偏校正
        # 计算 (X'X)^{-1}
        try:
            XtX_inv = inv(X.T @ X)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            XtX_inv = np.linalg.pinv(X.T @ X)

        # 计算残差
        residuals_lasso = Y_response - X @ B_lasso

        # 去偏校正
        B_debiased = B_lasso + XtX_inv @ X.T @ residuals_lasso

        # 提取系数
        if include_const:
            c_hat = B_debiased[0, :]
            self.Phi_hat = B_lasso[1:, :].T
            self.Phi_debiased = B_debiased[1:, :].T
        else:
            c_hat = np.zeros(N)
            self.Phi_hat = B_lasso.T
            self.Phi_debiased = B_debiased.T

        # 计算残差协方差矩阵
        residuals = Y_response - X @ B_debiased
        Sigma_hat = (residuals.T @ residuals) / T_eff

        # 计算标准误
        self.standard_errors = self._compute_standard_errors(
            XtX_inv, Sigma_hat, N, p, include_const
        )

        return {
            'Phi_lasso': self.Phi_hat,
            'Phi_debiased': self.Phi_debiased,
            'c': c_hat,
            'Sigma': Sigma_hat,
            'residuals': residuals,
            'standard_errors': self.standard_errors,
            'T_eff': T_eff
        }

    def _compute_standard_errors(self, XtX_inv: np.ndarray,
                                  Sigma: np.ndarray,
                                  N: int, p: int,
                                  include_const: bool) -> np.ndarray:
        """
        计算去偏估计量的标准误

        Parameters
        ----------
        XtX_inv : np.ndarray
            (X'X)^{-1}
        Sigma : np.ndarray
            残差协方差矩阵
        N : int
            变量数量
        p : int
            滞后阶数
        include_const : bool
            是否包含常数项

        Returns
        -------
        np.ndarray
            标准误矩阵
        """
        # 简化计算：使用对角元素
        if include_const:
            var_diag = np.diag(XtX_inv)[1:]  # 排除常数项
        else:
            var_diag = np.diag(XtX_inv)

        # 标准误矩阵
        se_matrix = np.zeros((N, N * p))
        for i in range(N):
            sigma_ii = Sigma[i, i]
            se_matrix[i, :] = np.sqrt(sigma_ii * var_diag)

        return se_matrix

    def test_coefficient(self, i: int, j: int, lag: int = 1) -> Dict[str, Any]:
        """
        检验单个系数是否显著

        Parameters
        ----------
        i : int
            响应变量索引
        j : int
            预测变量索引
        lag : int
            滞后阶数

        Returns
        -------
        Dict[str, Any]
            检验结果
        """
        if self.Phi_debiased is None:
            raise ValueError("模型尚未拟合")

        N = self.Phi_debiased.shape[0]
        coef_idx = (lag - 1) * N + j

        coef = self.Phi_debiased[i, coef_idx]
        se = self.standard_errors[i, coef_idx]

        # t统计量
        t_stat = coef / se if se > 0 else 0

        # 双侧p值（使用正态近似）
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

        return {
            'coefficient': coef,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01
        }
