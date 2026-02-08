"""
核范数正则化VAR估计器
用于低秩VAR模型估计
"""

import numpy as np
from numpy.linalg import svd, det
from typing import Tuple, Optional, Dict, Any

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


class NuclearNormVAR:
    """核范数正则化VAR估计器（低秩场景）"""

    def __init__(self, lambda_nuc: Optional[float] = None):
        """
        初始化核范数正则化VAR估计器

        Parameters
        ----------
        lambda_nuc : float, optional
            核范数正则化参数
        """
        self.lambda_nuc = lambda_nuc
        self.Phi_hat = None
        self.c_hat = None
        self.Sigma_hat = None
        self.residuals = None
        self.rank = None

    def build_design_matrix(self, Y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建VAR模型的设计矩阵

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状为 (T, N)
        p : int
            滞后阶数

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (设计矩阵X, 响应矩阵Y_response)
        """
        T, N = Y.shape
        T_eff = T - p

        # 构建滞后矩阵
        X = np.zeros((T_eff, N * p))
        for t in range(T_eff):
            for lag in range(p):
                X[t, lag*N:(lag+1)*N] = Y[t + p - lag - 1, :]

        Y_response = Y[p:, :]

        return X, Y_response

    def fit_cvxpy(self, Y: np.ndarray, p: int,
                   include_const: bool = True) -> Dict[str, Any]:
        """
        使用CVXPY进行核范数正则化估计

        目标函数：min ||Y - XΦ'||_F^2 / (2T) + λ ||Φ||_*

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
        if not HAS_CVXPY:
            raise ImportError("需要安装cvxpy: pip install cvxpy")

        T, N = Y.shape
        X, Y_response = self.build_design_matrix(Y, p)
        T_eff = T - p

        # 定义优化变量
        Phi = cp.Variable((N, N * p))

        if include_const:
            c = cp.Variable(N)
            Y_pred = X @ Phi.T + np.ones((T_eff, 1)) @ cp.reshape(c, (1, N))
        else:
            Y_pred = X @ Phi.T

        # 目标函数
        loss = cp.sum_squares(Y_response - Y_pred) / (2 * T_eff)

        if self.lambda_nuc is not None and self.lambda_nuc > 0:
            reg = self.lambda_nuc * cp.normNuc(Phi)
            objective = cp.Minimize(loss + reg)
        else:
            objective = cp.Minimize(loss)

        # 求解
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"优化问题求解失败: {problem.status}")

        # 提取结果
        self.Phi_hat = Phi.value

        if include_const:
            self.c_hat = c.value
        else:
            self.c_hat = np.zeros(N)

        # 计算残差
        if include_const:
            Y_fitted = X @ self.Phi_hat.T + self.c_hat
        else:
            Y_fitted = X @ self.Phi_hat.T

        self.residuals = Y_response - Y_fitted

        # 计算残差协方差矩阵
        self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff

        # 计算秩
        _, s, _ = svd(self.Phi_hat)
        self.rank = np.sum(s > 1e-6)

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
            'rank': self.rank,
            'T_eff': T_eff,
            'lambda_nuc': self.lambda_nuc
        }

    def fit_svd(self, Y: np.ndarray, p: int, rank: int,
                include_const: bool = True) -> Dict[str, Any]:
        """
        使用截断SVD进行低秩VAR估计（硬阈值方法）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        rank : int
            目标秩
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            估计结果
        """
        T, N = Y.shape
        X, Y_response = self.build_design_matrix(Y, p)
        T_eff = T - p

        # 首先进行OLS估计
        if include_const:
            X_full = np.column_stack([np.ones(T_eff), X])
        else:
            X_full = X

        # OLS估计
        B_ols = np.linalg.lstsq(X_full, Y_response, rcond=None)[0]

        if include_const:
            self.c_hat = B_ols[0, :]
            Phi_ols = B_ols[1:, :].T
        else:
            self.c_hat = np.zeros(N)
            Phi_ols = B_ols.T

        # 截断SVD
        U, s, Vt = svd(Phi_ols, full_matrices=False)
        s_truncated = np.zeros_like(s)
        s_truncated[:rank] = s[:rank]
        self.Phi_hat = U @ np.diag(s_truncated) @ Vt
        self.rank = rank

        # 计算残差
        if include_const:
            Y_fitted = X @ self.Phi_hat.T + self.c_hat
        else:
            Y_fitted = X @ self.Phi_hat.T

        self.residuals = Y_response - Y_fitted

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
            'rank': self.rank,
            'T_eff': T_eff,
            'singular_values': s
        }

    def fit(self, Y: np.ndarray, p: int, method: str = 'cvxpy',
            **kwargs) -> Dict[str, Any]:
        """
        拟合低秩VAR模型

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        method : str
            估计方法：'cvxpy' 或 'svd'
        **kwargs : dict
            其他参数

        Returns
        -------
        Dict[str, Any]
            估计结果
        """
        if method == 'cvxpy':
            return self.fit_cvxpy(Y, p, **kwargs)
        elif method == 'svd':
            return self.fit_svd(Y, p, **kwargs)
        else:
            raise ValueError(f"未知的估计方法: {method}")
