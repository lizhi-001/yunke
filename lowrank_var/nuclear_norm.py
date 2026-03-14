"""
核范数正则化VAR估计器
用于低秩VAR模型估计
"""

import numpy as np
from numpy.linalg import svd, det
from typing import Tuple, Optional, Dict, Any

from simulation.design_matrix import build_var_design_matrix

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
        return build_var_design_matrix(Y, p, include_const=False)

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

    def fit_rrr(self, Y: np.ndarray, p: int, rank: int,
                include_const: bool = True) -> Dict[str, Any]:
        """
        降秩回归（Reduced-Rank Regression）估计低秩VAR参数。

        直接求解 min_{B: rank(B)≤r} ||Y_resp - X_full @ B||²_F
        解析解：B_RRR = B_OLS @ V_r @ V_r.T
        其中 V_r 是拟合值矩阵 Ŷ = X_full @ B_OLS 的前 r 个右奇异向量。

        参考文献：
        - Anderson (1951). Estimating linear restrictions on regression
          coefficients for multivariate normal distributions.
        - Reinsel & Velu (1998). Multivariate Reduced-Rank Regression.

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状 (T, N)
        p : int
            滞后阶数
        rank : int
            目标秩
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            估计结果（与 fit_svd 返回格式一致）
        """
        T, N = Y.shape
        X, Y_response = self.build_design_matrix(Y, p)
        T_eff = T - p

        if include_const:
            X_full = np.column_stack([np.ones(T_eff), X])
        else:
            X_full = X

        # OLS 估计
        B_ols = np.linalg.lstsq(X_full, Y_response, rcond=None)[0]  # K × N
        Y_hat = X_full @ B_ols                                       # T_eff × N

        # 拟合值矩阵的 SVD；V_r 张成最优 rank 维响应子空间
        _, s_yhat, Vt_yhat = svd(Y_hat, full_matrices=False)
        V_r = Vt_yhat[:rank, :].T                                    # N × rank

        # 降秩投影：将 B_OLS 各列投影到 V_r 子空间
        B_rrr = B_ols @ (V_r @ V_r.T)                               # K × N

        if include_const:
            self.c_hat = B_rrr[0, :]
            self.Phi_hat = B_rrr[1:, :].T                            # N × (N*p)
        else:
            self.c_hat = np.zeros(N)
            self.Phi_hat = B_rrr.T

        self.rank = rank

        # 残差
        Y_fitted = X_full @ B_rrr
        self.residuals = Y_response - Y_fitted

        # 残差协方差
        self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff

        # 对数似然
        det_Sigma = det(self.Sigma_hat)
        if det_Sigma <= 0:
            det_Sigma = 1e-10
        log_likelihood = -0.5 * T_eff * (N * np.log(2 * np.pi) + np.log(det_Sigma) + N)

        # Phi_hat 的奇异值（用于诊断）
        _, s_phi, _ = svd(self.Phi_hat, full_matrices=False)

        return {
            'Phi': self.Phi_hat,
            'c': self.c_hat,
            'Sigma': self.Sigma_hat,
            'residuals': self.residuals,
            'log_likelihood': log_likelihood,
            'rank': self.rank,
            'T_eff': T_eff,
            'singular_values': s_phi
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
        elif method == 'rrr':
            return self.fit_rrr(Y, p, **kwargs)
        else:
            raise ValueError(f"未知的估计方法: {method}")
