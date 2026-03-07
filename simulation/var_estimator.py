"""
VAR模型估计模块
支持：OLS估计、Lasso-VAR（高维稀疏）、低秩VAR（核范数正则化）
"""

import numpy as np
from numpy.linalg import lstsq, det, pinv
from typing import Tuple, Optional, Dict, Any

from .design_matrix import build_var_design_matrix


class VAREstimator:
    """VAR模型估计器"""

    def __init__(self, method: str = 'ols'):
        """
        初始化VAR估计器

        Parameters
        ----------
        method : str
            估计方法：'ols', 'lasso', 'lowrank'
        """
        self.method = method
        self.Phi_hat = None
        self.Sigma_hat = None
        self.c_hat = None
        self.residuals = None
        self.log_likelihood = None

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
            X形状为 (T_eff, N*p + 1) 或 (T_eff, N*p)
            Y_response形状为 (T_eff, N)
        """
        return build_var_design_matrix(Y, p, include_const)

    def fit_ols(self, Y: np.ndarray, p: int,
                include_const: bool = True) -> Dict[str, Any]:
        """
        使用OLS估计VAR模型

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
            包含估计结果的字典
        """
        T, N = Y.shape
        X, Y_response = self.build_design_matrix(Y, p, include_const)
        T_eff = T - p

        # OLS估计：使用lstsq替代显式求逆，提升数值稳定性
        # B = (X'X)^{-1} X'Y 等价于求解 X @ B = Y 的最小二乘解
        B_hat, _, _, _ = lstsq(X, Y_response, rcond=None)

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

        # 计算残差协方差矩阵（使用MLE估计，与对数似然口径一致）
        # MLE协方差分母为T_eff，与对数似然公式中的假设一致
        self.Sigma_hat = (self.residuals.T @ self.residuals) / T_eff

        # 计算对数似然值
        self.log_likelihood = self._compute_log_likelihood(T_eff, N, self.Sigma_hat)

        return {
            'Phi': self.Phi_hat,
            'c': self.c_hat,
            'Sigma': self.Sigma_hat,
            'residuals': self.residuals,
            'log_likelihood': self.log_likelihood,
            'T_eff': T_eff
        }

    def _compute_log_likelihood(self, T_eff: int, N: int,
                                 Sigma: np.ndarray) -> float:
        """
        计算对数似然值

        Parameters
        ----------
        T_eff : int
            有效样本长度
        N : int
            变量数量
        Sigma : np.ndarray
            残差协方差矩阵

        Returns
        -------
        float
            对数似然值
        """
        det_Sigma = det(Sigma)
        if det_Sigma <= 0:
            det_Sigma = 1e-10  # 避免log(0)

        log_lik = -0.5 * T_eff * (N * np.log(2 * np.pi) + np.log(det_Sigma) + N)
        return log_lik

    def fit(self, Y: np.ndarray, p: int, **kwargs) -> Dict[str, Any]:
        """
        拟合VAR模型

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        **kwargs : dict
            其他参数

        Returns
        -------
        Dict[str, Any]
            估计结果
        """
        if self.method == 'ols':
            return self.fit_ols(Y, p, **kwargs)
        elif self.method == 'lasso':
            return self.fit_lasso(Y, p, **kwargs)
        elif self.method == 'lowrank':
            return self.fit_lowrank(Y, p, **kwargs)
        else:
            raise ValueError(f"未知的估计方法: {self.method}")

    def fit_segment(self, Y: np.ndarray, p: int, start: int, end: int,
                    include_const: bool = True) -> Dict[str, Any]:
        """
        对时间序列的一个片段进行VAR拟合

        Parameters
        ----------
        Y : np.ndarray
            完整时间序列
        p : int
            滞后阶数
        start : int
            起始位置
        end : int
            结束位置
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            估计结果
        """
        Y_segment = Y[start:end, :]
        return self.fit_ols(Y_segment, p, include_const)
