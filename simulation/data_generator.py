"""
数据生成模块 - VAR时间序列数据生成器
支持：平稳VAR、含断点VAR、稀疏系数矩阵、低秩系数矩阵
"""

import numpy as np
from numpy.linalg import eigvals
from typing import Tuple, Optional, List


class VARDataGenerator:
    """VAR时间序列数据生成器"""

    def __init__(self, seed: Optional[int] = None):
        """
        初始化数据生成器

        Parameters
        ----------
        seed : int, optional
            随机种子，用于结果复现
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def check_stationarity(Phi: np.ndarray) -> bool:
        """
        检验VAR系数矩阵是否满足平稳性条件

        Parameters
        ----------
        Phi : np.ndarray
            VAR系数矩阵，形状为 (N, N*p)，其中N为变量数，p为滞后阶数

        Returns
        -------
        bool
            True表示满足平稳性条件（所有特征值模<1）
        """
        N = Phi.shape[0]
        p = Phi.shape[1] // N

        # 构建伴随矩阵 (companion matrix)
        companion = np.zeros((N * p, N * p))
        companion[:N, :] = Phi
        if p > 1:
            companion[N:, :-N] = np.eye(N * (p - 1))

        # 计算特征值
        eigenvalues = eigvals(companion)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        return max_eigenvalue < 1

    def generate_stationary_phi(self, N: int, p: int = 1, sparsity: float = 1.0,
                                 scale: float = 0.3, max_attempts: int = 100) -> np.ndarray:
        """
        生成满足平稳性条件的VAR系数矩阵

        Parameters
        ----------
        N : int
            变量数量
        p : int
            滞后阶数
        sparsity : float
            稀疏度，0-1之间，1表示完全稠密，0.1表示10%非零
        scale : float
            系数的缩放因子，用于控制系数大小
        max_attempts : int
            最大尝试次数

        Returns
        -------
        np.ndarray
            满足平稳性条件的系数矩阵，形状为 (N, N*p)
        """
        for _ in range(max_attempts):
            # 生成随机系数矩阵
            Phi = self._rng.normal(loc=0.0, scale=scale, size=(N, N * p))

            # 应用稀疏性约束
            if sparsity < 1.0:
                mask = self._rng.random((N, N * p)) < sparsity
                Phi = Phi * mask

            # 检验平稳性
            if VARDataGenerator.check_stationarity(Phi):
                return Phi

        raise ValueError(f"无法在{max_attempts}次尝试内生成平稳的VAR系数矩阵")

    def generate_lowrank_phi(self, N: int, p: int = 1, rank: int = 2,
                              scale: float = 0.3,
                              target_spectral_radius: Optional[float] = None,
                              max_attempts: int = 100) -> np.ndarray:
        """
        生成低秩的VAR系数矩阵

        Parameters
        ----------
        N : int
            变量数量
        p : int
            滞后阶数
        rank : int
            目标秩
        scale : float
            系数的缩放因子（用于生成初始随机矩阵）
        target_spectral_radius : float, optional
            目标谱半径。若指定，则对生成的矩阵等比缩放至该谱半径，
            消除 seed 间因随机方向不同导致的谱半径差异。
            低秩矩阵自由度少（rank 个奇异值决定整体大小），不指定时
            seed 间谱半径方差极大（CV≈0.45），指定后完全消除该方差来源。
        max_attempts : int
            最大尝试次数

        Returns
        -------
        np.ndarray
            低秩且满足平稳性条件的系数矩阵
        """
        for _ in range(max_attempts):
            # 通过低秩分解生成：Phi = U @ V.T
            U = self._rng.normal(loc=0.0, scale=scale, size=(N, rank))
            V = self._rng.normal(loc=0.0, scale=scale, size=(N * p, rank))
            Phi = U @ V.T

            if target_spectral_radius is not None:
                # 直接归一化到目标谱半径，保留低秩方向（U, V），仅调整幅度
                r = np.max(np.abs(eigvals(Phi)))
                if r < 1e-10:
                    continue
                return Phi * (target_spectral_radius / r)

            if VARDataGenerator.check_stationarity(Phi):
                return Phi

        raise ValueError(f"无法在{max_attempts}次尝试内生成平稳的低秩VAR系数矩阵")

    def generate_var_series(self, T: int, N: int, p: int,
                            Phi: np.ndarray, Sigma: np.ndarray,
                            c: Optional[np.ndarray] = None,
                            burn_in: int = 100) -> np.ndarray:
        """
        生成VAR(p)时间序列

        Parameters
        ----------
        T : int
            样本长度
        N : int
            变量数量
        p : int
            滞后阶数
        Phi : np.ndarray
            系数矩阵，形状为 (N, N*p)
        Sigma : np.ndarray
            残差协方差矩阵，形状为 (N, N)
        c : np.ndarray, optional
            常数项向量，形状为 (N,)
        burn_in : int
            预热期长度，用于消除初始值影响

        Returns
        -------
        np.ndarray
            生成的时间序列，形状为 (T, N)
        """
        if c is None:
            c = np.zeros(N)

        total_length = T + burn_in
        Y = np.zeros((total_length, N))

        # 生成残差
        epsilon = self._rng.multivariate_normal(np.zeros(N), Sigma, total_length)

        # 迭代生成序列
        for t in range(p, total_length):
            Y_lag_ordered = Y[t-p:t, :][::-1].ravel()
            Y[t, :] = c + Phi @ Y_lag_ordered + epsilon[t, :]

        return Y[burn_in:, :]

    def generate_var_with_break(self, T: int, N: int, p: int,
                                 Phi1: np.ndarray, Phi2: np.ndarray,
                                 Sigma: np.ndarray, break_point: int,
                                 c: Optional[np.ndarray] = None,
                                 burn_in: int = 100) -> Tuple[np.ndarray, int]:
        """
        生成含结构断点的VAR时间序列

        Parameters
        ----------
        T : int
            样本长度
        N : int
            变量数量
        p : int
            滞后阶数
        Phi1 : np.ndarray
            断点前的系数矩阵
        Phi2 : np.ndarray
            断点后的系数矩阵
        Sigma : np.ndarray
            残差协方差矩阵
        break_point : int
            断点位置（相对于有效样本）
        c : np.ndarray, optional
            常数项向量
        burn_in : int
            预热期长度

        Returns
        -------
        Tuple[np.ndarray, int]
            (生成的时间序列, 实际断点位置)
        """
        if c is None:
            c = np.zeros(N)

        total_length = T + burn_in
        actual_break = break_point + burn_in
        Y = np.zeros((total_length, N))

        # 生成残差
        epsilon = self._rng.multivariate_normal(np.zeros(N), Sigma, total_length)

        # 迭代生成序列
        for t in range(p, total_length):
            Y_lag_ordered = Y[t-p:t, :][::-1].ravel()
            Phi = Phi1 if t < actual_break else Phi2
            Y[t, :] = c + Phi @ Y_lag_ordered + epsilon[t, :]

        return Y[burn_in:, :], break_point
