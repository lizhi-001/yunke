"""
低秩VAR的蒙特卡洛仿真模块
用于评估低秩VAR检验的第一类错误和统计功效
"""

import numpy as np
from typing import Dict, Any, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.data_generator import VARDataGenerator
from .lowrank_bootstrap import LowRankBootstrapInference


class LowRankMonteCarloSimulation:
    """低秩VAR的蒙特卡洛仿真"""

    def __init__(self, M: int = 1000, B: int = 500,
                 seed: Optional[int] = None, method: str = 'svd',
                 rank: Optional[int] = None,
                 lambda_nuc: Optional[float] = None):
        """
        Parameters
        ----------
        M : int
            蒙特卡洛重复次数
        B : int
            每次Bootstrap的重复次数
        seed : int, optional
            随机种子
        method : str
            估计方法：'svd' 或 'cvxpy'
        rank : int, optional
            指定秩
        lambda_nuc : float, optional
            核范数正则化参数
        """
        self.M = M
        self.B = B
        self.seed = seed
        self.method = method
        self.rank = rank
        self.lambda_nuc = lambda_nuc

    def evaluate_type1_error(self, N: int, T: int, p: int,
                              Phi: np.ndarray, Sigma: np.ndarray,
                              t: int, test_alpha: float = 0.05,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        评估第一类错误（H0为真时的拒绝率）

        Parameters
        ----------
        N, T, p : int
            模型维度、样本长度、滞后阶数
        Phi : np.ndarray
            真实低秩系数矩阵
        Sigma : np.ndarray
            残差协方差矩阵
        t : int
            已知的检验时间点
        test_alpha : float
            显著性水平
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        generator = VARDataGenerator()
        rejections = 0
        p_values = []
        successful_iterations = 0

        for m in range(self.M):
            if verbose and (m + 1) % 10 == 0:
                print(f"Monte Carlo iteration {m + 1}/{self.M}")

            try:
                Y = generator.generate_var_series(T, N, p, Phi, Sigma)

                bootstrap = LowRankBootstrapInference(
                    B=self.B, method=self.method,
                    rank=self.rank, lambda_nuc=self.lambda_nuc
                )
                result = bootstrap.test(Y, p, t, alpha=test_alpha)

                p_values.append(result['p_value'])
                successful_iterations += 1
                if result['reject_h0']:
                    rejections += 1
            except Exception as e:
                if verbose:
                    print(f"Iteration {m + 1} failed: {e}")
                continue

        type1_error = rejections / successful_iterations if successful_iterations > 0 else np.nan

        return {
            'type1_error': type1_error,
            'nominal_alpha': test_alpha,
            'test_point': t,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values),
            'size_distortion': type1_error - test_alpha if not np.isnan(type1_error) else np.nan,
            'method': self.method
        }

    def evaluate_power(self, N: int, T: int, p: int,
                        Phi1: np.ndarray, Phi2: np.ndarray,
                        Sigma: np.ndarray, break_point: int,
                        t: int, test_alpha: float = 0.05,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        评估统计功效（H1为真时的拒绝率）

        Parameters
        ----------
        Phi1 : np.ndarray
            断点前的低秩系数矩阵
        Phi2 : np.ndarray
            断点后的低秩系数矩阵
        break_point : int
            真实断点位置
        t : int
            已知的检验时间点
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        generator = VARDataGenerator()
        rejections = 0
        p_values = []
        successful_iterations = 0

        for m in range(self.M):
            if verbose and (m + 1) % 10 == 0:
                print(f"Monte Carlo iteration {m + 1}/{self.M}")

            try:
                Y, _ = generator.generate_var_with_break(
                    T, N, p, Phi1, Phi2, Sigma, break_point
                )

                bootstrap = LowRankBootstrapInference(
                    B=self.B, method=self.method,
                    rank=self.rank, lambda_nuc=self.lambda_nuc
                )
                result = bootstrap.test(Y, p, t, alpha=test_alpha)

                p_values.append(result['p_value'])
                successful_iterations += 1
                if result['reject_h0']:
                    rejections += 1
            except Exception as e:
                if verbose:
                    print(f"Iteration {m + 1} failed: {e}")
                continue

        power = rejections / successful_iterations if successful_iterations > 0 else np.nan

        return {
            'power': power,
            'alpha': test_alpha,
            'test_point': t,
            'true_break': break_point,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values),
            'method': self.method
        }
