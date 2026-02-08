"""
高维稀疏VAR的蒙特卡洛仿真模块
用于评估稀疏VAR检验的第一类错误和统计功效
"""

import numpy as np
from typing import Dict, Any, Optional, List
import sys
import os

# 添加父目录到路径以导入simulation模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.data_generator import VARDataGenerator
from .sparse_bootstrap import SparseBootstrapInference


class SparseMonteCarloSimulation:
    """高维稀疏VAR的蒙特卡洛仿真"""

    def __init__(self, M: int = 1000, B: int = 500, seed: Optional[int] = None,
                 estimator_type: str = 'lasso', alpha: Optional[float] = None):
        """
        初始化蒙特卡洛仿真

        Parameters
        ----------
        M : int
            蒙特卡洛重复次数
        B : int
            每次Bootstrap的重复次数
        seed : int, optional
            随机种子
        estimator_type : str
            估计方法：'lasso' 或 'debiased_lasso'
        alpha : float, optional
            正则化参数
        """
        self.M = M
        self.B = B
        self.seed = seed
        self.estimator_type = estimator_type
        self.alpha = alpha

    def evaluate_type1_error(self, N: int, T: int, p: int,
                              Phi: np.ndarray, Sigma: np.ndarray,
                              t: int, test_alpha: float = 0.05,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        评估第一类错误（H0为真时的拒绝率）

        Parameters
        ----------
        N : int
            变量数量
        T : int
            样本长度
        p : int
            滞后阶数
        Phi : np.ndarray
            真实系数矩阵（满足平稳性）
        Sigma : np.ndarray
            残差协方差矩阵
        t : int
            待检验的变点位置
        test_alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            第一类错误评估结果
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
                # 生成无结构变化的序列（H0为真）
                Y = generator.generate_var_series(T, N, p, Phi, Sigma)

                # 执行稀疏Bootstrap LR检验
                bootstrap = SparseBootstrapInference(
                    B=self.B,
                    estimator_type=self.estimator_type,
                    alpha=self.alpha
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
            'estimator_type': self.estimator_type
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
        N : int
            变量数量
        T : int
            样本长度
        p : int
            滞后阶数
        Phi1 : np.ndarray
            断点前的系数矩阵
        Phi2 : np.ndarray
            断点后的系数矩阵
        Sigma : np.ndarray
            残差协方差矩阵
        break_point : int
            真实断点位置
        t : int
            待检验的变点位置
        test_alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            功效评估结果
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
                # 生成含断点的序列（H1为真）
                Y, _ = generator.generate_var_with_break(
                    T, N, p, Phi1, Phi2, Sigma, break_point
                )

                # 执行稀疏Bootstrap LR检验
                bootstrap = SparseBootstrapInference(
                    B=self.B,
                    estimator_type=self.estimator_type,
                    alpha=self.alpha
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
            'estimator_type': self.estimator_type
        }

    def power_curve(self, N: int, T: int, p: int,
                     Phi_base: np.ndarray, Sigma: np.ndarray,
                     delta_values: List[float], break_point: int,
                     test_alpha: float = 0.05,
                     verbose: bool = True) -> Dict[str, Any]:
        """
        绘制功效曲线（Power vs Delta）

        Parameters
        ----------
        N : int
            变量数量
        T : int
            样本长度
        p : int
            滞后阶数
        Phi_base : np.ndarray
            基准系数矩阵
        Sigma : np.ndarray
            残差协方差矩阵
        delta_values : List[float]
            结构变化强度列表
        break_point : int
            断点位置
        test_alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            功效曲线数据
        """
        powers = []

        for delta in delta_values:
            if verbose:
                print(f"\nEvaluating power for delta = {delta}")

            # 构造断点后的系数矩阵
            Phi2 = Phi_base + delta * np.ones_like(Phi_base)

            # 确保Phi2满足平稳性
            if not VARDataGenerator.check_stationarity(Phi2):
                if verbose:
                    print(f"Warning: Phi2 with delta={delta} is not stationary")
                powers.append(np.nan)
                continue

            result = self.evaluate_power(
                N, T, p, Phi_base, Phi2, Sigma, break_point, break_point,
                test_alpha, verbose=False
            )
            powers.append(result['power'])

            if verbose:
                print(f"Power at delta={delta}: {result['power']:.4f}")

        return {
            'delta_values': delta_values,
            'powers': powers,
            'alpha': test_alpha,
            'T': T,
            'N': N,
            'p': p,
            'estimator_type': self.estimator_type
        }
