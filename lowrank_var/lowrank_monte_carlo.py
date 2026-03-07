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
from simulation.parallel import run_task_map
from .lowrank_bootstrap import LowRankBootstrapInference


def _iteration_seed(base_seed: Optional[int], iteration: int) -> Optional[int]:
    return None if base_seed is None else base_seed + iteration


def _type1_worker(task):
    seed, N, T, p, Phi, Sigma, t, test_alpha, B, method, rank, lambda_nuc = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)
        bootstrap = LowRankBootstrapInference(B=B, method=method, rank=rank, lambda_nuc=lambda_nuc)
        result = bootstrap.test(Y, p, t, alpha=test_alpha)
        return {'success': True, 'p_value': result['p_value'], 'reject_h0': result['reject_h0']}
    except Exception:
        return {'success': False}


def _power_worker(task):
    seed, N, T, p, Phi1, Phi2, Sigma, break_point, t, test_alpha, B, method, rank, lambda_nuc = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)
        bootstrap = LowRankBootstrapInference(B=B, method=method, rank=rank, lambda_nuc=lambda_nuc)
        result = bootstrap.test(Y, p, t, alpha=test_alpha)
        return {'success': True, 'p_value': result['p_value'], 'reject_h0': result['reject_h0']}
    except Exception:
        return {'success': False}


class LowRankMonteCarloSimulation:
    """低秩VAR的蒙特卡洛仿真"""

    def __init__(self, M: int = 1000, B: int = 500,
                 seed: Optional[int] = None, method: str = 'svd',
                 rank: Optional[int] = None,
                 lambda_nuc: Optional[float] = None,
                 n_jobs: int = 1):
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
        self.n_jobs = max(1, n_jobs)

    def _run_tasks(self, worker, tasks, verbose: bool):
        return run_task_map(
            worker,
            tasks,
            n_jobs=self.n_jobs,
            verbose=verbose,
            progress_every=10,
            progress_label="Monte Carlo iteration",
        )

    @staticmethod
    def _collect_results(results):
        rejections = 0
        p_values = []
        successful_iterations = 0
        for result in results:
            if not result.get('success'):
                continue
            p_values.append(result['p_value'])
            successful_iterations += 1
            if result['reject_h0']:
                rejections += 1
        return rejections, p_values, successful_iterations

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
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi, Sigma, t, test_alpha,
             self.B, self.method, self.rank, self.lambda_nuc)
            for m in range(self.M)
        ]
        results = self._run_tasks(_type1_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_results(results)

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
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi1, Phi2, Sigma, break_point,
             t, test_alpha, self.B, self.method, self.rank, self.lambda_nuc)
            for m in range(self.M)
        ]
        results = self._run_tasks(_power_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_results(results)

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
