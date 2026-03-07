"""
蒙特卡洛仿真模块
用于评估LR检验和Sup-LR检验的第一类错误和统计功效
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from .data_generator import VARDataGenerator
from .bootstrap import BootstrapInference
from .parallel import run_task_map


def _iteration_seed(base_seed: Optional[int], iteration: int) -> Optional[int]:
    return None if base_seed is None else base_seed + iteration


def _type1_at_point_worker(task: Tuple) -> Dict[str, Any]:
    seed, N, T, p, Phi, Sigma, t, alpha, B = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)
        bootstrap = BootstrapInference(B=B)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha)
        return {'success': True, 'p_value': result['p_value'], 'reject_h0': result['reject_h0']}
    except Exception:
        return {'success': False}


def _power_at_point_worker(task: Tuple) -> Dict[str, Any]:
    seed, N, T, p, Phi1, Phi2, Sigma, break_point, t, alpha, B = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)
        bootstrap = BootstrapInference(B=B)
        result = bootstrap.test_at_point(Y, p, t, alpha=alpha)
        return {'success': True, 'p_value': result['p_value'], 'reject_h0': result['reject_h0']}
    except Exception:
        return {'success': False}


def _type1_sup_worker(task: Tuple) -> Dict[str, Any]:
    seed, N, T, p, Phi, Sigma, alpha, trim, B = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y = generator.generate_var_series(T, N, p, Phi, Sigma)
        bootstrap = BootstrapInference(B=B)
        result = bootstrap.test(Y, p, alpha=alpha, trim=trim)
        return {'success': True, 'p_value': result['p_value'], 'reject_h0': result['reject_h0']}
    except Exception:
        return {'success': False}


def _power_sup_worker(task: Tuple) -> Dict[str, Any]:
    seed, N, T, p, Phi1, Phi2, Sigma, break_point, alpha, trim, B = task
    if seed is not None:
        np.random.seed(seed)

    generator = VARDataGenerator()
    try:
        Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)
        bootstrap = BootstrapInference(B=B)
        result = bootstrap.test(Y, p, alpha=alpha, trim=trim)
        return {
            'success': True,
            'p_value': result['p_value'],
            'reject_h0': result['reject_h0'],
            'estimated_break': result['estimated_break'],
        }
    except Exception:
        return {'success': False}


class MonteCarloSimulation:
    """蒙特卡洛仿真"""

    def __init__(self, M: int = 1000, B: int = 500,
                 seed: Optional[int] = None, n_jobs: int = 1):
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
        """
        self.M = M
        self.B = B
        self.seed = seed
        self.n_jobs = max(1, n_jobs)

    def _run_tasks(self, worker, tasks: List[Tuple], verbose: bool) -> List[Dict[str, Any]]:
        return run_task_map(
            worker,
            tasks,
            n_jobs=self.n_jobs,
            verbose=verbose,
            progress_every=10,
            progress_label="Monte Carlo iteration",
        )

    @staticmethod
    def _collect_binary_results(results: List[Dict[str, Any]]) -> Tuple[int, List[float], int]:
        rejections = 0
        p_values: List[float] = []
        successful_iterations = 0
        for result in results:
            if not result.get('success'):
                continue
            p_values.append(result['p_value'])
            successful_iterations += 1
            if result['reject_h0']:
                rejections += 1
        return rejections, p_values, successful_iterations

    def evaluate_type1_error_at_point(self, N: int, T: int, p: int,
                                       Phi: np.ndarray, Sigma: np.ndarray,
                                       t: int, alpha: float = 0.05,
                                       verbose: bool = True) -> Dict[str, Any]:
        """
        评估针对特定时间点t的第一类错误（H0为真时的拒绝率）

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
        alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            第一类错误评估结果
        """
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi, Sigma, t, alpha, self.B)
            for m in range(self.M)
        ]
        results = self._run_tasks(_type1_at_point_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_binary_results(results)

        type1_error = rejections / successful_iterations if successful_iterations > 0 else np.nan

        return {
            'type1_error': type1_error,
            'nominal_alpha': alpha,
            'test_point': t,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values),
            'size_distortion': type1_error - alpha if not np.isnan(type1_error) else np.nan
        }

    def evaluate_power_at_point(self, N: int, T: int, p: int,
                                 Phi1: np.ndarray, Phi2: np.ndarray,
                                 Sigma: np.ndarray, break_point: int,
                                 t: int, alpha: float = 0.05,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        评估针对特定时间点t的统计功效（H1为真时的拒绝率）

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
            待检验的变点位置（通常设为break_point）
        alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            功效评估结果
        """
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi1, Phi2, Sigma, break_point, t, alpha, self.B)
            for m in range(self.M)
        ]
        results = self._run_tasks(_power_at_point_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_binary_results(results)

        power = rejections / successful_iterations if successful_iterations > 0 else np.nan

        return {
            'power': power,
            'alpha': alpha,
            'test_point': t,
            'true_break': break_point,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values)
        }

    def evaluate_type1_error(self, N: int, T: int, p: int,
                              Phi: np.ndarray, Sigma: np.ndarray,
                              alpha: float = 0.05,
                              trim: float = 0.15,
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
        alpha : float
            显著性水平
        trim : float
            修剪比例
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            第一类错误评估结果
        """
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi, Sigma, alpha, trim, self.B)
            for m in range(self.M)
        ]
        results = self._run_tasks(_type1_sup_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_binary_results(results)

        # 计算实际第一类错误率（使用成功迭代次数作为分母）
        type1_error = rejections / successful_iterations if successful_iterations > 0 else np.nan

        return {
            'type1_error': type1_error,
            'nominal_alpha': alpha,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values),
            'size_distortion': type1_error - alpha if not np.isnan(type1_error) else np.nan
        }

    def evaluate_power(self, N: int, T: int, p: int,
                        Phi1: np.ndarray, Phi2: np.ndarray,
                        Sigma: np.ndarray, break_point: int,
                        alpha: float = 0.05,
                        trim: float = 0.15,
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
            断点位置
        alpha : float
            显著性水平
        trim : float
            修剪比例
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            功效评估结果
        """
        tasks = [
            (_iteration_seed(self.seed, m), N, T, p, Phi1, Phi2, Sigma, break_point, alpha, trim, self.B)
            for m in range(self.M)
        ]
        results = self._run_tasks(_power_sup_worker, tasks, verbose)
        rejections, p_values, successful_iterations = self._collect_binary_results(results)
        estimated_breaks = [result['estimated_break'] for result in results if result.get('success')]

        # 计算功效（使用成功迭代次数作为分母）
        power = rejections / successful_iterations if successful_iterations > 0 else np.nan

        # 计算断点估计的准确性
        estimated_breaks = np.array(estimated_breaks)
        break_estimation_bias = np.mean(estimated_breaks - break_point) if len(estimated_breaks) > 0 else np.nan
        break_estimation_rmse = np.sqrt(np.mean((estimated_breaks - break_point) ** 2)) if len(estimated_breaks) > 0 else np.nan

        return {
            'power': power,
            'alpha': alpha,
            'rejections': rejections,
            'M': self.M,
            'M_effective': successful_iterations,
            'p_values': np.array(p_values),
            'true_break': break_point,
            'estimated_breaks': estimated_breaks,
            'break_estimation_bias': break_estimation_bias,
            'break_estimation_rmse': break_estimation_rmse
        }

    def power_curve(self, N: int, T: int, p: int,
                     Phi_base: np.ndarray, Sigma: np.ndarray,
                     delta_values: List[float],
                     break_point: int,
                     alpha: float = 0.05,
                     trim: float = 0.15,
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
        alpha : float
            显著性水平
        trim : float
            修剪比例
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
                N, T, p, Phi_base, Phi2, Sigma, break_point,
                alpha, trim, verbose=False
            )
            powers.append(result['power'])

            if verbose:
                print(f"Power at delta={delta}: {result['power']:.4f}")

        return {
            'delta_values': delta_values,
            'powers': powers,
            'alpha': alpha,
            'T': T,
            'N': N,
            'p': p
        }
