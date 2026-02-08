"""
Bootstrap推断模块
用于LR检验和Sup-LR检验的Bootstrap校正
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .var_estimator import VAREstimator
from .sup_lr_test import LRTest, SupLRTest


class BootstrapInference:
    """Bootstrap推断"""

    def __init__(self, B: int = 500, seed: Optional[int] = None):
        """
        初始化Bootstrap推断

        Parameters
        ----------
        B : int
            Bootstrap重复次数
        seed : int, optional
            随机种子
        """
        self.B = B
        self.seed = seed
        self.bootstrap_statistics = None
        self.p_value = None
        self.critical_values = None

    def generate_pseudo_series(self, Y: np.ndarray, p: int,
                                Phi: np.ndarray, c: np.ndarray,
                                residuals: np.ndarray) -> np.ndarray:
        """
        生成Bootstrap伪序列

        Parameters
        ----------
        Y : np.ndarray
            原始时间序列
        p : int
            滞后阶数
        Phi : np.ndarray
            H0下估计的系数矩阵
        c : np.ndarray
            H0下估计的常数项
        residuals : np.ndarray
            H0下的残差

        Returns
        -------
        np.ndarray
            Bootstrap伪序列
        """
        T, N = Y.shape
        T_eff = len(residuals)

        # 残差重抽样（有放回）并居中处理
        # 居中处理确保Bootstrap残差均值为0，减少小样本偏移
        centered_residuals = residuals - np.mean(residuals, axis=0)
        indices = np.random.choice(T_eff, size=T_eff, replace=True)
        resampled_residuals = centered_residuals[indices, :]

        # 生成伪序列
        Y_star = np.zeros((T, N))
        # 使用原始数据的前p个观测值作为初始值
        Y_star[:p, :] = Y[:p, :]

        # 迭代生成
        for t in range(p, T):
            Y_lag_ordered = np.zeros(N * p)
            for lag in range(p):
                Y_lag_ordered[lag*N:(lag+1)*N] = Y_star[t-lag-1, :]

            epsilon_t = resampled_residuals[t - p, :] if t - p < T_eff else np.zeros(N)
            Y_star[t, :] = c + Phi @ Y_lag_ordered + epsilon_t

        return Y_star

    def bootstrap_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                               verbose: bool = False) -> Dict[str, Any]:
        """
        执行针对特定时间点t的Bootstrap LR检验

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            Bootstrap检验结果
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Step 1: 计算原始数据的LR统计量
        lr_test = LRTest()
        original_result = lr_test.compute_lr_at_point(Y, p, t)
        original_lr = original_result['lr_statistic']

        # 获取H0下的估计结果
        restricted_result = original_result['restricted_result']
        Phi_r = restricted_result['Phi']
        c_r = restricted_result['c']
        residuals_r = restricted_result['residuals']

        # Step 2: Bootstrap循环
        bootstrap_lr_values = []

        for b in range(self.B):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{self.B}")

            try:
                # 生成伪序列
                Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)

                # 计算伪序列的LR统计量
                lr_test_b = LRTest()
                result_b = lr_test_b.compute_lr_at_point(Y_star, p, t)
                bootstrap_lr_values.append(result_b['lr_statistic'])
            except Exception:
                continue

        self.bootstrap_statistics = np.array(bootstrap_lr_values)

        # Step 3: 计算p值
        self.p_value = np.mean(self.bootstrap_statistics >= original_lr)

        # Step 4: 计算临界值
        self.critical_values = {
            0.10: np.percentile(self.bootstrap_statistics, 90),
            0.05: np.percentile(self.bootstrap_statistics, 95),
            0.01: np.percentile(self.bootstrap_statistics, 99)
        }

        return {
            'original_lr': original_lr,
            'test_point': t,
            'p_value': self.p_value,
            'critical_values': self.critical_values,
            'bootstrap_statistics': self.bootstrap_statistics,
            'B_effective': len(self.bootstrap_statistics)
        }

    def test_at_point(self, Y: np.ndarray, p: int, t: int,
                       alpha: float = 0.05, verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的针对特定时间点的Bootstrap LR检验

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        alpha : float
            显著性水平
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            检验结果
        """
        result = self.bootstrap_lr_at_point(Y, p, t, verbose)

        # 做出决策
        reject_h0 = result['p_value'] <= alpha

        result['alpha'] = alpha
        result['reject_h0'] = reject_h0
        result['decision'] = f"拒绝H0：在时间点{t}存在结构性变化" if reject_h0 else "接受H0：不存在结构性变化"

        return result

    def bootstrap_sup_lr(self, Y: np.ndarray, p: int,
                          trim: float = 0.15,
                          verbose: bool = False) -> Dict[str, Any]:
        """
        执行Bootstrap Sup-LR检验

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        trim : float
            修剪比例
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            Bootstrap检验结果
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Step 1: 计算原始数据的Sup-LR统计量
        sup_lr_test = SupLRTest(trim=trim)
        original_result = sup_lr_test.compute_sup_lr(Y, p)
        original_sup_lr = original_result['sup_lr']

        # 获取H0下的估计结果
        restricted_result = original_result['restricted_result']
        Phi_r = restricted_result['Phi']
        c_r = restricted_result['c']
        residuals_r = restricted_result['residuals']

        # Step 2: Bootstrap循环
        bootstrap_sup_lr_values = []

        for b in range(self.B):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{self.B}")

            try:
                # 生成伪序列
                Y_star = self.generate_pseudo_series(Y, p, Phi_r, c_r, residuals_r)

                # 计算伪序列的Sup-LR统计量
                sup_lr_test_b = SupLRTest(trim=trim)
                result_b = sup_lr_test_b.compute_sup_lr(Y_star, p)
                bootstrap_sup_lr_values.append(result_b['sup_lr'])
            except Exception:
                continue

        self.bootstrap_statistics = np.array(bootstrap_sup_lr_values)

        # Step 3: 计算p值
        self.p_value = np.mean(self.bootstrap_statistics >= original_sup_lr)

        # Step 4: 计算临界值
        self.critical_values = {
            0.10: np.percentile(self.bootstrap_statistics, 90),
            0.05: np.percentile(self.bootstrap_statistics, 95),
            0.01: np.percentile(self.bootstrap_statistics, 99)
        }

        return {
            'original_sup_lr': original_sup_lr,
            'estimated_break': original_result['estimated_break'],
            'p_value': self.p_value,
            'critical_values': self.critical_values,
            'bootstrap_statistics': self.bootstrap_statistics,
            'B_effective': len(self.bootstrap_statistics)
        }

    def test(self, Y: np.ndarray, p: int, alpha: float = 0.05,
             trim: float = 0.15, verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的Bootstrap Sup-LR检验

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        alpha : float
            显著性水平
        trim : float
            修剪比例
        verbose : bool
            是否显示进度

        Returns
        -------
        Dict[str, Any]
            检验结果
        """
        result = self.bootstrap_sup_lr(Y, p, trim, verbose)

        # 做出决策
        reject_h0 = result['p_value'] <= alpha

        result['alpha'] = alpha
        result['reject_h0'] = reject_h0
        result['decision'] = "拒绝H0：存在结构性变化" if reject_h0 else "接受H0：不存在结构性变化"

        return result
