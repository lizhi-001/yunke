"""
秩选择模块
用于低秩VAR模型的秩自动选择
"""

import numpy as np
from numpy.linalg import svd
from typing import Tuple, Optional, Dict, Any, List


class RankSelector:
    """秩选择器"""

    def __init__(self):
        """初始化秩选择器"""
        self.selected_rank = None
        self.selection_results = None

    def select_by_eigenvalue_ratio(self, Y: np.ndarray, p: int,
                                    threshold: float = 0.9) -> Dict[str, Any]:
        """
        使用特征值比例选择秩

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        threshold : float
            累积方差解释比例阈值

        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        T, N = Y.shape
        T_eff = T - p

        # 构建设计矩阵
        X = np.zeros((T_eff, N * p))
        for t in range(T_eff):
            for lag in range(p):
                X[t, lag*N:(lag+1)*N] = Y[t + p - lag - 1, :]

        X_full = np.column_stack([np.ones(T_eff), X])
        Y_response = Y[p:, :]

        # OLS估计
        B_ols = np.linalg.lstsq(X_full, Y_response, rcond=None)[0]
        Phi_ols = B_ols[1:, :].T

        # SVD分解
        U, s, Vt = svd(Phi_ols, full_matrices=False)

        # 计算累积方差解释比例
        total_variance = np.sum(s ** 2)
        cumulative_variance = np.cumsum(s ** 2) / total_variance

        # 选择秩
        self.selected_rank = np.searchsorted(cumulative_variance, threshold) + 1
        self.selected_rank = min(self.selected_rank, len(s))

        return {
            'selected_rank': self.selected_rank,
            'singular_values': s,
            'cumulative_variance_ratio': cumulative_variance,
            'threshold': threshold
        }

    def select_by_information_criterion(self, Y: np.ndarray, p: int,
                                          max_rank: Optional[int] = None,
                                          criterion: str = 'bic') -> Dict[str, Any]:
        """
        使用信息准则选择秩

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        max_rank : int, optional
            最大秩
        criterion : str
            信息准则：'aic', 'bic'

        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        from .nuclear_norm import NuclearNormVAR

        T, N = Y.shape
        T_eff = T - p

        if max_rank is None:
            max_rank = min(N, N * p)

        ic_values = []
        log_likelihoods = []

        for rank in range(1, max_rank + 1):
            try:
                estimator = NuclearNormVAR()
                result = estimator.fit_svd(Y, p, rank=rank)

                log_lik = result['log_likelihood']
                log_likelihoods.append(log_lik)

                # 计算有效参数数量
                k = rank * (N + N * p - rank)  # 低秩矩阵的自由度

                # 计算信息准则
                if criterion == 'aic':
                    ic = -2 * log_lik + 2 * k
                elif criterion == 'bic':
                    ic = -2 * log_lik + k * np.log(T_eff)
                else:
                    raise ValueError(f"未知的信息准则: {criterion}")

                ic_values.append(ic)
            except Exception:
                ic_values.append(np.inf)
                log_likelihoods.append(-np.inf)

        # 选择最优秩
        self.selected_rank = np.argmin(ic_values) + 1

        return {
            'selected_rank': self.selected_rank,
            'ic_values': ic_values,
            'log_likelihoods': log_likelihoods,
            'criterion': criterion,
            'ranks_tested': list(range(1, max_rank + 1))
        }

    def select_by_cross_validation(self, Y: np.ndarray, p: int,
                                     max_rank: Optional[int] = None,
                                     n_splits: int = 5) -> Dict[str, Any]:
        """
        使用交叉验证选择秩

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        max_rank : int, optional
            最大秩
        n_splits : int
            交叉验证折数

        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        from .nuclear_norm import NuclearNormVAR

        T, N = Y.shape

        if max_rank is None:
            max_rank = min(N, N * p)

        # 时间序列交叉验证
        fold_size = T // (n_splits + 1)
        cv_errors = []

        for rank in range(1, max_rank + 1):
            rank_errors = []

            for fold in range(n_splits):
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, T)

                if test_end <= test_start + p:
                    continue

                Y_train = Y[:train_end, :]
                Y_test = Y[test_start:test_end, :]

                try:
                    # 在训练集上拟合
                    estimator = NuclearNormVAR()
                    result = estimator.fit_svd(Y_train, p, rank=rank)

                    # 在测试集上预测
                    Phi = result['Phi']
                    c = result['c']

                    # 计算预测误差
                    mse = 0
                    count = 0
                    for t in range(p, len(Y_test)):
                        Y_lag = np.zeros(N * p)
                        for lag in range(p):
                            Y_lag[lag*N:(lag+1)*N] = Y_test[t-lag-1, :]
                        Y_pred = c + Phi @ Y_lag
                        mse += np.sum((Y_test[t, :] - Y_pred) ** 2)
                        count += 1

                    if count > 0:
                        rank_errors.append(mse / count)
                except Exception:
                    continue

            if rank_errors:
                cv_errors.append(np.mean(rank_errors))
            else:
                cv_errors.append(np.inf)

        # 选择最优秩
        self.selected_rank = np.argmin(cv_errors) + 1

        return {
            'selected_rank': self.selected_rank,
            'cv_errors': cv_errors,
            'n_splits': n_splits,
            'ranks_tested': list(range(1, max_rank + 1))
        }
