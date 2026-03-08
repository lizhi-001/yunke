"""
交叉验证调参模块
用于高维稀疏VAR模型的正则化参数选择
"""

import os

import numpy as np
from typing import Tuple, Optional, Dict, Any, List

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

try:
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CrossValidationTuner:
    """交叉验证调参器"""

    def __init__(self, n_splits: int = 5, max_iter: int = 10000):
        """
        初始化交叉验证调参器

        Parameters
        ----------
        n_splits : int
            时间序列交叉验证的折数
        max_iter : int
            最大迭代次数
        """
        if not HAS_SKLEARN:
            raise ImportError("需要安装sklearn: pip install scikit-learn")

        self.n_splits = n_splits
        self.max_iter = max_iter
        self.optimal_alphas = None
        self.cv_results = None

    def tune_lasso_var(self, Y: np.ndarray, p: int,
                        alphas: Optional[np.ndarray] = None,
                        include_const: bool = True) -> Dict[str, Any]:
        """
        使用时间序列交叉验证选择最优正则化参数

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        alphas : np.ndarray, optional
            候选alpha值
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            调参结果
        """
        T, N = Y.shape
        T_eff = T - p

        # 构建设计矩阵
        X = np.zeros((T_eff, N * p))
        for t in range(T_eff):
            for lag in range(p):
                X[t, lag*N:(lag+1)*N] = Y[t + p - lag - 1, :]

        if include_const:
            X = np.column_stack([np.ones(T_eff), X])

        Y_response = Y[p:, :]

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # 对每个方程选择最优alpha
        self.optimal_alphas = []
        self.cv_results = []

        for i in range(N):
            y_i = Y_response[:, i]

            if alphas is not None:
                model = LassoCV(alphas=alphas, cv=tscv, max_iter=self.max_iter)
            else:
                model = LassoCV(cv=tscv, max_iter=self.max_iter)

            model.fit(X, y_i)

            self.optimal_alphas.append(model.alpha_)
            self.cv_results.append({
                'alpha': model.alpha_,
                'alphas_tested': model.alphas_,
                'mse_path': model.mse_path_,
                'coef': model.coef_
            })

        return {
            'optimal_alphas': self.optimal_alphas,
            'mean_alpha': np.mean(self.optimal_alphas),
            'cv_results': self.cv_results
        }

    def information_criterion_selection(self, Y: np.ndarray, p: int,
                                          alphas: np.ndarray,
                                          criterion: str = 'bic') -> Dict[str, Any]:
        """
        使用信息准则选择正则化参数

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据
        p : int
            滞后阶数
        alphas : np.ndarray
            候选alpha值
        criterion : str
            信息准则：'aic', 'bic', 'hqc'

        Returns
        -------
        Dict[str, Any]
            选择结果
        """
        from sklearn.linear_model import Lasso

        T, N = Y.shape
        T_eff = T - p

        # 构建设计矩阵
        X = np.zeros((T_eff, N * p))
        for t in range(T_eff):
            for lag in range(p):
                X[t, lag*N:(lag+1)*N] = Y[t + p - lag - 1, :]

        X = np.column_stack([np.ones(T_eff), X])
        Y_response = Y[p:, :]

        best_alphas = []
        ic_values = []

        for i in range(N):
            y_i = Y_response[:, i]
            best_ic = np.inf
            best_alpha = alphas[0]

            for alpha in alphas:
                model = Lasso(alpha=alpha, max_iter=self.max_iter)
                model.fit(X, y_i)

                # 计算RSS
                residuals = y_i - model.predict(X)
                rss = np.sum(residuals ** 2)

                # 计算有效参数数量（非零系数）
                k = np.sum(np.abs(model.coef_) > 1e-10) + 1  # +1 for intercept

                # 计算信息准则
                if criterion == 'aic':
                    ic = T_eff * np.log(rss / T_eff) + 2 * k
                elif criterion == 'bic':
                    ic = T_eff * np.log(rss / T_eff) + k * np.log(T_eff)
                elif criterion == 'hqc':
                    ic = T_eff * np.log(rss / T_eff) + 2 * k * np.log(np.log(T_eff))
                else:
                    raise ValueError(f"未知的信息准则: {criterion}")

                if ic < best_ic:
                    best_ic = ic
                    best_alpha = alpha

            best_alphas.append(best_alpha)
            ic_values.append(best_ic)

        return {
            'optimal_alphas': best_alphas,
            'mean_alpha': np.mean(best_alphas),
            'ic_values': ic_values,
            'criterion': criterion
        }
