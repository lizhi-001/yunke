"""
高维稀疏VAR的LR检验模块
使用Lasso/去偏Lasso估计进行结构变化检验
"""

import numpy as np
from typing import Dict, Any, Optional
from .lasso_var import LassoVAREstimator
from .debiased_lasso import DebiasedLassoVAR


class SparseLRTest:
    """高维稀疏VAR的LR结构变化检验"""

    def __init__(self, estimator_type: str = 'lasso', alpha: Optional[float] = None):
        """
        初始化稀疏LR检验

        Parameters
        ----------
        estimator_type : str
            估计方法：'lasso' 或 'debiased_lasso'
        alpha : float, optional
            正则化参数，如果为None则使用交叉验证选择
        """
        self.estimator_type = estimator_type
        self.alpha = alpha
        self.lr_statistic = None

    def _get_estimator(self):
        """获取估计器实例"""
        if self.estimator_type == 'lasso':
            return LassoVAREstimator(alpha=self.alpha)
        elif self.estimator_type == 'debiased_lasso':
            return DebiasedLassoVAR(alpha=self.alpha)
        else:
            raise ValueError(f"未知的估计方法: {self.estimator_type}")

    def _get_log_likelihood(self, result: Dict[str, Any], estimator_type: str) -> float:
        """从估计结果中提取对数似然值"""
        if 'log_likelihood' in result:
            return result['log_likelihood']
        else:
            # 对于去偏Lasso，需要手动计算对数似然
            T_eff = result['T_eff']
            Sigma = result['Sigma']
            N = Sigma.shape[0]
            det_Sigma = np.linalg.det(Sigma)
            if det_Sigma <= 0:
                det_Sigma = 1e-10
            return -0.5 * T_eff * (N * np.log(2 * np.pi) + np.log(det_Sigma) + N)

    def compute_lr_at_point(self, Y: np.ndarray, p: int, t: int,
                            include_const: bool = True) -> Dict[str, Any]:
        """
        针对特定时间点t计算LR统计量（使用稀疏估计）

        Parameters
        ----------
        Y : np.ndarray
            时间序列数据，形状为 (T, N)
        p : int
            滞后阶数
        t : int
            待检验的变点位置
        include_const : bool
            是否包含常数项

        Returns
        -------
        Dict[str, Any]
            包含LR统计量和相关信息的字典
        """
        T, N = Y.shape

        # 边界验证
        min_segment_size = p + 2
        if t < min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第一段样本量不足")
        if t > T - min_segment_size:
            raise ValueError(f"变点位置t={t}无效，第二段样本量不足")

        # H0: 无结构变化（约束模型）
        estimator_r = self._get_estimator()
        result_r = estimator_r.fit(Y, p, include_const)
        log_lik_r = self._get_log_likelihood(result_r, self.estimator_type)

        # H1: 在时间点t发生结构变化（分段拟合）
        Y1 = Y[:t, :]
        Y2 = Y[t:, :]

        estimator1 = self._get_estimator()
        result1 = estimator1.fit(Y1, p, include_const)
        log_lik_1 = self._get_log_likelihood(result1, self.estimator_type)

        estimator2 = self._get_estimator()
        result2 = estimator2.fit(Y2, p, include_const)
        log_lik_2 = self._get_log_likelihood(result2, self.estimator_type)

        # 非约束模型的总对数似然
        log_lik_u = log_lik_1 + log_lik_2

        # LR统计量
        self.lr_statistic = 2 * (log_lik_u - log_lik_r)

        return {
            'lr_statistic': self.lr_statistic,
            'test_point': t,
            'log_lik_restricted': log_lik_r,
            'log_lik_unrestricted': log_lik_u,
            'restricted_result': result_r,
            'segment1_result': result1,
            'segment2_result': result2,
            'estimator_type': self.estimator_type
        }
