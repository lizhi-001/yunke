"""
高维稀疏VAR的LR检验模块
使用Lasso/去偏Lasso估计进行已知断点下的结构断裂检验。

口径说明：
- H0：已知点处不存在结构断裂，整条时间序列共用一套参数；
- H1：已知点处存在结构断裂，断点前后使用两套参数；
- 第二段从断点后第一个响应开始生效，但允许其滞后项借用断点前的 p 个观测。
"""

import numpy as np
from typing import Dict, Any, Optional
from .lasso_var import LassoVAREstimator
from .debiased_lasso import DebiasedLassoVAR


class SparseLRTest:
    """高维稀疏VAR的LR结构变化检验"""

    def __init__(self, estimator_type: str = 'lasso', alpha: Optional[float] = None,
                 post_lasso_ols: bool = False):
        """
        初始化稀疏LR检验

        Parameters
        ----------
        estimator_type : str
            估计方法：'lasso' 或 'debiased_lasso'
        alpha : float, optional
            正则化参数，如果为None则使用交叉验证选择
        post_lasso_ols : bool
            是否使用Post-Lasso OLS重拟合
        """
        self.estimator_type = estimator_type
        self.alpha = alpha
        self.post_lasso_ols = post_lasso_ols
        self.lr_statistic = None
        self._estimator = None

    def _get_estimator(self):
        """获取估计器实例"""
        if self._estimator is not None:
            return self._estimator

        if self.estimator_type == 'lasso':
            self._estimator = LassoVAREstimator(alpha=self.alpha,
                                                 post_lasso_ols=self.post_lasso_ols)
        elif self.estimator_type == 'debiased_lasso':
            self._estimator = DebiasedLassoVAR(alpha=self.alpha)
        else:
            raise ValueError(f"未知的估计方法: {self.estimator_type}")
        return self._estimator

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
                            include_const: bool = True,
                            support_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        support_mask : np.ndarray, optional
            固定支撑集掩码（shape: n_features × N）。
            若提供，则 H0/H1 均用 OLS on 固定支撑集，实现固定支撑 Post-Lasso OLS。

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
        if support_mask is not None:
            estimator = LassoVAREstimator(alpha=self.alpha)
            result_r = estimator.fit_with_support(Y, p, support_mask, include_const)
        else:
            estimator = self._get_estimator()
            result_r = estimator.fit(Y, p, include_const)
        log_lik_r = self._get_log_likelihood(result_r, self.estimator_type)

        # H1: 在时间点t发生结构变化（结构断裂拟合）
        Y1 = Y[:t, :]
        Y2 = Y[t - p:, :]

        if support_mask is not None:
            result1 = estimator.fit_with_support(Y1, p, support_mask, include_const)
            log_lik_1 = self._get_log_likelihood(result1, self.estimator_type)
            result2 = estimator.fit_with_support(Y2, p, support_mask, include_const)
            log_lik_2 = self._get_log_likelihood(result2, self.estimator_type)
        else:
            result1 = estimator.fit(Y1, p, include_const)
            log_lik_1 = self._get_log_likelihood(result1, self.estimator_type)
            result2 = estimator.fit(Y2, p, include_const)
            log_lik_2 = self._get_log_likelihood(result2, self.estimator_type)

        if result_r['T_eff'] != result1['T_eff'] + result2['T_eff']:
            raise ValueError('H0 与 H1 的有效样本量不一致')

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
