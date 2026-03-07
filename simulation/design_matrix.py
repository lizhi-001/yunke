"""Utilities for building VAR lag design matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_var_design_matrix(
    Y: np.ndarray,
    p: int,
    include_const: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the lagged design matrix for a VAR(p) model.

    The implementation stacks lag blocks with vectorized slicing over time,
    which is substantially faster than nested Python loops when Monte Carlo and
    bootstrap repeatedly rebuild the same type of matrix.
    """
    T, _ = Y.shape
    if p <= 0:
        raise ValueError(f"滞后阶数p必须为正整数，当前p={p}")
    if T <= p:
        raise ValueError(f"样本长度T必须大于滞后阶数p，当前T={T}, p={p}")

    T_eff = T - p
    lagged_blocks = [Y[p - lag - 1:T - lag - 1, :] for lag in range(p)]
    X = np.hstack(lagged_blocks)

    if include_const:
        X = np.column_stack([np.ones(T_eff), X])

    Y_response = Y[p:, :]
    return X, Y_response
