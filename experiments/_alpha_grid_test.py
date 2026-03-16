"""debiased_lasso + lowrank_rrr type I error smoke test at p=3"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from simulation.data_generator import VARDataGenerator
from sparse_var.sparse_bootstrap import SparseBootstrapInference
from lowrank_var.lowrank_bootstrap import LowRankBootstrapInference
from concurrent.futures import ProcessPoolExecutor

T, p, t_star = 500, 3, 250
B = 200
M = 500
JOBS = 10
SEED = 42
TEST_ALPHA = 0.05

# --- sparse (debiased_lasso) ---
N_S = 20
SPARSITY = 0.05
ALPHA_L = 0.02

def _dense_scale(N, p):
    return min(0.3, 0.85 / (N * p) ** 0.5)

gen_s = VARDataGenerator(seed=SEED)
phi_s = gen_s.generate_stationary_phi(N_S, p, sparsity=SPARSITY, scale=_dense_scale(N_S, p))
Sigma_s = np.eye(N_S) * 0.5

def _mc_sparse(m):
    rng = np.random.default_rng(SEED * 10000 + m)
    gen_m = VARDataGenerator(seed=int(rng.integers(0, 2**31)))
    Y = gen_m.generate_var_series(T, N_S, p, phi_s, Sigma_s)
    boot = SparseBootstrapInference(B=B, seed=int(rng.integers(0, 2**31)),
                                     estimator_type="debiased_lasso", alpha=ALPHA_L)
    result = boot.test(Y, p, t_star, alpha=TEST_ALPHA)
    return result['reject_h0']

# --- lowrank (rrr) ---
N_L = 20
RANK = 2

# SR=0.40
gen_l40 = VARDataGenerator(seed=SEED)
phi_l40 = gen_l40.generate_lowrank_phi(N_L, p, rank=RANK, scale=0.3, target_spectral_radius=0.40)
Sigma_l = np.eye(N_L) * 0.5

# SR=0.30
gen_l30 = VARDataGenerator(seed=SEED)
phi_l30 = gen_l30.generate_lowrank_phi(N_L, p, rank=RANK, scale=0.3, target_spectral_radius=0.30)

def _mc_lowrank_40(m):
    rng = np.random.default_rng(SEED * 20000 + m)
    gen_m = VARDataGenerator(seed=int(rng.integers(0, 2**31)))
    Y = gen_m.generate_var_series(T, N_L, p, phi_l40, Sigma_l)
    boot = LowRankBootstrapInference(B=B, seed=int(rng.integers(0, 2**31)),
                                      method="rrr", rank=RANK)
    result = boot.test(Y, p, t_star, alpha=TEST_ALPHA)
    return result['reject_h0']

def _mc_lowrank_30(m):
    rng = np.random.default_rng(SEED * 20000 + m)
    gen_m = VARDataGenerator(seed=int(rng.integers(0, 2**31)))
    Y = gen_m.generate_var_series(T, N_L, p, phi_l30, Sigma_l)
    boot = LowRankBootstrapInference(B=B, seed=int(rng.integers(0, 2**31)),
                                      method="rrr", rank=RANK)
    result = boot.test(Y, p, t_star, alpha=TEST_ALPHA)
    return result['reject_h0']

if __name__ == '__main__':
    # lowrank SR=0.40
    print("Running lowrank_rrr SR=0.40 ...", flush=True)
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        res_40 = list(pool.map(_mc_lowrank_40, range(M)))
    r_40 = np.mean(res_40)
    se_40 = (r_40 * (1 - r_40) / M) ** 0.5
    print(f"lowrank_rrr SR=0.40  type1={r_40:.4f} ±{se_40:.4f}  ({sum(res_40)}/{len(res_40)})", flush=True)

    # lowrank SR=0.30
    print("Running lowrank_rrr SR=0.30 ...", flush=True)
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        res_30 = list(pool.map(_mc_lowrank_30, range(M)))
    r_30 = np.mean(res_30)
    se_30 = (r_30 * (1 - r_30) / M) ** 0.5
    print(f"lowrank_rrr SR=0.30  type1={r_30:.4f} ±{se_30:.4f}  ({sum(res_30)}/{len(res_30)})", flush=True)
