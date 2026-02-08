"""
完整仿真实验脚本
运行所有实验并收集结果
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/Users/lizhi/Developer/python/yunke')

from simulation import (
    VARDataGenerator,
    VAREstimator,
    SupLRTest,
    BootstrapInference,
    MonteCarloSimulation
)

# 创建结果目录
RESULTS_DIR = '/Users/lizhi/Developer/python/yunke/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 实验结果存储
all_results = {
    'experiment_info': {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': '高维稀疏VAR模型结构性变化检验仿真实验'
    },
    'experiments': {}
}


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# 实验1: 基础VAR模型 - 无结构变化检验
# ============================================================
def experiment_1_basic_var():
    print_section("实验1: 基础VAR模型 - 无结构变化检验 (H0为真)")

    # 参数设置
    N = 3  # 变量数量
    T = 200  # 样本长度
    p = 1  # 滞后阶数
    B = 200  # Bootstrap次数

    print(f"参数: N={N}, T={T}, p={p}, B={B}")

    # 生成数据
    generator = VARDataGenerator(seed=42)
    Phi = generator.generate_stationary_phi(N, p, scale=0.3)
    Sigma = np.eye(N) * 0.5

    print(f"\n系数矩阵 Phi:\n{Phi}")
    print(f"平稳性检验: {generator.check_stationarity(Phi)}")

    # 生成无结构变化的序列
    Y = generator.generate_var_series(T, N, p, Phi, Sigma)
    print(f"生成的时间序列形状: {Y.shape}")

    # 执行Sup-LR检验
    print("\n执行Sup-LR检验...")
    sup_lr_test = SupLRTest(trim=0.15)
    sup_lr_result = sup_lr_test.compute_sup_lr(Y, p)

    print(f"Sup-LR统计量: {sup_lr_result['sup_lr']:.4f}")
    print(f"估计的断点位置: {sup_lr_result['estimated_break']}")

    # Bootstrap推断
    print(f"\n执行Bootstrap推断 (B={B})...")
    bootstrap = BootstrapInference(B=B, seed=42)
    boot_result = bootstrap.test(Y, p, alpha=0.05)

    print(f"Bootstrap p值: {boot_result['p_value']:.4f}")
    print(f"临界值 (α=0.10): {boot_result['critical_values'][0.10]:.4f}")
    print(f"临界值 (α=0.05): {boot_result['critical_values'][0.05]:.4f}")
    print(f"临界值 (α=0.01): {boot_result['critical_values'][0.01]:.4f}")
    print(f"决策: {boot_result['decision']}")

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'B': B},
        'Phi': Phi.tolist(),
        'sup_lr': float(sup_lr_result['sup_lr']),
        'estimated_break': int(sup_lr_result['estimated_break']),
        'p_value': float(boot_result['p_value']),
        'critical_values': {str(k): float(v) for k, v in boot_result['critical_values'].items()},
        'reject_h0': bool(boot_result['reject_h0']),
        'decision': boot_result['decision']
    }

    return result


# ============================================================
# 实验2: 含结构断点的VAR模型检验
# ============================================================
def experiment_2_structural_break():
    print_section("实验2: 含结构断点的VAR模型检验 (H1为真)")

    # 参数设置
    N = 3
    T = 200
    p = 1
    B = 200
    break_point = 100  # 断点位置

    print(f"参数: N={N}, T={T}, p={p}, B={B}, 断点={break_point}")

    generator = VARDataGenerator(seed=42)

    # 生成两个不同的系数矩阵
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.2)

    # 创建明显不同的Phi2
    np.random.seed(123)
    Phi2 = generator.generate_stationary_phi(N, p, scale=0.2)
    # 增加差异
    delta = 0.25
    Phi2 = Phi2 + delta * np.sign(Phi2)

    # 确保Phi2平稳
    while not generator.check_stationarity(Phi2):
        Phi2 = Phi2 * 0.9

    Sigma = np.eye(N) * 0.5

    print(f"\n断点前系数矩阵 Phi1:\n{Phi1}")
    print(f"\n断点后系数矩阵 Phi2:\n{Phi2}")
    print(f"\n系数变化量 (Frobenius范数): {np.linalg.norm(Phi2 - Phi1, 'fro'):.4f}")

    # 生成含断点的序列
    Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)

    # 执行Bootstrap Sup-LR检验
    print(f"\n执行Bootstrap Sup-LR检验 (B={B})...")
    bootstrap = BootstrapInference(B=B, seed=42)
    boot_result = bootstrap.test(Y, p, alpha=0.05)

    print(f"Sup-LR统计量: {boot_result['original_sup_lr']:.4f}")
    print(f"估计的断点位置: {boot_result['estimated_break']} (真实: {break_point})")
    print(f"断点估计误差: {abs(boot_result['estimated_break'] - break_point)}")
    print(f"Bootstrap p值: {boot_result['p_value']:.4f}")
    print(f"决策: {boot_result['decision']}")

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'B': B, 'true_break': break_point},
        'Phi1': Phi1.tolist(),
        'Phi2': Phi2.tolist(),
        'coefficient_change_norm': float(np.linalg.norm(Phi2 - Phi1, 'fro')),
        'sup_lr': float(boot_result['original_sup_lr']),
        'estimated_break': int(boot_result['estimated_break']),
        'break_estimation_error': int(abs(boot_result['estimated_break'] - break_point)),
        'p_value': float(boot_result['p_value']),
        'reject_h0': bool(boot_result['reject_h0']),
        'decision': boot_result['decision']
    }

    return result


# ============================================================
# 实验3: 第一类错误评估 (Size)
# ============================================================
def experiment_3_type1_error():
    print_section("实验3: 第一类错误评估 (H0为真时的拒绝率)")

    # 参数设置 (为了演示使用较小的M和B)
    N = 2
    T = 100
    p = 1
    M = 50  # 蒙特卡洛次数
    B = 100  # Bootstrap次数
    alpha = 0.05

    print(f"参数: N={N}, T={T}, p={p}, M={M}, B={B}, α={alpha}")
    print("注意: 实际应用中建议 M=1000-5000, B=500")

    generator = VARDataGenerator(seed=42)
    Phi = generator.generate_stationary_phi(N, p, scale=0.3)
    Sigma = np.eye(N) * 0.5

    print(f"\n系数矩阵 Phi:\n{Phi}")

    # 蒙特卡洛仿真
    print(f"\n开始蒙特卡洛仿真 (M={M})...")
    mc = MonteCarloSimulation(M=M, B=B, seed=42)
    type1_result = mc.evaluate_type1_error(N, T, p, Phi, Sigma, alpha=alpha, verbose=True)

    print(f"\n===== 第一类错误评估结果 =====")
    print(f"名义显著性水平 α: {alpha}")
    print(f"实际拒绝率 (第一类错误): {type1_result['type1_error']:.4f}")
    print(f"Size失真: {type1_result['size_distortion']:.4f}")
    print(f"拒绝次数: {type1_result['rejections']} / {M}")

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'M': M, 'B': B, 'alpha': alpha},
        'Phi': Phi.tolist(),
        'type1_error': float(type1_result['type1_error']),
        'nominal_alpha': alpha,
        'size_distortion': float(type1_result['size_distortion']),
        'rejections': int(type1_result['rejections']),
        'p_values_mean': float(np.mean(type1_result['p_values'])),
        'p_values_std': float(np.std(type1_result['p_values']))
    }

    return result


# ============================================================
# 实验4: 统计功效评估 (Power)
# ============================================================
def experiment_4_power():
    print_section("实验4: 统计功效评估 (H1为真时的拒绝率)")

    # 参数设置
    N = 2
    T = 100
    p = 1
    M = 50
    B = 100
    alpha = 0.05
    break_point = 50

    print(f"参数: N={N}, T={T}, p={p}, M={M}, B={B}, α={alpha}, 断点={break_point}")

    generator = VARDataGenerator(seed=42)
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.3)

    # 创建不同强度的结构变化
    delta_values = [0.1, 0.2, 0.3]
    power_results = []

    Sigma = np.eye(N) * 0.5

    print(f"\n基准系数矩阵 Phi1:\n{Phi1}")

    for delta in delta_values:
        print(f"\n--- 结构变化强度 Δ = {delta} ---")

        # 构造Phi2
        Phi2 = Phi1 + delta * np.ones_like(Phi1)

        # 确保平稳
        attempts = 0
        while not generator.check_stationarity(Phi2) and attempts < 10:
            Phi2 = Phi2 * 0.9
            attempts += 1

        if not generator.check_stationarity(Phi2):
            print(f"警告: Δ={delta} 无法生成平稳的Phi2，跳过")
            power_results.append({'delta': delta, 'power': None, 'skipped': True})
            continue

        print(f"Phi2:\n{Phi2}")
        print(f"系数变化量: {np.linalg.norm(Phi2 - Phi1, 'fro'):.4f}")

        # 评估功效
        mc = MonteCarloSimulation(M=M, B=B, seed=42)
        power_result = mc.evaluate_power(N, T, p, Phi1, Phi2, Sigma, break_point,
                                          alpha=alpha, verbose=False)

        print(f"功效 (Power): {power_result['power']:.4f}")
        print(f"断点估计偏差: {power_result['break_estimation_bias']:.2f}")
        print(f"断点估计RMSE: {power_result['break_estimation_rmse']:.2f}")

        power_results.append({
            'delta': delta,
            'power': float(power_result['power']),
            'break_estimation_bias': float(power_result['break_estimation_bias']),
            'break_estimation_rmse': float(power_result['break_estimation_rmse']),
            'skipped': False
        })

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'M': M, 'B': B, 'alpha': alpha, 'break_point': break_point},
        'Phi1': Phi1.tolist(),
        'power_curve': power_results
    }

    return result


# ============================================================
# 实验5: 高维稀疏VAR模型估计
# ============================================================
def experiment_5_sparse_var():
    print_section("实验5: 高维稀疏VAR模型 (Lasso估计)")

    try:
        from sparse_var import LassoVAREstimator
    except ImportError as e:
        print(f"跳过稀疏VAR实验: {e}")
        return {'skipped': True, 'reason': str(e)}

    # 参数设置
    N = 10  # 高维
    T = 300
    p = 1
    sparsity = 0.2  # 20%非零

    print(f"参数: N={N}, T={T}, p={p}, 稀疏度={sparsity}")

    generator = VARDataGenerator(seed=42)

    # 生成稀疏系数矩阵
    Phi_true = generator.generate_stationary_phi(N, p, sparsity=sparsity, scale=0.3)
    Sigma = np.eye(N) * 0.5

    true_sparsity = np.mean(np.abs(Phi_true) < 1e-10)
    true_nonzero = np.sum(np.abs(Phi_true) > 1e-10)

    print(f"\n真实系数矩阵稀疏度: {true_sparsity:.2%}")
    print(f"真实非零系数数量: {true_nonzero} / {Phi_true.size}")

    # 生成数据
    Y = generator.generate_var_series(T, N, p, Phi_true, Sigma)

    # Lasso-VAR估计
    print("\n执行Lasso-VAR估计...")
    lasso_var = LassoVAREstimator(cv=5)
    lasso_result = lasso_var.fit(Y, p)

    estimated_sparsity = lasso_result['sparsity']
    estimated_nonzero = np.sum(np.abs(lasso_result['Phi']) > 1e-10)

    print(f"估计的稀疏度: {estimated_sparsity:.2%}")
    print(f"估计的非零系数数量: {estimated_nonzero} / {lasso_result['Phi'].size}")
    print(f"平均正则化参数 λ: {np.mean(lasso_result['alphas_used']):.6f}")

    # 计算估计误差
    estimation_error = np.linalg.norm(lasso_result['Phi'] - Phi_true, 'fro')
    print(f"估计误差 (Frobenius范数): {estimation_error:.4f}")

    # 获取非零系数信息
    nonzero_info = lasso_var.get_nonzero_coefficients()

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'true_sparsity': sparsity},
        'true_sparsity': float(true_sparsity),
        'true_nonzero_count': int(true_nonzero),
        'estimated_sparsity': float(estimated_sparsity),
        'estimated_nonzero_count': int(estimated_nonzero),
        'mean_lambda': float(np.mean(lasso_result['alphas_used'])),
        'estimation_error': float(estimation_error),
        'log_likelihood': float(lasso_result['log_likelihood']),
        'skipped': False
    }

    return result


# ============================================================
# 实验6: 低秩VAR模型估计
# ============================================================
def experiment_6_lowrank_var():
    print_section("实验6: 低秩VAR模型 (核范数正则化)")

    from lowrank_var import NuclearNormVAR, RankSelector

    # 参数设置
    N = 8
    T = 250
    p = 1
    true_rank = 2

    print(f"参数: N={N}, T={T}, p={p}, 真实秩={true_rank}")

    generator = VARDataGenerator(seed=42)

    # 生成低秩系数矩阵
    Phi_true = generator.generate_lowrank_phi(N, p, rank=true_rank, scale=0.3)
    Sigma = np.eye(N) * 0.5

    # 验证真实秩
    _, s_true, _ = np.linalg.svd(Phi_true)
    actual_rank = np.sum(s_true > 1e-6)

    print(f"\n真实系数矩阵的奇异值: {s_true[:5]}")
    print(f"实际秩: {actual_rank}")

    # 生成数据
    Y = generator.generate_var_series(T, N, p, Phi_true, Sigma)

    # 秩选择
    print("\n执行秩选择...")
    rank_selector = RankSelector()

    # 方法1: 特征值比例法
    ev_result = rank_selector.select_by_eigenvalue_ratio(Y, p, threshold=0.9)
    print(f"特征值比例法选择的秩: {ev_result['selected_rank']}")

    # 方法2: BIC
    bic_result = rank_selector.select_by_information_criterion(Y, p, max_rank=5, criterion='bic')
    print(f"BIC选择的秩: {bic_result['selected_rank']}")

    # 低秩VAR估计 (截断SVD)
    print("\n执行低秩VAR估计 (截断SVD)...")
    lowrank_var = NuclearNormVAR()
    lr_result = lowrank_var.fit_svd(Y, p, rank=true_rank)

    print(f"估计的秩: {lr_result['rank']}")
    print(f"对数似然值: {lr_result['log_likelihood']:.4f}")

    # 计算估计误差
    estimation_error = np.linalg.norm(lr_result['Phi'] - Phi_true, 'fro')
    print(f"估计误差 (Frobenius范数): {estimation_error:.4f}")

    result = {
        'parameters': {'N': N, 'T': T, 'p': p, 'true_rank': true_rank},
        'true_singular_values': s_true.tolist(),
        'actual_rank': int(actual_rank),
        'ev_selected_rank': int(ev_result['selected_rank']),
        'bic_selected_rank': int(bic_result['selected_rank']),
        'estimated_rank': int(lr_result['rank']),
        'estimation_error': float(estimation_error),
        'log_likelihood': float(lr_result['log_likelihood'])
    }

    return result


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#    高维稀疏VAR模型结构性变化检验 - 完整仿真实验    #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    start_time = datetime.now()

    # 运行所有实验
    print("\n开始运行实验...")

    # 实验1
    all_results['experiments']['exp1_basic_var'] = experiment_1_basic_var()

    # 实验2
    all_results['experiments']['exp2_structural_break'] = experiment_2_structural_break()

    # 实验3
    all_results['experiments']['exp3_type1_error'] = experiment_3_type1_error()

    # 实验4
    all_results['experiments']['exp4_power'] = experiment_4_power()

    # 实验5
    all_results['experiments']['exp5_sparse_var'] = experiment_5_sparse_var()

    # 实验6
    all_results['experiments']['exp6_lowrank_var'] = experiment_6_lowrank_var()

    # 计算总耗时
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    all_results['experiment_info']['duration_seconds'] = duration

    print_section("实验完成")
    print(f"总耗时: {duration:.2f} 秒")

    # 保存结果
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {results_file}")

    return all_results


if __name__ == '__main__':
    results = main()
