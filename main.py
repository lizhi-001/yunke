"""
VAR模型结构性变化检验 - 主程序入口
Structural Change Testing in VAR Models

基于开题报告《高维向量自回归模型的结构性变化检验》实现
支持：正常VAR模型检验 + 高维稀疏VAR模型检验
"""

import numpy as np
import argparse
from typing import Optional
from datetime import datetime

# 导入仿真模块
from simulation import (
    VARDataGenerator,
    VAREstimator,
    LRTest,
    SupLRTest,
    BootstrapInference,
    MonteCarloSimulation
)

# 导入稀疏VAR模块
from sparse_var import (
    LassoVAREstimator,
    DebiasedLassoVAR,
    SparseLRTest,
    SparseBootstrapInference,
    SparseMonteCarloSimulation
)


def demo_lr_at_point():
    """演示针对特定时间点的LR检验"""
    print("=" * 60)
    print("演示1: 针对特定时间点的LR结构变化检验")
    print("=" * 60)

    # 参数设置
    N = 3  # 变量数量
    T = 200  # 样本长度
    p = 1  # 滞后阶数
    test_point = 100  # 待检验的时间点

    # 生成数据
    generator = VARDataGenerator(seed=42)
    Phi = generator.generate_stationary_phi(N, p, scale=0.3)
    Sigma = np.eye(N) * 0.5

    print(f"\n参数设置: N={N}, T={T}, p={p}, 检验点t={test_point}")
    print(f"系数矩阵 Phi:\n{Phi}")

    # 生成无结构变化的序列（H0为真）
    Y = generator.generate_var_series(T, N, p, Phi, Sigma)
    print(f"\n生成的时间序列形状: {Y.shape}")

    # 执行针对特定点的LR检验
    print(f"\n执行针对时间点t={test_point}的LR检验...")
    lr_test = LRTest()
    result = lr_test.compute_lr_at_point(Y, p, test_point)

    print(f"LR统计量: {result['lr_statistic']:.4f}")

    # Bootstrap推断
    print(f"\n执行Bootstrap推断 (B=100)...")
    bootstrap = BootstrapInference(B=100, seed=42)
    boot_result = bootstrap.test_at_point(Y, p, test_point, alpha=0.05)

    print(f"Bootstrap p值: {boot_result['p_value']:.4f}")
    print(f"临界值 (α=0.05): {boot_result['critical_values'][0.05]:.4f}")
    print(f"决策: {boot_result['decision']}")

    return boot_result


def demo_structural_break_at_point():
    """演示含结构断点的VAR模型检验（针对特定点）"""
    print("\n" + "=" * 60)
    print("演示2: 含结构断点的VAR模型检验（针对特定点）")
    print("=" * 60)

    # 参数设置
    N = 3
    T = 200
    p = 1
    break_point = 100  # 真实断点位置
    test_point = 100   # 检验点（设为真实断点）

    generator = VARDataGenerator(seed=42)

    # 生成两个不同的系数矩阵
    Phi1 = generator.generate_stationary_phi(N, p, scale=0.2)
    Phi2 = generator.generate_stationary_phi(N, p, scale=0.2)
    Phi2 = Phi2 + 0.3 * np.ones_like(Phi2)

    while not generator.check_stationarity(Phi2):
        Phi2 = Phi2 * 0.9

    Sigma = np.eye(N) * 0.5

    print(f"\n断点前系数矩阵 Phi1:\n{Phi1}")
    print(f"\n断点后系数矩阵 Phi2:\n{Phi2}")
    print(f"\n真实断点位置: {break_point}, 检验点: {test_point}")

    # 生成含断点的序列
    Y, _ = generator.generate_var_with_break(T, N, p, Phi1, Phi2, Sigma, break_point)

    # 执行针对特定点的Bootstrap LR检验
    print(f"\n执行针对时间点t={test_point}的Bootstrap LR检验 (B=100)...")
    bootstrap = BootstrapInference(B=100, seed=42)
    result = bootstrap.test_at_point(Y, p, test_point, alpha=0.05)

    print(f"LR统计量: {result['original_lr']:.4f}")
    print(f"Bootstrap p值: {result['p_value']:.4f}")
    print(f"决策: {result['decision']}")

    return result


def demo_sparse_var_test():
    """演示高维稀疏VAR模型的结构变化检验"""
    print("\n" + "=" * 60)
    print("演示3: 高维稀疏VAR模型结构变化检验 (Lasso)")
    print("=" * 60)

    # 参数设置
    N = 10  # 高维
    T = 300
    p = 1
    sparsity = 0.2
    test_point = 150

    generator = VARDataGenerator(seed=42)

    # 生成稀疏系数矩阵
    Phi = generator.generate_stationary_phi(N, p, sparsity=sparsity, scale=0.3)
    Sigma = np.eye(N) * 0.5

    true_sparsity = np.mean(np.abs(Phi) < 1e-10)
    print(f"\n参数设置: N={N}, T={T}, p={p}, 检验点t={test_point}")
    print(f"真实稀疏度: {true_sparsity:.2%}")

    # 生成数据（H0为真）
    Y = generator.generate_var_series(T, N, p, Phi, Sigma)

    # 执行稀疏VAR的LR检验
    print("\n执行稀疏VAR的LR检验...")
    try:
        sparse_lr = SparseLRTest(estimator_type='lasso')
        lr_result = sparse_lr.compute_lr_at_point(Y, p, test_point)
        print(f"LR统计量: {lr_result['lr_statistic']:.4f}")

        # Bootstrap推断
        print("\n执行稀疏VAR的Bootstrap推断 (B=50)...")
        sparse_bootstrap = SparseBootstrapInference(B=50, estimator_type='lasso')
        boot_result = sparse_bootstrap.test(Y, p, test_point, alpha=0.05)

        print(f"Bootstrap p值: {boot_result['p_value']:.4f}")
        print(f"决策: {boot_result['decision']}")
        return boot_result
    except ImportError as e:
        print(f"跳过稀疏VAR检验: {e}")
        return None


def run_monte_carlo_simulation():
    """运行蒙特卡洛仿真并生成报告"""
    print("\n" + "=" * 60)
    print("蒙特卡洛仿真 - 评估检验性能")
    print("=" * 60)

    # 参数设置
    N = 2
    T = 100
    p = 1
    M = 30   # 蒙特卡洛次数
    B = 50   # Bootstrap次数
    test_point = 50

    generator = VARDataGenerator(seed=42)
    Phi = generator.generate_stationary_phi(N, p, scale=0.3)
    Sigma = np.eye(N) * 0.5

    print(f"\n参数设置: N={N}, T={T}, p={p}, M={M}, B={B}, 检验点t={test_point}")

    results = {}

    # 1. 正常VAR模型 - 第一类错误
    print("\n" + "-" * 40)
    print("1. 正常VAR模型 - 第一类错误评估 (H0为真)")
    print("-" * 40)
    mc = MonteCarloSimulation(M=M, B=B, seed=42)
    type1_result = mc.evaluate_type1_error_at_point(N, T, p, Phi, Sigma, test_point,
                                                     alpha=0.05, verbose=True)
    results['var_type1'] = type1_result
    print(f"\n第一类错误率: {type1_result['type1_error']:.4f}")
    print(f"名义显著性水平: {type1_result['nominal_alpha']}")
    print(f"Size失真: {type1_result['size_distortion']:.4f}")

    # 2. 正常VAR模型 - 功效
    print("\n" + "-" * 40)
    print("2. 正常VAR模型 - 功效评估 (H1为真)")
    print("-" * 40)
    Phi2 = Phi + 0.2 * np.ones_like(Phi)
    while not generator.check_stationarity(Phi2):
        Phi2 = Phi2 * 0.9

    power_result = mc.evaluate_power_at_point(N, T, p, Phi, Phi2, Sigma,
                                               break_point=test_point, t=test_point,
                                               alpha=0.05, verbose=True)
    results['var_power'] = power_result
    print(f"\n统计功效: {power_result['power']:.4f}")

    return results


def generate_report(results: dict):
    """生成分析报告"""
    print("\n" + "=" * 60)
    print("仿真分析报告")
    print("=" * 60)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-" * 40)
    print("一、正常VAR模型检验结果")
    print("-" * 40)

    if 'var_type1' in results:
        r = results['var_type1']
        print(f"  第一类错误率: {r['type1_error']:.4f} (名义水平: {r['nominal_alpha']})")
        print(f"  Size失真: {r['size_distortion']:.4f}")
        print(f"  有效迭代次数: {r['M_effective']}/{r['M']}")

    if 'var_power' in results:
        r = results['var_power']
        print(f"  统计功效: {r['power']:.4f}")
        print(f"  有效迭代次数: {r['M_effective']}/{r['M']}")

    print("\n" + "-" * 40)
    print("二、结论")
    print("-" * 40)

    if 'var_type1' in results:
        type1 = results['var_type1']['type1_error']
        if abs(type1 - 0.05) < 0.03:
            print("  - 第一类错误率接近名义水平，检验Size控制良好")
        elif type1 > 0.05:
            print("  - 第一类错误率偏高，检验可能过于激进")
        else:
            print("  - 第一类错误率偏低，检验可能过于保守")

    if 'var_power' in results:
        power = results['var_power']['power']
        if power > 0.8:
            print("  - 统计功效较高，检验能有效检测结构变化")
        elif power > 0.5:
            print("  - 统计功效中等，可考虑增加样本量")
        else:
            print("  - 统计功效较低，需要更大样本或更强的结构变化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VAR模型结构性变化检验'
    )
    parser.add_argument('--demo', type=str, default='all',
                        choices=['all', 'lr', 'break', 'sparse', 'mc', 'report'],
                        help='选择演示内容')

    args = parser.parse_args()

    print("=" * 60)
    print("VAR模型结构性变化检验")
    print("Structural Change Testing in VAR Models")
    print("=" * 60)

    if args.demo == 'all' or args.demo == 'lr':
        demo_lr_at_point()

    if args.demo == 'all' or args.demo == 'break':
        demo_structural_break_at_point()

    if args.demo == 'all' or args.demo == 'sparse':
        demo_sparse_var_test()

    if args.demo == 'all' or args.demo == 'mc' or args.demo == 'report':
        results = run_monte_carlo_simulation()
        generate_report(results)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
