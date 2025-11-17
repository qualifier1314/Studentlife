"""快速对比传统与增强层间耦合方法

此脚本同时运行两种方法并生成对比报告。
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import networkx as nx
from src.layers.combined import StaticLayerBuilder
from src.utils import NetworkBuilder
from src.inter_layer_coupling import InterLayerCoupling


def compare_methods():
    """对比传统与增强方法"""
    
    print("=" * 70)
    print("传统 vs 增强层间耦合方法对比")
    print("=" * 70)
    
    # 构建三层网络
    print("\n[1] 构建静态三层网络...")
    static_builder = StaticLayerBuilder()
    
    try:
        phys_static = static_builder.build_physical_static(mode='sum', normalize_sources=True)
        print(f"  Physical: {phys_static.number_of_nodes()} nodes, {phys_static.number_of_edges()} edges")
    except Exception as e:
        print(f"  Physical 构建失败: {e}")
        phys_static = nx.Graph()
    
    try:
        beh_static = static_builder.build_behavior_static(mode='sum', normalize_sources=True)
        print(f"  Behavior: {beh_static.number_of_nodes()} nodes, {beh_static.number_of_edges()} edges")
    except Exception as e:
        print(f"  Behavior 构建失败: {e}")
        beh_static = nx.Graph()
    
    try:
        edu_static = static_builder.build_education_static(network_type='co-enrollment')
        print(f"  Education: {edu_static.number_of_nodes()} nodes, {edu_static.number_of_edges()} edges")
    except Exception as e:
        print(f"  Education 构建失败: {e}")
        edu_static = nx.Graph()
    
    # 统一节点集合
    all_users = set(phys_static.nodes()) | set(beh_static.nodes()) | set(edu_static.nodes())
    nodes = sorted(all_users)
    N = len(nodes)
    print(f"\n  总节点数: {N}")
    
    # 确保所有图包含相同节点集
    for G in [phys_static, beh_static, edu_static]:
        G.add_nodes_from(nodes)
    
    # 获取邻接矩阵
    A_phy = nx.to_numpy_array(phys_static, nodelist=nodes, weight='weight')
    A_beh = nx.to_numpy_array(beh_static, nodelist=nodes, weight='weight')
    A_edu = nx.to_numpy_array(edu_static, nodelist=nodes, weight='weight')
    
    # 方法1：传统对角连接
    print("\n[2] 构建传统 supra 矩阵（omega × I）...")
    omega = 1.0
    I = np.eye(N)
    A_traditional = np.block([
        [A_phy, omega * I, omega * I],
        [omega * I, A_beh, omega * I],
        [omega * I, omega * I, A_edu],
    ])
    print(f"  Supra 维度: {A_traditional.shape}")
    print(f"  总边数（非零元素）: {np.count_nonzero(A_traditional)}")
    
    # 方法2：增强相似度耦合
    print("\n[3] 构建增强 supra 矩阵（相似度加权）...")
    layers = [
        ('Physical', phys_static),
        ('Behavior', beh_static),
        ('Education', edu_static),
    ]
    
    A_enhanced, coupling_dict = InterLayerCoupling.build_supra_with_enhanced_coupling(
        layers=layers,
        nodes=nodes,
        omega_diag=1.0,
        omega_sim=0.3,
        sim_method='common_neighbors',
        threshold=0.1,
        top_k=10,
        normalize=True,
    )
    print(f"  Supra 维度: {A_enhanced.shape}")
    print(f"  总边数（非零元素）: {np.count_nonzero(A_enhanced)}")
    
    # 统计层间边
    print("\n  层间耦合统计:")
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        diag = np.count_nonzero(np.diag(C) > 0)
        off_diag = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
        print(f"    {la} <-> {lb}:")
        print(f"      对角边: {diag}")
        print(f"      非对角边: {off_diag}")
    
    # 对比统计
    print("\n" + "=" * 70)
    print("对比统计")
    print("=" * 70)
    
    edges_trad = np.count_nonzero(A_traditional)
    edges_enh = np.count_nonzero(A_enhanced)
    increase_pct = (edges_enh / edges_trad - 1) * 100
    
    print(f"\n边数对比:")
    print(f"  传统方法: {edges_trad:,}")
    print(f"  增强方法: {edges_enh:,}")
    print(f"  增加数量: {edges_enh - edges_trad:,}")
    print(f"  增加比例: {increase_pct:.1f}%")
    
    # 密度对比
    total_possible = (3 * N) ** 2
    density_trad = edges_trad / total_possible
    density_enh = edges_enh / total_possible
    
    print(f"\n密度对比:")
    print(f"  传统方法: {density_trad:.4f}")
    print(f"  增强方法: {density_enh:.4f}")
    print(f"  密度提升: {(density_enh / density_trad - 1) * 100:.1f}%")
    
    # 度分布对比
    deg_trad = A_traditional.sum(axis=1)
    deg_enh = A_enhanced.sum(axis=1)
    
    print(f"\n度统计对比:")
    print(f"  传统方法 - 平均度: {deg_trad.mean():.2f}, 最大度: {deg_trad.max():.0f}")
    print(f"  增强方法 - 平均度: {deg_enh.mean():.2f}, 最大度: {deg_enh.max():.0f}")
    
    # 度相关性（Spearman）
    try:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(deg_trad, deg_enh)
        print(f"\n度排序相关性:")
        print(f"  Spearman 相关系数: {corr:.3f}")
        print(f"  P-value: {pval:.4e}")
    except ImportError:
        print("\n  (需安装 scipy 以计算度相关性)")
    
    # 导出对比报告
    print("\n[4] 导出对比报告...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'outputs', 'comparison')
    NetworkBuilder.ensure_dir(out_dir)
    
    # 保存对比统计
    comparison_stats = {
        'metric': [
            '总节点数', '总边数(传统)', '总边数(增强)', '边数增加',
            '增加比例(%)', '密度(传统)', '密度(增强)', '平均度(传统)', '平均度(增强)'
        ],
        'value': [
            N, edges_trad, edges_enh, edges_enh - edges_trad,
            increase_pct, density_trad, density_enh, deg_trad.mean(), deg_enh.mean()
        ]
    }
    
    stats_df = pd.DataFrame(comparison_stats)
    stats_path = os.path.join(out_dir, 'comparison_stats.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"  已导出: {stats_path}")
    
    # 保存度分布
    degree_df = pd.DataFrame({
        'node_index': range(3 * N),
        'degree_traditional': deg_trad,
        'degree_enhanced': deg_enh,
        'degree_increase': deg_enh - deg_trad,
    })
    degree_path = os.path.join(out_dir, 'degree_comparison.csv')
    degree_df.to_csv(degree_path, index=False, encoding='utf-8')
    print(f"  已导出: {degree_path}")
    
    # 生成 Markdown 报告
    report_lines = [
        "# 传统 vs 增强层间耦合方法对比报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 网络规模",
        f"- 总节点数: {N}",
        f"- 层数: 3",
        f"- Supra 矩阵维度: {3*N} × {3*N}",
        "",
        "## 边数对比",
        f"| 方法 | 边数 | 增加量 | 增加比例 |",
        f"|------|------|--------|----------|",
        f"| 传统 | {edges_trad:,} | - | - |",
        f"| 增强 | {edges_enh:,} | {edges_enh - edges_trad:,} | {increase_pct:.1f}% |",
        "",
        "## 密度对比",
        f"| 方法 | 密度 | 密度提升 |",
        f"|------|------|----------|",
        f"| 传统 | {density_trad:.4f} | - |",
        f"| 增强 | {density_enh:.4f} | {(density_enh / density_trad - 1) * 100:.1f}% |",
        "",
        "## 度统计对比",
        f"| 方法 | 平均度 | 最大度 | 标准差 |",
        f"|------|--------|--------|--------|",
        f"| 传统 | {deg_trad.mean():.2f} | {deg_trad.max():.0f} | {deg_trad.std():.2f} |",
        f"| 增强 | {deg_enh.mean():.2f} | {deg_enh.max():.0f} | {deg_enh.std():.2f} |",
        "",
        "## 层间耦合详情（增强方法）",
    ]
    
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        diag = np.count_nonzero(np.diag(C) > 0)
        off_diag = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
        report_lines.extend([
            f"### {la} ↔ {lb}",
            f"- 对角边（同一节点跨层）: {diag}",
            f"- 非对角边（不同节点跨层）: {off_diag}",
            f"- 非对角比例: {off_diag / (diag + off_diag) * 100:.1f}%",
            "",
        ])
    
    report_lines.extend([
        "## 结论与建议",
        "",
        f"1. **边数增长**: 增强方法比传统方法多 {increase_pct:.1f}% 的边",
        "   - 如果增长过大（>100%），建议提高阈值或降低 top_k",
        "   - 如果增长过小（<20%），可降低阈值或提高 omega_sim",
        "",
        f"2. **密度变化**: 密度提升 {(density_enh / density_trad - 1) * 100:.1f}%",
        "   - 密度适中时（0.01-0.1）适合多层分析",
        "   - 密度过高时考虑更严格的过滤条件",
        "",
        "3. **应用建议**:",
        "   - 多层中心性分析：优先使用增强方法",
        "   - 个体轨迹追踪：可使用传统方法",
        "   - 社团检测：增强方法能发现更多跨层相似角色",
        "",
        "4. **计算成本**:",
        f"   - 节点数 {N}：增强方法可接受",
        "   - 如需更快速度，降低 top_k 或提高 threshold",
    ])
    
    report_path = os.path.join(out_dir, 'comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  已导出: {report_path}")
    
    print("\n" + "=" * 70)
    print("对比完成！")
    print(f"输出目录: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    compare_methods()
