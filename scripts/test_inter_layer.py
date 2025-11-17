"""测试增强层间耦合功能的简化示例"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
from src.inter_layer_coupling import InterLayerCoupling


def test_inter_layer_coupling():
    """测试层间耦合构建"""
    
    print("=" * 60)
    print("测试增强层间耦合功能")
    print("=" * 60)
    
    # 创建简单的测试网络
    print("\n[1] 创建测试网络...")
    nodes = ['u00', 'u01', 'u02', 'u03', 'u04']
    N = len(nodes)
    
    # Physical 层：u00-u01, u01-u02, u02-u03
    G_phy = nx.Graph()
    G_phy.add_nodes_from(nodes)
    G_phy.add_edge('u00', 'u01', weight=1.0)
    G_phy.add_edge('u01', 'u02', weight=1.0)
    G_phy.add_edge('u02', 'u03', weight=1.0)
    print(f"  Physical: {G_phy.number_of_edges()} edges")
    
    # Behavior 层：u01-u02, u02-u03, u03-u04
    G_beh = nx.Graph()
    G_beh.add_nodes_from(nodes)
    G_beh.add_edge('u01', 'u02', weight=1.0)
    G_beh.add_edge('u02', 'u03', weight=1.0)
    G_beh.add_edge('u03', 'u04', weight=1.0)
    print(f"  Behavior: {G_beh.number_of_edges()} edges")
    
    # Education 层：u00-u02, u01-u03, u02-u04
    G_edu = nx.Graph()
    G_edu.add_nodes_from(nodes)
    G_edu.add_edge('u00', 'u02', weight=1.0)
    G_edu.add_edge('u01', 'u03', weight=1.0)
    G_edu.add_edge('u02', 'u04', weight=1.0)
    print(f"  Education: {G_edu.number_of_edges()} edges")
    
    # 测试对角耦合
    print("\n[2] 测试传统对角耦合...")
    C_diag = InterLayerCoupling.diagonal_coupling(N, omega=1.0)
    print(f"  对角耦合矩阵形状: {C_diag.shape}")
    print(f"  非零元素数: {np.count_nonzero(C_diag)}")
    
    # 测试相似度耦合
    print("\n[3] 测试相似度耦合（共同邻居）...")
    C_sim = InterLayerCoupling.similarity_coupling(
        G_phy, G_beh, nodes,
        omega_diag=1.0,
        omega_sim=0.5,
        sim_method='common_neighbors',
        threshold=0.1,
        top_k=3,
        normalize=True,
    )
    print(f"  相似度耦合矩阵形状: {C_sim.shape}")
    print(f"  对角元素数: {np.count_nonzero(np.diag(C_sim))}")
    print(f"  非对角元素数: {np.count_nonzero(C_sim - np.diag(np.diag(C_sim)))}")
    
    # 显示耦合矩阵（前5x5）
    print("\n  Physical-Behavior 耦合矩阵:")
    print("  ", " ".join(f"{n:>6}" for n in nodes))
    for i, u in enumerate(nodes):
        vals = " ".join(f"{C_sim[i, j]:6.3f}" for j in range(N))
        print(f"  {u} {vals}")
    
    # 测试完整 supra 矩阵构建
    print("\n[4] 测试完整 supra 矩阵构建...")
    layers = [
        ('Physical', G_phy),
        ('Behavior', G_beh),
        ('Education', G_edu),
    ]
    
    supra, coupling_dict = InterLayerCoupling.build_supra_with_enhanced_coupling(
        layers=layers,
        nodes=nodes,
        omega_diag=1.0,
        omega_sim=0.3,
        sim_method='common_neighbors',
        threshold=0.1,
        top_k=2,
        normalize=True,
    )
    
    print(f"  Supra 矩阵维度: {supra.shape} (应为 {3*N} x {3*N})")
    print(f"  总边数（非零元素）: {np.count_nonzero(supra)}")
    
    # 统计层间边
    print("\n[5] 层间耦合统计:")
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        diag = np.count_nonzero(np.diag(C) > 0)
        off_diag = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
        print(f"  {la} <-> {lb}:")
        print(f"    对角边: {diag}")
        print(f"    非对角边: {off_diag}")
    
    # 对比传统方法
    print("\n[6] 对比传统对角连接方法:")
    supra_traditional = np.block([
        [nx.to_numpy_array(G_phy, nodelist=nodes), np.eye(N), np.eye(N)],
        [np.eye(N), nx.to_numpy_array(G_beh, nodelist=nodes), np.eye(N)],
        [np.eye(N), np.eye(N), nx.to_numpy_array(G_edu, nodelist=nodes)],
    ])
    print(f"  传统方法总边数: {np.count_nonzero(supra_traditional)}")
    print(f"  增强方法总边数: {np.count_nonzero(supra)}")
    print(f"  增加的边数: {np.count_nonzero(supra) - np.count_nonzero(supra_traditional)}")
    
    print("\n" + "=" * 60)
    print("测试完成！增强层间耦合功能正常工作。")
    print("=" * 60)


if __name__ == '__main__':
    test_inter_layer_coupling()
