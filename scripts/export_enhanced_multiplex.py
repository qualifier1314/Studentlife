"""导出增强层间耦合的多层网络

使用基于相似度的层间耦合替代传统 omega*I 对角连接。
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


def export_enhanced_multiplex(
    omega_diag: float = 1.0,
    omega_sim: float = 0.3,
    sim_method: str = 'common_neighbors',
    threshold: float = 0.1,
    top_k: int = 10,
    output_dir: str = None,
):
    """导出增强层间耦合的多层网络
    
    Args:
        omega_diag: 对角耦合强度（同一节点跨层）
        omega_sim: 相似度耦合强度（不同节点跨层）
        sim_method: 相似度方法
            - 'common_neighbors': 共同邻居归一化
            - 'jaccard': Jaccard 系数
            - 'degree_corr': 度相关性
            - 'adamic_adar': Adamic-Adar 系数
            - 'resource_allocation': 资源分配指数
        threshold: 相似度阈值
        top_k: 每节点保留的跨层邻居数
        output_dir: 输出目录（默认 outputs/static_enhanced）
    """
    # 确定输出目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'outputs', 'static_enhanced')
    NetworkBuilder.ensure_dir(output_dir)
    
    print(f"[Info] 输出目录: {output_dir}")
    print(f"[Info] 层间耦合参数: omega_diag={omega_diag}, omega_sim={omega_sim}, method={sim_method}, threshold={threshold}, top_k={top_k}")
    
    # 构建静态三层网络
    print("\n[Step 1] 构建静态三层网络...")
    static_builder = StaticLayerBuilder()
    
    try:
        phys_static = static_builder.build_physical_static(
            mode='sum', proximity_threshold=-70, min_duration=300, 
            time_threshold=30, normalize_sources=True
        )
        print(f"  Physical: {phys_static.number_of_nodes()} nodes, {phys_static.number_of_edges()} edges")
    except Exception as e:
        print(f"  Physical 构建失败: {e}")
        phys_static = nx.Graph()
    
    try:
        beh_static = static_builder.build_behavior_static(
            mode='sum', similarity_threshold=0.7, top_k=10, normalize_sources=True
        )
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
    
    # 统一节点集合与编号
    print("\n[Step 2] 统一节点集合与编号...")
    all_users = set(phys_static.nodes()) | set(beh_static.nodes()) | set(edu_static.nodes())
    uid_number_map = NetworkBuilder.build_uid_numbering(all_users)
    nodes = sorted(all_users, key=lambda x: uid_number_map.get(x, 10**9))
    N = len(nodes)
    print(f"  总节点数: {N}")
    
    # 确保所有图包含相同节点集（补齐孤立节点）
    for G in [phys_static, beh_static, edu_static]:
        G.add_nodes_from(nodes)
    
    # 构建增强 supra 矩阵
    print("\n[Step 3] 构建增强层间耦合的 supra 邻接矩阵...")
    layers = [
        ('Physical', phys_static),
        ('Behavior', beh_static),
        ('Education', edu_static),
    ]
    
    supra, coupling_dict = InterLayerCoupling.build_supra_with_enhanced_coupling(
        layers=layers,
        nodes=nodes,
        omega_diag=omega_diag,
        omega_sim=omega_sim,
        sim_method=sim_method,
        threshold=threshold,
        top_k=top_k,
        normalize=True,
    )
    
    print(f"  Supra 矩阵维度: {supra.shape}")
    
    # 统计层间边
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        diag_edges = np.count_nonzero(np.diag(C) > 0)
        off_diag_edges = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
        print(f"  {la} <-> {lb}: 对角边={diag_edges}, 非对角边={off_diag_edges}")
    
    # 导出 supra 邻接矩阵
    print("\n[Step 4] 导出 supra 邻接矩阵...")
    col_names = [f"{lname}_{uid_number_map.get(n, n)}" for lname, _ in layers for n in nodes]
    supra_df = pd.DataFrame(supra, columns=col_names, index=col_names)
    supra_path = os.path.join(output_dir, 'supra_adjacency_enhanced.csv')
    supra_df.to_csv(supra_path, encoding='utf-8', float_format='%.6f')
    print(f"  已导出: {supra_path}")
    
    # 导出 UID 编号映射
    map_path = os.path.join(output_dir, 'uid_number_map.csv')
    NetworkBuilder.export_uid_numbering(uid_number_map, map_path)
    print(f"  已导出: {map_path}")
    
    # 导出层内边
    print("\n[Step 5] 导出层内边...")
    for lname, G in layers:
        fname_uw = os.path.join(output_dir, f'{lname.lower()}_edges_unweighted.csv')
        fname_w = os.path.join(output_dir, f'{lname.lower()}_edges_weighted.csv')
        NetworkBuilder.export_edge_list(G, fname_uw, weighted=False)
        NetworkBuilder.export_edge_list(G, fname_w, weighted=True)
        print(f"  已导出: {fname_uw}, {fname_w}")
    
    # 导出层间边
    print("\n[Step 6] 导出层间边...")
    inter_edges_path = os.path.join(output_dir, 'inter_layer_edges_enhanced.csv')
    InterLayerCoupling.export_inter_layer_edges(
        coupling_dict=coupling_dict,
        nodes=nodes,
        filepath=inter_edges_path,
        weight_threshold=0.0,
    )
    print(f"  已导出: {inter_edges_path}")
    
    # 导出多层边（层内+层间合并）
    print("\n[Step 7] 导出多层边列表...")
    rows = []
    # 层内边
    for lname, G in layers:
        for u, v, d in G.edges(data=True):
            w = float(d.get('weight', 1.0))
            if w > 0:
                rows.append({
                    'type': 'intra',
                    'src_layer': lname,
                    'dst_layer': lname,
                    'u': u,
                    'v': v,
                    'weight': w,
                })
    # 层间边
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                w = float(C[i, j])
                if w > 0:
                    rows.append({
                        'type': 'inter',
                        'src_layer': la,
                        'dst_layer': lb,
                        'u': u,
                        'v': v,
                        'weight': w,
                    })
    multi_path = os.path.join(output_dir, 'multiplex_edges_enhanced.csv')
    pd.DataFrame(rows).to_csv(multi_path, index=False, encoding='utf-8')
    print(f"  已导出: {multi_path}, 记录数: {len(rows)}")
    
    # 生成统计报告
    print("\n[Step 8] 生成统计报告...")
    report_lines = [
        "# 增强层间耦合多层网络统计报告",
        "",
        "## 参数设置",
        f"- omega_diag (对角耦合): {omega_diag}",
        f"- omega_sim (相似度耦合): {omega_sim}",
        f"- sim_method (相似度方法): {sim_method}",
        f"- threshold (相似度阈值): {threshold}",
        f"- top_k (每节点保留跨层邻居数): {top_k}",
        "",
        "## 网络规模",
        f"- 总节点数: {N}",
        f"- 层数: {len(layers)}",
        f"- Supra 矩阵维度: {supra.shape[0]} × {supra.shape[1]}",
        "",
        "## 层内网络统计",
    ]
    for lname, G in layers:
        report_lines.extend([
            f"### {lname}",
            f"- 节点数: {G.number_of_nodes()}",
            f"- 边数: {G.number_of_edges()}",
            f"- 平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}" if G.number_of_nodes() > 0 else "- 平均度: N/A",
            "",
        ])
    
    report_lines.extend([
        "## 层间耦合统计",
    ])
    for (la, lb), C in coupling_dict.items():
        if la >= lb:
            continue
        diag_edges = np.count_nonzero(np.diag(C) > 0)
        off_diag_edges = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
        diag_weight_sum = np.diag(C).sum()
        off_diag_weight_sum = (C - np.diag(np.diag(C))).sum()
        report_lines.extend([
            f"### {la} <-> {lb}",
            f"- 对角边数: {diag_edges}",
            f"- 非对角边数: {off_diag_edges}",
            f"- 对角权重和: {diag_weight_sum:.2f}",
            f"- 非对角权重和: {off_diag_weight_sum:.2f}",
            "",
        ])
    
    report_lines.extend([
        "## 理论依据与解释",
        "",
        "### 传统方法（omega*I 对角连接）的局限性",
        "- 仅连接同一学生在不同层的副本",
        "- 忽略跨层的真实关联（如物理接近但选课不同的学生）",
        "- 无法捕捉不同层上角色相似的节点间的信息传递",
        "",
        "### 增强方法（相似度加权层间耦合）的优势",
        "- 保留对角基础耦合（omega_diag），确保同一节点的跨层连接",
        "- 增加非对角耦合（omega_sim * similarity），反映不同节点在不同层上的角色相似性",
        "- 支持多种相似度度量（共同邻居、Jaccard、度相关性、Adamic-Adar 等）",
        "- 可配置阈值和 Top-K，控制耦合密度与计算复杂度",
        "",
        "### 适用场景",
        "- 多层网络中心性分析：更准确地识别跨层影响力节点",
        "- 多层社团检测：发现跨层相似角色的社团结构",
        "- 链路预测：利用跨层相似度提升预测精度",
        "- 信息传播模拟：更真实地模拟跨层信息流动",
        "",
        "### 参考文献",
        "- De Domenico, M., et al. (2013). Structural reducibility of multilayer networks. Nature communications, 4(1), 1-8.",
        "- Magnani, M., et al. (2013). Combinatorial analysis of multiple networks. arXiv preprint arXiv:1303.4986.",
        "- Kivelä, M., et al. (2014). Multilayer networks. Journal of complex networks, 2(3), 203-271.",
    ])
    
    report_path = os.path.join(output_dir, 'report_enhanced.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  已导出: {report_path}")
    
    print("\n[完成] 增强层间耦合多层网络导出成功！")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    # 默认参数
    export_enhanced_multiplex(
        omega_diag=1.0,
        omega_sim=0.3,
        sim_method='common_neighbors',
        threshold=0.1,
        top_k=10,
    )
