"""层间耦合增强模块：基于节点相似度/属性的跨层连接构建。

传统 supra 邻接矩阵用 omega*I 对角连接同一节点在不同层的副本，
该模块提供更细粒度的层间耦合方式：
- 保留对角基础耦合（omega_diag）
- 增加基于跨层相似度的非对角耦合（omega_similarity * similarity(i,j,alpha,beta)）
- 支持多种相似度度量（共同邻居、度相关性、属性相似度等）

理论依据：
- De Domenico et al., "Structural reducibility of multilayer networks" (2013)
- Magnani et al., "Combinatorial analysis of multiple networks" (2013)
- 跨层耦合权重应反映节点在不同层上的"角色相似性"或"信息传递可能性"
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class InterLayerCoupling:
    """层间耦合构建器"""

    @staticmethod
    def diagonal_coupling(N: int, omega: float = 1.0) -> np.ndarray:
        """传统对角耦合：omega * I
        
        Args:
            N: 节点数
            omega: 耦合强度
            
        Returns:
            N×N 对角矩阵
        """
        return np.eye(N) * omega

    @staticmethod
    def similarity_coupling(
        G_alpha: nx.Graph,
        G_beta: nx.Graph,
        nodes: List[str],
        omega_diag: float = 1.0,
        omega_sim: float = 0.5,
        sim_method: str = 'common_neighbors',
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        normalize: bool = True,
        attr_matrix: Optional[np.ndarray] = None,
        attr_weight: float = 0.5,
    ) -> np.ndarray:
        """基于相似度的层间耦合矩阵
        
        Args:
            G_alpha, G_beta: 两层网络（节点集合一致）
            nodes: 统一节点列表（顺序）
            omega_diag: 对角基础耦合强度
            omega_sim: 相似度加权系数
            sim_method: 相似度计算方法
                - 'common_neighbors': 共同邻居数归一化
                - 'jaccard': Jaccard 系数
                - 'adamic_adar': Adamic-Adar 系数
                - 'degree_corr': 度相关性
                - 'resource_allocation': 资源分配指数
            threshold: 相似度阈值（低于该值的非对角项置零）
            top_k: 每个节点保留的最相似跨层邻居数（None 表示全保留）
            normalize: 是否对相似度归一化到 [0,1]
            
        Returns:
            N×N 耦合矩阵
        """
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}
        
        # 计算相似度矩阵
        S = np.zeros((N, N))
        
        if sim_method == 'common_neighbors':
            # 共同邻居数 / max(deg_alpha[i], deg_beta[j])
            neighbors_a = {n: set(G_alpha.neighbors(n)) for n in nodes}
            neighbors_b = {n: set(G_beta.neighbors(n)) for n in nodes}
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i == j:
                        continue
                    common = len(neighbors_a[u] & neighbors_b[v])
                    denom = max(len(neighbors_a[u]), len(neighbors_b[v]), 1)
                    S[i, j] = common / denom
        
        elif sim_method == 'jaccard':
            # Jaccard: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
            neighbors_a = {n: set(G_alpha.neighbors(n)) for n in nodes}
            neighbors_b = {n: set(G_beta.neighbors(n)) for n in nodes}
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i == j:
                        continue
                    inter = len(neighbors_a[u] & neighbors_b[v])
                    union = len(neighbors_a[u] | neighbors_b[v])
                    S[i, j] = inter / union if union > 0 else 0.0
        
        elif sim_method == 'degree_corr':
            # 度相关性：(deg_a[i] * deg_b[j]) / (max_deg_a * max_deg_b)
            deg_a = np.array([G_alpha.degree(n) for n in nodes], dtype=float)
            deg_b = np.array([G_beta.degree(n) for n in nodes], dtype=float)
            max_da = deg_a.max() if deg_a.max() > 0 else 1.0
            max_db = deg_b.max() if deg_b.max() > 0 else 1.0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    S[i, j] = (deg_a[i] * deg_b[j]) / (max_da * max_db)
        
        elif sim_method == 'adamic_adar':
            # Adamic-Adar：sum_{z in CN(i,j)} 1/log(deg(z))
            neighbors_a = {n: set(G_alpha.neighbors(n)) for n in nodes}
            neighbors_b = {n: set(G_beta.neighbors(n)) for n in nodes}
            deg_a = {n: G_alpha.degree(n) for n in nodes}
            deg_b = {n: G_beta.degree(n) for n in nodes}
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i == j:
                        continue
                    common = neighbors_a[u] & neighbors_b[v]
                    aa = sum(1.0 / np.log(max(deg_a[z], deg_b[z], 2)) for z in common)
                    S[i, j] = aa
        
        elif sim_method == 'resource_allocation':
            # Resource Allocation: sum_{z in CN(i,j)} 1/deg(z)
            neighbors_a = {n: set(G_alpha.neighbors(n)) for n in nodes}
            neighbors_b = {n: set(G_beta.neighbors(n)) for n in nodes}
            deg_a = {n: G_alpha.degree(n) for n in nodes}
            deg_b = {n: G_beta.degree(n) for n in nodes}
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i == j:
                        continue
                    common = neighbors_a[u] & neighbors_b[v]
                    ra = sum(1.0 / max(deg_a[z], deg_b[z], 1) for z in common)
                    S[i, j] = ra
        
        else:
            raise ValueError(f"Unknown similarity method: {sim_method}")
        
        # 归一化
        if normalize:
            S_max = S.max()
            if S_max > 0:
                S /= S_max

        # 如果提供了属性相似度矩阵，则与结构相似度线性组合（归一后）
        if attr_matrix is not None:
            try:
                A = np.array(attr_matrix, dtype=float)
                if A.shape != S.shape:
                    raise ValueError("attr_matrix shape must match S shape")
                # 归一化属性相似度到 [0,1]
                A_min, A_max = A.min(), A.max()
                if A_max - A_min > 0:
                    A = (A - A_min) / (A_max - A_min)
                else:
                    A = np.zeros_like(A)
                # 将两者线性组合
                S = (1.0 - attr_weight) * S + attr_weight * A
                # 重新归一化
                if normalize:
                    S_max2 = S.max()
                    if S_max2 > 0:
                        S /= S_max2
            except Exception:
                # 若属性矩阵无效则忽略
                pass
        
        # 阈值过滤与 Top-K
        if threshold > 0:
            S[S < threshold] = 0.0
        
        if top_k is not None:
            # 每行（每个节点）保留 Top-K
            for i in range(N):
                row = S[i, :]
                if row.sum() == 0:
                    continue
                indices = np.argsort(-row)[:top_k]
                mask = np.ones(N, dtype=bool)
                mask[indices] = False
                S[i, mask] = 0.0
        
        # 构建耦合矩阵：对角 omega_diag + 非对角 omega_sim * S
        C = np.eye(N) * omega_diag + S * omega_sim
        return C

    @staticmethod
    def build_supra_with_enhanced_coupling(
        layers: List[Tuple[str, nx.Graph]],
        nodes: List[str],
        omega_diag: float = 1.0,
        omega_sim: float = 0.3,
        sim_method: str = 'common_neighbors',
        threshold: float = 0.1,
        top_k: Optional[int] = 10,
        normalize: bool = True,
        attr_matrix: Optional[np.ndarray] = None,
        attr_weight: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
        """构建增强版 supra 邻接矩阵
        
        Args:
            layers: [(layer_name, G), ...] 按顺序
            nodes: 统一节点列表
            omega_diag: 对角耦合强度
            omega_sim: 相似度耦合强度
            sim_method: 相似度方法
            threshold: 相似度阈值
            top_k: 每节点保留的跨层邻居数
            normalize: 是否归一化相似度
            
        Returns:
            (supra_matrix, coupling_dict)
            - supra_matrix: (L*N) × (L*N) 矩阵
            - coupling_dict: {(layer_i, layer_j): C_ij} 耦合矩阵字典（调试用）
        """
        L = len(layers)
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}
        
        # 层内邻接矩阵
        A_layers = []
        for lname, G in layers:
            A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
            A_layers.append(A)
        
        # 层间耦合矩阵（L×L 块，每块 N×N）
        coupling_dict = {}
        blocks = []
        for i in range(L):
            row_blocks = []
            for j in range(L):
                if i == j:
                    # 对角块：层内邻接
                    row_blocks.append(A_layers[i])
                else:
                    # 非对角块：层间耦合
                    if (layers[i][0], layers[j][0]) not in coupling_dict:
                        C_ij = InterLayerCoupling.similarity_coupling(
                            layers[i][1],
                            layers[j][1],
                            nodes,
                            omega_diag=omega_diag,
                            omega_sim=omega_sim,
                            sim_method=sim_method,
                            threshold=threshold,
                            top_k=top_k,
                            normalize=normalize,
                            attr_matrix=attr_matrix,
                            attr_weight=attr_weight,
                        )
                        coupling_dict[(layers[i][0], layers[j][0])] = C_ij
                        coupling_dict[(layers[j][0], layers[i][0])] = C_ij.T
                    row_blocks.append(coupling_dict[(layers[i][0], layers[j][0])])
            blocks.append(row_blocks)
        
        # 拼接为 supra 矩阵
        supra = np.block(blocks)
        return supra, coupling_dict

    @staticmethod
    def export_inter_layer_edges(
        coupling_dict: Dict[Tuple[str, str], np.ndarray],
        nodes: List[str],
        filepath: str,
        weight_threshold: float = 0.0,
    ):
        """导出层间边列表
        
        Args:
            coupling_dict: {(layer_i, layer_j): C_ij}
            nodes: 节点列表
            filepath: 导出 CSV 路径
            weight_threshold: 权重阈值（低于该值不导出）
        """
        import pandas as pd
        rows = []
        for (la, lb), C in coupling_dict.items():
            if la >= lb:
                continue
            N = len(nodes)
            for i in range(N):
                for j in range(N):
                    w = float(C[i, j])
                    if w > weight_threshold:
                        rows.append({
                            'type': 'inter' if i != j else 'diag',
                            'src_layer': la,
                            'dst_layer': lb,
                            'u': nodes[i],
                            'v': nodes[j],
                            'weight': w,
                        })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
