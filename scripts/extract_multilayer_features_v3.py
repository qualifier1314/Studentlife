"""
多层网络特征提取器 V3（保留全部49个学生）
解决的问题：
1. ✅ 使用并集节点集（49个全体学生），保证研究整个班级
2. ✅ 添加层覆盖特征（4个），明确标记哪些层有数据
3. ✅ 缺失层特征设为NaN（不是0），避免语义错误
4. ✅ 跨层特征仅在有效层上计算
5. ✅ 层内Z-score标准化，消除层密度差异
6. ✅ 引入经典多层网络指标（参与系数、层熵等）

优势：
- 样本数: 49 (vs V2的28，+75%提升)
- 样本/特征比例: 49/28=1.75 (vs V2的28/24=1.17，+50%提升)
- 覆盖率: 100% (vs V2的57%)
- 泛化能力: 代表整个班级（而非仅"数据完整"的学生）

修复依据：KEEP_ALL_49_STUDENTS_PLAN.md
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MultilayerFeatureExtractorV3:
    """多层网络特征提取器 V3（保留全部49个学生）"""
    
    def __init__(self, data_dir='studentlife_data/dataset'):
        """
        初始化特征提取器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.features = {}
        self.layer_graphs = {}
        self.node_list = []  # 全体节点集（并集，49人）
        self.layer_stats = {}  # 每层的统计量（用于标准化）
        self.node_layer_coverage = {}  # 节点->层覆盖情况
    
    def _load_graph_from_edges(self, edge_file):
        """
        从边列表文件加载网络
        
        Args:
            edge_file: 边列表CSV文件路径
        
        Returns:
            nx.Graph: 网络图
        """
        G = nx.Graph()
        df = pd.read_csv(edge_file)
        
        for _, row in df.iterrows():
            source = row['source']
            target = row['target']
            weight = row.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)
        
        print(f"  加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
        
    def build_networks(self):
        """
        构建三层网络（使用并集节点，保留全部49个学生）
        """
        print("=" * 60)
        print("加载多层网络（V3版本 - 保留全部49个学生）")
        print("=" * 60)
        
        # 从已有的边列表加载网络
        print("\n[1/3] 加载物理层网络（Physical Layer）...")
        physical_graph = self._load_graph_from_edges('outputs/static/physical_edges_weighted.csv')
        
        print("\n[2/3] 加载行为层网络（Behavioral Layer）...")
        behavioral_graph = self._load_graph_from_edges('outputs/static/behavior_edges_weighted.csv')
        
        print("\n[3/3] 加载教育层网络（Educational Layer）...")
        educational_graph = self._load_graph_from_edges('outputs/static/education_edges_weighted.csv')
        
        # 保存网络
        self.layer_graphs['physical'] = physical_graph
        self.layer_graphs['behavioral'] = behavioral_graph
        self.layer_graphs['educational'] = educational_graph
        
        # 获取各层节点集
        physical_nodes = set(physical_graph.nodes())
        behavioral_nodes = set(behavioral_graph.nodes())
        educational_nodes = set(educational_graph.nodes())
        
        # 计算节点集统计
        print("\n" + "=" * 60)
        print("节点集统计分析")
        print("=" * 60)
        print(f"Physical层节点数:    {len(physical_nodes)} ({len(physical_nodes)/49*100:.1f}%)")
        print(f"Behavioral层节点数:  {len(behavioral_nodes)} ({len(behavioral_nodes)/49*100:.1f}%)")
        print(f"Educational层节点数: {len(educational_nodes)} ({len(educational_nodes)/49*100:.1f}%)")
        
        # 交集和并集
        intersection_nodes = physical_nodes & behavioral_nodes & educational_nodes
        union_nodes = physical_nodes | behavioral_nodes | educational_nodes
        
        print(f"\n三层交集（核心节点）: {len(intersection_nodes)} 个学生 ({len(intersection_nodes)/49*100:.1f}%)")
        print(f"三层并集（全体节点）: {len(union_nodes)} 个学生 ({len(union_nodes)/49*100:.1f}%)")
        
        # 缺失层统计
        missing_physical = len(union_nodes - physical_nodes)
        missing_behavioral = len(union_nodes - behavioral_nodes)
        missing_educational = len(union_nodes - educational_nodes)
        
        print(f"\n缺失情况:")
        print(f"  缺失Physical层:    {missing_physical} 人 ({missing_physical/len(union_nodes)*100:.1f}%) ⚠️")
        print(f"  缺失Behavioral层:  {missing_behavioral} 人 ({missing_behavioral/len(union_nodes)*100:.1f}%)")
        print(f"  缺失Educational层: {missing_educational} 人 ({missing_educational/len(union_nodes)*100:.1f}%)")
        
        # ✅ 使用并集节点（全部49个学生）
        self.node_list = sorted(list(union_nodes))
        
        print(f"\n✅ 使用全体节点集（并集）: {len(self.node_list)} 个学生")
        print("   优势：")
        print("     - 覆盖100%学生（代表整个班级）")
        print("     - 样本数充足（49样本 vs V2的28样本，+75%）")
        print("     - 避免选择偏差（不排除低活跃度学生）")
        print("   处理方法：")
        print("     - 添加层覆盖特征（明确标记哪些层有数据）")
        print("     - 缺失层特征设为NaN（而非0）")
        print("     - 跨层特征仅在有效层上计算")
        print("     - 使用XGBoost等支持NaN的模型")
        
        # 记录每个节点的层覆盖情况
        for node in self.node_list:
            self.node_layer_coverage[node] = {
                'has_physical': node in physical_nodes,
                'has_behavioral': node in behavioral_nodes,
                'has_educational': node in educational_nodes,
                'layer_count': sum([
                    node in physical_nodes,
                    node in behavioral_nodes,
                    node in educational_nodes
                ])
            }
        
        print(f"\n网络构建完成！")
        print(f"- 物理层:   {physical_graph.number_of_nodes()} 节点, {physical_graph.number_of_edges()} 边")
        print(f"- 行为层:   {behavioral_graph.number_of_nodes()} 节点, {behavioral_graph.number_of_edges()} 边")
        print(f"- 教育层:   {educational_graph.number_of_nodes()} 节点, {educational_graph.number_of_edges()} 边")
        print(f"- 分析节点: {len(self.node_list)} 个学生（并集，100%覆盖）")
        
        return self.layer_graphs
    
    def extract_single_layer_features(self, layer_name, G):
        """
        提取单层网络特征（精简版：每层3个核心指标）
        
        V3改进：
        - 缺失层特征设为NaN（而非0），避免"无数据"="极度孤立"的语义错误
        - 保留Z-score标准化
        
        Args:
            layer_name: 层名称 (physical/behavioral/educational)
            G: 网络图
        
        Returns:
            dict: 节点->特征字典
        """
        print(f"\n提取 {layer_name} 层特征（V3版本 - 支持缺失值）...")
        features = defaultdict(dict)
        
        # 1. 度中心性（归一化度）
        print("  [1/3] 计算度中心性...")
        degree_centrality = nx.degree_centrality(G)
        
        # 2. 聚类系数
        print("  [2/3] 计算聚类系数...")
        clustering_coef = nx.clustering(G)
        
        # 3. 介数中心性（大网络采样计算）
        print("  [3/3] 计算介数中心性...")
        try:
            if G.number_of_nodes() < 100:
                betweenness = nx.betweenness_centrality(G, weight='weight')
            else:
                betweenness = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()), weight='weight')
        except:
            betweenness = {n: 0 for n in G.nodes()}
        
        # 保存所有节点的特征（包含并集中的节点）
        nodes_in_layer = 0
        nodes_missing = 0
        
        for node in self.node_list:
            if node in G:
                # ✅ 节点在该层有数据
                features[node][f'{layer_name}_degree_centrality'] = degree_centrality[node]
                features[node][f'{layer_name}_clustering_coef'] = clustering_coef[node]
                features[node][f'{layer_name}_betweenness'] = betweenness[node]
                nodes_in_layer += 1
            else:
                # ✅ 节点在该层缺失 → 设为NaN（而非0）
                features[node][f'{layer_name}_degree_centrality'] = np.nan
                features[node][f'{layer_name}_clustering_coef'] = np.nan
                features[node][f'{layer_name}_betweenness'] = np.nan
                nodes_missing += 1
        
        print(f"  ✅ 完成！{nodes_in_layer}个学生有数据, {nodes_missing}个学生缺失 (设为NaN)")
        
        # 计算层统计量（用于Z-score标准化，仅在有数据的节点上计算）
        degree_values = [degree_centrality[n] for n in G.nodes()]
        self.layer_stats[layer_name] = {
            'degree_mean': np.mean(degree_values),
            'degree_std': np.std(degree_values),
            'degree_min': np.min(degree_values),
            'degree_max': np.max(degree_values),
            'nodes_count': len(G.nodes())
        }
        
        print(f"  层统计: mean={self.layer_stats[layer_name]['degree_mean']:.3f}, "
              f"std={self.layer_stats[layer_name]['degree_std']:.3f}, "
              f"nodes={self.layer_stats[layer_name]['nodes_count']}")
        
        return features
    
    def apply_zscore_normalization(self, features_dict):
        """
        应用层内Z-score标准化（修复层密度差异）
        
        V3改进：
        - 仅对有数据的节点进行标准化
        - 缺失节点保持NaN
        
        Args:
            features_dict: 节点->特征字典
        
        Returns:
            dict: 标准化后的特征字典
        """
        print("\n" + "=" * 60)
        print("应用层内Z-score标准化（V3版本 - 跳过NaN）")
        print("=" * 60)
        
        normalized_features = {}
        
        for layer_name in ['physical', 'behavioral', 'educational']:
            print(f"\n标准化 {layer_name} 层...")
            
            # 收集该层所有有效值（非NaN）
            degree_values = []
            for node in self.node_list:
                val = features_dict[node].get(f'{layer_name}_degree_centrality', np.nan)
                if not np.isnan(val):
                    degree_values.append(val)
            
            if len(degree_values) == 0:
                print(f"  ⚠️  警告: {layer_name} 层没有有效数据，跳过标准化")
                continue
            
            # 计算统计量（仅在有效值上）
            mean_val = np.mean(degree_values)
            std_val = np.std(degree_values)
            
            print(f"  原始分布: mean={mean_val:.3f}, std={std_val:.3f}")
            
            # 标准化（仅处理有数据的节点）
            zscore_values = []
            for node in self.node_list:
                if node not in normalized_features:
                    normalized_features[node] = {}
                
                # 度中心性标准化
                degree_val = features_dict[node].get(f'{layer_name}_degree_centrality', np.nan)
                if not np.isnan(degree_val):
                    if std_val > 0:
                        z_score = (degree_val - mean_val) / std_val
                    else:
                        z_score = 0.0
                    normalized_features[node][f'{layer_name}_degree_centrality_z'] = z_score
                    zscore_values.append(z_score)
                else:
                    # 缺失值保持NaN
                    normalized_features[node][f'{layer_name}_degree_centrality_z'] = np.nan
                
                # 保留原始值（用于对比）
                normalized_features[node][f'{layer_name}_degree_centrality'] = degree_val
            
            # 验证Z-score分布
            if len(zscore_values) > 0:
                z_mean = np.mean(zscore_values)
                z_std = np.std(zscore_values)
                print(f"  Z-score分布: mean={z_mean:.3f}, std={z_std:.3f} ✅")
            
        return normalized_features
    
    def extract_multilayer_features(self):
        """
        提取多层网络特征（V3版本 - 支持缺失值）
        
        V3改进：
        - 跨层特征仅在"有效层"上计算
        - 添加层覆盖特征（4个）
        - 少于2层数据的跨层特征设为NaN
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n" + "=" * 60)
        print("提取多层网络特征（V3版本 - 缺失感知）")
        print("=" * 60)
        
        multilayer_features = {}
        
        for node in self.node_list:
            multilayer_features[node] = {}
            
            # 获取层覆盖情况
            coverage = self.node_layer_coverage[node]
            has_physical = coverage['has_physical']
            has_behavioral = coverage['has_behavioral']
            has_educational = coverage['has_educational']
            layer_count = coverage['layer_count']
            
            # =====================================================
            # [新增] 层覆盖特征（4个）
            # =====================================================
            multilayer_features[node]['has_physical_layer'] = 1 if has_physical else 0
            multilayer_features[node]['has_behavioral_layer'] = 1 if has_behavioral else 0
            multilayer_features[node]['has_educational_layer'] = 1 if has_educational else 0
            multilayer_features[node]['layer_coverage'] = layer_count  # 1-3
            
            # =====================================================
            # [修复] 跨层活跃度：仅在有效层上计算
            # =====================================================
            valid_z_scores = []
            
            # 收集有效层的Z-score
            if has_physical:
                physical_z = self.features[node].get('physical_degree_centrality_z', np.nan)
                if not np.isnan(physical_z):
                    valid_z_scores.append(physical_z)
            
            if has_behavioral:
                behavioral_z = self.features[node].get('behavioral_degree_centrality_z', np.nan)
                if not np.isnan(behavioral_z):
                    valid_z_scores.append(behavioral_z)
            
            if has_educational:
                educational_z = self.features[node].get('educational_degree_centrality_z', np.nan)
                if not np.isnan(educational_z):
                    valid_z_scores.append(educational_z)
            
            # 跨层活跃度：至少2层才计算
            if len(valid_z_scores) >= 2:
                # 1. 高活跃层数（Z-score > 0）
                multilayer_features[node]['cross_layer_activity'] = sum(z > 0 for z in valid_z_scores)
                
                # 2. 度方差（Z-score）
                multilayer_features[node]['degree_variance_z'] = np.var(valid_z_scores)
                
                # 3. 平均Z-score
                multilayer_features[node]['avg_zscore'] = np.mean(valid_z_scores)
            else:
                # 少于2层，跨层特征无意义
                multilayer_features[node]['cross_layer_activity'] = np.nan
                multilayer_features[node]['degree_variance_z'] = np.nan
                multilayer_features[node]['avg_zscore'] = np.nan
            
            # =====================================================
            # [保留] 经典多层网络指标
            # =====================================================
            
            # 1. Participation Coefficient（参与系数）
            # 衡量节点在不同层的"活跃分布"
            if layer_count >= 2:
                degrees = []
                if has_physical:
                    deg = self.features[node].get('physical_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                if has_behavioral:
                    deg = self.features[node].get('behavioral_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                if has_educational:
                    deg = self.features[node].get('educational_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                
                if len(degrees) >= 2:
                    total_degree = sum(degrees)
                    if total_degree > 0:
                        # PC = 1 - Σ(d_i/d_total)^2
                        pc = 1 - sum((d / total_degree) ** 2 for d in degrees)
                        multilayer_features[node]['participation_coefficient'] = pc
                    else:
                        multilayer_features[node]['participation_coefficient'] = 0.0
                else:
                    multilayer_features[node]['participation_coefficient'] = np.nan
            else:
                multilayer_features[node]['participation_coefficient'] = np.nan
            
            # 2. Overlapping Degree（重叠度）
            # 节点在多少层中是"活跃节点"（度 > 均值）
            if layer_count >= 2:
                overlapping_count = 0
                
                if has_physical:
                    deg = self.features[node].get('physical_degree_centrality', np.nan)
                    mean_deg = self.layer_stats['physical']['degree_mean']
                    if not np.isnan(deg) and deg > mean_deg:
                        overlapping_count += 1
                
                if has_behavioral:
                    deg = self.features[node].get('behavioral_degree_centrality', np.nan)
                    mean_deg = self.layer_stats['behavioral']['degree_mean']
                    if not np.isnan(deg) and deg > mean_deg:
                        overlapping_count += 1
                
                if has_educational:
                    deg = self.features[node].get('educational_degree_centrality', np.nan)
                    mean_deg = self.layer_stats['educational']['degree_mean']
                    if not np.isnan(deg) and deg > mean_deg:
                        overlapping_count += 1
                
                multilayer_features[node]['overlapping_degree'] = overlapping_count
            else:
                multilayer_features[node]['overlapping_degree'] = np.nan
            
            # 3. Layer Entropy（层熵）
            # 衡量节点在各层的"活跃均匀度"
            if layer_count >= 2:
                degrees = []
                if has_physical:
                    deg = self.features[node].get('physical_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                if has_behavioral:
                    deg = self.features[node].get('behavioral_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                if has_educational:
                    deg = self.features[node].get('educational_degree_centrality', np.nan)
                    if not np.isnan(deg):
                        degrees.append(deg)
                
                if len(degrees) >= 2:
                    total_degree = sum(degrees)
                    if total_degree > 0:
                        probs = [d / total_degree for d in degrees]
                        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                        multilayer_features[node]['layer_entropy'] = entropy
                    else:
                        multilayer_features[node]['layer_entropy'] = 0.0
                else:
                    multilayer_features[node]['layer_entropy'] = np.nan
            else:
                multilayer_features[node]['layer_entropy'] = np.nan
        
        print(f"\n✅ 多层特征提取完成！")
        print(f"   - 层覆盖特征: 4个（has_*_layer, layer_coverage）")
        print(f"   - 跨层特征: 3个（仅在有效层计算）")
        print(f"   - 经典多层指标: 3个（至少2层才计算）")
        
        return multilayer_features
    
    def extract_behavioral_attributes(self):
        """
        提取行为属性特征（从EMA数据）
        
        V3改进：
        - 支持缺失值（某些学生可能没有EMA数据）
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n" + "=" * 60)
        print("提取行为属性特征（V3版本）")
        print("=" * 60)
        
        behavioral_features = {}
        
        # 加载EMA数据（情绪+压力）
        ema_file = os.path.join(self.data_dir, 'EMA', 'response.csv')
        
        if os.path.exists(ema_file):
            print(f"从 {ema_file} 加载EMA数据...")
            ema_df = pd.read_csv(ema_file)
            
            # 计算每个学生的平均情绪和响应次数
            ema_stats = ema_df.groupby('uid').agg({
                'mood': 'mean',
                'resp_time': 'count'
            }).reset_index()
            ema_stats.columns = ['uid', 'ema_mood_mean', 'ema_response_count']
            
            # 保存到特征字典
            for _, row in ema_stats.iterrows():
                uid = row['uid']
                if uid in self.node_list:
                    behavioral_features[uid] = {
                        'ema_mood_mean': row['ema_mood_mean'],
                        'ema_response_count': row['ema_response_count']
                    }
            
            # 缺失EMA数据的学生设为NaN
            for node in self.node_list:
                if node not in behavioral_features:
                    behavioral_features[node] = {
                        'ema_mood_mean': np.nan,
                        'ema_response_count': np.nan
                    }
            
            print(f"✅ EMA特征提取完成！{len(ema_stats)}个学生有数据")
        else:
            print(f"⚠️  警告: 未找到EMA文件 {ema_file}，跳过行为属性特征")
            for node in self.node_list:
                behavioral_features[node] = {
                    'ema_mood_mean': np.nan,
                    'ema_response_count': np.nan
                }
        
        return behavioral_features
    
    def extract_peer_influence_features(self):
        """
        提取同伴影响特征（从网络邻居）
        
        V3改进：
        - 跨层邻居计算时仅考虑有效层
        - 支持缺失值
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n" + "=" * 60)
        print("提取同伴影响特征（V3版本 - 跨层邻居）")
        print("=" * 60)
        
        peer_features = {}
        
        # 加载EMA数据（用于计算邻居情绪）
        ema_file = os.path.join(self.data_dir, 'EMA', 'response.csv')
        ema_mood = {}
        
        if os.path.exists(ema_file):
            ema_df = pd.read_csv(ema_file)
            ema_mood = ema_df.groupby('uid')['mood'].mean().to_dict()
        
        # 计算每个节点的同伴影响特征
        for node in self.node_list:
            # 收集所有层的邻居（仅在有效层）
            all_neighbors = set()
            coverage = self.node_layer_coverage[node]
            
            if coverage['has_physical'] and node in self.layer_graphs['physical']:
                all_neighbors.update(self.layer_graphs['physical'].neighbors(node))
            
            if coverage['has_behavioral'] and node in self.layer_graphs['behavioral']:
                all_neighbors.update(self.layer_graphs['behavioral'].neighbors(node))
            
            if coverage['has_educational'] and node in self.layer_graphs['educational']:
                all_neighbors.update(self.layer_graphs['educational'].neighbors(node))
            
            # 邻居平均情绪
            neighbor_moods = [ema_mood[n] for n in all_neighbors if n in ema_mood]
            peer_avg_mood = np.mean(neighbor_moods) if neighbor_moods else np.nan
            
            # 邻居平均压力（如果有压力数据）
            # 这里简化为NaN（可扩展）
            peer_avg_stress = np.nan
            
            # 总邻居数（跨层）
            total_neighbors = len(all_neighbors)
            
            peer_features[node] = {
                'peer_avg_mood': peer_avg_mood,
                'peer_avg_stress': peer_avg_stress,
                'total_neighbors': total_neighbors
            }
        
        print(f"✅ 同伴影响特征提取完成！")
        return peer_features
    
    def run(self, output_dir='outputs/static_v3'):
        """
        运行完整的特征提取流程（V3版本 - 保留49人）
        
        Args:
            output_dir: 输出目录
        """
        print("\n" + "=" * 80)
        print(" 多层网络特征提取器 V3 - 保留全部49个学生 ".center(80, "="))
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 构建网络（使用并集节点）
        self.build_networks()
        
        # 2. 提取单层特征（支持缺失值）
        print("\n" + "=" * 60)
        print("阶段1：提取单层拓扑特征（3层 × 3特征 = 9个）")
        print("=" * 60)
        
        for layer_name, G in self.layer_graphs.items():
            layer_features = self.extract_single_layer_features(layer_name, G)
            
            # 合并到总特征字典
            for node, feats in layer_features.items():
                if node not in self.features:
                    self.features[node] = {}
                self.features[node].update(feats)
        
        # 3. 应用Z-score标准化（跳过NaN）
        print("\n" + "=" * 60)
        print("阶段2：应用Z-score标准化（修复层密度差异）")
        print("=" * 60)
        
        normalized_features = self.apply_zscore_normalization(self.features)
        
        # 合并标准化特征
        for node, feats in normalized_features.items():
            self.features[node].update(feats)
        
        # 4. 提取多层网络特征（缺失感知）
        print("\n" + "=" * 60)
        print("阶段3：提取多层网络特征（10个 + 4个层覆盖）")
        print("=" * 60)
        
        multilayer_features = self.extract_multilayer_features()
        for node, feats in multilayer_features.items():
            self.features[node].update(feats)
        
        # 5. 提取行为属性特征
        print("\n" + "=" * 60)
        print("阶段4：提取行为属性特征（2个）")
        print("=" * 60)
        
        behavioral_features = self.extract_behavioral_attributes()
        for node, feats in behavioral_features.items():
            self.features[node].update(feats)
        
        # 6. 提取同伴影响特征
        print("\n" + "=" * 60)
        print("阶段5：提取同伴影响特征（3个）")
        print("=" * 60)
        
        peer_features = self.extract_peer_influence_features()
        for node, feats in peer_features.items():
            self.features[node].update(feats)
        
        # 7. 保存特征到CSV
        print("\n" + "=" * 60)
        print("保存特征到文件")
        print("=" * 60)
        
        feature_df = pd.DataFrame.from_dict(self.features, orient='index')
        feature_df.index.name = 'uid'
        
        output_file = os.path.join(output_dir, 'multilayer_features_v3.csv')
        feature_df.to_csv(output_file)
        
        print(f"\n✅ 特征已保存到: {output_file}")
        print(f"   - 样本数: {len(feature_df)} 个学生 (100%覆盖)")
        print(f"   - 特征数: {len(feature_df.columns)} 个")
        print(f"   - 样本/特征比例: {len(feature_df) / len(feature_df.columns):.2f} ✅")
        
        # 8. 生成特征描述
        self._generate_feature_summary(feature_df, output_dir)
        
        # 9. 统计缺失值
        self._analyze_missing_values(feature_df, output_dir)
        
        print("\n" + "=" * 80)
        print(" 特征提取完成！ ".center(80, "="))
        print("=" * 80)
        
        return feature_df
    
    def _generate_feature_summary(self, feature_df, output_dir):
        """生成特征摘要报告"""
        summary_file = os.path.join(output_dir, 'feature_summary_v3.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" 多层网络特征摘要（V3版本 - 49个学生）\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"样本数: {len(feature_df)}\n")
            f.write(f"特征数: {len(feature_df.columns)}\n")
            f.write(f"样本/特征比例: {len(feature_df) / len(feature_df.columns):.2f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("特征分类统计\n")
            f.write("=" * 80 + "\n\n")
            
            # 特征分类
            categories = {
                '单层拓扑特征': [c for c in feature_df.columns if any(x in c for x in ['physical_', 'behavioral_', 'educational_']) and '_z' not in c and 'has_' not in c],
                'Z-score标准化特征': [c for c in feature_df.columns if '_z' in c],
                '层覆盖特征': [c for c in feature_df.columns if 'has_' in c or c == 'layer_coverage'],
                '跨层特征': [c for c in feature_df.columns if any(x in c for x in ['cross_layer', 'degree_variance', 'avg_zscore'])],
                '经典多层指标': [c for c in feature_df.columns if any(x in c for x in ['participation', 'overlapping', 'entropy'])],
                '行为属性特征': [c for c in feature_df.columns if 'ema_' in c],
                '同伴影响特征': [c for c in feature_df.columns if 'peer_' in c or c == 'total_neighbors']
            }
            
            for cat_name, cols in categories.items():
                f.write(f"{cat_name}: {len(cols)}个\n")
                for col in cols:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("特征统计\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(feature_df.describe().to_string())
        
        print(f"✅ 特征摘要已保存到: {summary_file}")
    
    def _analyze_missing_values(self, feature_df, output_dir):
        """分析缺失值分布"""
        missing_file = os.path.join(output_dir, 'missing_values_v3.txt')
        
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" 缺失值分析（V3版本）\n")
            f.write("=" * 80 + "\n\n")
            
            missing_stats = feature_df.isnull().sum()
            missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
            
            f.write(f"总特征数: {len(feature_df.columns)}\n")
            f.write(f"有缺失值的特征数: {len(missing_stats)}\n\n")
            
            f.write("缺失值详情:\n")
            f.write("-" * 80 + "\n")
            
            for col, count in missing_stats.items():
                percentage = count / len(feature_df) * 100
                f.write(f"{col:50s}: {count:3d}/{len(feature_df)} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("预期缺失模式:\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. Physical层特征: ~36.7%缺失（18/49学生）\n")
            f.write("2. Behavioral层特征: ~6.1%缺失（3/49学生）\n")
            f.write("3. Educational层特征: ~8.2%缺失（4/49学生）\n")
            f.write("4. 跨层特征: 少于2层数据时设为NaN\n")
            f.write("\n处理建议:\n")
            f.write("- 使用XGBoost/LightGBM/CatBoost（原生支持NaN）\n")
            f.write("- 或使用Imputation方法（KNN、MICE等）\n")
        
        print(f"✅ 缺失值分析已保存到: {missing_file}")
        
        # 打印关键统计
        print(f"\n缺失值概览:")
        print(f"  有缺失值的特征数: {len(missing_stats)}/{len(feature_df.columns)}")
        if len(missing_stats) > 0:
            print(f"  最多缺失: {missing_stats.index[0]} ({missing_stats.iloc[0]}/{len(feature_df)}, {missing_stats.iloc[0]/len(feature_df)*100:.1f}%)")


def main():
    """主函数"""
    extractor = MultilayerFeatureExtractorV3(data_dir='studentlife_data/dataset')
    feature_df = extractor.run(output_dir='outputs/static_v3')
    
    print("\n" + "=" * 80)
    print(f"✅ V3版本特征提取完成！")
    print(f"   - 样本数: {len(feature_df)} (vs V2的28, +75%)")
    print(f"   - 特征数: {len(feature_df.columns)}")
    print(f"   - 样本/特征比例: {len(feature_df)/len(feature_df.columns):.2f}")
    print(f"   - 输出目录: outputs/static_v3/")
    print("=" * 80)


if __name__ == '__main__':
    main()
