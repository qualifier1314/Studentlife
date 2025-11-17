"""
多层网络特征提取器
提取学生在多层社交网络中的结构特征、行为特征和时序特征
用于风险预测和干预策略设计

特征类别：
1. 单层网络特征（每层独立计算）
2. 多层网络特征（跨层综合）
3. 行为属性特征（从原始数据提取）
4. 时序特征（网络和行为的变化率）
5. 同伴影响特征（邻居的平均特征）
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

from src.layers.physical import SensingNetwork, DiningNetwork
from src.layers.behavior import AppUsageNetwork, CalendarNetwork, RawAccFeatNetwork
from src.layers.education import EducationNetwork
from src.layers.combined import StaticLayerBuilder


class MultilayerFeatureExtractor:
    """多层网络特征提取器"""
    
    def __init__(self, data_dir='studentlife_data/dataset'):
        """
        初始化特征提取器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.features = {}
        self.layer_graphs = {}
        self.node_list = []
    
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
        """构建三层网络（从已有的边列表文件加载）"""
        print("=" * 60)
        print("加载多层网络...")
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
        
        # 获取所有节点的并集
        all_nodes = set()
        for G in self.layer_graphs.values():
            all_nodes.update(G.nodes())
        self.node_list = sorted(list(all_nodes))
        
        print(f"\n网络构建完成！")
        print(f"- 物理层: {physical_graph.number_of_nodes()} 节点, {physical_graph.number_of_edges()} 边")
        print(f"- 行为层: {behavioral_graph.number_of_nodes()} 节点, {behavioral_graph.number_of_edges()} 边")
        print(f"- 教育层: {educational_graph.number_of_nodes()} 节点, {educational_graph.number_of_edges()} 边")
        print(f"- 总节点数: {len(self.node_list)}")
        
        return self.layer_graphs
    
    def extract_single_layer_features(self, layer_name, G):
        """
        提取单层网络特征
        
        Args:
            layer_name: 层名称 (physical/behavioral/educational)
            G: 网络图
        
        Returns:
            dict: 节点->特征字典
        """
        print(f"\n提取 {layer_name} 层特征...")
        features = defaultdict(dict)
        
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        
        # 聚类系数
        clustering_coef = nx.clustering(G)
        
        # 介数中心性（大网络计算慢，采样计算）
        try:
            if G.number_of_nodes() < 100:
                betweenness = nx.betweenness_centrality(G, weight='weight')
            else:
                # 采样计算
                betweenness = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()), weight='weight')
        except:
            betweenness = {n: 0 for n in G.nodes()}
        
        # 接近中心性（只在连通分量上计算）
        closeness = {}
        for component in nx.connected_components(G):
            subG = G.subgraph(component)
            if len(component) > 1:
                close = nx.closeness_centrality(subG, distance='weight')
                closeness.update(close)
            else:
                closeness[list(component)[0]] = 0.0
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, weight='weight', max_iter=100)
        except:
            pagerank = {n: 0 for n in G.nodes()}
        
        # 加权度（考虑边权重）
        weighted_degree = dict(G.degree(weight='weight'))
        
        # 保存特征
        for node in self.node_list:
            prefix = f"{layer_name}_"
            features[node][f"{prefix}degree_centrality"] = degree_centrality.get(node, 0.0)
            features[node][f"{prefix}clustering_coef"] = clustering_coef.get(node, 0.0)
            features[node][f"{prefix}betweenness"] = betweenness.get(node, 0.0)
            features[node][f"{prefix}closeness"] = closeness.get(node, 0.0)
            features[node][f"{prefix}pagerank"] = pagerank.get(node, 0.0)
            features[node][f"{prefix}weighted_degree"] = weighted_degree.get(node, 0.0) if node in G else 0.0
            features[node][f"{prefix}degree"] = G.degree(node) if node in G else 0
        
        return features
    
    def extract_multilayer_features(self):
        """
        提取多层网络特征（跨层综合）
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n提取多层网络特征（跨层综合）...")
        features = defaultdict(dict)
        
        # 获取每层的度中心性
        layer_degrees = {}
        for layer_name, G in self.layer_graphs.items():
            layer_degrees[layer_name] = nx.degree_centrality(G)
        
        for node in self.node_list:
            # 1. 跨层活跃度（在多少层中度>平均值）
            active_layers = 0
            degree_values = []
            for layer_name, degrees in layer_degrees.items():
                deg = degrees.get(node, 0.0)
                degree_values.append(deg)
                if deg > 0:
                    avg_deg = np.mean(list(degrees.values()))
                    if deg > avg_deg:
                        active_layers += 1
            
            features[node]['cross_layer_activity'] = active_layers
            
            # 2. 层间度差异（最大度 - 最小度）
            if degree_values:
                features[node]['degree_variance'] = np.max(degree_values) - np.min(degree_values)
                features[node]['degree_std'] = np.std(degree_values)
                features[node]['degree_mean'] = np.mean(degree_values)
            else:
                features[node]['degree_variance'] = 0.0
                features[node]['degree_std'] = 0.0
                features[node]['degree_mean'] = 0.0
            
            # 3. 层间一致性（度排名的Spearman相关性）
            # Physical vs Educational
            phys_deg = layer_degrees['physical'].get(node, 0.0)
            edu_deg = layer_degrees['educational'].get(node, 0.0)
            features[node]['physical_educational_consistency'] = phys_deg * edu_deg  # 简化版
            
            # 4. 邻居重叠度（不同层的邻居交集）
            phys_neighbors = set(self.layer_graphs['physical'].neighbors(node)) if node in self.layer_graphs['physical'] else set()
            edu_neighbors = set(self.layer_graphs['educational'].neighbors(node)) if node in self.layer_graphs['educational'] else set()
            beh_neighbors = set(self.layer_graphs['behavioral'].neighbors(node)) if node in self.layer_graphs['behavioral'] else set()
            
            # Jaccard相似度
            if len(phys_neighbors | edu_neighbors) > 0:
                features[node]['phys_edu_neighbor_overlap'] = len(phys_neighbors & edu_neighbors) / len(phys_neighbors | edu_neighbors)
            else:
                features[node]['phys_edu_neighbor_overlap'] = 0.0
            
            if len(phys_neighbors | beh_neighbors) > 0:
                features[node]['phys_beh_neighbor_overlap'] = len(phys_neighbors & beh_neighbors) / len(phys_neighbors | beh_neighbors)
            else:
                features[node]['phys_beh_neighbor_overlap'] = 0.0
            
            # 5. 总邻居数（所有层的并集）
            all_neighbors = phys_neighbors | edu_neighbors | beh_neighbors
            features[node]['total_unique_neighbors'] = len(all_neighbors)
        
        return features
    
    def extract_behavioral_attributes(self):
        """
        提取行为属性特征（从原始数据）
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n提取行为属性特征...")
        features = defaultdict(dict)
        
        # 初始化所有节点为0
        for node in self.node_list:
            features[node]['call_count'] = 0
            features[node]['call_duration'] = 0.0
            features[node]['sms_count'] = 0
            features[node]['app_usage_hours'] = 0.0
            features[node]['calendar_events'] = 0
            features[node]['physical_activity'] = 0.0
        
        # 1. 通话特征
        try:
            call_log_path = os.path.join(self.data_dir, 'call_log')
            if os.path.exists(call_log_path):
                for file_name in os.listdir(call_log_path):
                    if file_name.endswith('.csv'):
                        uid = file_name.replace('call_log_', '').replace('.csv', '')
                        if uid in self.node_list:
                            csv_file = os.path.join(call_log_path, file_name)
                            df = pd.read_csv(csv_file)
                            if not df.empty:
                                features[uid]['call_count'] = len(df)
                                # CALLS_duration列是通话时长（秒）
                                if 'CALLS_duration' in df.columns:
                                    duration = pd.to_numeric(df['CALLS_duration'], errors='coerce').fillna(0)
                                    features[uid]['call_duration'] = duration.sum() / 60.0  # 转为分钟
                print(f"  ✓ 通话特征: 提取了 {sum(1 for f in features.values() if f['call_count'] > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取通话特征失败: {e}")
        
        # 2. 短信特征
        try:
            sms_path = os.path.join(self.data_dir, 'sms')
            if os.path.exists(sms_path):
                for file_name in os.listdir(sms_path):
                    if file_name.endswith('.csv'):
                        uid = file_name.replace('sms_', '').replace('.csv', '')
                        if uid in self.node_list:
                            csv_file = os.path.join(sms_path, file_name)
                            df = pd.read_csv(csv_file)
                            features[uid]['sms_count'] = len(df)
                print(f"  ✓ 短信特征: 提取了 {sum(1 for f in features.values() if f['sms_count'] > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取短信特征失败: {e}")
        
        # 3. App使用特征（注意：app_usage目录结构可能不同）
        try:
            app_usage_path = os.path.join(self.data_dir, 'app_usage')
            if os.path.exists(app_usage_path):
                # app_usage可能有子目录
                for item in os.listdir(app_usage_path):
                    item_path = os.path.join(app_usage_path, item)
                    if os.path.isdir(item_path):
                        # 子目录模式
                        for file_name in os.listdir(item_path):
                            if file_name.endswith('.csv'):
                                uid = item
                                if uid in self.node_list:
                                    csv_file = os.path.join(item_path, file_name)
                                    df = pd.read_csv(csv_file)
                                    if not df.empty:
                                        # App使用记录数作为活跃度指标
                                        features[uid]['app_usage_hours'] = len(df) / 60.0
                print(f"  ✓ App使用特征: 提取了 {sum(1 for f in features.values() if f['app_usage_hours'] > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取App使用特征失败: {e}")
        
        # 4. 日历事件
        try:
            calendar_path = os.path.join(self.data_dir, 'calendar')
            if os.path.exists(calendar_path):
                for item in os.listdir(calendar_path):
                    item_path = os.path.join(calendar_path, item)
                    if os.path.isdir(item_path):
                        # 子目录模式
                        for file_name in os.listdir(item_path):
                            if file_name.endswith('.csv'):
                                uid = item
                                if uid in self.node_list:
                                    csv_file = os.path.join(item_path, file_name)
                                    df = pd.read_csv(csv_file)
                                    features[uid]['calendar_events'] = len(df)
                print(f"  ✓ 日历特征: 提取了 {sum(1 for f in features.values() if f['calendar_events'] > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取日历特征失败: {e}")
        
        # 5. EMA特征（Mood和Stress，JSON格式）
        import json
        try:
            # Mood数据
            mood_path = os.path.join(self.data_dir, 'EMA', 'response', 'Mood')
            if os.path.exists(mood_path):
                for file_name in os.listdir(mood_path):
                    if file_name.endswith('.json'):
                        uid = file_name.replace('Mood_', '').replace('.json', '')
                        if uid in self.node_list:
                            json_file = os.path.join(mood_path, file_name)
                            with open(json_file, 'r') as f:
                                mood_data = json.load(f)
                            
                            if mood_data and len(mood_data) > 0:
                                # 提取happy分数（1-5分）
                                happy_scores = []
                                for entry in mood_data:
                                    if 'happy' in entry:
                                        try:
                                            happy_scores.append(int(entry['happy']))
                                        except:
                                            pass
                                
                                if happy_scores:
                                    features[uid]['ema_mood_mean'] = np.mean(happy_scores)
                                    features[uid]['ema_mood_std'] = np.std(happy_scores)
                                else:
                                    features[uid]['ema_mood_mean'] = 0.0
                                    features[uid]['ema_mood_std'] = 0.0
                                
                                features[uid]['ema_response_count'] = len(mood_data)
                print(f"  ✓ EMA Mood特征: 提取了 {sum(1 for f in features.values() if f.get('ema_mood_mean', 0) > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取EMA Mood特征失败: {e}")
        
        try:
            # Stress数据
            stress_path = os.path.join(self.data_dir, 'EMA', 'response', 'Stress')
            if os.path.exists(stress_path):
                for file_name in os.listdir(stress_path):
                    if file_name.endswith('.json'):
                        uid = file_name.replace('Stress_', '').replace('.json', '')
                        if uid in self.node_list:
                            json_file = os.path.join(stress_path, file_name)
                            with open(json_file, 'r') as f:
                                stress_data = json.load(f)
                            
                            if stress_data and len(stress_data) > 0:
                                stress_scores = []
                                for entry in stress_data:
                                    if 'level' in entry:
                                        try:
                                            stress_scores.append(int(entry['level']))
                                        except:
                                            pass
                                
                                if stress_scores:
                                    features[uid]['ema_stress_mean'] = np.mean(stress_scores)
                                    features[uid]['ema_stress_std'] = np.std(stress_scores)
                                else:
                                    features[uid]['ema_stress_mean'] = 0.0
                                    features[uid]['ema_stress_std'] = 0.0
                print(f"  ✓ EMA Stress特征: 提取了 {sum(1 for f in features.values() if f.get('ema_stress_mean', 0) > 0)} 个学生的数据")
        except Exception as e:
            print(f"  ⚠️  提取EMA Stress特征失败: {e}")
        
        # 确保所有节点都有EMA特征（即使是0）
        for node in self.node_list:
            if 'ema_mood_mean' not in features[node]:
                features[node]['ema_mood_mean'] = 0.0
                features[node]['ema_mood_std'] = 0.0
            if 'ema_stress_mean' not in features[node]:
                features[node]['ema_stress_mean'] = 0.0
                features[node]['ema_stress_std'] = 0.0
            if 'ema_response_count' not in features[node]:
                features[node]['ema_response_count'] = 0
        
        return features
    
    def extract_peer_influence_features(self, behavioral_features):
        """
        提取同伴影响特征（邻居的平均特征）
        
        Args:
            behavioral_features: 行为特征字典
        
        Returns:
            dict: 节点->特征字典
        """
        print("\n提取同伴影响特征...")
        features = defaultdict(dict)
        
        # 对每层计算邻居的平均行为特征
        for layer_name, G in self.layer_graphs.items():
            for node in self.node_list:
                if node not in G:
                    features[node][f'{layer_name}_neighbor_avg_ema_mood'] = 0.0
                    features[node][f'{layer_name}_neighbor_avg_ema_stress'] = 0.0
                    features[node][f'{layer_name}_neighbor_count'] = 0
                    continue
                
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    features[node][f'{layer_name}_neighbor_avg_ema_mood'] = 0.0
                    features[node][f'{layer_name}_neighbor_avg_ema_stress'] = 0.0
                    features[node][f'{layer_name}_neighbor_count'] = 0
                else:
                    # 邻居的平均EMA情绪
                    neighbor_moods = [behavioral_features[n].get('ema_mood_mean', 0.0) for n in neighbors]
                    features[node][f'{layer_name}_neighbor_avg_ema_mood'] = np.mean(neighbor_moods)
                    
                    # 邻居的平均EMA压力
                    neighbor_stress = [behavioral_features[n].get('ema_stress_mean', 0.0) for n in neighbors]
                    features[node][f'{layer_name}_neighbor_avg_ema_stress'] = np.mean(neighbor_stress)
                    
                    # 邻居数量
                    features[node][f'{layer_name}_neighbor_count'] = len(neighbors)
        
        return features
    
    def combine_all_features(self):
        """
        组合所有特征
        
        Returns:
            pd.DataFrame: 特征矩阵
        """
        print("\n" + "=" * 60)
        print("开始特征提取流程...")
        print("=" * 60)
        
        all_features = defaultdict(dict)
        
        # 1. 单层网络特征
        for layer_name, G in self.layer_graphs.items():
            layer_features = self.extract_single_layer_features(layer_name, G)
            for node in self.node_list:
                all_features[node].update(layer_features[node])
        
        # 2. 多层网络特征
        multilayer_features = self.extract_multilayer_features()
        for node in self.node_list:
            all_features[node].update(multilayer_features[node])
        
        # 3. 行为属性特征
        behavioral_features = self.extract_behavioral_attributes()
        for node in self.node_list:
            all_features[node].update(behavioral_features[node])
        
        # 4. 同伴影响特征
        peer_features = self.extract_peer_influence_features(behavioral_features)
        for node in self.node_list:
            all_features[node].update(peer_features[node])
        
        # 转换为DataFrame
        df = pd.DataFrame.from_dict(all_features, orient='index')
        df.index.name = 'uid'
        
        # 填充缺失值
        df = df.fillna(0.0)
        
        print("\n" + "=" * 60)
        print("特征提取完成！")
        print("=" * 60)
        print(f"- 节点数: {len(df)}")
        print(f"- 特征数: {len(df.columns)}")
        print(f"\n特征类别统计:")
        print(f"  - Physical层特征: {len([c for c in df.columns if c.startswith('physical_')])}")
        print(f"  - Behavioral层特征: {len([c for c in df.columns if c.startswith('behavioral_')])}")
        print(f"  - Educational层特征: {len([c for c in df.columns if c.startswith('educational_')])}")
        print(f"  - 多层网络特征: {len([c for c in df.columns if 'cross_layer' in c or 'overlap' in c or 'variance' in c])}")
        print(f"  - 行为属性特征: {len([c for c in df.columns if any(x in c for x in ['call', 'sms', 'app', 'calendar', 'ema'])])}")
        print(f"  - 同伴影响特征: {len([c for c in df.columns if 'neighbor' in c])}")
        
        return df


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("多层网络特征提取器")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = 'outputs/static'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化提取器
    extractor = MultilayerFeatureExtractor()
    
    # 构建网络
    extractor.build_networks()
    
    # 提取所有特征
    features_df = extractor.combine_all_features()
    
    # 保存特征矩阵
    output_file = os.path.join(output_dir, 'multilayer_features.csv')
    features_df.to_csv(output_file)
    print(f"\n✓ 特征矩阵已保存到: {output_file}")
    
    # 显示前几行
    print("\n特征矩阵预览（前5行，前10列）:")
    print(features_df.iloc[:5, :10])
    
    # 保存特征描述
    desc_file = os.path.join(output_dir, 'feature_descriptions.txt')
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("多层网络特征说明\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【单层网络特征】\n")
        f.write("每层独立计算，包括:\n")
        f.write("- degree_centrality: 度中心性\n")
        f.write("- clustering_coef: 聚类系数\n")
        f.write("- betweenness: 介数中心性\n")
        f.write("- closeness: 接近中心性\n")
        f.write("- pagerank: PageRank值\n")
        f.write("- weighted_degree: 加权度\n")
        f.write("- degree: 度数\n\n")
        
        f.write("【多层网络特征】\n")
        f.write("跨层综合特征:\n")
        f.write("- cross_layer_activity: 跨层活跃度（在多少层中度>平均值）\n")
        f.write("- degree_variance: 层间度差异（最大度-最小度）\n")
        f.write("- degree_std: 层间度标准差\n")
        f.write("- degree_mean: 层间度均值\n")
        f.write("- physical_educational_consistency: 物理层与教育层一致性\n")
        f.write("- phys_edu_neighbor_overlap: 物理层与教育层邻居重叠度\n")
        f.write("- phys_beh_neighbor_overlap: 物理层与行为层邻居重叠度\n")
        f.write("- total_unique_neighbors: 总邻居数（所有层并集）\n\n")
        
        f.write("【行为属性特征】\n")
        f.write("从原始数据提取:\n")
        f.write("- call_count: 通话次数\n")
        f.write("- call_duration: 通话时长（分钟）\n")
        f.write("- sms_count: 短信数量\n")
        f.write("- app_usage_hours: App使用时长（小时）\n")
        f.write("- calendar_events: 日历事件数\n")
        f.write("- physical_activity: 物理活动量\n")
        f.write("- ema_mood_mean: EMA情绪均值\n")
        f.write("- ema_mood_std: EMA情绪标准差\n")
        f.write("- ema_stress_mean: EMA压力均值\n")
        f.write("- ema_stress_std: EMA压力标准差\n")
        f.write("- ema_response_count: EMA响应次数\n\n")
        
        f.write("【同伴影响特征】\n")
        f.write("邻居的平均特征:\n")
        f.write("- {layer}_neighbor_avg_ema_mood: 邻居平均情绪\n")
        f.write("- {layer}_neighbor_avg_ema_stress: 邻居平均压力\n")
        f.write("- {layer}_neighbor_count: 邻居数量\n\n")
        
        f.write(f"\n总特征数: {len(features_df.columns)}\n")
        f.write(f"总节点数: {len(features_df)}\n")
    
    print(f"\n✓ 特征说明已保存到: {desc_file}")
    
    # 生成特征统计报告
    print("\n" + "=" * 60)
    print("特征统计摘要")
    print("=" * 60)
    print(features_df.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\n" + "=" * 60)
    print("特征提取完成！可以开始预测任务了。")
    print("=" * 60)


if __name__ == '__main__':
    main()
