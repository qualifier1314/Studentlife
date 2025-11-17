"""
预测标签生成器
从时序数据生成三类预测目标：
1. 学业风险（学习行为下降）
2. 心理健康风险（EMA持续低落）
3. 社交健康风险（网络位置边缘化）
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionLabelGenerator:
    """预测标签生成器"""
    
    def __init__(self, data_dir='studentlife_data/dataset'):
        self.data_dir = data_dir
        self.labels = {}
        self.node_list = []
    
    def load_node_list(self, features_file='outputs/static/multilayer_features.csv'):
        """加载节点列表"""
        df = pd.read_csv(features_file)
        self.node_list = df['uid'].tolist()
        print(f"加载节点列表: {len(self.node_list)} 个学生")
        return self.node_list
    
    def generate_academic_risk_labels(self):
        """
        生成学业风险标签
        定义：基于通信行为（通话+短信）和网络位置的综合评估
        
        标签定义：
        - 1（高风险）: 通信活跃度低 + 教育层度低（学习参与少）
        - 0（低风险）: 正常参与
        """
        print("\n生成学业风险标签...")
        print("-" * 60)
        
        labels = {}
        features_df = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
        
        # 计算通信活跃度阈值（使用20%分位数，更宽松）
        call_threshold = features_df['call_count'].quantile(0.20)
        sms_threshold = features_df['sms_count'].quantile(0.20)
        edu_degree_threshold = features_df['educational_degree_centrality'].quantile(0.25)
        
        for uid in self.node_list:
            labels[uid] = 0
            
            if uid in features_df.index:
                call_count = features_df.loc[uid, 'call_count']
                sms_count = features_df.loc[uid, 'sms_count']
                edu_degree = features_df.loc[uid, 'educational_degree_centrality']
                behavioral_degree = features_df.loc[uid, 'behavioral_degree_centrality']
                
                # 高风险条件（更宽松）：
                # 1. 通信活跃度低（通话或短信低于20%分位数）
                # 2. 教育层度低（低于25%分位数）或行为层度很低
                if (call_count < call_threshold or sms_count < sms_threshold) and \
                   (edu_degree < edu_degree_threshold or behavioral_degree < 0.15):
                    labels[uid] = 1
        
        self.labels['academic_risk'] = labels
        
        # 统计
        risk_count = sum(labels.values())
        risk_rate = risk_count / len(labels) * 100
        print(f"  ✓ 基于通信行为+网络位置生成标签")
        print(f"  ✓ 阈值: Call<{call_threshold:.0f}, SMS<{sms_threshold:.0f}, EduDegree<{edu_degree_threshold:.3f}")
        print(f"  ✓ 生成完成:")
        print(f"    - 总学生数: {len(labels)}")
        print(f"    - 高风险学生: {risk_count} ({risk_rate:.1f}%)")
        print(f"    - 低风险学生: {len(labels) - risk_count} ({100 - risk_rate:.1f}%)")
        
        if risk_count > 0:
            print(f"\n  高风险学生特征:")
            for uid in self.node_list:
                if labels[uid] == 1 and uid in features_df.index:
                    call = features_df.loc[uid, 'call_count']
                    sms = features_df.loc[uid, 'sms_count']
                    edu = features_df.loc[uid, 'educational_degree_centrality']
                    behav = features_df.loc[uid, 'behavioral_degree_centrality']
                    print(f"    - {uid}: Call={call:.0f}, SMS={sms:.0f}, EduDegree={edu:.3f}, BehavDegree={behav:.3f}")
        
        return labels
    
    def generate_mental_health_risk_labels(self):
        """
        生成心理健康风险标签
        定义：基于EMA情绪/压力分数
        
        标签定义：
        - 1（高风险）: EMA平均分数低于阈值或高压力
        - 0（低风险）: EMA分数正常
        """
        print("\n生成心理健康风险标签...")
        print("-" * 60)
        
        labels = {}
        
        # 从特征矩阵读取EMA数据
        features_df = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
        
        for uid in self.node_list:
            labels[uid] = 0
            
            if uid in features_df.index:
                ema_mood = features_df.loc[uid, 'ema_mood_mean']
                ema_stress = features_df.loc[uid, 'ema_stress_mean']
                
                # 高风险条件：
                # 1. Mood分数很低（< 1.0，在1-5分范围内）
                # 2. Stress分数很高（> 2.5，在1-5分范围内）
                # 3. 两者都不好
                if (ema_mood > 0 and ema_mood < 1.0) or \
                   (ema_stress > 0 and ema_stress > 2.8):
                    labels[uid] = 1
                # 或者：mood低且stress高
                elif (ema_mood > 0 and ema_mood < 1.5) and \
                     (ema_stress > 0 and ema_stress > 2.5):
                    labels[uid] = 1
        
        self.labels['mental_health_risk'] = labels
        
        # 统计
        risk_count = sum(labels.values())
        risk_rate = risk_count / len(labels) * 100
        
        # 显示高风险学生的EMA分数
        print(f"  ✓ 基于真实EMA数据生成标签")
        print(f"  ✓ 生成完成:")
        print(f"    - 总学生数: {len(labels)}")
        print(f"    - 高风险学生: {risk_count} ({risk_rate:.1f}%)")
        print(f"    - 低风险学生: {len(labels) - risk_count} ({100 - risk_rate:.1f}%)")
        
        if risk_count > 0:
            print(f"\n  高风险学生EMA分数:")
            for uid in self.node_list:
                if labels[uid] == 1 and uid in features_df.index:
                    mood = features_df.loc[uid, 'ema_mood_mean']
                    stress = features_df.loc[uid, 'ema_stress_mean']
                    print(f"    - {uid}: Mood={mood:.2f}, Stress={stress:.2f}")
        
        return labels
    
    def generate_social_health_risk_labels(self):
        """
        生成社交健康风险标签
        定义：基于网络位置（度中心性）
        
        标签定义：
        - 1（高风险）: 多层度中心性都很低（社交边缘化）
        - 0（低风险）: 至少在一层中心度较高
        """
        print("\n生成社交健康风险标签...")
        print("-" * 60)
        
        labels = {}
        features_df = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
        
        # 使用更宽松的阈值（20%分位数）
        phys_threshold = features_df['physical_degree_centrality'].quantile(0.20)
        behav_threshold = features_df['behavioral_degree_centrality'].quantile(0.20)
        edu_threshold = features_df['educational_degree_centrality'].quantile(0.25)
        
        for uid in self.node_list:
            if uid in features_df.index:
                physical_degree = features_df.loc[uid, 'physical_degree_centrality']
                behavioral_degree = features_df.loc[uid, 'behavioral_degree_centrality']
                educational_degree = features_df.loc[uid, 'educational_degree_centrality']
                cross_layer_activity = features_df.loc[uid, 'cross_layer_activity']
                
                # 高风险条件（更宽松）：
                # 至少两层度都很低，或者跨层活跃度<=1
                low_count = sum([
                    physical_degree < phys_threshold,
                    behavioral_degree < behav_threshold,
                    educational_degree < edu_threshold
                ])
                
                if low_count >= 2 or cross_layer_activity <= 1:
                    labels[uid] = 1
                else:
                    labels[uid] = 0
            else:
                labels[uid] = 0
        
        self.labels['social_health_risk'] = labels
        
        # 统计
        risk_count = sum(labels.values())
        risk_rate = risk_count / len(labels) * 100
        print(f"  ✓ 基于多层网络位置生成标签")
        print(f"  ✓ 阈值: Phys<{phys_threshold:.3f}, Behav<{behav_threshold:.3f}, Edu<{edu_threshold:.3f}")
        print(f"  ✓ 生成完成:")
        print(f"    - 总学生数: {len(labels)}")
        print(f"    - 高风险学生: {risk_count} ({risk_rate:.1f}%)")
        print(f"    - 低风险学生: {len(labels) - risk_count} ({100 - risk_rate:.1f}%)")
        
        if risk_count > 0 and risk_count <= 15:
            print(f"\n  高风险学生网络位置:")
            for uid in self.node_list:
                if labels[uid] == 1 and uid in features_df.index:
                    phys = features_df.loc[uid, 'physical_degree_centrality']
                    behav = features_df.loc[uid, 'behavioral_degree_centrality']
                    edu = features_df.loc[uid, 'educational_degree_centrality']
                    cross = features_df.loc[uid, 'cross_layer_activity']
                    print(f"    - {uid}: Phys={phys:.3f}, Behav={behav:.3f}, Edu={edu:.3f}, CrossLayer={cross:.0f}")
        
        return labels
    
    def save_labels(self, output_file='outputs/static/prediction_labels.csv'):
        """保存所有标签到CSV"""
        # 创建DataFrame
        df_dict = {'uid': self.node_list}
        for task_name, labels_dict in self.labels.items():
            df_dict[task_name] = [labels_dict.get(uid, 0) for uid in self.node_list]
        
        df = pd.DataFrame(df_dict)
        
        # 保存
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ 标签文件已保存到: {output_file}")
        
        return df
    
    def generate_all_labels(self):
        """生成所有预测标签"""
        print("\n" + "=" * 60)
        print("预测标签生成")
        print("=" * 60)
        
        # 加载节点列表
        self.load_node_list()
        
        # 生成三类标签
        self.generate_academic_risk_labels()
        self.generate_mental_health_risk_labels()
        self.generate_social_health_risk_labels()
        
        # 保存标签
        labels_df = self.save_labels()
        
        print("\n" + "=" * 60)
        print("标签生成完成！")
        print("=" * 60)
        
        # 显示标签概览
        print("\n标签概览:")
        print(labels_df.describe())
        
        return labels_df


def main():
    """主函数"""
    generator = PredictionLabelGenerator()
    labels_df = generator.generate_all_labels()
    
    print("\n预览前10行:")
    print(labels_df.head(10))


if __name__ == '__main__':
    main()
