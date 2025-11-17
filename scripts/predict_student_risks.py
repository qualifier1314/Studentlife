"""
学生风险预测模型
使用多层网络特征预测三类风险：
1. 学业风险
2. 心理健康风险
3. 社交健康风险
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StudentRiskPredictor:
    """学生风险预测器"""
    
    def __init__(self, features_file='outputs/static/multilayer_features.csv',
                 labels_file='outputs/static/prediction_labels.csv'):
        self.features_file = features_file
        self.labels_file = labels_file
        self.features = None
        self.labels = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """加载特征和标签"""
        print("\n加载数据...")
        print("-" * 60)
        
        # 加载特征
        features_df = pd.read_csv(self.features_file, index_col='uid')
        
        # 移除行为属性特征（当前为0）
        behavior_cols = [c for c in features_df.columns if 
                        'call' in c or 'sms' in c or 'app_usage' in c or 
                        'ema' in c or 'neighbor_avg' in c or
                        'calendar' in c or 'physical_activity' in c]
        
        features_df = features_df.drop(columns=behavior_cols, errors='ignore')
        
        print(f"  ✓ 加载特征矩阵: {features_df.shape}")
        print(f"    - 学生数: {features_df.shape[0]}")
        print(f"    - 特征数: {features_df.shape[1]}")
        
        # 加载标签
        labels_df = pd.read_csv(self.labels_file, index_col='uid')
        print(f"  ✓ 加载标签矩阵: {labels_df.shape}")
        
        # 对齐索引
        common_uids = features_df.index.intersection(labels_df.index)
        self.features = features_df.loc[common_uids]
        self.labels = labels_df.loc[common_uids]
        
        print(f"  ✓ 对齐后样本数: {len(common_uids)}")
        
        # 显示特征类型
        print(f"\n特征类别:")
        single_layer = [c for c in self.features.columns if 
                       'physical_' in c or 'behavioral_' in c or 'educational_' in c]
        multi_layer = [c for c in self.features.columns if 
                      'cross_layer' in c or 'inter_layer' in c or 
                      'total_unique' in c or 'neighbor_overlap' in c]
        print(f"    - 单层网络特征: {len(single_layer)}")
        print(f"    - 多层网络特征: {len(multi_layer)}")
        
        return self.features, self.labels
    
    def train_model(self, task_name, y, n_folds=5):
        """
        训练预测模型
        
        参数:
        - task_name: 任务名称
        - y: 标签向量
        - n_folds: 交叉验证折数
        """
        print(f"\n训练模型: {task_name}")
        print("-" * 60)
        
        X = self.features.values
        
        # 检查类别分布
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        print(f"  类别分布: 正类={pos_count}, 负类={neg_count}")
        
        if pos_count < 2:
            print(f"  ⚠️  正类样本过少（<2），跳过该任务")
            return None
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建随机森林模型
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=min(n_folds, pos_count), shuffle=True, random_state=42)
        
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            print(f"  ✓ 交叉验证 AUC-ROC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        except Exception as e:
            print(f"  ⚠️  交叉验证失败: {e}")
            cv_scores = np.array([0.5])
        
        # 训练最终模型（使用全部数据）
        model.fit(X_scaled, y)
        
        # 预测
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        # 评估
        print(f"\n  分类报告:")
        report = classification_report(y, y_pred, target_names=['低风险', '高风险'], 
                                      zero_division=0)
        print("  " + report.replace('\n', '\n  '))
        
        # AUC-ROC
        try:
            auc_score = roc_auc_score(y, y_proba)
            print(f"  AUC-ROC: {auc_score:.3f}")
        except Exception as e:
            auc_score = 0.5
            print(f"  AUC-ROC: 无法计算")
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        print(f"\n  混淆矩阵:")
        print(f"    预测负  预测正")
        print(f"  真负  {cm[0, 0]:3d}     {cm[0, 1]:3d}")
        print(f"  真正  {cm[1, 0]:3d}     {cm[1, 1]:3d}")
        
        # 保存模型和结果
        self.models[task_name] = {
            'model': model,
            'scaler': scaler,
            'cv_scores': cv_scores,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return model
    
    def train_all_tasks(self):
        """训练所有预测任务"""
        print("\n" + "=" * 60)
        print("多层网络风险预测模型训练")
        print("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 训练三个任务
        tasks = {
            'academic_risk': '学业风险',
            'mental_health_risk': '心理健康风险',
            'social_health_risk': '社交健康风险'
        }
        
        for task_name, task_label in tasks.items():
            y = self.labels[task_name].values
            self.train_model(task_name, y)
        
        # 生成总结报告
        self.generate_summary()
    
    def generate_summary(self):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("模型性能总结")
        print("=" * 60)
        
        summary_data = []
        
        for task_name, model_info in self.models.items():
            if model_info is not None:
                summary_data.append({
                    '任务': task_name.replace('_', ' ').title(),
                    'CV AUC (均值)': f"{model_info['cv_scores'].mean():.3f}",
                    'CV AUC (标准差)': f"{model_info['cv_scores'].std():.3f}",
                    '训练集 AUC': f"{model_info['auc_score']:.3f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n")
            print(summary_df.to_string(index=False))
            
            # 保存到文件
            output_dir = 'outputs/static'
            os.makedirs(output_dir, exist_ok=True)
            
            summary_file = os.path.join(output_dir, 'prediction_model_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("多层网络风险预测模型 - 性能总结\n")
                f.write("=" * 60 + "\n\n")
                f.write(summary_df.to_string(index=False))
                f.write("\n\n生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            print(f"\n✓ 总结报告已保存到: {summary_file}")
    
    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性"""
        print("\n生成特征重要性图表...")
        
        output_dir = 'outputs/static/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        for task_name, model_info in self.models.items():
            if model_info is None:
                continue
            
            model = model_info['model']
            importances = model.feature_importances_
            
            # 获取Top N特征
            indices = np.argsort(importances)[::-1][:top_n]
            top_features = [self.features.columns[i] for i in indices]
            top_importances = importances[indices]
            
            # 绘图
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_importances, color='steelblue')
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features - {task_name.replace("_", " ").title()}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # 保存
            output_file = os.path.join(output_dir, f'feature_importance_{task_name}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {task_name}: {output_file}")
    
    def analyze_feature_importance(self):
        """分析特征重要性并生成报告"""
        print("\n" + "=" * 60)
        print("特征重要性分析")
        print("=" * 60)
        
        self.plot_feature_importance()
        
        # 生成文本报告
        output_dir = 'outputs/static'
        report_file = os.path.join(output_dir, 'feature_importance_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("多层网络特征重要性分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            for task_name, model_info in self.models.items():
                if model_info is None:
                    continue
                
                f.write(f"\n{task_name.replace('_', ' ').title()}\n")
                f.write("-" * 60 + "\n")
                
                model = model_info['model']
                importances = model.feature_importances_
                
                # 排序
                indices = np.argsort(importances)[::-1]
                
                f.write("\nTop 10 重要特征:\n")
                for i, idx in enumerate(indices[:10], 1):
                    feature_name = self.features.columns[idx]
                    importance = importances[idx]
                    f.write(f"  {i:2d}. {feature_name:40s} {importance:.4f}\n")
                
                # 按类别统计
                f.write("\n特征类别贡献:\n")
                single_layer_importance = sum([importances[i] for i, col in enumerate(self.features.columns)
                                              if 'physical_' in col or 'behavioral_' in col or 'educational_' in col])
                multi_layer_importance = sum([importances[i] for i, col in enumerate(self.features.columns)
                                             if 'cross_layer' in col or 'inter_layer' in col or 
                                             'total_unique' in col or 'neighbor_overlap' in col])
                
                total = single_layer_importance + multi_layer_importance
                if total > 0:
                    f.write(f"  - 单层网络特征: {single_layer_importance:.3f} ({single_layer_importance/total*100:.1f}%)\n")
                    f.write(f"  - 多层网络特征: {multi_layer_importance:.3f} ({multi_layer_importance/total*100:.1f}%)\n")
        
        print(f"\n✓ 特征重要性报告已保存到: {report_file}")


def main():
    """主函数"""
    predictor = StudentRiskPredictor()
    
    # 训练所有任务
    predictor.train_all_tasks()
    
    # 分析特征重要性
    predictor.analyze_feature_importance()
    
    print("\n" + "=" * 60)
    print("预测模型训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
