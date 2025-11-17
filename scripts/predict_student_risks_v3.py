"""
学生风险预测模型 V3（支持缺失值）
使用XGBoost训练49个学生的多层网络特征

特点：
1. 原生支持NaN缺失值（无需imputation）
2. 使用49个全体学生（vs V2的28人）
3. 支持层覆盖特征（has_*_layer, layer_coverage）
4. 5-Fold交叉验证，每折9-10个样本
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# XGBoost原生支持NaN
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost未安装，请运行: pip install xgboost")
    XGBOOST_AVAILABLE = False
    sys.exit(1)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StudentRiskPredictorV3:
    """学生风险预测器 V3（支持缺失值）"""
    
    def __init__(self, features_file='outputs/static_v3/multilayer_features_v3.csv',
                 labels_file='outputs/static/prediction_labels.csv'):
        self.features_file = features_file
        self.labels_file = labels_file
        self.features = None
        self.labels = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def load_data(self):
        """加载特征和标签"""
        print("\n" + "=" * 80)
        print(" 加载数据（V3版本 - 49个学生） ".center(80, "="))
        print("=" * 80)
        
        # 加载V3特征（49样本）
        features_df = pd.read_csv(self.features_file, index_col='uid')
        print(f"\n✓ 加载V3特征矩阵: {features_df.shape}")
        print(f"  - 学生数: {features_df.shape[0]} (100%覆盖)")
        print(f"  - 特征数: {features_df.shape[1]}")
        print(f"  - 样本/特征比例: {features_df.shape[0] / features_df.shape[1]:.2f}")
        
        # 统计缺失值
        missing_count = features_df.isnull().sum().sum()
        missing_ratio = missing_count / features_df.size * 100
        print(f"\n缺失值统计:")
        print(f"  - 总缺失值: {missing_count}/{features_df.size} ({missing_ratio:.1f}%)")
        
        # 显示缺失最多的特征
        missing_per_feature = features_df.isnull().sum().sort_values(ascending=False)
        missing_per_feature = missing_per_feature[missing_per_feature > 0].head(5)
        if len(missing_per_feature) > 0:
            print(f"  - 缺失最多的特征（Top 5）:")
            for col, count in missing_per_feature.items():
                print(f"    • {col}: {count}/{len(features_df)} ({count/len(features_df)*100:.1f}%)")
        
        # 加载标签
        labels_df = pd.read_csv(self.labels_file, index_col='uid')
        print(f"\n✓ 加载标签矩阵: {labels_df.shape}")
        
        # 对齐索引
        common_uids = features_df.index.intersection(labels_df.index)
        self.features = features_df.loc[common_uids]
        self.labels = labels_df.loc[common_uids]
        
        print(f"\n✓ 对齐后样本数: {len(common_uids)}")
        
        # 显示特征类别
        print(f"\n特征类别分析:")
        
        single_layer = [c for c in self.features.columns if 
                       any(x in c for x in ['physical_', 'behavioral_', 'educational_']) 
                       and 'has_' not in c and '_z' not in c]
        zscore_features = [c for c in self.features.columns if '_z' in c]
        coverage_features = [c for c in self.features.columns if 'has_' in c or c == 'layer_coverage']
        multi_layer = [c for c in self.features.columns if 
                      any(x in c for x in ['cross_layer', 'participation', 'overlapping', 'entropy', 'avg_zscore', 'degree_variance'])]
        behavior_attr = [c for c in self.features.columns if 'ema_' in c or 'peer_' in c or 'total_neighbors' in c]
        
        print(f"  - 单层拓扑特征: {len(single_layer)}个")
        print(f"  - Z-score标准化: {len(zscore_features)}个")
        print(f"  - 层覆盖特征: {len(coverage_features)}个 ← V3新增")
        print(f"  - 多层网络特征: {len(multi_layer)}个")
        print(f"  - 行为属性特征: {len(behavior_attr)}个")
        
        return self.features, self.labels
    
    def train_model(self, task_name, y, n_folds=5):
        """
        训练XGBoost模型（支持NaN）
        
        参数:
        - task_name: 任务名称
        - y: 标签向量
        - n_folds: 交叉验证折数
        """
        print("\n" + "=" * 80)
        print(f" 训练模型: {task_name} ".center(80, "="))
        print("=" * 80)
        
        X = self.features.values
        
        # 检查类别分布
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_ratio = pos_count / len(y) * 100
        
        print(f"\n类别分布:")
        print(f"  - 正类（高风险）: {pos_count}个 ({pos_ratio:.1f}%)")
        print(f"  - 负类（低风险）: {neg_count}个 ({100-pos_ratio:.1f}%)")
        
        if pos_count < 2 or neg_count < 2:
            print(f"\n⚠️  样本过少（正类<2或负类<2），跳过该任务")
            return None
        
        # 计算类别权重（处理不平衡）
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # 创建XGBoost模型（原生支持NaN）
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            missing=np.nan,  # ✅ 关键：指定NaN为缺失值标记
            scale_pos_weight=scale_pos_weight,  # 处理类别不平衡
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        print(f"\n模型配置:")
        print(f"  - 模型类型: XGBoost")
        print(f"  - 树数量: 100")
        print(f"  - 最大深度: 3")
        print(f"  - 学习率: 0.1")
        print(f"  - 缺失值处理: 原生支持NaN ✅")
        print(f"  - 类别权重: {scale_pos_weight:.2f}")
        
        # 交叉验证
        print(f"\n执行 {n_folds}-Fold 交叉验证...")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_auc_scores = []
        fold_predictions = []
        fold_true_labels = []
        fold_feature_importance = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算指标
            accuracy = (y_pred == y_test).mean()
            
            # AUC（需要至少两个类别）
            if len(np.unique(y_test)) >= 2:
                auc = roc_auc_score(y_test, y_pred_proba)
                cv_auc_scores.append(auc)
            else:
                auc = np.nan
            
            cv_scores.append(accuracy)
            fold_predictions.extend(y_pred_proba)
            fold_true_labels.extend(y_test)
            
            # 特征重要性
            fold_feature_importance.append(model.feature_importances_)
            
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
            print(f"  Fold {fold_idx}: "
                  f"训练={len(train_idx)}样本, 测试={len(test_idx)}样本, "
                  f"Acc={accuracy:.3f}, AUC={auc_str}")
        
        # 计算平均指标
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        
        if len(cv_auc_scores) > 0:
            mean_auc = np.mean(cv_auc_scores)
            std_auc = np.std(cv_auc_scores)
        else:
            mean_auc = np.nan
            std_auc = np.nan
        
        print(f"\n交叉验证结果:")
        print(f"  - 平均准确率: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        if not np.isnan(mean_auc):
            print(f"  - 平均AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        else:
            print(f"  - 平均AUC: N/A（某些折缺少类别）")
        
        # 最终训练（在全部数据上）
        print(f"\n在全部{len(X)}个样本上训练最终模型...")
        model.fit(X, y)
        
        # 特征重要性
        feature_names = self.features.columns.tolist()
        mean_importance = np.mean(fold_feature_importance, axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性（Top 10）:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.4f}")
        
        # 保存结果
        self.models[task_name] = model
        self.feature_importance[task_name] = importance_df
        self.results[task_name] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'cv_scores': cv_scores,
            'cv_auc_scores': cv_auc_scores,
            'predictions': fold_predictions,
            'true_labels': fold_true_labels,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'pos_count': pos_count,
            'neg_count': neg_count
        }
        
        return model
    
    def run_all_tasks(self):
        """运行所有预测任务"""
        print("\n" + "=" * 80)
        print(" 运行所有预测任务 ".center(80, "="))
        print("=" * 80)
        
        # 任务定义
        tasks = {
            '学业风险': 'academic_risk',
            '心理健康风险': 'mental_health_risk',
            '社交孤立风险': 'social_isolation_risk'
        }
        
        for task_name, label_col in tasks.items():
            if label_col in self.labels.columns:
                y = self.labels[label_col]
                self.train_model(task_name, y)
            else:
                print(f"\n⚠️  标签列 '{label_col}' 不存在，跳过任务 '{task_name}'")
        
        return self.results
    
    def generate_report(self, output_dir='outputs/static_v3'):
        """生成训练报告"""
        print("\n" + "=" * 80)
        print(" 生成训练报告 ".center(80, "="))
        print("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成文本报告
        report_file = os.path.join(output_dir, 'training_results_v3.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" V3版本训练结果报告（49个学生） ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据信息:\n")
            f.write(f"  - 样本数: {self.features.shape[0]}\n")
            f.write(f"  - 特征数: {self.features.shape[1]}\n")
            f.write(f"  - 样本/特征比例: {self.features.shape[0] / self.features.shape[1]:.2f}\n")
            f.write(f"  - 缺失值比例: {self.features.isnull().sum().sum() / self.features.size * 100:.1f}%\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("各任务性能\n")
            f.write("=" * 80 + "\n\n")
            
            for task_name, result in self.results.items():
                f.write(f"{task_name}:\n")
                f.write(f"  - 样本数: {result['n_samples']}\n")
                f.write(f"  - 正类: {result['pos_count']}个 ({result['pos_count']/result['n_samples']*100:.1f}%)\n")
                f.write(f"  - 负类: {result['neg_count']}个 ({result['neg_count']/result['n_samples']*100:.1f}%)\n")
                f.write(f"  - 平均准确率: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}\n")
                if not np.isnan(result['mean_auc']):
                    f.write(f"  - 平均AUC: {result['mean_auc']:.3f} ± {result['std_auc']:.3f}\n")
                else:
                    f.write(f"  - 平均AUC: N/A\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("特征重要性（Top 15）\n")
            f.write("=" * 80 + "\n\n")
            
            for task_name, importance_df in self.feature_importance.items():
                f.write(f"{task_name}:\n")
                for i, row in importance_df.head(15).iterrows():
                    f.write(f"  {i+1:2d}. {row['feature']:40s}: {row['importance']:.4f}\n")
                f.write("\n")
        
        print(f"✓ 文本报告已保存: {report_file}")
        
        # 2. 保存特征重要性CSV
        for task_name, importance_df in self.feature_importance.items():
            importance_file = os.path.join(output_dir, f'feature_importance_{task_name}.csv')
            importance_df.to_csv(importance_file, index=False)
            print(f"✓ 特征重要性已保存: {importance_file}")
        
        # 3. 生成可视化图表
        self._plot_results(output_dir)
        
        print("\n✓ 报告生成完成！")
    
    def _plot_results(self, output_dir):
        """生成可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 性能对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        task_names = list(self.results.keys())
        accuracies = [self.results[t]['mean_accuracy'] for t in task_names]
        acc_stds = [self.results[t]['std_accuracy'] for t in task_names]
        aucs = [self.results[t]['mean_auc'] if not np.isnan(self.results[t]['mean_auc']) else 0 
                for t in task_names]
        auc_stds = [self.results[t]['std_auc'] if not np.isnan(self.results[t]['std_auc']) else 0 
                    for t in task_names]
        
        # 准确率
        ax1 = axes[0]
        bars = ax1.bar(range(len(task_names)), accuracies, yerr=acc_stds, 
                       capsize=5, alpha=0.7, color='steelblue')
        ax1.set_xticks(range(len(task_names)))
        ax1.set_xticklabels(task_names, rotation=15, ha='right')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_title('V3模型准确率', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # AUC
        ax2 = axes[1]
        bars = ax2.bar(range(len(task_names)), aucs, yerr=auc_stds,
                       capsize=5, alpha=0.7, color='coral')
        ax2.set_xticks(range(len(task_names)))
        ax2.set_xticklabels(task_names, rotation=15, ha='right')
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.set_title('V3模型AUC', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='合理阈值(0.7)')
        ax2.axhline(y=0.85, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='上限(0.85)')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=9)
        
        for i, (bar, auc) in enumerate(zip(bars, aucs)):
            height = bar.get_height()
            if auc > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{auc:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'v3_model_performance.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 性能对比图已保存")
        plt.close()
        
        # 2. 特征重要性图（仅第一个任务）
        if len(self.feature_importance) > 0:
            first_task = list(self.feature_importance.keys())[0]
            importance_df = self.feature_importance[first_task].head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            bars = ax.barh(range(len(importance_df)), importance_df['importance'].values,
                          alpha=0.7, color='forestgreen')
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'].values, fontsize=9)
            ax.set_xlabel('特征重要性', fontsize=12)
            ax.set_title(f'特征重要性排名（{first_task}）', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'].values)):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                       f'{imp:.4f}',
                       ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'v3_feature_importance.png'), dpi=300, bbox_inches='tight')
            print(f"✓ 特征重要性图已保存")
            plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print(" V3版本学生风险预测（49个学生 + XGBoost） ".center(80, "="))
    print("=" * 80)
    
    # 创建预测器
    predictor = StudentRiskPredictorV3(
        features_file='outputs/static_v3/multilayer_features_v3.csv',
        labels_file='outputs/static/prediction_labels.csv'
    )
    
    # 加载数据
    predictor.load_data()
    
    # 运行所有任务
    results = predictor.run_all_tasks()
    
    # 生成报告
    predictor.generate_report(output_dir='outputs/static_v3')
    
    print("\n" + "=" * 80)
    print(" 训练完成！ ".center(80, "="))
    print("=" * 80)
    print("\n生成的文件:")
    print("  - outputs/static_v3/training_results_v3.txt")
    print("  - outputs/static_v3/feature_importance_*.csv")
    print("  - outputs/static_v3/v3_model_performance.png")
    print("  - outputs/static_v3/v3_feature_importance.png")


if __name__ == '__main__':
    main()
