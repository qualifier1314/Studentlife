"""
特征组合对比实验：量化多层网络特征的贡献
目标：为复杂网络方法论论文提供核心实验证据

实验设计：
1. 特征组合对比（5组）：量化多层特征提升幅度
2. 模型对比（3种）：验证特征质量而非模型过拟合
3. 层重要性分析：发现哪一层最关键
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FeatureCombinationComparison:
    """特征组合对比实验类"""
    
    def __init__(self, features_file='outputs/static/multilayer_features_no_behavior.csv',
                 labels_file='outputs/static/prediction_labels.csv'):
        self.features_file = features_file
        self.labels_file = labels_file
        self.features = None
        self.labels = None
        self.feature_groups = {}
        self.results = {}
    
    def load_data(self):
        """加载特征和标签"""
        print("\n" + "=" * 70)
        print("加载数据")
        print("=" * 70)
        
        # 加载特征
        features_df = pd.read_csv(self.features_file, index_col='uid')
        print(f"  ✓ 加载特征矩阵: {features_df.shape}")
        
        # 加载标签
        labels_df = pd.read_csv(self.labels_file, index_col='uid')
        print(f"  ✓ 加载标签矩阵: {labels_df.shape}")
        
        # 对齐索引
        common_uids = features_df.index.intersection(labels_df.index)
        self.features = features_df.loc[common_uids]
        self.labels = labels_df.loc[common_uids]
        
        print(f"  ✓ 对齐后样本数: {len(common_uids)}")
        
        # 移除全零特征
        non_zero_cols = self.features.columns[(self.features != 0).any()]
        self.features = self.features[non_zero_cols]
        print(f"  ✓ 移除零值特征后: {self.features.shape[1]} 特征")
        
        # 定义特征组
        self._define_feature_groups()
        
        return self.features, self.labels
    
    def _define_feature_groups(self):
        """定义特征组（针对复杂网络方法论论文）"""
        print(f"\n定义特征组（方法论视角）...")
        
        all_cols = self.features.columns.tolist()
        
        # 1. Physical层特征
        physical_features = [c for c in all_cols if 'physical_' in c and 'neighbor' not in c]
        
        # 2. Behavioral层特征
        behavioral_features = [c for c in all_cols if 'behavioral_' in c and 'neighbor' not in c]
        
        # 3. Educational层特征
        educational_features = [c for c in all_cols if 'educational_' in c and 'neighbor' not in c]
        
        # 4. 单层网络特征（所有单层特征的并集）
        single_layer = list(set(physical_features + behavioral_features + educational_features))
        
        # 5. 多层网络特征（跨层结构特征）
        multi_layer = [c for c in all_cols if 
                      'cross_layer' in c or 'degree_variance' in c or 'degree_std' in c or
                      'degree_mean' in c or 'consistency' in c or 'overlap' in c or
                      'total_unique' in c]
        
        # 6. 行为属性特征
        behavioral_attr = [c for c in all_cols if 
                          'call' in c or 'sms' in c or 'app_usage' in c or 
                          'calendar' in c or 'physical_activity' in c or
                          ('ema' in c and 'neighbor' not in c)]
        
        # 7. 同伴影响特征（基于网络结构的传播）
        peer_influence = [c for c in all_cols if 'neighbor' in c]
        
        self.feature_groups = {
            # 单层对比
            'Physical Layer Only': physical_features,
            'Behavioral Layer Only': behavioral_features,
            'Educational Layer Only': educational_features,
            
            # 方法论对比（核心）
            'Single-Layer Features': single_layer,
            'Multi-Layer Features': multi_layer,
            'Single+Multi (Topology)': single_layer + multi_layer,
            
            # 完整特征
            'Topology+Attributes': single_layer + multi_layer + behavioral_attr,
            'All Features': single_layer + multi_layer + behavioral_attr + peer_influence,
        }
        
        print(f"\n  【单层对比】:")
        print(f"    - Physical Layer Only: {len(physical_features)} 特征")
        print(f"    - Behavioral Layer Only: {len(behavioral_features)} 特征")
        print(f"    - Educational Layer Only: {len(educational_features)} 特征")
        
        print(f"\n  【方法论对比（核心）】:")
        print(f"    - Single-Layer Features: {len(single_layer)} 特征")
        print(f"    - Multi-Layer Features: {len(multi_layer)} 特征")
        print(f"    - Single+Multi (Topology): {len(single_layer + multi_layer)} 特征")
        
        print(f"\n  【完整特征】:")
        print(f"    - Topology+Attributes: {len(single_layer + multi_layer + behavioral_attr)} 特征")
        print(f"    - All Features: {len(all_cols)} 特征")
    
    def compare_feature_combinations(self, task_name, y, n_folds=5):
        """对比不同特征组合"""
        print(f"\n" + "=" * 70)
        print(f"对比实验: {task_name}")
        print("=" * 70)
        
        # 检查类别分布
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_ratio = pos_count / len(y) * 100
        print(f"\n类别分布: 正类={pos_count} ({pos_ratio:.1f}%), 负类={neg_count} ({100-pos_ratio:.1f}%)")
        
        if pos_count < 2:
            print(f"  ⚠️  正类样本过少（<2），跳过该任务")
            return None
        
        results = {}
        
        # 定义对比顺序（按论文逻辑）
        comparison_order = [
            # 单层对比
            ('Physical Layer Only', 'Physical Layer Only'),
            ('Behavioral Layer Only', 'Behavioral Layer Only'),
            ('Educational Layer Only', 'Educational Layer Only'),
            
            # 方法论对比（核心）
            ('Single-Layer Features', 'Single-Layer Features'),
            ('Multi-Layer Features', 'Multi-Layer Features'),
            ('Single+Multi (Topology)', 'Single+Multi (Topology)'),
            
            # 完整特征
            ('Topology+Attributes', 'Topology+Attributes'),
            ('All Features', 'All Features'),
        ]
        
        for display_name, group_name in comparison_order:
            if group_name not in self.feature_groups:
                continue
            
            feature_cols = self.feature_groups[group_name]
            if len(feature_cols) == 0:
                print(f"\n  ⚠️  {display_name}: 无可用特征，跳过")
                continue
            
            print(f"\n{'='*70}")
            print(f"{display_name} ({len(feature_cols)} 特征)")
            print(f"{'='*70}")
            
            X = self.features[feature_cols].values
            
            # 处理NaN
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Random Forest
            rf_results = self._train_and_evaluate(
                'Random Forest', 
                RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     random_state=42, class_weight='balanced'),
                X_scaled, y, n_folds, pos_count
            )
            
            # Logistic Regression
            lr_results = self._train_and_evaluate(
                'Logistic Regression',
                LogisticRegression(max_iter=1000, random_state=42, 
                                 class_weight='balanced', penalty='l2', C=1.0),
                X_scaled, y, n_folds, pos_count
            )
            
            # SVM（只在样本数不太多时）
            if len(y) < 100:
                svm_results = self._train_and_evaluate(
                    'SVM',
                    SVC(kernel='rbf', probability=True, random_state=42, 
                       class_weight='balanced', C=1.0),
                    X_scaled, y, n_folds, pos_count
                )
            else:
                svm_results = None
            
            results[display_name] = {
                'n_features': len(feature_cols),
                'Random Forest': rf_results,
                'Logistic Regression': lr_results,
                'SVM': svm_results
            }
        
        self.results[task_name] = results
        return results
    
    def _train_and_evaluate(self, model_name, model, X, y, n_folds, pos_count):
        """训练并评估模型"""
        print(f"\n  [{model_name}]")
        
        # 交叉验证
        actual_folds = min(n_folds, pos_count)
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_auc_mean = cv_scores.mean()
            cv_auc_std = cv_scores.std()
            print(f"    交叉验证 AUC: {cv_auc_mean:.3f} (±{cv_auc_std:.3f})")
        except Exception as e:
            print(f"    ⚠️  交叉验证失败: {e}")
            cv_auc_mean = 0.0
            cv_auc_std = 0.0
        
        # 训练最终模型
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            else:
                y_proba = y_pred
            
            # 评估指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average='binary', zero_division=0
            )
            
            try:
                train_auc = roc_auc_score(y, y_proba)
            except:
                train_auc = 0.0
            
            print(f"    训练集 AUC: {train_auc:.3f}")
            print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
        except Exception as e:
            print(f"    ⚠️  训练失败: {e}")
            train_auc = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'train_auc': train_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def generate_comparison_report(self):
        """生成对比报告（面向复杂网络论文）"""
        print("\n" + "=" * 70)
        print("生成对比实验报告")
        print("=" * 70)
        
        output_dir = 'outputs/static'
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成CSV报告
        csv_file = os.path.join(output_dir, 'feature_combination_comparison.csv')
        
        rows = []
        for task_name, task_results in self.results.items():
            for feature_combo, combo_results in task_results.items():
                n_features = combo_results['n_features']
                
                for model_name, model_results in combo_results.items():
                    if model_name == 'n_features' or model_results is None:
                        continue
                    
                    rows.append({
                        'Task': task_name,
                        'Feature_Combination': feature_combo,
                        'N_Features': n_features,
                        'Model': model_name,
                        'CV_AUC_Mean': model_results['cv_auc_mean'],
                        'CV_AUC_Std': model_results['cv_auc_std'],
                        'Train_AUC': model_results['train_auc'],
                        'Precision': model_results['precision'],
                        'Recall': model_results['recall'],
                        'F1': model_results['f1']
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"\n✓ 对比结果已保存到: {csv_file}")
        
        # 显示关键结果（论文核心发现）
        print(f"\n关键结果摘要（面向复杂网络论文）:")
        print("=" * 70)
        
        for task_name in self.results.keys():
            print(f"\n【{task_name}】")
            task_df = df[df['Task'] == task_name]
            
            # 使用Random Forest的结果（最稳定）
            rf_df = task_df[task_df['Model'] == 'Random Forest'].copy()
            rf_df = rf_df.sort_values('CV_AUC_Mean', ascending=False)
            
            if len(rf_df) > 0:
                # 最佳组合
                best = rf_df.iloc[0]
                print(f"\n  最佳特征组合: {best['Feature_Combination']}")
                print(f"    - CV AUC: {best['CV_AUC_Mean']:.3f} (±{best['CV_AUC_Std']:.3f})")
                print(f"    - 特征数: {int(best['N_Features'])}")
                
                # 核心对比：单层 vs 多层
                single_layer = rf_df[rf_df['Feature_Combination'] == 'Single-Layer Features']
                multi_layer = rf_df[rf_df['Feature_Combination'] == 'Multi-Layer Features']
                single_multi = rf_df[rf_df['Feature_Combination'] == 'Single+Multi (Topology)']
                
                if len(single_layer) > 0:
                    baseline_auc = single_layer.iloc[0]['CV_AUC_Mean']
                    print(f"\n  【方法论贡献】")
                    print(f"    单层特征基线: {baseline_auc:.3f}")
                    
                    if len(multi_layer) > 0:
                        multi_auc = multi_layer.iloc[0]['CV_AUC_Mean']
                        print(f"    仅多层特征: {multi_auc:.3f}")
                    
                    if len(single_multi) > 0:
                        combined_auc = single_multi.iloc[0]['CV_AUC_Mean']
                        abs_improvement = combined_auc - baseline_auc
                        rel_improvement = (combined_auc / baseline_auc - 1) * 100
                        
                        print(f"    单层+多层: {combined_auc:.3f}")
                        print(f"    ➜ 绝对提升: +{abs_improvement:.3f}")
                        print(f"    ➜ 相对提升: +{rel_improvement:.1f}%")
                
                # 层重要性对比
                print(f"\n  【层重要性分析】")
                for layer_name in ['Physical Layer Only', 'Behavioral Layer Only', 'Educational Layer Only']:
                    layer_res = rf_df[rf_df['Feature_Combination'] == layer_name]
                    if len(layer_res) > 0:
                        layer_auc = layer_res.iloc[0]['CV_AUC_Mean']
                        print(f"    {layer_name}: {layer_auc:.3f}")
                
                # 模型一致性检验
                print(f"\n  【模型一致性】")
                single_multi_all = task_df[task_df['Feature_Combination'] == 'Single+Multi (Topology)']
                for _, row in single_multi_all.iterrows():
                    print(f"    {row['Model']}: {row['CV_AUC_Mean']:.3f} (±{row['CV_AUC_Std']:.3f})")
        
        return df
    
    def plot_comparison_results(self):
        """绘制对比结果图（面向论文）"""
        print("\n生成对比图表...")
        
        output_dir = 'outputs/static/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        for task_name, task_results in self.results.items():
            # 图1: 特征组合对比（Random Forest）
            self._plot_feature_combination_bars(task_name, task_results, output_dir)
            
            # 图2: 模型一致性验证
            self._plot_model_consistency(task_name, task_results, output_dir)
    
    def _plot_feature_combination_bars(self, task_name, task_results, output_dir):
        """绘制特征组合对比柱状图"""
        # 提取Random Forest结果
        feature_combos = []
        aucs = []
        stds = []
        n_features_list = []
        
        for feature_combo, combo_results in task_results.items():
            rf_res = combo_results.get('Random Forest')
            if rf_res:
                feature_combos.append(feature_combo)
                aucs.append(rf_res['cv_auc_mean'])
                stds.append(rf_res['cv_auc_std'])
                n_features_list.append(combo_results['n_features'])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(feature_combos))
        bars = ax.bar(x, aucs, yerr=stds, capsize=5, 
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # 标注特征数
        for i, (bar, n_feat) in enumerate(zip(bars, n_features_list)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.02,
                   f'n={n_feat}',
                   ha='center', va='bottom', fontsize=9, color='darkred')
        
        # 添加基线
        if 'Single-Layer Features' in feature_combos:
            baseline_idx = feature_combos.index('Single-Layer Features')
            baseline_auc = aucs[baseline_idx]
            ax.axhline(y=baseline_auc, color='red', linestyle='--', linewidth=2, 
                      label=f'Single-Layer Baseline ({baseline_auc:.3f})')
        
        ax.set_xlabel('Feature Combination', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cross-Validation AUC-ROC', fontsize=13, fontweight='bold')
        ax.set_title(f'Feature Combination Comparison - {task_name}\n(Random Forest, 5-Fold CV)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_combos, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, min(1.0, max(aucs) + max(stds) + 0.1)])
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 
                                  f'feature_combo_{task_name.replace(" ", "_").lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {task_name} 特征组合对比: {output_file}")
    
    def _plot_model_consistency(self, task_name, task_results, output_dir):
        """绘制模型一致性验证图"""
        # 选择关键特征组合
        key_combos = ['Single-Layer Features', 'Multi-Layer Features', 'Single+Multi (Topology)']
        
        model_names = ['Random Forest', 'Logistic Regression', 'SVM']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(key_combos))
        width = 0.25
        
        colors = ['steelblue', 'orange', 'green']
        
        for i, model_name in enumerate(model_names):
            aucs = []
            for combo in key_combos:
                if combo in task_results:
                    model_res = task_results[combo].get(model_name)
                    if model_res:
                        aucs.append(model_res['cv_auc_mean'])
                    else:
                        aucs.append(0)
                else:
                    aucs.append(0)
            
            ax.bar(x + i*width, aucs, width, label=model_name, 
                  color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Feature Combination', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Validation AUC-ROC', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Consistency - {task_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(key_combos, rotation=30, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 
                                  f'model_consistency_{task_name.replace(" ", "_").lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {task_name} 模型一致性: {output_file}")
    
    def run_all_comparisons(self):
        """运行所有对比实验"""
        print("\n" + "=" * 70)
        print("特征组合对比实验")
        print("目标：量化多层网络特征的方法论贡献")
        print("=" * 70)
        
        # 加载数据
        self.load_data()
        
        # 对每个任务运行对比实验
        tasks = {
            'mental_health_risk': 'Mental Health Risk',
            'social_health_risk': 'Social Health Risk'
        }
        
        for task_name, task_label in tasks.items():
            if task_name not in self.labels.columns:
                continue
            
            y = self.labels[task_name].values
            self.compare_feature_combinations(task_label, y)
        
        # 生成报告和图表
        results_df = self.generate_comparison_report()
        self.plot_comparison_results()
        
        print("\n" + "=" * 70)
        print("对比实验完成！")
        print("=" * 70)
        print("\n核心发现（论文卖点）:")
        print("  1. 多层网络特征相对于单层特征的提升幅度")
        print("  2. 不同模型的一致性（证明是特征质量而非模型过拟合）")
        print("  3. 各层网络的相对重要性")
        print("\n输出文件:")
        print("  - outputs/static/feature_combination_comparison.csv")
        print("  - outputs/static/figures/feature_combo_*.png")
        print("  - outputs/static/figures/model_consistency_*.png")
        
        return results_df


def main():
    """主函数"""
    comparator = FeatureCombinationComparison()
    results_df = comparator.run_all_comparisons()
    
    print("\n结果预览（前20行）:")
    print(results_df.head(20).to_string())


if __name__ == '__main__':
    main()
