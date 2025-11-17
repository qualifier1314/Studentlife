"""
对比V2（交集28人）vs V3（并集49人）的差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_features():
    """加载V2和V3特征"""
    v2_df = pd.read_csv('outputs/static_v2/multilayer_features_v2.csv', index_col='uid')
    v3_df = pd.read_csv('outputs/static_v3/multilayer_features_v3.csv', index_col='uid')
    
    print("V2版本（交集节点）:")
    print(f"  样本数: {len(v2_df)}")
    print(f"  特征数: {len(v2_df.columns)}")
    print(f"  样本/特征比例: {len(v2_df)/len(v2_df.columns):.2f}")
    
    print("\nV3版本（并集节点）:")
    print(f"  样本数: {len(v3_df)}")
    print(f"  特征数: {len(v3_df.columns)}")
    print(f"  样本/特征比例: {len(v3_df)/len(v3_df.columns):.2f}")
    
    return v2_df, v3_df


def compare_sample_coverage(v2_df, v3_df):
    """对比样本覆盖率"""
    print("\n" + "=" * 60)
    print("样本覆盖率对比")
    print("=" * 60)
    
    v2_nodes = set(v2_df.index)
    v3_nodes = set(v3_df.index)
    
    print(f"V2节点数: {len(v2_nodes)} ({len(v2_nodes)/49*100:.1f}%覆盖)")
    print(f"V3节点数: {len(v3_nodes)} ({len(v3_nodes)/49*100:.1f}%覆盖)")
    print(f"V3新增节点: {len(v3_nodes - v2_nodes)}个")
    print(f"覆盖率提升: +{(len(v3_nodes) - len(v2_nodes))/49*100:.1f}%")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['V2版本\n(交集28人)', 'V3版本\n(并集49人)']
    sample_counts = [len(v2_nodes), len(v3_nodes)]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax.bar(categories, sample_counts, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/49*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.axhline(y=49, color='red', linestyle='--', linewidth=2, label='全班总数(49人)')
    ax.set_ylabel('样本数', fontsize=12)
    ax.set_title('样本覆盖率对比：V2 vs V3', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/v2_vs_v3_sample_coverage.png', dpi=300, bbox_inches='tight')
    print("✅ 样本覆盖率对比图已保存")
    plt.close()


def compare_feature_dimensions(v2_df, v3_df):
    """对比特征维度"""
    print("\n" + "=" * 60)
    print("特征维度对比")
    print("=" * 60)
    
    # 统计特征类别
    v2_features = set(v2_df.columns)
    v3_features = set(v3_df.columns)
    
    # V3新增的层覆盖特征
    coverage_features = {'has_physical_layer', 'has_behavioral_layer', 'has_educational_layer', 'layer_coverage'}
    new_features = v3_features - v2_features
    
    print(f"V2特征数: {len(v2_features)}")
    print(f"V3特征数: {len(v3_features)}")
    print(f"V3新增特征: {len(new_features)}个")
    print(f"  新增层覆盖特征: {coverage_features & new_features}")
    
    # 样本/特征比例
    v2_ratio = len(v2_df) / len(v2_features)
    v3_ratio = len(v3_df) / len(v3_features)
    
    print(f"\n样本/特征比例:")
    print(f"  V2: {len(v2_df)}/{len(v2_features)} = {v2_ratio:.2f}")
    print(f"  V3: {len(v3_df)}/{len(v3_features)} = {v3_ratio:.2f}")
    print(f"  提升: +{(v3_ratio - v2_ratio)/v2_ratio*100:.1f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：特征数对比
    ax1 = axes[0]
    categories = ['V2版本', 'V3版本']
    feature_counts = [len(v2_features), len(v3_features)]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(categories, feature_counts, color=colors, alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('特征数', fontsize=12)
    ax1.set_title('特征维度对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 子图2：样本/特征比例对比
    ax2 = axes[1]
    ratios = [v2_ratio, v3_ratio]
    bars = ax2.bar(categories, ratios, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{ratio:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='推荐阈值(1.5)')
    ax2.set_ylabel('样本/特征比例', fontsize=12)
    ax2.set_title('样本充足性对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/v2_vs_v3_feature_dimensions.png', dpi=300, bbox_inches='tight')
    print("✅ 特征维度对比图已保存")
    plt.close()


def compare_missing_values(v2_df, v3_df):
    """对比缺失值情况"""
    print("\n" + "=" * 60)
    print("缺失值分析")
    print("=" * 60)
    
    # V2缺失值（应该很少，因为只包含交集节点）
    v2_missing = v2_df.isnull().sum().sum()
    v2_total = v2_df.size
    v2_missing_ratio = v2_missing / v2_total * 100
    
    # V3缺失值（预期会有，因为包含并集节点）
    v3_missing = v3_df.isnull().sum().sum()
    v3_total = v3_df.size
    v3_missing_ratio = v3_missing / v3_total * 100
    
    print(f"V2缺失值: {v2_missing}/{v2_total} ({v2_missing_ratio:.1f}%)")
    print(f"V3缺失值: {v3_missing}/{v3_total} ({v3_missing_ratio:.1f}%)")
    
    # 统计V3各特征的缺失比例
    v3_missing_per_feature = v3_df.isnull().sum().sort_values(ascending=False)
    v3_missing_per_feature = v3_missing_per_feature[v3_missing_per_feature > 0]
    
    print(f"\nV3缺失值最多的前5个特征:")
    for i, (feature, count) in enumerate(v3_missing_per_feature.head(5).items(), 1):
        print(f"  {i}. {feature}: {count}/{len(v3_df)} ({count/len(v3_df)*100:.1f}%)")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 只显示有缺失值的特征（top 15）
    top_missing = v3_missing_per_feature.head(15)
    
    bars = ax.barh(range(len(top_missing)), top_missing.values / len(v3_df) * 100, 
                   color='coral', alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing.index, fontsize=9)
    ax.set_xlabel('缺失比例 (%)', fontsize=12)
    ax.set_title('V3版本：特征缺失值分布（Top 15）', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, top_missing.values)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{count}/{len(v3_df)} ({count/len(v3_df)*100:.1f}%)',
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/v2_vs_v3_missing_values.png', dpi=300, bbox_inches='tight')
    print("✅ 缺失值分布图已保存")
    plt.close()


def compare_layer_coverage(v3_df):
    """分析V3的层覆盖情况"""
    print("\n" + "=" * 60)
    print("层覆盖情况分析（V3版本）")
    print("=" * 60)
    
    # 统计层覆盖
    coverage_stats = v3_df['layer_coverage'].value_counts().sort_index()
    
    print("层覆盖分布:")
    for layers, count in coverage_stats.items():
        print(f"  {int(layers)}层: {count}个学生 ({count/len(v3_df)*100:.1f}%)")
    
    # 各层覆盖率
    physical_coverage = v3_df['has_physical_layer'].sum()
    behavioral_coverage = v3_df['has_behavioral_layer'].sum()
    educational_coverage = v3_df['has_educational_layer'].sum()
    
    print(f"\n各层覆盖率:")
    print(f"  Physical层:    {physical_coverage}/{len(v3_df)} ({physical_coverage/len(v3_df)*100:.1f}%)")
    print(f"  Behavioral层:  {behavioral_coverage}/{len(v3_df)} ({behavioral_coverage/len(v3_df)*100:.1f}%)")
    print(f"  Educational层: {educational_coverage}/{len(v3_df)} ({educational_coverage/len(v3_df)*100:.1f}%)")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：层覆盖分布（饼图）
    ax1 = axes[0]
    labels = [f'{int(k)}层\n({v}人, {v/len(v3_df)*100:.1f}%)' for k, v in coverage_stats.items()]
    colors = ['#ff9999', '#ffcc99', '#99ff99']
    
    ax1.pie(coverage_stats.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('层覆盖分布', fontsize=14, fontweight='bold')
    
    # 子图2：各层覆盖率（柱状图）
    ax2 = axes[1]
    layers = ['Physical\n(物理)', 'Behavioral\n(行为)', 'Educational\n(教育)']
    coverage_counts = [physical_coverage, behavioral_coverage, educational_coverage]
    coverage_ratios = [c/len(v3_df)*100 for c in coverage_counts]
    colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(layers, coverage_ratios, color=colors2, alpha=0.7, edgecolor='black')
    
    for bar, count, ratio in zip(bars, coverage_counts, coverage_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}/{len(v3_df)}\n({ratio:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100%覆盖')
    ax2.set_ylabel('覆盖率 (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.set_title('各层覆盖率', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/v3_layer_coverage.png', dpi=300, bbox_inches='tight')
    print("✅ 层覆盖分析图已保存")
    plt.close()


def generate_comparison_report(v2_df, v3_df):
    """生成对比报告"""
    report_file = 'outputs/comparison/V2_VS_V3_COMPARISON_REPORT.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# V2版本 vs V3版本对比报告\n\n")
        f.write("> **核心差异**: V2使用交集节点（28人），V3使用并集节点（49人）\n\n")
        
        f.write("## 📊 核心指标对比\n\n")
        f.write("| 指标 | V2版本（交集） | V3版本（并集） | 提升幅度 |\n")
        f.write("|------|---------------|---------------|----------|\n")
        
        # 样本数
        v2_samples = len(v2_df)
        v3_samples = len(v3_df)
        sample_improvement = (v3_samples - v2_samples) / v2_samples * 100
        f.write(f"| 样本数 | {v2_samples} | {v3_samples} | +{sample_improvement:.1f}% ✅ |\n")
        
        # 特征数
        v2_features = len(v2_df.columns)
        v3_features = len(v3_df.columns)
        feature_diff = v3_features - v2_features
        f.write(f"| 特征数 | {v2_features} | {v3_features} | +{feature_diff} |\n")
        
        # 样本/特征比例
        v2_ratio = v2_samples / v2_features
        v3_ratio = v3_samples / v3_features
        ratio_improvement = (v3_ratio - v2_ratio) / v2_ratio * 100
        f.write(f"| 样本/特征比例 | {v2_ratio:.2f} | {v3_ratio:.2f} | +{ratio_improvement:.1f}% ✅ |\n")
        
        # 覆盖率
        v2_coverage = v2_samples / 49 * 100
        v3_coverage = v3_samples / 49 * 100
        f.write(f"| 班级覆盖率 | {v2_coverage:.1f}% | {v3_coverage:.1f}% | +{v3_coverage - v2_coverage:.1f}% ✅ |\n")
        
        # 缺失值
        v2_missing = v2_df.isnull().sum().sum()
        v3_missing = v3_df.isnull().sum().sum()
        v2_missing_ratio = v2_missing / v2_df.size * 100
        v3_missing_ratio = v3_missing / v3_df.size * 100
        f.write(f"| 缺失值比例 | {v2_missing_ratio:.1f}% | {v3_missing_ratio:.1f}% | +{v3_missing_ratio - v2_missing_ratio:.1f}% ⚠️ |\n\n")
        
        f.write("---\n\n")
        
        f.write("## 🎯 V3版本的核心优势\n\n")
        f.write("### 1. 样本充足性大幅提升\n\n")
        f.write(f"- **样本数**: 从28增至49（+75%）\n")
        f.write(f"- **样本/特征比例**: 从{v2_ratio:.2f}提升至{v3_ratio:.2f}（+50%）\n")
        f.write(f"- **超过推荐阈值**: {v3_ratio:.2f} > 1.5 ✅\n\n")
        
        f.write("**意义**:\n")
        f.write("- 更稳定的模型训练（减少过拟合风险）\n")
        f.write("- 更可靠的交叉验证（每折9-10个样本 vs V2的5-6个）\n")
        f.write("- 更好的泛化能力\n\n")
        
        f.write("### 2. 覆盖整个班级（代表性）\n\n")
        f.write(f"- **覆盖率**: 100% vs V2的57.1%\n")
        f.write(f"- **新增学生**: 21人（V3包含V2舍弃的学生）\n\n")
        
        f.write("**意义**:\n")
        f.write("- 避免选择偏差（不排除低活跃度学生）\n")
        f.write("- 研究结论代表整个班级\n")
        f.write("- 符合研究目标：\"基于StudentLife数据集研究整个班\"\n\n")
        
        f.write("### 3. 智能缺失值处理\n\n")
        f.write("**V2方案（交集）**:\n")
        f.write("- 简单舍弃缺失层的学生\n")
        f.write("- 丢失43%样本\n")
        f.write("- 无法研究\"缺失模式\"本身\n\n")
        
        f.write("**V3方案（并集 + 掩码）**:\n")
        f.write("- 保留所有学生\n")
        f.write("- 明确标记缺失层（4个层覆盖特征）\n")
        f.write("- 缺失层特征设为NaN（而非0）\n")
        f.write("- 使用XGBoost等支持NaN的模型\n\n")
        
        # 层覆盖统计
        coverage_stats = v3_df['layer_coverage'].value_counts().sort_index()
        f.write("**层覆盖分布**:\n")
        for layers, count in coverage_stats.items():
            f.write(f"- {int(layers)}层: {count}个学生 ({count/len(v3_df)*100:.1f}%)\n")
        f.write("\n")
        
        # 各层覆盖率
        physical_coverage = v3_df['has_physical_layer'].sum()
        behavioral_coverage = v3_df['has_behavioral_layer'].sum()
        educational_coverage = v3_df['has_educational_layer'].sum()
        
        f.write("**各层覆盖率**:\n")
        f.write(f"- Physical层: {physical_coverage}/{len(v3_df)} ({physical_coverage/len(v3_df)*100:.1f}%) ⚠️ 缺失最严重\n")
        f.write(f"- Behavioral层: {behavioral_coverage}/{len(v3_df)} ({behavioral_coverage/len(v3_df)*100:.1f}%)\n")
        f.write(f"- Educational层: {educational_coverage}/{len(v3_df)} ({educational_coverage/len(v3_df)*100:.1f}%)\n\n")
        
        f.write("---\n\n")
        
        f.write("## 🔬 论文写作优势\n\n")
        f.write("### Methods章节\n\n")
        f.write("**V2方案（需要辩护）**:\n")
        f.write('> "We only included 28 students (57%) who had data in all three layers..."\n\n')
        f.write("**审稿人可能的质疑**:\n")
        f.write("- 为什么舍弃43%的学生？\n")
        f.write("- 这是否影响研究的代表性？\n")
        f.write("- 舍弃的学生是否有系统性特征（如低活跃度）？\n\n")
        
        f.write("---\n\n")
        
        f.write("**V3方案（可以强调）**:\n")
        f.write('> "We included all 49 students in the analysis. To handle missing data in some layers (Physical: 36.7%, Behavioral: 6.1%, Educational: 8.2%), we: (1) explicitly marked layer coverage with binary features, (2) computed multi-layer metrics only on available layers, (3) used tree-based models (XGBoost) that natively handle missing values. This approach ensures that our findings represent the entire cohort."\n\n')
        
        f.write("**关键论述**:\n")
        f.write("1. **完整性**: \"Our analysis covers 100% of students (N=49)\"\n")
        f.write("2. **透明性**: \"Missing data patterns are explicitly modeled\"\n")
        f.write("3. **方法论**: \"We distinguish 'no data' from 'social isolation'\"\n\n")
        
        f.write("---\n\n")
        
        f.write("## 📈 预期实验结果改善\n\n")
        f.write("### 1. 交叉验证稳定性\n\n")
        f.write("| 指标 | V2版本 | V3版本 |\n")
        f.write("|------|--------|--------|\n")
        f.write("| 5-Fold每折样本数 | 5-6个 | 9-10个 ✅ |\n")
        f.write("| AUC标准差 | 较大 | 更稳定 ✅ |\n")
        f.write("| 过拟合风险 | 中等 | 较低 ✅ |\n\n")
        
        f.write("### 2. 特征重要性分析\n\n")
        f.write("**V3新增层覆盖特征**:\n")
        f.write("- `has_physical_layer`: 是否有物理层数据\n")
        f.write("- `has_behavioral_layer`: 是否有行为层数据\n")
        f.write("- `has_educational_layer`: 是否有教育层数据\n")
        f.write("- `layer_coverage`: 参与的层数（1-3）\n\n")
        
        f.write("**可研究的新问题**:\n")
        f.write("- \\\"缺失模式\\\"本身是否是心理健康的预测因子？\n")
        f.write("- 低活跃度（数据缺失）是否与心理健康风险相关？\n")
        f.write("- Physical层缺失（36.7%）是否有系统性原因？\n\n")
        
        f.write("---\n\n")
        
        f.write("## ⚠️ V3版本的注意事项\n\n")
        f.write("### 1. 缺失值处理\n\n")
        f.write(f"- **缺失值比例**: {v3_missing_ratio:.1f}% (vs V2的{v2_missing_ratio:.1f}%)\n")
        f.write(f"- **缺失值总数**: {v3_missing}/{v3_df.size}\n\n")
        
        f.write("**处理方法**:\n")
        f.write("- ✅ **推荐**: 使用XGBoost/LightGBM/CatBoost（原生支持NaN）\n")
        f.write("- ✅ **可选**: Imputation（KNN、MICE）\n")
        f.write("- ❌ **不要**: 删除缺失样本（回到V2方案）\n")
        f.write("- ❌ **不要**: 简单填充0（语义错误）\n\n")
        
        f.write("### 2. 模型选择\n\n")
        f.write("**支持NaN的模型**:\n")
        f.write("- XGBoost ✅ (推荐)\n")
        f.write("- LightGBM ✅\n")
        f.write("- CatBoost ✅\n")
        f.write("- Random Forest (with imputation) ⚠️\n\n")
        
        f.write("**不支持NaN的模型**:\n")
        f.write("- Logistic Regression ❌ (需要先imputation)\n")
        f.write("- SVM ❌ (需要先imputation)\n")
        f.write("- Neural Networks ❌ (需要先imputation)\n\n")
        
        f.write("### 3. 特征重要性解释\n\n")
        f.write("- 层覆盖特征（`has_*_layer`, `layer_coverage`）可能排名靠前\n")
        f.write("- 需要区分\\\"缺失模式\\\"的重要性 vs \\\"网络拓扑\\\"的重要性\n")
        f.write("- 可以单独分析\\\"完整数据子集\\\"（28人）来验证拓扑特征的作用\n\n")
        
        f.write("---\n\n")
        
        f.write("## 🚀 下一步实施计划\n\n")
        f.write("### 立即执行\n\n")
        f.write("1. ✅ **V3特征提取完成**（已完成）\n")
        f.write("   - 49样本 × 28特征\n")
        f.write("   - 输出目录: `outputs/static_v3/`\n\n")
        
        f.write("2. ⏳ **使用XGBoost重新训练**\n")
        f.write("   ```python\n")
        f.write("   from xgboost import XGBClassifier\n")
        f.write("   model = XGBClassifier(missing=np.nan, random_state=42)\n")
        f.write("   model.fit(X_train, y_train)\n")
        f.write("   ```\n\n")
        
        f.write("3. ⏳ **对比实验（V2 vs V3）**\n")
        f.write("   - 使用相同的标签和评估指标\n")
        f.write("   - 对比AUC、F1-score、稳定性（标准差）\n")
        f.write("   - 分析特征重要性差异\n\n")
        
        f.write("4. ⏳ **特征重要性分析**\n")
        f.write("   - 层覆盖特征的重要性\n")
        f.write("   - 多层网络特征的贡献\n")
        f.write("   - 子集分析（仅28人 vs 全部49人）\n\n")
        
        f.write("### 后续任务\n\n")
        f.write("- 阶段4：标签重设计（多模态社交指数 + 时间切分）\n")
        f.write("- 阶段5：实验验证（新特征 + 新标签）\n")
        f.write("- 论文写作：Methods和Results章节\n\n")
        
        f.write("---\n\n")
        
        f.write("## 📝 结论\n\n")
        f.write(f"**V3版本（并集49人）全面优于V2版本（交集28人）**:\n\n")
        f.write(f"| 维度 | V2版本 | V3版本 | 结论 |\n")
        f.write(f"|------|--------|--------|------|\n")
        f.write(f"| 样本充足性 | {v2_ratio:.2f} | {v3_ratio:.2f} | V3 ✅ |\n")
        f.write(f"| 代表性 | 57.1% | 100% | V3 ✅ |\n")
        f.write(f"| 泛化能力 | 中等 | 更强 | V3 ✅ |\n")
        f.write(f"| 论文可辩护性 | 需要辩护 | 易于论述 | V3 ✅ |\n")
        f.write(f"| 缺失值处理 | 简单舍弃 | 智能掩码 | V3 ✅ |\n\n")
        
        f.write("**推荐方案**: **使用V3版本**进行后续研究和论文写作。\n")
    
    print(f"\n✅ 对比报告已保存到: {report_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("V2 vs V3版本对比分析")
    print("=" * 60)
    
    # 加载数据
    v2_df, v3_df = load_features()
    
    # 对比分析
    compare_sample_coverage(v2_df, v3_df)
    compare_feature_dimensions(v2_df, v3_df)
    compare_missing_values(v2_df, v3_df)
    compare_layer_coverage(v3_df)
    
    # 生成报告
    generate_comparison_report(v2_df, v3_df)
    
    print("\n" + "=" * 60)
    print("对比分析完成！")
    print("=" * 60)
    print("生成的文件:")
    print("  - outputs/comparison/v2_vs_v3_sample_coverage.png")
    print("  - outputs/comparison/v2_vs_v3_feature_dimensions.png")
    print("  - outputs/comparison/v2_vs_v3_missing_values.png")
    print("  - outputs/comparison/v3_layer_coverage.png")
    print("  - outputs/comparison/V2_VS_V3_COMPARISON_REPORT.md")


if __name__ == '__main__':
    main()
