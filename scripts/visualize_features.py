"""
特征可视化脚本
生成特征分布图、相关性热力图等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_distributions(df, output_dir='outputs/static/figures'):
    """绘制主要特征的分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择重要特征
    important_features = [
        'physical_degree_centrality',
        'behavioral_degree_centrality',
        'educational_degree_centrality',
        'cross_layer_activity',
        'degree_variance',
        'total_unique_neighbors'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    feature_names_cn = {
        'physical_degree_centrality': '物理层度中心性',
        'behavioral_degree_centrality': '行为层度中心性',
        'educational_degree_centrality': '教育层度中心性',
        'cross_layer_activity': '跨层活跃度',
        'degree_variance': '层间度差异',
        'total_unique_neighbors': '总邻居数'
    }
    
    for idx, feature in enumerate(important_features):
        ax = axes[idx]
        df[feature].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(feature_names_cn.get(feature, feature), fontsize=12)
        ax.set_xlabel('值', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 特征分布图已保存到: {output_dir}/feature_distributions.png")
    plt.close()


def plot_correlation_heatmap(df, output_dir='outputs/static/figures'):
    """绘制特征相关性热力图"""
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算相关矩阵
    corr_matrix = df[numeric_cols].corr()
    
    # 绘制热力图（使用matplotlib的imshow）
    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(numeric_cols, fontsize=6)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('相关系数', fontsize=10)
    
    plt.title('特征相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 相关性热力图已保存到: {output_dir}/feature_correlation_heatmap.png")
    plt.close()


def plot_layer_comparison(df, output_dir='outputs/static/figures'):
    """对比三层网络的中心性"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = ['physical', 'behavioral', 'educational']
    layer_names = ['物理层', '行为层', '教育层']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    x = np.arange(len(df))
    width = 0.25
    
    for idx, (layer, name, color) in enumerate(zip(layers, layer_names, colors)):
        feature = f'{layer}_degree_centrality'
        values = df[feature].values
        ax.bar(x + idx * width, values, width, label=name, color=color, alpha=0.8)
    
    ax.set_xlabel('学生ID', fontsize=12)
    ax.set_ylabel('度中心性', fontsize=12)
    ax.set_title('三层网络度中心性对比', fontsize=14, pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{i+1}' for i in range(len(df))], rotation=90, fontsize=6)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_centrality_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 层对比图已保存到: {output_dir}/layer_centrality_comparison.png")
    plt.close()


def plot_multilayer_features(df, output_dir='outputs/static/figures'):
    """绘制多层网络特征的散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 跨层活跃度 vs 总邻居数
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['cross_layer_activity'], df['total_unique_neighbors'], 
                           c=df['degree_mean'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax1.set_xlabel('跨层活跃度', fontsize=12)
    ax1.set_ylabel('总邻居数', fontsize=12)
    ax1.set_title('跨层活跃度 vs 总邻居数', fontsize=13)
    ax1.grid(alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('平均度中心性', fontsize=10)
    
    # 层间度差异 vs 邻居重叠度
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['degree_variance'], df['phys_edu_neighbor_overlap'], 
                           c=df['cross_layer_activity'], cmap='plasma', s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('层间度差异', fontsize=12)
    ax2.set_ylabel('物理-教育层邻居重叠度', fontsize=12)
    ax2.set_title('层间度差异 vs 邻居重叠度', fontsize=13)
    ax2.grid(alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('跨层活跃度', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multilayer_features_scatter.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 多层特征散点图已保存到: {output_dir}/multilayer_features_scatter.png")
    plt.close()


def generate_feature_summary(df, output_dir='outputs/static'):
    """生成特征摘要报告"""
    summary_file = os.path.join(output_dir, 'feature_summary_report.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("多层网络特征提取报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"节点数量: {len(df)}\n")
        f.write(f"特征数量: {len(df.columns)}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("1. 网络中心性统计\n")
        f.write("=" * 70 + "\n\n")
        
        # 找出每层的Top-5中心节点
        for layer in ['physical', 'behavioral', 'educational']:
            f.write(f"\n{layer.capitalize()}层 Top-5 中心节点:\n")
            f.write("-" * 50 + "\n")
            top5 = df.nlargest(5, f'{layer}_degree_centrality')[[f'{layer}_degree_centrality', f'{layer}_degree']]
            for idx, (uid, row) in enumerate(top5.iterrows(), 1):
                f.write(f"  {idx}. {uid}: 度中心性={row[f'{layer}_degree_centrality']:.3f}, 度={row[f'{layer}_degree']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("2. 跨层活跃度分析\n")
        f.write("=" * 70 + "\n\n")
        
        activity_counts = df['cross_layer_activity'].value_counts().sort_index()
        f.write("跨层活跃度分布:\n")
        f.write("-" * 50 + "\n")
        for activity, count in activity_counts.items():
            percentage = count / len(df) * 100
            f.write(f"  活跃于 {int(activity)} 层: {count} 人 ({percentage:.1f}%)\n")
        
        # 找出跨层活跃度最高的节点
        f.write("\n跨层活跃度最高的Top-10节点:\n")
        f.write("-" * 50 + "\n")
        top_active = df.nlargest(10, 'cross_layer_activity')[['cross_layer_activity', 'total_unique_neighbors', 'degree_mean']]
        for idx, (uid, row) in enumerate(top_active.iterrows(), 1):
            f.write(f"  {idx}. {uid}: 活跃度={int(row['cross_layer_activity'])}, 总邻居={int(row['total_unique_neighbors'])}, 平均度={row['degree_mean']:.3f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("3. 节点类型识别\n")
        f.write("=" * 70 + "\n\n")
        
        # 识别通才型（所有层都高度中心）
        generalists = df[(df['physical_degree_centrality'] > 0.7) & 
                        (df['behavioral_degree_centrality'] > 0.2) & 
                        (df['educational_degree_centrality'] > 0.8)]
        f.write(f"通才型节点（所有层都中心）: {len(generalists)} 个\n")
        if len(generalists) > 0:
            f.write("  节点: " + ", ".join(generalists.index.tolist()) + "\n")
        
        # 识别专才型（只在单层中心）
        specialists_phys = df[(df['physical_degree_centrality'] > 0.7) & 
                             (df['behavioral_degree_centrality'] < 0.3) & 
                             (df['educational_degree_centrality'] < 0.5)]
        f.write(f"\n专才型节点-物理层: {len(specialists_phys)} 个\n")
        if len(specialists_phys) > 0:
            f.write("  节点: " + ", ".join(specialists_phys.index.tolist()) + "\n")
        
        # 识别边缘型（所有层都边缘）
        peripherals = df[(df['physical_degree_centrality'] < 0.3) & 
                        (df['behavioral_degree_centrality'] < 0.2) & 
                        (df['educational_degree_centrality'] < 0.5)]
        f.write(f"\n边缘型节点（所有层都边缘）: {len(peripherals)} 个\n")
        if len(peripherals) > 0:
            f.write("  节点: " + ", ".join(peripherals.index.tolist()) + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("4. 邻居重叠度分析\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"物理-教育层邻居重叠度均值: {df['phys_edu_neighbor_overlap'].mean():.3f}\n")
        f.write(f"物理-行为层邻居重叠度均值: {df['phys_beh_neighbor_overlap'].mean():.3f}\n")
        
        high_overlap = df.nlargest(5, 'phys_edu_neighbor_overlap')[['phys_edu_neighbor_overlap', 'total_unique_neighbors']]
        f.write("\n邻居重叠度最高的Top-5节点（物理-教育）:\n")
        f.write("-" * 50 + "\n")
        for idx, (uid, row) in enumerate(high_overlap.iterrows(), 1):
            f.write(f"  {idx}. {uid}: 重叠度={row['phys_edu_neighbor_overlap']:.3f}, 总邻居={int(row['total_unique_neighbors'])}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ 特征摘要报告已保存到: {summary_file}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("特征可视化与分析")
    print("=" * 60)
    
    # 加载特征数据
    features_file = 'outputs/static/multilayer_features.csv'
    print(f"\n加载特征数据: {features_file}")
    df = pd.read_csv(features_file, index_col='uid')
    print(f"✓ 加载完成: {len(df)} 个节点, {len(df.columns)} 个特征")
    
    # 创建输出目录
    output_dir = 'outputs/static/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种可视化
    print("\n生成可视化图表...")
    print("-" * 60)
    plot_feature_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_layer_comparison(df, output_dir)
    plot_multilayer_features(df, output_dir)
    
    # 生成摘要报告
    print("\n生成特征摘要报告...")
    print("-" * 60)
    generate_feature_summary(df)
    
    print("\n" + "=" * 60)
    print("所有可视化和分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
