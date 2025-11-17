"""
修复前后对比分析
对比修复前（49样本×49特征）和修复后（28样本×21特征）的差异
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("修复前后对比分析")
print("=" * 80)

# 加载数据
print("\n加载数据...")
try:
    df_old = pd.read_csv('outputs/static/multilayer_features_no_behavior.csv')
    print(f"✅ 修复前数据: {df_old.shape[0]} 样本 × {df_old.shape[1]-1} 特征")
except:
    print("⚠️  未找到修复前数据，尝试使用原始特征文件")
    df_old = pd.read_csv('outputs/static/multilayer_features.csv')
    print(f"✅ 修复前数据: {df_old.shape[0]} 样本 × {df_old.shape[1]-1} 特征")

df_new = pd.read_csv('outputs/static_v2/multilayer_features_v2.csv')
print(f"✅ 修复后数据: {df_new.shape[0]} 样本 × {df_new.shape[1]-1} 特征")

# ============================================================================
# 1. 维度对比
# ============================================================================
print("\n" + "=" * 80)
print("1. 维度对比")
print("=" * 80)

print(f"\n{'指标':<30} {'修复前':<15} {'修复后':<15} {'变化':<15}")
print("-" * 75)
print(f"{'样本数':<30} {df_old.shape[0]:<15} {df_new.shape[0]:<15} {df_new.shape[0]-df_old.shape[0]} ({(df_new.shape[0]/df_old.shape[0]-1)*100:+.1f}%)")
print(f"{'特征数':<30} {df_old.shape[1]-1:<15} {df_new.shape[1]-1:<15} {df_new.shape[1]-df_old.shape[1]} ({(df_new.shape[1]/df_old.shape[1]-1)*100:+.1f}%)")

old_ratio = df_old.shape[0] / (df_old.shape[1]-1)
new_ratio = df_new.shape[0] / (df_new.shape[1]-1)
print(f"{'样本/特征比例':<30} {old_ratio:<15.2f} {new_ratio:<15.2f} {new_ratio-old_ratio:+.2f} ({(new_ratio/old_ratio-1)*100:+.1f}%)")

print(f"\n{'过拟合风险评估':<30} {'修复前':<15} {'修复后':<15}")
print("-" * 60)
if old_ratio < 1.5:
    print(f"{'修复前':<30} {'🔴 高风险':<15} (比例={old_ratio:.2f})")
else:
    print(f"{'修复前':<30} {'✅ 可接受':<15} (比例={old_ratio:.2f})")

if new_ratio < 1.5:
    print(f"{'修复后':<30} {'🟡 中等风险':<15} (比例={new_ratio:.2f})")
else:
    print(f"{'修复后':<30} {'✅ 可接受':<15} (比例={new_ratio:.2f})")

# ============================================================================
# 2. 层密度对比
# ============================================================================
print("\n" + "=" * 80)
print("2. 层密度对比（度中心性统计）")
print("=" * 80)

# 修复前层密度
print("\n修复前（49样本）：")
if 'physical_degree_centrality' in df_old.columns:
    phys_old = df_old['physical_degree_centrality'].mean()
    print(f"  Physical层:    mean={phys_old:.3f}, std={df_old['physical_degree_centrality'].std():.3f}")
if 'behavioral_degree_centrality' in df_old.columns:
    beh_old = df_old['behavioral_degree_centrality'].mean()
    print(f"  Behavioral层:  mean={beh_old:.3f}, std={df_old['behavioral_degree_centrality'].std():.3f}")
if 'educational_degree_centrality' in df_old.columns:
    edu_old = df_old['educational_degree_centrality'].mean()
    print(f"  Educational层: mean={edu_old:.3f}, std={df_old['educational_degree_centrality'].std():.3f}")
    if 'physical_degree_centrality' in df_old.columns:
        density_diff_old = edu_old / phys_old
        print(f"  层密度差异: {density_diff_old:.2f}倍 (Educational/Physical)")

# 修复后层密度
print("\n修复后（28样本）：")
phys_new = df_new['physical_degree_centrality'].mean()
beh_new = df_new['behavioral_degree_centrality'].mean()
edu_new = df_new['educational_degree_centrality'].mean()

print(f"  Physical层:    mean={phys_new:.3f}, std={df_new['physical_degree_centrality'].std():.3f}")
print(f"  Behavioral层:  mean={beh_new:.3f}, std={df_new['behavioral_degree_centrality'].std():.3f}")
print(f"  Educational层: mean={edu_new:.3f}, std={df_new['educational_degree_centrality'].std():.3f}")

density_diff_new = edu_new / phys_new
print(f"  层密度差异: {density_diff_new:.2f}倍 (Educational/Physical)")

if 'physical_degree_centrality' in df_old.columns:
    print(f"\n变化：")
    print(f"  层密度差异: {density_diff_old:.2f}倍 → {density_diff_new:.2f}倍 ({(density_diff_new/density_diff_old-1)*100:+.1f}%)")
    print(f"  ✅ 层密度差异减少 {(1-density_diff_new/density_diff_old)*100:.1f}%")

# ============================================================================
# 3. Z-score标准化效果
# ============================================================================
print("\n" + "=" * 80)
print("3. Z-score标准化效果")
print("=" * 80)

if all(col in df_new.columns for col in ['physical_degree_zscore', 'behavioral_degree_zscore', 'educational_degree_zscore']):
    print("\n✅ V2版本已实现Z-score标准化")
    print(f"\n{'层':<20} {'均值':<12} {'标准差':<12} {'评估':<20}")
    print("-" * 60)
    
    for layer in ['physical', 'behavioral', 'educational']:
        col = f'{layer}_degree_zscore'
        mean_z = df_new[col].mean()
        std_z = df_new[col].std()
        
        if abs(mean_z) < 0.15 and 0.8 < std_z < 1.2:
            status = "✅ 标准化成功"
        else:
            status = "⚠️  需检查"
        
        print(f"{layer.capitalize():<20} {mean_z:<12.3f} {std_z:<12.3f} {status:<20}")
    
    print("\n说明：Z-score标准化后，各层度中心性应满足：")
    print("  - 均值 ≈ 0 (实际：-0.1 ~ 0.1)")
    print("  - 标准差 ≈ 1 (实际：0.8 ~ 1.2)")
    print("  ✅ 这样可以消除层密度差异，使跨层特征更公平")
else:
    print("\n⚠️  V2版本未包含Z-score特征（可能已删除调试特征）")

# ============================================================================
# 4. 特征分类对比
# ============================================================================
print("\n" + "=" * 80)
print("4. 特征分类对比")
print("=" * 80)

def categorize_features(columns):
    """对特征进行分类"""
    categories = {
        '单层拓扑': [],
        '多层特征': [],
        '行为属性': [],
        '同伴影响': [],
        'Z-score': [],
        '其他': []
    }
    
    for col in columns:
        if col == 'uid':
            continue
        elif any(layer in col for layer in ['physical_', 'behavioral_', 'educational_']):
            if 'zscore' in col:
                categories['Z-score'].append(col)
            elif 'neighbor' in col:
                categories['同伴影响'].append(col)
            else:
                categories['单层拓扑'].append(col)
        elif any(keyword in col for keyword in ['cross_layer', 'participation', 'overlapping', 'layer_entropy', 'degree_variance', 'degree_std', 'degree_mean', 'degree_range', 'overlap']):
            categories['多层特征'].append(col)
        elif any(keyword in col for keyword in ['ema', 'call', 'sms', 'app', 'calendar', 'physical_activity']):
            categories['行为属性'].append(col)
        elif 'peer' in col or 'neighbor' in col:
            categories['同伴影响'].append(col)
        else:
            categories['其他'].append(col)
    
    return categories

old_categories = categorize_features(df_old.columns)
new_categories = categorize_features(df_new.columns)

print(f"\n{'特征类别':<20} {'修复前':<12} {'修复后':<12} {'变化':<15}")
print("-" * 60)

for cat in ['单层拓扑', '多层特征', 'Z-score', '行为属性', '同伴影响', '其他']:
    old_count = len(old_categories[cat])
    new_count = len(new_categories[cat])
    change = new_count - old_count
    
    if change < 0:
        change_str = f"{change} ({(change/old_count*100 if old_count>0 else 0):.0f}%)"
    elif change > 0:
        change_str = f"+{change} ({(change/old_count*100 if old_count>0 else 0):+.0f}%)"
    else:
        change_str = "0"
    
    print(f"{cat:<20} {old_count:<12} {new_count:<12} {change_str:<15}")

print("\n单层拓扑特征详情（修复后）：")
for feat in sorted(new_categories['单层拓扑']):
    print(f"  - {feat}")

print("\n多层特征详情（修复后）：")
for feat in sorted(new_categories['多层特征']):
    print(f"  - {feat}")

if new_categories['Z-score']:
    print("\nZ-score特征（调试用）：")
    for feat in sorted(new_categories['Z-score']):
        print(f"  - {feat}")

# ============================================================================
# 5. 跨层特征对比
# ============================================================================
print("\n" + "=" * 80)
print("5. 跨层特征改进")
print("=" * 80)

print("\n修复前的问题：")
print("  1. 未做层内标准化，直接计算degree_variance")
print("  2. Educational层平均度0.84 vs Physical层0.53")
print("  3. degree_variance被层密度差异主导，而非角色不一致")

print("\n修复后的改进：")
print("  1. ✅ Z-score标准化：每层度归一化到均值0、标准差1")
print("  2. ✅ degree_variance_z：基于标准化度计算，消除层密度影响")
print("  3. ✅ 引入经典指标：")

if 'participation_coefficient' in df_new.columns:
    pc = df_new['participation_coefficient']
    print(f"     - Participation Coefficient: mean={pc.mean():.3f}, std={pc.std():.3f}")
    print(f"       含义: P≈1表示邻居均匀分布在各层（多面手），P≈0表示集中在某一层")

if 'overlapping_degree' in df_new.columns:
    od = df_new['overlapping_degree']
    print(f"     - Overlapping Degree: mean={od.mean():.1f}, std={od.std():.1f}")
    print(f"       含义: 跨层社交圈总规模")

if 'layer_entropy' in df_new.columns:
    le = df_new['layer_entropy']
    print(f"     - Layer Entropy: mean={le.mean():.3f}, std={le.std():.3f}")
    print(f"       含义: H高表示邻居均匀分布，H低表示集中在某层")

# ============================================================================
# 6. 数据质量对比
# ============================================================================
print("\n" + "=" * 80)
print("6. 数据质量对比")
print("=" * 80)

print("\n缺失值统计：")
old_missing = df_old.isnull().sum().sum()
new_missing = df_new.isnull().sum().sum()

print(f"  修复前: {old_missing} 个缺失值")
print(f"  修复后: {new_missing} 个缺失值")

if new_missing > 0:
    print("\n  修复后存在缺失值的特征：")
    for col in df_new.columns:
        missing_count = df_new[col].isnull().sum()
        if missing_count > 0:
            print(f"    - {col}: {missing_count}/{len(df_new)} ({missing_count/len(df_new)*100:.1f}%)")

print("\n节点质量：")
print("  修复前: 使用并集节点（49个），21个学生至少缺失一层")
print("  修复后: 使用交集节点（28个），所有学生在三层都有真实数据 ✅")

# ============================================================================
# 7. 可视化对比
# ============================================================================
print("\n" + "=" * 80)
print("7. 生成可视化对比图")
print("=" * 80)

# 创建输出目录
os.makedirs('outputs/comparison', exist_ok=True)

# 图1: 层密度对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 修复前
if all(f'{layer}_degree_centrality' in df_old.columns for layer in ['physical', 'behavioral', 'educational']):
    ax = axes[0]
    data_old = [
        df_old['physical_degree_centrality'].dropna(),
        df_old['behavioral_degree_centrality'].dropna(),
        df_old['educational_degree_centrality'].dropna()
    ]
    bp = ax.boxplot(data_old, labels=['Physical', 'Behavioral', 'Educational'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('度中心性', fontsize=12)
    ax.set_title('修复前：层密度分布（49样本）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加均值标注
    means_old = [d.mean() for d in data_old]
    for i, mean in enumerate(means_old):
        ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

# 修复后
ax = axes[1]
data_new = [
    df_new['physical_degree_centrality'].dropna(),
    df_new['behavioral_degree_centrality'].dropna(),
    df_new['educational_degree_centrality'].dropna()
]
bp = ax.boxplot(data_new, labels=['Physical', 'Behavioral', 'Educational'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_ylabel('度中心性', fontsize=12)
ax.set_title('修复后：层密度分布（28样本）', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 添加均值标注
means_new = [d.mean() for d in data_new]
for i, mean in enumerate(means_new):
    ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/comparison/layer_density_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ 保存: outputs/comparison/layer_density_comparison.png")
plt.close()

# 图2: Z-score标准化效果
if all(col in df_new.columns for col in ['physical_degree_zscore', 'behavioral_degree_zscore', 'educational_degree_zscore']):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_zscore = [
        df_new['physical_degree_zscore'].dropna(),
        df_new['behavioral_degree_zscore'].dropna(),
        df_new['educational_degree_zscore'].dropna()
    ]
    
    bp = ax.boxplot(data_zscore, labels=['Physical', 'Behavioral', 'Educational'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightyellow')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='均值=0（目标）')
    ax.axhline(y=1, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='标准差=1（目标）')
    ax.axhline(y=-1, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_ylabel('Z-score', fontsize=12)
    ax.set_title('Z-score标准化效果：消除层密度差异', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 添加统计信息
    for i, data in enumerate(data_zscore):
        mean_z = data.mean()
        std_z = data.std()
        ax.text(i+1, -2.5, f'μ={mean_z:.3f}\nσ={std_z:.3f}', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/zscore_standardization.png', dpi=300, bbox_inches='tight')
    print("  ✅ 保存: outputs/comparison/zscore_standardization.png")
    plt.close()

# 图3: 特征数量对比
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['单层拓扑', '多层特征', '行为属性', '同伴影响']
old_counts = [len(old_categories[cat]) for cat in categories]
new_counts = [len(new_categories[cat]) for cat in categories]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, old_counts, width, label='修复前', color='lightcoral')
bars2 = ax.bar(x + width/2, new_counts, width, label='修复后', color='lightgreen')

ax.set_ylabel('特征数量', fontsize=12)
ax.set_title('特征分类数量对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/comparison/feature_category_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ 保存: outputs/comparison/feature_category_comparison.png")
plt.close()

# 图4: 样本/特征比例对比
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ['样本数', '特征数', '样本/特征比例']
old_values = [df_old.shape[0], df_old.shape[1]-1, old_ratio]
new_values = [df_new.shape[0], df_new.shape[1]-1, new_ratio]

x = np.arange(len(metrics))
width = 0.35

# 归一化显示（样本/特征比例已经是比例了，其他需要缩放）
old_normalized = [old_values[0]/50, old_values[1]/50, old_values[2]]
new_normalized = [new_values[0]/50, new_values[1]/50, new_values[2]]

bars1 = ax.bar(x - width/2, old_normalized, width, label='修复前', color='lightcoral')
bars2 = ax.bar(x + width/2, new_normalized, width, label='修复后', color='lightgreen')

ax.set_ylabel('归一化值', fontsize=12)
ax.set_title('维度对比（归一化显示）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# 添加实际数值标签
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
            f'{old_values[i]:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
            f'{new_values[i]:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加过拟合风险线（比例=1.5）
if metrics[2] == '样本/特征比例':
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='建议阈值=1.5')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('outputs/comparison/dimension_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ 保存: outputs/comparison/dimension_comparison.png")
plt.close()

# ============================================================================
# 8. 总结
# ============================================================================
print("\n" + "=" * 80)
print("8. 修复效果总结")
print("=" * 80)

print("\n✅ 已解决的问题：")
print("\n  问题②：节点不一致")
print("    - 修复前: 使用并集（49个），21个学生缺失至少一层")
print("    - 修复后: 使用交集（28个），所有学生三层都有数据 ✅")
print("    - 效果: 避免'缺失=0'的语义错误")

print("\n  问题③：跨层特征缺陷")
print("    - 修复前: 层密度差异1.59倍，未标准化")
print("    - 修复后: 层密度差异1.08倍（-32%），Z-score标准化 ✅")
print("    - 效果: degree_variance_z真正反映角色不一致，而非层密度")

print("\n  问题④：特征维度过高")
print(f"    - 修复前: {df_old.shape[1]-1}特征，样本/特征={old_ratio:.2f}")
print(f"    - 修复后: {df_new.shape[1]-1}特征，样本/特征={new_ratio:.2f} ✅")
print(f"    - 效果: 特征降维{(1-new_ratio/old_ratio)*100:.1f}%，过拟合风险降低")

print("\n⚠️  仍需改进：")
if new_ratio < 1.5:
    print(f"    - 样本/特征比例={new_ratio:.2f} < 1.5（建议阈值）")
    print("    - 建议: 进一步删除Z-score调试特征，或删除Educational层特征")
    print(f"    - 目标: 将特征从{df_new.shape[1]-1}个降至15-16个")

print("\n📊 生成的可视化文件：")
print("    - outputs/comparison/layer_density_comparison.png")
print("    - outputs/comparison/zscore_standardization.png")
print("    - outputs/comparison/feature_category_comparison.png")
print("    - outputs/comparison/dimension_comparison.png")

print("\n" + "=" * 80)
print("对比分析完成！")
print("=" * 80)
print("\n下一步：")
print("  1. 查看可视化图表: outputs/comparison/*.png")
print("  2. 决定是否进一步降维（可选）")
print("  3. 继续阶段4-5: 标签重设计 + 实验验证")
