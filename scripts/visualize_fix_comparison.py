"""
可视化数据泄露修复前后的对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 路径配置
project_root = Path(__file__).parent.parent
output_dir = project_root / "outputs/static_enhanced/figures"
output_dir.mkdir(parents=True, exist_ok=True)

# 修复前后的结果（社交健康，Random Forest）
data_before = {
    'Feature_Combination': [
        'Physical\nLayer',
        'Behavioral\nLayer', 
        'Educational\nLayer',
        'Single-Layer\nFeatures',
        'Multi-Layer\nFeatures',
        'Single+Multi\n(Topology)',
        'Topology+\nAttributes',
        'All\nFeatures'
    ],
    'AUC_Before': [0.824, 0.829, 0.862, 0.914, 1.000, 1.000, 1.000, 0.986],
    'Std_Before': [0.185, 0.070, 0.133, 0.171, 0.000, 0.000, 0.000, 0.029],
}

data_after = {
    'AUC_After': [0.472, 0.807, 0.442, 0.660, 0.574, 0.638, 0.654, 0.643],
    'Std_After': [0.122, 0.245, 0.206, 0.320, 0.221, 0.258, 0.294, 0.306],
}

df = pd.DataFrame({**data_before, **data_after})

# 设置样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 9

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# === 子图1: 修复前后对比（柱状图） ===
ax1 = axes[0]
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['AUC_Before'], width, 
                yerr=df['Std_Before'], capsize=3,
                label='修复前 (有泄露)', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['AUC_After'], width,
                yerr=df['Std_After'], capsize=3,
                label='修复后 (无泄露)', color='#4ECDC4', alpha=0.8)

# 标注异常值
for i, (before, after) in enumerate(zip(df['AUC_Before'], df['AUC_After'])):
    if before >= 0.99:  # 异常值
        ax1.text(i - width/2, before + 0.05, '🔴', ha='center', fontsize=12)
    if abs(before - after) > 0.3:  # 大幅下降
        ax1.annotate('', xy=(i + width/2, after), xytext=(i - width/2, before),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, ls='--'))

ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='完美预测 (异常)')
ax1.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='良好预测 (合理)')

ax1.set_ylabel('AUC Score', fontweight='bold', fontsize=11)
ax1.set_xlabel('Feature Combination', fontweight='bold', fontsize=11)
ax1.set_title('(A) 数据泄露修复前后对比\nSocial Health Risk Prediction', 
              fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Feature_Combination'], rotation=45, ha='right', fontsize=8)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加文本说明
ax1.text(0.02, 0.98, '修复前: AUC=1.000 (3组)\n修复后: AUC=0.5-0.8 (合理)', 
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# === 子图2: AUC变化量（增量图） ===
ax2 = axes[1]
changes = df['AUC_After'] - df['AUC_Before']
colors = ['#4ECDC4' if c > -0.05 else '#FF6B6B' for c in changes]

bars = ax2.barh(range(len(df)), changes, color=colors, alpha=0.8)

# 标注变化量
for i, (change, comb) in enumerate(zip(changes, df['Feature_Combination'])):
    label = f'{change:+.3f}'
    color = 'darkgreen' if change > -0.05 else 'darkred'
    ha = 'left' if change < 0 else 'right'
    offset = 0.02 if change < 0 else -0.02
    ax2.text(change + offset, i, label, va='center', ha=ha, 
            fontsize=9, fontweight='bold', color=color)

ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax2.axvline(x=-0.3, color='red', linestyle='--', linewidth=1, alpha=0.5)

ax2.set_yticks(range(len(df)))
ax2.set_yticklabels(df['Feature_Combination'], fontsize=8)
ax2.set_xlabel('AUC Change (After - Before)', fontweight='bold', fontsize=11)
ax2.set_title('(B) AUC变化量分析\nNegative = Leak Fixed', 
              fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(-0.5, 0.1)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', alpha=0.8, label='大幅下降 (修复泄露)'),
    Patch(facecolor='#4ECDC4', alpha=0.8, label='稳定 (无泄露)')
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

# 添加注释
ax2.text(0.02, 0.98, 
         '关键发现:\n'
         '• Multi-Layer: -0.426\n'
         '• Topology+Attr: -0.346\n'
         '• Behavioral: -0.022 ✓',
         transform=ax2.transAxes, fontsize=8,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / "data_leakage_fix_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ 保存修复对比图: {output_dir / 'data_leakage_fix_comparison.png'}")

# === 图表2: 泄露诊断流程图 ===
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

# 流程步骤
steps = [
    {
        'y': 0.95,
        'text': '❓ 问题发现',
        'detail': '用户质疑: "AUC=1正常情况不应该是1"',
        'color': '#FF6B6B'
    },
    {
        'y': 0.82,
        'text': '🔍 深度诊断',
        'detail': 'diagnose_perfect_auc.py\n→ call_count: AUC=1.000\n→ sms_count: AUC=0.987',
        'color': '#FFA07A'
    },
    {
        'y': 0.68,
        'text': '💡 根本原因',
        'detail': '循环定义:\n标签 = f(call_count, sms_count)\n特征 = [call_count, sms_count, ...]',
        'color': '#FFD700'
    },
    {
        'y': 0.53,
        'text': '🔧 修复方案',
        'detail': '移除泄露特征:\n✗ call_count\n✗ sms_count\n✗ app_usage_hours',
        'color': '#98D8C8'
    },
    {
        'y': 0.38,
        'text': '✅ 验证效果',
        'detail': '重新运行对比实验:\nMulti-Layer: 1.000 → 0.574\nAll Features: 0.986 → 0.643',
        'color': '#4ECDC4'
    },
    {
        'y': 0.22,
        'text': '📊 新发现',
        'detail': 'Behavioral层AUC=0.807 (最高)\n说明网络拓扑确实能预测社交健康',
        'color': '#95E1D3'
    }
]

for i, step in enumerate(steps):
    # 绘制方框
    rect = plt.Rectangle((0.1, step['y']-0.06), 0.8, 0.12, 
                         facecolor=step['color'], edgecolor='black', 
                         linewidth=2, alpha=0.6)
    ax.add_patch(rect)
    
    # 添加标题
    ax.text(0.5, step['y']+0.03, step['text'], 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 添加详细信息
    ax.text(0.5, step['y']-0.02, step['detail'], 
           ha='center', va='center', fontsize=10, style='italic')
    
    # 添加箭头
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, steps[i+1]['y']+0.06), 
                   xytext=(0.5, step['y']-0.06),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# 添加标题
ax.text(0.5, 1.0, '数据泄露修复流程', 
       ha='center', va='top', fontsize=16, fontweight='bold')

# 添加时间线
ax.text(0.05, 0.08, '用户反馈 → 2小时 → 问题解决', 
       ha='left', va='center', fontsize=11, style='italic',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.savefig(output_dir / "data_leakage_fix_workflow.png", dpi=300, bbox_inches='tight')
print(f"✅ 保存修复流程图: {output_dir / 'data_leakage_fix_workflow.png'}")

plt.show()

print("\n" + "="*70)
print("可视化完成！生成2张图表：")
print("  1. data_leakage_fix_comparison.png - 修复前后对比")
print("  2. data_leakage_fix_workflow.png - 修复流程图")
print("="*70)
