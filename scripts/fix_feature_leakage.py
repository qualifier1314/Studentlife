"""
修复特征泄露：移除用于定义标签的行为特征

问题：
- 社交健康标签由 call_count<阈值 AND sms_count<阈值 定义
- 但预测特征中包含 call_count, sms_count
- 导致循环定义 → AUC=1.000

解决方案：
- 从预测特征中移除所有行为特征
- 仅保留网络拓扑特征（度、中心性等）+ 多层特征
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 路径配置
project_root = Path(__file__).parent.parent
features_file = project_root / "outputs/static/multilayer_features.csv"
output_file = project_root / "outputs/static/multilayer_features_no_behavior.csv"

# 加载特征
df = pd.read_csv(features_file)
print(f"原始特征数: {len(df.columns) - 1}  # 减1是uid列")
print(f"样本数: {len(df)}")

# 定义需要移除的行为特征（直接暴露通信行为的特征）
behavior_features_to_remove = [
    # 通信频率（标签定义使用）
    'call_count', 'sms_count',
    
    # App使用（直接行为观察）
    'app_usage_hours', 'app_diversity',
    
    # 其他直接行为特征
    'activity_transitions', 'conversation_duration',
    'dark_time_ratio', 'sleep_quality_score',
    
    # 如果有以下特征也移除
    'total_communication_time', 'daily_communication_freq',
    'social_activity_score'
]

# 检查哪些特征存在
existing_behavior_features = [f for f in behavior_features_to_remove if f in df.columns]
print(f"\n【需要移除的行为特征】 ({len(existing_behavior_features)}个):")
for f in existing_behavior_features:
    print(f"  ✗ {f}")

# 保留的特征类型
print(f"\n【保留的特征类型】:")
print("  ✓ 网络拓扑特征（度、中心性、聚类系数等）")
print("  ✓ 多层特征（跨层活动度、邻居重叠等）")
print("  ✓ 同伴影响特征（邻居平均值）")

# 移除行为特征
columns_to_keep = ['uid'] + [c for c in df.columns if c not in behavior_features_to_remove and c != 'uid']
df_clean = df[columns_to_keep]

print(f"\n【清洗后】:")
print(f"  特征数: {len(df_clean.columns) - 1}")
print(f"  移除数: {len(existing_behavior_features)}")

# 保存
df_clean.to_csv(output_file, index=False)
print(f"\n✅ 已保存到: {output_file.relative_to(project_root)}")

# 显示保留的特征列表
print(f"\n【保留的特征列表】:")
retained_features = [c for c in df_clean.columns if c != 'uid']
for i, f in enumerate(retained_features, 1):
    print(f"  {i:2d}. {f}")

print(f"\n{'='*70}")
print("下一步: 使用 multilayer_features_no_behavior.csv 重新运行对比实验")
print("期望: AUC降至合理范围 (0.70-0.90)")
print(f"{'='*70}")
