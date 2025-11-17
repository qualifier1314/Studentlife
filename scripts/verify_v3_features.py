import pandas as pd
import numpy as np

df = pd.read_csv('outputs/static_v3/multilayer_features_v3.csv', index_col='uid')

print('='*60)
print('V3特征数据质量验证')
print('='*60)

print(f'\n1. 基本信息:')
print(f'   样本数: {len(df)}')
print(f'   特征数: {len(df.columns)}')
print(f'   样本/特征比例: {len(df)/len(df.columns):.2f}')

print(f'\n2. 层覆盖特征统计:')
coverage_cols = ['has_physical_layer', 'has_behavioral_layer', 'has_educational_layer', 'layer_coverage']
print(df[coverage_cols].describe())

print(f'\n3. 层覆盖分布:')
print(df['layer_coverage'].value_counts().sort_index())

print(f'\n4. 缺失值最多的特征（Top 5）:')
missing = df.isnull().sum().sort_values(ascending=False).head(5)
for col, count in missing.items():
    print(f'   {col}: {count}/{len(df)} ({count/len(df)*100:.1f}%)')

print(f'\n5. Z-score特征验证（应该均值≈0, 标准差≈1）:')
zscore_cols = [c for c in df.columns if '_z' in c]
if zscore_cols:
    for col in zscore_cols[:3]:
        vals = df[col].dropna()
        print(f'   {col}: mean={vals.mean():.3f}, std={vals.std():.3f}')

print('\n✅ V3特征数据验证完成！')
