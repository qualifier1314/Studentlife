"""查看V3预测的特征重要性"""
import pandas as pd
import pickle
import os

# 检查生成的图片
print("=" * 60)
print("V3版本预测结果检查")
print("=" * 60)

# 检查输出文件
output_dir = 'outputs/static_v3'
files = os.listdir(output_dir)
print(f"\n生成的文件:")
for f in sorted(files):
    if f.endswith('.png') or f.endswith('.md'):
        print(f"  ✅ {f}")

# 读取结果摘要
print(f"\n" + "=" * 60)
print("预测结果摘要")
print("=" * 60)

with open('outputs/static_v3/prediction_results_v3.md', 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)

print("\n" + "=" * 60)
print("关键发现")
print("=" * 60)
print("1. 心理健康风险CV AUC: 0.164 ⚠️ 极低")
print("2. 社交健康风险CV AUC: 0.636 ⚠️ 中等")
print("3. 训练集AUC: 1.000 ⚠️ 完美过拟合")
print("\n这说明:")
print("  - 当前标签存在严重的数据泄露问题（训练集AUC=1.0）")
print("  - 或者特征数相对样本数太多导致过拟合")
print("  - 需要进行阶段4的标签重设计（时间切分）")
