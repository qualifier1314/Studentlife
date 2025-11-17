"""
深度诊断：为什么AUC仍然是1.000？

可能原因：
1. 标签定义使用call_count/sms_count，而这些特征高度相关behavioral_degree
2. Behavioral网络本身是从通话/短信数据构建的 → 完全泄露
3. 样本量太小（11个正类）导致完全可分

解决方案：
- 检查Behavioral层的构建逻辑
- 如果Behavioral层=通话网络，那么behavioral_degree就是call_count的函数
- 需要移除Behavioral层特征，或重新定义标签
"""

import pandas as pd
import numpy as np


def diagnose_leakage():
    """诊断数据泄露"""
    print("=" * 70)
    print("深度诊断：AUC=1.000的根本原因")
    print("=" * 70)
    
    # 加载数据
    features = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
    labels = pd.read_csv('outputs/static/prediction_labels.csv', index_col='uid')
    
    # 合并
    df = features.copy()
    df['social_risk'] = labels['social_health_risk']
    
    # 1. 检查call_count与网络特征的相关性
    print("\n【1】call_count/sms_count 与网络度的相关性:")
    print("-" * 70)
    
    degree_cols = ['physical_degree', 'behavioral_degree', 'educational_degree']
    
    for feat in ['call_count', 'sms_count']:
        print(f"\n{feat}:")
        for deg in degree_cols:
            if feat in df.columns and deg in df.columns:
                corr = df[feat].corr(df[deg])
                print(f"  vs {deg}: {corr:+.3f} {'🔴 极高相关' if abs(corr) > 0.8 else ('🟡 高相关' if abs(corr) > 0.5 else '✅ 低相关')}")
    
    # 2. 检查标签与特征的分离度
    print("\n\n【2】高风险 vs 低风险组的特征分布:")
    print("-" * 70)
    
    high_risk = df[df['social_risk'] == 1]
    low_risk = df[df['social_risk'] == 0]
    
    key_features = ['cross_layer_activity', 'total_unique_neighbors', 
                   'behavioral_degree', 'call_count']
    
    print(f"\n样本数: 高风险={len(high_risk)}, 低风险={len(low_risk)}")
    print(f"\n特征分布对比:")
    print(f"{'特征':<30} {'高风险均值':<15} {'低风险均值':<15} {'差异':<10} {'完全分离?'}")
    print("-" * 80)
    
    for feat in key_features:
        if feat in df.columns:
            hr_mean = high_risk[feat].mean()
            lr_mean = low_risk[feat].mean()
            diff = lr_mean - hr_mean
            
            # 检查是否完全分离（两组无重叠）
            hr_max = high_risk[feat].max()
            lr_min = low_risk[feat].min()
            separated = hr_max < lr_min
            
            print(f"{feat:<30} {hr_mean:<15.1f} {lr_mean:<15.1f} {diff:<10.1f} {'🔴 是' if separated else '✅ 否'}")
    
    # 3. 检查多层特征的唯一值数量
    print("\n\n【3】多层特征的唯一值数量（如果太少说明特征不够区分）:")
    print("-" * 70)
    
    multilayer_cols = [c for c in df.columns if 'cross' in c or 'overlap' in c or 'total_unique' in c]
    
    for feat in multilayer_cols:
        unique_count = df[feat].nunique()
        print(f"{feat:<40} {unique_count} 个唯一值")
    
    # 4. 检查Behavioral层的构建逻辑（推测）
    print("\n\n【4】Behavioral层构建逻辑推测:")
    print("-" * 70)
    
    if 'behavioral_degree' in df.columns and 'call_count' in df.columns:
        corr_call = df['call_count'].corr(df['behavioral_degree'])
        corr_sms = df['sms_count'].corr(df['behavioral_degree'])
        
        print(f"behavioral_degree vs call_count: {corr_call:+.3f}")
        print(f"behavioral_degree vs sms_count: {corr_sms:+.3f}")
        
        if abs(corr_call) > 0.8 or abs(corr_sms) > 0.8:
            print("\n🔴 警告：Behavioral层可能直接由通话/短信网络构建！")
            print("   → 标签基于call/sms → behavioral_degree基于call/sms → 完全泄露")
        else:
            print("\n✅ Behavioral层与call/sms相关性不高，不是主要泄露源")
    
    # 5. 找出完美预测的特征
    print("\n\n【5】寻找能完美预测标签的单个特征:")
    print("-" * 70)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    
    X_social = df['social_risk'].values
    
    perfect_features = []
    for feat in df.columns:
        if feat in ['social_risk', 'mental_health_risk', 'academic_risk']:
            continue
        
        try:
            X = df[[feat]].values
            
            # 训练简单决策树
            clf = DecisionTreeClassifier(max_depth=1, random_state=42)
            clf.fit(X, X_social)
            y_pred = clf.predict_proba(X)[:, 1]
            
            auc = roc_auc_score(X_social, y_pred)
            
            if auc > 0.95 or auc < 0.05:  # 非常高或非常低（反向预测）
                perfect_features.append((feat, auc))
                print(f"  {feat:<40} AUC={auc:.3f} 🔴 完美预测")
        except:
            pass
    
    # 6. 给出修复建议
    print("\n\n【6】修复建议:")
    print("=" * 70)
    
    if len(perfect_features) > 0:
        print(f"\n发现 {len(perfect_features)} 个能完美预测标签的特征：")
        for feat, auc in sorted(perfect_features, key=lambda x: abs(x[1]-0.5), reverse=True)[:5]:
            print(f"  - {feat} (AUC={auc:.3f})")
        
        print("\n🔴 建议方案A（推荐）：")
        print("  1. 从预测特征中移除这些泄露特征")
        print("  2. 仅使用Physical和Educational层特征 + 多层特征")
        print("  3. 移除所有Behavioral层特征（如果它由call/sms构建）")
        
        print("\n🟡 建议方案B（备选）：")
        print("  1. 完全重新定义标签")
        print("  2. 使用外部问卷数据（如loneliness, social_support）")
        print("  3. 如果没有问卷，改为无监督聚类任务")
        
        print("\n🟢 建议方案C（折中）：")
        print("  1. 接受AUC=1.000，但在论文中说明：")
        print("     '标签基于通信频率（call/sms），")
        print("      网络特征能完美预测是因为它们捕捉了相同的社交行为维度'")
        print("  2. 强调这是'社交行为模式识别'而非'预测'")
        print("  3. 改变论文定位：从'预测'变为'表征学习'")


if __name__ == '__main__':
    diagnose_leakage()
