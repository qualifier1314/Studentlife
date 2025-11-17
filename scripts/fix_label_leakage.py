"""
重新生成社交健康风险标签（修复数据泄露问题）

问题诊断：
- 原标签生成使用了网络拓扑特征（degree_centrality, cross_layer_activity）
- 这些特征又被用于预测，导致数据泄露
- 结果：AUC=1.000（异常）

解决方案：
- 使用行为数据（call_count, sms_count）定义标签
- 这些是独立的行为指标，不是从网络拓扑计算的
- 定义：通话和短信都低于25%分位数 → 社交行为极低 → 高风险
"""

import pandas as pd
import numpy as np
import os


def generate_social_health_labels_no_leakage():
    """
    生成社交健康风险标签（无数据泄露版本）
    
    定义基于行为数据而非网络拓扑：
    - 通话次数低于25%分位数
    - 短信次数低于25%分位数
    - 两者同时满足 → 社交行为极低 → 高风险
    """
    print("\n" + "=" * 70)
    print("重新生成社交健康风险标签（无数据泄露）")
    print("=" * 70)
    
    # 加载特征
    features_df = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
    
    # 检查行为特征是否存在
    required_features = ['call_count', 'sms_count']
    for feat in required_features:
        if feat not in features_df.columns:
            print(f"  ❌ 缺少特征: {feat}")
            return None
    
    print(f"\n使用行为数据定义标签（独立于网络拓扑）:")
    print(f"  - call_count: 通话次数（直接从call_log数据计算）")
    print(f"  - sms_count: 短信次数（直接从sms数据计算）")
    print(f"  - 定义: 两者都低于25%分位数 → 社交行为极低 → 高风险")
    
    # 计算阈值
    call_threshold = features_df['call_count'].quantile(0.25)
    sms_threshold = features_df['sms_count'].quantile(0.25)
    
    print(f"\n阈值:")
    print(f"  - call_count < {call_threshold:.0f} 次/学期")
    print(f"  - sms_count < {sms_threshold:.0f} 条/学期")
    
    # 生成标签
    labels = {}
    for uid in features_df.index:
        call_count = features_df.loc[uid, 'call_count']
        sms_count = features_df.loc[uid, 'sms_count']
        
        # 高风险条件：通话和短信都极低
        if call_count < call_threshold and sms_count < sms_threshold:
            labels[uid] = 1
        else:
            labels[uid] = 0
    
    # 统计
    risk_count = sum(labels.values())
    risk_rate = risk_count / len(labels) * 100
    
    print(f"\n生成结果:")
    print(f"  - 总学生数: {len(labels)}")
    print(f"  - 高风险学生: {risk_count} ({risk_rate:.1f}%)")
    print(f"  - 低风险学生: {len(labels) - risk_count} ({100 - risk_rate:.1f}%)")
    
    # 显示高风险学生的行为数据
    if risk_count > 0 and risk_count <= 20:
        print(f"\n高风险学生行为数据:")
        for uid in features_df.index:
            if labels[uid] == 1:
                call = features_df.loc[uid, 'call_count']
                sms = features_df.loc[uid, 'sms_count']
                print(f"  - {uid}: call={call:.0f}, sms={sms:.0f}")
    
    return labels


def verify_no_leakage(labels):
    """
    验证标签与网络拓扑特征的相关性
    如果相关性很高，说明仍有泄露
    """
    print("\n" + "=" * 70)
    print("数据泄露检验")
    print("=" * 70)
    
    features_df = pd.read_csv('outputs/static/multilayer_features.csv', index_col='uid')
    
    # 将标签添加到DataFrame
    features_df['label'] = features_df.index.map(labels)
    
    # 检查与网络拓扑特征的相关性
    topology_features = [
        'physical_degree_centrality',
        'behavioral_degree_centrality',
        'educational_degree_centrality',
        'cross_layer_activity',
        'total_unique_neighbors'
    ]
    
    print("\n标签与网络拓扑特征的相关性（应该<0.5）:")
    
    high_corr_features = []
    for feat in topology_features:
        if feat in features_df.columns:
            corr = features_df['label'].corr(features_df[feat])
            abs_corr = abs(corr)
            
            if abs_corr < 0.3:
                status = "✅ 安全"
            elif abs_corr < 0.5:
                status = "🟡 中等"
            else:
                status = "🔴 高风险"
                high_corr_features.append((feat, corr))
            
            print(f"  {feat}: {corr:+.3f} {status}")
    
    # 检查与行为特征的相关性
    behavioral_features = ['call_count', 'sms_count', 'ema_mood_mean', 'ema_stress_mean']
    
    print("\n标签与行为特征的相关性（预期较高，因为用它们定义标签）:")
    for feat in behavioral_features:
        if feat in features_df.columns:
            corr = features_df['label'].corr(features_df[feat])
            print(f"  {feat}: {corr:+.3f}")
    
    # 结论
    print("\n泄露检验结论:")
    if len(high_corr_features) == 0:
        print("  ✅ 未发现严重数据泄露（所有拓扑特征相关性<0.5）")
        print("  ✅ 可以安全使用拓扑特征进行预测")
        return True
    else:
        print(f"  🔴 发现{len(high_corr_features)}个高相关拓扑特征：")
        for feat, corr in high_corr_features:
            print(f"     - {feat}: {corr:+.3f}")
        print("  ⚠️  建议移除这些特征或重新定义标签")
        return False


def update_labels_file():
    """更新标签文件"""
    print("\n" + "=" * 70)
    print("更新标签文件")
    print("=" * 70)
    
    # 加载现有标签
    labels_df = pd.read_csv('outputs/static/prediction_labels.csv', index_col='uid')
    
    # 生成新的社交健康标签
    new_labels = generate_social_health_labels_no_leakage()
    
    if new_labels is None:
        print("  ❌ 标签生成失败")
        return
    
    # 更新社交健康标签列
    for uid in labels_df.index:
        if uid in new_labels:
            labels_df.loc[uid, 'social_health_risk'] = new_labels[uid]
    
    # 备份原文件
    backup_file = 'outputs/static/prediction_labels_backup.csv'
    labels_df_old = pd.read_csv('outputs/static/prediction_labels.csv', index_col='uid')
    labels_df_old.to_csv(backup_file)
    print(f"\n  ✓ 原标签已备份到: {backup_file}")
    
    # 保存更新后的标签
    output_file = 'outputs/static/prediction_labels.csv'
    labels_df.to_csv(output_file)
    print(f"  ✓ 新标签已保存到: {output_file}")
    
    # 验证泄露
    is_safe = verify_no_leakage(new_labels)
    
    # 对比旧标签
    print("\n对比新旧标签:")
    old_labels = labels_df_old['social_health_risk'].to_dict()
    
    changed = 0
    for uid in labels_df.index:
        if uid in old_labels and uid in new_labels:
            if old_labels[uid] != new_labels[uid]:
                changed += 1
    
    print(f"  - 旧标签正类比例: {sum(old_labels.values())/len(old_labels)*100:.1f}%")
    print(f"  - 新标签正类比例: {sum(new_labels.values())/len(new_labels)*100:.1f}%")
    print(f"  - 标签变化数: {changed}/{len(labels_df)} ({changed/len(labels_df)*100:.1f}%)")
    
    if is_safe:
        print("\n" + "=" * 70)
        print("✅ 标签更新成功！无数据泄露风险")
        print("=" * 70)
        print("\n下一步:")
        print("  1. 重新运行对比实验: python scripts/feature_combination_comparison.py")
        print("  2. 检查AUC是否降到合理范围（0.70-0.90）")
        print("  3. 如果AUC仍然很高但<1.0，说明多层特征确实有效")
    else:
        print("\n" + "=" * 70)
        print("⚠️  标签更新完成，但仍有泄露风险")
        print("=" * 70)


def main():
    """主函数"""
    update_labels_file()


if __name__ == '__main__':
    main()
