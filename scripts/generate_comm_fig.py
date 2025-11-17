"""社群可视化工具：生成社群相关图表

功能：
1. 社群规模分布图（柱状图）
2. 社群间连接热力图（权重矩阵）
3. 社群内部密度对比图
4. 社群属性富集可视化（若有 enrichment 文件）
5. 核心成员分布图（若有 core 文件）

输入：
- outputs/static/communities_uid.csv
- outputs/static/communities_uid_summary.csv
- outputs/static/communities_uid_enrichment.csv (可选)
- outputs/static/communities_uid_core.csv (可选)
- outputs/static/multiplex_edges_enhanced.csv

输出：
- outputs/static/figures/community_*.png
"""

import os
import sys
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(base_dir):
    """加载社群相关数据文件"""
    out_dir = os.path.join(base_dir, 'outputs', 'static')
    
    # 必需文件
    comm_path = os.path.join(out_dir, 'communities_uid.csv')
    if not os.path.isfile(comm_path):
        raise FileNotFoundError(f"未找到 {comm_path}，请先运行 detect_communities.py")
    communities_df = pd.read_csv(comm_path)
    
    edges_path = os.path.join(out_dir, 'multiplex_edges_enhanced.csv')
    if not os.path.isfile(edges_path):
        edges_path = os.path.join(out_dir, 'multiplex_edges_weighted.csv')
    if not os.path.isfile(edges_path):
        raise FileNotFoundError("未找到多层边文件")
    edges_df = pd.read_csv(edges_path)
    
    # 可选文件
    summary_df = None
    summary_path = os.path.join(out_dir, 'communities_uid_summary.csv')
    if os.path.isfile(summary_path):
        summary_df = pd.read_csv(summary_path)
    
    enrichment_df = None
    enrich_path = os.path.join(out_dir, 'communities_uid_enrichment.csv')
    if os.path.isfile(enrich_path):
        enrichment_df = pd.read_csv(enrich_path)
    
    core_df = None
    core_path = os.path.join(out_dir, 'communities_uid_core.csv')
    if os.path.isfile(core_path):
        core_df = pd.read_csv(core_path)
    
    return out_dir, communities_df, edges_df, summary_df, enrichment_df, core_df


def plot_community_size_distribution(summary_df, save_dir):
    """社群规模分布柱状图"""
    if summary_df is None or summary_df.empty:
        print("跳过社群规模分布图：无 summary 数据")
        return
    
    plt.figure(figsize=(10, 6))
    data = summary_df.sort_values('size', ascending=False).head(20)
    
    colors = cm.tab10(np.linspace(0, 1, len(data)))
    plt.bar(range(len(data)), data['size'], color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('社群编号', fontsize=12)
    plt.ylabel('成员数量', fontsize=12)
    plt.title('社群规模分布 (Top 20)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(data)), [f"C{c}" for c in data['community']], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'community_size_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def plot_inter_community_heatmap(communities_df, edges_df, save_dir):
    """社群间连接权重热力图"""
    # 构建 uid -> community 映射
    uid2comm = dict(zip(communities_df['uid'], communities_df['community']))
    
    # 统计社群间连接权重
    comm_pairs = defaultdict(float)
    edges_df.columns = [str(c).strip().lower() for c in edges_df.columns]
    for _, row in edges_df.iterrows():
        u = str(row['u'])
        v = str(row['v'])
        w = float(row.get('weight', 1.0))
        if u == v or w <= 0:
            continue
        cu = uid2comm.get(u, -1)
        cv = uid2comm.get(v, -1)
        if cu == -1 or cv == -1:
            continue
        pair = tuple(sorted([cu, cv]))
        comm_pairs[pair] += w
    
    # 构建矩阵
    comms = sorted(set(communities_df['community']))
    n = len(comms)
    matrix = np.zeros((n, n))
    comm_idx = {c: i for i, c in enumerate(comms)}
    
    for (c1, c2), w in comm_pairs.items():
        i = comm_idx.get(c1, -1)
        j = comm_idx.get(c2, -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += w
            matrix[j, i] += w
    
    # 归一化（按行最大值）
    row_max = matrix.max(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_max, where=row_max > 0, out=np.zeros_like(matrix))
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='归一化权重')
    plt.xlabel('社群编号', fontsize=12)
    plt.ylabel('社群编号', fontsize=12)
    plt.title('社群间连接强度热力图', fontsize=14, fontweight='bold')
    plt.xticks(range(n), [f"C{c}" for c in comms], rotation=45, ha='right')
    plt.yticks(range(n), [f"C{c}" for c in comms])
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'inter_community_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def plot_community_internal_density(enrichment_df, save_dir):
    """社群内部密度对比图"""
    if enrichment_df is None or enrichment_df.empty:
        print("跳过社群内部密度图：无 enrichment 数据")
        return
    
    data = enrichment_df.sort_values('size', ascending=False).head(15)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = range(len(data))
    ax1.bar(x, data['size'], alpha=0.6, color='steelblue', label='社群规模')
    ax1.set_xlabel('社群编号', fontsize=12)
    ax1.set_ylabel('成员数量', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in data['community']], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    if 'avg_internal_edge_weight' in data.columns:
        ax2.plot(x, data['avg_internal_edge_weight'], color='orangered', marker='o', linewidth=2, markersize=6, label='平均内部边权重')
        ax2.set_ylabel('平均内部边权重', fontsize=12, color='orangered')
        ax2.tick_params(axis='y', labelcolor='orangered')
    
    plt.title('社群规模与内部连接密度', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'community_internal_density.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def plot_core_member_distribution(core_df, save_dir):
    """核心成员分布饼图"""
    if core_df is None or core_df.empty:
        print("跳过核心成员分布图：无 core 数据")
        return
    
    core_counts = core_df.groupby('community')['is_core'].sum().sort_values(ascending=False).head(10)
    
    if core_counts.empty:
        print("跳过核心成员分布图：无核心成员")
        return
    
    plt.figure(figsize=(10, 8))
    colors = cm.Set3(np.linspace(0, 1, len(core_counts)))
    wedges, texts, autotexts = plt.pie(
        core_counts, 
        labels=[f"C{c}" for c in core_counts.index],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.title('核心成员在各社群的分布 (Top 10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'core_member_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def plot_enrichment_comparison(enrichment_df, save_dir):
    """社群属性富集对比雷达图（示例：情绪/压力/回复率）"""
    if enrichment_df is None or enrichment_df.empty:
        print("跳过富集对比图：无 enrichment 数据")
        return
    
    # 选择有数值的属性列
    num_cols = ['avg_mood', 'avg_stress', 'avg_response_rate']
    available = [c for c in num_cols if c in enrichment_df.columns and enrichment_df[c].notna().any()]
    
    if len(available) < 2:
        print("跳过富集对比图：可用数值属性不足")
        return
    
    # 选择前6个社群
    top_comms = enrichment_df.sort_values('size', ascending=False).head(6)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('avg_', '').replace('_', ' ').title() for c in available], fontsize=10)
    
    colors = cm.tab10(np.linspace(0, 1, len(top_comms)))
    
    for idx, (_, row) in enumerate(top_comms.iterrows()):
        values = [row.get(c, 0.0) for c in available]
        # 归一化到 [0, 1]
        max_val = max(values) if max(values) > 0 else 1.0
        values_norm = [v / max_val for v in values]
        values_norm += values_norm[:1]
        
        ax.plot(angles, values_norm, 'o-', linewidth=2, label=f"C{row['community']}", color=colors[idx])
        ax.fill(angles, values_norm, alpha=0.15, color=colors[idx])
    
    ax.set_ylim(0, 1)
    plt.title('社群属性对比 (归一化)', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'community_enrichment_radar.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir, communities_df, edges_df, summary_df, enrichment_df, core_df = load_data(base_dir)
    
    # 创建图表输出目录
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print("\n[生成社群可视化图表...]")
    
    # 1. 社群规模分布
    plot_community_size_distribution(summary_df, fig_dir)
    
    # 2. 社群间连接热力图
    plot_inter_community_heatmap(communities_df, edges_df, fig_dir)
    
    # 3. 社群内部密度
    plot_community_internal_density(enrichment_df, fig_dir)
    
    # 4. 核心成员分布
    plot_core_member_distribution(core_df, fig_dir)
    
    # 5. 富集属性对比
    plot_enrichment_comparison(enrichment_df, fig_dir)
    
    print(f"\n完成：社群可视化图表已保存至 {fig_dir}")


if __name__ == '__main__':
    main()
