"""多层网络社群划分（supra 与 uid 聚合两种视角）

输入优先使用增强结果：outputs/static/multiplex_edges_enhanced.csv；
如不存在则回退到 outputs/static/multiplex_edges_weighted.csv。

输出：
- outputs/static/communities_supra.csv: (layer, uid) 节点的社区划分
- outputs/static/communities_uid.csv: uid 聚合后的社区划分（更易解释）
- 控制台打印简要统计与 modularity

可选：通过命令行参数调整最小社区规模过滤等。
"""

import os
import sys
from collections import defaultdict, Counter
import math
import argparse

# 允许从项目根运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import networkx as nx


def load_multiplex_edges(base_dir: str):
    out_dir = os.path.join(base_dir, 'outputs', 'static')
    path_enh = os.path.join(out_dir, 'multiplex_edges_enhanced.csv')
    path_trad = os.path.join(out_dir, 'multiplex_edges_weighted.csv')
    if os.path.isfile(path_enh):
        print(f"[Load] {path_enh}")
        df = pd.read_csv(path_enh)
    elif os.path.isfile(path_trad):
        print(f"[Load] {path_trad}")
        df = pd.read_csv(path_trad)
    else:
        raise FileNotFoundError("未找到 multiplex_edges_enhanced.csv 或 multiplex_edges_weighted.csv，请先运行 src/example.py 生成。")
    # 统一列名
    df.columns = [str(c).strip().lower() for c in df.columns]
    # 期望列：type, src_layer, dst_layer, u, v, weight
    need = {'type','src_layer','dst_layer','u','v','weight'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"多层边CSV缺少所需列: {missing}")
    # 清理缺失与非法权重
    df = df.dropna(subset=['src_layer','dst_layer','u','v'])
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
    return df, out_dir


def build_supra_graph(df: pd.DataFrame) -> nx.Graph:
    """构建 supra 图：节点为 (layer,u)；包括层内与层间边。"""
    H = nx.Graph()
    for _, row in df.iterrows():
        lt = row['src_layer']
        rt = row['dst_layer']
        u = str(row['u'])
        v = str(row['v'])
        w = float(row['weight'])
        if w <= 0:
            continue
        su = (lt, u)
        sv = (rt, v)
        if H.has_edge(su, sv):
            H[su][sv]['weight'] += w
        else:
            H.add_edge(su, sv, weight=w)
    return H


def build_uid_graph(df: pd.DataFrame) -> nx.Graph:
    """构建 uid 聚合图：节点为 uid；
    - 层内边（type=intra, u!=v）计入 uid-uid 边；
    - 层间边（type=inter, u!=v）也计入 uid-uid 边；
    - 忽略对角跨层边（u==v），避免人为增强自连对聚类的影响。
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        u = str(row['u'])
        v = str(row['v'])
        w = float(row['weight'])
        if w <= 0 or u == v:
            continue
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def greedy_communities_weighted(G: nx.Graph):
    """使用 NetworkX 贪心模块度方法进行加权社群划分。"""
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return []
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    return [set(c) for c in comms]


def label_assignment_from_communities(comms, nodes):
    label = {}
    for cid, members in enumerate(comms):
        for n in members:
            label[n] = cid
    # 对未覆盖节点（孤立点）赋予新社区编号
    max_cid = len(comms)
    for n in nodes:
        if n not in label:
            label[n] = max_cid
            max_cid += 1
    return label


def modularity_safe(G: nx.Graph, label_dict):
    try:
        # 将标签字典转为社区列表
        comm_map = defaultdict(set)
        for n, c in label_dict.items():
            comm_map[c].add(n)
        comms = list(comm_map.values())
        return nx.algorithms.community.modularity(G, comms, weight='weight')
    except Exception:
        return float('nan')


def parse_args():
    parser = argparse.ArgumentParser(description="多层网络社区检测与核心成员识别")
    parser.add_argument('--core-percentile', type=float, default=0.8, help='内部加权度进入核心的分位阈值 (0-1)')
    parser.add_argument('--core-zscore', type=float, default=1.0, help='内部加权度 z-score 阈值 (备选条件)')
    parser.add_argument('--core-max', type=int, default=None, help='每个社区最多保留的核心成员数 (None=不限制)')
    parser.add_argument('--min-community-size', type=int, default=2, help='小于该规模的社区全部视为核心或单独输出')
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, out_dir = load_multiplex_edges(base_dir)

    # 1) Supra 图社群
    print("\n[Supra] 构建 supra 图并进行社群划分...")
    H = build_supra_graph(df)
    print(f"supra 节点数={H.number_of_nodes()}, 边数={H.number_of_edges()}")
    comms_H = greedy_communities_weighted(H)
    labels_H = label_assignment_from_communities(comms_H, H.nodes())
    Q_H = modularity_safe(H, labels_H)
    sizes_H = sorted([len(c) for c in comms_H], reverse=True)
    print(f"supra 社区数={len(comms_H)}, 前5个社区规模={sizes_H[:5]}, modularity≈{Q_H:.4f}")

    # 导出 supra 标签
    rows = []
    for (layer, uid), c in labels_H.items():
        rows.append({'layer': layer, 'uid': uid, 'community': c})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'communities_supra.csv'), index=False, encoding='utf-8')
    print(f"[Export] communities_supra.csv")

    # 2) UID 聚合图社群（更易解释）
    print("\n[UID] 构建 uid 聚合图并进行社群划分...")
    G = build_uid_graph(df)
    print(f"uid 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    comms_G = greedy_communities_weighted(G)
    labels_G = label_assignment_from_communities(comms_G, G.nodes())
    Q_G = modularity_safe(G, labels_G)
    sizes_G = sorted([len(c) for c in comms_G], reverse=True)
    print(f"uid 社区数={len(comms_G)}, 前5个社区规模={sizes_G[:5]}, modularity≈{Q_G:.4f}")

    # 导出 uid 标签
    rows = []
    for uid, c in labels_G.items():
        rows.append({'uid': uid, 'community': c})
    path_uid = os.path.join(out_dir, 'communities_uid.csv')
    pd.DataFrame(rows).to_csv(path_uid, index=False, encoding='utf-8')
    print(f"[Export] communities_uid.csv")

    # 3) 额外摘要：每个社区的规模与代表成员
    com2members = defaultdict(list)
    for uid, c in labels_G.items():
        com2members[c].append(uid)
    summary = []
    for c, members in com2members.items():
        summary.append({'community': c, 'size': len(members)})
    sum_df = pd.DataFrame(summary).sort_values('size', ascending=False)
    path_sum = os.path.join(out_dir, 'communities_uid_summary.csv')
    sum_df.to_csv(path_sum, index=False, encoding='utf-8')
    print(f"[Export] communities_uid_summary.csv (按规模降序)")

    # 4) 社区属性富集分析（survey/EMA/通信特征 + 结构指标）
    print("\n[Enrichment] 分析社区属性富集...")
    enrichment_rows = []
    # 尝试加载通信特征（如存在）
    comm_feat_path = os.path.join(out_dir, 'communication_features.csv')
    comm_df = None
    if os.path.isfile(comm_feat_path):
        try:
            comm_df = pd.read_csv(comm_feat_path)
            if 'uid' in comm_df.columns:
                comm_df.set_index('uid', inplace=True, drop=False)
        except Exception:
            comm_df = None

    # 收集节点属性（survey/EMA）
    try:
        from src.layers.education import NodeAttributes  # 复用已有类
        attrs_all = NodeAttributes().collect_attributes()
    except Exception:
        attrs_all = {}

    # 预计算结构指标：度、加权度、社区内部密度
    weighted_degree = dict(G.degree(weight='weight'))
    degree = dict(G.degree())

    # 计算社区内部边权密度（内部总权重 / 成员数选择标准化）
    for cid, members in com2members.items():
        member_set = set(members)
        # 内部总权重
        internal_weight = 0.0
        internal_edges = 0
        for u in members:
            for v in G.neighbors(u):
                if v in member_set:
                    w = float(G[u][v].get('weight', 1.0))
                    internal_weight += w
                    internal_edges += 1
        # 每条边被双计，除以2
        internal_weight /= 2.0
        internal_edges //= 2
        # 简单密度指标：internal_weight / max(1, internal_edges)
        density_w = internal_weight / max(1, internal_edges)
        # 汇总成员属性统计
        ages = []
        majors = Counter()
        years = Counter()
        mood = []
        stress = []
        response_rates = []
        comm_feat_means = Counter()
        comm_feat_counts = Counter()

        for u in members:
            a = attrs_all.get(u, {})
            survey = (a.get('survey') or {})
            ema = (a.get('ema') or {})
            try:
                if 'age' in survey:
                    ages.append(float(survey.get('age')))
            except Exception:
                pass
            if 'major' in survey and survey.get('major') not in [None, '', 'nan']:
                majors[str(survey.get('major'))] += 1
            if 'year' in survey and survey.get('year') not in [None, '', 'nan']:
                years[str(survey.get('year'))] += 1
            for k in ['avg_mood','avg_stress','response_rate']:
                try:
                    val = ema.get(k)
                    if val is not None:
                        if k == 'avg_mood':
                            mood.append(float(val))
                        elif k == 'avg_stress':
                            stress.append(float(val))
                        elif k == 'response_rate':
                            response_rates.append(float(val))
                except Exception:
                    pass
            # 通信特征
            if comm_df is not None and u in comm_df.index:
                row = comm_df.loc[u]
                for col in comm_df.columns:
                    if col == 'uid':
                        continue
                    try:
                        v = float(row[col])
                        comm_feat_means[col] += v
                        comm_feat_counts[col] += 1
                    except Exception:
                        continue

        def _avg(lst):
            return sum(lst)/len(lst) if lst else None
        avg_age = _avg(ages)
        avg_mood = _avg(mood)
        avg_stress = _avg(stress)
        avg_response = _avg(response_rates)
        # 主专业/年级 top3
        top_majors = ";".join([f"{m}:{c}" for m,c in majors.most_common(3)]) if majors else ''
        top_years = ";".join([f"{y}:{c}" for y,c in years.most_common(3)]) if years else ''
        # 通信特征均值
        comm_avgs = {}
        for col, total in comm_feat_means.items():
            cnt = comm_feat_counts.get(col, 0)
            if cnt > 0:
                comm_avgs[col] = total / cnt

        enrichment_rows.append({
            'community': cid,
            'size': len(members),
            'internal_edges': internal_edges,
            'internal_weight': round(internal_weight, 3),
            'avg_internal_edge_weight': round(density_w, 3),
            'avg_age': round(avg_age, 2) if avg_age is not None else None,
            'top_majors': top_majors,
            'top_years': top_years,
            'avg_mood': round(avg_mood, 3) if avg_mood is not None else None,
            'avg_stress': round(avg_stress, 3) if avg_stress is not None else None,
            'avg_response_rate': round(avg_response, 3) if avg_response is not None else None,
            **{f'comm_{k}_mean': round(v,3) for k,v in comm_avgs.items()},
        })

    enrich_path = os.path.join(out_dir, 'communities_uid_enrichment.csv')
    pd.DataFrame(enrichment_rows).to_csv(enrich_path, index=False, encoding='utf-8')
    print(f"[Export] communities_uid_enrichment.csv (社区属性与结构富集)")

    # 5) 核心成员识别
    print("\n[Core] 识别社区核心成员...")
    core_rows = []
    for cid, members in com2members.items():
        member_set = set(members)
        # 计算内部加权度 / 内部度
        internal_wdeg = {}
        internal_deg = {}
        for u in members:
            w_sum = 0.0
            d_cnt = 0
            for v in G.neighbors(u):
                if v in member_set:
                    w = float(G[u][v].get('weight', 1.0))
                    w_sum += w
                    d_cnt += 1
            internal_wdeg[u] = w_sum
            internal_deg[u] = d_cnt
        values = list(internal_wdeg.values())
        if not values:
            continue
        mean_v = sum(values)/len(values)
        std_v = math.sqrt(sum((x-mean_v)**2 for x in values)/len(values)) if len(values)>1 else 0.0
        # 分位阈值
        sorted_vals = sorted(values)
        pct_idx = int((len(sorted_vals)-1) * args.core_percentile)
        pct_threshold = sorted_vals[pct_idx]
        # 标记核心条件：内部加权度 >= 分位阈值 或 zscore >= core_zscore
        core_candidates = []
        for u in members:
            z = (internal_wdeg[u]-mean_v)/std_v if std_v>1e-9 else 0.0
            is_core = False
            if len(members) <= args.min_community_size:
                is_core = True
            elif internal_wdeg[u] >= pct_threshold or z >= args.core_zscore:
                is_core = True
            core_candidates.append((u, internal_deg[u], internal_wdeg[u], z, is_core))
        # 若限制最大数量：按内部加权度排序只保留前 core_max
        if args.core_max is not None:
            # 筛选初步核心后再限制数量
            prelim = [c for c in core_candidates if c[4]]
            prelim.sort(key=lambda x: x[2], reverse=True)
            keep_set = set(u for u,_,_,_,_ in prelim[:args.core_max])
        else:
            keep_set = {u for u,_,_,_,flag in core_candidates if flag}
        # 输出全部成员含核心标记
        for rank,(u,ideg,iwdeg,z,is_core) in enumerate(sorted(core_candidates, key=lambda x: x[2], reverse=True), start=1):
            core_rows.append({
                'community': cid,
                'uid': u,
                'internal_degree': ideg,
                'internal_weighted_degree': round(iwdeg,3),
                'internal_wdeg_zscore': round(z,3),
                'global_weighted_degree': round(weighted_degree.get(u,0.0),3),
                'is_core': 1 if u in keep_set else 0,
                'core_rank_by_internal_weight': rank,
                'core_threshold_weight': round(pct_threshold,3),
                'core_mean_weight': round(mean_v,3),
                'core_std_weight': round(std_v,3),
            })
    core_path = os.path.join(out_dir, 'communities_uid_core.csv')
    pd.DataFrame(core_rows).to_csv(core_path, index=False, encoding='utf-8')
    print(f"[Export] communities_uid_core.csv (核心成员识别)")

    print("\n完成：已在 outputs/static 下导出社群划分、富集与核心成员。")


if __name__ == '__main__':
    main()
