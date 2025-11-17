import os
import pandas as pd
import numpy as np

"""
从 outputs/static 下的三层边列表构建：
1) supra 邻接矩阵（3N x 3N）；
2) 多层边CSV（包含层内边与层间耦合边）。

要求：已存在 uid_number_map.csv 用于统一学生编号。
输出文件：
- outputs/static/supra_adjacency.csv
- outputs/static/multi_layer_edges.csv
"""


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_uid_map(out_dir: str):
    map_path = os.path.join(out_dir, 'uid_number_map.csv')
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"未找到 {map_path}，请先生成 UID 编号映射")
    df = pd.read_csv(map_path)
    if not {'uid', 'number'}.issubset(df.columns):
        raise ValueError("uid_number_map.csv 应包含列：uid, number")
    df = df.sort_values('number')
    uids = list(df['uid'])
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    return uids, uid_to_idx


def build_layer_adj(out_dir: str, fname: str, uid_to_idx: dict, N: int) -> np.ndarray:
    path = os.path.join(out_dir, fname)
    A = np.zeros((N, N), dtype=float)
    if not os.path.isfile(path):
        return A
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        u = str(r['source'])
        v = str(r['target'])
        w = r['weight'] if 'weight' in df.columns else 1.0
        try:
            w = float(w) if pd.notna(w) else 0.0
        except Exception:
            w = 0.0
        if u in uid_to_idx and v in uid_to_idx:
            i = uid_to_idx[u]
            j = uid_to_idx[v]
            if i != j:
                A[i, j] += w
                A[j, i] += w
    return A


def export_supra_and_edges():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'outputs', 'static')
    ensure_dir(out_dir)

    uids, uid_to_idx = load_uid_map(out_dir)
    N = len(uids)

    A_phy = build_layer_adj(out_dir, 'physical_edges_weighted.csv', uid_to_idx, N)
    A_beh = build_layer_adj(out_dir, 'behavior_edges_weighted.csv', uid_to_idx, N)
    A_edu = build_layer_adj(out_dir, 'education_edges_weighted.csv', uid_to_idx, N)

    # supra 邻接矩阵
    w = 1.0
    I = np.eye(N)
    top = np.concatenate([A_phy, w * I, w * I], axis=1)
    mid = np.concatenate([w * I, A_beh, w * I], axis=1)
    bot = np.concatenate([w * I, w * I, A_edu], axis=1)
    A_multi = np.concatenate([top, mid, bot], axis=0)

    cols = [f'phy_{i+1}' for i in range(N)] + [f'beh_{i+1}' for i in range(N)] + [f'edu_{i+1}' for i in range(N)]
    rows = cols
    sup_df = pd.DataFrame(A_multi, columns=cols, index=rows)
    sup_path = os.path.join(out_dir, 'supra_adjacency.csv')
    sup_df.to_csv(sup_path, encoding='utf-8', float_format='%.6f')

    # 多层边（层内 + 层间耦合）
    def add_intra(fname: str, layer: str, store: list):
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            return
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            u = str(r['source'])
            v = str(r['target'])
            wv = r['weight'] if 'weight' in df.columns else 1.0
            try:
                wv = float(wv) if pd.notna(wv) else 0.0
            except Exception:
                wv = 0.0
            store.append({'type': 'intra', 'src_layer': layer, 'dst_layer': layer, 'u': u, 'v': v, 'weight': wv})

    edges = []
    add_intra('physical_edges_weighted.csv', 'phy', edges)
    add_intra('behavior_edges_weighted.csv', 'beh', edges)
    add_intra('education_edges_weighted.csv', 'edu', edges)

    for u in uids:
        edges.append({'type': 'inter', 'src_layer': 'phy', 'dst_layer': 'beh', 'u': u, 'v': u, 'weight': w})
        edges.append({'type': 'inter', 'src_layer': 'phy', 'dst_layer': 'edu', 'u': u, 'v': u, 'weight': w})
        edges.append({'type': 'inter', 'src_layer': 'beh', 'dst_layer': 'edu', 'u': u, 'v': u, 'weight': w})

    multi_path = os.path.join(out_dir, 'multi_layer_edges.csv')
    pd.DataFrame(edges).to_csv(multi_path, index=False, encoding='utf-8')

    print(f"[Export] 已导出 supra 邻接矩阵 -> {sup_path}")
    print(f"[Export] 已导出多层边 -> {multi_path}")


if __name__ == '__main__':
    export_supra_and_edges()