import os
import sys
import csv

# 添加项目根目录到Python路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # project root containing 'src'
sys.path.append(PROJECT_ROOT)

from src.layers.education import EducationNetwork


def ensure_dir(dir_path: str):
    if dir_path and not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def export_edges_csv(G, out_dir: str):
    ensure_dir(out_dir)
    unweighted_path = os.path.join(out_dir, 'education_edges_unweighted.csv')
    weighted_path = os.path.join(out_dir, 'education_edges_weighted.csv')

    # 无权边
    with open(unweighted_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source', 'target'])
        for u, v in G.edges():
            writer.writerow([u, v])

    # 有权边
    with open(weighted_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source', 'target', 'weight'])
        for u, v, d in G.edges(data=True):
            writer.writerow([u, v, float(d.get('weight', 1.0))])

    return unweighted_path, weighted_path


def main():
    edu = EducationNetwork()
    res = edu.build_network(network_type='co-enrollment', time_window=None)
    G = res.get('all')
    if G is None:
        print('No education graph produced.')
        return

    print(f"Education Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'static')
    unweighted_path, weighted_path = export_edges_csv(G, out_dir)
    print(f"[Export] Wrote: {unweighted_path} and {weighted_path}")


if __name__ == '__main__':
    main()