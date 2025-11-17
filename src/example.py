import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 可视化支持
from src.layers.communication import CallLogNetwork, SMSNetwork
from src.layers.physical import SensingNetwork, DiningNetwork
from src.layers.behavior import AppUsageNetwork, CalendarNetwork, RawAccFeatNetwork
from src.layers.education import EducationNetwork, NodeAttributes
from src.layers.combined import StaticLayerBuilder
from src.utils import NetworkBuilder
from src.layers.comm_features import CommunicationFeatureExtractor
from src.inter_layer_coupling import InterLayerCoupling  # 增强层间耦合

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局配置：是否使用增强层间耦合
USE_ENHANCED_COUPLING = True  # True=增强方法, False=传统方法

def visualize_network(G, title, time_key=None, label_map=None, save_dir=None):
    """可视化单层网络

    Args:
        G: 网络图
        title: 图标题
        time_key: 可选的时间窗口标签
        label_map: 可选的 UID→编号 映射，用于统一可视化编号
        save_dir: 可选的保存目录；若提供，则额外保存 PNG 图片
    """
    if G.number_of_nodes() == 0:
        print(f"跳过可视化 {title} - 网络中没有节点")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 如果节点数太多，仅显示度最高的前60个以保证可读性
    if G.number_of_nodes() > 120:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:60]
        G_sub = G.subgraph(top_nodes)
        print(f"网络节点数较多，仅显示度数最高的60个节点的子图")
    else:
        G_sub = G
    
    # 计算节点大小（基于度中心性）
    try:
        degree_centrality = nx.degree_centrality(G_sub)
        node_sizes = [v * 3000 + 300 for v in degree_centrality.values()]
    except:
        node_sizes = 500
    
    # 计算边的宽度（基于权重）
    try:
        edges_weights = [d['weight'] for u, v, d in G_sub.edges(data=True)]
        if edges_weights:
            edge_widths = [w * 5 + 0.5 for w in edges_weights]
        else:
            edge_widths = 1.0
    except:
        edge_widths = 1.0
    
    # 使用spring布局
    pos = nx.spring_layout(G_sub, k=1, iterations=50)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G_sub, pos, width=edge_widths, edge_color='gray', alpha=0.6)
    
    # 始终绘制编号标签（统一编号优先），提高可读性
    try:
        if label_map:
            labels = {n: str(label_map.get(n, n)) for n in G_sub.nodes()}
            nx.draw_networkx_labels(
                G_sub,
                pos,
                labels=labels,
                font_size=9,
                font_color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1)
            )
        else:
            nx.draw_networkx_labels(
                G_sub,
                pos,
                font_size=9,
                font_color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1)
            )
    except Exception:
        pass
    
    title_str = f"{title}"
    if time_key:
        title_str += f" - {time_key}"
    plt.title(title_str)
    plt.axis('off')
    plt.tight_layout()
    # 保存图片（若指定目录）
    if save_dir:
        NetworkBuilder.ensure_dir(save_dir)
        safe_title = ''.join(ch if ch.isalnum() or ch in ' _-' else '_' for ch in title)
        suffix = f"_{time_key}" if time_key else ''
        out_path = os.path.join(save_dir, f"{safe_title}{suffix}.png")
        try:
            plt.savefig(out_path, dpi=150)
            print(f"[Saved] {out_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")

    plt.show()


def visualize_network_with_communities(G, communities_dict, title, label_map=None, save_dir=None):
    """可视化网络并按社群着色
    
    Args:
        G: 网络图
        communities_dict: {uid: community_id} 社群标签字典
        title: 图标题
        label_map: 可选的 UID→编号 映射
        save_dir: 可选的保存目录
    """
    if G.number_of_nodes() == 0:
        print(f"跳过可视化 {title} - 网络中没有节点")
        return
    
    plt.figure(figsize=(14, 10))
    
    # 如果节点数太多，仅显示度最高的前80个
    if G.number_of_nodes() > 100:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:80]
        G_sub = G.subgraph(top_nodes)
        print(f"网络节点数较多，仅显示度数最高的80个节点的子图")
    else:
        G_sub = G
    
    # 社群颜色映射
    unique_comms = set(communities_dict.get(n, -1) for n in G_sub.nodes())
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10' if len(unique_comms) <= 10 else 'tab20')
    comm_colors = {c: cmap(i / max(len(unique_comms), 1)) for i, c in enumerate(sorted(unique_comms))}
    
    node_colors = [comm_colors.get(communities_dict.get(n, -1), 'gray') for n in G_sub.nodes()]
    
    # 节点大小基于度中心性
    try:
        degree_centrality = nx.degree_centrality(G_sub)
        node_sizes = [v * 3000 + 300 for v in degree_centrality.values()]
    except:
        node_sizes = 500
    
    # 边宽度基于权重
    try:
        edges_weights = [d.get('weight', 1.0) for u, v, d in G_sub.edges(data=True)]
        edge_widths = [w * 3 + 0.5 for w in edges_weights]
    except:
        edge_widths = 1.0
    
    # 使用spring布局
    pos = nx.spring_layout(G_sub, k=1.5, iterations=50, seed=42)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G_sub, pos, width=edge_widths, edge_color='gray', alpha=0.4)
    
    # 绘制标签
    try:
        if label_map:
            labels = {n: str(label_map.get(n, n)) for n in G_sub.nodes()}
        else:
            labels = {n: str(n) for n in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, pos, labels=labels, font_size=8, font_color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5)
        )
    except Exception:
        pass
    
    plt.title(f"{title} - 社群划分可视化", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=comm_colors[c], label=f'Community {c}') 
                      for c in sorted(unique_comms) if c != -1]
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    if save_dir:
        NetworkBuilder.ensure_dir(save_dir)
        safe_title = ''.join(ch if ch.isalnum() or ch in ' _-' else '_' for ch in title)
        out_path = os.path.join(save_dir, f"{safe_title}_communities.png")
        try:
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"[Saved] {out_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")
    plt.show()


def visualize_multiplex_3d(G_phy: nx.Graph, G_beh: nx.Graph, G_edu: nx.Graph, label_map=None, omega: float = 1.0, save_path: str = None):
    """三层多重网络的3D可视化（参考示意图）。

    - 节点放置在三个平面 z=0,1,2；
    - 层内边绘制在各自平面；
    - 层间边用垂直虚线连接同一学生的三层副本；
    """
    layers = [G_phy, G_beh, G_edu]
    layer_names = ['Physical', 'Behavior', 'Education']
    colors = ['#5CC8B8', '#F5D79C', '#F28C79']

    # 统一节点集合与布局（以所有出现过的学生为集合）
    all_nodes = set()
    for G in layers:
        all_nodes |= set(G.nodes())
    if not all_nodes:
        print("多层3D可视化：无节点，跳过")
        return

    # 使用 spring_layout 在二维平面确定统一位置
    base_G = nx.Graph()
    base_G.add_nodes_from(all_nodes)
    # 用三层的边合并一个稀疏参考图用于布局（不影响结果，只用于坐标）
    for G in layers:
        base_G.add_edges_from(G.edges())
    pos2d = nx.spring_layout(base_G, k=1.0, iterations=200)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    # 绘制各层节点与边
    for li, (G, lname, color) in enumerate(zip(layers, layer_names, colors)):
        z = li  # 0,1,2 三个平面
        xs = [pos2d[n][0] for n in all_nodes]
        ys = [pos2d[n][1] for n in all_nodes]
        zs = [z] * len(all_nodes)
        # 节点
        ax.scatter(xs, ys, zs, s=30, c=color, alpha=0.85, edgecolors='k', linewidths=0.3)
        # 边（只画该层存在的边）
        for u, v, d in G.edges(data=True):
            x = [pos2d[u][0], pos2d[v][0]]
            y = [pos2d[u][1], pos2d[v][1]]
            zline = [z, z]
            w = float(d.get('weight', 1.0))
            ax.plot(x, y, zline, color='gray', alpha=0.6, linewidth=0.8 + 0.8 * w)
        # 层标题
        ax.text2D(0.02, 0.95 - 0.08 * li, f"{lname}", transform=ax.transAxes, fontsize=12, color=color)

    # 层间边：同一学生的垂直虚线连接（omega 仅用于样式权示意）
    for n in all_nodes:
        x = [pos2d[n][0]] * 3
        y = [pos2d[n][1]] * 3
        z = [0, 1, 2]
        ax.plot(x, y, z, color='black', alpha=min(0.1 + 0.2 * omega, 0.6), linestyle='dashed', linewidth=0.8)

    # 显示编号标签（在顶层 z=2 旁标注）
    if label_map:
        for n in all_nodes:
            lx, ly = pos2d[n]
            ax.text(lx, ly, 2.02, s=str(label_map.get(n, n)), fontsize=8, color='black')
    else:
        for n in all_nodes:
            lx, ly = pos2d[n]
            ax.text(lx, ly, 2.02, s=str(n), fontsize=8, color='black')

    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=180)
            print(f"[Saved] 3D 多层网络图: {save_path}")
        except Exception as e:
            print(f"保存3D图失败: {e}")
    plt.show()

def main():
    # 初始化各层网络构建器
    call_network = CallLogNetwork()
    sms_network = SMSNetwork()
    sensing_network = SensingNetwork()
    dining_network = DiningNetwork()
    app_network = AppUsageNetwork()
    calendar_network = CalendarNetwork()
    education_network = EducationNetwork()
    rawaccfeat_network = RawAccFeatNetwork()  # 新增
    static_builder = StaticLayerBuilder()

    # 统一 UID→编号 映射，供该函数内所有层可视化使用
    try:
        layer_folders = ['call_log', 'sms', 'sensing', 'dining', 'dinning', 'app_usage', 'calendar', 'education', 'rawaccfeat']
        all_users = NetworkBuilder.get_all_users_union(layer_folders)
        uid_number_map = NetworkBuilder.build_uid_numbering(all_users)
    except Exception:
        uid_number_map = None
    
    # 构建基础社交层（通话网络，按周统计）
    print("构建通话网络...")
    try:
        call_graphs = call_network.build_network(
            weight_type='duration',
            time_window='W'
        )
        print(f"通话网络构建完成，包含 {len(call_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建通话网络时出错: {e}")
        call_graphs = {}
    
    # 构建短信网络
    print("构建短信网络...")
    try:
        sms_graphs = sms_network.build_network(
            weight_type='count',
            time_window='W'
        )
        print(f"短信网络构建完成，包含 {len(sms_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建短信网络时出错: {e}")
        sms_graphs = {}
    
    # 构建物理接触网络（示例参数）
    print("构建物理接触网络...")
    try:
        sensing_graphs = sensing_network.build_network(
            proximity_threshold=10,  # 根据实际数据单位调整
            min_duration=300,  # 5分钟
            time_window='W'  # 改为按周
        )
        print(f"物理接触网络构建完成，包含 {len(sensing_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建物理接触网络时出错: {e}")
        sensing_graphs = {}
    
    # 构建就餐共现网络
    print("构建就餐共现网络...")
    try:
        dining_graphs = dining_network.build_network(
            time_threshold=30,  # 30分钟内视为共同就餐
            time_window='W'  # 改为按周
        )
        print(f"就餐共现网络构建完成，包含 {len(dining_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建就餐共现网络时出错: {e}")
        dining_graphs = {}
    
    # 构建应用使用相似度网络
    print("构建应用使用网络...")
    try:
        app_graphs = app_network.build_network(
            network_type='similarity',
            similarity_threshold=0.2,
            top_k=10,  # 每个用户保留相似度最高的10个邻居
            time_window='W'  # 添加时间窗口
        )
        print(f"应用使用网络构建完成，包含 {len(app_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建应用使用网络时出错: {e}")
        app_graphs = {}
    
    # 构建日历共现网络
    print("构建日历共现网络...")
    try:
        calendar_graphs = calendar_network.build_network(
            network_type='co-occurrence',
            time_window='W'  # 添加时间窗口
        )
        print(f"日历共现网络构建完成，包含 {len(calendar_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建日历共现网络时出错: {e}")
        calendar_graphs = {}
    
    # 构建教育网络
    print("构建教育网络...")
    try:
        education_graphs = education_network.build_network(
            network_type='co-enrollment',
            time_window='W'  # 添加时间窗口
        )
        print(f"教育网络构建完成，包含 {len(education_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建教育网络时出错: {e}")
        education_graphs = {}
        # 尝试不使用时间窗口构建教育网络
        try:
            education_single = education_network.build_network(network_type='co-enrollment')
            print("教育网络构建完成（无时间窗口）")
        except Exception as e2:
            print(f"构建教育网络（无时间窗口）时出错: {e2}")
            education_single = nx.Graph()
    
    # 构建行为同步网络 (新增)
    print("构建行为同步网络...")
    try:
        rawaccfeat_graphs = rawaccfeat_network.build_network(
            similarity_threshold=0.7,
            top_k=10,
            time_window='W'
        )
        print(f"行为同步网络构建完成，包含 {len(rawaccfeat_graphs)} 个时间窗口")
    except Exception as e:
        print(f"构建行为同步网络时出错: {e}")
        rawaccfeat_graphs = {}
    
    # 收集节点属性
    print("收集节点属性...")
    node_attrs = NodeAttributes()
    attributes = node_attrs.collect_attributes()
    
    # 可视化各层网络
    print("可视化各层网络...")
    
    # 可视化通话网络 - 如果没有时间窗口，则直接可视化
    print(f"通话网络共有 {len(call_graphs)} 个时间窗口")
    if len(call_graphs) > 0:
        for time_key, graph in call_graphs.items():  # 显示所有时间窗口
            print(f"可视化通话网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "通话网络", time_key, label_map=uid_number_map)
    else:
        # 尝试构建无时间窗口的通话网络
        try:
            call_res = call_network.build_network(weight_type='duration', time_window=None)
            if isinstance(call_res, dict):
                call_single = call_res.get('all', next(iter(call_res.values()), nx.Graph()))
            else:
                call_single = call_res
            print(f"可视化通话网络（无时间窗口）, 节点数: {call_single.number_of_nodes()}, 边数: {call_single.number_of_edges()}")
            visualize_network(call_single, "通话网络", label_map=uid_number_map)
        except Exception as e:
            print(f"构建无时间窗口通话网络时出错: {e}")
    
    # 可视化短信网络
    print(f"短信网络共有 {len(sms_graphs)} 个时间窗口")
    if len(sms_graphs) > 0:
        for time_key, graph in sms_graphs.items():
            print(f"可视化短信网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "短信网络", time_key, label_map=uid_number_map)
    else:
        # 尝试构建无时间窗口的短信网络
        try:
            sms_res = sms_network.build_network(weight_type='count', time_window=None)
            if isinstance(sms_res, dict):
                sms_single = sms_res.get('all', next(iter(sms_res.values()), nx.Graph()))
            else:
                sms_single = sms_res
            print(f"可视化短信网络（无时间窗口）, 节点数: {sms_single.number_of_nodes()}, 边数: {sms_single.number_of_edges()}")
            visualize_network(sms_single, "短信网络", label_map=uid_number_map)
        except Exception as e:
            print(f"构建无时间窗口短信网络时出错: {e}")
    
    # 可视化物理接触网络
    print(f"物理接触网络共有 {len(sensing_graphs)} 个时间窗口")
    if len(sensing_graphs) > 0:
        for time_key, graph in sensing_graphs.items():
            print(f"可视化物理接触网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "物理接触网络", time_key, label_map=uid_number_map)
    else:
        # 尝试构建无时间窗口的物理接触网络
        try:
            sensing_res = sensing_network.build_network(proximity_threshold=10, min_duration=300, time_window=None)
            if isinstance(sensing_res, dict):
                sensing_single = sensing_res.get('all', next(iter(sensing_res.values()), nx.Graph()))
            else:
                sensing_single = sensing_res
            print(f"可视化物理接触网络（无时间窗口）, 节点数: {sensing_single.number_of_nodes()}, 边数: {sensing_single.number_of_edges()}")
            visualize_network(sensing_single, "物理接触网络", label_map=uid_number_map)
        except Exception as e:
            print(f"构建无时间窗口物理接触网络时出错: {e}")
    
    # 可视化就餐共现网络
    print(f"就餐共现网络共有 {len(dining_graphs)} 个时间窗口")
    if len(dining_graphs) > 0:
        for time_key, graph in dining_graphs.items():
            print(f"可视化就餐共现网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "就餐共现网络", time_key, label_map=uid_number_map)
    else:
        # 尝试构建无时间窗口的就餐共现网络
        try:
            dining_res = dining_network.build_network(time_threshold=30, time_window=None)
            if isinstance(dining_res, dict):
                dining_single = dining_res.get('all', next(iter(dining_res.values()), nx.Graph()))
            else:
                dining_single = dining_res
            print(f"可视化就餐共现网络（无时间窗口）, 节点数: {dining_single.number_of_nodes()}, 边数: {dining_single.number_of_edges()}")
            visualize_network(dining_single, "就餐共现网络", label_map=uid_number_map)
        except Exception as e:
            print(f"构建无时间窗口就餐共现网络时出错: {e}")
    
    # 可视化应用使用网络
    print(f"应用使用网络共有 {len(app_graphs)} 个时间窗口")
    for time_key, graph in app_graphs.items():
        print(f"可视化应用使用网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
        visualize_network(graph, "应用使用网络", time_key, label_map=uid_number_map)
    
    # 可视化日历共现网络
    print(f"日历共现网络共有 {len(calendar_graphs)} 个时间窗口")
    for time_key, graph in calendar_graphs.items():
        print(f"可视化日历共现网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
        visualize_network(graph, "日历共现网络", time_key, label_map=uid_number_map)
    
    # 可视化教育网络
    print(f"教育网络共有 {len(education_graphs)} 个时间窗口")
    if len(education_graphs) > 0:
        for time_key, graph in education_graphs.items():
            print(f"可视化教育网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "教育网络", time_key, label_map=uid_number_map)
    else:
        # 使用无时间窗口的教育网络
        try:
            if 'education_single' in locals():
                print(f"可视化教育网络（无时间窗口）, 节点数: {education_single.number_of_nodes()}, 边数: {education_single.number_of_edges()}")
                visualize_network(education_single, "教育网络", label_map=uid_number_map)
        except Exception as e:
            print(f"可视化教育网络时出错: {e}")
    
    # 可视化行为同步网络
    print(f"行为同步网络共有 {len(rawaccfeat_graphs)} 个时间窗口")
    if len(rawaccfeat_graphs) > 0:
        for time_key, graph in rawaccfeat_graphs.items():
            print(f"可视化行为同步网络 - 时间窗口: {time_key}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
            visualize_network(graph, "行为同步网络", time_key, label_map=uid_number_map)
    else:
        # 尝试构建无时间窗口的行为同步网络
        try:
            raf_res = rawaccfeat_network.build_network(similarity_threshold=0.7, top_k=10, time_window=None)
            if isinstance(raf_res, dict):
                rawaccfeat_single = raf_res.get('all', next(iter(raf_res.values()), nx.Graph()))
            else:
                rawaccfeat_single = raf_res
            print(f"可视化行为同步网络（无时间窗口）, 节点数: {rawaccfeat_single.number_of_nodes()}, 边数: {rawaccfeat_single.number_of_edges()}")
            visualize_network(rawaccfeat_single, "行为同步网络", label_map=uid_number_map)
        except Exception as e:
            print(f"构建无时间窗口行为同步网络时出错: {e}")
    
    # 示例：分析某一时间窗口的网络
    time_keys = list(call_graphs.keys()) if len(call_graphs) > 0 else []
    if time_keys:
        time_key = time_keys[0]
        print(f"\n时间窗口 {time_key} 的多层网络分析:")
        
        # 收集该时间窗口的所有网络
        layer_graphs = {
            '通话': call_graphs.get(time_key, nx.Graph()),
            '短信': sms_graphs.get(time_key, nx.Graph()),
            '物理接触': sensing_graphs.get(time_key, nx.Graph()),
            '就餐': dining_graphs.get(time_key, nx.Graph()),
            '应用使用': app_graphs.get(time_key, nx.Graph()),
            '日历事件': calendar_graphs.get(time_key, nx.Graph()),
            '教育': education_graphs.get(time_key, nx.Graph()),
            '行为同步': rawaccfeat_graphs.get(time_key, nx.Graph())
        }
        
        # 分析各层网络指标
        for layer_name, G in layer_graphs.items():
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                try:
                    degree_centrality = nx.degree_centrality(G)
                    clustering_coefficients = nx.clustering(G)
                    
                    print(f"\n{layer_name}网络:")
                    print(f"  节点数: {G.number_of_nodes()}")
                    print(f"  边数: {G.number_of_edges()}")
                    print(f"  平均度中心性: {sum(degree_centrality.values())/len(degree_centrality):.4f}")
                    print(f"  平均聚类系数: {sum(clustering_coefficients.values())/len(clustering_coefficients):.4f}")
                except Exception as e:
                    print(f"  无法计算指标: {e}")
            else:
                print(f"\n{layer_name}网络: 无有效节点或边")
    else:
        print("没有可用的时间窗口进行多层网络分析")

if __name__ == "__main__":
    # 仅构建并可视化四个静态综合网络（无时间窗口）
    print("构建并可视化静态综合网络...")
    static_builder = StaticLayerBuilder()
    # 导出目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'outputs', 'static')
    NetworkBuilder.ensure_dir(out_dir)

    # 统一 UID→编号 映射：跨所有层收集学生UID并生成编号
    try:
        layer_folders = ['call_log', 'sms', 'sensing', 'dining', 'dinning', 'app_usage', 'calendar', 'education', 'rawaccfeat']
        all_users = NetworkBuilder.get_all_users_union(layer_folders)
        uid_number_map = NetworkBuilder.build_uid_numbering(all_users)
        map_path = os.path.join(out_dir, 'uid_number_map.csv')
        NetworkBuilder.export_uid_numbering(uid_number_map, map_path)
        print(f"[Export] UID 编号映射已导出到 {map_path}，学生数 {len(uid_number_map)}")
    except Exception as e:
        print(f"生成 UID 编号映射时出错: {e}")
        uid_number_map = None

    fig_dir = os.path.join(out_dir, 'figures')

    # Communication（改为节点属性层导出）
    try:
        print("提取通信行为特征并导出为节点属性...")
        comm_feat_extractor = CommunicationFeatureExtractor()
        comm_df = comm_feat_extractor.build_node_features()
        out_features_path = os.path.join(out_dir, 'communication_features.csv')
        comm_df.to_csv(out_features_path, index=False, encoding='utf-8')
        print(f"[Export] 通信特征已导出到 {out_features_path}，学生数 {len(comm_df)}")
    except Exception as e:
        print(f"提取通信特征时出错: {e}")

    # Communication 不构建学生-学生静态图；仅导出节点通信行为特征
    print("通信层按隐私约束作为节点属性使用，不构建学生-学生边。")

    # Physical
    try:
        phys_static = static_builder.build_physical_static(mode='sum', proximity_threshold=-70, min_duration=300, time_threshold=30, normalize_sources=True)
        print(f"静态 Physical 网络：节点数 {phys_static.number_of_nodes()}，边数 {phys_static.number_of_edges()}")
        NetworkBuilder.export_edge_list(phys_static, os.path.join(out_dir, 'physical_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(phys_static, os.path.join(out_dir, 'physical_edges_weighted.csv'), weighted=True)
        print(f"[Export] 静态 Physical 边已导出到 {out_dir}")
        visualize_network(phys_static, "Physical", label_map=uid_number_map, save_dir=fig_dir)
    except Exception as e:
        print(f"构建静态 Physical 网络时出错: {e}")
        import networkx as nx
        phys_static = nx.Graph()
        NetworkBuilder.export_edge_list(phys_static, os.path.join(out_dir, 'physical_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(phys_static, os.path.join(out_dir, 'physical_edges_weighted.csv'), weighted=True)

    # Behavior
    try:
        beh_static = static_builder.build_behavior_static(mode='sum', similarity_threshold=0.7, top_k=10, normalize_sources=True)
        print(f"静态 Behavior 网络：节点数 {beh_static.number_of_nodes()}，边数 {beh_static.number_of_edges()}")
        NetworkBuilder.export_edge_list(beh_static, os.path.join(out_dir, 'behavior_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(beh_static, os.path.join(out_dir, 'behavior_edges_weighted.csv'), weighted=True)
        print(f"[Export] 静态 Behavior 边已导出到 {out_dir}")
        visualize_network(beh_static, "Behavior", label_map=uid_number_map, save_dir=fig_dir)
    except Exception as e:
        print(f"构建静态 Behavior 网络时出错: {e}")
        import networkx as nx
        beh_static = nx.Graph()
        NetworkBuilder.export_edge_list(beh_static, os.path.join(out_dir, 'behavior_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(beh_static, os.path.join(out_dir, 'behavior_edges_weighted.csv'), weighted=True)

    # Education
    try:
        edu_static = static_builder.build_education_static(network_type='co-enrollment')
        print(f"静态 Education 网络：节点数 {edu_static.number_of_nodes()}，边数 {edu_static.number_of_edges()}")
        NetworkBuilder.export_edge_list(edu_static, os.path.join(out_dir, 'education_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(edu_static, os.path.join(out_dir, 'education_edges_weighted.csv'), weighted=True)
        print(f"[Export] 静态 Education 边已导出到 {out_dir}")
        visualize_network(edu_static, "Education", label_map=uid_number_map, save_dir=fig_dir)
    except Exception as e:
        print(f"构建静态 Education 网络时出错: {e}")
        import networkx as nx
        edu_static = nx.Graph()
        NetworkBuilder.export_edge_list(edu_static, os.path.join(out_dir, 'education_edges_unweighted.csv'), weighted=False)
        NetworkBuilder.export_edge_list(edu_static, os.path.join(out_dir, 'education_edges_weighted.csv'), weighted=True)

    # 生成3D多层网络可视化
    try:
        fig3d_path = os.path.join(fig_dir, 'Multiplex3D.png')
        visualize_multiplex_3d(phys_static, beh_static, edu_static, label_map=uid_number_map, omega=1.0, save_path=fig3d_path)
    except Exception as e:
        print(f"生成3D多层网络可视化失败: {e}")

    # 导出 supra 邻接矩阵（3N x 3N）
    try:
        import pandas as pd
        import numpy as np
        # 使用之前生成的 UID→编号映射作为 label_map（若不存在则为 None）
        label_map = 'uid_number_map' in locals() and uid_number_map or None
        # 统一节点顺序（按照编号映射排序）
        def sorted_nodes(G):
            nodes = list(G.nodes())
            if label_map:
                nodes.sort(key=lambda x: label_map.get(x, 10**9))
            else:
                nodes.sort()
            return nodes

        nodes = sorted_nodes(phys_static)
        N = len(nodes)
        
        # 确保所有图包含相同节点集
        for G in [phys_static, beh_static, edu_static]:
            G.add_nodes_from(nodes)
        
        # ---------------- 额外：构建节点属性相似度矩阵（用于 composite 耦合） ----------------
        attr_matrix = None
        attr_weight = 0.4  # 属性相似度在最终相似度中的权重（可调）
        try:
            import pandas as _pd
            import numpy as _np

            # attributes 来自 NodeAttributes.collect_attributes()，之前已收集
            # 尝试加载通信特征（若已导出）作为补充数值特征
            comm_feat_path = os.path.join(out_dir, 'communication_features.csv')
            comm_df = None
            if os.path.isfile(comm_feat_path):
                try:
                    comm_df = _pd.read_csv(comm_feat_path)
                    comm_df.set_index('uid', inplace=True, drop=False)
                except Exception:
                    comm_df = None

            # 组织特征向量：优先使用 survey/ema numeric，兼容 communication_features
            feat_list = []
            for u in nodes:
                vec = []
                au = attributes.get(u, {}) if attributes else {}
                survey = au.get('survey', {}) if au else {}
                ema = au.get('ema', {}) if au else {}
                # 采集常见数值属性（可扩展）
                for k in ['age', 'social_score', 'engagement_score']:
                    try:
                        vec.append(float(survey.get(k, 0.0)))
                    except Exception:
                        vec.append(0.0)
                for k in ['avg_mood', 'avg_stress', 'response_rate']:
                    try:
                        vec.append(float(ema.get(k, 0.0)))
                    except Exception:
                        vec.append(0.0)
                # 补充通信特征行（若存在）
                if comm_df is not None and u in comm_df.index:
                    try:
                        row = comm_df.loc[u]
                        # 取所有数值列
                        for c in comm_df.columns:
                            try:
                                v = float(row[c])
                                vec.append(v)
                            except Exception:
                                continue
                    except Exception:
                        pass
                feat_list.append(vec)

            # 构建矩阵并计算余弦相似度
            M = _np.array([_np.array(v, dtype=float) if len(v) > 0 else _np.zeros(1) for v in feat_list])
            if M.size > 0:
                # 如果所有向量都是零，跳过
                norms = _np.linalg.norm(M, axis=1)
                if (norms > 0).any():
                    M_norm = M / (norms[:, None] + 1e-8)
                    A = M_norm.dot(M_norm.T)
                    # 把 cosine similarity 从 [-1,1] 映射到 [0,1]
                    A = (A + 1.0) / 2.0
                    attr_matrix = A
        except Exception as _e:
            print(f"构建属性相似度矩阵失败: {_e}")

        # 构造分层邻接矩阵
        A_phy = nx.to_numpy_array(phys_static, nodelist=nodes, weight='weight')
        A_beh = nx.to_numpy_array(beh_static, nodelist=nodes, weight='weight')
        A_edu = nx.to_numpy_array(edu_static, nodelist=nodes, weight='weight')
        
        if USE_ENHANCED_COUPLING:
            # === 增强方法：相似度加权层间耦合 ===
            print("\n[使用增强层间耦合方法]")
            layers = [
                ('Physical', phys_static),
                ('Behavior', beh_static),
                ('Education', edu_static),
            ]
            
            # 可配置参数（默认改为更宽松以便发现跨 id 的层间耦合）
            # 说明：增大 omega_sim / 降低 threshold / 允许更多 top_k 会生成更多非对角跨层边
            omega_diag = 1.0        # 对角耦合强度（保留）
            omega_sim = 0.6         # 相似度耦合强度（提高，使相似度项更显著）
            sim_method = 'jaccard'  # 相似度方法，jaccard 常被用作跨层邻居重叠的稳定指标
            threshold = 0.0         # 相似度阈值：降为 0 以减少误删潜在相似边（可在后续调回）
            top_k = None            # 每节点保留的跨层邻居数：None 表示不限制（可改为数值以减少稀疏性）
            
            print(f"  参数: omega_diag={omega_diag}, omega_sim={omega_sim}")
            print(f"  相似度方法: {sim_method}, 阈值: {threshold}, Top-K: {top_k}")
            
            A_multi, coupling_dict = InterLayerCoupling.build_supra_with_enhanced_coupling(
                layers=layers,
                nodes=nodes,
                omega_diag=omega_diag,
                omega_sim=omega_sim,
                sim_method=sim_method,
                threshold=threshold,
                top_k=top_k,
                normalize=True,
                attr_matrix=attr_matrix,
                attr_weight=attr_weight,
            )
            
            # 统计层间边
            print("  层间耦合统计:")
            for (la, lb), C in coupling_dict.items():
                if la >= lb:
                    continue
                diag_edges = np.count_nonzero(np.diag(C) > 0)
                off_diag_edges = np.count_nonzero(C - np.diag(np.diag(C)) > 0)
                print(f"    {la} <-> {lb}: 对角边={diag_edges}, 非对角边={off_diag_edges}")
            
            supra_path = os.path.join(out_dir, 'supra_adjacency_enhanced.csv')
            multi_edges_path = os.path.join(out_dir, 'multiplex_edges_enhanced.csv')
            
            # 导出层间边详细列表
            inter_edges_path = os.path.join(out_dir, 'inter_layer_edges_enhanced.csv')
            InterLayerCoupling.export_inter_layer_edges(
                coupling_dict=coupling_dict,
                nodes=nodes,
                filepath=inter_edges_path,
                weight_threshold=0.0,
            )
            print(f"  [Export] 层间边已导出到 {inter_edges_path}")
            # --- 新增：层间耦合摘要（对角 vs 非对角计数）并导出 top-k 非对角最大权重边 ---
            try:
                import numpy as _np
                import pandas as _pd

                total_diag = 0
                total_inter = 0
                inter_rows = []
                for (la, lb), C in coupling_dict.items():
                    if la >= lb:
                        continue
                    # 计数
                    total_diag += _np.count_nonzero(_np.diag(C) > 0)
                    total_inter += _np.count_nonzero(C - _np.diag(_np.diag(C)) > 0)
                    # 收集非对角边用于排序
                    Nloc = C.shape[0]
                    for i in range(Nloc):
                        for j in range(Nloc):
                            if i == j:
                                continue
                            w = float(C[i, j])
                            if w > 0:
                                inter_rows.append({
                                    'src_layer': la,
                                    'dst_layer': lb,
                                    'u': nodes[i],
                                    'v': nodes[j],
                                    'weight': w,
                                })

                print(f"  层间耦合汇总: total_diag={total_diag}, total_inter={total_inter}, total_inter_records={len(inter_rows)}")

                # 导出 top 20 非对角权重最大边
                if inter_rows:
                    inter_df = _pd.DataFrame(inter_rows)
                    inter_df_sorted = inter_df.sort_values('weight', ascending=False)
                    topk = 20
                    top_df = inter_df_sorted.head(topk)
                    top_path = os.path.join(out_dir, 'inter_layer_top_inter_edges.csv')
                    top_df.to_csv(top_path, index=False, encoding='utf-8')
                    print(f"  [Export] Top {topk} 非对角层间边已导出到 {top_path}")
            except Exception as _e:
                print(f"  无法生成层间耦合摘要: {_e}")
            
        else:
            # === 传统方法：omega * I 对角连接 ===
            print("\n[使用传统对角耦合方法]")
            I = np.eye(N) * 1.0
            omega = 1.0
            top = np.hstack([A_phy, omega * I, omega * I])
            mid = np.hstack([omega * I, A_beh, omega * I])
            bot = np.hstack([omega * I, omega * I, A_edu])
            A_multi = np.vstack([top, mid, bot])
            
            supra_path = os.path.join(out_dir, 'supra_adjacency.csv')
            multi_edges_path = os.path.join(out_dir, 'multiplex_edges_weighted.csv')
            coupling_dict = None

        # 导出 supra 邻接矩阵
        pd.DataFrame(A_multi).to_csv(supra_path, index=False)
        print(f"[Export] Supra 邻接矩阵已导出到 {supra_path}，维度 {A_multi.shape}")

        # 导出多层边列表（含层内与层间）
        rows = []
        # 层内边
        for lname, A in [('phy', A_phy), ('beh', A_beh), ('edu', A_edu)]:
            for i in range(N):
                for j in range(N):
                    w = float(A[i, j])
                    if w > 0:
                        rows.append({
                            'type': 'intra',
                            'src_layer': lname,
                            'dst_layer': lname,
                            'u': nodes[i],
                            'v': nodes[j],
                            'weight': w
                        })
        
        # 层间边
        if USE_ENHANCED_COUPLING and coupling_dict:
            # 增强方法：从 coupling_dict 导出
            layer_map = {'Physical': 'phy', 'Behavior': 'beh', 'Education': 'edu'}
            for (la, lb), C in coupling_dict.items():
                if la >= lb:
                    continue
                la_short = layer_map.get(la, la)
                lb_short = layer_map.get(lb, lb)
                for i in range(N):
                    for j in range(N):
                        w = float(C[i, j])
                        if w > 0:
                            rows.append({
                                'type': 'inter',
                                'src_layer': la_short,
                                'dst_layer': lb_short,
                                'u': nodes[i],
                                'v': nodes[j],
                                'weight': w
                            })
        else:
            # 传统方法：对角连接
            omega = 1.0
            for i in range(N):
                uid = nodes[i]
                for pair in [('phy','beh'),('phy','edu'),('beh','edu')]:
                    rows.append({
                        'type': 'inter',
                        'src_layer': pair[0],
                        'dst_layer': pair[1],
                        'u': uid,
                        'v': uid,
                        'weight': omega
                    })

        pd.DataFrame(rows).to_csv(multi_edges_path, index=False, encoding='utf-8')
        print(f"[Export] 多层边列表已导出到 {multi_edges_path}，记录数 {len(rows)}")
        
    except Exception as e:
        import traceback
        print(f"导出 supra 邻接矩阵/多层边失败: {e}")
        traceback.print_exc()

    # ============ 社群划分与可视化 ============
    print("\n[社群划分与可视化]")
    try:
        # 加载社群标签（若已生成）
        comm_path = os.path.join(out_dir, 'communities_uid.csv')
        if os.path.isfile(comm_path):
            import pandas as pd
            comm_df = pd.read_csv(comm_path)
            communities_dict = dict(zip(comm_df['uid'], comm_df['community']))
            print(f"  加载社群标签：{len(communities_dict)} 个节点，{len(set(communities_dict.values()))} 个社群")
            
            # 可视化各层网络的社群划分
            print("  生成社群可视化图...")
            visualize_network_with_communities(phys_static, communities_dict, "Physical Layer", label_map=uid_number_map, save_dir=fig_dir)
            visualize_network_with_communities(beh_static, communities_dict, "Behavior Layer", label_map=uid_number_map, save_dir=fig_dir)
            visualize_network_with_communities(edu_static, communities_dict, "Education Layer", label_map=uid_number_map, save_dir=fig_dir)
            
            # 生成社群统计图表
            print("  生成社群统计图表...")
            try:
                import subprocess
                venv_python = os.path.join(base_dir, 'student-complex', 'Scripts', 'python.exe')
                comm_fig_script = os.path.join(base_dir, 'scripts', 'generate_comm_fig.py')
                if os.path.isfile(venv_python) and os.path.isfile(comm_fig_script):
                    subprocess.run([venv_python, comm_fig_script], check=True)
                else:
                    print(f"  提示：可手动运行 scripts/generate_comm_fig.py 生成更多社群图表")
            except Exception as e:
                print(f"  自动生成社群图表失败: {e}")
                print(f"  请手动运行: python scripts/generate_comm_fig.py")
        else:
            print(f"  未找到社群标签文件 {comm_path}")
            print(f"  请先运行: python scripts/detect_communities.py")
    except Exception as e:
        print(f"社群可视化失败: {e}")
        import traceback
        traceback.print_exc()