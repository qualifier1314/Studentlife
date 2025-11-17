import networkx as nx
from typing import Dict, List, Set, Optional

from .communication import CallLogNetwork, SMSNetwork
from .physical import SensingNetwork, DiningNetwork
from .behavior import AppUsageNetwork, CalendarNetwork
from .education import EducationNetwork
from ..utils import NetworkBuilder


class StaticLayerBuilder:
    """构建四个静态综合单层网络：communication、physical、behavior、education。

    - communication: 合并通话与短信网络
    - physical: 合并物理接近与共同就餐网络
    - behavior: 合并日历共现与行为同步网络
    - education: 教育网络（同班/二分图），静态
    """

    def __init__(self):
        self.builder = NetworkBuilder()
        # 统一学生节点集：跨三层（physical/behavior/education）确保相同节点集合
        self._global_user_dirs = [
            'sensing', 'dining', 'dinning',  # physical
            'calendar', 'app_usage',         # behavior
            'education'                      # education
        ]

    def _get_global_users(self) -> Set[str]:
        """返回全局学生UID集合，用于在三层之间保持一致的节点集。"""
        try:
            return self.builder.get_all_users_union(self._global_user_dirs)
        except Exception:
            # 回退：逐层拼接
            users: Set[str] = set()
            for d in self._global_user_dirs:
                try:
                    users |= self.builder.get_all_users(d)
                except Exception:
                    continue
            return users

    # --------------------------- 通用合并工具 ---------------------------
    def _merge_graphs(
        self,
        graphs: List[nx.Graph],
        users: Set[str],
        mode: str = 'binary',  # 'binary' 或 'sum'
        normalize_sources: bool = True,
        final_normalize: bool = False,
        only_student_edges: bool = True,
    ) -> nx.Graph:
        """将多个图合并为静态图。

        Args:
            graphs: 待合并的图列表
            users: 节点集（学生 UID 集合）
            mode: 合并模式，'binary' 为任一源有边则置 1；'sum' 为源权重相加
            normalize_sources: 是否先对每个源图做权重归一化（便于不同量纲相加）
            final_normalize: 是否对合并后的图再做一次归一化
        """
        G = self.builder.create_base_graph(users)

        for H in graphs:
            if H is None:
                continue
            Hn = H.copy()
            if normalize_sources:
                Hn = self.builder.normalize_weights(Hn)
            for u, v, d in Hn.edges(data=True):
                # 可选：仅保留学生 UID 节点之间的边
                if only_student_edges and ((u not in users) or (v not in users)):
                    continue
                w = d.get('weight', 1.0)
                if mode == 'binary':
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=1.0)
                else:  # sum
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += float(w)
                    else:
                        G.add_edge(u, v, weight=float(w))

        if final_normalize:
            G = self.builder.normalize_weights(G)
        return G

    # --------------------------- Communication ---------------------------
    def build_communication_static(
        self,
        mode: str = 'binary',  # 'binary' 或 'sum'
        call_weight_type: str = 'duration',  # 'duration' 或 'count'
        normalize_sources: bool = True,
        allow_external: bool = False,
    ) -> nx.Graph:
        """静态 communication 层：合并通话与短信网络。"""
        users = self.builder.get_all_users('call_log') | self.builder.get_all_users('sms')

        call_net = CallLogNetwork()
        sms_net = SMSNetwork()

        try:
            call_res = call_net.build_network(weight_type=call_weight_type, time_window=None)
            call_G = call_res['all'] if isinstance(call_res, dict) else call_res
        except Exception:
            call_G = nx.Graph()
            call_G.add_nodes_from(users)

        try:
            sms_res = sms_net.build_network(weight_type='count', time_window=None)
            sms_G = sms_res['all'] if isinstance(sms_res, dict) else sms_res
        except Exception:
            sms_G = nx.Graph()
            sms_G.add_nodes_from(users)

        return self._merge_graphs(
            [call_G, sms_G],
            users,
            mode=mode,
            normalize_sources=normalize_sources,
            only_student_edges=not allow_external,
        )

    # --------------------------- Physical ---------------------------
    def build_physical_static(
        self,
        mode: str = 'binary',
        proximity_threshold: Optional[float] = None,
        min_duration: int = 300,
        time_threshold: int = 30,
        normalize_sources: bool = True,
    ) -> nx.Graph:
        """静态 physical 层：合并物理接近与共同就餐网络。"""
        users = self._get_global_users()

        sense_net = SensingNetwork()
        dine_net = DiningNetwork()

        try:
            sense_res = sense_net.build_network(
                proximity_threshold=proximity_threshold,
                min_duration=min_duration,
                time_window=None,
            )
            sense_G = sense_res['all'] if isinstance(sense_res, dict) else sense_res
        except Exception:
            sense_G = nx.Graph()
            sense_G.add_nodes_from(users)

        try:
            dine_res = dine_net.build_network(
                time_threshold=time_threshold,
                time_window=None,
            )
            dine_G = dine_res['all'] if isinstance(dine_res, dict) else dine_res
        except Exception:
            dine_G = nx.Graph()
            dine_G.add_nodes_from(users)

        return self._merge_graphs([sense_G, dine_G], users, mode=mode, normalize_sources=normalize_sources)

    # --------------------------- Behavior ---------------------------
    def build_behavior_static(
        self,
        mode: str = 'binary',
        similarity_threshold: float = 0.2,
        top_k: int = 10,
        normalize_sources: bool = True,
    ) -> nx.Graph:
        """静态 behavior 层：合并日历共现与应用使用相似度网络。"""
        users = self._get_global_users()

        cal_net = CalendarNetwork()
        app_net = AppUsageNetwork()

        # 日历共现
        try:
            cal_res = cal_net.build_network(network_type='co-occurrence', time_window=None)
            cal_G = cal_res['all'] if isinstance(cal_res, dict) else cal_res
        except Exception:
            cal_G = nx.Graph()
            cal_G.add_nodes_from(users)

        # 应用使用相似度网络
        try:
            app_res = app_net.build_network(
                network_type='similarity',
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                time_window=None,
            )
            app_G = app_res['all'] if isinstance(app_res, dict) else app_res
        except Exception:
            app_G = nx.Graph()
            app_G.add_nodes_from(users)

        return self._merge_graphs([cal_G, app_G], users, mode=mode, normalize_sources=normalize_sources)

    # --------------------------- Education ---------------------------
    def build_education_static(
        self,
        network_type: str = 'co-enrollment',  # 或 'bipartite'
    ) -> nx.Graph:
        """静态 education 层：教育网络。"""
        users = self._get_global_users()
        edu_net = EducationNetwork()
        try:
            edu_res = edu_net.build_network(network_type=network_type, time_window=None)
            edu_G = edu_res['all'] if isinstance(edu_res, dict) else edu_res
        except Exception:
            edu_G = nx.Graph()
            edu_G.add_nodes_from(users)
        return edu_G