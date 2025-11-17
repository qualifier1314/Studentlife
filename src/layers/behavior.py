from typing import Dict, Set, Optional, Tuple, List
import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
import sqlite3
from collections import defaultdict
from ..utils import NetworkBuilder

class AppUsageNetwork:
    """应用使用行为网络构建类"""
    
    def __init__(self):
        self.folder = "app_usage"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     network_type: str = 'similarity',
                     similarity_threshold: float = 0.2,
                     top_k: Optional[int] = None,
                     time_window: Optional[str] = None
                     ) -> Dict[str, nx.Graph]:
        """构建应用使用网络
        
        Args:
            network_type: 网络类型，'similarity'为用户相似度网络，'bipartite'为用户-应用二分网络
            similarity_threshold: 相似度阈值（用于similarity类型）
            top_k: 每个用户保留的最相似邻居数量（用于similarity类型）
            time_window: 时间窗口
            
        Returns:
            Dict[str, nx.Graph]: 时间窗口到对应网络的映射
        """
        users = self.builder.get_all_users(self.folder)
        networks: Dict[str, nx.Graph] = {}

        if time_window is None:
            networks['all'] = self.builder.create_base_graph(users)

        app_usage_by_window: Dict[str, Dict[str, Dict[str, int]]] = {}
        all_apps_by_window: Dict[str, Set[str]] = {}
        users_seen_by_window: Dict[str, Set[str]] = {}
        folder_path = self.builder.resolve_folder(self.folder)
        for f in glob.glob(os.path.join(folder_path, "*.csv")):
            uid = self.builder.user_from_fname(f)
            try:
                df = pd.read_csv(f, on_bad_lines='skip', engine='python')
            except Exception as e:
                print(f"警告: 无法读取文件 {f}: {e}，跳过该文件")
                continue

            df.columns = [c.lower() for c in df.columns]
            app_col = next((c for c in df.columns if any(x in c for x in ['app','package','application'])), None)
            timestamp_col = next((c for c in df.columns if 'time' in c), None)
            if app_col is None:
                continue

            if time_window and timestamp_col:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                    time_groups = self.builder.time_window_split(df, timestamp_col, time_window)
                    for time_key, group_df in time_groups.items():
                        app_usage_by_window.setdefault(time_key, defaultdict(lambda: defaultdict(int)))
                        all_apps_by_window.setdefault(time_key, set())
                        users_seen_by_window.setdefault(time_key, set()).add(uid)
                        self._collect_app_usage(app_usage_by_window[time_key], all_apps_by_window[time_key], uid, group_df, app_col)
                except Exception as e:
                    print(f"警告: 处理文件 {f} 时间分组时出错: {e}，跳过该文件")
                    continue
            else:
                time_key = 'all'
                app_usage_by_window.setdefault(time_key, defaultdict(lambda: defaultdict(int)))
                all_apps_by_window.setdefault(time_key, set())
                users_seen_by_window.setdefault(time_key, set()).add(uid)
                self._collect_app_usage(app_usage_by_window[time_key], all_apps_by_window[time_key], uid, df, app_col)

        for time_key, usage_dict in app_usage_by_window.items():
            users_list = sorted(users_seen_by_window.get(time_key, list(users)))
            all_apps = sorted(all_apps_by_window.get(time_key, set()))
            app_index = {app: i for i, app in enumerate(all_apps)}
            user_app_matrix = np.zeros((len(users_list), len(all_apps)))

            for i, user in enumerate(users_list):
                if user in usage_dict:
                    for app, count in usage_dict[user].items():
                        j = app_index.get(app)
                        if j is not None:
                            user_app_matrix[i, j] = count

            G = self.builder.create_base_graph(set(users_list))
            if network_type == 'similarity':
                sim_matrix = self._cosine_similarity(user_app_matrix)
                for i in range(len(users_list)):
                    candidates = [(j, float(sim_matrix[i, j])) for j in range(i+1, len(users_list)) if sim_matrix[i, j] >= similarity_threshold]
                    if top_k:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        candidates = candidates[:top_k]
                    for j, sim in candidates:
                        G.add_edge(users_list[i], users_list[j], weight=sim)
                self.builder.normalize_weights(G)
            else:
                G.clear()
                G.add_nodes_from(users_list, bipartite=0)
                G.add_nodes_from(all_apps, bipartite=1)
                for i, user in enumerate(users_list):
                    for app, j in app_index.items():
                        w = user_app_matrix[i, j]
                        if w > 0:
                            G.add_edge(user, app, weight=w)

            networks[time_key] = G

        return networks
    
    # 添加手动实现的余弦相似度计算函数
    def _cosine_similarity(self, matrix):
        """计算矩阵行向量之间的余弦相似度"""
        # 归一化向量
        normalized = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        # 计算余弦相似度
        return np.dot(normalized, normalized.T)
    
    def _collect_app_usage(self,
                          app_usage: Dict,
                          all_apps: Set[str],
                          user: str,
                          df: pd.DataFrame,
                          app_col: str):
        """收集应用使用数据"""
        for _, row in df.iterrows():
            app = str(row[app_col])
            if pd.isna(app):
                continue
            app_usage[user][app] += 1
            all_apps.add(app)

class CalendarNetwork:
    """日历事件网络构建类"""
    
    def __init__(self):
        self.folder = "calendar"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     network_type: str = 'co-occurrence',
                     time_window: Optional[str] = None
                     ) -> Dict[str, nx.Graph]:
        """构建日历事件网络
        
        Args:
            network_type: 网络类型，'co-occurrence'为共同参与网络，'bipartite'为用户-事件二分网络
            time_window: 时间窗口
            
        Returns:
            Dict[str, nx.Graph]: 时间窗口到对应网络的映射
        """
        users = self.builder.get_all_users(self.folder)
        networks: Dict[str, nx.Graph] = {}

        if time_window is None:
            networks['all'] = self.builder.create_base_graph(users)

        events_by_window: Dict[str, Dict[str, Set[str]]] = {}
        user_events_by_window: Dict[str, Dict[str, Set[str]]] = {}
        event_times_by_window: Dict[str, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = {}

        folder_path = self.builder.resolve_folder(self.folder)
        for f in glob.glob(os.path.join(folder_path, "*.csv")):
            uid = self.builder.user_from_fname(f)
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            event_col = next((c for c in df.columns if any(x in c for x in ['event','id'])), None)
            start_col = next((c for c in df.columns if any(x in c for x in ['start','begin','time'])), None)
            end_col = next((c for c in df.columns if 'end' in c), None)
            attendees_col = next((c for c in df.columns if any(x in c for x in ['attendee','participant'])), None)
            if not event_col or not start_col:
                print(f"警告: 文件 {f} 缺少事件或开始时间列，跳过该文件")
                continue

            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
            if end_col:
                df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

            if time_window:
                time_groups = self.builder.time_window_split(df, start_col, time_window)
                for time_key, group_df in time_groups.items():
                    events_by_window.setdefault(time_key, defaultdict(set))
                    user_events_by_window.setdefault(time_key, defaultdict(set))
                    event_times_by_window.setdefault(time_key, {})
                    if time_key not in networks:
                        networks[time_key] = self.builder.create_base_graph(users)
                    self._collect_event_data(events_by_window[time_key], user_events_by_window[time_key], event_times_by_window[time_key], uid, group_df, event_col, start_col, end_col, attendees_col)
            else:
                key = 'all'
                events_by_window.setdefault(key, defaultdict(set))
                user_events_by_window.setdefault(key, defaultdict(set))
                event_times_by_window.setdefault(key, {})
                self._collect_event_data(events_by_window[key], user_events_by_window[key], event_times_by_window[key], uid, df, event_col, start_col, end_col, attendees_col)

        for time_key, G in networks.items():
            events = events_by_window.get(time_key, {})
            user_events = user_events_by_window.get(time_key, {})
            event_times = event_times_by_window.get(time_key, {})
            if network_type == 'co-occurrence':
                for event_id, participants in events.items():
                    plist = sorted(list(participants))
                    for i in range(len(plist)):
                        for j in range(i+1, len(plist)):
                            u1, u2 = plist[i], plist[j]
                            if G.has_edge(u1, u2):
                                G[u1][u2]['weight'] += 1
                            else:
                                G.add_edge(u1, u2, weight=1)
                self.builder.normalize_weights(G)
            else:
                G.clear()
                G.add_nodes_from(set(user_events.keys()), bipartite=0)
                G.add_nodes_from(set(events.keys()), bipartite=1)
                for user, user_event_set in user_events.items():
                    for event_id in user_event_set:
                        G.add_edge(user, event_id)
                        if event_id in event_times:
                            start, end = event_times[event_id]
                            duration = (end - start).total_seconds() if (start is not None and end is not None) else 3600
                            G[user][event_id]['weight'] = duration

        return networks
    
    def _collect_event_data(self,
                           events: Dict[str, Set[str]],
                           user_events: Dict[str, Set[str]],
                           event_times: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
                           user: str,
                           df: pd.DataFrame,
                           event_col: str,
                           start_col: str,
                           end_col: Optional[str],
                           attendees_col: Optional[str]):
        """收集事件参与数据"""
        for _, row in df.iterrows():
            event_id = str(row[event_col])
            if pd.isna(event_id):
                continue
                
            # 记录事件时间
            start_time = row[start_col]
            end_time = row[end_col] if end_col else None
            event_times[event_id] = (start_time, end_time)
            
            # 添加参与者
            if attendees_col and not pd.isna(row[attendees_col]):
                # 假设attendees是分号分隔的用户列表
                attendees = [a.strip() for a in str(row[attendees_col]).split(';')]
                events[event_id].update(attendees)
                for attendee in attendees:
                    user_events[attendee].add(event_id)
            else:
                # 如果没有明确的参与者列表，就只添加当前用户
                events[event_id].add(user)
                user_events[user].add(event_id)

# // 添加新的RawAccFeatNetwork类
class RawAccFeatNetwork:
    """原始加速度特征网络构建类"""
    
    def __init__(self):
        self.folder = "rawaccfeat"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     similarity_threshold: float = 0.7,
                     top_k: Optional[int] = None,
                     time_window: Optional[str] = 'W'
                     ) -> Dict[str, nx.Graph]:
        """构建行为同步网络
        
        Args:
            similarity_threshold: 行为同步性阈值
            top_k: 每个用户保留的最相似邻居数量
            time_window: 时间窗口，默认按周
            
        Returns:
            Dict[str, nx.Graph]: 时间窗口到对应网络的映射
        """
        users = self.builder.get_all_users(self.folder)
        networks: Dict[str, nx.Graph] = {}

        if time_window is None:
            networks['all'] = self.builder.create_base_graph(users)

        acc_features = defaultdict(dict)
        folder_path = self.builder.resolve_folder(self.folder)

        for f in glob.glob(os.path.join(folder_path, "*.csv")):
            try:
                uid = self.builder.user_from_fname(f)
                df = pd.read_csv(f)
            except Exception as e:
                print(f"警告: 无法读取文件 {f}: {e}，跳过该文件")
                continue
            df.columns = [c.lower() for c in df.columns]
            timestamp_col = next((c for c in df.columns if 'time' in c), None)
            feature_cols = [c for c in df.columns if c != timestamp_col]
            if time_window and timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                time_groups = self.builder.time_window_split(df, timestamp_col, time_window)
                for time_key, group_df in time_groups.items():
                    if time_key not in networks:
                        networks[time_key] = self.builder.create_base_graph(users)
                    if feature_cols:
                        acc_features[uid][time_key] = group_df[feature_cols].mean(numeric_only=True).values
            else:
                if feature_cols:
                    acc_features[uid]['all'] = df[feature_cols].mean(numeric_only=True).values

        for f in glob.glob(os.path.join(folder_path, "*.db")):
            uid = self.builder.user_from_fname(f)
            try:
                conn = sqlite3.connect(f)
                cur = conn.cursor()
                tbl = None
                for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                    tbl = row[0]
                    break
                if tbl is None:
                    conn.close()
                    continue
                df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
                conn.close()
            except Exception as e:
                print(f"警告: 读取数据库 {f} 出错: {e}")
                continue
            df.columns = [str(c).lower() for c in df.columns]
            ts_candidates = [c for c in df.columns if 'time' in c or 'timestamp' in c]
            timestamp_col = ts_candidates[0] if ts_candidates else None
            numeric_cols = [c for c in df.columns if c != timestamp_col]
            if timestamp_col is not None and time_window:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                except Exception:
                    pass
                time_groups = self.builder.time_window_split(df, timestamp_col, time_window)
                for time_key, group_df in time_groups.items():
                    if time_key not in networks:
                        networks[time_key] = self.builder.create_base_graph(users)
                    if numeric_cols:
                        acc_features[uid][time_key] = group_df[numeric_cols].mean(numeric_only=True).values
            else:
                if numeric_cols:
                    acc_features[uid]['all'] = df[numeric_cols].mean(numeric_only=True).values

        for time_key, G in networks.items():
            users_with_features = [u for u in acc_features.keys() if time_key in acc_features[u]]
            if len(users_with_features) < 2:
                continue
            feature_matrix = np.array([acc_features[u][time_key] for u in users_with_features])
            if feature_matrix.size == 0:
                continue
            corr = np.corrcoef(feature_matrix)
            for i in range(len(users_with_features)):
                pairs = [(j, float(corr[i, j])) for j in range(i+1, len(users_with_features)) if corr[i, j] >= similarity_threshold]
                if top_k:
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    pairs = pairs[:top_k]
                for j, w in pairs:
                    G.add_edge(users_with_features[i], users_with_features[j], weight=w)
            self.builder.normalize_weights(G)

        return networks
