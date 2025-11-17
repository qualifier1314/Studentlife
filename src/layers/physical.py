from typing import Dict, Set, Optional, Tuple
import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta
from ..utils import NetworkBuilder

class SensingNetwork:
    """物理接触网络构建类（基于蓝牙/WiFi近距离检测）"""
    
    def __init__(self):
        self.folder = "sensing"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     proximity_threshold: float = None,  # 近距离阈值（取决于数据中的距离/信号强度单位）
                     min_duration: int = 300,  # 最小接触持续时间（秒）
                     time_window: Optional[str] = 'D'
                     ) -> Dict[str, nx.Graph]:
        """构建物理接触网络
        
        Args:
            proximity_threshold: 判定为近距离接触的阈值
            min_duration: 最小接触持续时间（秒）
            time_window: 时间窗口
            
        Returns:
            Dict[str, nx.Graph]: 时间窗口到对应网络的映射
        """
        users = self.builder.get_all_users(self.folder)
        networks: Dict[str, nx.Graph] = {}
        
        if time_window is None:
            networks['all'] = self.builder.create_base_graph(users)
            
        durations_by_window: Dict[str, Dict[Tuple[str, str], float]] = {}
        last_contact_by_window: Dict[str, Dict[Tuple[str, str], pd.Timestamp]] = {}
        
        folder_path = self.builder.resolve_folder(self.folder)
        for f in glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True):
            uid = self.builder.user_from_fname(f)
            try:
                # 修改：添加错误处理参数以避免解析错误
                df = pd.read_csv(f, on_bad_lines='skip', engine='python')
            except Exception as e:
                print(f"警告: 无法读取文件 {f}: {e}，跳过该文件")
                continue
            
            # 标准化列名（根据实际数据调整）
            df.columns = [c.lower() for c in df.columns]
            timestamp_col = next((c for c in df.columns if 'time' in c.lower()), None)
            # 修改：提供默认值避免StopIteration异常
            target_col = next((c for c in df.columns 
                            if any(x in c.lower() for x in ['target', 'other', 'detected'])), None) or 'target'
            # 修改：提供默认值避免StopIteration异常
            strength_col = next((c for c in df.columns 
                               if any(x in c.lower() for x in ['strength', 'distance', 'rssi'])), None) or 'strength'
            
            # 检查必要列是否存在
            if timestamp_col is None:
                print(f"警告: 文件 {f} 中未找到时间戳列，跳过该文件")
                continue
            
            if target_col is None:
                print(f"警告: 文件 {f} 中未找到目标用户列，跳过该文件")
                continue
            
            # 修改：添加错误处理
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            except Exception as e:
                print(f"警告: 文件 {f} 中时间戳列解析出错: {e}，跳过该文件")
                continue
            
            if time_window:
                try:
                    time_groups = self.builder.time_window_split(df, timestamp_col, time_window)
                    for time_key, group_df in time_groups.items():
                        if time_key not in networks:
                            networks[time_key] = self.builder.create_base_graph(users)
                        durations_by_window.setdefault(time_key, {})
                        last_contact_by_window.setdefault(time_key, {})
                        self._process_contacts(networks[time_key], uid, group_df,
                                               timestamp_col, target_col, strength_col,
                                               proximity_threshold, min_duration,
                                               durations_by_window[time_key], last_contact_by_window[time_key])
                except Exception as e:
                    print(f"警告: 处理文件 {f} 时间分组时出错: {e}，跳过该文件")
                    continue
            else:
                key = 'all'
                if key not in networks:
                    networks[key] = self.builder.create_base_graph(users)
                durations_by_window.setdefault(key, {})
                last_contact_by_window.setdefault(key, {})
                self._process_contacts(networks[key], uid, df,
                                       timestamp_col, target_col, strength_col,
                                       proximity_threshold, min_duration,
                                       durations_by_window[key], last_contact_by_window[key])
        
        # 将累计的接触时间添加为边权重
        for time_key, G in networks.items():
            for (u1, u2), duration in durations_by_window.get(time_key, {}).items():
                if duration >= min_duration:
                    G.add_edge(u1, u2, weight=duration)
            self.builder.normalize_weights(G)
            
        return networks
    
    def _process_contacts(self,
                         G: nx.Graph,
                         user: str,
                         df: pd.DataFrame,
                         timestamp_col: str,
                         target_col: str,
                         strength_col: Optional[str],
                         proximity_threshold: Optional[float],
                         min_duration: int,
                         contact_durations: Dict[Tuple[str, str], float],
                         last_contact: Dict[Tuple[str, str], pd.Timestamp]):
        """处理接触数据"""
        for _, row in df.iterrows():
            target = str(row[target_col])
            if pd.isna(target) or user == target:
                continue
                
            if strength_col and proximity_threshold is not None:
                try:
                    val = float(row[strength_col])
                except Exception:
                    val = None
                if val is not None:
                    col = strength_col.lower()
                    if 'rssi' in col:
                        if val < float(proximity_threshold):
                            continue
                    else:
                        if val > float(proximity_threshold):
                            continue
            
            curr_time = row[timestamp_col]
            pair = tuple(sorted([user, target]))
            
            # 更新接触持续时间
            if pair in last_contact:
                time_diff = (curr_time - last_contact[pair]).total_seconds()
                if time_diff <= 300:  # 假设5分钟内的检测为连续接触
                    if pair not in contact_durations:
                        contact_durations[pair] = 0
                    contact_durations[pair] += time_diff
            
            last_contact[pair] = curr_time

class DiningNetwork:
    """共享就餐位置网络构建类"""
    
    def __init__(self):
        self.folder = "dining"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     time_threshold: int = 30,
                     time_window: Optional[str] = 'D'
                     ) -> Dict[str, nx.Graph]:
        """构建共同就餐网络
        
        Args:
            time_threshold: 判定为同时就餐的时间差阈值（分钟）
            time_window: 时间窗口
            
        Returns:
            Dict[str, nx.Graph]: 时间窗口到对应网络的映射
        """
        # 兼容 d\u2192dining/dinning 两种拼写
        users = set()
        users.update(self.builder.get_all_users("dining"))
        users.update(self.builder.get_all_users("dinning"))
        networks: Dict[str, nx.Graph] = {}
        
        if time_window is None:
            networks['all'] = self.builder.create_base_graph(users)
            
        # 存储用户就餐记录
        dining_records: Dict[str, Dict[str, list]] = {}
        
        # 同样兼容两种目录名
        for folder_name in ("dining", "dinning"):
            folder_path = self.builder.resolve_folder(folder_name)
            for f in glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True) + \
                     glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True):
                uid = self.builder.user_from_fname(f)
                df = None
                try:
                    df = pd.read_csv(f, on_bad_lines='skip', engine='python')
                except Exception:
                    try:
                        df = pd.read_csv(f, sep='\t', engine='python', header=None)
                    except Exception:
                        try:
                            df = pd.read_csv(f, delim_whitespace=True, engine='python', header=None)
                        except Exception:
                            print(f"警告: 无法解析文件 {f}")
                            continue
                df.columns = [str(c).lower() for c in df.columns]
                timestamp_col = next((c for c in df.columns if 'time' in c), None)
                location_col = next((c for c in df.columns if any(x in c for x in ['location','place','dining'])), None)
                if timestamp_col is None and len(df.columns) >= 2:
                    timestamp_col = df.columns[0]
                if location_col is None and len(df.columns) >= 2:
                    location_col = df.columns[1]
                if timestamp_col is None or location_col is None:
                    print(f"警告: 文件 {f} 缺少时间或位置列，跳过")
                    continue
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                except Exception:
                    print(f"警告: 文件 {f} 时间列解析失败，跳过")
                    continue
                if time_window:
                    time_groups = self.builder.time_window_split(df, timestamp_col, time_window)
                    for time_key, group_df in time_groups.items():
                        if time_key not in networks:
                            networks[time_key] = self.builder.create_base_graph(users)
                        dining_records.setdefault(time_key, {})
                        self._process_dining_records(dining_records[time_key], uid, group_df, timestamp_col, location_col)
                else:
                    key = 'all'
                    dining_records.setdefault(key, {})
                    self._process_dining_records(dining_records[key], uid, df, timestamp_col, location_col)
        
        # 构建共同就餐网络
        for time_key, records in dining_records.items():
            G = networks[time_key]
            self._build_co_dining_edges(G, records, time_threshold)
            self.builder.normalize_weights(G)
            
        return networks
    
    def _process_dining_records(self,
                               records: Dict[str, list],
                               user: str,
                               df: pd.DataFrame,
                               timestamp_col: str,
                               location_col: str):
        """处理就餐记录"""
        if user not in records:
            records[user] = []
            
        for _, row in df.iterrows():
            time = row[timestamp_col]
            location = str(row[location_col])
            if pd.isna(location):
                continue
            
            records[user].append((location, time))
    
    def _build_co_dining_edges(self,
                              G: nx.Graph,
                              records: Dict[str, list],
                              time_threshold: int):
        """构建共同就餐网络边"""
        for u1 in records:
            for u2 in records:
                if u1 >= u2:  # 避免重复
                    continue
                    
                co_dining_count = 0
                for loc1, t1 in records[u1]:
                    for loc2, t2 in records[u2]:
                        if loc1 == loc2 and \
                           abs((t1 - t2).total_seconds()) <= time_threshold * 60:
                            co_dining_count += 1
                
                if co_dining_count > 0:
                    G.add_edge(u1, u2, weight=co_dining_count)