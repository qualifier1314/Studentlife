import pandas as pd
import networkx as nx
import os
import glob
from datetime import datetime
from collections import defaultdict

class CallLogNetwork:
    def __init__(self):
        from ..utils import NetworkBuilder
        self.folder = "call_log"
        self.builder = NetworkBuilder()
        folder_path = self.builder.resolve_folder(self.folder)
        self.user_files = glob.glob(os.path.join(folder_path, "call_log_*.csv"))
    
    def build_network(self, weight_type='count', time_window='D', phone_uid_map=None, phone_uid_map_path=None):
        """
        构建通话网络
        weight_type: 'count', 'duration'
        time_window: 'D'(daily), 'W'(weekly), 'M'(monthly)
        """
        graphs = {}
        # 加载号码→UID映射（若提供路径或可自动发现）
        try:
            from ..utils import NetworkBuilder
            if phone_uid_map is None:
                phone_uid_map = NetworkBuilder.load_phone_uid_map(phone_uid_map_path)
        except Exception:
            phone_uid_map = {}
        
        for file_path in self.user_files:
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
                
                # 检查必要的列是否存在
                df.columns = [str(c).strip().lower() for c in df.columns]
                uid = self.builder.user_from_fname(file_path)
                # 统一时间列名
                time_col = next((c for c in df.columns if any(x in c for x in ['timestamp','datetime','time','date','start'])), None)
                # 对端列
                peer_col = next((c for c in df.columns if any(x in c for x in ['callee','other','target','contact','peer','to','number','id'])), None)
                # 主体列
                caller_col = next((c for c in df.columns if any(x in c for x in ['caller','source','from','user'])), None)
                # 时长列
                dur_col = next((c for c in df.columns if any(x in c for x in ['duration','length','dur','seconds','sec'])), None)
                
                if time_col is None:
                    print(f"警告: 文件 {file_path} 中未找到时间列，跳过该文件")
                    continue
                if peer_col is None and ('callee' not in df.columns):
                    print(f"警告: 文件 {file_path} 中未找到对端用户列，跳过该文件")
                    continue
                
                # 使用统一切片逻辑：若无精确时间或无时间列 -> 静态('all')
                from ..utils import NetworkBuilder
                grouped = NetworkBuilder.split_or_all(df, time_col, time_window)
                for time_key, group in grouped.items():
                    if time_key not in graphs:
                        graphs[time_key] = nx.Graph()
                    
                    G = graphs[time_key]
                    
                    # 根据权重类型计算边权重
                    for _, row in group.iterrows():
                        # 固定主叫为文件名中的UID，避免电话号码干扰
                        caller = uid
                        callee_val = row[peer_col] if peer_col in group.columns else row.get('callee', None)
                        if pd.isna(callee_val):
                            continue
                        callee_raw = str(callee_val)
                        # 映射被叫为UID（若可能）
                        if callee_raw.startswith('u'):
                            callee = callee_raw
                        else:
                            norm = NetworkBuilder.normalize_phone(callee_raw)
                            callee = phone_uid_map.get(norm, '')
                        if not callee:
                            # 若无法映射为学生UID，则保留原始标识（电话号码等）
                            callee = callee_raw
                        duration = float(row[dur_col]) if (dur_col and pd.notna(row.get(dur_col))) else 1.0
                        
                        if caller == callee:
                            continue
                            
                        if G.has_edge(caller, callee):
                            if weight_type == 'duration':
                                G[caller][callee]['weight'] += duration
                            else:
                                G[caller][callee]['weight'] += 1
                        else:
                            if weight_type == 'duration':
                                G.add_edge(caller, callee, weight=duration)
                            else:
                                G.add_edge(caller, callee, weight=1)
                                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        return graphs

class SMSNetwork:
    def __init__(self):
        from ..utils import NetworkBuilder
        self.folder = "sms"
        self.builder = NetworkBuilder()
        folder_path = self.builder.resolve_folder(self.folder)
        self.user_files = glob.glob(os.path.join(folder_path, "sms_*.csv"))
    
    def build_network(self, weight_type='count', time_window='D', phone_uid_map=None, phone_uid_map_path=None):
        """
        构建短信网络
        weight_type: 'count'
        time_window: 'D'(daily), 'W'(weekly), 'M'(monthly)
        """
        graphs = {}
        # 加载号码→UID映射（若提供路径或可自动发现）
        try:
            from ..utils import NetworkBuilder
            if phone_uid_map is None:
                phone_uid_map = NetworkBuilder.load_phone_uid_map(phone_uid_map_path)
        except Exception:
            phone_uid_map = {}
        
        for file_path in self.user_files:
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
                
                # 检查必要的列是否存在
                df.columns = [str(c).strip().lower() for c in df.columns]
                uid = self.builder.user_from_fname(file_path)
                # 统一时间列名
                time_col = next((c for c in df.columns if any(x in c for x in ['timestamp','datetime','time','date','start'])), None)
                # 对端列
                peer_col = next((c for c in df.columns if any(x in c for x in ['receiver','other','target','contact','peer','to','number','id'])), None)
                # 主体列
                sender_col = next((c for c in df.columns if any(x in c for x in ['sender','from','user'])), None)
                if time_col is None:
                    print(f"警告: 文件 {file_path} 中未找到时间列，跳过该文件")
                    continue
                if peer_col is None and ('receiver' not in df.columns):
                    print(f"警告: 文件 {file_path} 中未找到接收者列，跳过该文件")
                    continue
                
                from ..utils import NetworkBuilder
                grouped = NetworkBuilder.split_or_all(df, time_col, time_window)
                for time_key, group in grouped.items():
                    if time_key not in graphs:
                        graphs[time_key] = nx.Graph()
                    
                    G = graphs[time_key]
                    
                    # 计算边权重
                    for _, row in group.iterrows():
                        # 固定发信人为文件名中的UID，避免电话号码干扰
                        sender = uid
                        recv_val = row[peer_col] if peer_col in group.columns else row.get('receiver', None)
                        if pd.isna(recv_val):
                            continue
                        receiver_raw = str(recv_val)
                        # 映射接收者为UID（若可能）
                        if receiver_raw.startswith('u'):
                            receiver = receiver_raw
                        else:
                            norm = NetworkBuilder.normalize_phone(receiver_raw)
                            receiver = phone_uid_map.get(norm, '')
                        if not receiver:
                            # 若无法映射为学生UID，则保留原始标识（电话号码等）
                            receiver = receiver_raw
                        
                        if sender == receiver:
                            continue
                            
                        if G.has_edge(sender, receiver):
                            G[sender][receiver]['weight'] += 1
                        else:
                            G.add_edge(sender, receiver, weight=1)
                            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        return graphs