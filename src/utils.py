import os
import glob
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime, timedelta

class NetworkBuilder:
    DATA_ROOT = r"e:\Complex network\Complex Network\student_complex\studentlife_data\dataset"
    TOP_ROOT = os.path.dirname(DATA_ROOT)
    
    @staticmethod
    def user_from_fname(fname: str) -> str:
        """从文件名中提取用户ID"""
        base = os.path.basename(fname)
        parts = base.split('_')
        for p in parts:
            if p.startswith('u'):
                return p.split('.')[0]
        # 若文件名不包含以 'u' 开头的片段，则视为无有效用户ID
        return ''
    
    @staticmethod
    def get_all_users(folder: str) -> Set[str]:
        """获取指定文件夹及其子目录下所有用户ID（支持 csv/txt/db）"""
        users: Set[str] = set()
        folder_path = NetworkBuilder.resolve_folder(folder)
        if not os.path.isdir(folder_path):
            return users
        patterns = ["**/*.csv", "**/*.txt", "**/*.db"]
        for pat in patterns:
            for f in glob.glob(os.path.join(folder_path, pat), recursive=True):
                uid = NetworkBuilder.user_from_fname(f)
                if uid:
                    users.add(uid)
        return users
    
    @staticmethod
    def create_base_graph(users: Set[str]) -> nx.Graph:
        """创建包含所有用户节点的基础图"""
        G = nx.Graph()
        G.add_nodes_from(users)
        return G
    
    @staticmethod
    def time_window_split(df, 
                         time_col: str,
                         window: str = 'D') -> Dict[str, object]:
        """按时间窗口分割数据
        
        Args:
            df: 输入数据框
            time_col: 时间列名
            window: 时间窗口，'D'为按天，'W'为按周
        
        Returns:
            Dict[str, pd.DataFrame]: 按时间窗口分组的数据字典
        """
        import pandas as pd
        if time_col not in df.columns:
            return {'all': df}
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isna().all():
            return {'all': df}
        # 若仅有日期（没有时分秒），默认视为静态层
        if df[time_col].dt.normalize().equals(df[time_col]):
            return {'all': df}
        return dict(tuple(df.groupby(pd.Grouper(key=time_col, freq=window))))
    
    @staticmethod
    def normalize_weights(G: nx.Graph, method: str = 'minmax') -> nx.Graph:
        """归一化边权重
        
        Args:
            G: 输入图
            method: 归一化方法，'minmax'或'zscore'
        
        Returns:
            nx.Graph: 归一化后的图
        """
        weights = [d['weight'] for _,_,d in G.edges(data=True)]
        if not weights:
            return G
            
        if method == 'minmax':
            min_w, max_w = min(weights), max(weights)
            if max_w > min_w:
                for u,v,d in G.edges(data=True):
                    G[u][v]['weight'] = (d['weight'] - min_w) / (max_w - min_w)
        elif method == 'zscore':
            mean_w, std_w = np.mean(weights), np.std(weights)
            if std_w > 0:
                for u,v,d in G.edges(data=True):
                    G[u][v]['weight'] = (d['weight'] - mean_w) / std_w
                    
        return G

    @staticmethod
    def resolve_folder(folder: str) -> str:
        """解析数据目录，优先 dataset/，若不存在则回退到顶层 studentlife_data/。"""
        candidate_1 = os.path.join(NetworkBuilder.DATA_ROOT, folder)
        candidate_2 = os.path.join(NetworkBuilder.TOP_ROOT, folder)
        if os.path.isdir(candidate_1):
            return candidate_1
        return candidate_2

    # -------------------- 号码映射工具 --------------------
    @staticmethod
    def normalize_phone(phone: str) -> str:
        """规范化电话号码，仅保留数字字符。

        说明：映射文件中的号码可能包含国家码、括号或连字符；
        统一为纯数字字符串以便匹配。
        """
        if phone is None:
            return ''
        s = str(phone)
        digits = ''.join(ch for ch in s if ch.isdigit())
        return digits

    @staticmethod
    def find_phone_uid_map_path() -> Optional[str]:
        """尝试在数据根目录下自动查找 phone_uid_map*.csv 文件路径。"""
        roots = [NetworkBuilder.DATA_ROOT, NetworkBuilder.TOP_ROOT]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for f in glob.glob(os.path.join(root, "**", "phone_uid_map*.csv"), recursive=True):
                return f
        return None

    @staticmethod
    def load_phone_uid_map(map_path: Optional[str] = None) -> Dict[str, str]:
        """加载电话号码到学生UID的映射。

        优先使用显式的 `map_path`；若未提供，则在数据目录自动搜索 `phone_uid_map*.csv`。
        映射中，号码与表内号码统一为纯数字以避免格式差异。

        支持的列名：
        - 号码列：['phone','number','contact','peer','to'] 中任意一个
        - UID列：['uid','user','id'] 中任意一个（必须以 'u' 开头的ID才会被接受）
        """
        mapping: Dict[str, str] = {}
        path = map_path or NetworkBuilder.find_phone_uid_map_path()
        if not path or not os.path.isfile(path):
            return mapping
        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
            df.columns = [str(c).strip().lower() for c in df.columns]
            phone_col = next((c for c in df.columns if c in ['phone','number','contact','peer','to']), None)
            uid_col = next((c for c in df.columns if c in ['uid','user','id']), None)
            if phone_col is None or uid_col is None:
                return mapping
            for _, row in df.iterrows():
                raw_phone = row.get(phone_col)
                uid = str(row.get(uid_col, '')).strip()
                if not isinstance(raw_phone, (str, int)):
                    continue
                norm = NetworkBuilder.normalize_phone(raw_phone)
                if not norm:
                    continue
                if isinstance(uid, str) and uid.startswith('u'):
                    mapping[norm] = uid
        except Exception:
            return {}
        return mapping

    @staticmethod
    def split_or_all(df, time_col: str, window: Optional[str]) -> Dict[str, object]:
        """依据时间窗口返回分片；若无时间或指定为静态则返回 {'all': df}。"""
        if window is None:
            return {'all': df}
        return NetworkBuilder.time_window_split(df, time_col, window)

    # -------------------- 导出与目录工具 --------------------
    @staticmethod
    def ensure_dir(dir_path: str):
        """确保目录存在。"""
        if dir_path and not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def export_edge_list(G: nx.Graph, filepath: str, weighted: bool = False):
        """导出图的边列表到 CSV。
        
        Args:
            G: 图
            filepath: 目标 CSV 文件路径
            weighted: 是否导出权重列
        """
        import pandas as pd
        rows = []
        for u, v, d in G.edges(data=True):
            if weighted:
                rows.append({
                    'source': u,
                    'target': v,
                    'weight': float(d.get('weight', 1.0))
                })
            else:
                rows.append({
                    'source': u,
                    'target': v,
                })
        # 确保空表也包含列头
        if weighted:
            df = pd.DataFrame(rows, columns=['source', 'target', 'weight'])
        else:
            df = pd.DataFrame(rows, columns=['source', 'target'])
        NetworkBuilder.ensure_dir(os.path.dirname(filepath))
        df.to_csv(filepath, index=False)

    # -------------------- 全局 UID 编号映射 --------------------
    @staticmethod
    def get_all_users_union(folders: List[str]) -> Set[str]:
        """获取多个数据文件夹的用户ID并取并集。"""
        users: Set[str] = set()
        for folder in folders:
            try:
                users |= NetworkBuilder.get_all_users(folder)
            except Exception:
                # 某些目录可能不存在，忽略
                continue
        return users

    @staticmethod
    def build_uid_numbering(users: Set[str]) -> Dict[str, int]:
        """为学生UID生成稳定的 1..N 编号映射。

        规则：
        - 优先按 UID 的数字部分排序（如 u1 < u2 < u10）；
        - 若 UID 不符合上述格式，则按字典序排序；
        - 编号从 1 开始，以便在图上更直观。
        """
        def sort_key(u: str):
            try:
                if isinstance(u, str) and u.startswith('u') and u[1:].isdigit():
                    return (0, int(u[1:]))
            except Exception:
                pass
            return (1, str(u))

        ordered = sorted(list(users), key=sort_key)
        return {uid: i + 1 for i, uid in enumerate(ordered)}

    @staticmethod
    def export_uid_numbering(mapping: Dict[str, int], filepath: str):
        """将 UID→编号 映射导出为 CSV。"""
        import pandas as pd
        rows = [{'uid': uid, 'number': num} for uid, num in mapping.items()]
        df = pd.DataFrame(rows, columns=['uid', 'number'])
        NetworkBuilder.ensure_dir(os.path.dirname(filepath))
        df.to_csv(filepath, index=False, encoding='utf-8')