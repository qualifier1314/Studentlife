import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from ..utils import NetworkBuilder


class CommunicationFeatureExtractor:
    """将通话与短信日志转为每个学生的通信行为特征（节点属性）。

    特征示例：
    - calls_total: 通话总次数
    - call_duration_total: 通话总时长
    - outgoing_ratio_calls: 去电比例（去电次数 / (去电+来电)）
    - sms_total: 短信总条数
    - outgoing_ratio_sms: 发短信比例
    - comm_freq_day: 平均每日通信事件数（通话+短信）
    - comm_freq_week: 平均每周通信事件数（通话+短信）
    - unique_contacts: 去重联系人数量（按映射后的UID或号码的数字化形式）
    - contact_entropy: 联系人分布的香农熵（越高越分散）
    """

    def __init__(self):
        self.builder = NetworkBuilder()

    def _find_time_col(self, cols):
        cols = [str(c).strip().lower() for c in cols]
        return next((c for c in cols if any(x in c for x in ['timestamp', 'datetime', 'time', 'date', 'start'])), None)

    def _find_duration_col(self, cols):
        cols = [str(c).strip().lower() for c in cols]
        return next((c for c in cols if any(x in c for x in ['duration', 'length', 'dur', 'seconds', 'sec'])), None)

    def _find_peer_col(self, cols, peer_candidates):
        cols = [str(c).strip().lower() for c in cols]
        return next((c for c in cols if any(x in c for x in peer_candidates)), None)

    def _find_dir_col(self, cols):
        cols = [str(c).strip().lower() for c in cols]
        return next((c for c in cols if any(x in c for x in ['direction', 'type', 'calltype'])), None)

    def _find_actor_col(self, cols, actor_candidates):
        cols = [str(c).strip().lower() for c in cols]
        return next((c for c in cols if any(x in c for x in actor_candidates)), None)

    def _normalize_contact(self, val, phone_uid_map):
        if pd.isna(val):
            return ''
        s = str(val)
        if s.startswith('u'):
            return s
        norm = NetworkBuilder.normalize_phone(s)
        if norm in phone_uid_map:
            return phone_uid_map[norm]
        # 返回数字化号码（避免暴露原号码格式）
        return norm

    def build_node_features(self, phone_uid_map_path: str = None) -> pd.DataFrame:
        users = self.builder.get_all_users('call_log') | self.builder.get_all_users('sms')
        phone_uid_map = NetworkBuilder.load_phone_uid_map(phone_uid_map_path)

        # 初始化聚合容器
        stats = {
            uid: {
                'uid': uid,
                'calls_total': 0,
                'call_duration_total': 0.0,
                'outgoing_calls': 0,
                'incoming_calls': 0,
                'sms_total': 0,
                'outgoing_sms': 0,
                'incoming_sms': 0,
                'days_events': Counter(),
                'weeks_events': Counter(),
                'contact_counts': Counter(),
            }
            for uid in users
        }

        # 处理通话日志
        call_path = self.builder.resolve_folder('call_log')
        for f in glob.glob(os.path.join(call_path, "call_log_*.csv")):
            uid = self.builder.user_from_fname(f)
            if not uid:
                continue
            try:
                df = pd.read_csv(f, on_bad_lines='skip', engine='python')
            except Exception:
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            time_col = self._find_time_col(df.columns)
            dur_col = self._find_duration_col(df.columns)
            peer_col = self._find_peer_col(df.columns, ['callee', 'other', 'target', 'contact', 'peer', 'to', 'number', 'id'])
            dir_col = self._find_dir_col(df.columns)
            caller_col = self._find_actor_col(df.columns, ['caller', 'source', 'from', 'user'])

            for _, row in df.iterrows():
                # 通话事件计数
                stats[uid]['calls_total'] += 1
                # 时长
                if dur_col and pd.notna(row.get(dur_col)):
                    try:
                        stats[uid]['call_duration_total'] += float(row.get(dur_col))
                    except Exception:
                        pass

                # 方向：优先用 caller==uid 判定，否则用 direction/type
                outgoing = None
                if caller_col and pd.notna(row.get(caller_col)):
                    outgoing = (str(row.get(caller_col)) == uid)
                elif dir_col and pd.notna(row.get(dir_col)):
                    dval = str(row.get(dir_col)).lower()
                    outgoing = ('out' in dval or 'send' in dval)
                if outgoing is True:
                    stats[uid]['outgoing_calls'] += 1
                elif outgoing is False:
                    stats[uid]['incoming_calls'] += 1

                # 联系人计数与频次
                contact = ''
                if peer_col:
                    contact = self._normalize_contact(row.get(peer_col), phone_uid_map)
                elif 'callee' in df.columns:
                    contact = self._normalize_contact(row.get('callee'), phone_uid_map)
                if contact:
                    stats[uid]['contact_counts'][contact] += 1

                # 频率（按日/周）
                if time_col and pd.notna(row.get(time_col)):
                    try:
                        t = pd.to_datetime(row.get(time_col), errors='coerce')
                        if pd.notna(t):
                            stats[uid]['days_events'][t.date()] += 1
                            stats[uid]['weeks_events'][(t.year, t.week if hasattr(t, 'week') else t.weekofyear)] += 1
                    except Exception:
                        pass

        # 处理短信日志
        sms_path = self.builder.resolve_folder('sms')
        for f in glob.glob(os.path.join(sms_path, "sms_*.csv")):
            uid = self.builder.user_from_fname(f)
            if not uid:
                continue
            try:
                df = pd.read_csv(f, on_bad_lines='skip', engine='python')
            except Exception:
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            time_col = self._find_time_col(df.columns)
            peer_col = self._find_peer_col(df.columns, ['receiver', 'other', 'target', 'contact', 'peer', 'to', 'number', 'id'])
            dir_col = self._find_dir_col(df.columns)
            sender_col = self._find_actor_col(df.columns, ['sender', 'from', 'user'])

            for _, row in df.iterrows():
                stats[uid]['sms_total'] += 1

                outgoing = None
                if sender_col and pd.notna(row.get(sender_col)):
                    outgoing = (str(row.get(sender_col)) == uid)
                elif dir_col and pd.notna(row.get(dir_col)):
                    dval = str(row.get(dir_col)).lower()
                    outgoing = ('out' in dval or 'send' in dval)
                if outgoing is True:
                    stats[uid]['outgoing_sms'] += 1
                elif outgoing is False:
                    stats[uid]['incoming_sms'] += 1

                contact = ''
                if peer_col:
                    contact = self._normalize_contact(row.get(peer_col), phone_uid_map)
                elif 'receiver' in df.columns:
                    contact = self._normalize_contact(row.get('receiver'), phone_uid_map)
                if contact:
                    stats[uid]['contact_counts'][contact] += 1

                if time_col and pd.notna(row.get(time_col)):
                    try:
                        t = pd.to_datetime(row.get(time_col), errors='coerce')
                        if pd.notna(t):
                            stats[uid]['days_events'][t.date()] += 1
                            stats[uid]['weeks_events'][(t.year, t.week if hasattr(t, 'week') else t.weekofyear)] += 1
                    except Exception:
                        pass

        # 汇总为 DataFrame
        rows = []
        for uid in sorted(users):
            s = stats[uid]
            total_dir_calls = s['outgoing_calls'] + s['incoming_calls']
            total_dir_sms = s['outgoing_sms'] + s['incoming_sms']
            outgoing_ratio_calls = (s['outgoing_calls'] / total_dir_calls) if total_dir_calls > 0 else np.nan
            outgoing_ratio_sms = (s['outgoing_sms'] / total_dir_sms) if total_dir_sms > 0 else np.nan

            day_freq = (sum(s['days_events'].values()) / len(s['days_events'])) if len(s['days_events']) > 0 else 0.0
            week_freq = (sum(s['weeks_events'].values()) / len(s['weeks_events'])) if len(s['weeks_events']) > 0 else 0.0

            # 联系人唯一数与熵
            unique_contacts = len(s['contact_counts'])
            total_contacts_events = sum(s['contact_counts'].values())
            if total_contacts_events > 0 and unique_contacts > 0:
                p = np.array(list(s['contact_counts'].values()), dtype=float) / float(total_contacts_events)
                contact_entropy = float(-np.sum(p * np.log(p + 1e-12)))
            else:
                contact_entropy = 0.0

            rows.append({
                'uid': uid,
                'calls_total': s['calls_total'],
                'call_duration_total': s['call_duration_total'],
                'outgoing_ratio_calls': outgoing_ratio_calls,
                'sms_total': s['sms_total'],
                'outgoing_ratio_sms': outgoing_ratio_sms,
                'comm_freq_day': day_freq,
                'comm_freq_week': week_freq,
                'unique_contacts': unique_contacts,
                'contact_entropy': contact_entropy,
            })

        return pd.DataFrame(rows)