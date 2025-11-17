from typing import Dict, Set, Optional, Tuple, List
import os
import glob
import networkx as nx
from collections import defaultdict
import json
from ..utils import NetworkBuilder

class EducationNetwork:
    """教育网络构建类"""
    
    def __init__(self):
        self.folder = "education"
        self.builder = NetworkBuilder()
        
    def build_network(self,
                     network_type: str = 'co-enrollment',  # 'co-enrollment' 或 'bipartite'
                     time_window: Optional[str] = None
                     ) -> Dict[str, nx.Graph]:
        """构建教育网络（基于 class.csv 的同课共现网络，支持二分图投影）。

        - 优先解析 `education/class.csv` 内容的第一列作为 `uid`，其余列作为课程代码。
        - 不再依赖文件名中的 `uXX` 片段来识别学生。
        - `time_window` 对 `class.csv` 无效，统一返回静态结果 `{'all': G}`。
        """
        folder_path = self.builder.resolve_folder(self.folder)

        # 尝试找到 class.csv
        class_csv = None
        for f in glob.glob(os.path.join(folder_path, "**", "class.csv"), recursive=True):
            class_csv = f
            break

        if class_csv is None:
            # 未找到 class.csv，返回空图字典
            empty_users: Set[str] = set()
            return {'all': self.builder.create_base_graph(empty_users)}

        # 可选：加载课程元信息用于校验
        valid_courses: Optional[Set[str]] = None
        info_json = None
        for jf in glob.glob(os.path.join(folder_path, "**", "class_info.json"), recursive=True):
            info_json = jf
            break
        if info_json and os.path.isfile(info_json):
            try:
                with open(info_json, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                valid_courses = set(map(str, data.keys()))
            except Exception:
                valid_courses = None

        # 解析 class.csv：第一列 uid，后续为课程代码
        # 使用原生 csv 解析，避免对 pandas 的依赖
        rows: List[List[str]] = []
        try:
            import csv
            with open(class_csv, 'r', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                for r in reader:
                    # 保留原样，逐列去空白
                    rows.append([c.strip() for c in r])
        except Exception:
            # 兜底尝试不同编码
            try:
                import csv
                with open(class_csv, 'r', encoding='latin-1') as fh:
                    reader = csv.reader(fh)
                    for r in reader:
                        rows.append([c.strip() for c in r])
            except Exception:
                return {'all': self.builder.create_base_graph(set())}

        users: Set[str] = set()
        course_to_students: Dict[str, Set[str]] = defaultdict(set)
        student_to_courses: Dict[str, Set[str]] = defaultdict(set)

        for row in rows:
            if not row:
                continue
            # 第一列为 uid
            raw_uid = str(row[0]).strip()
            if not raw_uid or not raw_uid.startswith('u'):
                # 跳过非 uid 行（如标题、空行）
                continue
            uid = raw_uid
            users.add(uid)

            # 后续列为课程代码
            for i in range(1, len(row)):
                course_id = str(row[i]).strip()
                if not course_id or course_id.lower() == 'nan' or course_id == '...':
                    continue
                # 可选课程校验
                if valid_courses is not None and (course_id not in valid_courses):
                    # 若课程名不在 class_info.json 中，仍然接纳，但可在此处进行清洗/修正规则
                    pass
                course_to_students[course_id].add(uid)
                student_to_courses[uid].add(course_id)

        # 创建图（静态：all）
        G = self.builder.create_base_graph(users)

        if network_type == 'co-enrollment':
            # 对每一门课，枚举修读该课的学生对，累加共享课程计数
            for course_id, students in course_to_students.items():
                # 跳过空/仅单人课
                if len(students) < 2:
                    continue
                # 枚举两两组合
                sorted_students = sorted(students)
                for i in range(len(sorted_students)):
                    for j in range(i + 1, len(sorted_students)):
                        s1 = sorted_students[i]
                        s2 = sorted_students[j]
                        if G.has_edge(s1, s2):
                            G[s1][s2]['weight'] += 1.0
                        else:
                            G.add_edge(s1, s2, weight=1.0)
        else:
            # 学生-课程二分图
            G.clear()
            G.add_nodes_from(users, bipartite=0)
            G.add_nodes_from(list(course_to_students.keys()), bipartite=1)
            for student, courses in student_to_courses.items():
                for c in courses:
                    G.add_edge(student, c, weight=1.0)

        return {'all': G}
    
    def _collect_class_data(self, 
                           courses: Dict[str, Set[str]],
                           student_courses: Dict[str, Set[str]],
                           users: Set[str],
                           df: object):
        """专门处理class.csv数据（整表解析，不依赖文件名中的 uid）。"""
        # 保留兼容函数但不再被调用（主逻辑已改为原生 csv）
        try:
            iterable = getattr(df, 'iterrows', None)
            if not callable(iterable):
                return
            for _, row in df.iterrows():
                if len(row) == 0:
                    continue
                raw_uid = str(row.iloc[0]).strip()
                if not raw_uid or not raw_uid.startswith('u'):
                    continue
                uid = raw_uid
                users.add(uid)
                for i in range(1, len(row)):
                    course_id = str(row.iloc[i]).strip()
                    if course_id and course_id.lower() != 'nan' and course_id != '...':
                        courses[course_id].add(uid)
                        student_courses[uid].add(course_id)
        except Exception:
            # 兼容性方法，解析失败时忽略
            pass

    def _collect_course_data(self,
                            student: str,  # 保留签名以兼容调用，但方法现已不使用
                            courses: Dict[str, Set[str]],
                            student_courses: Dict[str, Set[str]],
                            course_times: Dict[str, Tuple[Optional[object], Optional[object]]],
                            df: object,
                            course_col: str,
                            start_col: Optional[str],
                            end_col: Optional[str]):
        """旧版方法占位，避免外部调用报错。当前教育层不再从非 class.csv 文件收集课程。"""
        # 不进行处理，保留兼容性。
        return

class NodeAttributes:
    """节点属性收集类"""
    
    def __init__(self):
        self.ema_folder = "EMA"
        self.survey_folder = "survey"
        self.builder = NetworkBuilder()
    
    def collect_attributes(self) -> Dict[str, Dict[str, Dict]]:
        """收集节点属性
        
        Returns:
            Dict[str, Dict[str, Dict]]: user -> {'ema': {...}, 'survey': {...}}
        """
        attributes = {}
        
        # 收集EMA/response数据（兼容多目录名与递归子目录）
        ema_dirs = [self.ema_folder, 'ema', 'EMA_response', 'responses', 'response']
        for d in ema_dirs:
            ema_folder = self.builder.resolve_folder(d)
            if not os.path.isdir(ema_folder):
                continue
            for f in glob.glob(os.path.join(ema_folder, "**", "*.csv"), recursive=True):
                uid = self.builder.user_from_fname(f)
                if uid not in attributes:
                    attributes[uid] = {'ema': {}, 'survey': {}}
                try:
                    import pandas as pd
                    df = pd.read_csv(f, on_bad_lines='skip', engine='python')
                except Exception:
                    try:
                        import pandas as pd
                        df = pd.read_csv(f, sep='\t', engine='python')
                    except Exception:
                        continue
                self._process_ema_data(attributes[uid]['ema'], df)
        
        # 收集Survey数据（递归）
        survey_dirs = [self.survey_folder, 'surveys', 'questionnaire']
        for d in survey_dirs:
            survey_folder = self.builder.resolve_folder(d)
            if not os.path.isdir(survey_folder):
                continue
            for f in glob.glob(os.path.join(survey_folder, "**", "*.csv"), recursive=True):
                uid = self.builder.user_from_fname(f)
                if uid not in attributes:
                    attributes[uid] = {'ema': {}, 'survey': {}}
                try:
                    import pandas as pd
                    df = pd.read_csv(f, on_bad_lines='skip', engine='python')
                except Exception:
                    continue
                self._process_survey_data(attributes[uid]['survey'], df)
        
        return attributes
    
    def _process_ema_data(self, ema_dict: Dict, df: object):
        """处理EMA数据
        
        这里需要根据实际的EMA数据结构进行调整。以下是示例处理：
        - 计算情绪和压力的平均值
        - 统计回复率
        """
        import pandas as pd
        df.columns = [str(c).lower() for c in df.columns]
        
        # 将可数值列转换为数值用于均值计算
        def to_numeric_cols(cols):
            out = []
            for c in cols:
                try:
                    s = pd.to_numeric(df[c], errors='coerce')
                    if s.notna().any():
                        out.append(c)
                except Exception:
                    continue
            return out
        
        # 情绪
        mood_cols = to_numeric_cols([c for c in df.columns if 'mood' in c or 'emotion' in c])
        if mood_cols:
            ema_dict['avg_mood'] = df[mood_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
        
        # 压力
        stress_cols = to_numeric_cols([c for c in df.columns if 'stress' in c])
        if stress_cols:
            ema_dict['avg_stress'] = df[stress_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
        
        # 回复率：优先使用状态列，其次使用回答非空比例
        status_cols = [c for c in df.columns if any(x in c for x in ['completed','response','answered','status','submit'])]
        response_rate = None
        for c in status_cols:
            s = df[c].astype(str).str.lower()
            # 将常见真值映射为1
            truthy = s.isin(['1','true','yes','y','submitted','completed'])
            falsy = s.isin(['0','false','no','n'])
            if truthy.any() or falsy.any():
                response_rate = truthy.mean()
                break
        if response_rate is None:
            # 估计：每行是否有任何非时间/ID字段的非空回答
            meta_cols = [c for c in df.columns if any(x in c for x in ['time','timestamp','date','id','user','uid','survey','type'])]
            answer_cols = [c for c in df.columns if c not in meta_cols]
            if answer_cols:
                response_rate = df[answer_cols].notna().any(axis=1).mean()
            else:
                response_rate = 0.0
        ema_dict['response_rate'] = float(response_rate)
    
    def _process_survey_data(self, survey_dict: Dict, df: object):
        """处理Survey数据
        
        这里需要根据实际的Survey数据结构进行调整。以下是示例处理：
        - 提取人口统计学特征
        - 提取社交能力自评分数
        - 提取学习参与度指标
        """
        # 修改：确保列名都是字符串后再进行处理
        import pandas as pd
        df.columns = [str(c).lower() for c in df.columns]
        
        # 示例：处理人口统计学特征
        demo_cols = ['age', 'gender', 'year', 'major']
        for col in demo_cols:
            matches = [c for c in df.columns if col in c]
            if matches:
                survey_dict[col] = df[matches[0]].iloc[0]
        
        # 示例：处理社交能力自评
        social_cols = [c for c in df.columns 
                      if any(x in c for x in ['social', 'communication', 'interaction'])]
        if social_cols:
            # 修改：只对可以转换为数值的列进行计算
            numeric_social_cols = []
            for col in social_cols:
                # 检查列是否可以转换为数值
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_social_cols.append(col)
                except (ValueError, TypeError):
                    continue
            if numeric_social_cols:
                survey_dict['social_score'] = df[numeric_social_cols].mean().mean()
        
        # 示例：处理学习参与度
        engage_cols = [c for c in df.columns 
                      if any(x in c for x in ['engage', 'participate', 'attend'])]
        if engage_cols:
            # 修改：只对可以转换为数值的列进行计算
            numeric_engage_cols = []
            for col in engage_cols:
                # 检查列是否可以转换为数值
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_engage_cols.append(col)
                except (ValueError, TypeError):
                    continue
            if numeric_engage_cols:
                survey_dict['engagement_score'] = df[numeric_engage_cols].mean().mean()


    class EducationEnhancedNetwork:
        """增强版教育网络构建：在共修课程基础上，融合多源教育相关证据。

        证据来源（均为可选）：
        - 共修课程（co-enrollment）：来自 education/class.csv（与 EducationNetwork 相同）
        - 学术类日历共现（calendar academic co-attendance）：对日历事件按关键词过滤，参与者两两连边
        - 学习类 App 使用相似度（edu app similarity）：筛选学习/学术相关 App 后计算用户间余弦相似度
        - 相同专业/年级（same major/year）：来自 survey 的属性同类连边（弱证据）

        合并方式：先对各来源子图可选归一化，再按权重线性相加或二值化。
        返回静态图 {'all': G}。
        """

        def __init__(self):
            self.builder = NetworkBuilder()

        def build_network(
            self,
            # 各来源权重（线性相加时有效）
            w_co_enroll: float = 1.0,
            w_calendar: float = 0.6,
            w_edu_app: float = 0.6,
            w_same_major: float = 0.3,
            # 归一化/合并策略
            normalize_sources: bool = True,
            final_normalize: bool = True,
            combine_mode: str = 'sum',  # 'sum' or 'binary'
            # 学术日历过滤（关键词任一命中即视为学术事件）
            calendar_keywords: Optional[List[str]] = None,
            # 学习类 App 过滤（关键词/正则片段任一命中）
            edu_app_keywords: Optional[List[str]] = None,
            app_similarity_threshold: float = 0.25,
            app_top_k: Optional[int] = 10,
        ) -> Dict[str, nx.Graph]:
            import os
            import glob
            import pandas as pd
            import numpy as np
            import networkx as nx
            from collections import defaultdict

            # 默认关键词（可按实际数据调整/扩展）
            if calendar_keywords is None:
                calendar_keywords = [
                    'class', 'course', 'lecture', 'lab', 'seminar', 'tutorial', 'exam', 'quiz', 'assignment',
                    'meeting', 'advisor', 'office hour', 'office-hour', 'thesis', 'defense'
                ]
            if edu_app_keywords is None:
                edu_app_keywords = [
                    'moodle', 'canvas', 'blackboard', 'edx', 'coursera', 'khan', 'udemy',
                    'slides', 'doc', 'pdf', 'word', 'excel', 'powerpoint', 'onenote', 'notion', 'evernote',
                    'github', 'gitlab', 'coding', 'ide', 'leetcode'
                ]

            # 统一用户集合（education/calendar/app_usage/survey）
            users = self.builder.get_all_users_union(['education', 'calendar', 'app_usage', 'survey'])
            G_final = self.builder.create_base_graph(users)

            # ---------------- 1) 共修课程 ----------------
            try:
                edu_base = EducationNetwork()
                edu_res = edu_base.build_network(network_type='co-enrollment', time_window=None)
                G_enroll = edu_res.get('all', next(iter(edu_res.values()))) if isinstance(edu_res, dict) else edu_res
            except Exception:
                G_enroll = nx.Graph(); G_enroll.add_nodes_from(users)

            # ---------------- 2) 学术类日历共现 ----------------
            G_cal = self.builder.create_base_graph(users)
            try:
                cal_path = self.builder.resolve_folder('calendar')
                def _is_academic(name: str) -> bool:
                    if not name:
                        return False
                    s = str(name).lower()
                    return any(k in s for k in calendar_keywords)

                events = defaultdict(set)  # event_id -> participants
                # 兼容常见列：['event','id','title','summary','description'] 参与者列 ['attendee','participant','participants']
                for f in glob.glob(os.path.join(cal_path, "**", "*.csv"), recursive=True):
                    try:
                        df = pd.read_csv(f, on_bad_lines='skip', engine='python')
                    except Exception:
                        continue
                    if df is None or df.shape[0] == 0:
                        continue
                    df.columns = [str(c).lower() for c in df.columns]
                    event_id_col = next((c for c in df.columns if c in ['event','id','event_id']), None)
                    title_col = next((c for c in df.columns if c in ['title','summary','name','description','subject']), None)
                    attendees_col = next((c for c in df.columns if any(x in c for x in ['attendee','participant','participants'])), None)

                    for _, row in df.iterrows():
                        # 事件标识与名称
                        ev_id = str(row[event_id_col]) if event_id_col and pd.notna(row.get(event_id_col)) else None
                        ev_title = str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else ''
                        if not _is_academic(ev_title):
                            continue

                        # 参与者：若无显式参与者则回退为文件所属 uid
                        uid = self.builder.user_from_fname(f)
                        part_set: Set[str] = set()
                        if attendees_col and pd.notna(row.get(attendees_col)):
                            parts = [p.strip() for p in str(row.get(attendees_col)).split(';') if str(p).strip()]
                            for p in parts:
                                part_set.add(p)
                        if uid:
                            part_set.add(uid)
                        if not part_set:
                            continue
                        key = ev_id or f"{os.path.basename(f)}#{_}"
                        events[key].update(p for p in part_set if p in users)

                # 事件参与者两两连边
                for ev, members in events.items():
                    if len(members) < 2:
                        continue
                    ms = sorted(members)
                    for i in range(len(ms)):
                        for j in range(i+1, len(ms)):
                            u, v = ms[i], ms[j]
                            if G_cal.has_edge(u, v):
                                G_cal[u][v]['weight'] += 1.0
                            else:
                                G_cal.add_edge(u, v, weight=1.0)
            except Exception:
                pass

            # ---------------- 3) 学习类 App 使用相似度 ----------------
            G_app = self.builder.create_base_graph(users)
            try:
                app_path = self.builder.resolve_folder('app_usage')
                # user -> Counter(app->count)
                from collections import Counter
                usage = defaultdict(Counter)

                def _is_edu_app(name: str) -> bool:
                    if not name:
                        return False
                    s = str(name).lower()
                    return any(k in s for k in edu_app_keywords)

                for f in glob.glob(os.path.join(app_path, "*.csv")):
                    uid = self.builder.user_from_fname(f)
                    if not uid:
                        continue
                    try:
                        df = pd.read_csv(f, on_bad_lines='skip', engine='python')
                    except Exception:
                        continue
                    if df is None or df.shape[0] == 0:
                        continue
                    df.columns = [str(c).lower() for c in df.columns]
                    app_col = next((c for c in df.columns if any(x in c for x in ['app','package','application','bundle'])), None)
                    if not app_col:
                        continue
                    for _, row in df.iterrows():
                        app = row.get(app_col)
                        if pd.isna(app):
                            continue
                        if _is_edu_app(str(app)):
                            usage[uid][str(app)] += 1

                # 构造用户-应用矩阵（仅教育类app）
                all_apps: List[str] = sorted({a for c in usage.values() for a in c.keys()})
                if len(all_apps) >= 1:
                    app_idx = {a: i for i, a in enumerate(all_apps)}
                    us = [u for u in users if u in usage]
                    if len(us) >= 2:
                        mat = np.zeros((len(us), len(all_apps)), dtype=float)
                        for i, u in enumerate(us):
                            for a, cnt in usage[u].items():
                                j = app_idx.get(a)
                                if j is not None:
                                    mat[i, j] = float(cnt)
                        # 余弦相似度
                        def _cosine(M):
                            M2 = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                            return np.dot(M2, M2.T)
                        S = _cosine(mat)
                        for i in range(len(us)):
                            pairs = [(j, float(S[i, j])) for j in range(i+1, len(us)) if S[i, j] >= app_similarity_threshold]
                            if app_top_k:
                                pairs.sort(key=lambda x: x[1], reverse=True)
                                pairs = pairs[:app_top_k]
                            for j, w in pairs:
                                G_app.add_edge(us[i], us[j], weight=w)
            except Exception:
                pass

            # ---------------- 4) 相同专业/年级（Survey） ----------------
            G_attr = self.builder.create_base_graph(users)
            try:
                attrs = NodeAttributes().collect_attributes()
                # 相同专业 +0.5，相同期别/年级 +0.5（累计后归一化）
                for u in users:
                    for v in users:
                        if u >= v:
                            continue
                        wu = 0.0
                        au = attrs.get(u, {})
                        av = attrs.get(v, {})
                        su = (au.get('survey') or {})
                        sv = (av.get('survey') or {})
                        try:
                            if su.get('major') and sv.get('major') and str(su.get('major')) == str(sv.get('major')):
                                wu += 1.0
                        except Exception:
                            pass
                        try:
                            if su.get('year') and sv.get('year') and str(su.get('year')) == str(sv.get('year')):
                                wu += 1.0
                        except Exception:
                            pass
                        if wu > 0:
                            G_attr.add_edge(u, v, weight=wu)
            except Exception:
                pass

            # ---------------- 合并各来源 ----------------
            def _prep(H: nx.Graph, w: float) -> nx.Graph:
                Hn = H.copy()
                if normalize_sources:
                    Hn = self.builder.normalize_weights(Hn)
                if w != 1.0:
                    for u, v, d in Hn.edges(data=True):
                        d['weight'] = float(d.get('weight', 1.0)) * float(w)
                return Hn

            sources = [
                _prep(G_enroll, w_co_enroll),
                _prep(G_cal, w_calendar),
                _prep(G_app, w_edu_app),
                _prep(G_attr, w_same_major),
            ]

            for H in sources:
                for u, v, d in H.edges(data=True):
                    if combine_mode == 'binary':
                        if not G_final.has_edge(u, v):
                            G_final.add_edge(u, v, weight=1.0)
                    else:
                        w = float(d.get('weight', 1.0))
                        if G_final.has_edge(u, v):
                            G_final[u][v]['weight'] += w
                        else:
                            G_final.add_edge(u, v, weight=w)

            if final_normalize:
                G_final = self.builder.normalize_weights(G_final)

            return {'all': G_final}