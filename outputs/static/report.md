# 学生行为多层网络构建与可视化报告

生成时间：自动生成（基于仓库当前数据）

本报告汇总了各单层网络的可视化图形、对应数据来源及数据集简述，并在结尾评估当前多层网络的可行性以及在其基础上开展“创新关键节点识别”研究的可行性。

---

## 单层网络（静态综合层）

以下三个静态综合单层是目前用于多层网络构建的基础层：Physical、Behavior、Education。每一层的可视化图片已保存在 `outputs/static/figures/` 目录。

### Physical（物理接近+共同就餐）
- 可视化：`outputs/static/figures/Physical.png`
- 数据来源：
  - `studentlife_data/dataset/sensing/**/*.csv`
  - `studentlife_data/dataset/dining/**` 与 `studentlife_data/dataset/dinning/**`
- 数据集信息（简述）：
  - 物理接触数据基于近距离检测（蓝牙/WiFi/RSSI等），包含时间戳、被检测对象、强度/距离等字段；就餐数据包含就餐时间与地点，支持 `.csv` 或 `.txt`。
  - 近距离接触按阈值与最小持续时间累计边权；共同就餐按时间差阈值判断是否同时就餐并累计共现次数。
- 构建逻辑：对物理接近与共同就餐两个图进行权重归一化后合并，得到静态 Physical 层。

### Behavior（日历事件共现+应用使用相似度）
- 可视化：`outputs/static/figures/Behavior.png`
- 数据来源：
  - `studentlife_data/dataset/calendar/*.csv`
  - `studentlife_data/dataset/app_usage/*.csv`
- 数据集信息（简述）：
  - 日历数据包含事件ID、开始/结束时间、参与者列表等；应用使用记录包含应用包名/名称、时间戳等。
  - 日历层按事件参与者两两共现生成边；应用层按用户-应用计数矩阵计算余弦相似度，阈值与 `top_k` 控制邻接边集。
- 构建逻辑：将“日历共现图”和“应用相似度图”权重归一化后合并，得到静态 Behavior 层。

### Education（同课共修）
- 可视化：`outputs/static/figures/Education.png`
- 数据来源：
  - `studentlife_data/dataset/education/**/class.csv`
  - 可选：`class_info.json`（课程元信息校验）
- 数据集信息（简述）：
  - `class.csv` 第一列为学生 `uid`，其余列为该学生修读的课程代码；构建时对每门课的学生集合进行两两组合，边权为共享课程计数。
  - 也支持构建“学生-课程二分图”，但用于多层时通常采用共修投影。
- 构建逻辑：直接以同课共享次数作为边权，形成静态 Education 层。

---

## 原始单层网络（作为静态层的来源）
- 物理接触（SensingNetwork）：`dataset/sensing/**/*.csv`，按近距离接触累计时长构边。
- 共同就餐（DiningNetwork）：`dataset/dining/**` 与 `dataset/dinning/**`，按同时就餐共现次数构边。
- 应用使用（AppUsageNetwork）：`dataset/app_usage/*.csv`，按用户间使用相似度（余弦）构边。
- 日历事件（CalendarNetwork）：`dataset/calendar/*.csv`，按事件参与者共现构边。
- 教育（EducationNetwork）：`dataset/education/**/class.csv`，按同课共享次数构边。

注：因隐私设计，call_log/sms 的对端为单向哈希，无法合法映射为学生-学生边；因此 communication 不构建单层网络，改为节点属性层（见下）。

---

## Communication 节点属性层（不作为学生-学生边）
- 数据来源：`dataset/call_log/*.csv`、`dataset/sms/*.csv`（对端哈希，不公布学生映射）
- 特征示例：
  - `calls_total`、`call_duration_total`、`sms_total`、`unique_contacts`（按哈希去重）
  - `avg_daily_comm`、`avg_weekly_comm`、`outgoing_ratio`、`night_ratio`、`weekend_ratio`
  - 集中度与分布：`top1_contact_share`、`top5_contact_share`、`contact_entropy`、`contact_gini`
  - 动态性：`burstiness`（变异系数）、`trend_slope_weekly`（趋势）
- 产物：`outputs/static/communication_features.csv`

这些通信特征作为节点属性与多层结构特征（度/介数/参与系数等）联合用于关键节点识别与解释。

注：当前多层构建以静态综合层（Communication/Physical/Behavior/Education）为主；上述原始单层是这些综合层的来源。

---

## 多层网络构建与导出
- 建模设定（静态、存在即为边）：对含时间的数据集，将全期内任意一次交互视为存在边；静态网络保留“出现过即连边”，并同时导出加权版本（强度为次数/时长等累计）。
- 构建方式：以 Physical、Behavior、Education 三层为例，统一节点顺序后分别生成三层邻接矩阵 `A_phy`、`A_beh`、`A_edu`，用 `omega * I` 表示层间同一学生的耦合，拼接成 `3N × 3N` 的 `supra` 邻接矩阵。
- 导出产物：
  - `outputs/static/supra_adjacency.csv`
  - `outputs/static/multi_layer_edges.csv` 或示例内生成的 `outputs/static/multiplex_edges_weighted.csv`
- 可视化：3D 多层示意图（`Multiplex3D.png`，若已生成）。

---

## 可行性评估
- 多层网络可行性：
  - 从目录结构与现有脚本看，数据加载、静态层合并、统一编号映射、`supra` 拼接与边导出流程完整，且 `outputs/static/` 已存在对应产物，说明构建链路是可行的。
  - 可进一步在时间维度上扩展（按周/月），但需保证各层的时间切片对齐与节点集统一。
- 创新关键节点识别的可行性：
  - 在多层网络上开展关键节点识别完全可行。可选方法包括：
    - 多层/超网络版 PageRank、层感知度中心性（如跨层参与系数、层间耦合权衡的度/介数）。
    - 多层 k-core 分解、跨层社团参与度、跨层桥接性（识别跨层连通的“关键桥”节点）。
    - 结合节点属性（EMA/Survey）与行为层（App/Calendar/AccFeat）的特征，进行监督/弱监督排序与解释。
  - 注意事项：
    - 层间权重 `omega` 和各层归一化会显著影响排序结果；建议通过敏感性分析选择稳健参数。
    - 节点集统一和数据质量（缺失/噪声）会影响跨层指标一致性，需要做缺失补全或稳健度处理。

---

## 参考运行
若需重新生成图与导出，请在项目根目录运行：

```
python src/example.py
```

生成后，查看：
- 单层图：`outputs/static/figures/*.png`
- 边与矩阵：`outputs/static/*.csv`

如需我将本报告转换为 PPT（包含上述四层图与要点），可以在当前环境安装 `python-pptx` 后生成（未在 `requirements.txt` 中）。如果你希望我直接生成 PPT，请告知，我会自动安装依赖并输出 `outputs/static/report.pptx`。