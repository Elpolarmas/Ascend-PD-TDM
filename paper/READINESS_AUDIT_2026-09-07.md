# PD-TDM 完整初稿准备度审查（2026-09-07）

## 1. 结论与审查范围

项目已有系统实现、中文完整长稿、T6/MaaS/F5 数据和候选图，足以启动完整英文初稿。
目前尚不满足 ICASSP R0：中文稿重建为 14 页；缺正式模板、数字 manifest、机制图和
最终紧凑表；baseline 命名、证据等级与测量表述未统一。优先工作是收敛故事、数字回源、
重写与验收；增加模态不能替代公平性和方法正确性的补强。

本轮读取 paper 的计划、claim、材料、文献审计、全部 sections，检查绘图/指标计算代码、
selector、MaaS 两份原始 JSON 和图表聚合，核对会议官方说明及最近邻一手来源。
这不是全部原始 T6/遥测的逐条复算，也没有执行新的 NPU 实验；相应事项仍待 T1 关闭。

## 2. 现有资产与准确状态

| 项目 | 观察 | 对 R0 的意义 |
|---|---|---|
| 中文正文 | 七个 section 文件（含摘要），ctexart；干净 XeLaTeX/BibTeX 重建 14 页 | 是素材库，需重写为四页英文论证 |
| 引用 | references.bib 已有 EcoServe OSDI 2026 正式条目；完成 BibTeX 后引用可解析 | 无需从零补文献；最终要改为官方数字引用样式 |
| 图表 | T6/MaaS/F5 PDF 已存在，但绘图脚本仍有 Sarathi-Serve baseline 图例 | 已有图不能等同于已验收主图 |
| 数字清单 | figure_metrics.json 存在；manifest.json 不存在 | 摘要/表图每个数字还缺来源、版本、窗口和聚合闭环 |
| 写作骨架 | WRITING_PLAN 已限定三条 claim，STORYBOARD 不存在 | 9/7 冻结一页故事，避免沿用长稿章节比重 |
| 多模态 | 已归档的 VLM_EXPERIMENT_REPORT 是实验前设计，含公开基准、模拟先验与 TBD | 不能作为本项目已完成 VLM 验证 |
| 历史状态 | D-018 仍写 T0 暂停，旧 TASKS 写 9/6 成稿 | D-020 覆盖；旧说明保留历史但不得再驱动执行 |

## 3. 必须补齐的内容（按阻断程度）

### P0-A：贡献与实际实现对齐

1. 统一 baseline 为 vLLM-Ascend CP；Sarathi-Serve 是技术来源，不能将本地 CP 实现
   的数据写成 Sarathi-Serve 实测。正文、caption、图例、claim ledger 必须全部一致。
2. 最近邻 EcoServe 已做同实例时间分离。主张应落在单 TP 副本内 bounded continuation、
   迭代级 pure-phase 与显式阶段服务机会的组合，以及 Ascend 上的证据；不能以改名或
   换平台本身替代 novelty。写一段具体设计差异，而不是仅声明“场景不同”。
3. 说明实际使用的 ratio、bucket cap、urgency、starvation、controller 与 fallback。
   selector 含多种 override；不能因论文不讨论 PID 就假定实验是严格静态 ratio。
4. token bound 约束工作量，不证明固定 wall-clock stall 上界；每请求平均 TPOT 不保证
   每个 token 间隔。删去或限定“bounded stall 保证”“满足所有 token tail”等推断。
5. Qwen3-8B/TP=2 是已选择的部署配置；当前材料不证明 8B 权重必须占满两张 64GB NPU。
   用预算约束表述，不写未经测量的显存必要性。

### P0-B：数字、统计和公平性

- T6 以 m31-fix 三 seed 为主证据，但 headline 仍要 raw → per-seed → aggregate 回算。
  conv +34.8pp、code k=2.8 +77.8pp、code k=4.9 约+403% 是不同工作点/指标。
  展示完整选定 SLO 曲线与离散度；3 seed 不足以随意宣称统计显著。
- 公平性表不只写“同两卡同模型”：需版本、backend/graph、token budget、KV/并发限制、
  arrival、输出策略、warm-up、超时和统计窗口。backend 不同会限制因果归因。
- `aggregate_window` 按到达落在 `[warmup,end)` 的请求聚合，并把符合 SLO 请求的完整
  输出 token 除以窗口时长；这属于 arrival-cohort 口径，需说明排空与完成情况，不能
  自动当作稳态最大可持续吞吐或直接窗口内发出的 token 数。
- 请求达标率和 token Goodput 必须分开：变长输出时
  `sum(o_i*I_i)/T != sum(o_i)/T * sum(I_i)/N`（一般情形）。VLM 报告中的简化乘法
  仅在等输出长度或相应加权条件下成立。
- TPOT p99 在当前主指标中是“每请求平均 TPOT 的跨请求分位数”，不是 token ITL p99；
  如保留“无长停顿”claim，需另看逐 token tail，否则删除该保证。

### P0-C：MaaS 计量与缺失值（本轮已回源）

来源：`results/maas_replay/maas_{sarathi,pdtdm}_rs0.10.json` 的 summary、window、joined。
内部文件名 sarathi 保留作 provenance，不改原始路径。

| 字段 | CP | PD-TDM |
|---|---:|---:|
| 提交/成功请求 | 35,441 / 35,441 | 35,441 / 35,441 |
| 客户端错误 | 0 | 0 |
| 到达范围（秒） | 0–4559.8413 | 0–4559.8413 |
| 配置 duration / warm-up（秒） | 34,200 / 30 | 34,200 / 30 |
| 指标窗口（秒） | [30,34200) | [30,34200) |
| 窗内请求 | 35,345 | 35,345 |
| 窗内 matched / unmatched | 35,324 / 21 | 35,342 / 3 |
| TTFT mean / p99（ms，summary） | 952.16 / 1316.275 | 808.44 / 1179.984 |

**已确认：**76 分钟是到达时间轴，34,170 秒是现有指标分母；不是同一种时长。
这不足以直接证明哪个数错了，但正文只写“76 分钟 replay”而不解释分母会误导读者。
历史 V12_REPORT 还记录另一组实际完成时间，T1 需核查其来源/定义；在此之前不擅自
用 76 分钟或实际运行时长重算 headline。

`maas_rows` 只留下 matched 且延迟齐全的成功请求；`maas_sensitivity` 用这些请求数作
attainment 分母。F5 绘图则用 n_in_window。最终需统一主口径（建议全部窗内 arrivals，
错误/未知按未证实达标处理），并单列 matched-only sensitivity。MaaS 两边 unmatched
数量不同，0 client error 不意味着 100% 遥测匹配。

×1.2 是 post-hoc SLO sensitivity。正文可保留经过审核的 TTFT shift 与原始宽松 SLO
ceiling，但不得将事后阈值当作预注册业务目标。

### P0-D：四页写作闭环

- 英文稿由问题/差异 → 设计 → 设置与主证据 → 支持与局限 → 结论构成。
- 两图一紧凑表：机制时间线、T6 主图、MaaS/边界。旧四路对比、整张 F5 矩阵、历史
  kernel 讨论和长控制器说明退出正文。
- 方法需让读者能复现最小调度规则；评估需让读者理解请求筛选、SLO 与配置选择。
- 所有图和 caption 直接解释支持哪条 claim；正式样式、字体、页数和引用属于 R0 门槛。
- 不用“limitation”掩盖无法确认的主要数字、配置或选择偏差；有争议证据应删/降级。

## 4. 补实验决策与降级路线

| 优先级 | 内容 | 为什么需要 | 无新设备时的 R0 策略 |
|---|---|---|---|
| must，离线可做 | 数字回源、配置核查、统计口径、telemetry invariant 抽查 | 决定已有结果是否可信 | 必做，不能以 deadline 豁免 |
| 最优先条件性 run | CP budget sensitivity | 固定2048未排除 baseline tuning 替代解释 | 明确只比较固定配置，不称 best-tuned 优势；保留审稿风险 |
| 条件性 run | ratio sensitivity | 验证阶段接口的交换价值 | 只写可配置接口/观测行为，不称最优控制 |
| 条件性 run | 无 timeout 的多 seed 负对照 | 支撑稳定边界而非错误/超时导致的退化 | 现有负区仅作为探索性信号，删精确泛化百分比 |
| 因果 claim 才 must | 同 backend、等工作量隔离 | 排除执行路径差异 | 删除单变量机制归因 |
| useful/暂缓 | VLM/跨模型 | 增强外部适用性与会议关联 | 先完成 Qwen3 论文，不把先验当结果 |
| omit | PID、大规模多平台、全部矩阵重跑 | 无助三天内关闭当前核心论证 | 本轮停止 |

实验矩阵、准入/退出时间见 TASKS。不能靠单 seed 搜参后只选最好结果作稳定 headline；
需披露探索过程并独立/配对验证。若公平性无法成立，必须调整论断或重新评估投稿准备度。

## 5. Related Work 还应补什么

现有17项文献数量已经足够，缺的是紧凑的差异论证和引用与 claim 的对应。

- Sarathi-Serve：承认 chunked prefill 与 stall-free batching 的既有贡献；直接测量对象
  是 vLLM-Ascend CP。依据 [OSDI论文](https://apanwariisc.github.io/publications/osdi-2024-sarathi-serve/osdi24-sarathiserve.pdf)。
- EcoServe：同实例时间分离加多实例滚动协调是最近邻，应明确 PD-TDM 的单副本、迭代级
  调度区别。当前 BibTeX 已对应 [OSDI 2026 正式条目](https://www.usenix.org/conference/osdi26/presentation/du)，
  不必换成旧 arXiv 标题；不声称首次 temporal disaggregation。
- DistServe/Splitwise：强调资源池与 KV 转移边界；不把两卡1P1D实验推广到一般分离系统。
- 代表性空间复用：选1–2篇说明对可控计算分区/并发的依赖；只陈述已核验的实现约束，
  不把当前部署限制写成所有 Ascend 硬件永远不支持的结论。
- 正文未引用的文献不必强塞；arXiv 条目在 R0 前核对是否已有正式版本。此次仅复核
  关键最近邻，不宣称完成截至9/7的穷尽文献检索。

## 6. 投稿准备与验收

官方 [CFP](https://2027.ieeeicassp.org/call-for-papers/) 仍列 2026-09-16 全文截止，并包括
Machine Learning and Generative AI / Applied Signal Processing Systems。由此判断文本
serving 可以论证会议关联，但不保证评审认可，不能仅靠会议 scope 宣称 venue-fit 已关闭。

[当届 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) 要求英文、最多4页
技术内容、总计最多5页；第5页只放允许的非技术内容。还需核对 ≥9pt、字体嵌入、作者
与投稿元数据一致；2027要求每位作者 ORCID。采用当届提供的模板，不套用旧会议模板。

R0 必须同时交付 PDF、源码、数字 manifest 和 review note。后者重点请导师判断：
novelty 是否足够、baseline 公平性是否可接受、MaaS 口径是否清楚、会议关联是否有说服力。
作者/单位/资助等未知项不得虚构，投稿前补齐。

## 7. 本轮验证记录

- 在 `/tmp/ascend-paper-audit/` 用 XeLaTeX → BibTeX → XeLaTeX 多轮构建中文原稿，
  产出14页，非正式模板；引用可解析。此验证不代表英文 R0 已完成。
- 已直接读取 MaaS 两份 raw JSON 并核对上表；未修改原始结果与测量脚本。
- 本轮不运行 NPU、不更改调度器、不发送导师消息、不进行投稿。
- 时间表、入口状态与 claim ledger 的局部纠偏记录在 D-020；尚未完成的正文、图表、
  manifest 和实验保留未完成状态。
