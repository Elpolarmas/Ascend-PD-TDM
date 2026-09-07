# PD-TDM ICASSP 2027 任务总表

> D-020，2026-09-07 重排；唯一权威执行队列。内部时间均为北京时间。
> **从今天开始准备，2026-09-09 22:00 前冻结完整英文 R0 初稿。**
> 旧版 9/6 初稿、9/7 送导师、9/5 低交互及 V0 阻塞安排全部取消；历史材料保留在
> `../memory/archive/paper_cleanup_2026-09-07/`。完成状态以实际产物验收，不按日期自动完成。

## 1. 目标与范围

交付一篇能独立通读、事实可追溯、可由导师评审的完整英文论文，包含真实图表、参考文献、
局限与结论、可编译源码及 PDF。不能用提纲、中文长稿或核心结果占位代替。
合格初稿不等于已达到充分的录用竞争力，公平性和 novelty 风险必须显式管理。

- 目标沿用 ICASSP 2027 regular paper；9/14 内部定稿，9/15 计划提交。
- 官方全文截止 9/16（投稿页明确对应北京时间9/17 20:00）；4 页技术内容，第 5 页仅参考文献、资助致谢及伦理声明等允许内容。
  2026-09-07 已核对 [CFP](https://2027.ieeeicassp.org/call-for-papers/) 和
  [Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)。
- 主对比：同平台 vLLM-Ascend Chunked Prefill (vLLM-Ascend CP) vs PD-TDM。
- 主证据：Qwen3-8B、2×Ascend 910B3、TP=2、T6 m31-fix 三 seed。
- MaaS 是支持性证据，F5 是探索性边界，VLM 是条件性增强项，不阻塞 R0。
- 主张限于 bounded pure-phase 设计、已评估配置的端到端收益和边界；不声称纯调度
  单因素因果增益、在线最优 ratio、首次时间分离或跨平台普适性。

官方当前未列regular paper的单独摘要截止日；摘要随全文提交。日期与来源见README。

## 2. 三天时间盒

| 日期与截止 | 工作 | 必须交付 | 验收/失败处理 |
|---|---|---|---|
| 9/7 剩余时间，22:00 | T0 故事；T1 数字与公平性审计；T2 模板布局 | STORYBOARD、最小 manifest、公平性/测量表、正式模板骨架 | T6 headline 回源；MaaS 窗口和缺失匹配交代清楚；不明数字删除或降级 |
| 9/8 12:00 | 完成 T1/T2；T4 补实验决策 | 两张图与紧凑表方案冻结；must/useful/omit 清单 | 实验准入关闭；不让新实验进入成稿关键路径 |
| 9/8 22:00 | T3 英文全文，摘要最后写 | 从标题到参考文献完整的英文 PDF | 真实结果，无缺失章节；允许尚待润色，不能仍是提纲 |
| 9/9 12:00 | T6 内容与证据冻结 | 最终图表、数字清单、limitations、引用 | 只纳入已审计结果；未完成 VLM/补实验退出 R0 |
| 9/9 18:00 | 全文模拟审稿并修正 | 四页技术内容的候选 PDF | 核对 novelty、方法、指标、baseline、统计、图文与引用 |
| **9/9 22:00** | R0 验收与冻结 | `build/PD-TDM_R0_2026-09-09.pdf`、源码、manifest、REVIEW_NOTE | 所有硬门槛通过；未达标则明确报告阻断，不标成合格初稿 |

若 9/7 剩余时间不足，任务顺延到 9/8 上午，先削减非必要图表和实验；保留 9/8 晚全文
成稿与 9/9 审查时间。此时间盒不是后台自动执行或定时发送承诺。

## 3. 执行队列

### T0 — 故事与贡献冻结（P0，9/7）

- [x] 重排日期，完成首轮仓库审查，记录 D-020。
- [ ] `STORYBOARD.md`：一句话 thesis、问题、最近邻差异、三条 claim、章节证据映射。
- [ ] 区分“部署只分配一个 TP 副本”与“模型必须 TP=2 才能装下”；无证据不写后者。
- [ ] 保留设计事实、限定端到端结果、探索性边界；实际 ratio/urgency/controller 配置
  与论文所描述方法一致。

### T1 — 数字与公平性审计（P0，9/7–9/8 上午）

- [ ] 复算 T6 conv k=3.0/S1、code k=2.8/S1 和 k=4.9/S1；分别保存三 seed 值、中位数、
  配对差值与离散程度，不拼接不同工作点最大增益。
- [ ] 核对模型/精度、TP、版本/commit、attention backend、graph 模式、token budget、
  max_num_seqs、KV 设置、arrival seed、warm-up、窗口、错误与匹配数。
- [ ] 明确 TTFT、每请求平均 TPOT、跨请求 P99 与逐 token ITL tail 的区别。
- [ ] 固定 `A=N_joint/N_arrivals`、`G_tok=sum(o_i*I_joint)/T`；变长输出时不能用
  `raw output throughput × request attainment` 代替 token Goodput。
- [ ] MaaS 区分 76 分钟到达流、配置统计窗口 `[30,34200)` 秒与实际完成/排空时间；
  审核 34,170 秒分母，不擅自换成 4,560 秒。
- [ ] MaaS 披露总请求 35,441、窗内 35,345、CP/PD unmatched 21/3；延迟分布、
  attainment 的样本集合分别说明；×1.2 只能标 post-hoc。
- [ ] 建立 `data/manifest.json`：全部入稿数字的 raw 路径、hash、配置、seed、窗口、SLO、
  脚本/命令、聚合规则、raw/derived/post-hoc、审核状态；不能只复制旧 summary。

核心配置公平性无法说明时，删除相应强比较或补齐证据，不能只用 limitations 掩盖。
若 backend 不同且无隔离证据，只报告部署实现的端到端差异。

### T2 — 模板、机制图与结果图（P0，9/7–9/8 上午）

- [x] 接入当届官方 spconf/IEEEbib，数字引用和双栏版式，XeLaTeX 编译通过（9/7）。
- [ ] 将现有9页中文内容重写为英文4+1页；模板接入不等于篇幅/语言验收通过。
- [ ] Fig.1：相同 chunk bound 的 mixed CP vs bounded pure-P/pure-D 时间线；标明
  shared KV、continuation 和 phase opportunity，不画未测得的精确时间比例。
- [ ] Fig.2：T6 conv/code 严格 SLO，优先 attainment 与三 seed 离散度；明确负载轴。
- [ ] 紧凑表承载 MaaS/边界；设置文字保留公平配置，删去无法审计的数据单元。
- [ ] 图例与正文 baseline 一致；最终尺寸字体和可读性符合要求，不直接搬旧三路图。

### T3 — 完整英文正文（P0，9/8 晚）

- [ ] Introduction：具体部署预算、CP 控制量缺口、最近邻差异、三条贡献。
- [ ] Design：phase invariant、整轮 chunk bound、partial continuation、ratio/fallback；
  token bound 不等于严格时间上界或逐 token SLO 保证。
- [ ] Evaluation：设置与指标、T6、MaaS、边界、局限；明确观测与推断。
- [x] Related Work：Sarathi/CP、DistServe/Splitwise、EcoServe 和代表性空间复用，
  每类说明差异；用一手引用，避免只增加论文数量。
- [x] Conclusion 与约 100–150 词 abstract 使用同一组当前已核实 claim。

### T4/T5 — 最小补实验（9/8 12:00 准入关闭）

具体风险见 `READINESS_AUDIT_2026-09-07.md`。只有预计能在 9/9 12:00 前完成运行、
分析与核查才准入；无设备、失败或超时则收缩论断，不扩展研究主线。

| 候选 | 优先级与触发条件 | 最小设计 | 无法完成时 |
|---|---|---|---|
| CP budget sensitivity | 最优先条件性补实验；保留广义/调参公平优越性 claim 时必须 | win 点 CP 512/1024/2048/4096，固定 arrival/SLO；探索后冻结配置再做配对验证，披露选择过程 | 只比较固定 2048，不称优于 best-tuned CP；显式披露风险 |
| 静态 ratio sensitivity | 声称可调接口的实证收益时必须 | 一 win、一 boundary，少量冻结 ratio，记录实际相位与 overrides | 只声称实现配置接口，不称最优/通用规律 |
| 干净负对照 | 稳定负区定量 claim 的前提 | 先找已有无 error/timeout 且请求齐全 cell；不足才补三 seed | 只保留探索性边界，不报稳定退化百分比 |
| 同 backend 隔离 | pure-phase 单因素因果 claim 的前提 | 等工作量，同 backend/graph | 删除 kernel/纯调度归因，保留实现端到端比较 |
| VLM | useful，非 R0 门槛 | correctness、公平输入和 encoder 可观测性先通过，至多两个负载点配对比较 | 暂缓；公开数字、加250ms模拟和预期区间不算本项目实测 |

新实验需独立结果目录、固定配置/seed、日志、单 run 超时、总截止与失败处理。
不重启 PID、全矩阵、跨 GPU/Sarathi 复现、全版本移植或 1P1D 第二主线。

### T6/R0 — 9/9 完整初稿硬验收

- [ ] 全英文，所有必要章节齐全；标题、摘要、引言、结论 claim 一致。
- [ ] 核心数字回源复算；窗口和分母闭合；图表、正文、manifest 一致。
- [ ] 方法与实跑配置一致；不把接口写成已验证的自适应能力。
- [ ] 最近邻差异、公平性、局限可被独立审阅，无虚构结果/未证实强因果。
- [ ] 正式模板技术内容 ≤4 页、总计 ≤5 页；图表可读，引用与交叉引用完整。
- [ ] 干净构建通过，无核心 TODO/TBD、缺图、空表；作者信息真实，未知元数据在
  review note 明示，投稿前补齐，不虚构姓名/单位。
- [ ] PDF、源码、manifest、review note 成套冻结；note 聚焦 novelty、baseline、
  测量口径与 venue relevance，列明剩余风险。

## 4. 后续节点

| 日期 | 交付 |
|---|---|
| 9/10 | R0 供导师第一轮评审；用户安排发送，反馈未到先自行审查 |
| 9/10–9/11 | 处理拒稿级问题与最小补强，不等待反馈才修正 |
| 9/12 | R1，数字/引用/语言复核与第二轮模拟审稿 |
| 9/13 | 技术内容冻结；未关闭问题形成明确投稿风险决策 |
| 9/14 | 内部定稿，作者/ORCID/EDICS/元数据与 PDF 格式检查 |
| 9/15 | 计划投稿并核对回执，实际提交由用户安排 |
| 9/16 | 官方截止缓冲 |

当前入口：**T0 → T1/T2 → T3 → T6/R0**；补实验服务于成稿。
