# 决策日志

> 记录每次方向调整的来龙去脉。任何跟 `PROJECT.md` 「不能偏离的几条原则」抵触的修改 → 必须先在这里写一条新决策。时间倒序。

格式:每条独立,固定字段。状态 = active(当前生效)/ superseded(被新决策取代)/ pending(候选,未拍板)。

---

## D-022(2026-09-07)清理 paper 为单一 ICASSP 投稿工作区

**状态:active**

- `paper/` 只保留当前英文四页稿、官方 spconf/IEEEbib 与官方示例、当前计划/证据文档、
  引用文献、正文实际使用的 T6 PDF 图和最小 paper-facing 数据。
- 旧任务表、中文模板兼容文件、VLM 条件性方案、未引用候选图和空表目录移至
  `memory/archive/paper_cleanup_2026-09-07/`，不删除。
- 新增 `experiments/plot_icassp_main_figure.py`，只生成当前正文引用的主图，防止重绘时
  将候选图重新写回 paper。
- 构建目录当前只保留 `main.pdf`；LaTeX 辅助文件可由 `latexmk` 重建。

---

## D-021(2026-09-07)接入ICASSP2027官方LaTeX模板与截止时间核验

**状态:active**

- **触发:** 用户要求核查单独abstract截止并直接接入官方模板。
- **实现:** main改为article+官方spconf；IEEEbib数字引用；新增authors与过渡中文字体文件。
  保留原sections及原始数据；旧入口文件归档；官方文件URL/hash记录在SOURCES.json。
- **验收:** XeLaTeX/latexmk编译通过，9页US Letter PDF，字体嵌入、引用解析；已检查首页。
  仍为中文长稿、作者占位和旧图表，不能标记英文四页R0完成。
- **日期:** 官方regular投稿说明未列独立摘要截止；全文截止明确为北京时间9/17 20:00。
  内部9/9初稿、9/15计划提交不变。依据paper/README.md所列官方页面。

---

## D-020(2026-09-07)以9/9完整初稿重排计划，先关闭论文证据与表达缺口

**状态:active**

- **触发:** 用户要求从今天开始，以9/9完整合格初稿为目标，先重设计划再审查补充内容。
- **取代:** TASKS D-019旧日期及D-018的T0暂停/V0前置规则；保留D-017同平台与三claim范围。
- **安排:** 9/7故事、数字审计与模板；9/8晚完整英文稿；9/9审查并于22:00冻结R0。
- **优先级:** 配置公平、数字回源、统计口径、最近邻差异和四页完整性为必做；VLM与新增
  实验只有能关闭具体claim风险且不阻塞成稿才准入。不得用未审计数字换取按时完成。
- **发现:** 中文稿14页；manifest/机制主图未落地；MaaS到达窗口与统计分母不同、unmatched
  请求与attainment分母需要说明；不能把request attainment直接乘raw token throughput。
- **产物:** paper/TASKS.md、READINESS_AUDIT_2026-09-07.md；旧任务表已归档。
- **范围:** 本轮完成计划与审查；不声称已完成英文论文或完整数据复算。9/10供导师评审，
  9/14内部定稿、9/15计划提交；消息发送和投稿由用户安排。

---

## D-018(2026-08-19)新增 ICASSP venue-fit / 多模态模型决策门，T0 暂停

**状态:部分被 D-020 取代；V0 不再阻塞 T0，模态候选分析保留。**

- **触发:**
  - 在冻结 T0 故事前，需要确认 ICASSP 作为信号处理旗舰会议是否要求增加音频、视频或
    图像模型实验，以增强论文与目标社区的相关性。
  - ICASSP 2027 官方 scope 同时包含 Applied Signal Processing Systems、Machine Learning
    and Generative AI、Speech and Language Processing、Image/Video/Multidimensional SP；
    因此文本 LLM serving 并非天然越界，但“通用系统工作与 SP 社区的关联”仍是评审风险。
  - 本地版本审计显示 Qwen2.5-VL-7B 有 vLLM-Ascend 在线 serving benchmark，且 PD-TDM
    scheduler 保留 encoder-input scheduling；Qwen2-Audio 在该版本只能方便地做离线推理，
    OpenAI-compatible server 不支持 audio input，Whisper 未支持。

- **新决策:**
  1. 暂停 T0，先执行 V0 venue-fit 决策门；多模态实验是条件性增强项，不因会议名称直接
     设为必做。
  2. 候选模型优先选择 **Qwen2.5-VL-7B-Instruct**，因为它与当前 7B/8B 自回归 decoder
     规模接近、同一 vLLM-Ascend 版本有在线 benchmark 路径，并能真实引入图像信号输入。
  3. 不选择纯图像生成/图形学模型：PD-TDM 的 Prefill/Decode 调度对象和 TTFT/TPOT 指标
     不可直接迁移，会产生一篇新的论文主线。
  4. 不选择 Qwen2-Audio/Whisper 作为当前首选：在线 serving harness 或平台支持不满足
     最小改动原则。若后续已有可用在线音频服务路径，可重新评估。
  5. VL 实验只有同时满足以下条件才进入 ICASSP 正文：无需修改 PD-TDM 核心机制；CP 与
     PD-TDM 语义公平；encoder 开销可记录并解释；结果对 claim 提供新信息；最终只占一行
     结果或一个小子图。否则不做/不报，以 ML & Generative AI / Applied SP Systems framing
     建立 venue relevance。

- **V0 输出:**
  - venue/track 定位；是否需要模态实验；候选模型与数据集；最小实验矩阵；Go/No-Go 标准；
    对四页故事和 claim 的影响。

---

## D-017(2026-08-13)目标切换为 ICASSP 2027 四页投稿，先冻结故事再按风险补实验

**状态:active(2026-08-13)**

- **触发:**
  - 导师建议赶 ICASSP 2027；正式论文截稿为 2026-09-16，技术内容限 4 页。
  - 现有 D-016 计划以“完整系统因果闭环”为完成门槛，任务范围偏向长篇系统论文，
    不适合 ICASSP 四页稿的时间与篇幅约束。
  - 项目实验均运行于 vLLM-Ascend / Ascend 910B3，无法也没有必要跨平台复现 GPU
    Sarathi-Serve；导师确认直接比较同平台的 vLLM-Ascend Chunked Prefill 即可。

- **新决策:**
  1. 当前唯一投稿目标为 **ICASSP 2027 regular paper**；内部完成日为 2026-09-14，
     2026-09-15 提交，官方截止日仅作缓冲。
  2. 直接 baseline 固定命名为 **vLLM-Ascend Chunked Prefill (vLLM-Ascend CP)**。
     Sarathi-Serve 只作为 Chunked Prefill 的思想来源和相关工作，不作跨 GPU/NPU 性能比较，
     不写“PD-TDM 优于 Sarathi-Serve”。
  3. 主线固定为：Chunked Prefill 控制单次 Prefill 工作上限，但 P/D 跨 iteration 的服务
     份额仍隐式依赖 mixed batch；PD-TDM 用 bounded pure-phase iteration 将 chunk bound
     与阶段服务机会解耦，在 Prefill 压力高且 Decode 有 TPOT 余量时改善联合 SLO，
     deep-decode/TPOT-tight 区域作为负对照。
  4. 当前论文只保留三类 claim：设计事实、Ascend 同平台端到端结果、适用边界。
     不把 PID、动态最优 ratio、固定 mixed-batch tax、kernel 根因或跨平台普适性列为贡献。
  5. 先用现有数据完成四页初稿，再做 reviewer-risk audit；只有会破坏主 claim 的缺口才
     进入补实验关键路径。完整 Sarathi 移植、动态控制器、全矩阵重跑和跨 GPU 对比停止。
  6. T6 三 seed Azure burst 为主证据；MaaS 为外部有效性支持；F5/F5b/F5d 只承担边界与
     负对照，其中 single-seed/timeout 限制必须显式披露。

- **执行入口:**
  - `paper/TASKS.md`：唯一权威任务队列，按任务逐项推进。
  - `paper/WRITING_PLAN.md`：ICASSP 四页结构和篇幅预算。
  - `paper/CLAIMS.md`：下一任务中按本决策冻结并统一 baseline 命名。

- **对 D-016 的影响:**
  - D-016 的代码路径、Attention dispatch 和旧 FIA 消融审计仍有效。
  - D-016 将 Force-FIA、等工作量微基准和 best-tuned CP 全部设为投稿前硬门槛的安排被
    本决策取代；这些实验改为初稿完成后的条件性风险项。

---

## D-016(2026-08-06)论文核心收敛为 bounded phase-pure，旧 mixed-tax/FIA 归因重新开放

**状态:superseded by D-017 for submission planning; technical audit remains valid**

- **触发:**
  - 将 PD-TDM 转换为论文时，重新审计 vLLM v0.11.0 / vllm-ascend
    v0.11.0rc1、`c3_cp`、`TDMScheduler`、ModelRunner 和 Attention dispatch。
  - 发现旧 narrative 同时把 `c3_cp` 直接称作 Sarathi-Serve、把 C3 简化成“始终
    FIA”，并用可能未传播到 TP worker 的 Scheduler-time monkey patch 得出“kernel
    贡献约等于零”。这些结论不能继续作为核心 A 的已验证证据。

- **新决策:**
  1. 当前论文核心固定为 **A：保留 Prefill chunk cap 的 bounded phase-pure
     scheduling，与 Sarathi-style mixed Chunked Prefill 的比较**。
  2. 动态 PD ratio 是后续优化方向，不承担当前论文第一贡献；ratio 在当前论文中准确
     描述为 P iteration 的长期频率/信用，chunk cap 控制单次 P 工作量。
  3. `c3_cp` 正式命名为 **vLLM V1 Chunked Prefill (Sarathi-style)**，不再直接称为
     Sarathi-Serve。vLLM V1 已实现 Sarathi 核心在线调度思想，但不是完整研究系统原样
     合入。
  4. 当前 CP 与 TDM 均使用 native state dispatch：CP 的 mixed/partial Prefill 进入
     FIA，pure initial Prefill / Decode-only 走专用路径；TDM 的 pure P/D 主要走专用
     路径。“C3 始终 FIA”作废。
  5. 旧 `m31_fia` 消融降为 **未验证**。必须把 Force-FIA 放到模型 worker 初始化路径，
     并用 rank 日志、调用计数或 NPU profiler 证明实际执行后方可引用。
  6. 核心 A 的必做验证为：运行时 P/D + Attention-state telemetry、CP token-budget
     tuning、worker 级 TDM-ForceFIA、同 FIA 等工作量 Mixed-vs-Pure 微基准，以及
     代表性双 SLO 端到端验证。
  7. 在验证完成前，摘要、Motivation、Contribution 和 Conclusion 不定稿；稳定章节和
     实验方案继续整理。

- **核心假设:**
  - H1：统一 backend 和等工作量后，目标 workload/SLO 区域存在
    `T_P(P)+T_D(D)+T_switch < T_mix(P,D)`。
  - H2：Prefill 压力高、TTFT 紧且 Decode 有 TPOT 余量时，PD-TDM 优势增大。
  - H3：Decode-heavy、TPOT 极紧、低负载或 mixed cost 很小时，优势缩小或消失。

- **影响文档:**
  - 新增 `PAPER_CORE_A_PLAN.md` 作为论文/实验当前主入口。
  - `README.md`、`PROJECT.md`、`design/paper.md` 增加 D-016 覆盖提示。
  - 后续实验完成后再更新 `EXPERIMENTS.md`、`FINDINGS.md` 和论文正文。

- **Supersedes / reopens:**
  - 覆盖 D-014 中把 `c3_cp` 直接命名为 Sarathi chunked prefill/open-source impl 的表述。
  - 重新开放 D-009“kernel 贡献约等于零”的结论；旧消融在确认 TP worker 生效前无效。
  - 重新开放 D-015/F0+“mixed-batch tax 已由现有 cycle telemetry 因果证实”的结论；
    现有数据只证明 bundle-level 时间差，需同 FIA、等工作量实验归因。

---

## D-015(2026-05-26)导师汇报 3 challenges + first-principles 辩护 + F0+ 升 P0 / F5 降 P1

**状态:active(2026-05-26)**

- **触发:**
  - 5/26 mentor 汇报回来,老师对 PD-TDM 提出 3 个核心质疑(原话):
    1. **逻辑漏洞**:chunked prefill 本身就是要减少 prefill 对 decode 干扰才做混合 batch + 优先 decode;PD-TDM 把 prefill / decode 拆开,等于又回到 prefill 干扰 decode,直觉上更差 — 凭什么反而好?
    2. **负载偏置**:实验优势是否特定于数据集(prefill 长 decode 短)?是否要构造 decode-heavy 工作负载先确认不是负载特性带来的
    3. **prior art**:TDM 这种思想很简单(除 SLO 优化),P/D 时分复用机制有没有别的工作做过?(用户自行 web search,不在我的任务里)

- **关于 challenge 2(负载偏置)— 实测 trace token 分布,确认老师质疑成立:**
  | Trace | ContextTokens mean / median / p90 | GeneratedTokens mean / median / p90 |
  |---|---|---|
  | conv | 1155 / 1020 / 2735 | **211 / 129 / 424** |
  | code | 2048 / 1469 / 5194 | **28 / 13 / 55** |
  - conv prompt:output ≈ 5×,code prompt:output ≈ 73× — **两个 Azure trace 都是 prefill-leaning**,不存在真正 decode-heavy regime
  - 之前底稿口头说 "conv 输入输出差不多" 是错的,要在 paper §5 Caveat 明写 "当前 4-way 结果只覆盖 prefill-leaning regime;decode-heavy 情形未验证"

- **关于 challenge 1(逻辑漏洞)— 三柱 first-principles 辩护(待 F0+ kernel telemetry 实证):**
  1. **chunk_budget 才是 Sarathi 真正止血机制,"同 batch 混合" 不是**
     - Sarathi OSDI'24 Fig.7 自己也承认:chunk_size 从 ∞ 降到 2048 带来 TPOT 改善占绝大部分;混合 batch 在 chunk 限到 2048 后边际贡献很小
     - PD-TDM 保留 chunk_budget=2048(D-006),只是把"混合"换成"phase-pure" → Sarathi 防 stall 的核心机制我们继承,没丢
  2. **混合 batch 在 NPU 上有 variable-query-length 税**
     - 混合 iter 的 attention call 里 query_len = `[chunk_size, 1, 1, ..., 1]` 极不均匀
     - 910B3 attention kernel 对变长 query 支持不像 GPU FlashAttn 便宜,触发 padding / 多次 kernel launch / sub-optimal tiling
     - c3 telemetry(5/25)实测 mixed iter 220ms vs PD-TDM phase-pure prefill 150ms + decode 39ms = cycle 189ms,差额 ~70ms 即 mixed batch tax
  3. **decode "等待时间" ≠ "P iter 时长",要看 selector 节奏 + kernel 满血**
     - Sarathi 每个 iter 220ms,decode 一直被 P 拖
     - PD-TDM 在 P iter 时 decode 等 ~150ms,但 D iter 时 decode kernel 满血(pure decode = 39ms vs mixed 含 decode ~80ms),平均 TPOT 反而更优
  - **类比**:洗衣 + 洗碗放一锅煮 vs 分开洗,后者更快 — 因为水温/清洁剂/转速各自最优。LLM 中 P (compute-bound)/ D (memory-bound) 硬件最优工作点不同,强行混在一个 attention call 反而双输

- **F0+ kernel/iter cost breakdown 升 P0(原 F0 Vanilla CB telemetry 补救合并):**
  - 目标:直接实证三柱辩护第 2 柱 — 把 mixed iter 和 phase-pure iter 的 attention kernel 耗时拆出 `variable_query_length_overhead` 这一项
  - 现状:D-013 patch 已给 iter-level 计时(c3 telemetry / m31 telemetry 各 1 cell);可能不需要重跑,只需要重新解析现有 JSON + 加 kernel-level 拆分(如果 telemetry granularity 不够再补 cell)
  - 输出:数字形如 "NPU mixed iter 有 ~15-20% variable-query overhead,phase-pure 省掉这部分"
  - 优先级 reason:**决定性**(直接验证 / 推翻 first-principles 模型)+ **便宜**(可能 0 新 run)+ **paper 武器**(mechanism-level evidence 比 workload sweep 难反驳)

- **F5 decode-heavy synthetic sweep 降 P1(从 P0 调整):**
  - 原计划:`synth_decode_heavy`(prompt 64-128, output 1024-2048)+ `synth_balanced` + `synth_prefill_heavy`,3 paradigm × 12 run × 3 wl ≈ 36 cell,T6 1 晚跑完
  - 降级理由:即使 decode-heavy 上 PD-TDM 赢,也不解释**为什么** — 老师下一句还是"凭什么不混合反而好",逻辑漏洞没补;**必须先 F0+ 把 mechanism 钉住再做 scope 验证**
  - 顺序:F0+ 出结果后,根据 first-principles 模型预测 decode-heavy 表现 → F5 验证或推翻预测 → paper §5.3 收紧 scope claim
  - 决策点:**如果 F0+ 数据显示 variable-query overhead 其实很小(<5%),first-principles 模型不成立 → 立即停 F5,回到机制重新找;不要花 1 晚验证一个错的预测**

- **paper 写作影响:**
  - §3 性能分析:把现在的 "cycle time 模型" 换成三柱 first-principles 框架(待 F0+ 实证后定稿)
  - §5 Caveat:加 "Azure conv/code 都是 prefill-leaning(实测 mean prompt/output ratio = 5× / 73×)→ decode-heavy regime 未验证";F5 跑完后根据结果再收紧或放宽
  - §3.1 motivation 重排:**承认 Sarathi 混合 batch + decode 优先是 GPU 上的最优解,我们的贡献是"NPU 上 variable-query 税 + cycle 长度差异让 phase-pure 反超"** — 不要否定 Sarathi 在 GPU 上的成功
  - §2 related work:等用户 web search 反馈 prior art,确认无前作 / 区分 contribution(challenge 3)

- **跟进事项 ranking(更新自 D-014):**
  - **P0**: F0+(kernel/iter cost breakdown)— 解析 c3 / m31 telemetry,拆 variable-query overhead;1-2 天
  - **P1**: F5(decode-heavy synthetic sweep)— 等 F0+ 后跑;1 晚 wall
  - **P1**: prior art 调研(user 自行 web search)— 决定 §2 related work + §1 differentiation
  - **P2**: L3 real Azure trace(phase 2 计划内,长 wall)
  - **P3**: phase_iters semantic bug fix(D-013 残留,不影响数据)

- **影响文档:**
  - `README.md`:加 D-015 到历史决策,跟进事项 ranking 更新
  - `FINDINGS.md`:加 Azure trace token 分布实测块
  - `MENTOR_DISCUSSION.md`:§3 性能分析待 F0+ 后重写,§5 Caveat 加 prefill-leaning 限定

- **Supersedes:**
  - D-014 跟进事项 F5(decode-heavy)P0 → P1
  - D-014 跟进事项 F0(Vanilla CB telemetry 补救)合并到 F0+(kernel cost breakdown)

---

## D-014(2026-05-26)Baseline 重定位:c1 → Vanilla CB(老 c3 chunk=8192)+ NPU 复现 Sarathi finding + 短 prompt 反例

**状态:active(2026-05-26 finalize)**

- **触发:**
  - 5/26 准备 mentor 汇报底稿时质疑 c1 命名 — 文献意义 "vanilla CB" 通常指 mixed batch(prefill+decode 同 batch),Sarathi 论文比较的就是这个;但我们的 c1(vllm-ascend AscendScheduler default + `chunked_prefill_enabled=False`)实际是 admit-driven phase-pure(单 iter 单 phase),**跟主流文献 vanilla CB 不是同一种 design**
  - 用户提醒早期 T6 sweep 跑的 c3 用 `chunked_prefill_enabled=True` + `max_num_batched_tokens=8192`,数据保存在 `results/phase_2_t6_burst_goodput/*_nonpid/c3_cp_qps0.0.json`(42 dirs × 3 seeds);Azure trace prompt cap=7000 < 8192 → chunk 实际不触发 = **mixed batch + 长 prompt 整 iter 一次跑完**,**真正的 vanilla CB equivalent**
  - 这些数据 D-013 期间被 framing 成 "c3 unfair 版本",换成 chunk=2048 的 c3-fair 作 fair baseline;**实际上两者关系不是 "fair vs unfair",是 "vanilla CB vs Sarathi chunked prefill"**

- **修订:**
  - **Baseline 重命名 + 数据源映射:**
    | Paper 名 | 原 config 名 | 数据源 |
    |---|---|---|
    | **Vanilla CB**(主流文献 baseline) | 老 c3 (`c3_cp` + chunk=8192) | `phase_2_t6_burst_goodput/*_nonpid/c3_cp_qps0.0.json` |
    | **Sarathi chunked prefill**(主流文献 baseline) | c3-fair (`c3_cp` + chunk=2048) | `c3_chunk2048_supplement/` |
    | **PD-TDM**(本工作) | m31-fix (`c2_tdm_m31_2048_fix`) | `m31fix_validate/` |
    | c4_pd 1P1D disagg(reference) | 不变 | `c4_pd_supplement/` |
  - **c1(原 vllm-ascend default phase-pure)从 paper 主线 baseline 移除** — 它在 LLM serving 文献里没有标准命名,是 NPU-specific niche design,容易引起 reviewer 困惑。代码 / 数据保留,paper 不报
  - aggregate JSON `m31fix_phase1_pointwise.json` 加 `vanilla_cb` config 列(`posthoc_m31fix_phase1.py` 改:加 `CONFIG_VCB = "vanilla_cb"` + path resolver 指向 nonpid 目录的 `c3_cp_qps0.0.json`)
  - paper 主图:`plot_paper_main_figures.py` PARADIGMS_4WAY 把 `c1_baseline` 换成 `vanilla_cb`;新加两个 heatmap function:F2b(PD-TDM vs Vanilla CB 总优势)、F2c(Sarathi vs Vanilla CB NPU 复现 + 反例视觉证据);共出 18 张 PNG(6 Pareto × 2 variant + 3 heatmap × 2 variant)

- **NPU 上首次复现 Sarathi 论文 OSDI'24 核心 finding(F2c heatmap 实测,3-seed median Δ pp = Sarathi − Vanilla CB):**
  ```
  conv s1:  -6.2  -6.5  -6.2  -3.5  -1.6  -0.5  +1.1
  conv s2:  +1.2  -3.0  +2.4 +10.1 +14.6 +17.8 +22.6
  conv s3:  -5.2  -2.0  +1.2 +12.9 +24.9 +26.2 +34.9
  code s1: +13.0 +22.4  -1.1  +1.7  +0.9  +1.1  +5.5
  code s2: +16.3 +43.6 +61.6 +70.6 +74.5 +75.2 +74.8
  code s3:  +0.6  +7.1 +32.5 +87.5 +80.0 +80.6 +80.6
  ```
  - **code workload(长 prompt)上 Sarathi 大胜 Vanilla CB**:max +88pp(code 2.8 s3)— **NPU 上首次直接量化 chunked prefill 把长 prompt 灾难救活的 finding**
  - **conv s1 / s3 低 k 上 Sarathi 反输 Vanilla CB(-2 ~ -6.5pp)**:**fresh finding**,Sarathi 论文没明示 — 短 prompt 在 chunk=8192 整 iter 一次跑完 ttft 更快,chunk=2048 切多份反向拖慢首 token

- **PD-TDM 跟两个 baseline 关系明确化(F2 + F2b heatmap):**
  - **vs Sarathi(F2,unique contribution)**:phase-pure variant of chunked prefill,strict SLO 上 +30~+77pp 增益
  - **vs Vanilla CB(F2b,总优势)**:NPU 复现 Sarathi 部分 + phase-pure 额外增益的总和,全谱完胜,max +87.5pp(code 2.8 s3)
  - **paper 须明确归 credit 给 Sarathi 的 chunk 上限设计**(vs Vanilla CB 的胜利绝大部分来自 chunked prefill,**不能 claim 是 PD-TDM 独占**);PD-TDM 真正独占的 contribution = chunked prefill 内部 phase-pure 的额外增益(F2)

- **数据无须新跑,iter telemetry 待小补:**
  - Vanilla CB paradigm-level 数据完整可用(42 cells × 3 seeds,跟 c1 同 sweep 一起跑的)
  - **Vanilla CB iter telemetry 缺**(D-013 patch 是 5/25 加的,老 c3 是 5/22 跑的) — 待 F0 补救(~30min wall,1-3 cell 用 D-013 patch 重跑)
  - **当前已可重画 18 张 PNG**(`posthoc_m31fix_phase1.py` + `plot_paper_main_figures.py --variant {3way,4way}` 已修),paradigm-level Pareto / heatmap 全部 ready

- **影响文档:**
  - `README.md`:当前状态 + thesis 版本 + Outdir 段说明 vanilla_cb 数据源
  - `T6_FINDINGS.md`:§"c3 vs c1 baseline 复现" 整段重写为 "Vanilla CB vs Sarathi" + NPU 复现 finding + 短 prompt 反例
  - `FINDINGS.md`:加 D-014 finding 块
  - `EXPERIMENTS.md`:paradigm 命名表更新 + Vanilla CB 数据源指向
  - `PROJECT.md` §2.4 baseline 列表:c1 移除,Vanilla CB 加入

- **影响代码:**
  - `experiments/posthoc_m31fix_phase1.py`:加 `CONFIG_VCB`,_path_for / ALL_CONFIGS / SHORT 都加 vanilla_cb
  - `experiments/plot_paper_main_figures.py`:PARADIGMS_4WAY 把 c1_baseline 换 vanilla_cb;加 `plot_advantage_heatmap_vs_vanilla` + `plot_sarathi_vs_vanilla_heatmap`(F2b / F2c);共出 9 个 figure × 2 variant = 18 PNG

- **影响范围 / 不能偏离的原则更新:**
  - **D-011 原 §4 "对手 = 4-way paradigm comparison" baseline 命名更新**:C1/C3/M3.1/c4_pd → Vanilla CB / Sarathi / PD-TDM / c4_pd
  - 新加一条:**Paper baseline 命名必须跟主流 LLM serving 文献对应**(Vanilla CB / Sarathi chunked prefill 都是文献术语,c1 这种 NPU-specific 命名不再用)
  - 新加一条:**NPU 复现 Sarathi finding + 短 prompt 反例**作为 paper §3.1 motivation 实证 + §5.3 caveat fresh finding

- **跟进事项:**
  - F0(P0+,~30min wall):Vanilla CB iter telemetry 补救(1-3 cell × D-013 patch 后 scheduler 重跑,关闭 §4.1 单步实测表 "待补 telemetry" 标签)
  - 其它跟进 F1-F4 见 D-013 / MENTOR_DISCUSSION.md §7

- **Supersedes:**
  - D-013 §"c3 vs c1 baseline 复现(部分)" framing:**改成 Sarathi vs Vanilla CB 基准对照**;原 "conv strict 上 c3 反输 c1" caveat 实际是 NPU c1 特有现象,**主线 baseline 切换后该 caveat 不再适用**
  - D-013 paper 主线 4-way baseline(C1/C3/M3.1/c4_pd)→ D-014 重定位为 4-way(Vanilla CB / Sarathi / PD-TDM / c4_pd)

---

## D-013(2026-05-24 启动 / 2026-05-25 finalize)chunked_schedule bug fix + Phase 1 全 m31 重跑 + 机制 review

**状态:active(5/25 Phase 1 数据 finalize + 机制 telemetry verify)**

- **触发:**
  - 5/23 LONG_PROMPT_SMOKE 跑完得"决定性 negative result":m31 在 long-prompt code 上 0%~67.8% meet 而 c3-fair 100% meet
  - 5/24 用户质疑 chunk 配置(m31 cmdline 8192 vs c3 cmdline 2048),deeper 挖发现 m31 内部 `prefill_chunk_tokens=2048` 实际硬 cap,两者 chunk granularity 实际相同
  - 进一步看 iter trace 发现 **277/1073 个 "decode" iter 实际跑 prefill chunk(reqs=1, tokens=2048)而 0 decode 输出**
  - 跟踪到 `chunked_schedule.py` waiting loop 没 gate 在 phase,decode iter 偷塞 prefill 但因 L396 `len(scheduled_req_ids)==0` 判断 skip decode loop → 双输

- **修复:**
  - `vllm-ascend/vllm_ascend/core/tdm/chunked_schedule.py` L194 waiting loop while 改:
    ```python
    while (self.phase == "prefill" and self.waiting
           and token_budget > 0 and chunk_budget > 0):
    ```
  - 注释为 Diff #6,引用 design intent comment(L13, L120-125)证明这是漏写不是 design choice

- **修复验证(5 cell × 3 seed critical scan):**
  - long_prompt code @ (1000/500):0% → 100% meet,gp +19% vs c3-fair
  - T6 code k1.4 s2:bug -9pp → fix +1pp(swing +10pp)
  - T6 code k1.4 s3:bug -12pp → fix 0pp tied(swing +12pp)
  - T6 conv k1.4 s1:bug +27pp / fix +27pp(strict TTFT m31 winning regime 不变,大幅胜)
  - T6 conv k1.4 s3:bug +3.9pp / fix +2.7pp(基本不变,bug 在短 prompt 影响小)
  - **结论:m31-fix 在测过 5 cell 上全部 ≥ c3-fair,无 cell 输**

- **memo 数字校正(T6_FINDINGS 普遍夸大):**
  - "code s3 -43pp 最戏剧" → 实测 -12pp(夸大 3.5×)
  - "code s2 -23pp" → 实测 -9pp(夸大 2.5×)
  - "conv s3 tied" → 实测 +3.9pp(已是微赢)
  - "code s4 -8pp" → 实测 0pp tied
  - 三个文档(T6_FINDINGS / LONG_PROMPT_SMOKE / mech_brainstorm)5/24 已加 SUPERSEDED 头

- **数据 invalidate:**
  - T6 `phase_2_t6_burst_goodput/*_pid` 168 个 m31 cells 全废
  - `azure_m31_fia_ablation/` kernel ablation 6 cells 也带 bug(若需要再重跑)
  - c1/c3 数据不受影响,复用

---

### Phase 1 sweep 完成(2026-05-25 01:31:54 UTC)

- script `/tmp/run_m31fix_phase1.sh`,log `/tmp/m31fix_phase1.log`,output `results/m31fix_validate/`
- 实测 129 cells(126 标准 matrix + 3 carryover),0 失败 run
- post-hoc 脚本:`experiments/posthoc_m31fix_phase1.py`
- 聚合数据:`results/phase_2_post/m31fix_phase1_pointwise.json`

---

### Phase 1 结果 — Paradigm-level winning region(m31-fix vs c3-fair,3-seed median)

**conv 21/21 全胜**(12 decisive +5pp ↑ / 9 small win;0 tie / 0 loss)

| SLO | k=0.5 | k=1.0 | k=1.4 | k=1.8 | k=2.2 | k=2.6 | k=3.0 |
|---|---|---|---|---|---|---|---|
| s1 200/120 | +9.9pp | +23.8 | +26.7 | +28.3 | +30.5 | +30.9 | **+34.8** |
| s2 300/150 | +2.8 | +7.2 | +3.0 | +7.0 | +6.4 | +5.3 | +6.3 |
| s3 500/200 | +4.0 | +2.3 | +3.3 | +2.6 | +2.2 | +2.5 | +1.9 |

**code 9 decisive / 11 tied / 1 noise loss**(strict 大胜,mid/loose 撞 saturation ceiling)

| SLO | k=0.7 | k=1.4 | k=2.1 | k=2.8 | k=3.5 | k=4.2 | k=4.9 |
|---|---|---|---|---|---|---|---|
| s1 500/200 | +12.3pp | +27.9 | **+71.5** | **+77.8** | **+77.2** | **+77.4** | **+73.9** |
| s2 500/700 | +2.5 | +0.6 | +2.1 | 0.0 | +0.6 | +0.6 | +0.9 |
| s3 2000/400 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0(都 100%) |

**Goodput**(m31-fix vs c3-fair Δ tok/s):conv s1 k=3.0 时 +1870;code s1 k=4.9 时 +351。

**Latency 全维度 m31 胜**(winner count tally,21 cells × 6 metric):
- conv:m31 ttft_mean 20/21,ttft_p99 17/21,tpot_p99 20/21;c3 拿 0;c1 只 ttft_p99 偶 2 cell
- code:m31 ttft/tpot mean/p99 **4 维全 21/21**(c3/c1 在任一 latency metric 上 0 wins)

### Phase 1 结果 — Thesis 修正

**原 D-012 framing**:"deliberately trades tail latency for mean latency to maximize goodput"
**Phase 1 数据不支持**:m31 不仅 mean 比 c3 低,**tail (p99) 也全程低**(code tpot_p99 c3=228ms 稳定 vs m31=190ms;ttft_p99 c3=900 vs m31=720)。**mean 和 tail 同时改善,不是 trade-off**。

**修正后的 framing(待 paper 落定):**
> Phase-pure temporal multiplexing 把 mixed-batch 的"持续低幅 prefill-decode 干扰"替换为"集中高效的 phase-specific batch"。一个 cycle (prefill_iter + decode_iter) 时间(实测 189ms)**短于** c3 单 mixed iter 时间(实测 ~225ms),差距来自消除 mixed-attention overhead(varlen attention with mixed query lengths 的 kernel 路径 overhead)。
>
> mean 和 tail 同步改善的根因:cycle 时间分母变小了,不是"trade-off"。

**Caveat(paper 须讲清):**
- 此结论限定 Azure burst trace + Qwen3-8B + 2-NPU + chunk_budget=2048
- 极重 prefill demand regime 下(如长 prompt + 高频 burst),m31 cycle 退化为"多 prefill iter + 1 decode iter",cycle 时间变长,**trade-off 此时会重新暴露** — 当前 burst regime 没触发(实测 phase run length 都 ≈ 1 iter)

---

### Phase 1 结果 — Telemetry 机制 verify(code k2.8 s1 cell)

**m31 实际 phase 行为**(window 5-90s,898 iters):
- prefill iter:n=449,mean 150ms,batch_num_tokens 永远 2048,batch_num_reqs mean 2.16 max 6
- decode iter:n=449,mean 39ms,batch_num_reqs mean 29 max 40
- **严格 1:1 alternation**(每 phase run = 1 iter,burst 满负载下)
- decode-iter inter-arrival p50/p99 = 189/203 ms —— 跟实测 tpot p99=190ms 一致

**关键 cycle 时间对比(burst 满负载)**:
| 系统 | 单 iter / cycle 时间 |
|---|---|
| c3 单 mixed iter | ~225 ms(实测 tpot p99) |
| m31 prefill iter + decode iter cycle | 150 + 39 = **189 ms** |

差 36ms / 16% → 直接解释 tpot 优势。

**8 个 cell verify**(conv 全 k × s1 + code 全 k × s1 + 2 loose SLO cell):
- 严格 1:1 仅出现在 code k2.8 s1(burst 满,waiting_depth=64 持续堆积)
- 绝大部分 cell:1 prefill iter 后跟 84-422 个 decode iter(轻 / 中负载)
- 实际 ratio 由 **"有没有 prefill 待消化"** 驱动,不由 PID target 驱动

---

### Phase 1 结果 — PID 不是优势来源

- PID 把 target_ratio 推到 ratio_max=0.8 撞顶,但**实际 prefill iter 占比 12-50%** 由 waiting queue 状态决定
- 8 cell 中 7 cell 都见 target_ratio 0.80 max,实际 ratio 范围 12% (conv k0.5 轻负载) 到 50% (code k2.8 burst 满)
- ReLU 单向上推让 ratio 容易上不容易下,与 D-011 "PID 退出 paper contribution" 一致
- **paper 报 m31 = static ratio per workload,PID 不进 contribution**(D-011 + D-012 立场不变)

---

### Phase 1 结果 — 配置 fairness 确认

| config | server max_num_batched_tokens | 内部 chunk 限制 | 单 iter 最大 prefill |
|---|---|---|---|
| c1 | 8192 | 无 | 8192(unified,no chunking;> 8192 prompt reject) |
| c3 | **2048** | 无(chunk = budget) | 2048(chunked prefill,prefill chunk + decode 共享) |
| m31 | 8192(死代码) | **chunk_budget=2048** | 2048(per iter,多 reqs 共享) |

- **c3 与 m31 chunk size 完全公平(都 2048)**
- m31 的 server budget=8192 实际**完全无效**(prefill iter 被 chunk_budget=2048 cap,decode iter 实际 ~29 token 远小于任一 cap)
- c1=8192 是 unified 唯一合理配置;改 2048 会 reject 长 prompt(c1 没 chunked prefill 路径);"unified + chunked" design point 已被 c3 cover,**c1 无需 fairness ablation**
- 三家配置代表三个 paradigm 的标准设计选择,non-controlled-variable 不是 unfairness

---

### Phase 1 结果 — c3 vs c1 baseline 复现(部分)

- **code 上 c3 大幅胜 c1**:meet% Δ +14~+75pp;tpot_p99 Δ -195~-2506ms(c1 在 code 上 tpot 完全爆炸 2000-2700ms)— 符合预期
- **conv s1/s2 低 k:c3 反而输 c1**:conv s1 c3 meet% 比 c1 -4~-19pp,ttft_p99 c3 比 c1 高 +20~+60ms
  - 物理原因:c3 chunked prefill 把短 prompt 也分 chunk,首 token 延迟比 c1 unified 高 ~40ms
  - 这是 chunked prefill 在短 prompt 高频负载下的弱点
- paper 须讲清此 caveat;c3 vs c1 不是普适成立

---

### 发现的 code bug(Phase 1 副产物,影响有限)

**phase_iters 语义 bug**(`engine.py:30-40`):
- `apply()` 提前更新 `self._phase = candidate`,导致 `reconcile()` 比较失效
- `phase_iters` 实际记录"controller decision 被 parent obey 的连续 iter 数",而非"当前 phase 持续 iter 数"
- 后果:`HardConstraints.enforce()` 用 buggy phase_iters → `constraint_min_slice` 永远不触发(0%),`constraint_max_slice` 误触发 46%(但因 controller planned 通常本来就想切,无害)
- **实际 phase 切换 100% 由 controller selector + parent_auto_flip 驱动,HardConstraints guard 实际不起作用**
- 不影响 Phase 1 数据(guard 本来就 effectively 不存在);修复后预计行为不变,但代码正确性需要修

---

### 跟进事项

1. **修 phase_iters 语义 bug**(代码正确性,可能不影响数据,sanity verify 一下)
2. **m31@chunk_budget ∈ {2048, 4096, 8192} ablation sweep**:扫 chunk size sweet spot;预计 chunk 越大,burst 期 ttft 越低但 tail 越崩 —— 这能直接量化"trade tail for mean" trade-off 的边界
3. **rewrite `T6_FINDINGS.md` / `LONG_PROMPT_SMOKE.md`**:revised conclusions 用 fix 数据
4. **跨 workload 机制 verify**(conv 高 k 段 telemetry):确认 phase-pure 机制不只 code 特例
5. **决定是否进 Phase 2**(Qwen3-4B 跨模型 / BurstGPT trace / mixed workload / c4_pd)

---

- **影响范围 / 不能偏离的原则更新:**
  - 不动 D-011/D-012 框架(thesis 重新成立,但"trade tail for mean" framing 需修正为"phase-pure batching 让 cycle 时间变短,mean/tail 同步改善")
  - 新加 1 条:**任何 cite m31 数据前必须确认是 fix 版**(检查 outdir 是 `m31fix_validate/` 还是旧 `phase_2_t6_burst_goodput/*_pid/`)
  - 新加 1 条:**c3 vs c1 不普适胜出**,paper Section 5 须 explicit 讲 conv strict SLO 上 c3 输 c1 的反例

---

## D-012(2026-05-20)thesis 三层结构化 + goal-first framing + 双 SLO goodput + 三层实验 hierarchy

- **触发:** 5/20 session 完整对比 `past_meeting/thesis_framing.md`(用户 5/19 写,L1/L2/L3 三层结构 + doc framing)跟 memory D-011 / 本 session 累积的方法学讨论(物理 framing + TPOT vs TBT + SLO 方法学),逐个 reconcile 三个 conflict:
  1. **Conflict 1(主指标)**: 单 TPOT_p99 vs 双 SLO goodput。**用 azure_p15 已有 4-way raw p99 数据 force decision**:conv w1@1860 上 M3.1 vs C3 在 TPOT_p99 几乎打平(177 vs 175),但 TTFT_mean M3.1 比 C3 好 86ms;code w1@570 上 M3.1 TPOT_p99=1735ms 比 C3=689ms 差 2.5×。**单 p99 指标会丢失 mean 改善信号**(conv 看不到我们赢)。
  2. **Conflict 2(M3.1 静态化 + SLO 方法)**: PID 运行时耦合 vs static ratio + relative-to-ideal SLO。**用 memory FINDINGS line 144/286 已有数据 force decision**:P1.6e 实测 PID 跨 4 档 SLO 都钉 ratio_max=0.8,driver warmup=20s 已过滤 5-10s 启动爬升期。**现有 M3.1 数据 = effectively static ratio=0.8 with profile**。
  3. **Conflict 3(mechanism proof 方法)**: synthetic controlled experiment vs post-hoc telemetry decomposition。**用户提议三层结合**:B (post-hoc on existing telemetry) 出 defining figure 数据版 → C (trace-sampled synthetic + Poisson QPS sweep) 出 winning region heatmap → L3 (real trace replay) 出 generalization Pareto。
- **关键 narrative 升级(本次新增,跟 D-011 兼容但更深):**
  - **Goal-first thesis statement(替代 D-011 mechanism-first 措辞):**
    > "PD-TDM **deliberately trades tail latency for mean latency** to maximize goodput on accelerators without spatial partitioning. This trade-off is **net-positive on TTFT-sensitive workloads** (long-prompt, high-concurrency) and **net-negative on TPOT-sensitive workloads** (long-output), which we explicitly cede. Double-SLO goodput surfaces this designed trade-off as the primary metric."
  - **三层结构(对齐 thesis_framing.md L1/L2/L3,但修正 L3 含义):**
    - **L1 Interference Shifting Framework** = paradigm 分析层(为什么不同 paradigm SLO 表现不同) — analytical foundation,**不是独立 contribution,是 PD-TDM 的设计合理性来源**
    - **L2 Phase-Pure Temporal Multiplexing** = mechanism 层(怎么实现 mean→tail 转移) — token bucket + static ratio
    - **L3 Per-Workload Static Ratio** = ratio 选择层(怎么选 ratio) — **修正自 thesis_framing.md 的 "SLO-Aware Ratio Selection"**,因为 L3 narrative 强度待 H2 verify
  - **主指标 = TTFT P99 + TPOT P99 双 SLO goodput**(DistServe / Semi-PD 同款,不是单 TPOT_p99 也不是 TBT):
    - 不选 TBT:MuxWise/Sarathi 用 TBT 因为他们卖点是消除瞬时 stall,我们是 paradigm-level 转移,TBT 会把 NPU jitter 跟 paradigm 差异混在一起 under-state 我们 contribution
    - 单 p99 会丢失 mean 改善信号(conv 上 M3.1 双 p99 跟 C3 持平但 mean 大胜)
    - 双 SLO 形式:max sustainable QPS at attainment ≥ 90% under (TTFT < T_ttft) AND (TPOT < T_tpot)
  - **SLO 方法学 = Sarathi 风格 relative-to-ideal × 5×/25×,absolute + relative 双标注**:
    - 拒绝 doc 的 absolute grid {200,500,1000,1500} × {50,100,200,300}(因为是 GPU 经验值,跨硬件不适配,容易出现"全输"或"全 ceiling"的两边坑)
    - 也拒绝早期 reference profile 用 trace-based 低 qps 测 ideal(实证 conv qps=3.18 / code qps=1.05 有 10× workload bias,code tpot_p99=1791ms 完全污染)
    - 用 micro-benchmark(concurrent=1,Semi-PD 同款)per (model, workload) 测 ideal_ttft / ideal_tpot
    - 4-tier:tight (5×) / mid-1 (10×) / mid-2 (15×) / loose (25×) × ideal,absolute ms + "× ideal" footnote 双标注
  - **M3.1 = static ratio per workload(B2 路径)**:
    - PID 代码保留 internal(不删,跟 D-011 一致)
    - Paper 报为 "M3.1 = phase-pure + static ratio,per-workload profiled via one-time scan"
    - 现有 v8/v8b/azure_p15 数据全部复用(等价于 static ratio=0.8 + 5-10s warmup,driver warmup=20s 已过滤)
    - 跟 D-011 "SLO-aware fast-loop selector" thesis 的关系:**fast-loop 重新定位为 paper 不主报机制**,L3 主报 per-workload static ratio profiling 协议(配置层面 SLO 自适应,DistServe goodput formulation 同粒度)
  - **三层实验 hierarchy(冲突 3 resolution)**:
    - **L1: B Post-hoc**(0 wall)— 从 azure_p15 telemetry 后处理出 mean/tail interference,填 defining figure 数据版 → S5.3
    - **L2: C Trace-sampled synthetic**(12-18h)— Poisson QPS + 从 trace 经验分布采样 prompt/output → 受控 QPS 变量 + 真实 prompt 长度分布,出 winning region heatmap → S5.4
    - **L3: Real trace replay**(24-36h)— Azure trace scaled replay + bursty arrival,出 goodput Pareto frontier → S5.2 主图
    - **三层互补**:L1 证 framework holds(干扰坐标);L2 证 mechanism 可预测(winning region 边界);L3 证 generalization(real-world)
- **跟 D-011 的精确关系:**
  - **D-011 active 不变的部分:**
    - PD-TDM paradigm 在 "lacking SM partitioning support" scope 内
    - 4-way comparison(C1/C3/M3.1/c4_pd)
    - MuxWise scope clause complementary positioning
    - 慢回路 PID 不进 paper,starvation override 不进 paper,graph-aware F only
    - vllm-ascend v0.11.0rc1 lock,kernel 不作主图
    - Framing C scope(conceptual = lacking SM partition,evaluation = Ascend 910B3)
    - 4-paper comparison(Sarathi / DistServe / Semi-PD / MuxWise)定位
  - **D-011 被修正的部分:**
    - thesis 措辞:"PD-TDM paradigm + SLO-aware fast-loop selector" → "PD-TDM = L1+L2+L3 三层集成 + goal-first framing"(narrative 更系统化)
    - SLO meet% 不再作为 secondary(D-011 写"secondary = SLO meet% at fixed QPS")→ secondary 改为 SLO attainment curve,主报锁双 SLO goodput
    - "fast-loop selector 是 mechanism-level 加成"(D-011 措辞)→ "fast-loop 探索过 PID 钉 max ratio,paper 报为 static ratio per workload"(C5 honest finding)
  - **D-011 新增的部分:**
    - 三层结构 narrative(L1/L2/L3)
    - 双 SLO goodput 主指标 + Sarathi 风格 SLO 方法学
    - 三层实验 hierarchy(B → C → L3)
    - Goal-first framing(替代 mechanism-first)
    - 干扰转移操作分类(T_couple / T_chunk / T_temporal / T_spatial / T_disagg)
- **本次数据回顾(D-012 决策依据):**
  - **azure_p15 4-way raw p99**(本 session 首次系统看):
    - conv w1@1860:M3.1 ttft_mean=220ms / ttft_p99=825ms / tpot_p99=177ms vs C3 ttft_mean=306 / ttft_p99=788 / tpot_p99=175 — **conv 上 M3.1 TTFT_mean 大幅赢,TPOT 持平,无 visible trade-off**
    - code w1@570:M3.1 ttft_mean=391ms / tpot_p99=1735ms vs C3 ttft_mean=676 / tpot_p99=689 — **code 上 M3.1 TTFT_mean 赢但 TPOT_p99 输 2.5×,visible trade-off**
  - **v8/v8b matrix complete**(7 offset × 3 seed,conv only,c1 vs c2_m31_2048):
    - off1860 (HIGH peak, long-prompt): M3.1 稳赢,确认 winning regime
    - off2160 (HIGH-short): M3.1 输,strict/mid SLO 都一致 → paradigm-level 真反例
  - **static_scan_3a**(M1+chunk ratio sweep,可作 H2 第一手数据)
- **新的 open question(D-012 待 verify):**
  - **H2: PID 钉 0.8 是 workload-adaptive(H1)还是 mechanism-saturated(H2)**? — phase-pure + 单向 urgency(FINDINGS line 157)可能 force ratio 永远撞 ceiling。**影响 L3 narrative 强度**:H1 → "per-workload ratio" 完整 narrative;H2 → "ratio = ratio_max universal,phase-pure 单向有益" 简化 narrative。无论哪种结果都不破坏 thesis,只影响 paper L3 措辞。
  - **C3 在 NPU 反常根因**:azure_p15 显示 conv 上 C3 ttft_p99 比 M3.1 还差(对应 reviewer attack #6)— 需要从 c3_cp telemetry 找根因(chunk_tokens=2048 太大?FIA kernel issue?vllm-ascend 调度问题?)
- **实验工作分 4 phase(详见 `experiment_plan.md`):**
  - **Phase 0(1-2 天):** T3 micro-benchmark ideal(15 min)+ T4 H2 diagnostic(2h post-hoc → 可能 1-2 天 rerun)+ T2 post-hoc B(1 天)+ T1 本决策记录
  - **Phase 1(3-4 天):** T5 trace-sampled synthetic L2 实验,192 runs 12-18h wall
  - **Phase 2(3-5 天):** T6 4-way Azure trace QPS sweep,384 runs 24-36h wall(可与 Phase 1 并行)
  - **Phase 3(3-5 天):** T7 C3 根因 + T8 c4_pd 补跑 + BurstGPT + Mixed workload
  - **Paper writing:** Phase 4 并行起 W2,实验数据 ready 后写 S5-S6
- **影响文档:**
  - `DECISIONS.md`:本条 D-012
  - `README.md`:当前状态节更新到 2026-05-20,最近决策加 D-012
  - `EXPERIMENTS.md`:加 azure_p15 4-way raw p99 数据点入主表,加 4 phase 实验
  - `FINDINGS.md`:加 conv/code workload TPOT_p99 asymmetric trade-off finding,加 H2 待 verify 条目,加 C3 NPU 反常待根因
  - `PROJECT.md`:主原则更新(SLO 方法学 + 双 SLO 指标 + 三层结构)
  - `design/paper.md`:S5 evaluation 主图改双 SLO Pareto,S6 加 regime analysis,S7 加 4-paper 精确定位
  - `past_meeting/thesis_framing.md`:doc 整体 align 本决策(L3 narrative 等 H2 verify 后再细化)
- **影响代码:**
  - 暂时无需新代码改动(B2 路径数据复用)
  - 需要写新 driver(micro-benchmark concurrent=1)
  - 需要写新 trace-sampler(L2 实验用,从 trace 经验分布采样 prompt/output)
  - 后续如果 H2 verify 是 H1 → 需要写 code workload static ratio scan
- **风险:**
  - **H2 verify 结果可能简化 L3 narrative**:如果是 H2,L3 contribution 措辞要从 "per-workload ratio mapping" 收缩为 "static ratio = ratio_max universal recommendation"。Mitigation:H2 不破坏 thesis,只影响 contribution claim 强度,提前预案准备两版措辞
  - **三层实验 hierarchy 整体 wall time 大**:Phase 0+1+2+3 ≈ 2-3 周纯实验。Mitigation:Phase 1 + Phase 2 可并行(不同 NPU 节点),paper writing W2 起并行
  - **azure_p15 数据是 PID-based M3.1**:严格说不是 static-ratio M3.1。Mitigation:用 P1.6e finding 论证 PID 在数据里实际等价于 static@0.8,driver warmup 已过滤启动期。如果 reviewer 攻 → 提供 H2 verify 结果作 evidence
  - **Code workload 上 paradigm-level loss 不可避免**:azure_p15 实证 M3.1 TPOT_p99 = 1735ms 远输 C3 = 689ms。Mitigation:goal-first framing 把这个 trade-off 写成 deliberate design choice 而非 implementation bug,S6.2 explicit cede 给 C3
- **状态:** active
- **Supersedes:**
  - D-011 部分内容:thesis 措辞("fast-loop selector"→"三层结构 + static ratio"),secondary metric 定义("SLO meet% at fixed QPS"→"SLO attainment curve"),实验清单(原 5 个 W1 task → 三层 hierarchy + 4 phase)
  - D-011 主体不变:scope / 4-way / MuxWise positioning / vllm-ascend lock / 4-paper comparison
- **替代关系:** 本决策不取代 D-011,而是在 D-011 框架内 refine narrative + metric + experiment structure

---

## D-011(2026-05-18)thesis 完整重定位:PD-TDM paradigm + SLO-aware fast-loop selector + Framing C scope

- **触发:** P1.9d + P1.9e + 完整 reviewer attack 分析三件事一起 force 重新定位 thesis。
  1. **P1.9d 4-way ablation 实证:** A 无 SLO 自适应 / B 只慢回路 PID / C 只快回路 / D 慢+快(M3.1 完整)。saturated 上 D-A ≈ 0 ~ +3.75pp(noise 内),**B-A ≈ 0(慢回路 saturated 上无贡献)**,**双维度协同 D ≈ max(B, C) 不存在**。conv 上 C-A = +1-2.5pp(快回路主贡献),code 上 C-A ≈ 0
  2. **P1.9e workload transition 实证:** M3.1 vs M1@r08 在 conv2code transition 上 -2.68pp,在 code2conv 上 -0.21pp(noise)。**慢回路 PID 在 transition 上也无 +Δ,saturated regime 唯一可能有 +Δ 的场景被实证证伪**
  3. **Reviewer attack 系统打磨**(用户 5/17-18 多轮讨论):
     - A 类 (novelty):"PD-TDM 是 Sarathi-Serve + DistServe 的 interpolation" / "Mechanism well-studied" → 需要新 angle defense
     - B 类 (scope/significance):"single-node scope 太窄" / "+1-2.5pp 太边际" → 当前 SLO meet% metric 跟 DistServe goodput metric 不可比较,被 attack 量级小
     - D 类 (specific finding):MuxWise (ASPLOS'26) 已经做了 GPU 上的 PD multiplexing,但**明确排除 NPU 平台**("supports intra-process spatial sharing")—— 这反而是 explicit gap
     - 用户指出 1P:1D disagg 不是 fair baseline(资源受限被迫配置),不能 anchor "disagg 反优势" 论点
     - 用户指出 thesis 原意是 goodput at SLO 提升,P1.x 只测 SLO meet% 是 methodological 缺口
- **旧 thesis(D-002 / D-009 / D-010 联合状态):**
  - 「分段控制器 + 双维度协同」:慢回路 PID 启动爬升 + 边界保护;快回路 selector iter 粒度双向预警
  - 论文 framing 隐含 "M3.1 stack vs C3 stack" paradigm bundle 对比(D-010),"M3.1 是另一个 regime 的方案"(D-009)
  - 主 metric = SLO meet% at fixed QPS(saturated regime)
- **新 thesis(D-011):**
  - **核心 claim:** PD-TDM 是 single-node multi-device accelerators(lacking SM partitioning support)上的 temporal P/D multiplexing paradigm,提供 goodput-centric improvement under SLO constraints。SLO-aware fast-loop selector 在 conv tight-TTFT regime 提供 mechanism-level 加成
  - **删除 claim:**
    - 慢回路 PID 不再作为 contribution(P1.9d/e 全证伪;代码保留 internal,paper 不提)
    - 双维度协同不再作为 thesis 核心(P1.9d D ≈ max(B,C) 实证证伪)
    - Starvation tpot override 删除(P1.7c 触发率 0.5-0.8% + ±1pp noise,无效;代码保留,paper 不提)
    - Graph-aware 模块 D-005 三选一定为 F only(不进 contribution,只在 Implementation 节一句 fairness setup)
  - **保留 claim:**
    - phase-pure paradigm 物理 fingerprint(iter trace decode silence p99 800-940ms,99.7% prefill 时 decode queue 非空)
    - paradigm-level Δ 双向 trade-off(conv tight TTFT M3.1 +7.58pp / conv relaxed C3 dominate / code 两者都瘫)
    - SLO-aware fast-loop selector(token bucket + urgency_ttft per-request sorting)= mechanism contribution
    - target_ratio = static config 设计选择(不动态调,paper Section 5.6 sweep + recommended config)
  - **新引入的 conceptual contribution:**
    - **Interference shifting framework** 作为 paradigm choice 的 unified lens:Chunked Prefill = mean-domain interference,PD-TDM = tail-domain interference,Spatial multiplexing (MuxWise) = avoid interference (requires SM partition),Disaggregation = eliminate interference (requires cheap KV transfer)
    - **Goodput-centric metric framework:** Primary metric = max sustainable QPS at SLO target ≥ X%(DistServe-compatible);Secondary metric = SLO meet% at fixed QPS(saturated quality view)。Pareto frontier 作为主图呈现
- **Paper scope strategy(Framing C hybrid):**
  - **Conceptual scope:** "Single-node multi-device accelerators **lacking SM partitioning support**"(包括 NPU / older GPUs / 未来 ASIC)。这个 scope 是 MuxWise paper "supports intra-process spatial sharing" scope 的 explicit complement set
  - **Evaluation scope:** Ascend 910B3 NPU 作为 representative platform(2 张 NPU,TP=2)
  - **Title:** 不锁 NPU,Section 1 scope statement 明确(待写作期定 title 具体措辞)
- **完整 paper structure(9 sections):**
  - S1 Intro / S2 Background+Motivation(完整 4-paradigm landscape) / S3 Design(PD-TDM + fast-loop) / S4 Implementation(state probe 三件套 + fairness setup) / S5 Evaluation(Pareto frontier + 4-way + ablation + cross-model + physical evidence) / S6 Discussion(interference shifting unified lens) / S7 Related Work(cite MuxWise + Semi-PD + Sarathi + DistServe) / S8 Limitations(single platform / single family / no online switching) / S9 Conclusion
- **实验需要补的(W1 启动):**
  - **QPS sweep**(8 QPS × 4 configs × 2 traces × 3 seeds):goodput-centric metric 数据,1-2 天 wall
  - **跨模型 sweep**(Qwen3-4B + Qwen3-8B 已确定,不加 Llama / Llama-3.2 / MoE):2-3 天 wall
  - **c4_pd 补跑**(Azure trace + Qwen3-8B,跟主 evaluation 框架对齐):1-2 天
  - **BurstGPT trace 加入**:1-2 天
  - **Mixed workload(conv + code 同时到达)**:1-2 天,decide 是否扩 winning regime 到第 2 个
- **Defense framework summary(完整 reviewer attack defense):**
  | Attack 类 | Defense anchor |
  |---|---|
  | A1 PD-TDM 是 interpolation | Interference shifting reframe + V 字形 trade-off 主图(非 monotone)+ mixed workload 数据 |
  | A2 Mechanism well-studied | Writing 上承认 primitive 不新,定位为 paradigm realization 一部分 |
  | B1 single-node scope 窄 | Resource utilization industry consensus + 撤回 disagg 反优势 anchor + 1P:1D 降级 reference point |
  | B2 +1-2.5pp 太边际 | Goodput-centric metric framework + Pareto frontier + paradigm Δ +7.58pp 作 headline |
  | C1 没跑 Sarathi official | vLLM-CP 是 Sarathi paradigm 的 open-source impl,vllm-ascend 已是 production stack |
  | C2 单一 hardware | Framing C scope generic("lacking SM partitioning")+ paradigm physical independence argument + cross-model 部分弥补 |
  | D1 disagg 反优势 Ascend-specific | B1 修订后已 defused,1P:1D 是 budget reference,不 claim 普遍反优势 |
  | D2 phase-pure trade-off 已 cite | **MuxWise (ASPLOS'26) explicit 排除 NPU,我们填 gap;Framing C scope 是 MuxWise scope 的 complement set;Interference shifting 是新 unified lens** |
  | E1 target_ratio hyperparameter | Paper 主动 disclaim,Section 5.6 sensitivity sweep |
  | E2 复现性 | 全 stack 开源(vllm-ascend Apache 2.0 + TDM 插件)+ Ascend 商业硬件 + 公开 trace |
- **影响文档:**
  - `PROJECT.md`:完整重写(本次 W1 第 2 步)
  - `EXPERIMENTS.md`:加 P1.9d / P1.9e 入主表,加 W1 待跑实验
  - `FINDINGS.md`:加 P1.9d/e 实证条目,interference shifting framework 作为 unified lens 记录
  - `design/paper.md`:完整重写 paper outline(W2 起做)
  - `design/slo_pid.md`:慢回路 PID 角色 final 收尾(代码留 internal,paper 不写)
  - `design/sampling.md`:Sampling 模块降级为 evaluation methodology(SLO grid 校准),不进 paper contribution
- **影响代码:**
  - 慢回路 PID(`SLOReactiveController`)代码保留 internal use,paper 不写
  - Starvation tpot override(`starvation_decouple_bucket` flag)代码保留 internal,paper 不写
  - Mixed_mode 开关(interfaces.md 缺口 #5)**不再做** —— 之前认为需要它做 phase-pure 单变量 ablation,现在 paradigm-level Δ 不归因到单机制(D-010 立场延续),不需要这个 ablation
  - Graph-aware 模块代码保持现状(D-005 已 settled F only)
- **风险:**
  - **QPS sweep 数据可能不利:** lower QPS 上 paradigm 差异可能消失或反向。如果 PD-TDM 在所有 SLO target 下都不能 sustain 比 C3 更高 QPS,B2 defense 弱化。Mitigation: 多个 SLO target reporting(60/70/80/90),Pareto frontier 让 reviewer 看完整 picture
  - **Mixed workload 可能不赢:** 如果 conv+code 混合到达上 PD-TDM 不能赢 C3,winning regime 仅限 conv tight-TTFT(narrow scope)
  - **Framing C scope claim 部分 hypothetical:** "older GPUs / future ASIC" 在 lacking SM partition class 内但我们没实测验证。Mitigation: Limitations 节诚实承认 + scope generalization 作为 conceptual/future-work
  - **MuxWise scope clause 解读:** 我们 explicit cite MuxWise 排除 NPU 的句子作为 motivation。如果 MuxWise reviewer 不同意我们解读,可能 attack。Mitigation: cite 原句,framework 镜像 MuxWise scope clause,leave interpretation to readers
  - **Top venue(EuroSys / SoCC)中签率 15-25%**(reviewer 仍可能觉得 contribution incremental + single platform)。Mitigation: 主投 B 中下(IPDPS / DSN / Middleware)+ fallback CCF-C(IISWC / ICPADS / HPCC)
- **状态:** active
- **Supersedes:**
  - D-002(分段控制器 + 双维度协同):双维度协同部分 superseded;分段控制器收缩为 fast-loop only
  - D-005(Graph 模块三选一 pending):final 定为 F only
  - PROJECT.md L17-21 不能偏离 §1(thesis 旧表述):被 D-011 新 thesis 取代

---

## D-010(2026-05-15)phase Δ 归因层级修正 + 双维度 ablation 结构

- **触发:** 用户 5/15 讨论指出 D-009 表述有两个方法论漏洞:
  1. 「phase-pure 贡献 86-101%」是把 paradigm-level 差异强行归因到 phase-pure 单变量。M3.1 vs C3 整套代码 stack 都不同(TDMScheduler / vLLM CP scheduler、chunk_tokens、max_num_batched_tokens、graph 配置),phase Δ 实际上是 bundle vs bundle 总差,内部机制不可单变量归因
  2. SLO 自适应本来就是双维度设计(D-002):慢回路 PID(tick 粒度,启动爬升 + 边界保护)+ 快回路 selector(iter 粒度,双向预警)。「PID 钉 ratio_max 因此 PID 没用」这种判断只看了慢回路一条腿,真正的 SLO 自适应力量在快回路(P1.7b 未实装)
- **旧:**
  - D-009 narrative:「M3.1 vs C3 差距 86-101% 来自 phase-pure 调度」(单机制归因)
  - 论文 ablation 隐含拿 C3 当 same-stack 对照
- **新:**
  - phase Δ 表述软化为 「TDM-paradigm vs CP-paradigm bundle 总差」,**不再分配到具体机制**。论文写法上明列两套 bundle 内容,不试图 isolate phase-pure
  - mechanism-level claim 改由**同 stack ablation**支撑:
    | claim | 同 stack 对照 | 状态 |
    |---|---|---|
    | phase-pure(P/D 时分复用)有用 | TDM mixed_mode on/off | **需要加 mixed_mode 开关** |
    | 慢回路 PID 真正角色 | M1 静态 ratio 扫到 ratio_max vs M3.1 | 不依赖新代码,可立即跑 |
    | 快回路双向预警有用 | M3.1(慢) vs M3.1+P1.7b(慢+快) | 依赖 P1.7b 实装 |
    | 双维度真协同 | 只快 vs 慢+快 | 同上 |
  - C3 在新结构里定位为 **外部 reference point**(end-to-end SOTA 对比),不当 ablation。代码 stack 差异这个 confound 在 reference 对比里是合法的,不需要消除
  - 论文 threats to validity 加两条:bundle 内不可拆 + TDMScheduler 跟 vLLM CP scheduler 实现质量差异 inherent confound
- **影响文档:** `design/paper.md`(§6 贡献声明 + §7 TtV + 新 ablation 矩阵),`design/slo_pid.md`(慢回路角色重定位),`design/interfaces.md`(加 mixed_mode 缺口),`FINDINGS.md`(P1.8 节注解 paradigm-level),`EXPERIMENTS.md`(三条 pending 实验)
- **影响代码:** TDMScheduler 加 `mixed_mode` 开关(中等工程量,几天)。M1 ratio_max 扫描只是配置改动,不动代码
- **launch 参数对齐(2026-05-15 收窄):** 既然接受 paradigm-level Δ 不归因到单机制,launch 参数对齐就不再是 D-010 根基,大部分参数本来就是 paradigm 各自的合理选择。只保留两件 sanity check:
  - c1/c3/m31 是否都开 graph capture(`enforce_eager` 一致)— 执行细节不一致是 bug
  - c3 默认 `chunk_tokens` 是多少 — 写论文时要知道 c3 跑在什么配置上;决定要不要补 c3@chunk=2048 做 sensitivity(不是对齐,是确认 c3 不在非最优点)
  - **不再要求** `max_num_seqs` / `max_num_batched_tokens` 等 paradigm-specific 参数对齐:强制对齐反而违反 paradigm 公平比较原则
- **风险:**
  - M1 ratio_max 扫描如果显示 M1@ratio_max ≈ M3.1,慢回路 PID 自适应贡献被实证弱化 → 接受,论文写"PID 提供启动爬升 + 边界保护"而非"在线 SLO 自适应"。这是诚实表述,不戳穿 thesis(thesis 第二条腿在快回路)
  - thesis 闭环更依赖 P1.7b。P1.7b 拖延 → mechanism-level claim 缺第二维度,paradigm-level reference 也不够
- **状态:** active

---

## D-009(2026-05-14)dedicated kernel 完全移出 contribution,phase-pure 双向化

- **触发:** P1.8 m31_fia ablation(`results/azure_m31_fia_ablation/`)。强制 M3.1 attention 走 FIA 后,所有指标变化 ≤ 13ms 或 < 1%,跟原 m31_2048 统计等价。M3.1 vs C3 总差距 86-101% 来自 phase-pure 调度本身,**kernel 贡献 ≈ 0**。同时 phase-pure 在 saturated 长 prompt 上 TPOT_p99 输 C3 992ms (M3.1 是 C3 的 2.5×),e2e 输 7.2s
- **旧:**
  - PROJECT.md 不能偏离 §5:「kernel 速度不作为论文主图,只解释 C3 反常」(隐含 kernel 仍是优势组件之一)
  - paper narrative 隐含「M3.1 vs C3 = phase-pure + dedicated kernel」两层叠加
- **新:**
  - kernel 完全移出 contribution。论文不再提 dedicated kernel 速度优势,vllm-ascend v0.13+ 删除 dedicated kernels 不威胁我们 narrative
  - phase-pure 显式双向化:**TTFT 紧 + 短 prompt → phase-pure 赢;TPOT 紧 + 长 prompt → mixed 赢**。M3.1 不再宣称单方面优于 C3
  - SLO-adaptive TDM 的卖点收紧为「根据 workload regime 动态选择 phase-pure / mixed」,这正好 motivate 分段控制器(D-002)+ 双向 selector(P1.7b)。论文 narrative 从「M3.1 全面胜出」改为「regime-aware 选择,在区分带 dominate」
- **影响文档:** `PROJECT.md` 不能偏离 §5(改写,kernel 移除)+ §其他论点连带核查,`design/paper.md`(主 narrative 重写 contribution section),`FINDINGS.md` Gap-5 已加,`EXPERIMENTS.md` 加 `azure_m31_fia_ablation/` 入主表
- **影响代码:** 无(`force_fia_attention` 字段 + `attn_patch.py` 保留作为可复现 ablation,默认 False 不影响主路径)
- **风险:** thesis narrative 从「M3.1 是更好的方案」变成「M3.1 是另一个 regime 的方案」,审稿要看 SLO-adaptive 部分是否真的能做出 regime-aware 切换效果 → P1.7b 双向 selector 必须 work,不然 thesis 论点缺第二条腿
- **状态:** active

---

## D-001(2026-05-14)P1.6h calibrated baseline 推迟到 P1.7b 阶段

- **触发:** 用户 5/14 要求 memory 整理 + 决定 P1.6f/g/h/i 脚本未跑就遗弃(见 D-004)。其中 P1.6h 跟其他三个不同:它是用调过的 SLO 档(conv=200ms)重跑全部 baseline 出主结果,跟「主报只在调过的档下报」直接相关
- **旧:** P1.6h 作为独立 sweep(6 calls × 22min)单独跑
- **新:** 脚本删除,实验推迟。P1.7b 阶段实装快回路 selector 时,baseline 顺便用调过的档跑一次,合并产出主结果
- **影响文档:** `EXPERIMENTS.md`(列为 pending,等 P1.7b),`PROJECT.md`「不能偏离 §2」(临时承认 `azure_main` strict 档作为临时主结果),`FINDINGS.md` 未决问题节
- **影响代码:** 无(脚本已随 D-004 删除)
- **风险:** 如果 P1.7b 拖延,主结果用 strict 档时间被拉长,审稿可能质疑。可接受
- **状态:** active

---

## D-002(2026-05-13)thesis 改为「分段控制器 + 双维度协同」

- **触发:** P1.6e 跨 4 档 SLO sweep(`results/azure_p16e/`)conv ratio 钉 max 80%+ 跨所有 SLO 档,`err_tpot_eff` 非零率最高 3.6%(PASS 标准是 >20%)。PID 在边界几乎必然单输入,**不是闭环反馈**
- **旧:** thesis claim = 「SLO-aware 闭环反馈控制 P/D 比例,tpot/ttft 双 SLO 作 runtime 受控变量」
- **新:** thesis claim = 「分段控制器:慢回路 PID 负责启动期把 ratio 爬升到合理值 + 饱和稳态做边界保护;快回路 selector 负责 iter 粒度 TTFT/TPOT 双向预警(P1.7b 待实装)」
- **影响文档:** `PROJECT.md` 不能偏离 §1,`design/slo_pid.md` 修正节,`design/paper.md` narrative,`FINDINGS.md` Gap-1
- **影响代码:** 无(代码层面慢回路 PID 已经是这个行为;快回路待 P1.7b 实装)
- **状态:** active(待 P1.7b 验证补全第二条腿)

---

## D-003(2026-05-13)Sampling 模块定位转向

- **触发:** P1.6e finding:SLO 档校准让 M2.4 屏蔽 99.5% → 5.4%(对反馈链有巨大影响),但 ratio 仍钉 max。SLO 校准**必要但不充分**
- **旧:** Sampling 模块 claim = 「提供 SLO target 让 PID 在双输入工作区工作」
- **新:** Sampling 模块的三个真实价值:
  1. 定义系统可服务边界(code 类工作负载无论 SLO 怎么调违例都极高 → 出 admission control 范畴)
  2. 提供主报结果的目标档(thesis 报这个档下 meet_slo%)
  3. 校准 PID 的「启动期爬升目标」,让爬升收敛到工程意义点而非物理不可达档
- **影响文档:** `design/sampling.md` 修正节,`PROJECT.md` § 三模块的 Sampling 节,`FINDINGS.md` Gap-2
- **影响代码:** 暂无(模块未实装)
- **状态:** active

---

## D-004(2026-05-14)P1.6f/g/i sensitivity sweep 全部遗弃

- **触发:** 用户 5/14 评估认为方向已定(P1.7b 是 thesis 第二条腿),P1.6f(target_violation_rate sweep)/ P1.6g(ratio_max sweep)/ P1.6i(chunk on azure)这类 sensitivity 实验价值降级
- **旧:** 三个 sensitivity sweep 脚本就位,各 ~2-2.4h wall,准备一夜串联跑
- **新:** 全部删除(P1.6f / P1.6g / P1.6h / P1.6i 共 9 文件:4 run sh + 4 posthoc py + 1 execute_guide)。P1.6h 单独推迟(D-001),其他三个永久遗弃
- **影响文档:** `EXPERIMENTS.md` 已废弃实验节,`FINDINGS.md` 已废弃路径节
- **影响代码:** 删除 `experiments/run_p16{f,g,h,i}*.sh`、`experiments/posthoc_p16{f,g,h,i}*.py`、`memory/p16_sweep_execute_guide.md`
- **状态:** active

---

## D-005(2026-05-12)Graph 模块降级为论文 finding

- **触发:** P1.0(FFTS+ baseline)+ P1.0b(AIV 23 sizes vs FFTS+ 15 sizes)。AIV 反而比 FFTS+ 输 -1.5~-23.8pp,**capture 数量非单调**。vllm-ascend 已内置 cudagraph_capture_sizes + decode 自动 pad
- **旧:** Graph 自适应模块是 TDM 三大控制模块之一(SLO + Sampling + Graph)
- **新:** Graph 模块没有算法空间,降级为论文 finding。三模块变两模块(Sampling + SLO 控制器)。论文形态待论文写作前三选一:
  - **G+F**:作为独立章节写入论文
  - **D**:合并到 chunking 节
  - **F only**:只留作 finding 引用,不进 contribution
- **影响文档:** `PROJECT.md` § 三模块,`design/paper.md`,`FINDINGS.md` Gap-4
- **影响代码:** 无(模块本来就没独立实装,只在 chunking 节有 capture sizes 对齐逻辑)
- **状态:** pending(三选一,论文写作前定)

---

## D-006(2026-05-08)chunking 不做 dynamic controller,收敛为静态 chunk=2048

- **触发:** p1_chunk_scan 实测,chunk_tokens ∈ {512, 1024, 2048, 4096, 8192} pareto 单调,chunk=2048 是 sweet spot。没有双向 trade-off,无 dynamic 设计空间
- **旧:** M3.4 设计是 SLOReactiveController 加 chunk_tokens 输出维度(运行时调)
- **新:** chunking 静态 chunk_tokens=2048,不做 dynamic controller。M3 的 thesis 价值收敛为「推进区分带边界」(让区分带从 ttft∈[500,1500] × tpot∈[100,200] 扩到更紧的档)
- **影响文档:** `design/chunking.md`、`design/paper.md`,`FINDINGS.md` Gap-3
- **影响代码:** 无(M3.4 controller hook 未实装,留 `prefill_chunk_tokens` 配置字段静态用)
- **状态:** active

---

## D-007(2026-05-07)锁 vllm-ascend v0.11.0rc1

- **触发:** vllm-project/vllm-ascend [PR #4623](https://github.com/vllm-project/vllm-ascend/pull/4623) 在 v0.13.0 删除 AscendScheduler(我们继承的父类),没有 phase-aware 替代接口。v0.13.0+ 强制走上游 V1 Scheduler + chunked prefill + FIA kernel(等价于 C3)。我们环境受限不能跑并行 v0.18 supplementary
- **旧:** 跟随上游版本升级
- **新:** 所有实验 / 论文锁 v0.11.0rc1。论文 §implementation 主动写明锁定理由,把 narrative 升级为「v0.11.0rc1 是最后一个保有 phase-aware AscendScheduler 的版本,恰好是研究 phase-aware 调度价值的最后窗口」
- **影响文档:** `PROJECT.md` § 版本锁,`design/paper.md` § threats to validity
- **影响代码:** 实验编排都锁这个版本
- **状态:** active(不可逆)

---

## D-008(2026-05-07)长 prompt sweep 双 pivot

- **触发:** P0-3 长 prompt SLO grid sweep(`results/long_prompt_sweep/`)发现:
  - strict 档 (ttft<500 / tpot<50):所有 4 个 config 趴地板 < 5%
  - loose 档 (ttft>2000 / tpot>300):所有 4 个 config 100%
  - 区分带 ttft∈[500, 1500] × tpot∈[100, 200]:M2.7 一致胜出
- **旧:**
  1. strict SLO 档作为 evaluation 主档
  2. M3 chunking 计划做在线 controller 调 chunk_tokens
- **新:**
  1. Sampling 模块从设计文档升格为实装目标(workload-conditional SLO 框架)
  2. M3 chunking 收敛为「推进区分带边界」(进一步发展为 D-006)
- **影响文档:** `design/sampling.md`、`design/chunking.md`、`PROJECT.md` 不能偏离 §2
- **影响代码:** 无
- **状态:** superseded by D-003(Sampling 模块进一步转向)+ D-006(chunking 收敛静态)

---

# 候选 / 暂未拍板

(写在这里的是脑暴中的方向,尚未变成 active 决策。落地前讨论)

- **(无候选)**——P1.7b 双向 selector 设计还在 `design/slo_pid.md` 快回路节讨论,未到决策阶段
