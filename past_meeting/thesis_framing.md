# PD-TDM Thesis Framing (2026-05-19)

> 基于四篇核心论文 (Sarathi-Serve / DistServe / Semi-PD / MuxWise) 的完整定位分析。
> 本文档是 paper writing 的 blueprint，所有 section 内容已细化到可执行级别。

---

## 1. PD 调度技术三代演进

### 1.1 三代技术全景

| 代际 | 代表论文 | 核心思路 | 对 P/D 干扰的处理 | 硬件要求 | NPU 可用? |
|---|---|---|---|---|---|
| **1st: PD 耦合优化** | Sarathi-Serve (OSDI'24) | Chunked prefill，prefill 切块与 decode 交替 | **减少干扰**：tail→mean，无法根除 | 无特殊要求 | ✅ |
| **2nd: PD 分离** | DistServe (OSDI'24) | P/D 分到不同 GPU，从根源消除干扰 | **消除干扰**：物理隔离，代价=KV 传输 | ≥2 GPU | ✅ (c4_pd) |
| **3rd: PD 动态混部 (空分)** | Semi-PD (EuroSys'25) / MuxWise (ASPLOS'26) | 单卡 SM 空间分区动态分配 P/D 计算资源 | **规避干扰**：空分隔离，综合前两代优点 | **SM/MPS 分区** | ❌ |
| **2.5th: PD 时分复用 (我们的位置)** | **PD-TDM (本文)** | 单卡时间维度 P/D 分相执行 | **转移干扰**：mean→tail，唯一不需要空分硬件的 P/D 隔离方案 | 无特殊要求 | ✅ |

### 1.2 MuxWise 的关键 scope clause（我们的 entry point）

MuxWise (ASPLOS'26) 明确声明其方案要求 "supports intra-process spatial sharing" (SM 分区能力)。这意味着：

- **所有 3rd-gen PD 动态混部方案天生排除 Ascend NPU**
- NPU 平台上，PD 调度的可选方案被压缩为：耦合 (C1/C3) vs 时分 (M3.1) vs 分离 (c4_pd, 需要 ≥2 卡)
- **"在无空分能力的加速器上，时分复用是唯一能在单卡内实现 P/D 隔离的机制"** — 这是我们的核心 motivation

### 1.3 PD 调度策略空间 (单卡内)

```
                 P/D 共享 iter               P/D 独占 iter
                 (interference 存在)          (interference 隔离)

空间维度         统一 batch (C1)              SM 分区 (MuxWise, Semi-PD)
                 P+D 同 batch 混跑            P/D 各自占 SM 集合
                                              需要 GPU SM 分区 ← NPU 无此能力

时间维度         Chunked Prefill (C3)         PD-TDM (M3.1, 我们的位置)
                 P 切块与 D 交替              P-iter 和 D-iter 分相执行
                 interference → mean          interference → tail
```

**我们的位置：** 时间维度 × P/D 独占 iter。这个格子目前是空的 — Sarathi/C3 在时间维度做了切分但没有独占，MuxWise 做了独占但是空间的，PDM/Drift 做了时分但缺系统化研究 + SLO 视角 + NPU 平台。

---

## 2. PD-TDM 的三层结构

PD-TDM 不是 "一个 framework + 一个实现"，而是一个**三层集成的完整方案**。三层各自解决一个子问题，合在一起构成 PD-TDM 的完整贡献。

```
Layer 1: Interference Shifting Framework (分析层)
         → 理解问题：为什么不同 PD 调度策略有不同的 SLO 表现？
         → 答案：因为它们把干扰推到了延迟分布的不同位置

Layer 2: Phase-Pure Temporal Multiplexing (机制层)
         → 解决问题：如何在无空分能力的加速器上主动控制干扰的位置？
         → 答案：通过 phase-pure scheduling + ratio knob 实现干扰转移

Layer 3: SLO-Aware Ratio Selection (自适应层)
         → 使用方法：给定 SLO 结构，如何选择干扰位置？
         → 答案：SLO 结构 → 干扰预算 → 最优 ratio 的映射
```

### 2.1 Layer 1: Interference Shifting Framework（分析工具）

**这不是独立的"贡献"——它是为 PD-TDM 提供设计合理性的分析语言。**

P/D 干扰的本质：**prefill 计算占用 decode iter 的时间，导致 decode token 产出被延迟。** 不同调度策略的本质区别，在于它们把这种延迟推到 latency distribution 的哪个位置。

**操作性定义：**

```
Mean-domain interference：干扰均匀分布在大多数 decode token 上
  → 每个 token 都慢一点，CDF 整体右移，shape 不变
  → 伤害 TTFT（prefill 被 decode 挤占）+ TPOT_mean

Tail-domain interference：干扰集中在少数 decode token 上
  → 大部分 token 正常速度，少数 token 极慢（prefill iter 期间 decode silence）
  → 伤害 TPOT_p99，不影响 TPOT_mean 和 TTFT
```

**四种干扰转移操作：**

| 操作 | 范式 | 对干扰分布的影响 | 物理机制 |
|---|---|---|---|
| **T_couple** | C1 Unified | mean↑, tail↑ | P 和 D 同 iter 竞争，每个 token 都被干扰 |
| **T_chunk** | C3 Chunked Prefill | mean↑, tail↓ | P 被切块 → 更多 iter 有少量干扰，但单 iter 对 decode 的阻塞时间缩短 |
| **T_temporal** | M3.1 PD-TDM | mean↓, tail↑ | P 和 D 分相 → 纯 decode iter 零干扰，但 prefill iter 期间 decode 完全沉默 |
| **T_spatial** | MuxWise (NPU 不可达) | mean≈0, tail≈0 | SM 隔离 → 两个 phase 各占独立计算单元，无干扰 |
| **T_disagg** | DistServe / c4_pd | mean≈0, tail≈0 | 物理分卡 → 零干扰 |

### 2.2 When Does Temporal Win? — The Physics of Interference Under Workload Structure

本节回答审稿人最可能问的问题：*"Why should temporal multiplexing outperform chunked prefill — and when?"*

**核心物理链条：**

长 prompt + 高并发 → P/D 竞争白热化 → mean interference 急剧恶化 → temporal 的 mean→tail 转移收益最大。

```
长 prompt (数百到数千 token)
  → 单次 prefill 时间长（数百 ms 到 s 级）
  → 在耦合调度 (C1/C3) 中，一次 prefill 长时间阻塞整个 batch 的 decode
  → TTFT 因为 prefill queue 堆积而恶化（所有请求等 prefill slot）
  → C3 的 chunking 把一次长 prefill 切分成多次短 prefill：
    正面：单次阻塞时间变短 → tail interference↓ → TPOT_p99↓
    代价：更多 iter 有 prefill 混入 → mean interference↑ → TTFT↑
  → C3 的 trade-off 方向恰好是：牺牲 TTFT 改善 TPOT

高并发 (qps ≥ 8, waiting queue + running queue 同时深)
  → P 和 D 同时竞争 iter 时间 → "谁等谁" 的问题不可回避
  → 耦合方案 (C1/C3) 无法结构化重分配 P vs D 的时间比例
  → 每 iter 的 batch 构成由 scheduler 的贪心逻辑决定，
     非结构化的 P/D 混合导致两者相互拖慢
  → 时分方案 (M3.1) 通过 ratio 显式控制 P:D 时间分配，
     这是一个耦合方案没有的自由度

PD-TDM 在这个场景下的物理行为：
  纯 prefill iter:
    - 长 prompt 的 TTFT 从 "等 decode iter 结束" 中解放
    - 集中跑 prefill → waiting queue 快速清空 → TTFT↓
    - Chunked prefill 在 P iter 内仍然生效 (chunk_tokens=2048)
  
  纯 decode iter:
    - 在 prefill iter 之间有 decode silence (p99 800-940ms)
    - 但高并发下 running queue 有足够 decode token 可产
    - Silence 的边际影响被 diluted: 沉默 800ms 后 D iter 密集产出
    
  Ratio knob:
    - 高 ratio (0.8) → 更多 P iter → TTFT 更低，但 decode silence 更频繁
    - 在长 prompt 场景下，单次 P iter 的 silence 已经很大，
      调高 ratio 增加 silence 频率的边际代价递减
    - 所以最优 ratio 在长 prompt 下偏高 (0.7-0.8)
    - 这是 SLO-aware ratio selection (L3) 的物理依据
```

**反过来——什么时候 temporal 不赢：**

| 条件 | 物理原因 |
|---|---|
| **短 prompt** (prefill 很快完成) | TTFT 本就不是瓶颈，temporal 的 decode silence 代价没有收益对冲 |
| **低并发** (queue 不深) | 不需要结构化重分配 P:D 时间，跑完就完了 |
| **TPOT 极度敏感** (code 类 workload) | decode silence 的 tail 代价 >> TTFT 改善的收益。这类 workload 下 C3 的反向 trade-off (牺牲 TTFT 保 TPOT) 更有利 |
| **QPS 过低** (<< hardware 饱和点) | 所有 config 都能轻松满足 SLO，差异消失 |

**这为三层框架提供了 predictive power：**

- Layer 1 预测：temporal 在 **mean interference 是瓶颈** 时赢
- Layer 2 解释：长 prompt + 高并发 = mean interference 最严重的物理条件
- Layer 3 量化：在此条件下高 ratio (0.7-0.8) 是最优选择

这个预测链条使 PD-TDM 不是一个 post-hoc 调参碰到的 winner，而是 **"给定 workload 结构，可以预测哪个范式更优"** 的 principled framework。这是它与已有工作的本质区别——Sarathi/MuxWise 都没有提供这种 regime prediction 能力。

### 2.2 Layer 2: Phase-Pure Temporal Multiplexing（控制手段）

**Framework 告诉你"干扰可以被转移"，PD-TDM 告诉你"在 NPU 上如何转移"。**

在无 SM 分区能力的加速器上，temporal multiplexing 是唯一能在单卡内实现 P/D 隔离的机制。具体的控制手段：

```
时间轴:
|-- P iter --|-- D iter --|-- D iter --|-- P iter --|-- D iter --|-- D iter --|...

- P iter: 100% prefill，从 waiting queue 取请求
- D iter: 100% decode，只推进 running requests
- 切换决策由 ratio 控制: ratio = P_iters / (P_iters + D_iters)
- Token bucket: 每个 iter 累计 ratio credit，credit ≥ 1 且 queue 非空 → P iter
```

**Ratio 是干扰转移的控制 knob：**
- 高 ratio → 更多 P iter → prefill 更及时 → 低 mean interference → TTFT 友好
- 低 ratio → 更多 D iter → decode 不被打断 → 低 tail interference → TPOT 友好
- Ratio 直接决定系统在干扰空间中的位置

### 2.3 Layer 3: SLO-Aware Ratio Selection（使用方法）

**Ratio 有了，怎么选？这是 PD-TDM 的 SLO 自适应层。**

SLO 自适应在 PD-TDM 里发生在**配置选择层面**（而非在线调参层面）：

```
给定: SLO target (TTFT < T_ttft, TPOT < T_tpot)
      workload profile (conv / code / mixed)
      
Step 1: SLO 结构分析
  - TTFT 紧 + TPOT 松 → mean interference 是瓶颈 → 需要低 mean → 高 ratio
  - TTFT 松 + TPOT 紧 → tail interference 是瓶颈 → 需要低 tail → 低 ratio
  
Step 2: 干扰预算分配
  - 根据 SLO 结构确定可容忍的 mean/tail interference 上限
  
Step 3: Ratio 选择
  - Conv (TTFT-sensitive): 推荐 ratio = 0.8
  - Code (TPOT-sensitive): 推荐 ratio = 0.7
```

**为什么 "static suffices" 不是放弃自适应，而是自适应的正确粒度：**

同一 workload 的 SLO 结构是稳定的 — 不需要 iter 粒度动态调整。Ratio sensitivity sweep (S5) 展示 ratio 对 goodput 的效果是单调可预测的，一次 profiling → 固定配置。这与 DistServe 的 goodput formulation（也是配置层面优化）是同构的。

**跟已有 SLO-aware 工作的对比：**

| | Sarathi-Serve | DistServe | MuxWise | PD-TDM |
|---|---|---|---|---|
| 有 SLO 概念？ | 隐式 (chunk size 影响 TTFT/TPOT) | 明确 (goodput formulation) | 明确 (SLO-aware SM partition) | 明确 (SLO→ratio mapping) |
| SLO 自适应粒度 | — | 配置层面 (node assignment) | 在线 (dynamic SM reallocation) | 配置层面 (per-workload ratio) |
| 自适应机制 | — | goodput maximization | SM partition ratio 动态调 | interference budget → ratio |
| 在 NPU 可用？ | ✅ (C3) | ✅ (c4_pd, 需 2 卡) | ❌ | ✅ |

### 2.4 Defining Figure（论文最重要的图）

```
         tail_interference
              ↑
         high |
              |    ● M3.1 (PD-TDM)
              |     低 mean, 高 tail
              |     "interference shifted to tail"
              |     TTFT 友好, TPOT_p99 敏感
              |     ratio = 0.8
              |
              |         ● C1 (Unified)
              |          中 mean, 中 tail
              |          "interference everywhere"
              |
              |    ○ MuxWise (requires SM partition)
              |    ★ c4_pd (1P1D, 2-card)
              |     零 mean, 接近零 tail
              |     "interference eliminated"
              |     ← ratio 控制移动方向
         low  |         ● C3 (Chunked Prefill)
              |          高 mean, 低 tail
              |          "interference shifted to mean"
              |          TPOT 友好, TTFT 敏感
              |
              └──────────────────────────→ mean_interference
                   low              high
```

**这张图的三重含义：**
1. **解释过去：** 统一解释已有的 PD 调度策略——它们都是 "干扰转移操作"，只是转移方向和程度不同
2. **定位 PD-TDM：** PD-TDM 填了 temporal cell，通过 ratio knob 在 mean-vs-tail 轴上主动选择位置（而非被动接受硬件决定的干扰分布）
3. **指导使用：** SLO 结构决定目标位置 → ratio 是到达目标位置的 knob → SLO-aware ratio selection 是使用这个 knob 的方法

**这张图在论文中的三个出现位置：**
1. **S2 (Background & Motivation):** 概念版，用 toy example 建立坐标系和四种转移操作
2. **S3 (Design):** 在 PD-TDM 设计中被引用——"this is the design space, and PD-TDM occupies the temporal cell"
3. **S5 (Evaluation):** 数据版，真实数据填入坐标系，验证预测位置与实际位置一致

---

## 3. Thesis Statement

### 完整版

> *PD-TDM is an SLO-aware prefill-decode temporal multiplexing scheduler for accelerators without spatial partitioning support. It is built on the insight that different PD scheduling strategies are fundamentally **interference shifting operations** — they relocate prefill-decode interference to different domains of the decode latency distribution. PD-TDM layers three components into one integrated solution: (1) the **interference shifting framework**, which provides the analytical language to understand why different strategies perform differently under SLO constraints; (2) **phase-pure temporal multiplexing**, which implements interference shifting via a single ratio-controlled knob; and (3) **SLO-aware ratio selection**, which maps SLO structure to optimal interference positioning at the configuration level. Together, these three layers make PD-TDM the first complete SLO-aware PD scheduling solution for the class of accelerators that lack spatial multiplexing hardware.*

### 一句版

> *PD-TDM: SLO-aware temporal multiplexing that shifts P/D interference to the latency tail, trading TPOT_p99 for TTFT and goodput — the only single-card mechanism that actively controls interference positioning on accelerators without spatial partitioning.*

### 三层拆解

| 层 | 解决的问题 | 对应设计 | 在论文中的位置 |
|---|---|---|---|
| **Layer 1: Analysis** | 为什么不同 PD 策略有不同的 SLO 表现？ | Interference Shifting Framework | S2 |
| **Layer 2: Mechanism** | 如何在 NPU 上控制干扰的位置？ | Phase-pure scheduling + ratio knob | S3.2-S3.3 |
| **Layer 3: Adaptation** | 给定 SLO，怎么选干扰位置？ | SLO-aware ratio selection | S3.4 |
| **统一** | **三者一起构成 PD-TDM 的完整贡献** | **PD-TDM** | S1, S3-S6 |

### 跟已有工作的本质区别

已有工作各自只做了 "控制" (Sarathi = chunking to reduce tail, without SLO-awareness) 或 "消除" (DistServe = eliminate via disaggregation, at KV transfer cost) 或 "规避" (MuxWise = avoid via spatial isolation, requires SM partition)。**PD-TDM 第一个做的是 "理解→控制→使用" 的完整闭环。**

---

## 4. Contribution Statement

**核心贡献是一个三层集成的方案 (PD-TDM)，拆解为五个可独立评估的子贡献：**

| # | Contribution | 层次 | 审稿人视角 |
|---|---|---|---|
| **C1** | **PD-TDM: 三层集成的 SLO-aware temporal multiplexing** — 在无空分能力的加速器上，首次实现 "理解干扰 (framework) → 控制干扰 (phase-pure + ratio) → 使用干扰 (SLO→ratio mapping)" 完整闭环 | 系统 (三层合一) | "一个完整的 solution，不只一个 idea" |
| **C2** | **Interference Shifting Framework** — 将 PD 调度策略统一解释为 "干扰转移操作"，定义 mean-domain vs tail-domain interference，为 PD-TDM 提供设计合理性 | 分析层 (Layer 1) | "给了我新的 lens 来理解 PD 调度" |
| **C3** | **SLO-Aware Ratio Selection** — 将 SLO 自适应正确定位在配置层面：SLO 结构 → 干扰预算 → 最优 ratio，同一 workload 的 SLO 结构稳定，一次 profiling 确定 ratio | 自适应层 (Layer 3) | "SLO 自适应被重新定义为合理的工程选择" |
| **C4** | **First systematic PD scheduling characterization on Ascend NPU** — 4-way paradigm comparison (C1/C3/M3.1/c4_pd) with goodput-centric evaluation，首次在 NPU 上完整刻画 PD 调度设计空间 | 实证 (跨层验证) | "首次在 NPU 上做了这么全的对比" |
| **C5** | **Finding: static ratio suffices** — 在线动态控制器 (PID / fast-loop selector) 不带来统计显著的额外收益；ratio 的单调可预测性使 per-workload 配置足够 | 实证发现 | "诚实的发现，对实践者有指导意义" |

### 什么不是 contribution（明确不自称）

- Phase-pure scheduling 的想法本身 (PDM/Drift 已有) — 在 Background 中 cite
- Token bucket / urgency sorting 等具体调度算法 — S3 一笔带过，它们是实现细节不是贡献
- PID controller / fast-loop selector — 不写进 paper 主体 (在 S5.5 最多一段 honest negative result)
- Kernel 优化 — 不写
- 与 MuxWise 的性能对比 — 不做（平台不同，不可比）

---

## 5. Paper Structure (9 Sections, 详细到段落)

### S1: Introduction (~2 pages)

**段落 1 — 背景钩子:**
LLM serving 的核心矛盾：prefill (compute-bound) 和 decode (memory-bound) 对硬件资源的需求互补但互相干扰。如何管理 P/D 干扰决定了 serving 系统的 goodput。

**段落 2 — 三代 PD 调度与一个未填的 cell:**
- 1st gen (Sarathi): chunked prefill, 减少干扰但无法根除。做了 "控制" 但没有 SLO 视角
- 2nd gen (DistServe): PD 分离, 根除干扰但有 KV 传输代价
- 3rd gen (Semi-PD / MuxWise): 空分动态混部, 单卡内隔离 — **需要 SM 分区硬件，NPU 不可用**
- **未被系统研究的 cell:** 单卡内 temporal multiplexing + SLO-aware — 无空分能力的加速器上唯一的 P/D 隔离路径

**段落 3 — Gap statement:**
当前没有任何方案在无空分能力的加速器上实现 "理解干扰 → 控制干扰 → 使用干扰" 的完整闭环。已有时分复用工作 (PDM/Drift) 缺 SLO 视角与系统化研究。

**段落 4 — PD-TDM:**
PD-TDM 是一个三层集成的 SLO-aware PD 调度方案：Layer 1 (Interference Shifting Framework) 解释干扰如何影响 SLO；Layer 2 (Phase-Pure Temporal Multiplexing) 通过 ratio knob 控制干扰位置；Layer 3 (SLO-Aware Ratio Selection) 将 SLO 结构映射为最优 ratio。三层共同构成在无空分能力的加速器上首个完整的 SLO-aware PD 调度 solution。

**段落 5 — Contributions (列表, 对应 §4 的 C1-C5):**
1. PD-TDM: 三层集成的 SLO-aware temporal multiplexing
2. Interference Shifting Framework: 统一 PD 调度分析语言 (Layer 1)
3. SLO-Aware Ratio Selection: 配置层面的 SLO 自适应 (Layer 3)
4. First NPU characterization with goodput-centric eval (4-way comparison)
5. "Static ratio suffices" finding

**段落 6 — 一句结果预告:**
"On Ascend NPU with Qwen3-8B, PD-TDM achieves X% higher max sustainable QPS at TTFT<Y compared to chunked prefill, while maintaining comparable TPOT_p99 at Z% SLO meet rate."

---

### S2: Background & Motivation (~3 pages)

**2.1 LLM Inference Preliminaries (0.5 page)**
- Prefill vs Decode 的计算特征 (compute-bound vs memory-bound)
- TTFT / TPOT / SLO 的定义
- Goodput = output tokens/s that meet SLO constraints

**2.2 PD Scheduling Design Space (1 page)**

用 taxonomy tree 展示 PD 调度策略的完整版图：

```
PD Interference Management Strategies
├── Coupled (P/D share iteration)
│   ├── Unified batching (baseline)
│   └── Chunked prefill (Sarathi-Serve, OSDI'24)
│       Effect: tail→mean interference shift
├── Disaggregated (P/D on separate devices)
│   └── Full disaggregation (DistServe, OSDI'24)
│       Effect: eliminate interference, pay KV transfer cost
├── Spatial co-location (P/D share GPU, partitioned SMs)
│   ├── Semi-PD (EuroSys'25)
│   └── MuxWise (ASPLOS'26)
│       Effect: avoid interference via SM isolation
│       Requirement: SM partitioning ← NPU lacks this
└── Temporal co-location (P/D share device, separate time slices)
    └── PD-TDM (this work)
        Effect: mean→tail interference shift
        Requirement: none (universal)
```

**2.3 The Interference Shifting Framework (1 page) — PD-TDM 的分析基础 (Layer 1)**

明确这段是 PD-TDM 的设计合理性来源，不是独立的 contribution。

- 引入 mean-domain vs tail-domain interference 的操作性定义
- 四种干扰转移操作 (T_couple / T_chunk / T_temporal / T_spatial / T_disagg)
- Defining Figure (概念版): 各范式在干扰空间中的理论位置
- 核心 insight: **没有哪个范式在所有维度上优于其他 — 选择取决于 SLO 结构。** PD-TDM 的原理就是 "用 ratio 主动选择干扰位置，而不是被动接受"

**2.4 The NPU Gap & Motivation for PD-TDM (0.5 page)**
- Ascend NPU 没有 SM 分区 → 3rd-gen 方案不可用
- NPU 上唯一可比的方案：C1 (unified) / C3 (chunked prefill) / c4_pd (1P1D, 2 卡)
- 时分复用是唯一能在单卡内实现 P/D 隔离的机制 — 但需要做到 SLO-aware 才完整
- MuxWise scope clause 引用

**2.5 PD-TDM Overview (0.3 page) — 承上启下**
- 一句话预告三层结构：理解干扰 (Layer 1) → 控制干扰 (Layer 2) → 使用干扰 (Layer 3)
- 这张 overview 为 S3 的设计详述做准备

---

### S3: PD-TDM Design (~3 pages)

**3.1 Design Overview: Three Layers, One System (0.3 page)**

```
┌─────────────────────────────────────────────────────────┐
│ PD-TDM                                                  │
│                                                         │
│  Input: SLO target (TTFT, TPOT) + workload profile      │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Layer 3: SLO-Aware Ratio Selection                │  │
│  │ SLO structure → interference budget → ratio       │  │
│  │ "Where should we be on the interference map?"     │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │ ratio                         │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │ Layer 2: Phase-Pure Temporal Multiplexing         │  │
│  │ ratio → P-iter / D-iter dispatch via token bucket │  │
│  │ "How do we move interference to the right place?" │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │ phase decision (P/D)           │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │ AscendScheduler → NPU Execution                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌ - - - - - - - - - - - - - - - - - - - - - - - - ┐  │
│  │ Layer 1: Interference Shifting Framework         │  │
│  │ (analytical foundation — explains WHY in S2)      │  │
│  └ - - - - - - - - - - - - - - - - - - - - - - - - ┘  │
│                                                         │
│  State Probe (per-iter monitoring, for evaluation)       │
└─────────────────────────────────────────────────────────┘
```

**3.2 Layer 2: Phase-Pure Temporal Multiplexing (1 page)**

**物理机制：**

```
时间轴:
|-- P iter --|-- D iter --|-- D iter --|-- P iter --|-- D iter --|-- D iter --|...

- P iter: 100% prefill, 从 waiting queue 取请求, 最多 chunk_tokens 个 prefill token
- D iter: 100% decode, 只推进 running requests 的 decode
- 切换决策由 ratio 控制: ratio = P_iters / (P_iters + D_iters)
```

**Token Bucket 实现：**
- 每个 iter 累计 ratio × bucket_size 的 prefill credit (cap=4, 防 burst)
- 当 credit ≥ 1 且 waiting queue 非空 → P iter；否则 → D iter
- Chunked prefill 在 P iter 内生效 (chunk_tokens=2048)

**Ratio 作为干扰控制 knob：**
- 高 ratio (0.8) → 更多 P iter → prefill 更及时 → 低 mean interference → TTFT 友好，但 decode silence 更多 → 高 tail
- 低 ratio (0.5) → 更多 D iter → decode 不被打断 → 低 tail interference → TPOT 友好，但 prefill 堆积 → 高 mean
- Ratio 直接决定系统在 defining figure (S2) 中的位置

**3.3 Layer 3: SLO-Aware Ratio Selection (0.8 page)**

**设计思路：** SLO 自适应发生在配置选择层面（不是在线调参层面），因为同一 workload 的 SLO 结构稳定。

**SLO→Ratio 映射逻辑：**

```
给定 SLO target (T_ttft, T_tpot) + workload profile:

Step 1: 判断 SLO 结构
  - TTFT-saturated? (mean interference 是瓶颈) → 需要低 mean → 高 ratio
  - TPOT-saturated? (tail interference 是瓶颈) → 需要低 tail → 低 ratio

Step 2: 量化干扰预算
  - 可容忍的 mean_interference ≈ T_ttft / ttft_baseline - 1
  - 可容忍的 tail_interference ≈ T_tpot / tpot_baseline - 1

Step 3: Ratio 选择
  - Conv (TTFT-sensitive): ratio = 0.8 (默认推荐)
  - Code (TPOT-sensitive): ratio = 0.7 (默认推荐)
  - Sensitivity sweep (S5.5) 提供 ratio→goodput 的完整 mapping
```

**为什么是 "配置层面" 而非 "在线"：**
- SLO 结构和 workload class 在部署时就已知，不需要运行时探测
- Ratio 对 goodput 的效果单调可预测 (见 S5.5 sensitivity sweep)
- 在线动态调整在稳态下等价于 "推到 ratio_max 后钉住" (P1.9a 预期结论)
- 这与 DistServe 的 goodput formulation (配置层面优化 node assignment) 是同一粒度

**跟 DistServe / MuxWise 的 SLO 自适应对比 (一小段)：**
- DistServe: 配置层面优化 node assignment for goodput maximization → 需要 ≥2 设备
- MuxWise: 在线动态调 SM partition ratio for SLO → 需要 SM 分区 (NPU 不可用)
- PD-TDM: 配置层面 per-workload ratio selection → 单卡可用，无特殊硬件要求

**3.4 System Architecture & State Probe (0.5 page)**

```
trace ──┐
        ├─► SLO grid calibration (offline, per-workload, §5.1)
hw+model┘        │
                 │  SLO targets
                 ▼
SLO-Aware Ratio Selection (L3)
       │
       │  ratio
       ▼
Token Bucket + Phase-Pure Dispatch (L2)
       │
       │  phase decision (P/D)
       ▼
AscendScheduler ──► NPU Execution
       │
       ▼
State Probe (QueueMonitor / RequestTracker / Telemetry)
  → per-iter & per-request structured logging (S4)
```

三个基础设施组件 (S4 详述):
- QueueMonitor: per-iter O(1) snapshot of scheduler internal state
- RequestTracker: per-request lifecycle (admit → first_token → per_token → finish)
- Telemetry: 4-stream structured JSONL logging (iter/req/ctrl/chunk)

**3.5 Configurations for Comparison (0.3 page)**

| Config | P/D Mode | Chunking | Ratio | 角色 |
|---|---|---|---|---|
| C1 | Unified (mixed) | None | N/A | 耦合 baseline |
| C3 | Unified (mixed) | chunked prefill, chunk=2048 | N/A | Sarathi 范式在 vllm-ascend 上的实现 |
| M3.1 | Phase-pure (temporal) | chunked prefill, chunk=2048 | static (0.7-0.8) | **PD-TDM (我们的方案)** |
| c4_pd | Disaggregated (1P1D) | None | N/A | 分离 baseline (2 卡, budget reference) |

---

### S4: Implementation (~2 pages)

**4.1 vllm-ascend Integration (0.8 page)**
- 基于 vllm-ascend v0.11.0rc1 + AscendScheduler
- TDM 插件通过 `scheduler_cls` 配置注入
- Phase-pure dispatch 逻辑在 `TDMScheduler.schedule()` 中
- 与 vLLM upstream 的接口兼容性讨论 (v0.13.0+ 删除了 AscendScheduler)

**4.2 State Probe Infrastructure (0.7 page)**
- QueueMonitor: per-iter O(1) snapshot of scheduler internal state
- RequestTracker: per-request lifecycle (admit → first_token → per_token → finish)
- Telemetry: 4-stream structured JSONL logging (iter/req/ctrl/chunk)

**4.3 Graph Capture Fairness (0.3 page)**
- cudagraph_capture_sizes across configs 确保公平对比
- vllm-ascend 内置的 decode auto-padding
- 不使用 graph 优化作为 contribution (D-005)

**4.4 Open-Source Artifact (0.2 page)**
- 全 stack 开源 (Apache 2.0)
- 实验脚本 + post-hoc 分析可复现

---

### S5: Evaluation (~5 pages)

**5.1 Experimental Setup (1 page)**

| 项目 | 配置 |
|---|---|
| Hardware | 2 × Ascend 910B3 (64GB HBM each), TP=2 |
| Software | CANN 8.3.rc1, vllm-ascend v0.11.0rc1, PyTorch 2.7.1 |
| Models | Qwen3-4B, Qwen3-8B |
| Traces | Azure conv, Azure code, BurstGPT, Mixed (conv+code) |
| SLO Grid | TTFT ∈ {200, 500, 1000, 1500}ms × TPOT ∈ {50, 100, 200, 300}ms |
| Metrics | Primary: goodput at SLO (max sustainable QPS at meet% ≥ X%) / Secondary: SLO meet% at fixed QPS |
| Configs | C1 / C3 / M3.1 / c4_pd (4-way) |
| Seeds | 3 seeds, mean ± std |
| Duration | 60s per run, 20s warmup |

**5.2 Goodput-Centric Pareto Frontier (1 page) — 主图**

- 主图: x = QPS, y = SLO meet%, 4 条曲线 (C1/C3/M3.1/c4_pd)
- 竖直虚线标注 max sustainable QPS at meet% ≥ 80% / 90%
- Key finding: "M3.1 sustains X% higher QPS than C3 at SLO meet% ≥ 90% for conv workload"

**5.3 Interference Decomposition (1 page) — 核心实证**

- 从 decode token latency 的完整分布中提取 mean_interference 和 tail_interference
- 填入 §2.3 的 defining figure（概念版 → 数据版）
- 展示 C1/C3/M3.1/c4_pd 四个点在坐标系中的实际位置
- Sub-figure: per-request decode_intervals 的 CDF (直观展示 mean shift vs tail stretch)
- Sub-figure: iter trace — 时间轴上的 decode silence 模式 (physical fingerprint)

**5.4 SLO Meet% at Fixed QPS (0.5 page) — 辅助图**

- 四档 SLO grid 的 heatmap 或 grouped bar chart
- 展示 M3.1 在 TTFT-tight 档的优势和 TPOT-tight 档的劣势
- 支撑 interference shifting 的论述

**5.5 Mechanism Analysis (0.8 page)**

- **Static ratio sensitivity:** ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.8} 下 goodput 的变化
  - 展示 "one knob, monotonic/predictable effect"
- **Online controller comparison:** M1@static_ratio_max vs M3.1@PID
  - 预期: 两者在稳态下统计等价 → "static suffices"
- **Fast-loop exploration (optional, brief):** 简短提及我们探索了更复杂的 urgency-based 在线机制但未发现统计显著的改善 (one paragraph, honest negative result)

**5.6 Cross-Model & Cross-Trace (0.7 page)**
- Qwen3-4B 上的 QPS sweep
- Azure code / BurstGPT / Mixed workload
- 展示 winning regime 的稳定性

**5.7 Physical Evidence: Iter Trace Fingerprint (0.5 page)**
- Time-series plot: decode queue length during prefill iters
- "99.7% of prefill iters observe non-empty decode queue"
- "Decode silence p99 = 800-940ms during prefill iters"
- 这是 temporal multiplexing 物理代价的直接证据

---

### S6: Discussion (~2 pages)

**6.1 Interference Shifting as a Unified Lens (0.8 page)**

回到 defining figure（数据版），讨论：
- 每个范式在干扰空间中的位置是物理必然的，不是实现细节
- 选择范式 = 选择牺牲 latency distribution 的哪个部分
- 这个框架可以覆盖已有工作 (Sarathi = T_chunk, DistServe = T_disagg, MuxWise = T_spatial) 和未来工作 (新的干扰转移操作)

**6.2 Regime Analysis: When Does Temporal Win? (1 page)**

不是简单的 "temporal 在 TTFT 紧时赢"——那只是故事的一半。完整的 regime analysis 需要 **二维交叉**：SLO 结构 × Workload 结构。

**6.2.1 SLO 结构维度**

| SLO 结构 | 瓶颈在干扰的哪个域 | 范式倾向 |
|---|---|---|
| TTFT 紧, TPOT 松 | Mean interference | Temporal (低 mean → TTFT 友好) |
| TTFT 松, TPOT 紧 | Tail interference | Chunked Prefill (低 tail → TPOT 友好) |
| 都紧 | 两者都超 | 物理不可达 → 降 QPS 或升级硬件 |
| 都松 | 都无瓶颈 | 无差异 → 选最简单的 (C1) |

**6.2.2 Workload 结构维度**

| Workload 特征 | 对 mean interference 的影响 | 对 temporal 的利好程度 |
|---|---|---|
| **长 prompt** (数百-数千 token) | 单次 prefill 长 → 耦合方案中 decode 被长时间阻塞 → mean 急剧恶化 | **最强利好** |
| **高并发** (QPS ≥ 8, P+D queue 同时深) | P 和 D 同时竞争 iter → 谁等谁都有代价，无结构化分配手段 | **强利好** |
| **P/D 竞争强** (prefill 请求和 decode 请求同时到) | Coupled 方案只能靠贪心 batching 随机分配 P:D 时间 | **强利好** |
| 短 prompt (< 100 token) | Prefill 很快完成 → mean 本就不严重 | 无利好 (temporal silence 纯代价) |
| 低并发 (QPS ≤ 4) | Queue 不深 → 不需要结构化重分配 | 无利好 |
| code TPOT-sensitive workload | tail 代价 > mean 改善 | 不利 (C3 的反向 trade-off 更匹配) |

**6.2.3 二维交叉：何时选 PD-TDM**

| | TTFT 紧 + TPOT 松 | TTFT 松 + TPOT 紧 | 都紧 | 都松 |
|---|---|---|---|---|
| **长 prompt + 高并发** | ✅ **M3.1 最强优势** | △ C3 更优 | 物理不可达 | 无差异 |
| **长 prompt + 低并发** | ✅ M3.1 有优势 | △ C3 更优 | 物理不可达 | 无差异 |
| **短 prompt + 高并发** | △ M3.1 微优势 | △ C3 更优 | 物理不可达 | 无差异 |
| **短 prompt + 低并发** | 无差异 | 无差异 | 物理不可达 | 无差异 |
| **code (TPOT 敏感) + 任何** | △ 取决于 prompt 长度 | ✅ **C3 最强优势** | 物理不可达 | 无差异 |

✅ = 明确推荐  △ = 需要具体分析  ✗ = 明确不推荐

**核心 insight：PD-TDM 的优势不是 universal 的——它是在特定 regime（长 prompt + 高并发 + TTFT 紧 SLO）下有结构性优势的方案。框架 (L1) 告诉你这个 regime 是什么，机制 (L2) 告诉你为什么在这个 regime 下有效，自适应 (L3) 告诉你在这个 regime 下怎么配置。**

**这个发现对论文叙事的意义：**

- "PD-TDM 在所有场景都好" → 脆弱，审稿人挑一个反例就推翻
- "PD-TDM 在 X 条件下好，Y 条件下不好，且框架可以预测哪个是 X" → 强韧，审稿人觉得有 insight

这与已有工作的差异：Sarathi 没有系统分析 "chunked prefill 在什么 workload 下最有效"；MuxWise 分析了自己的 sensitivity 但没给出跨 paradigm 的 regime 对比。PD-TDM 通过 interference shifting framework 首次给出了这种跨 paradigm 的 regime prediction。

**6.3 Static Ratio: Why Simple Works (0.3 page)**
- ratio 直接映射到干扰坐标位置
- 在线动态调 ratio 在这个 knob 上没有提供额外自由度
- 对实践者的建议: profile 一次 workload → 选 ratio → 不动

**6.4 Intra-Node Disaggregation: The c4_pd Reference Point (0.4 page)**
- c4_pd (1P1D) 在没有 KV 传输开销的情况下 (同机共享显存) 提供了理论最优
- 但需要 2 卡 → 资源利用率只有 temporal 的一半 (每卡只做一件事)
- 这是 budget-equivalent 对比: "用同样的 2 卡, temporal 可以同时服务 P+D, 而 1P1D 每卡只管一件事"

---

### S7: Related Work (~1.5 pages)

按 taxonomy 组织：

**PD Coupling Optimization:**
- Sarathi-Serve (OSDI'24): chunked prefill, 我们的 C3 baseline
- DeepSpeed-FastGen, vLLM: 其他 chunked prefill 实现

**PD Disaggregation:**
- DistServe (OSDI'24): goodput-centric PD separation, 我们的 goodput metric 参照
- Llumnix, Splitwise: 其他 disaggregation 方案

**Spatial PD Co-location:**
- MuxWise (ASPLOS'26): SLO-aware SM partitioning — **explicit scope clause 排除 NPU, 我们的 complementary work**
- Semi-PD (EuroSys'25): profiling-based PD mode switching
- DuetServe, RAPID-Serve: 其他空间复用方案

**Temporal Multiplexing:**
- PDM, Drift: 早期的时分复用方案 — **缺 SLO 视角 + 系统化 evaluation**
- GANDIVA (OSDI'18): DL 训练时分复用，非推理

**LLM Serving Systems:**
- vLLM, TGI, TensorRT-LLM: 通用 serving 框架

**我们的差异化:** interference shifting framework 提供了一个统一的分析语言 + 首次在 NPU 上系统研究 temporal cell + MuxWise 的明确 complementary 定位

---

### S8: Limitations (~0.8 page)

1. **Single hardware platform:** 只在 Ascend 910B3 上实测，"lacking SM partitioning" 这一类内其他硬件 (older GPUs, edge ASICs) 未验证 → 在 Discussion 中作为 conceptual claim 而非 empirical claim
2. **Single model family:** 只测了 Qwen3 系列 → cross-family 验证留给 future work
3. **Static ratio requires per-workload profiling:** non-trivial 的前置开销 → 但不影响结论 (ratio 的 effect 是可预测的)
4. **No online adaptation:** 不支持 request-level 的动态调整 → future work 可以研究 lightweight adaptation
5. **vllm-ascend version lock:** v0.11.0rc1, 社区已删除 AscendScheduler → 论文写作时诚实声明

---

### S9: Conclusion (~0.3 page)

- PD-TDM 展示了在无空分能力的加速器上，时分复用是可行的 P/D 隔离机制
- Interference shifting framework 为 PD 调度设计空间提供了统一的分析语言
- Simple static ratio suffices — 不需要 complex online controller
- Future: 扩展到更多硬件平台，研究 lightweight online adaptation

---

## 6. 与已有四篇论文的关系定位

| 论文 | 跟我们的关系 | 我们在 paper 里怎么处理 |
|---|---|---|
| **Sarathi-Serve** | Chunked prefill 的提出者，我们的 C3 baseline | S2 作为 coupling 方案的代表详述；S5 作为 baseline 对比 |
| **DistServe** | Goodput 概念的提出者，PD 分离的 formalization | S2 作为 disaggregation 方案的代表详述；S5 的 goodput 定义参照其 formalism；c4_pd 作为 budget-equivalent 分离 baseline |
| **Semi-PD** | 早期空分混部方案 | S2 列入 spatial co-location 类别；S7 作为 related work |
| **MuxWise** | **最相关也最关键** — 空分混部的 SOTA，明确排除 NPU | S1 scope statement 引用其 scope clause；S2 列入 spatial co-location 类别并标注 "excludes NPU"；S7 作为 complementary work；**不做实验对比**（平台不同，不可比） |

---

## 7. 实验清单 (paper 成立的前置条件)

### 必须完成 (P0)

| # | 实验 | 内容 | Wall 时间 | 支撑的 claim |
|---|---|---|---|---|
| 1 | QPS sweep | [2,4,6,8,10,12,14,16] × C1/C3/M3.1/c4_pd × conv/code × 3 seeds × Qwen3-8B | 1-2 天 | C3 (goodput-centric eval) |
| 2 | Cross-model QPS sweep | 同上 matrix, Qwen3-4B | 2-3 天 | C3 (cross-model consistency) |
| 3 | Interference decomposition | 从已有 telemetry 数据 post-hoc 提取 mean/tail interference 坐标 | 0 天 wall (分析) | C1 (interference shifting framework 实证) |
| 4 | P1.9a static ratio scan | M1@ratio ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8} vs M3.1 | 1 天 | C4 (static suffices) |
| 5 | c4_pd 补跑 | Azure trace + Qwen3-8B, 跟进 evaluation 框架 | 1-2 天 | 4-way comparison 完整性 |

### 强烈建议 (P1)

| # | 实验 | 内容 | Wall 时间 | 支撑的 claim |
|---|---|---|---|---|
| 6 | BurstGPT trace | 写 loader + sweep | 1-2 天 | Cross-trace robustness |
| 7 | Mixed workload (conv+code) | Trace 生成 + sweep | 1-2 天 | 是否扩 winning regime |
| 8 | C3 反常根因分析 | Telemetry 分析 + profiling | 0.5-1 天 (分析) | 防御审稿质疑 "C3 为什么在 NPU 上反常" |
| 9 | Ratio sensitivity 系统整理 | 已有数据, 整理成论文图 | 0.5 天 (分析) | S5.5 mechanism analysis |
| 10 | Controlled microbenchmark | 固定 8 decode + 周期性 prefill inject, 对比 C1/C3/M3.1 的 per-token latency trace | 1 天 wall | 最直观的 visual evidence for interference shifting |

### 不需要做

| 实验 | 理由 |
|---|---|
| MuxWise 对比 | NPU 不支持 SM 分区，物理不可跑 |
| Semi-PD 对比 | 同上 |
| Llama/Mistral cross-family | 适配成本高，Limitations 诚实交代 |
| Fast-loop selector 修复 (P1.7b) | 已证伪，不值得花时间；改为 "static suffices" 叙事 |
| Mixed_mode ablation | D-010 已决定 paradigm-level Δ 不归因到单机制 |

---

## 8. 写作时间线 (4 周)

| 周 | 任务 |
|---|---|
| **W1 (当前)** | 实验: QPS sweep + P1.9a + c4_pd + interference decomposition post-hoc |
| **W2** | Paper draft S1-S4 (Intro / Background+Framework / Design / Implementation) + 实验: cross-model + BurstGPT + mixed workload |
| **W3** | Paper draft S5-S9 (Evaluation / Discussion / Related Work / Limitations / Conclusion) + 实验收尾 |
| **W4** | Polish + 选 venue + 投递 |

**Venue 策略:**
- 优先: IPDPS / DSN / Middleware (CCF-B)
- Fallback: IISWC / ICPADS / HPCC (CCF-C)

---

## 9. 关键写作原则

1. **PD-TDM 是主角，三层是它的组成。** 不要说 "我们贡献了 framework + 实现了一个系统"，要说 "PD-TDM 是一个三层集成的方案：理解干扰 (L1) → 控制干扰 (L2) → 使用干扰 (L3)"。

2. **每一层都不能独立存在。** Layer 1 没有 L2/L3 是空谈；Layer 2 没有 L1 是盲目设计 (为什么 phase-pure 有用？) 没有 L3 是不完整 (ratio 怎么选？)；Layer 3 没有 L1/L2 是无根据的 heuristic。

3. **SLO 自适应没有被放弃，而是被正确定位了。** "Static ratio suffices" 不是 "SLO 自适应不重要"，而是 "SLO 自适应的正确粒度在配置层面"。跟 DistServe 的 goodput formulation 是同一粒度。

4. **诚实比夸大更能通过审稿。** 主动写明 limitation (single platform, cross-family 未测, fast-loop 不 work)。审稿人会因为诚实而信任你的数据。

5. **MuxWise 是盟友不是对手。** 每次提到 MuxWise 都要强调 "他们填了 spatial cell, 我们填了 temporal cell — 合在一起，PD 调度设计空间才完整"。

6. **"Simple" 是 feature 不是 bug。** 不要为机制简单道歉。用数据展示 "ratio 是唯一重要的 knob，它的效果单调可预测" — 这是对后续工作的指导，不是弱点。

---

## 10. 审稿人可能攻击的点 & 预置防御

| 攻击 | 防御位置 | 防御策略 |
|---|---|---|
| "Phase-pure scheduling is not new (PDM/Drift)" | S2, S7 | 承认 cite，我们的贡献不在 phase-pure 本身。PD-TDM 的贡献是三层集成：理解 WHY (L1) + 实现 HOW (L2) + 使用 WHEN (L3)。PDM/Drift 只做了 L2 的一部分 |
| "Mechanism too simple (just a static ratio)" | S3.3-S3.4, S6.3 | Ratio 本身不简单——它是三层框架推导出的控制 knob。"Static suffices" 是经过在线控制器实证证伪后的发现，不是一开始就选择简单 |
| "SLO adaptation claim is weak (no online controller)" | S3.3 | SLO 自适应的正确粒度在配置层面，不是在线。跟 DistServe 的 goodput formulation 是同一粒度。同一 workload 的 SLO 结构稳定 → per-workload ratio 是完整自适应 |
| "Is the framework a contribution or just design rationale?" | S1, S2, §C2 | Framework 是 PD-TDM 的 Layer 1（分析基础），不是独立贡献。它的价值在于使 PD-TDM 的设计有了 "why"——跟已有工作 "we do X and it works" 的本质区别 |
| "Single platform, single model family" | S8 | 诚实承认，解释原因 (NPU 适配限制)。三层框架的 conceptual claim 不依赖具体平台——L1 是物理必然，L3 的方法论是通用的 |
| "C3 on NPU is abnormally bad — unfair baseline" | S5.7, S8 | 分析根因 (FIA kernel on NPU)，诚实交代；但这正是 motivation 的一部分 — NPU 上 coupling 方案有平台相关的劣势，temporal 可以绕过 |
| "Why not compare with MuxWise?" | S1, S2.4 | MuxWise requires SM partitioning — NPU 不可达；scope 互补。我们的贡献是 "PD 调度选择在无空分硬件上应该怎么做"，不是 "跟 MuxWise 谁更好" |
| "Static ratio needs per-workload profiling" | S5.5, S6.3 | 承认，开销一次性的。MuxWise/Semi-PD 也需要 profiling-based 配置。Ratio sensitivity sweep 提供推荐值 |
| "Winning regime is narrow (only long-prompt + high-concurrency + TTFT-tight)" | S2.2, S6.2 | **这正是框架的 predictive power。** "PD-TDM 在所有场景好" 是脆弱 claim，"PD-TDM 在框架预测的场景赢、在预测的场景输" 是强韧 finding。我们提供了 **regime prediction 能力**——已有工作 (Sarathi/MuxWise) 都没给出跨 paradigm 的 regime 对比 |
| "Goodput improvement is marginal in some regimes" | S5.2, S6.2 | Pareto frontier 展示完整 picture；claim 不是 "always better"。真正的贡献是给出 "WHEN temporal wins" 的 regime analysis — 这本身就是知识贡献 |
| "Three layers sound like three separate ideas stitched together" | S1, S3.1 | 它们不是三个独立 idea——它们是一个工程方案的三步逻辑链。L1→L2→L3 之间是依赖关系：没有 L1，L2 不知道为什么 phase-pure 有用；没有 L3，L2 不知道怎么选 ratio。已有工作各自只做了其中一步 |

---

*本文档是 paper writing 的 living blueprint。任何 narrative 级修改先更新本文档。*
