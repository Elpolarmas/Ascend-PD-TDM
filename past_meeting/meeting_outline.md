# 导师汇报提纲 — NPU 单卡 PD 时分复用调度系统

> 汇报日期：2026-04-03（已完成） | 后续讨论：2026-04-08
> 模型：**Qwen3-7B**（导师建议，取代原 Qwen3-4B） | 硬件：2×Ascend 910B3 (64GB HBM)
> 图表：`/vllm-workspace/lzn-pro/figures/`

---

## 导师反馈纪要（2026-04-08）

**1. 模型选型**
- 建议改用 **Qwen3-7B** 作为主测模型（替代 Qwen3-4B）
- 现阶段**不必过度追求大模型（30B+）和真实生产负载**：先把机制跑通、把增量收益讲清楚；大模型/真实 trace 留到后期再补
- 这意味着 §五.3.2「小模型局限」的论述可以弱化，重点不再是"小模型不够大"，而是"7B 已能展示 P/D 互补效应"

**2. 相关工作对比缺失（最主要的批评）**
- 当前提纲对相关工作只在 §三 比较了"算子级 vs iteration 级"两条技术路线，**没有系统性地对比已有的单卡 PD 混部 / 时分复用工作**
- 必须新增一节明确回答："我们和 Sarathi-Serve / DistServe / Semi-PD / DuetServe / MuxWise / RAPID-Serve / PDM/Drift / Dilu 的差异在哪里？为什么这些工作不能直接套用到 NPU？"
- 已在下方新增 **§二.5「相关工作对比」**，需在下次汇报中重点讲解

**3. 后续动作**
- 切换 baseline 到 Qwen3-7B 重跑 Exp B/D/E（资源画像 / 并发 / Phased vs Unified）
- 把 idea_proposal.md §1.2/§1.3 的相关工作梳理 + tdm_related_work.md 的七维对比，浓缩成 1-2 张幻灯片级别的对比表，融入下次汇报

---

## 一、问题与动机（2 min）

**核心优化目标**：Goodput——即满足 SLO 约束的有效请求吞吐量。现有架构在真实动态负载下难以最大化 Goodput。

> 展示 **fig1**（P/D 资源互补）+ **fig2**（并发 vs 吞吐/TPOT）

**两个核心问题**：
- **问题（1）**：耦合架构（P/D 同 batch，同 batch 干扰无法根除）与分离架构（P/D 跨卡物理绑定，KV Cache 跨物理资源迁移开销 + 必然需要 ≥2 卡）均存在不可消除的结构性代价，**无一种方法能在单卡上根据实时负载自适应调整 P/D 执行策略**。
- **问题（2）**：PD 动态混部沿空间复用（依赖 SM/CU/MPS 等硬件能力）与时间复用（硬件普适但多采用跨多迭代生效的固定比例或启发式策略）两条路径发展，**无一种方法能兼顾硬件普适性与迭代粒度的调度灵活性**。

**资源互补性支撑**：Prefill = compute-bound（AICore 69.8%），Decode = memory-bound（HBM BW 25.1%）；1→64 并发吞吐增 58× 但 TPOT 恒定——NPU 在中低并发下算力大量空闲，P/D 资源需求互补 → 时分调度存在提升空间。

**方案**：选择硬件普适性更强的时间复用路径，在 iteration 粒度上引入 SLO 余量反馈控制，在通用加速器上同时兼顾硬件普适性与迭代粒度的负载感知调度。

---

## 二、TDM 核心概念澄清（3 min）

### 什么是时分复用？

**TDM 不是每个 iteration 做 P/D 的二选一**，而是在滑动时间窗口内，动态调整三种执行模式的比例：

```
时间轴（每个字母 = 一个 iteration = 一次 ACL Graph replay）：
  D D D P D D D D P D D P P D D D D D D P D D ...
  ←─ 低负载：P 稀疏 ──→ ←─ 突发请求 ─→ ←─ 恢复 ─→
高并发场景退化为 Mixed：
  D D M M D D M D D D ...
```

**关键等价关系**：1 iteration = 1 batch = 1 ACL Graph replay。Iteration 是最小切换单元（ACL Graph 的原子性），但调度看的是窗口级 P:D:M 比例。

**三种执行模式**：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| PREFILL | 纯 prefill iteration | 队列深、decode 空闲、低并发 |
| DECODE | 纯 decode iteration | TPOT 紧张、无新请求 |
| MIXED | Chunked Prefill（P+D 混合 batch） | 高并发、负载平稳 |

**TDM 是 Chunked Prefill 的超集**：CP = 全部 iteration 都选 MIXED 的特例。TDM 动态选择最优模式组合。

> 展示 **fig_concept**（时间轴 + 三模式切换示意）

**实验支撑**（Exp E）：Phased 并非一致优于 Unified——低并发 +10%，高并发 -20% → **不是"分离一定好"或"混合一定好"，需要动态选择**。

---

## 三、为什么是 Iteration 级？（2 min）

> 展示 **fig8**（架构总览图）

我们评估了两条技术路线：

| 维度 | 算子级多流交错 | Iteration 级时分复用 |
|---|---|---|
| 可行性 | **不可行** | **可行** |
| 原因 | ACL Graph FULL 模式不可拆分；PIECEWISE 劣化 26% | iteration 是 ACL Graph 的自然执行边界 |
| 改动范围 | model_runner + acl_graph + attention | 仅 AscendScheduler |

**结论**：NPU 的 ACL Graph 机制决定了 iteration 是最自然的时分边界。一次 graph replay = 一个 iteration，在此粒度切换零额外开销。

---

## 四、系统设计 — 核心创新 + 落地配套（8 min）

**定位**：以 **SLO 自适应控制模块** 为核心创新，**分层硬件状态采集模块** 为其提供低开销的硬件状态输入，共同实现迭代粒度的负载感知调度；为使该方法在采用预编译计算图的加速器上高效落地，配套设计 **图识别控制模块**。

### 模块一：SLO 自适应控制模块（核心创新）

**核心控制变量**：prefill 插入率 r_p —— 不是每次做决策，而是维护一个平滑变化的比例。

**V1（原型）：优先级规则 + 防饿死**

```
状态 S = (Q, R, ttft_slack, tpot_slack)

决策（按优先级）：
  1. tpot_slack < T_tpot              → DECODE（保 decode SLO）
  2. Q > 0 且 ttft_slack < T_ttft     → PREFILL（防 TTFT 违约）
  3. consecutive_d > max_consecutive   → PREFILL（防 prefill 饿死）
  4. Q > 0 且 R == 0                  → PREFILL
  5. Q == 0                           → DECODE
  6. R > threshold_high               → MIXED（高并发用 CP 更优）
  7. 否则 → 根据硬件利用率反馈选择
```

V1 本质 reactive：等 SLO 快违约才切换，可能震荡和滞后。

**V2（目标）：AIMD 自适应比例控制**（借鉴 TGS 拥塞控制思想）
- 将 prefill 插入率 r_p 类比 TCP 发送窗口
- TPOT violation → multiplicative decrease（r_p × 0.5）
- TPOT 达标且 Q > 0 → additive increase（r_p + α）
- r_p 有下界（防 prefill 饿死）、上界（防 decode 饿死）
- 高并发自动切 MIXED 模式
- 渐进收敛到最优 P:D:M 比例，无需手调

**V1 先行理由**：快速验证方向是否有增量收益，再迭代算法复杂度。

**与现有方案的对比**（Exp E3 公平 2 卡实验）：

> 展示 **fig6**（TPS 对比）+ **fig7**（延迟对比）

| 方案 | TPS vs Baseline | 特点 |
|---|---|---|
| Unified TP=2（无 CP） | 基准 | P/D 混合，无隔离 |
| Unified TP=2 + CP | **+5%~+11%** | Chunked Prefill，强 baseline |
| Phased TP=2 | +2%~+7% | 粗粒度全 P→全 D（一刀切） |
| Disagg 1P1D | **-2%~-12%** | 物理分离，资源浪费 |

### 模块二：图识别控制模块（落地配套）

> 展示 **fig3**（graph shape 间距 + eager boundary）+ **fig4**（padding 浪费）

**问题**：48 种已编译 graph shape，小 batch 间距大（1→2→8→16），padding 浪费平均 23%，最坏 62.5%。精确匹配比 padding 高 6% TPS，大 shape 下效率差达 22%。

**Layer 1 和 Layer 2 的关系（串行决策 + 反馈修正）**：

```
Layer 1（Phase Ratio Controller）决定"做什么模式" → 时间维度
Layer 2（Graph-Aware Batch Shaping）决定"做多大" → 容量维度

Step 1 — Layer 1 决定 Phase（P / D / M）
Step 2 — Layer 2 决定 Batch Size，对齐 graph shape
  PREFILL: total_tokens > 512 → eager; ≤512 → 对齐 graph shape
  DECODE: 请求数对齐 graph shape
Step 3 — Layer 2 反馈修正 Layer 1
  Q 不够凑好 shape → 建议延迟切换
  running 在 shape 间距中间 → 建议等精确匹配点再切
```

**例子**：running=33，新来 2 个请求
- Layer 1：ttft_slack 充裕 → DECODE
- Layer 2：33 pad 到 40（浪费 17.5%），等 1-2 iter 降到 32（精确匹配）→ 建议等
- 最终：保持 DECODE，running=32 后切 PREFILL

**原创性**：调研 12 篇核心论文 + 5 篇 TDM 论文，**无一将编译图约束引入调度决策**。

---

## 五、相关工作对比（5 min，**本次新增**）

> 导师反馈中点名要求补充：原提纲完全没有系统性的相关工作对比。
> **两条线必须分开讲**：
> - **A 线 — 单卡 PD 混部**：来自 LLM serving 社区，问题是"P 和 D 在一张卡上怎么放" → idea_proposal.md §1.2 / paper_template.md
> - **B 线 — 通用时分复用**：来自 GPU 调度 / serverless / 训推混合社区，问题是"多任务怎么时分共享 GPU" → tdm_related_work.md
> - **我们位于两条线的交叉点**：把 B 线的 TDM 调度思想，搬到 A 线的 LLM P/D 场景，且首次落到 NPU + 编译图约束下

---

### 5.1 A 线：单卡 PD 混部相关工作

按"P 和 D 如何在一张卡上共享"的机制分四类：

| 类别 | 代表工作 | 核心机制 | NPU 可迁移性 |
|---|---|---|---|
| **耦合执行（Chunked Prefill）** | Sarathi-Serve (OSDI'24), SplitFuse (DeepSpeed-FastGen) | P+D 混进同一 batch，chunk size 控平衡 | ✅ 可，vllm-ascend 默认未开启；**我们最强 baseline** |
| **跨卡 PD 分离** | DistServe (OSDI'24), Llumnix (OSDI'24) | P/D 放到不同 GPU，KV cache 跨卡传输 | ⚠️ 与"单卡"目标矛盾；KV 迁移开销大 |
| **单卡空间复用** | Semi-PD, Nexus, DuetServe, **MuxWise** (2.2× goodput), **RAPID-Serve** (4.1× tput) | SM partition / MPS / CU masking / GreenContext 把一张卡切成 P 区和 D 区 | ❌ **NPU 无对等硬件抽象，整类方案不可迁移** |
| **Kernel 级 overlap** | POD-Attention (ASPLOS'25) | 在 attention kernel 内让 P/D 算子并行 | ❌ ACL Graph FULL 不可拆分，PIECEWISE 模式实测 -26% |
| **单卡时间复用（思路最近）** | **PDM/Drift** (arXiv'24) | batch-level 时间切分，soft SLO 约束 | ✅ 思路可借鉴，但粒度粗、无硬件感知 |

**A 线小结**：
1. NPU 上**空间复用路径已被硬件封死**（无 SM partition / CU masking / MPS）→ 这是"为什么是时分"的硬性原因
2. CP 是 NPU 上唯一可用的 LLM 端方案，但 vllm-ascend 默认关闭 → 我们要做的是**把 CP 这条路再向前推一步**：iteration 级动态 P/D/M 切换 vs CP 的"全 Mixed"
3. 与 PDM/Drift 的关键差异：**iteration 粒度 + 双 SLO + 编译图感知**（Drift 是 batch 粒度 + soft SLO + 无图感知）

---

### 5.2 B 线：通用时分复用相关工作

来源：Dilu 引用链中的五篇核心 TDM 论文（详见 `tdm_related_work.md`）。这些工作面向 serverless / 训推混合 / 容器云，**不是 LLM P/D 专用**，但调度思想（配额、自省、自适应）可以借鉴。

| 论文 | 场景 | 切换粒度 | 隔离机制 | 自适应 | 与我们的关系 |
|---|---|---|---|---|---|
| **FaST-GShare** [19] | Serverless 推理 | ms 级窗口 token 配额 | MPS 空间 + token 限流 | 静态 profiling | 借鉴：phase iteration 配额（连续最多 N 个 P-iter）+ 时空矩形匹配启发 Graph shape × phase |
| **GaiaGPU** [20] | 容器云 vGPU | 中粒度运行时 | API 拦截 + 软/硬限 | 周期性调整 | 较远；只借鉴"动态再分配"的总体思想 |
| **TGS** [47] | DL 训练混部 | kernel 级 AIMD | 速率控制隐式隔离 | 类 TCP 拥塞控制 | **直接借鉴**：V2 phase ratio 用 AIMD，TPOT violation = 拥塞信号，r_p × 0.5 multiplicative decrease |
| **FaaSwap** [54] | Serverless 多模型 | 模型粒度 swap | 优先级队列 + 异步 swap | 历史 RRC 估算 | 较远；只借鉴优先级队列思想 |
| **Dilu** (ASPLOS'25) | Serverless DL 混合 | ~5ms token 周期 | 状态机 + KLC 实时自省 | 多因子 profiling + 2D co-scaling | **最直接借鉴**：(a) 状态机 NORMAL→THROTTLE→EMERGENCY 映射到我们 SLO slack 阈值；(b) 2D co-scaling 映射到 phase ratio (快调) + KV cache 预算 (慢调)；(c) 离线画像表是其多因子 profiling 的简化 |

**七维细对比（节选自 tdm_related_work.md，重点行）**：

| 维度 | FaST-GShare | GaiaGPU | TGS | FaaSwap | Dilu | **NPU PD-TDM (ours)** |
|---|---|---|---|---|---|---|
| 切换粒度 | ms 窗口 | 运行时 | kernel | 模型级 | 5ms token | **iteration (~20ms, ACL Graph 自然边界)** |
| 共享对象 | 多 function | 多容器 | 训练任务 | 多模型 | 多任务 DL | **单模型 P/D 两阶段** |
| 切换开销 | 低 | 中 | 低 | **高（模型 swap）** | 低 | **零（共享权重 + KV cache）** |
| SLO 形式 | 吞吐（隐式） | QoS 等级 | 优先级 | 单一延迟 | 延迟 + goodput | **TTFT + TPOT 双 SLO** |
| 编译图感知 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **ACL Graph shape-aware** |
| 目标硬件 | GPU+MPS | GPU | GPU | GPU | GPU | **NPU（首个）** |

**B 线小结**：
1. **本质区别**：B 线五篇全是"多任务/多模型共享 GPU"，我们是"单 LLM 内 P/D 两阶段共享 NPU"——共享权重和 KV cache，**切换零开销**，这是其他方案都不具备的结构性优势
2. **可借鉴的具体机制**：TGS 的 AIMD（→ V2 phase ratio 控制）+ Dilu 的状态机 + 2D co-scaling（→ 快调 phase ratio / 慢调 KV 预算）+ FaST-GShare 的配额（→ 防 P 饿死的连续 D 上限）
3. **B 线无一涉及编译图**，进一步确认 Graph-Aware Batch Shaping 的原创性

---

### 5.3 我们的差异化定位（A 线 + B 线交叉点）

```
                A 线: 单卡 PD 混部           B 线: 通用时分复用
                (LLM serving 社区)         (调度/serverless 社区)
                       │                          │
            CP / Drift / Semi-PD          TGS / Dilu / FaST-GShare
                       │                          │
                       └────────► 交叉 ◄──────────┘
                                    │
                              NPU PD-TDM (ours)
                  把 B 线的 TDM 调度思想 (AIMD/状态机/配额)
                  搬到 A 线的 LLM P/D 场景，且首次落地到
                  NPU + 编译图约束下
```

**四个核心差异点**：
1. **PD 动态混部范畴内首个兼顾硬件普适性与迭代粒度调度灵活性的方法**——空分路径硬件依赖（A 线），时分路径现有调度粒度偏粗（A+B 两线）
2. **在 iteration 粒度上引入 SLO 余量反馈控制**——现有时分方案多采用跨多迭代生效的固定比例或启发式策略，缺乏随每次迭代更新的反馈机制
3. **唯一同时建模 TTFT + TPOT 双 SLO 的时分方法**——LLM serving 特有的双延迟指标
4. **预编译图加速器（NPU ACL Graph / GPU CUDA Graph）上 iteration 是零切换开销的天然粒度**——一次 graph replay = 一个 iteration，在此粒度切换无额外开销；本方法因此配套设计图识别控制模块以高效落地

> **汇报话术**：
> "相关工作我分两条线讲。**A 线是 LLM 圈的单卡 PD 混部**——CP（Sarathi）、Drift、Semi-PD、MuxWise、RAPID-Serve——这条线最强的几个方案都依赖 GPU 的 SM partition 或 CU masking，NPU 没有这些硬件能力，所以**空间复用整类方案被硬件封死**，剩下的可行项只有 CP 和时间复用。**B 线是通用的 GPU 时分复用**——FaST-GShare、GaiaGPU、TGS、FaaSwap、Dilu——它们解决的是多租户共享 GPU，调度思想（AIMD、状态机、配额）很有用，但都不是 LLM P/D 专用、不感知编译图、且都有切换开销。**我们位于两条线的交叉点**：把 B 线的 TDM 调度思想搬到 A 线的 LLM P/D 场景，落到 NPU + ACL Graph 这个新约束下，得到一个零切换开销、双 SLO 约束、图感知的 iteration 级调度器。"

---

## 六、问题与挑战（3 min）

### 3.1 尚未实现的核心模块
- Phase Switching V1 的具体实现和阈值确定
- Graph-Aware Batch Shaping 的调度器集成
- 离线 (batch_size, phase) → latency 画像表

### 3.2 模型规模与负载（**已根据导师反馈调整**）
- 主测模型升级为 **Qwen3-7B**（取代 Qwen3-4B）：prefill 时长更明显，P/D 互补效应更易观察
- **现阶段不必过度追求大模型 / 真实负载**（导师明确指示）：先讲清机制和增量收益
- 大模型（30B+）和真实 trace 留作后期补充实验，不作为当前阻塞项
- 当前硬件限制：910B3×2 跑 30B+ 模型 KV cache 空间极小，本身也不现实

### 3.3 Chunked Prefill 是强 baseline
- CP 已提供 +5~11% 提升，TDM 需论证增量价值
- 但 vllm-ascend 默认关闭 CP（NPU 支持可能不完善）→ 如果 NPU 上 CP 受限，TDM 价值更大

---

## 七、下一步计划（2 min）

### 短期（1-2 周）
1. **切换主测模型为 Qwen3-7B**，重跑 Exp B（资源画像）/ Exp D（并发）/ Exp E（Phased vs Unified）
2. 实现 Phase Switching V1 原型（AscendScheduler 中增加 P/D 切换）
3. End-to-end benchmark：原型 vs Unified+CP，验证增量收益
4. 离线画像表采集（Ascend Profiler）

### 中期（2-4 周）
5. V2 AIMD 自适应算法
6. Graph-Aware Batch Shaping 集成
7. ~~更大模型测试（Qwen3-30B）~~ → 导师指示**降低优先级**，不作为当前阶段目标

### 长期
7. 论文撰写
8. 向 vllm-ascend 社区提交 PR

---

## 附录：图表清单

| 图号 | 文件名 | 内容 |
|---|---|---|
| fig1 | fig1_exp_b_resource_profile.png | P/D 资源互补性 |
| fig2 | fig2_exp_d_concurrency.png | 并发 vs 吞吐/延迟 |
| fig3 | fig3_exp_a_acl_graph.png | ACL Graph shape + eager boundary |
| fig4 | fig4_exp_f_graph_aware.png | Padding 浪费 + Graph-Aware 对比 |
| fig6 | fig6_exp_e3_fair_compare.png | 4 方案 TPS 对比（2 卡公平） |
| fig7 | fig7_exp_e3_latency.png | TTFT/TPOT 延迟对比 |
| fig8 | fig8_architecture_overview.png | TDM 三层架构总览 |
