# Idea Proposal: NPU-TDM — SLO-Aware Time-Division Multiplexing for Prefill-Decode Co-serving on Single Ascend NPU

---

## 1. 问题定义与 Motivation

### 1.1 背景

LLM 推理服务的核心优化目标是 **Goodput**——即满足 SLO 约束的有效请求吞吐量（TTFT 与 TPOT 双延迟指标共同约束）。每个请求经历两个计算特征截然不同的阶段：

- **Prefill**：计算密集（compute-bound），处理完整 prompt，生成 KV cache，决定 TTFT
- **Decode**：访存密集（memory-bandwidth-bound），逐 token 自回归生成，决定 TPOT

两阶段对硬件资源的需求存在天然互补：Prefill 饱和算力但带宽有余，Decode 饱和带宽但算力空闲。如何在真实动态负载下协调两阶段的资源占用，成为 LLM 推理服务效率优化的核心议题。

### 1.2 现有方案的两类结构性局限

现有 LLM 推理服务在 P/D 资源组织上演化出三种架构范式：**耦合架构**（P/D 同 batch）、**分离架构**（P/D 跨卡物理绑定）、**动态混部**（单卡内按需协调 P/D）。其中动态混部又分为空间复用与时间复用两条子路径。

#### 问题（1）：耦合与分离架构均存在不可消除的结构性代价，在真实动态负载下难以最大化 Goodput

- **耦合架构**（Chunked Prefill / SplitFuse；代表：Sarathi-Serve OSDI'24、DeepSpeed-FastGen）：Prefill 与 Decode 共享同 batch，**同 batch 干扰无法根除**——Prefill 推高 Decode TPOT、Decode 争抢带宽影响 Prefill；即便内部有 Chunked Prefill、动态块大小等启发式调优，其"P/D 混合同 batch"的基础执行形态固定，无法在两阶段占比上做结构性重分配。
- **分离架构**（Disaggregation；代表：DistServe OSDI'24、Llumnix OSDI'24）：P/D 绑定到不同物理加速卡，通过物理隔离消除同 batch 干扰，但代价是**KV Cache 跨物理资源迁移开销**、且必然需要 ≥2 卡。内部虽有实例弹性伸缩、跨实例请求迁移等动态机制，但"P/D 跨卡物理绑定"的基础执行形态同样固定，**无法在单卡上根据动态负载调节 P/D 占比**。

两类架构的代价分别源于其基础执行形态，**无一种方法能在单卡上根据实时负载自适应调整 P/D 执行策略以最大化 Goodput**。

#### 问题（2）：PD 动态混部未能兼顾硬件普适性与真实动态负载下的调度灵活性

为突破上述局限，业界提出 **PD 动态混部** 范式——在单卡内按需协调 P/D，具体沿两条子路径发展：

- **空间复用路径**（代表：Semi-PD、DuetServe、MuxWise ASPLOS'26 2.2× goodput、RAPID-Serve 4.1× throughput）：依赖 GPU 硬件的空间分区能力（SM partition / CU masking / MPS / GreenContext）将单卡切出 P 区与 D 区。调度灵活但 **硬件依赖性强**——在无空间分区能力的加速器（如华为 Ascend NPU，缺乏 SM / CU 概念与 MPS 等价机制）上整类方案失效。
- **时间复用路径**（代表：PDM/Drift arXiv'24，及通用 TDM 工作 FaST-GShare / GaiaGPU / TGS / FaaSwap / Dilu）：P/D 在时间轴上交替占用全卡，硬件普适，但现有方案 **多采用跨多个迭代生效的固定比例或启发式策略** 安排 P/D 交替，缺乏随每次迭代实时更新的 SLO 余量反馈机制；且阶段互斥性（每一次迭代决策瞬时决定 TPOT 余量与 TTFT 累积）对调度机制的负载感知能力与反馈更新频率提出了更高要求，现有时分方案尚未满足。
- **Kernel 级 overlap**（POD-Attention ASPLOS'25）：仅在 attention kernel 内让 P/D 算子并行，**不构成独立调度层**，且在预编译图加速器（ACL Graph FULL 模式）上实测劣化 26%。

**两条路径各有所长但各有所缺**：空分路径调度灵活但硬件依赖，时分路径硬件普适但调度粒度与反馈频率偏粗；**无一种现有方法能在通用加速器上同时实现硬件普适性与迭代粒度的负载感知调度**，Goodput 仍存在明显瓶颈。

### 1.3 Motivation 总结

> 在 Ascend NPU 上，低至中并发场景下算力在 decode 阶段存在巨大浪费（实测 1→64 并发吞吐 58× 近线性扩展，TPOT 恒定，表明大量算力在 decode 间隙空闲）。我们能否选择硬件普适性更强的时间复用路径作为基础，在 iteration 粒度上引入 SLO 余量感知的反馈控制，从而在通用加速器上同时兼顾硬件普适性与迭代粒度的负载感知调度，最大化 Goodput？

---

## 2. 方法概述：NPU-TDM

### 2.1 核心思想

**Time-Division Multiplexing（TDM）**：在单张 NPU 上，通过在滑动时间窗口内动态调整 Prefill、Decode 和 Mixed 三种执行模式的比例，实现资源的时间维度复用。

**关键概念澄清**：

TDM **不是**每个 iteration 做 P/D 的二选一决策，而是**控制一段时间内 P:D:M 的执行比例**。在 vLLM 的执行模型中，1 iteration = 1 batch = 1 ACL Graph replay，iteration 是 NPU 上最小的不可拆分执行单元。TDM 以此为最小切换粒度，但调度策略看的是窗口级的比例关系：

```
时间轴示例（每个字母 = 一个 iteration）：
  D D D P D D D D P D D P P D D D D D D P D D ...
  ←─ 低负载：P 稀疏 ──→ ←─ 突发请求 ─→ ←─ 恢复 ─→

高并发场景可能退化为 Mixed（Chunked Prefill）模式：
  D D M M D D M D D D ...
```

**三种执行模式（动作空间 A = {PREFILL, DECODE, MIXED}）**：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| **PREFILL** | 纯 prefill iteration，集中处理等待队列 | 队列深、decode 空闲、低并发 |
| **DECODE** | 纯 decode iteration，全力推进已有请求 | TPOT 紧张、无新请求 |
| **MIXED** | Chunked Prefill，P/D 混合在同一 batch | 高并发、负载平稳，混合效率更高 |

将 Mixed 纳入动作空间使 TDM 成为 Chunked Prefill 的**超集**：CP 是 TDM 的一个特例（全部 iteration 都选 MIXED）。TDM 的核心价值在于根据负载动态选择最优模式组合。

**实验验证（Exp E）**：Phased（纯 P/D 分离）并非一致优于 Unified（混合执行）——低并发/长 prompt 时 Phased +10%，高并发/混合负载时 Phased -20%。这直接论证了**动态模式选择**的必要性。

区别于 chunked prefill（P/D 始终混合）和空间分离（P/D 占用不同 SM / CU），TDM 在时间轴上根据负载特征动态选择最优的 P/D 执行模式。

### 2.2 三层调度架构（主创新 + 落地配套）

方法以 **SLO 自适应控制模块** 为核心创新，**分层硬件状态采集模块** 为其提供低开销的硬件状态输入，共同实现迭代粒度的负载感知调度；同时，为使该方法在采用预编译计算图的加速器（ACL Graph / CUDA Graph）上高效落地，配套设计 **图识别控制模块** 处理 shape 对齐。三者定位：

- **SLO 自适应控制模块（核心创新）**：根据 SLO 余量与待处理请求数等实时负载指标，基于闭环反馈机制在迭代粒度上动态调节 P/D 执行比例参数，并设置防饿死硬约束保障两阶段请求均不被无限延迟。
- **分层硬件状态采集模块（核心支撑）**：离线画像 + 在线低频采样的两层架构，为 SLO 控制模块提供低开销的硬件状态输入。
- **图识别控制模块（落地配套）**：在预编译图加速器上，根据执行模式与已编译 shape 集合决定批次组成，与 SLO 控制模块双向反馈（如当前队列凑不出好 shape 时建议延迟切换）。该模块是本方法在预编译图加速器上的适配层，不在同一核心创新层级。

```
┌─────────────────────────────────────────────────────────────┐
│  核心创新：SLO 自适应控制模块（Phase Ratio Controller）        │
│  输入：SLO 余量、待处理请求数、硬件状态                       │
│  输出：下一迭代的 P/D 执行模式                                 │
│  机制：闭环反馈调节 P/D 执行比例 + 防饿死硬约束                │
├─────────────────────────────────────────────────────────────┤
│  核心支撑：分层硬件状态采集模块                                │
│  离线画像（Ascend Profiler）+ 在线低频采样（DCMI AICore）     │
│  产出：(batch_size, phase) → latency/util 画像 + 实时 AICore% │
├─────────────────────────────────────────────────────────────┤
│  落地配套：图识别控制模块（Graph-Aware Batch Shaping）        │
│  输入：SLO 控制模块选定的执行模式 + 已编译 graph shape 集合    │
│  输出：批次组成（请求集合 + 对齐后的 batch size）              │
│  反馈：队列凑不出好 shape 时建议 SLO 控制模块延迟切换           │
└─────────────────────────────────────────────────────────────┘

关键等价关系：1 iteration = 1 batch = 1 ACL Graph replay
SLO 控制决定"做什么模式"（时间维度），图识别决定"做多大"（容量维度）
```

### 2.3 Layer 1: Phase Ratio Controller（SLO-Aware）

**核心变量**：prefill 插入率 r_p（每 N 个 decode iteration 插入一个 prefill iteration）

TDM 的调度本质不是每个 iteration 的二选一决策，而是**在滑动时间窗口内控制 P:D 的执行比例**。这个比例通过 prefill 插入率 r_p 控制——r_p 越高，prefill 越频繁，TTFT 越低但 TPOT 可能恶化。

**形式化定义：**

```
目标：max Goodput  （即最大化满足 SLO 的有效请求吞吐量）
其中 Goodput = Σ I(TTFT_i ≤ SLO_TTFT ∧ TPOT_i ≤ SLO_TPOT) / T

状态空间 S = (Q, R, ttft_slack, tpot_slack, u_aicore, u_hbm_bw)
  Q: 等待队列深度（pending prefill requests）
  R: 正在 decode 的请求数
  ttft_slack: 队首请求距 TTFT SLO 的剩余时间
  tpot_slack: decode 请求中最紧迫的 TPOT 余量
  u_aicore: AICore 利用率（DCMI 采样）
  u_hbm_bw: HBM 带宽利用率（DCMI 采样）

动作空间 A = {PREFILL, DECODE, MIXED}
  PREFILL: 下一 iteration 执行纯 prefill batch
  DECODE:  下一 iteration 执行纯 decode batch
  MIXED:   下一 iteration 执行 chunked prefill（P+D 混合 batch）
```

**V1（原型）：优先级规则 + 防饿死机制**

```
切换策略（带防饿死约束）：
  1. 若 tpot_slack < T_tpot → DECODE（保 decode SLO，最高优先级）
  2. 若 Q > 0 且 ttft_slack < T_ttft → PREFILL（防 TTFT 违约）
  3. 若 Q > 0 且 R == 0 → PREFILL（无 decode 任务）
  4. 若 Q == 0 → DECODE
  5. 若 consecutive_decode > max_consecutive → PREFILL（防 prefill 饿死）
  6. 若 consecutive_prefill > max_consecutive → DECODE（防 decode 饿死）
  7. 否则：根据 u_aicore / u_hbm_bw 的互补程度动态选择

  高并发扩展：
  8. 若 R > threshold_high_concurrency 且 Q > 0 → MIXED（高并发下混合更高效）
```

V1 的局限：本质是 reactive 的（等 SLO 快违约才切换），可能震荡和滞后。

**V2（目标）：AIMD 自适应比例控制**（借鉴 TGS 拥塞控制思想）

```
核心变量：r_p = prefill 插入率（每 1/r_p 个 decode iteration 插一次 prefill）

AIMD 更新规则：
  每个时间窗口（如 W=10 iterations）检查一次：
  - 若 TPOT violation detected → r_p ← r_p × β   (β=0.5, Multiplicative Decrease)
  - 若 TPOT 达标且 Q > 0   → r_p ← r_p + α   (α=0.1, Additive Increase)
  - 若 Q == 0               → r_p ← 0          (无需 prefill)

防饿死保证：
  - r_p 有下界 r_min > 0（只要 Q > 0），保证 prefill 不被无限延迟
  - 连续 decode 有上界 max_consecutive_d，超过强制插入 prefill

模式选择：
  - 低并发（R < threshold）：使用纯 P/D 交替（TDM 主模式）
  - 高并发（R ≥ threshold）：切换为 MIXED 模式（退化为 Chunked Prefill）
  - 切换依据：实验 E 数据表明高并发下混合执行更高效
```

AIMD 的优势：不是每次做决策，而是维护一个**平滑变化的比例**，避免 reactive 切换的震荡。prefill 插入率渐进收敛到当前负载下的最优值。

**与现有工作的区别：**
- vs Sarathi-Serve：不只混合执行，而是动态选择纯 P / 纯 D / Mixed 的最优组合
- vs DistServe：不跨卡，省去 KV cache 迁移
- vs DuetServe/Semi-PD：不依赖 SM partition，适用于无 SM 概念的 NPU
- vs PDM/Drift：显式 SLO 约束驱动 + AIMD 比例控制（非 heuristic），含 Mixed 模式
- vs Chunked Prefill：CP 是 TDM 的特例（r_mixed = 1.0），TDM 动态选择最优模式

### 2.4 Layer 2: Graph-Aware Batch Shaping

**NPU 特有问题**：ACL Graph 为每种 batch size 独立编译（~1.4s/graph），未命中已编译 shape 则 fallback 到 eager mode，性能骤降。

**关键发现（源码分析 + 实验 F）**：
- ACL Graph 的 pad/eager 判断基于 `total_num_scheduled_tokens`（所有请求的 token 总数），而非请求数
- **Decode 阶段是 Graph 的主要受益者**：每请求 1 token，total_tokens = 并发请求数（通常 ≤512），在 graph 范围内
- **Prefill 阶段**：total_tokens = prompt 长度之和，长 prompt 大概率超过 max_compiled_size(512) → eager fallback；仅少量短 prompt 场景可走 graph
- 源码中 decode 有专门的 FULL mode graph 编译（`uniform_decode=True`），体现其主导地位
- **简单"向下取整"的 Graph-Aware 策略不可行**（实验 F 验证：prefill 场景 -10%~-20%），正确方向是"向上凑"——等凑够好 shape 再执行

**策略（主要面向 Decode 阶段）**：
- 离线预编译覆盖高频 batch size 的 graph
- 调度器在组 decode batch 时，**主动对齐已编译的 graph shapes**
- 允许轻微"浪费"batch capacity（padding tokens）以匹配 graph
- **Layer 1 感知 Graph shape**：切换决策考虑当前队列深度能否凑出好的 graph shape，避免低效的小 batch padding（浪费最坏 62.5%）
- Prefill 阶段：长 prompt 走 eager 模式，不受 graph 约束；少量短 prompt 可走 graph

```
GraphAwareBatchShaping(requests, phase, compiled_shapes):
  total_tokens = sum(token_count(r) for r in requests)  # decode: len(requests), prefill: sum(prompt_lens)
  if total_tokens > max_compiled_size:
    return eager_batch(requests)  # prefill 长 prompt 场景
  target_size = nearest_compiled_shape(total_tokens)
  if target_size >= total_tokens:
    return pad_batch(requests, target_size)  # 填充至已编译 shape
  else:
    return split_batch(requests, compiled_shapes)  # 拆分为多个已编译 shape
```

### 2.5 Layer 3: 分层硬件内省（Dilu 思想借鉴）

**离线画像（Ascend Profiler）**：
- 扫描 (batch_size, phase) 组合
- 建立 (batch_size, phase) → (latency, AICore%, HBM_BW%, graph_hit) 画像表
- 指导 Layer 1 的 SLO 预测和 Layer 2 的 batch size 选择

**在线监测（DCMI）**：
- HBM BW（~1.1ms/call）：每 iteration 采样 → 实时感知 decode 带宽压力
- AICore（~60ms/call）：异步线程低频采样 → 周期性监测 prefill 算力利用
- 用于 Layer 1 状态空间中的 u_aicore 和 u_hbm_bw

---

## 3. 与现有工作的系统对比

| 维度 | Sarathi | DistServe | Semi-PD | DuetServe | MuxWise | RAPID-Serve | PDM/Drift | **NPU-TDM (Ours)** |
|---|---|---|---|---|---|---|---|---|
| 设备数 | 1 | ≥2 | 1 | 1 | 1 | 1 | 1 | **1** |
| P/D 模式 | chunked混合 | 跨卡分离 | 空间partition | 自适应复用 | 空间partition | 空间partition | 时间interleave | **时分复用** |
| 调度粒度 | token/chunk | request | SM/resource | SM/batch | SM+layer-wise | CU masking | batch | **iteration** |
| SLO 建模 | heuristic | 显式 | implicit | Roofline显式 | 显式(decode优先) | 显式(中等) | soft | **显式闭环** |
| 硬件感知 | 无 | 无 | GPU SM/MPS | GPU SM | GreenContext | CU masking | 无 | **ACL Graph + DCMI** |
| 自适应 | 静态chunk | 静态分配 | profiling | 按需切换 | 动态SM+preempt | profiling驱动 | heuristic | **实时反馈闭环** |
| 目标平台 | GPU | GPU | GPU | GPU | GPU(Pascal+) | AMD/NVIDIA GPU | GPU | **Ascend NPU** |
| 代表性能 | 2.6× capacity | 7.4× requests | goodput↑ | TBT稳定 | 2.2× goodput | 4.1× throughput | goodput↑ | **待实测** |

---

## 4. 预先实验数据（Preliminary Results）

> 以下实验在单卡 Ascend 910B3（64GB HBM）上完成，模型为 Qwen3-4B，软件栈 vllm 0.11.0 + vllm-ascend 0.11.0rc1。
> 可视化图表见 `lzn-pro/result/` 目录。

### 4.1 实验 A：ACL Graph 行为分析

**目的**：量化 NPU 编译图约束对调度的影响。

**关键结果**：
- 48 种预编译 graph shape，范围 [1, 512]，编译总耗时 84.4s（一次性）
- **Padding 浪费**：prefill 小 batch 平均 23.4%，decode 平均 16.5%（最坏 62.5%，bs=3→8）
- **Eager 退化**：超过 max_compiled_size(512) 触发 eager mode，**per_token 延迟 +26%**（78ms→99ms）
- Graph 模式间切换无额外开销

**对 TDM 的意义**：
- **ACL Graph 主要作用于 decode 阶段**：decode 的 total_tokens = 并发请求数（通常 ≤512），在 graph 范围内；prefill 的 total_tokens = prompt 长度之和，长 prompt 大概率走 eager
- TDM 让 decode batch size 更可预测 → 更好匹配预编译 shape → 减少 padding 浪费
- Graph-Aware Batch Shaping 核心价值：减少 decode 阶段的 padding 浪费；prefill 阶段仅在少量短 prompt 时受益

![Exp A](result/exp_a_acl_graph.png)

### 4.2 实验 B：P/D 资源利用率画像

**目的**：量化 Prefill 和 Decode 阶段的资源互补程度，验证 TDM 的核心假设。

| Scenario | AICore% | HBM BW% | 特征 |
|---|---|---|---|
| **Prefill-heavy** | **69.8** | **15.8** | Compute-bound，高算力低带宽 |
| **Decode-heavy** | **42.6** | **25.1** | Memory-bound，高带宽较低算力 |
| Mixed | 33.9 | 26.1 | 接近 decode-heavy（decode 主导） |

**核心发现**：
- Prefill 的 AICore 利用率是 Decode 的 **1.64×**，Decode 的 HBM BW 是 Prefill 的 **1.59×**
- 两阶段资源需求**互补** → 时分复用可在时间维度上提升整体资源利用率
- Decode 占总推理时间 ~67%（10.89s vs 5.49s），其间 ~27% AICore 闲置 → TDM 可利用空间

![Exp B](result/exp_b_pd_profile.png)

### 4.3 实验 C：DCMI 在线采样开销

**目的**：验证硬件内省机制的可行性——在推理过程中做 DCMI 采样，吞吐下降是否可接受。

| Config | TPS (tok/s) | Overhead |
|---|---|---|
| No sampling (baseline) | 670.1 | 0% |
| HBM BW 2ms | 621.4 | 7.3% |
| HBM BW 5ms / 10ms / 20ms | 612-618 | **~8%（频率无关）** |
| AICore 100ms | 683.1 | **~0%** |

**核心发现**：
- HBM BW 采样开销 ~8% 且**与采样频率无关**（2ms 和 20ms 开销相同），瓶颈是 Python 线程 GIL 争用而非 DCMI 调用本身
- AICore 100ms 低频采样开销可忽略
- **推荐策略**：AICore 在线低频采样 + 离线画像表覆盖 HBM BW 信息；或用 C 扩展绕过 GIL

![Exp C](result/exp_c_dcmi_overhead.png)

### 4.4 实验 D：并发负载扩展性

**目的**：确定 TDM 的适用场景边界——什么负载下方案才有优势。

| Concurrency | TTFT(ms) | TPOT(ms) | TPS (tok/s) |
|---|---|---|---|
| 1 | 27.7 | 21.4 | 46.7 |
| 4 | 15.0 | 22.2 | 180.5 |
| 16 | 4.3 | 23.2 | 693.4 |
| 64 | 1.9 | 23.6 | 2728.5 |

**核心发现**：
- **TPOT 恒定 ~21-27ms**（1→64 并发不退化），decode batching 非常高效
- 吞吐 **58.4× 近线性扩展**（1→64），低并发时 NPU 严重 underutilized
- 低并发（1-4）仅 47-180 tok/s vs 峰值 2728 → **TDM 最佳切入场景**
- TPOT 恒定意味着在 decode 间隙插入 prefill **不会严重影响现有 decode 请求延迟**

![Exp D](result/exp_d_qps_scaling.png)

### 4.5 实验 E：PD Unified vs PD Phased 对比

**目的**：验证 P/D 分阶段执行（TDM 的极端形式：先全 P 再全 D）是否一致优于混合执行，论证自适应调度的必要性。

| Workload | Unified TPS | Phased TPS | 变化 |
|---|---|---|---|
| short_4req | 157.9 | 173.1 | **+9.6%** |
| short_16req | 759.4 | 647.3 | **-14.8%** |
| short_32req | 1460.5 | 1305.6 | **-10.6%** |
| long_4req | 185.1 | 184.4 | ~0% |
| long_16req | 629.2 | 697.3 | **+10.8%** |
| mixed_16req | 749.1 | 599.4 | **-20.0%** |

**核心发现**：
- P/D 分阶段执行**并非一致优于混合执行**：效果随 workload 变化显著（-20% ~ +10.8%）
- **低并发 / 长 prompt** 场景：phased 有优势，P/D 集中执行减少干扰
- **高并发 / 混合** 场景：phased 劣势明显，unified 的 P/D 重叠执行更高效
- **核心论点**：不是"分离一定比混合好"，而是需要一个**智能的 phase switching policy**，根据负载特征动态选择最佳策略 → 这正是 NPU-TDM 的核心价值

![Exp E](result/exp_e_pd_compare.png)

### 4.6 实验 F：Graph-Aware Batch Shaping vs Naive Batching

**目的**：量化调度器主动对齐已编译 Graph shape 的收益，验证 Graph-Aware 策略设计方向。

**关键结果**：

*精确匹配 vs Padding 吞吐对比*：精确匹配 per-req TPS 比需 padding 高 **+6.0%**（42.4 vs 40.0）

*TDM 模拟场景对比*：

| 场景 | Naive TPS | Graph-Aware TPS | Speedup |
|---|---|---|---|
| prefill_sparse (1-5) | 128.1 | 102.6 | **-19.9%** |
| prefill_moderate (5-20) | 464.8 | 418.2 | **-10.0%** |
| decode_growing (4→40) | 856.4 | 893.5 | **+4.3%** |
| decode_stable (~32) | 1252.0 | 1260.2 | **+0.7%** |

*同一 Graph Shape 下 per-token 效率差*：shape=64 满载 vs 最低载效率差 **+22.0%**

**核心发现**：
- **简单"向下取整"Graph-Aware 策略在 prefill 小 batch 场景反效果**（-10% ~ -20%）：减少请求数的代价 > padding 浪费
- **Decode 场景有正收益**（+0.7% ~ +4.3%）：大 batch 区间 shape 间距小，向下取整损失少
- **正确方向是"向上凑"**：等凑够好 shape 再切换，而非减少请求来精确匹配
- **Layer 1 和 Layer 2 耦合设计被验证**：Phase Switching 不能只看 SLO，还要考虑能否凑出好 graph shape

![Exp F](result/exp_f_graph_aware.png)

### 4.7 综合 Motivation 图

![Motivation Summary](result/motivation_summary.png)

左图展示 P/D 资源互补性，右图展示低并发下 NPU 利用率不足——两者共同支撑 TDM 的必要性。

---

## 5. 后续实验设计

### 5.1 实验环境
- **硬件**：1 × Ascend 910B3（64GB HBM）
- **软件**：CANN 8.3.RC1, PyTorch 2.7.1, vllm 0.11.0, vllm-ascend 0.11.0rc1
- **模型**：Qwen3-4B（主要），Qwen3-0.6B（辅助验证）

### 5.2 Baseline 对比
| Baseline | 说明 |
|---|---|
| vllm-ascend (原版) | 标准 continuous batching，无 P/D 调度 |
| Chunked Prefill | Sarathi-Serve 风格，P/D 混合 batch |
| Static TDM | 固定 P:D iteration 比例（如 1:5），无自适应 |
| **NPU-TDM (完整)** | 本方案：SLO-aware + Graph-aware + DCMI 反馈 |

### 5.3 评估指标
- **吞吐**：Goodput（满足 SLO 的有效吞吐）、Raw Throughput (tokens/s)
- **延迟**：P50/P99 TTFT、P50/P99 TPOT、End-to-end Latency
- **SLO 达标率**：TTFT SLO attainment、TPOT SLO attainment
- **资源利用率**：AICore utilization、HBM BW utilization
- **调度开销**：Phase 切换延迟、Graph 命中率

### 5.4 实验矩阵

**Exp 1: 不同负载强度下的 Goodput 对比**
- 负载：轻（QPS=1）、中（QPS=5）、重（QPS=20）、突发（burst）
- 指标：Goodput、SLO attainment
- 预期：NPU-TDM 在中高负载下 goodput 显著优于 baseline

**Exp 2: SLO 敏感性分析**
- 变量：TTFT SLO（50/100/200ms）、TPOT SLO（20/50/100ms）
- 指标：SLO attainment、Throughput tradeoff
- 预期：NPU-TDM 在严格 SLO 下仍保持高 attainment

**Exp 3: 消融实验**
- 逐步移除组件：SLO-aware switching / Graph-aware batching / DCMI feedback
- 验证各组件贡献

**Exp 4: Workload 特征影响**
- 变量：prompt 长度分布（短/长/混合）、output 长度分布
- 使用 ShareGPT / LMSYS-Chat 等真实 trace
- 预期：TDM 在混合 workload 下优势最明显

**Exp 5: ACL Graph 感知效果**
- 对比：有/无 graph-aware batch shaping
- 指标：Graph hit rate、Eager fallback 比例、吞吐影响

---

## 6. 预期贡献与创新点

### 6.1 学术贡献

1. **首个同时兼顾硬件普适性与迭代粒度调度灵活性的 PD 动态混部方法**
   - 针对问题（1）：耦合/分离架构在单卡上均无法根据动态负载自适应调整 P/D
   - 针对问题（2）：现有动态混部无法兼顾硬件普适性与调度灵活性
   - 选择硬件普适性更强的时间复用路径作为基础，在 iteration 粒度上引入 SLO 余量感知的反馈控制
   - 在通用加速器（含 Ascend NPU 等无空间分区硬件）上均可部署，论证 TDM 是此类加速器上可行的 P/D 动态混部路径

2. **SLO-Aware Phase Ratio Controller with Closed-Loop Control（核心创新）**
   - 针对问题（2）：现有时分方案多采用跨多个迭代生效的固定比例或启发式策略，缺乏随每次迭代实时更新的 SLO 余量反馈机制
   - 区别于 Sarathi/SplitFuse 的 heuristic chunk、PDM/Drift 跨多迭代生效的固定比例
   - 基于 DCMI 实时硬件反馈形成闭环；在迭代粒度上持续逼近当前负载下的 Goodput 最优
   - 配合防饿死硬约束保障 P/D 两阶段均不被无限延迟

3. **ACL Graph-Aware Batch Shaping**
   - 首次将编译器约束引入调度决策（MuxWise 使用 CUDA Graphs 但仅用于 decode 加速，未纳入调度；其余 11 篇核心工作均不涉及编译图约束）
   - 解决 NPU graph 编译开销（~1.4s/graph）与调度灵活性的矛盾
   - **主要面向 decode 阶段**：decode 的 total_tokens = 并发请求数（通常 ≤512），天然在 graph 范围内；prefill 的 total_tokens = prompt 长度之和，长 prompt 大概率 eager fallback
   - TDM 的天然优势：P/D 分离后 decode batch size 更可预测，graph 命中率更高
   - 实验 F 验证：简单"向下取整"策略反效果（-10%~-20%），正确方向是 Layer 1 感知 graph shape，等凑够好 shape 再切换

4. **分层硬件内省机制**
   - 离线画像（Ascend Profiler）+ 在线监测（DCMI）的分层设计
   - HBM BW 1.1ms/call 支持每 iteration 采样，AICore 60ms/call 异步低频采样
   - 借鉴 Dilu 的 multi-factor profiling 思想，适配 NPU DCMI 接口特性

### 6.2 系统贡献

- 基于 vllm-ascend 的端到端实现
- 可复现的 Ascend NPU 推理服务评估框架
- 实测数据：NPU P/D 特征画像、DCMI 开销、ACL Graph 行为

---

## 7. 潜在风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| TDM 切换开销过大 | 频繁切换抵消收益 | 设置最小连续执行 iteration 数；切换开销实测 |
| ACL Graph 预编译覆盖不足 | eager fallback 频繁 | 分析真实 workload 的 batch size 分布，优先编译高频 shape |
| DCMI 采样干扰推理 | 监测本身影响性能 | **实测**：AICore 100ms ~0% 开销；HBM BW 线程 ~8% 开销（可用离线画像替代） |
| SLO 预测不准 | 切换决策失误 | 保守策略（decode 优先）+ 离线画像辅助 |
| 与现有 GPU 方案难以直接对比 | 审稿人质疑 | 在相同 workload 下对比 NPU baseline；强调 NPU-specific 贡献 |

---

## 8. 时间规划

| 阶段 | 内容 | 时间 |
|---|---|---|
| Phase 1 | 离线 profiling + 画像表构建 | Week 1-2 |
| Phase 2 | Layer 1 基础版 TDM scheduler 实现 | Week 2-4 |
| Phase 3 | Layer 2 Graph-aware batch shaping | Week 4-5 |
| Phase 4 | Layer 3 DCMI 在线监测集成 | Week 5-6 |
| Phase 5 | 完整实验 + 论文撰写 | Week 6-10 |

---

## 9. 论文定位

- **目标会议**：系统顶会（OSDI/SOSP/ATC/EuroSys）或 AI 系统交叉（MLSys）
- **卖点**：PD 动态混部范畴内首个兼顾硬件普适性与迭代粒度调度灵活性的方法 + SLO 余量反馈控制 + Graph-aware 落地配套
- **叙事线**：
  1. 耦合/分离架构均存在结构性代价，无法在单卡上根据动态负载自适应调整 P/D → 业界转向 PD 动态混部
  2. 现有 PD 动态混部两条路径：空分调度灵活但依赖硬件（SM / CU / MPS 等），时分硬件普适但现有方案调度粒度偏粗、缺乏迭代级 SLO 反馈
  3. **无方法能同时兼顾硬件普适性与迭代粒度调度灵活性** → 本方法填补这一空白
  4. 选择硬件普适性更强的时间复用路径作为基础，在 iteration 粒度上引入 SLO 余量反馈控制；配套图识别控制以在预编译图加速器上高效落地
  5. Ascend NPU 是本方法尤其契合的落地场景（无空间分区硬件，且有 ACL Graph 预编译约束）；在 NPU 上取得显著收益
- **最接近竞争者**：DuetServe（空分路径自适应复用）、MuxWise（空分路径 + SLO）、PDM/Drift（时分路径前作）
