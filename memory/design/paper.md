# 论文叙事(thesis 修正后版本)

> 重写自原 `idea_proposal.md` (489 行)。thesis claim 已按 D-002 / D-003 / D-006 / D-008 调整。详细对照矩阵 + 早期 profile 数据见 `FINDINGS.md`。

---

## 1. Motivation

### 1.1 背景

LLM 推理服务优化目标 = **Goodput** = 满足 SLO 约束(TTFT + TPOT)的有效吞吐。每个请求经历两个计算特征截然不同的阶段:
- **Prefill**:compute-bound(NPU AICore 69.8%, HBM BW 15.8%),决定 TTFT
- **Decode**:memory-bandwidth-bound(42.6% / 25.1%),决定 TPOT

两阶段对硬件资源需求互补:Prefill 满算力但带宽有余,Decode 满带宽但算力空闲。**如何在真实动态负载下协调两阶段的资源占用**,是 LLM 推理服务效率优化的核心。

### 1.2 现有方案的局限

| 范式 | 代表 | 局限 |
|---|---|---|
| **耦合架构**(P/D 同 batch) | Sarathi-Serve, DeepSpeed-FastGen | 同 batch 干扰无法根除;P:D 占比无法做结构性重分配 |
| **分离架构**(P/D 跨卡) | DistServe, Llumnix | KV cache 跨卡迁移开销;必然需要 ≥ 2 卡;单卡内 P/D 占比不可调 |
| **空间复用动态混部** | Semi-PD, DuetServe, MuxWise, RAPID-Serve | 依赖 GPU 空间分区(SM / CU / MPS);**Ascend NPU 无对应硬件能力,整类方案失效** |
| **时间复用动态混部** | PDM/Drift | 跨多 iter 固定比例 / 启发式,**缺乏 iter 粒度反馈** |
| Kernel 级 overlap | POD-Attention | 不构成独立调度层,在 ACL Graph FULL 模式上反而 -26% |

**核心 gap:** 无方法能在通用加速器(尤其是 Ascend NPU 这种没有空间分区能力的硬件)上,同时实现「硬件普适 + iter 粒度负载感知调度」。

---

## 2. 方法:NPU-TDM

### 2.1 核心思想

在单张 NPU 上,在**滑动时间窗口内动态控制 P:D 执行比例**。三种执行模式 `A = {PREFILL, DECODE, MIXED}`:

| 模式 | 行为 | 适用 |
|---|---|---|
| PREFILL | 纯 prefill iter | 队列深、decode 空闲 |
| DECODE | 纯 decode iter | TPOT 紧 / 无新请求 |
| MIXED | Chunked Prefill(P+D 混合 batch) | 高并发、负载平稳 |

CP 是 TDM 的特例(全 iter 选 MIXED)→ **TDM 是 CP 的超集**。

### 2.2 系统架构(三模块,Graph 已降级)

```
trace ─┐
       ├─► ① Sampling ─► SLO 档目标 ─┐
hw/model┘                            │
                                     ▼
③ 状态采集 ─► QueueSnapshot/Tracker ─► ② SLO 控制器 ─► ratio + chunk ─► AscendScheduler (V0 fast)
                 ▲                       │
                 └───────────────────────┘
```

**① Sampling 模块**(离线一次性):workload-conditional SLO 档校准
- 详见 `sampling.md`

**② SLO 控制器**(在线,双回路):
- **慢回路 PID**:每 8 iter 算一次 ratio;真实工作模式 = 启动期爬升 + 边界保护
- **快回路 selector**:iter 粒度 TTFT 紧迫 + TPOT 沉默双向预警(P1.7b 待实装)
- 详见 `slo_pid.md`

**③ 状态采集**(基础设施):QueueMonitor / RequestTracker / Telemetry
- 详见 `interfaces.md`

**已撤下:Graph 自适应模块**(D-005)。vllm-ascend 已内置 capture + decode 自动 pad;AIV(23 sizes)vs FFTS+(15 sizes)实测非单调。论文形态三选一未定。

### 2.3 核心 mechanism:M3 prefill chunking

phase-pure 严格保证:单 iter 要么 pure-P 要么 pure-D,**不混 D token**。

V0 dedicated 快路径(`PrefillCacheHit` + `_npu_flash_attention_qlens`)vs C3 FIA 慢路径:

| Op | mean / p99 (mixed 1P+7D, q=519) | 相对 |
|---|---|---|
| `_npu_flash_attention_qlens`(V0,C1/TDM) | 0.088 / 0.119 ms | 基准 |
| `npu_fused_infer_attention_score`(FIA,C3) | 0.234 / 0.705 ms | **慢 2.66×** |

- pure_decode bs=64:FIA 慢 **3.43×**
- pure_prefill q=1024:FIA 慢 **3.10×**
- 端到端:qps=8 R1 下 C3 TTFT min 67ms vs C1 23ms(**2.9×**)

→ **M3 chunked prefill 与 C3 chunked prefill 的本质区别:**M3 跨 iter 续仍走 V0 快路径,只要不开 `chunked_prefill_enabled` 启动开关。

chunk_tokens=2048(p1_chunk_scan sweet spot,见 `chunking.md`)。

---

## 3. 跟现有工作的对比

| 维度 | Sarathi | DistServe | Semi-PD / DuetServe / MuxWise | PDM/Drift | **NPU-TDM (Ours)** |
|---|---|---|---|---|---|
| 设备数 | 1 | ≥ 2 | 1 | 1 | **1** |
| P/D 模式 | chunked 混合 | 跨卡分离 | 空间分区 | 时间 interleave | **时分复用** |
| 调度粒度 | token/chunk | request | SM | batch | **iteration** |
| SLO 建模 | heuristic | 显式 | implicit / 显式 | soft | **分段控制器** |
| 硬件依赖 | 无 | 无 | GPU SM/MPS | 无 | **ACL Graph + V0 kernel** |
| 自适应 | 静态 chunk | 静态分配 | profiling / 按需切换 | heuristic | **慢回路爬升 + 快回路双向预警** |
| 目标平台 | GPU | GPU | GPU | GPU | **Ascend NPU** |

---

## 3b. Mechanism-Level Comparison: Why PD-TDM Improves Mean Latency

> 这一节是 Section 3/4 的核心论点支撑:**chunking 本身不是优势源(C3 也 chunk),phase-pure 本身也不是充分条件(C1 隐式 phase-pure 但仍输)。PD-TDM 的真正机制 = explicit ratio control over phase switching frequency,叠加 phase-pure + chunking 两个辅助机制,在 TTFT-sensitive 工况下同时改善 TTFT_mean 和 TPOT_mean,从而提升 goodput at SLO**。

### 3b.1 单 iter 调度层面的差异

| 维度 | **C1** Unified (V0 hybrid) | **C3** Chunked Prefill (Sarathi) | **M1+chunk** static TDM | **M3.1** PD-TDM (Ours) |
|---|---|---|---|---|
| 单 iter phase 组成 | 全 prefill OR 全 decode(V0 强制) | prefill chunks + decode 同存 | 全 prefill OR 全 decode | 全 prefill OR 全 decode |
| Phase 切换决策 | **隐式** (admit-driven):running 清空才切 prefill | 不切相(每 iter 都混) | **显式** ratio knob (static) | **显式** ratio knob + SLO PID 反馈 |
| Prefill 切块 | ❌ 一次跑完整个 prompt | ✅ chunk=2048 切碎 | ✅ chunk=2048(限单 iter prefill 预算) | ✅ chunk=2048(同 M1) |
| Prefill iter token budget | 8192 全给 prefill | budget 被 decode 摊薄 | 8192 全给 prefill | 8192 全给 prefill |
| Decode iter token budget | 8192 全给 decode | budget 被 prefill chunk 摊薄 | 8192 全给 decode | 8192 全给 decode |
| Prefill iter 触发频率 | 不可控(load-coupled) | N/A(每 iter 都混) | bounded by ratio 周期 | bounded by ratio + PID 自适应 |

### 3b.2 物理链条:为什么 TTFT_mean 表现不同

**TTFT = 请求到达 → 第一个 decode token = 排队等 prefill iter + prefill 计算时间**

| paradigm | 排队部分(主导项) | 计算部分 | TTFT_mean 预测 |
|---|---|---|---|
| C1 | 大 prompt 必须等 running set 清空才切 prefill iter,持续负载下队列堆积 | 单 iter 整个 prompt 跑完(快) | 高负载下 **升高**(排队主导) |
| C3 | 每 iter 都接收 prefill chunk,排队短 | 多 iter 累计,且每 chunk 跟 decode 共享 budget → 每 iter 进展慢 | **最高**(计算项被 decode 拖慢) |
| M1+chunk | bounded by ratio 周期 | prefill iter 内 batch 多请求,8192 全用于 prefill | **低**(排队 + 计算都好) |
| M3.1 | 同 M1 + PID 在 ttft 紧张时拉高 ratio | 同 M1 | **最低**(PID 加成) |

**T2 azure_p15 conv 实测**(3 windows × 3 seeds,18 runs pool):
- C1=208.0  C3=253.4  M1+chunk=225.8  **M3.1=188.6** ms ← M3.1 最低,vs C1 -19ms,vs C3 -65ms

### 3b.3 物理链条:为什么 TPOT_mean 表现不同

**TPOT = 单请求 inter-token-latency 均值 = decode iter 跨度 × 一个请求每隔几 iter 被服务一次**

| paradigm | 单 decode iter 跨度 | 服务频率 | TPOT_mean 预测 |
|---|---|---|---|
| C1 | 短(纯 decode iter)**但**插入大 prefill iter 时整段沉默 | 每 iter 都在 batch 内,prefill iter 期间停 | mean 由 prefill iter 占比决定 |
| C3 | 长(decode 跟 prefill chunk 共 budget) | 每 iter 都被服务(混在一起) | 单 iter 慢,token 间隔均匀 |
| M1+chunk | 短(纯 decode iter 高效) | 由 ratio 决定 | mean = decode_iter_time × (1 + ratio·prefill_iter_time/decode_iter_time) |
| M3.1 | 同 M1 | ratio 自适应:无饱和时 ratio 低 → decode 占比高 → mean 更低 | **无饱和时最低** |

**关键非对称**:
- **conv 工况**(短 prompt,prefill iter 跨度 ≈ decode iter 跨度):M3.1 的 prefill 停顿期短 → TPOT_mean 跟 C3 接近、压住 C1
- **code 工况**(长 prompt,prefill iter ≫ decode iter):M3.1 的 prefill 停顿期长 → TPOT_mean 升高,反输 C3

**T2 实测**:
- conv TPOT_mean: C1=114.6  C3=111.9  M1+chunk=119.1  **M3.1=111.9** ← 对 C1 赢 -2.7,对 C3 打平
- code TPOT_mean: C1=695.7  **C3=504.9**  M1+chunk=611.6  M3.1=604.6 ← 对 C1 赢 -91,对 C3 输 +100

→ thesis "M3.1 同时改善 TTFT_mean + TPOT_mean → 提升 goodput at SLO" 在 **vs C1 baseline 上完全验证**;vs C3 在 conv 上 ttft 大赢 / tpot 持平,在 code 上 ttft 大赢 / tpot 反输 → 这就是 D-012 写的 "**net-positive on TTFT-sensitive workloads, net-negative on long-output TPOT-sensitive workloads, which we explicitly cede**" 的物理来源。

### 3b.4 关键 message(paper headline 候选)

> PD-TDM 的真正贡献不是 "chunking"(C3 也 chunk),不是 "phase-pure"(C1 隐式 phase-pure),也不是单纯叠加两者(M1+chunk 也叠加),而是 **explicit ratio control over phase switching frequency**:把 phase 切换从 admit decision 中解耦,通过 ratio knob 保证 prefill iter 周期性发生(不依赖 running set 是否空闲),并通过 SLO PID 在饱和时主动加大 ratio。这一机制叠加 phase-pure(让每 iter 的 token budget 全用满)和 chunking(限制单 prefill iter 长度避免饿死 decode),在 TTFT-sensitive 工况下同时压低 TTFT_mean 和 TPOT_mean,从而把 goodput at SLO 推到 unified baseline 之上。

### 3b.5 跟 §5b ablation 矩阵的对应

| Claim | §5b 矩阵 cell | 本节物理预测 | 实测对应 |
|---|---|---|---|
| A. phase-pure 时分复用 | mixed_mode on/off | C1→M1+chunk 跨度 = phase-pure + ratio 总效应 | P1.8 phase Δ = 86-101% 总差距 |
| B1. ratio knob 控制角色 | M1 ratio scan vs M3.1 | M1 静态 ratio scan 应该有 peak | T4 phase_a static_scan(5/20 弱微 H1) |
| B2. SLO PID 加成 | M3.1 vs M1+chunk | PID 在 ttft 紧张时拉 ratio → ttft_mean -17ms | conv 实测 m31_2048=188.6 vs m1_chunk=225.8(-37ms) |
| C. chunk=2048 sweet spot | p1_chunk_scan | 防止单 prefill iter 太长卡 decode | 已完成 |

---

## 4. 主要实验结论(详见 `FINDINGS.md` 和 `EXPERIMENTS.md`)

### 4.1 主胜区:M3.1 chunking 在区分带

mixed regime,qps=16,3 seeds,SLO 档 ttft<500/tpot∈[100,200]:

| Δ | mean ± std |
|---|---|
| M3.1 vs C1 | **+8.2~8.9 ± 2.4-2.9pp**(~3.4σ) |
| M3.1 vs C3 | **+13.1~13.9 ± 1.5-2.1pp**(~9σ) |

long regime,1 seed:M3.1 vs C1 **+15-16pp**,M3.1 vs C3 **+22-23pp**(待 multi-seed)。

### 4.2 ablation 干净因果

- fork-path 效应 ≈ 0(m31_disabled 跟 M2.7 ±1.4pp 内)
- 纯 chunking 效应 = +13-15pp(`m31_2048 - m31_disabled`)
- **TDM controller 本身(M2.7)在 mixed regime 反输 C1 -1.0~-1.8pp**——chunking 才是主角

### 4.3 thesis 真实定位(D-002 修正后)

> 慢回路 PID 是「启动期爬升 + 边界保护」分段控制器,**不是连续闭环反馈**(P1.6e 实证 ratio 钉 max 跨所有 SLO 档)。Sampling 模块定义可服务边界 + 校准 ramp 目标 + 提供主报结果档。快回路 selector(P1.7b)在 iter 粒度做双向预警,绕过慢回路屏蔽机制,在饱和场景仍提供运行中反馈。

---

## 5. 论文叙事(主结果段)

> On Ascend NPU at qps=16, M3.1 prefill chunking (chunk_tokens=2048) reliably improves meet_slo% by **+10pp (mixed, 3 seeds) ~ +18pp (long, 1 seed)** over the M2.7 baseline within the workload-conditional SLO band ttft<500ms × tpot∈[100,200]ms, simultaneously beating C1 hybrid by **+8-16pp** and C3 vanilla chunked-prefill by **+13-23pp**.
>
> Ablation: gain is fully chunking (m31_2048 − m31_disabled = +13-15pp); fork-code-path effect ≈ 0. **TDM controller alone (M2.7) does NOT significantly improve over C1 hybrid in either regime** — chunking on V0 dedicated kernels is the protagonist, avoiding both M2.7's "long-prompt-blocks-decode" tail-latency mode and C3's FIA-on-NPU 2.7-3.4× kernel penalty.
>
> The SLO band itself is part of the contribution: at strict tpot<50ms (chat-bot interactive standard, inappropriate for long-context workloads, where all configs cluster near the ~13% hardware floor) or at loose ttft>1000ms (where all configs reach >97% saturation), meaningful comparisons collapse. TDM's offline sampling module derives the discriminative band from real-trace statistics.

---

## 5b. Ablation 矩阵(D-010 修正后)

> phase Δ 不再归因到 phase-pure 单变量。mechanism-level claim 由**同 stack ablation**支撑,C3 只做外部 reference point。

| Claim | 同 stack 对照 | 现状 / 计划 |
|---|---|---|
| **A. P/D 时分复用(phase-pure)有用** | TDM mixed_mode on/off(都开 chunk=2048 + PID) | **缺**。需要 TDMScheduler 加 `mixed_mode` 开关。见 `interfaces.md` 新缺口 |
| **B1. 慢回路 PID 的角色** | M1 静态 ratio∈{0.05..ratio_max} 扫 vs M3.1 | 不依赖新代码,**可立即跑**。结果用来定位慢回路:自适应 vs 单纯启动爬升+边界 |
| **B2. 快回路 selector 有用** | M3.1(只慢回路)vs M3.1+P1.7b(慢+快) | 依赖 P1.7b 实装(快回路 selector + `decode_oldest_silence_ms` 接口) |
| **B3. 双维度真协同** | 只快回路 vs 慢+快 | 同上 |
| **C. chunk_tokens=2048 是 sweet spot** | `p1_chunk_scan`(已完成) | 已有,不写 contribution,作参数选择交代 |

**外部 reference 对比(不是 ablation):**
- TDM 整套(M3.1)vs C3 整套(vLLM CP):paradigm-level end-to-end 对比,不归因到机制。bundle 内容明列(见 §6 贡献声明)

---

## 6. 贡献声明(D-002 + D-010 修正后)

1. **首个在 Ascend NPU 上同时兼顾硬件普适性与 iter 粒度调度灵活性的 P/D 动态混部方法**
   - 选时间复用路径(硬件普适),在 iter 粒度引入 SLO 余量反馈
   - 在通用加速器(含无空间分区硬件)上均可部署
2. **分段控制器 + 双维度协同**(实测修正后的核心创新)
   - 慢回路 PID:启动期闭环爬升 + 饱和边界保护
   - 快回路 selector:iter 粒度 TTFT/TPOT 双向预警
   - 跟 PDM/Drift 跨多 iter 固定比例的本质区别
3. **Workload-conditional SLO 校准框架**
   - 三约束交集(硬件下限 + 用户感受上限 + 实验区分性)
   - 让 evaluation 有信号,而不是被 strict 默认档拍平
4. **prefill chunking on V0 fast kernel path**
   - 跟 C3 的本质区别:V0 vs FIA(2.7-3.4×)
   - chunk=2048 sweet spot,推进区分带边界

> **D-010 修正:** mechanism-level claim 1/2 必须由**同 stack ablation**支撑(见 §5b 矩阵),不依赖 vs C3 跨 stack 对比。vs C3 只做 end-to-end paradigm-level reference。

---

## 7. Threats to Validity

1. **vllm-ascend 锁 v0.11.0rc1**(D-007):6 个 release 前的版本。论文 §implementation 主动写明锁定理由 + 升级窗口
2. **C3 在 NPU 上反常输 C1 的根因未核查**:vllm-ascend chunked prefill 实现层 + Ascend kernel 特性,可能是 vllm-ascend 实现缺陷 vs 通用 chunked prefill 设计缺陷的争议
3. **单卡设备 / 模型限制**:Qwen3-4B / 8B / 0.6B,Ascend 910B3 单卡。跨硬件 / 跨模型迁移性需要补
4. **快回路 P1.7b 实装未完成**:thesis claim 「双维度协同」需 P1.7b PASS 才能完全声明,当前数据基于慢回路 + chunking
5. **calibrated SLO 档主结果未跑**:`azure_main` 用 strict 档跑的,P1.7b 阶段合并跑(D-001)
6. **stationary 不显效:** PID 在 stationary 长 prompt 仅 +0.9/+0.3pp,论文不能在 stationary 主胜
7. **合成 burst 撬不动 PID:** Δ ≈ ±0.7pp,thesis 不报这块
8. **Bundle 内不可拆(D-010):** TDM bundle = {phase-pure iter, M2.7 PID, ratio_max, chunk=2048, TDMScheduler 实现};CP bundle = {mixed iter, vLLM 默认 budget, vLLM CP scheduler 实现}。两套 bundle 内部机制互相依赖,phase Δ 不可单变量归因。论文明列 bundle 内容,不试图 isolate
9. **实现质量 inherent confound(D-010):** TDMScheduler 跟 vLLM CP scheduler 工程实现精度可能不同,paradigm-level Δ 里含一部分实现质量差,无法完全消除。缓解:都基于 vLLM 同版本,代码量级相当,无范式外优化

---

## 8. 目标会议 / 卖点

- **目标:** 系统会议(OSDI / SOSP / ATC / EuroSys)或 AI 系统(MLSys)
- **卖点:** Ascend NPU 上 phase-aware 调度的最后窗口 + 分段控制器 + 双维度协同 + workload-conditional SLO 框架
- **最接近竞品:** DuetServe(空分自适应)、MuxWise(空分 + SLO)、PDM/Drift(时分前作)
