# 四篇 LLM Serving 论文实验设置与性能指标对比

> **论文列表**
> - **Sarathi-Serve**: "Efficient LLM Inference with Stall-Free Batching" — OSDI 2024
> - **DistServe**: "Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving" — OSDI 2024
> - **Semi-PD**: "Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage" — 预印本
> - **MuxWise**: "Towards High-Goodput LLM Serving with Prefill-decode Multiplexing" — ASPLOS 2026

---

## 一、核心思路与架构定位

| 维度 | Sarathi-Serve | DistServe | Semi-PD | MuxWise |
|------|:---:|:---:|:---:|:---:|
| 核心思路 | Chunked-prefill + stall-free batching | Prefill/Decoding 物理分离到不同 GPU | 同GPU Phase-Wise Disaggregation + Unified Storage | 同GPU SM级 Spatial Multiplexing |
| 分离粒度 | 无分离（调度级） | 跨GPU物理分离 | 同GPU进程级分离（MPS） | 同GPU SM级空间复用（GreenContext） |
| 解决的核心问题 | Prefill导致decode stall + pipeline bubble | Prefill-decoding 计算干扰，耦合导致无法独立优化并行策略 | Disaggregation 的存储不均衡 + 静态资源分配无法适应动态负载 | Disaggregation 资源闲置/冗余计算 + Chunked-prefill 的 SLO-利用率矛盾 |
| 是否消除 prefill-decode 干扰 | 缓解（通过chunking减小单次prefill长度） | **根本消除**（物理分离到不同GPU） | **根本消除**（同GPU进程级隔离） | **根本消除**（SM级硬件空间隔离） |
| KV Cache 处理 | 同GPU直接共享 | 跨GPU传输（声称 <0.1% 开销） | **Unified Storage**（prefill/decode共享同一KV pool，无需传输） | 同GPU直接共享 |
| 资源调整方式 | Token budget 调整 chunk 大小 | 离线 placement 搜索算法（分钟级，一次性） | **SLO-aware 动态调整**（实时） | **Per-iteration SM partition 动态调整** |

---

## 二、核心评估指标定义

| 指标 | Sarathi-Serve | DistServe | Semi-PD | MuxWise |
|------|:---:|:---:|:---:|:---:|
| **Prefill 延迟** | P50 TTFT | 隐含在 SLO 约束中 | P90/P99 TTFT | P99 TTFT |
| **Decode 延迟** | **P99 TBT**（主要 SLO 指标） | **TPOT** | P90/P99 **TPOT** | **P99 TBT**（强调比 TPOT 更严格） |
| **SLO 达标率** | 隐含在 capacity 定义中 | **SLO Attainment**（≥90% 为核心阈值，附录含 99%） | **SLO Attainment**（≥90%） | SLO Attainment |
| **吞吐/Capacity** | **Serving Capacity**：满足 P99 TBT SLO + 队列不爆炸（median scheduling delay < 2s）的 max req/s | **Goodput**：满足 TTFT+TPOT 双重 SLO 约束且 ≥90% 请求达标的 max per-GPU rate (req/s) | Per-GPU rate + SLO Attainment 曲线 | **Goodput**：满足 TBT SLO 的 peak throughput |
| **鲁棒性/其他** | Throughput-Latency tradeoff 曲线 | SLO Scale 鲁棒性（缩放 SLO 看最严格可承受值）、Latency Breakdown（5阶段） | Average end-to-end latency speedup、P99 TTFT/TPOT | Token throughput、GPU utilization、SLO attainment vs rate |

### 关键指标差异：TPOT vs TBT

- **TPOT (Time Per Output Token)**：所有 decode token 的平均耗时，DistServe 和 Semi-PD 使用。可能掩盖个别 token 的高延迟。
- **TBT (Time Between Tokens)**：每个 decode token 的实际耗时。Sarathi-Serve 和 MuxWise 使用，MuxWise 明确指出 TBT 比 TPOT **更严格**，因为 TPOT 平均化会隐藏尾部 token 的延迟问题。

### 关键指标差异：Goodput/Capacity 定义

- **DistServe**：TTFT + TPOT 双重 SLO 约束下，90% 请求达标时的最大 per-GPU rate。通过「SLO Attainment 曲线」上的竖线确定。
- **Sarathi-Serve**：P99 TBT SLO 约束下，median scheduling delay < 2s（确保队列不爆炸）时的最大可持续 rate。
- **MuxWise**：P99 TBT SLO 约束下的 peak throughput，强调 "goodput" 概念。

---

## 三、SLO 设定方法对比

| 论文 | SLO 设定方式 | 具体数值 |
|------|-------------|---------|
| **Sarathi-Serve** | P99 TBT = **5× (strict) / 25× (relaxed)** 倍「无 prefill 干扰下单次 decode iteration 执行时间」（prefill=4K, bs=32 条件下测得） | Mistral-7B: 0.1s/0.5s; Yi-34B: 0.2s/1s; LLaMA2-70B: 1s/5s; Falcon-180B: 1s/5s |
| **DistServe** | 基于应用场景**经验值**设定 | Chatbot: TTFT 0.25-4.0s / TPOT 0.1-0.2s; Code: TTFT 0.125s / TPOT 0.2s; Summarization: TTFT 15s / TPOT 0.15s |
| **Semi-PD** | **7.5× (tight) / 10× (loose)** 倍单请求延迟 | Llama3-8B ShareGPT: 0.3s/0.15s (tight); LongBench: 2.25s/0.13s (tight) |
| **MuxWise** | TBT SLO 直接设为固定值（引用 prior work） | **50ms** (Llama-8B) / **100ms** (Llama-70B)；TTFT 不作为硬约束 |

---

## 四、硬件环境

| 论文 | GPU | 互联 | 节点规模 |
|------|-----|------|---------|
| **Sarathi-Serve** | A100-80GB (×4/节点), A40-48GB (×8) | NVLink intra-node, 100Gbps Ethernet cross-node | Azure NC96ads v4, 最多 2 节点 |
| **DistServe** | A100-80GB SXM (×8/节点) | NVLink intra-node, **25Gbps** cross-node | 4 节点 32 GPU |
| **Semi-PD** | A100-80GB SXM4, A100-40GB SXM4, **H200-141GB** SXM5, A800-80GB | NVLink intra-node, **200Gbps** cross-node | 4 种平台，最多 4 节点（405B） |
| **MuxWise** | A100-80GB (×8), H100-SMX5-80GB (×8), **H200-SMX5-141GB** (×8) | NVLink (600 GB/s) | 单节点为主 |

### 硬件趋势

- DistServe 故意使用低跨节点带宽（25Gbps）以测试 KV cache 传输压力
- Semi-PD 和 MuxWise 已覆盖 **H200** 等最新 GPU
- MuxWise 的 NVLink 带宽最高（600 GB/s），为 SM 级空间复用提供基础

---

## 五、模型与并行策略

| 论文 | 模型 | 注意力机制 | 最大模型 | 并行策略 |
|------|------|:---:|---------|---------|
| **Sarathi-Serve** | Mistral-7B, Yi-34B, LLaMA2-70B, Falcon-180B | GQA | 180B (Dense) | TP-2, TP4-PP2 |
| **DistServe** | OPT-13B, OPT-66B, OPT-175B | **MHA**（故意选经典注意力以增大传输压力） | 175B (Dense) | intra-op=1/4/8, inter-op=3 |
| **Semi-PD** | Llama3-8B/70B/405B, DeepSeek-V2-Lite(16B), DeepSeek-V3(671B) | GQA / MLA | **671B (MoE)** | TP=1/4/8, PP=4, FP8(DeepSeek-V3) |
| **MuxWise** | Llama-8B, Llama-70B, Qwen3-235B (22B activated MoE) | GQA | 235B (MoE) | TP=8 |

### 模型选择趋势

- 早期论文（DistServe）用 OPT + MHA 来最大化传输压力
- 后续论文统一转向 **Llama 系列 + GQA**，更贴近实际部署
- Semi-PD 和 MuxWise 开始覆盖 **MoE 模型**（DeepSeek-V3, Qwen3-235B），代表最新趋势
- Semi-PD 是唯一覆盖 671B 超大 MoE 的论文

---

## 六、工作负载与数据集

### 6.1 数据集对比

| 论文 | 数据集 | Prompt 特征 | Output 特征 |
|------|--------|------------|-------------|
| **Sarathi-Serve** | openchat_sharegpt4 | median 1730, P90 5696 | median 415, P90 834 |
| | arxiv_summarization | median 7059, P90 12985 | median 208, P90 371 |
| **DistServe** | ShareGPT | avg 755 | avg 200 |
| | HumanEval | avg 171 | avg 98 |
| | LongBench | avg 1738 | avg 91 |
| **Semi-PD** | ShareGPT, LongBench, MATH-500 | ShareGPT avg ~251; LongBench 长文本 | 多样 |
| | Synthetic (95% ShareGPT + 5% irregular ~4K) | 异构混合 | — |
| **MuxWise** | Real-world Conversation & Tool&Agent traces | avg ~7.5K（远长于传统ShareGPT的226） | 多样，突发性强 |
| | ShareGPT, OpenThoughts (短入长出), LooGLE (长入短出) | 三种典型模式 | — |

### 6.2 请求到达模式

| 论文 | 到达模式 |
|------|---------|
| **Sarathi-Serve** | Poisson 分布生成 |
| **DistServe** | Poisson 分布生成（数据集无时间戳） |
| **Semi-PD** | Poisson 分布（实例级）；真实 trace 缩放（集群级） |
| **MuxWise** | **真实生产 trace**（突发性极强，up to 13× spike within 1min）+ Poisson（合成数据） |

### 6.3 工作负载趋势

- 早期使用纯 Poisson 合成数据
- MuxWise 率先使用**真实生产 trace**，有强突发性
- 工作负载日趋多样化：短对话、长文本、代码、数学、长入短出/短入长出

---

## 七、Baseline 对比

| 论文 | Baselines | 备注 |
|------|-----------|------|
| **Sarathi-Serve** | vLLM (3 batch sizes: 32/64/128), Orca | Orca 无 PagedAttention；vLLM 使用 prefill-prioritizing scheduler |
| **DistServe** | vLLM, DeepSpeed-MII | DeepSpeed-MII 无法跑 OPT-175B（kernel 限制） |
| **Semi-PD** | DistServe, vLLM (default + SplitFuse), SGLang, NVIDIA Dynamo | 覆盖 disaggregated + unified + 集群级方案 |
| **MuxWise** | Chunked-prefill (SGLang), NanoFlow, LoongServe, SGLang-PD | 覆盖 chunked-prefill + 动态disaggregation + 静态disaggregation |

---

## 八、核心性能提升汇总

### 8.1 Sarathi-Serve — Capacity 提升

| 场景 | vs Orca | vs vLLM |
|------|---------|---------|
| Yi-34B, strict SLO (openchat) | **4.0×** | **3.7×** |
| Mistral-7B, strict SLO | 2.78× | 2.15× |
| LLaMA2-70B (TP4-PP2), strict SLO | **6.3×** | **4.3×** |
| Falcon-180B (TP4-PP2), strict SLO | 5.62× | 4.60× |
| Throughput-Latency Tradeoff (Mistral-7B, strict 100ms) | — | **3.5×** |
| Falcon-180B PP (strict SLO) | — | **3.6×** |

- Chunked-prefill 开销：chunk=512 时 ~25%，chunk=2048 时可忽略
- Token budget：512 (strict) / 2048 (relaxed)
- Pipeline bubble 显著减少

### 8.2 DistServe — Goodput 提升

| 场景 | vs vLLM | vs DeepSpeed-MII |
|------|---------|-----------------|
| Chatbot OPT-13B | 2.0× rate | 1.6× rate |
| Chatbot OPT-66B | 4.6× rate | **7.4×** rate |
| Chatbot OPT-175B | ~3× rate | N/A（MII 不支持 175B） |
| Code Completion OPT-66B | **5.7×** rate, 1.4× tighter SLO | 1.6× rate, 1.4× tighter SLO |
| Summarization OPT-66B | 4.3× rate, **12.6×** tighter SLO | 1.8× rate, 2.6× tighter SLO |
| SLO 鲁棒性（1.8×–3.2× tighter vs vLLM） | — | — |

- KV Cache 传输开销：**< 0.1%** 总延迟，95% 请求 < 30ms
- Latency Breakdown：prefill execution + decoding execution 占绝大部分

### 8.3 Semi-PD — E2E Latency & SLO Attainment 提升

| 场景 | 提升 |
|------|------|
| DeepSeek-V2-Lite (ShareGPT) | **1.27–2.40×** end-to-end latency speedup |
| DeepSeek-V3 (MATH-500) | **1.49–2.58×** end-to-end latency speedup |
| SLO Attainment vs vLLM-S (Llama) | **1.55×** higher request rate |
| SLO Attainment vs vLLM-D (Llama) | **1.72×** higher request rate |
| SLO Attainment vs DistServe (Llama) | **1.62×** higher request rate |
| vs semi-PD (100,100) static baseline | **1.11×** (dynamic adjusting 带来的额外提升) |
| Cluster-scale (Dynamo, 1P3D4S vs 2P6D) | **>2× TPOT reduction** |

- 关键发现：Dynamic adjusting 使 semi-PD 在 P90/P99 的 TTFT 和 TPOT 上**均保持平稳**，而其他系统在 request rate 升高时出现延迟爆炸
- P99 指标下半-PD 优势更明显

### 8.4 MuxWise — Goodput 提升

| 场景 | 提升 |
|------|------|
| Real-world workloads, 99%-ile TTFT vs chunked-prefill | **3.57×** avg speedup |
| vs NanoFlow | **5.98×** |
| vs LoongServe | **4.65×** |
| vs SGLang-PD | **1.66×** |
| Peak goodput (avg over all SOTA baselines) | **2.20×** (up to **3.06×**) |
| H100/H200 larger models (Qwen-235B) | TTFT **2.28×**, TBT **1.81×** |
| ShareGPT synthetic | Goodput 1.9× vs chunked, 1.73× vs NanoFlow, 9.5× vs LoongServe |
| LooGLE (long input) | 1.71× vs chunked, 2× vs NanoFlow |
| OpenThoughts (short input, long output) | 2× vs chunked/NanoFlow/SGLang-PD |
| Short requests / single GPU (Llama-8B) | 1.2× vs chunked-prefill |

- Memory overhead：6.2%（CUDA Graph，6种partition配置）
- Runtime overhead：< 1.5%（layer-wise prefill launch）
- SLO attainment 在 Tool&Agent 负载下保持 >90%

---

## 九、额外开销分析

| 论文 | 开销来源 | 量化 |
|------|---------|------|
| **Sarathi-Serve** | Chunked-prefill 重复 KV-cache 读取 | ~25% overhead (chunk=512), negligible (chunk=2048) |
| **DistServe** | KV Cache 跨GPU传输 | **< 0.1%** 总延迟，95% 请求 < 30ms |
| **Semi-PD** | MPS 进程切换开销 | 声称低开销，未给出精确数字 |
| **MuxWise** | CUDA Graph 内存（6 partition × batch sizes） | **6.2%** memory; Layer-wise launch **< 1.5%** runtime |

---

## 十、Ablation Study 对比

| 论文 | Ablation 内容 | 关键结论 |
|------|-------------|---------|
| **Sarathi-Serve** | chunked-prefills-only vs hybrid-batching-only vs combined | 两者结合效果最优：单独 chunked 增加 TTFT，单独 hybrid-batching 增加 TBT |
| **DistServe** | vLLM++ (best parallelism) vs DistServe-Low vs DistServe-High | vLLM++ = vLLM（调整并行度无帮助）；Disaggregation 是关键 |
| **Semi-PD** | semi-PD(100,100) static vs semi-PD(dynamic) | Dynamic adjusting 额外提升 1.11× SLO attainment |
| **MuxWise** | Scheduling details extraction; per-component overhead | 不同workload下SM partition自动适应；layer-wise launch开销<1.5% |

---

## 十一、总结矩阵

```
                         DistServe   Sarathi-Serve   Semi-PD      MuxWise
                         (OSDI'24)   (OSDI'24)       (preprint)   (ASPLOS'26)

架构范式                  跨GPU分离    调度融合         同GPU进程分离  同GPU SM级复用
消除prefill-decode干扰     ✓(根本)      △(缓解)         ✓(根本)       ✓(根本)
无需KV传输                 ✗           ✓               ✓            ✓
资源动态调整               ✗(静态)     △(token budget) ✓(实时)       ✓(per-iteration)
大模型/PP友好              △           ✓               △            △
新硬件(H200)支持           ✗           ✗               ✓            ✓
MoE模型支持                ✗           ✗               ✓(671B)      ✓(235B)
真实生产Trace             ✗           ✗               △(部分)       ✓
主要SLO指标              TTFT+TPOT   P99 TBT         TTFT+TPOT    P99 TBT
最大性能提升               7.4× rate   6.3× capacity   2.58× e2e    3.06× goodput
SLO最严格可达             12.6× tighter —              —            —
```

### 演进趋势

1. **从跨GPU到同GPU**：DistServe (跨GPU) → Semi-PD/MuxWise (同GPU细粒度)，避免网络传输开销，同时保留消除干扰的优势
2. **从静态到动态**：资源分配从离线搜索（DistServe）演进到 per-iteration 动态调整（MuxWise）
3. **指标日趋严格**：从 TPOT（平均化）演进到 P99 TBT（单token尾部延迟）
4. **模型覆盖扩大**：从 OPT/Llama Dense 模型拓展到 DeepSeek-V3/Qwen MoE 模型
5. **负载更真实**：从纯 Poisson 合成数据到真实生产 trace（含突发流量）
6. **硬件更新**：从 A100 到 H200，支持更大显存和更新架构
