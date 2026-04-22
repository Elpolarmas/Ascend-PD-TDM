# 论文精读模板

---

## Paper: Semi-PD (Hong et al., arXiv'25)

### 1. 调度机制
- **P/D 复用方式：**  
  基于多进程服务MPS在单 GPU 内进行 prefill 与 decode 的并行复用，通过将 GPU 资源（SM、memory bandwidth 等）在两类任务之间进行**空间划分 + 共享执行**，而不是完全串行或跨 GPU 分离。

- **调度粒度：**  
  - 以 **kernel / micro-batch / SM 资源粒度** 为主  
  - 属于 intra-GPU 的 fine-grained resource allocation  
  - 不同于 token-level 调度，更偏资源层复用

- **切换策略：**  
  - 不依赖频繁的 time-slicing 切换  
  - 采用**动态资源分配（resource re-partitioning）**  
  - 根据 workload（prefill vs decode 比例）调整资源占比  
  - 本质是“并行执行 + 资源重分配”，而不是阶段切换

---


### 2. SLO 建模
- **优化目标：**  
  - 提升整体吞吐（goodput）  
  - 同时控制 latency（TTFT + TPOT）

- **TTFT / TPOT 建模方式：**  
  - TTFT：由 prefill 阶段主导  
  - TPOT：由 decode 阶段主导  
  - 通过资源隔离减少 prefill 对 decode 的干扰  
  - 未采用严格解析模型，主要基于 profiling + 系统经验

- **约束形式：**  
  - 无严格数学优化约束（非显式优化问题）  
  - SLO 通过系统调优间接满足  
  - 属于 **implicit SLO-aware**（弱形式）

---

### 3. 关键设计选择 & 理由
- **为什么选这种方案：**
  - Prefill 与 Decode 计算特性差异明显：
    - Prefill：计算密集 + 长序列
    - Decode：延迟敏感 + 逐 token 生成
  - 传统 PD 分离存在：
    - KV cache 迁移成本
    - GPU 利用率下降
  - Semi-PD 选择：
    - 单卡内 disaggregation
    - 资源隔离 + 共享执行

---

- **主要 trade-off：**

  1. **Isolation vs Utilization**
     - 强隔离 → latency 更稳定  
     - 弱隔离 → 利用率更高  

  2. **Flexibility vs Complexity**
     - 动态 partition 提高适应性  
     - 但调度复杂度上升  

  3. **Memory vs Compute contention**
     - KV cache 与 compute 争抢带宽  
     - 需要 careful partition  

  4. **Throughput vs Tail Latency**
     - 提高吞吐可能影响 tail latency  

---

### 4. 实验设置
- **硬件：**
  - 数据中心级 GPU（如 NVIDIA A100 或同级别）
  - 重点评估单 GPU / 单机场景

- **模型：**
  - Transformer-based LLM（如 LLaMA 系列）
  - 多规模模型（small / medium / large）

- **workload：**
  - 混合请求：
    - 不同 prompt length
    - 不同 decode length
  - 模拟真实服务负载（可能包含 burst / 随机到达）

- **baseline：**
  - 标准 batching serving（如 vLLM 类系统）
  - PD 分离系统（如 DistServe）
  - naive co-location（无隔离）

---


### 5. 核心实验结论（2-3 个关键数据点）
- 在混合 workload 下：
  - **吞吐（goodput）显著提升**
  - 同时保持或改善 TTFT / TPOT

- 相比 PD 分离：
  - **减少 KV cache 迁移开销**
  - 提高单 GPU 利用率

- 在高负载场景：
  - 能稳定 decode latency  
  - 减少 prefill 对 decode 的干扰  

---

### 6. 局限性 / 没解决的问题
- **论文自述：**
  - 资源 partition 依赖 heuristic / profiling  
  - 对 workload 变化适应性有限  
  - 极端负载下难以同时满足所有 SLO  
  - intra-GPU 仍存在资源竞争

---

- **个人判断：**

  1. **缺乏显式 SLO 控制**
     - 未形成严格的 SLO 优化闭环  

  2. **调度粒度仍偏粗**
     - 非 token-level / time-slicing 调度  

  3. **缺少全局调度视角**
     - 未涉及 cluster-level 调度  

  4. **动态自适应能力有限**
     - partition 策略需要人工调优  

---

## Paper: Nexus (Shi et al., arXiv'25)

---

### 1. 调度机制

- **P/D 复用方式：**  
  基于单卡时空复用机制，在单个 GPU 引擎内部实现 Prefill 和 Decode 的解耦（intra-GPU disaggregation）。通过并发执行共享 GPU 资源，避免了跨卡（跨实例）分离所带来的通信与硬件开销

- **调度粒度：**  
  - 属于 intra-GPU 的细粒度资源划分  
  - 调度维度不仅包括计算能力（Compute capacity），还联合划分了内存占用（Memory footprint）和内存带宽争抢（Bandwidth contention）

- **切换策略：**  
  - 采用主动式（Proactive）动态资源分配，区别于以往依赖被动反馈（reactive feedback loops）或离线启发式规则的方法
  - 实时评估系统状态，预测 workload 的资源冲突，主动调整 Prefill 和 Decode 的资源配比以适应动态变化的请求

---

### 2. SLO 建模

- **优化目标：**  
  - 在提升整体吞吐量（Throughput）的同时，严格优化延迟指标（显著降低 TTFT 和 TBT/TPOT）

- **TTFT / TPOT 建模方式：**  
  - 建立了一个轻量级解析代价模型（Analytical cost model） 
  - 将延迟（Latency）显式建模为资源分配比例、prompt 长度和 KV cache 使用量的函数
  - 核心洞察：GPU 资源存在“边际收益递减（diminishing returns）”特性。当某阶段资源分配超过饱和点后，继续增加资源对降低延迟的贡献微乎其微  

- **约束形式：**  
  - 属于 explicit / proactive SLO-aware  
  - 通过解析模型前置预测性能，主动计算出能够满足 TTFT 和 TBT 约束的最优资源划分点，形成闭环  

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**
  - Chunked Prefill 虽然能把预填充和解码混合组 batch 提升利用率，但会导致严重的细粒度阶段干扰（phase interference） 
  - 传统的跨卡 PD 分离（Engine-level disaggregation）虽然能物理隔离干扰，但需要成倍的硬件成本，且面临高昂的 KV cache 传输开销
  - 利用单卡内的解耦架构，结合边际收益递减规律，将多余资源精准剥离给另一阶段

- **主要 trade-off：**

  1. **Diminishing Returns vs. Resource Allocation**
     - 不再追求单一阶段占满 100% 资源，而是寻找“饱和点”，将溢出算力分配给另一任务  

  2. **Compute vs. Memory Bandwidth**
     - LLM 推理不仅是算力竞争，更是带宽争抢。Nexus 必须在资源划分时同时 trade-off 这两个维度的配比 
---

### 4. 实验设置

- **硬件：** 商用数据中心级 GPU（Commodity GPUs，无需定制 kernel 或特殊硬件支持）  
- **模型：** 主流 Decoder-only Transformer LLM  
- **workload：** 多样化的生产规模负载（production-scale LLMs and traffic），涵盖动态到达和变化的 prompt/decode 长度  
- **baseline：** 标准 vLLM（包含 Chunked Prefill） / SGLang / 跨卡分离部署架构（vLLM-disaggregation）

---

### 5. 核心实验结论（2-3 个关键数据点）

- 相比基线系统（单卡场景）：吞吐量（Throughput）相比 vLLM 最高提升 2.2倍，相比 SGLang 最高提升 2倍 / 延迟大幅下降：TTFT 相比 vLLM 降低最高达 20倍，TBT 降低 2.5倍
- 相比分离式架构（跨卡场景）：在仅使用一半 GPU 资源的情况下，吞吐量依然比 vLLM-disaggregation 高出 1.4倍，证明了 intra-GPU 高效共享的资源优势

---

### 6. 局限性 / 未解决问题

- **论文自述：** 极度依赖解析代价模型（Analytical cost model）的准确性，对于架构变动较大的新模型（如复杂的 MoE 结构）可能需要重新拟合或调整模型参数 
- **个人判断：**
  1. **单卡显存天花板（Memory Wall）**
     - 虽然解决了计算和带宽的动态划分，但 Prefill 和 Decode 都在单卡内，意味着它们必须共享同一张卡的 KV Cache 池。在超长上下文（Long Context）的高并发场景下，单卡显存容量极易成为硬瓶颈 

  2. **多机多卡集群调度的协同**
     - 论文聚焦在单机引擎内部（Intra-engine）的精细化控制。如果放大到 Data Center 级别，这种细粒度的单机 partition 如何与全局的请求路由（Global Router）策略打通配合，尚未深入探讨  
  
  3. **异构硬件的泛化难度**
     - 该机制利用了特定 GPU 架构下的资源饱和特性，如果迁移到内存层级和计算单元完全不同的 NPU 架构上，这种调优策略与代价模型可能面临较大重构  

---

## Paper: DuetServe (Gao et al., arXiv'25)

---

### 1. 调度机制

- **P/D 复用方式：**  
  - 本质是一种**单卡内时空复用（spatial-temporal multiplexing on a single hardware card）**的方案
  - 在单个 GPU 内部实现了“解耦级别”（disaggregation-level）的隔离机制，兼具资源共享和性能隔离的优势

- **调度粒度：**  
  - SM (Streaming Multiprocessor) 级别进行极细粒度的动态空间划分 

- **切换策略：**  
  - 采用**自适应按需激活（Adaptive & On-demand multiplexing）**策略 
  - 默认情况下，系统在聚合模式（aggregated mode）下运行以最大化硬件利用率
  - 只有当预测到资源抢占会威胁到延迟目标时，才会动态切分 SM 给不同的执行阶段
  - 为了极速切换，设计了无中断执行引擎（interruption-free execution engine），直接消除了传统的 CPU-GPU 同步调度开销

---

### 2. SLO 建模

- **优化目标：**  
  - 在严守延迟 SLO（特别是极其敏感的 TBT）的硬性约束下，最大化系统总体吞吐量（Throughput）

- **TTFT / TPOT 建模方式：**  
  - 引入了一个极具特色的核心组件：注意力感知 Roofline 解析模型（Attention-aware roofline analytical model）  
  - 该模型向下深钻到算子级别（operator-level），综合计算密集度和访存特征，能够非常精准地预测单次迭代的延迟（iteration latency）

- **约束形式：**  
  - 属于 前瞻显式约束（Proactive / Explicit SLO-aware）
  - 调度器不依赖事后补偿，而是通过 Roofline 模型提前“预见”潜在的 TBT 违规（TBT degradation）。一旦报警，**划分优化器（Partitioning Optimizer）**会立刻介入，求解出当前状态下能满足延迟的最优 SM 切分比例

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**
  - 传统混合组批（Hybrid/Chunked Batching）的痛点： Prefill 像“推土机”（算力密集），Decode 像“流水线”（访存敏感、延迟敏感）。强行捏合往往导致长 Prefill 请求引发严重的细粒度资源干扰，导致 Decode 的 TBT 出现灾难性毛刺 
  - 跨卡分离的痛点： 物理级别的 P/D 分离虽然隔离彻底，但跨节点/跨卡的 KV Cache 传输带来了极高的网络通信开销和硬件成本
  -  DuetServe 的选择： 采用单卡内的时空解耦，这既享受了共享显存带来的零传输开销红利，又在关键时刻提供了物理级（SM 级）的性能隔离

- **主要 trade-off：**
  1. **聚合执行（利用率） vs. 空间隔离（保延迟）**
     - 默认聚合能把 GPU 算力吃干抹净，但风险极高；强行隔离则会导致算力碎片化。DuetServe 的解法是“平时聚合，危急时隔离”，在刀尖上平衡了这一矛盾 

  2. **调度灵活性 vs. 引擎开销**
     - 要在微秒/毫秒级别动态重划 SM 比例，传统的 Host 端发起 Kernel 的方式开销太大。引入无中断执行引擎就是牺牲了部分系统实现复杂度，换取了极致的 context switch 效率  

---

### 4. 实验设置

- **硬件：** GPU（A100 类）  
- **模型：** LLaMA / GPT 类  
- **workload：** 模拟真实世界中具有高度变动性的生产级请求流（动态的 Prompt 长度与生成的 Decode 长度） /   重点施压场景：突发的长文本 Prefill 请求混合大量的实时 Decode 任务
- **baseline：** 传统的聚合服务框架（如包含 Chunked Prefill 优化的系统） / 跨卡/跨实例的 P/D 分离架构 /   其他固定的单卡资源划分方案

---

### 5. 核心实验结论

- 极限吞吐突破： 在受到严格 TBT SLO（例如限制 99 分位延迟毛刺）的强约束下，系统提供的有效吞吐量远超传统基于时间切片或静态分配的基线系统  
- 抗干扰能力强： 当负载中突然混入极端超长文本（引发计算洪峰）时，得益于提前预测和秒级的 SM 切分，Decode 请求的 TBT 曲线依然保持平滑，未出现显著退化（degradation）  
- 隐形切换代价： 证明了其动态重新划分 SM 以及引擎流转的代价极低，几乎不会对 GPU 的整体 Goodput 造成实质性损耗 

---

### 6. 局限性 / 未解决问题

- **论文自述：** 极度依赖 Roofline 代价模型的准确率，如果预测失准，会导致 SM 划分比例错误，进而引发严重的 SLO 违规或算力浪费
- **个人判断：**
  1. **底层物理资源的“隐性暗战”**
     - 虽然在计算单元（SM）层面实现了精确隔离，但同一张卡内的 L2 Cache 和 Global Memory Bandwidth 依然是共享的。在极端的“双高”并发下（大量高并发 Decode + 超长上下文 Prefill），底层访存通道的物理拥堵依然会引发不可预测的延迟抖动 

  2. **模型泛化与解耦代价**
     - 基于“算子级特征”构建的 Roofline 模型与当前的 Attention 机制强绑定。如果底层模型结构发生巨变（如采用更复杂的 MoE 路由结构或非 Transformer 架构的线性 Attention），这一套代价评估模型可能需要全盘推翻重写 
  
  3. **单卡显存天花板依然存在**
     - 既然是单卡时空复用，所有的 KV Cache 必须硬塞在同一张卡的显存池里。随着超长上下文时代的到来，即便 SM 调度再完美，显存容量（Memory Capacity）会先一步成为系统崩溃的硬瓶颈  

---

## Paper: MuxWise (arXiv'26)

---

### 1. 调度机制

- **P/D 复用方式：**  
  - 提出 intra-GPU prefill-decode (PD) multiplexing 新范式：在同一 GPU 内通过空间划分（different SMs）让 prefill 和 decode 并行执行，同时共享单个 KV cache pool（解耦 compute 与 memory management）。避免了跨 GPU disaggregation 的 KV cache 迁移/碎片问题，以及 chunked-prefill 的 SLO-利用率两难

- **调度粒度：**  
  - 以 SM 资源 + layer-wise (for prefill) / graph-level (for decode) 为主
  - 使用 GreenContext 实现低开销 intra-process 空间复用
  - Prefill 被拆分成 layers (PLs) 以对齐执行时间，实现 bubble-less multiplexing；decode 使用 CUDA graphs
  - 属于 fine-grained intra-GPU resource partitioning，支持动态 SM 再分配

- **切换策略：**  
  - 采用 动态 SM 资源分区（re-partitioning） 而非频繁 time-slicing
  - SLO-aware dispatcher 根据当前请求模式保留“just-enough” SMs 给 decode（满足 TBT SLO），剩余分配给 prefill
  - Query-based synchronization（polling CUDA events）实现异步合并，无阻塞
  - 支持可选 non-recursive preemption（短请求可打断长 prefill），开销低（microseconds 级 reconfiguration

---

### 2. SLO 建模

- **优化目标：**  
  - 在严格满足SLOs 的前提下，最大化系统有效吞吐（High-Goodput）。

- **TTFT / TPOT 建模方式：**  
  - TTFT 由 prefill 阶段决定，TBT 由 decode 主导
  - Contention-tolerant estimator：offline solo-run predictor（计算复杂度建模，偏差 <9%） + contention guard（通过 grid sampling 捕获带宽争用，最大 slowdown ~30%）
  - 提供 worst-case 估计，优先严格保证 decode SLO，prefill SLO 通过剩余资源 + preemption 间接满足

- **约束形式：**  
  - Explicit SLO-aware（强形式）：dispatcher 基于 worst-case 估计精确保留 SMs 给 decode
  - Prefill SLO 非硬保证（超载时会 violation，表示达到容量上限）
  - 形成闭环：profiling-based 估计 → 动态分配 → preemption 保障

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**
  - Prefill（compute-intensive，长序列）与 Decode（memory-bound，延迟敏感）特性差异大
  - 传统方案痛点：PD disaggregation 导致 KV cache 碎片/迁移开销 + 资源闲置；chunked-prefill 在小 chunk 时利用率低、大 chunk 时违反 TBT SLO
  - MuxWise 选择 intra-GPU spatial multiplexing：动态适应 compute 分配、解耦 compute/memory、独立执行两阶段，实现高利用率 + SLO 保障

- **主要 trade-off：**
  1. **SLO 严格性 vs. Utilization”**
     - 强 decode SLO 保证（保留 just-enough SMs）可能略微保守，但远优于 chunked-prefill 的两难和 disaggregation 的碎片问题 

  2. **Compute Flexibility vs. Contention**
     - 动态 SM 分配适应性强，但引入带宽争用（用 contention guard 缓解，可能导致一定 under-utilization）
  
  3. **Preemption Benefit vs. Complexity**
     - 改善短请求 TTFT，但增加调度复杂度（non-recursive 设计避免级联 violation
  
  4. **Intra-GPU 效率 vs. 全局视角**
     - 单 GPU 内高 goodput，但大规模部署仍可与 disaggregation 互补

---

### 4. 实验设置

- **硬件：** 8× A100-80GB（NVLink），额外验证 8× H100-SXM5-80GB 和 8× H200-SXM5-141GB
- **模型：** Llama-8B、Llama-70B、Qwen3-235B（MoE，激活 22B）
- **workload：** 真实 trace（ShareGPT Conversation、Tool&Agent、LooGLE、OpenThoughts），含 bursty 到达（峰值 13×）/ 合成 workload：不同 input/output 长度组合（moderate、long-input/short-output、short-input/long-output）
- **baseline：** Chunked-prefill 系统（SGLang、NanoFlow、LoongServe） / PD disaggregation（SGLang-PD、static disaggregation）

---

### 5. 核心实验结论

- 在真实混合 workload 下（Llama-8B/70B）：goodput 平均提升 2.20×（最高 3.06×），同时严格满足 TBT SLO（chunked-prefill 和 NanoFlow 常违反）
- 99%-ile TTFT 显著改善：相比 chunked-prefill 提升 3.57×，NanoFlow 5.98×，SGLang-PD 1.66×；GPU 利用率从 ~63-66% 提升至 ~84-88%
- 相比 PD disaggregation：避免 KV cache 碎片（hit rate 优势明显），token throughput 大幅领先，同时消除迁移开销

---

### 6. 局限性 / 未解决问题
  
- **论文自述：**
  - Prefill SLO 非硬保证，violation 表示系统已达峰值容量
  - Contention guard 基于粗粒度 offline profiling（~7K samples，12 小时），偏保守可能限制极致利用率
  - 存在少量内存开销（GreenContext 4MB，CUDA graphs ~6.2%）和 layer-wise launch 开销（<1.5%）
  - 主要针对单实例/单机优化；依赖支持 intra-process spatial sharing 的 GPU（Pascal+），不适用于纯 time-sharing 或 inter-process（如 MPS/MIG）
- **个人判断：**
  1. **SLO 控制较强但非完美闭环**
     - Decode 严格保证，prefill 仍偏 indirect；极端 burst 下可能需人工/上层干预 

  2. **调度粒度精细但仍有 bubbles**
     - Bubble-less 设计有效，但 fine-grained 调度下 bubble ratio 略高于某些 chunked 方法（好在不影响 goodput）
  
  3. **缺少 cluster-level 全局调度**
     - 聚焦 intra-GPU multiplexing，可与 disaggregation 互补，但未覆盖大规模多机协调
  
  4. **自适应能力依赖 profiling**
     - Estimator 需 offline 准备，动态性好但对全新 workload/硬件的适配仍需一定成本

---

## Paper: POD-Attention (Kamath et al., arXiv'24 / ASPLOS'25)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 POD-Attention custom kernel：在单 GPU 内通过 hybrid batch attention 实现 full prefill-decode overlap，让 prefill 和 decode 在同一 kernel 中并发执行。

- **调度粒度：**  
  - 以 Kernel-level resource allocation 为主  
  - 同时利用 compute（prefill）和 memory bandwidth（decode）

- **切换策略：**  
  - Minimal interference 的 concurrent execution  
  - 无需 batch 分离或显式切换，直接在 kernel 内 overlap

---

### 2. SLO 建模

- **优化目标：**  
  - 更高 throughput + 更低 latency，尤其在 attention 阶段。

- **TTFT / TPOT 建模方式：**  
  - Overlap 同时改善 TTFT 和 TPOT，无额外调度开销  
  - 通过 kernel 内在机制减少 interference

- **约束形式：**  
  - Kernel 内在保障 low interference（弱形式）  
  - 依赖上层系统配合

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 现有 overlap 方法（chunked-prefill 等）不充分，无法同时最大化 compute 和 bandwidth  
  - POD-Attention 针对 prefill（compute-heavy）和 decode（memory-heavy）的资源差异，设计专用 kernel 实现 true full overlap

- **主要 trade-off：**  

  1. **Kernel complexity vs. Speedup**  
     - 实现复杂度高，但 attention 加速显著  

  2. **Full overlap vs. Hardware generality**  
     - 最大化利用率，但需特定 GPU 支持  

  3. **Attention-only vs. Full system**  
     - 专注 attention 瓶颈，可与 Sarathi-Serve 等集成  

  4. **Throughput vs. Implementation effort**

---

### 4. 实验设置

- **硬件：**  
  - Standard GPUs（NVIDIA 系列）

- **模型：**  
  - 多种 LLM

- **workload：**  
  - Offline 和 online inference 混合场景

- **baseline：**  
  - Sarathi-Serve 等上层系统（集成后进一步提升）

---

### 5. 核心实验结论

- Attention computation speedup 最高 **59%**（平均 **28%**）  
- 整体 throughput 提升高达 **22%**  
- 与 Sarathi-Serve 集成后 high-throughput low-latency 表现进一步改善

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - Kernel-specific 实现，对不同 hardware/attention variants 的泛化需额外工作  
  - 主要优化 attention 阶段，其他瓶颈仍需上层配合  

- **个人判断：**  

  1. **Kernel-level 突破**  
     - 首次实现 true full prefill-decode overlap  

  2. **SLO 控制间接**  
     - 通过加速间接改善 latency  

  3. **与上层系统互补性极强**  
     - 是 Sarathi-Serve 等的最佳底层增强  

  4. **未来潜力大**  
     - 可进一步扩展到其他 operators

---

## Paper: Sarathi-Serve (Kwon et al., OSDI'24)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 chunked-prefill + piggybacking：在单 GPU 内将长 prefill 拆分成 near-equal sized chunks，与 decode tokens 交织执行，实现 stall-free 的时间复用。

- **调度粒度：**  
  - 以 Iteration-level + uniform chunk sizes 为主  
  - Decode-maximal batching：每个 batch 包含一个 prefill chunk + 尽可能多的 decode tokens

- **切换策略：**  
  - Stall-free scheduling，piggyback decode slots 填充 bubble  
  - 每次 iteration 结束时动态调整 batch 组成，无需暂停 ongoing decodes

---

### 2. SLO 建模

- **优化目标：**  
  - 平衡 throughput-latency tradeoff，在满足 latency SLO 的前提下最大化 serving capacity。

- **TTFT / TPOT 建模方式：**  
  - TTFT 由 chunk size 控制，TPOT/TBT 通过 stall-free 保障  
  - 隐式优化两阶段 tradeoff

- **约束形式：**  
  - SLO-aware chunk sizing（中等强度）  
  - 通过调度策略自然满足 TBT SLO

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 传统 continuous batching 存在大量 bubble 和 SLO violation；纯 chunked-prefill 小 chunk 时利用率低、大 chunk 时违反 TBT  
  - Sarathi-Serve 通过 chunked-prefill + decode-maximal batching 实现 compute saturation，同时保持低 latency

- **主要 trade-off：**  

  1. **Chunk size vs. Utilization**  
     - 优化 sweet spot，避免传统两难  

  2. **Stall-free vs. Residual interference**  
     - 显著减少 bubble，但仍有一定 interference  

  3. **Single-GPU efficiency vs. Kernel support**  
     - 无需 disaggregation，部署简单  

  4. **Throughput vs. Tail latency**  
     - 大幅提升 serving capacity

---

### 4. 实验设置

- **硬件：**  
  - A100/H100 等，单 GPU 和 pipeline parallelism 配置

- **模型：**  
  - Mistral-7B、Yi-34B、Falcon-180B 等多种 LLM

- **workload：**  
  - 混合 interactive traces，包含不同 prompt/output length

- **baseline：**  
  - vLLM、Orca 等标准 continuous batching 系统

---

### 5. 核心实验结论

- Serving capacity 单 GPU 最高提升 **2.6×**（Mistral-7B），pipeline parallelism 下高达 **5.6×**（Falcon-180B）  
- 在保持低 tail latency 的同时显著提升 throughput  
- 有效 taming throughput-latency tradeoff

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - Chunking 仍引入一定 residual interference  
  - Chunk size tuning 对 workload 有一定敏感性  

- **个人判断：**  

  1. **Tradeoff 优化经典**  
     - 是 chunked-prefill 的最优实现之一  

  2. **SLO 控制实用**  
     - 但非 explicit worst-case 建模  

  3. **单实例高效**  
     - 可与 disaggregation 或 kernel-level（如 POD-Attention）结合  

  4. **实际部署价值高**

---

## Paper: DistServe (Zhong et al., OSDI'24)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 **Prefill-Decode disaggregation**：将 prefill 和 decode 计算完全分离到不同 GPU 实例上运行，实现跨实例的 phase-level disaggregation，存储（KV cache）通过高效通信在实例间传递。

- **调度粒度：**  
  - 以 Phase-level + instance-level 为主  
  - Prefill 实例和 decode 实例独立调度  
  - 属于 cross-GPU global scheduling

- **切换策略：**  
  - 采用 bandwidth-aware placement 和 online scheduling optimization  
  - 动态 routing 请求到合适实例，无需频繁 time-slicing  
  - KV cache 通过高效通信自动管理，实现“并行执行 + 异步传输”

---

### 2. SLO 建模

- **优化目标：**  
  - 在严格满足 TTFT 和 TPOT SLO 的前提下，最大化系统 goodput（有效吞吐）。

- **TTFT / TPOT 建模方式：**  
  - TTFT 由 prefill 实例主导（专注 compute-bound），TPOT 由 decode 实例主导（专注 memory-bound）  
  - 通过 simulator-driven search 量化资源需求和通信开销

- **约束形式：**  
  - Explicit SLO-aware（强形式）：co-optimize resource allocation 和 parallelism strategy  
  - 形成闭环：SLO 要求 → placement 优化 → online scheduling

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 传统 co-located serving 中 prefill 和 decode 严重相互干扰，导致 latency variance 大；全统一调度无法分别优化两阶段特性  
  - DistServe 选择跨 GPU disaggregation：彻底消除计算干扰，同时通过 tailored parallelism 分别优化 TTFT 和 TPOT

- **主要 trade-off：**  

  1. **Isolation vs. Communication overhead**  
     - 强 phase 隔离提升 SLO 稳定性，但引入 KV cache 传输开销  

  2. **Goodput vs. Deployment complexity**  
     - 显著提升 goodput，但集群管理更复杂  

  3. **Resource matching vs. Affinity**  
     - 高 node-affinity 集群效果最佳  

  4. **Single-phase optimization vs. Unified**  
     - 各阶段可独立达到最优，但需协调通信

---

### 4. 实验设置

- **硬件：**  
  - Multi-GPU clusters（高/低 node-affinity 配置）

- **模型：**  
  - OPT-13B/66B/175B、Llama 等多种 LLM

- **workload：**  
  - 混合真实服务负载：chat、code completion、document summarization 等，包含 variable length 和 bursty 到达

- **baseline：**  
  - 标准 co-located continuous batching 系统（vLLM 类）  
  - 其他 disaggregation baseline

---

### 5. 核心实验结论

- 在多种 LLM 和 workload 下，能服务 **7.4×** 更多请求，或支持 **12.6×** 更严格的 SLO，同时 >90% 请求满足 latency 约束  
- 相比 SOTA 系统，goodput 显著提升，latency 分布更稳定  
- 有效消除 prefill-decode 干扰，资源利用率更高

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - KV cache 传输开销在低 affinity 集群中仍较明显  
  - 未深度讨论 fault tolerance（单 decode GPU 故障可能影响多个 prefill）  

- **个人判断：**  

  1. **全局视角优秀**  
     - 真正实现 phase-level 解耦，是 disaggregation 的代表作  

  2. **SLO 控制严格**  
     - Goodput-optimized 设计形成完整闭环  

  3. **集群部署友好**  
     - 但对网络带宽要求较高  

  4. **与 intra-GPU 系统高度互补**  
     - 可结合 Semi-PD 等进一步优化小规模场景

---

## Paper: Llumnix (OSDI'24)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 cross-instance dynamic rescheduling：在多实例环境下通过 live migration of requests 和 KV states，实现全局 P/D 负载均衡和资源复用。

- **调度粒度：**  
  - 以 Runtime rescheduling across instances 为主  
  - 属于 system-level global scheduling  
  - 支持 request-level 和 KV cache 迁移

- **切换策略：**  
  - Efficient live migration（类似 OS context switch）  
  - 动态 placement 和 priority-aware rescheduling  
  - 无需等待请求完成即可全局调整

---

### 2. SLO 建模

- **优化目标：**  
  - 改善 tail latency、支持 priority differentiation 并提升 cost efficiency。

- **TTFT / TPOT 建模方式：**  
  - 通过 rescheduling 缓解 heterogeneous requests 导致的 variance  
  - 全局视角优化 TTFT 和 TPOT 分布

- **约束形式：**  
  - SLO/priority-aware 动态调度（中等强度）  
  - 形成闭环：监控 → migration → 负载均衡

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 单实例系统在 unpredictable workloads 下易出现 fragmentation 和 load imbalance  
  - Llumnix 通过 multi-instance 全局 rescheduling 提供 isolation、priority 和更好利用率

- **主要 trade-off：**  

  1. **Migration overhead vs. Isolation**  
     - 高效 migration 设计大幅降低开销  

  2. **Global view vs. Complexity**  
     - 全局优化效果好，但系统复杂度上升  

  3. **Tail latency vs. Average**  
     - 显著改善 tail latency  

---

### 4. 实验设置

- **硬件：**  
  - Multi-instance GPU setups

- **模型：**  
  - Various LLMs

- **workload：**  
  - Heterogeneous real-world traces

- **baseline：**  
  - vLLM 等 single-instance engines

---

### 5. 核心实验结论

- Tail latency 改善一个数量级  
- High-priority 请求加速高达 **1.5×**  
- 相同 tail latency 下 cost savings 高达 **36%**

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - Migration overhead 在极端场景仍存在  
  - 依赖底层 inference engine  

- **个人判断：**  

  1. **全局调度代表作**  
     - 填补了 cluster-level 空白  

  2. **SLO 支持全面**  
     - 优先级和 tail latency 控制优秀  

  3. **与 intra-GPU 系统互补性强**  

  4. **实际部署价值高**


---


## Paper: RAPID-Serve (Masood et al., arXiv'26)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 intra-GPU P/D concurrency：在单 GPU 内让 prefill 和 decode 并发执行但不混合成单一 batch，各阶段独立推进，实现 spatial-like overlap 和资源隔离。

- **调度粒度：**  
  - 以 Adaptive resource management（CU masking / SM partitioning）为主  
  - 属于 fine-grained intra-GPU resource partitioning  
  - 不同于 token-level 调度，更偏资源层复用

- **切换策略：**  
  - 采用 profiling-driven 的**动态资源重划分（resource re-partitioning）**  
  - 根据当前 workload 比例和 SLO 反馈调整资源占比  
  - 本质是“并行执行 + 低开销重分配”，避免频繁 time-slicing 和 KV 迁移

---

### 2. SLO 建模

- **优化目标：**  
  - 在满足 latency SLOs（尤其是 ITL/TBT）的前提下，最大化 throughput 和 resource utilization。

- **TTFT / TPOT 建模方式：**  
  - TTFT 由 prefill 阶段主导，TPOT 由 decode 阶段主导  
  - 通过 phase-specific profiling 量化资源需求和干扰  
  - 采用运行时 feedback 动态调整

- **约束形式：**  
  - Explicit SLO-aware（中等强度）：通过资源控制器直接优化 SLO 达成率  
  - 优先保障 decode 的 TPOT 稳定性，prefill 使用剩余资源

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - Hybrid batching 会显著 inflate decode latency，全 disaggregation 带来 KV transfer overhead 和 weights 重复  
  - RAPID-Serve 选择单 GPU 内 disaggregated computation + unified storage，既消除计算干扰，又保持高利用率

- **主要 trade-off：**  

  1. **Isolation vs. Utilization**  
     - 强资源隔离提升 latency 稳定性，但可能略微降低峰值利用率  

  2. **Flexibility vs. Complexity**  
     - 动态 partitioning 适应性强，但引入资源控制器  

  3. **Hardware dependency vs. Generality**  
     - 依赖 CU masking 等特性，但效果显著  

  4. **Single-GPU focus vs. Cluster**  
     - 单卡高效，可作为 disaggregation 的补充

---

### 4. 实验设置

- **硬件：**  
  - AMD Instinct GPUs 及 NVIDIA 等 resource-constrained 配置

- **模型：**  
  - 标准 Transformer-based LLMs

- **workload：**  
  - Diverse P/D ratios、SLO-sensitive 混合真实负载，包含 bursty 到达

- **baseline：**  
  - Hybrid batching 系统  
  - 全 PD disaggregation 系统  
  - Naive co-location

---

### 5. 核心实验结论

- Unconstrained throughput 最高提升 **4.1×**（平均 1.7×）  
- SLO-constrained 场景下 throughput 最高提升 **32×+**（平均 4.9×）  
- 在 resource-limited 环境下 latency 稳定性和 GPU 利用率显著优于 baseline

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - 强依赖 hardware-specific features（如 CU masking）和 profiling  
  - 极端负载下 interference 仍需 careful tuning  

- **个人判断：**  

  1. **实用性强**  
     - 特别适合资源受限场景  

  2. **SLO 控制较好**  
     - 但非最严格的 worst-case 建模  

  3. **缺少全局 cluster 视角**  
     - 主要聚焦单 GPU 优化  

  4. **部署复杂度适中**  
     - 优于全 disaggregation

---

## Paper: Orca (Yu et al., OSDI'22)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 Iteration-level scheduling + selective batching：在单 GPU 内通过同一 iteration 混合 prefill 和 decode tokens，实现 requests 动态加入与离开，属于经典的时间复用（continuous batching）机制。

- **调度粒度：**  
  - 以 Iteration / Step 级别为主  
  - Selective batching：仅对非-Attention 操作 batching，Attention 单独处理以支持 variable lengths

- **切换策略：**  
  - 每次 iteration（生成一个 token 的前向传播）结束时，动态踢出已完成请求并加入新 prefill 请求  
  - FCFS-like continuous batching，无需等待整个 request 完成

---

### 2. SLO 建模

- **优化目标：**  
  - 降低 queueing latency 和 head-of-line blocking，提升 overall throughput 和 responsiveness。

- **TTFT / TPOT 建模方式：**  
  - 通过 iteration-level 执行大幅减少等待时间，改善 latency 分布  
  - 隐式优化 TTFT 和 TPOT 一致性

- **约束形式：**  
  - Implicit SLO-aware（弱形式）  
  - 通过调度自然缓解 tail latency

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 传统 request-level scheduling 在 autoregressive 生成中会导致已完成请求阻塞新请求，产生严重 head-of-line blocking  
  - Orca 通过 iteration-level + selective batching 解决 variable iteration counts 和 non-batchable 操作的问题

- **主要 trade-off：**  

  1. **Batching efficiency vs. Responsiveness**  
     - Selective batching 略微牺牲部分效率，但大幅提升 latency 一致性  

  2. **Simplicity vs. Padding waste**  
     - 避免 request-level 的 padding/staling  

  3. **Scalability vs. Implementation**  
     - 在大模型上 throughput 显著提升，但需修改底层执行引擎

---

### 4. 实验设置

- **硬件：**  
  - Multi-GPU 集群，支持数百亿参数规模模型

- **模型：**  
  - GPT-3 类 Transformer generative models

- **workload：**  
  - Variable output lengths 的生成任务

- **baseline：**  
  - NVIDIA FasterTransformer 等 request-level 系统

---

### 5. 核心实验结论

- 在 GPT-3 175B 上，相同 latency 下 throughput 提升高达 **36.9×**  
- 显著减少 head-of-line blocking，提升系统 scalability 和资源利用率  
- 在真实生成负载下 latency 分布更加均匀

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - 未针对极长上下文或 MoE 模型深度优化  
  - Memory management 在超大规模时仍有挑战  

- **个人判断：**  

  1. **奠基性工作**  
     - 开启了 iteration-level continuous batching 时代  

  2. **SLO 控制较为基础**  
     - 缺乏显式 SLO 建模和预留  

  3. **单系统视角**  
     - 未涉及 P/D disaggregation 或 cluster-level 调度  

  4. **对后续工作影响深远**  
     - 是 vLLM、SplitFuse 等系统的核心基础


---

## Paper: SplitFuse (DeepSpeed-FastGen, 2024)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 Dynamic SplitFuse 机制：在单 GPU 内通过 token-level composition 将长 prompt 动态拆分成小 chunks，与正在进行的 decode tokens 组合成固定大小的 forward pass，实现 prefill 和 decode 在同一 iteration 中高度交织执行，属于时间复用（time-multiplexing）+ token composition 的混合范式。

- **调度粒度：**  
  - 以 Iteration-level + token budget（固定 forward token 数量）为主  
  - 属于 fine-grained token-level continuous batching  
  - 支持 dynamic prompt decomposition 和 unification

- **切换策略：**  
  - 每次 iteration 结束时动态 composition，无需 time-slicing 或显式切换  
  - 结合 paged/blocked KV cache 实现 stall-free 调度，长 prompt 可被“切片”逐步处理

---

### 2. SLO 建模

- **优化目标：**  
  - 最大化 effective throughput（goodput），同时显著降低 average latency 和 tail latency，提供一致的 TTFT/TPOT。

- **TTFT / TPOT 建模方式：**  
  - TTFT 和 TPOT 均通过固定 forward token size 减少 variance，实现 token-level predictability  
  - 基于 token-throughput curve 特性进行隐式优化

- **约束形式：**  
  - Implicit SLO-aware（弱形式）  
  - 通过动态 composition 自然控制延迟，无显式 worst-case 资源保留

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - Prefill 计算密集、decode 内存受限，且请求长度高度异构；传统 continuous batching 在长 prompt 下会导致严重 latency variance 和 bubble  
  - SplitFuse 通过将 prefill “切片”并与 decode 融合，让系统始终运行在 throughput saturation region

- **主要 trade-off：**  

  1. **Chunking overhead vs. Latency consistency**  
     - 引入少量拆分开销，但大幅降低 tail latency  

  2. **Flexibility vs. Implementation complexity**  
     - 对 variable length 适应性极强，但引擎实现更复杂  

  3. **Long-prompt advantage vs. Short-prompt**  
     - 在长 prompt/short output 场景收益最大，其他场景收益相对有限  

  4. **Single-engine focus vs. Cluster integration**  
     - 单实例高效，但需与上层调度结合

---

### 4. 实验设置

- **硬件：**  
  - 多种 GPU 配置（单机与多机测试）

- **模型：**  
  - 多种 LLM 架构（Decoder-only Transformer）

- **workload：**  
  - 长 prompt 重载、interactive generation、真实 variable length traces

- **baseline：**  
  - vLLM、Orca 等标准 continuous batching 系统

---

### 5. 核心实验结论

- Effective throughput 最高提升 **2.3×**，平均 latency 降低约 **2×**，token-level tail latency 最高降低 **3.7×**（vs. vLLM）  
- 在长 prompt workloads 下显著改善一致性和 responsiveness，实现 near-perfect scalability  
- GPU 利用率大幅提高，latency 分布更加稳定

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - 对 short prompt/long output 场景优势不明显  
  - 依赖 token 分析，对全新 workload 适配仍需一定 tuning  

- **个人判断：**  

  1. **SLO 控制较为间接**  
     - 依赖 composition 隐式保障，而非显式保留  

  2. **调度粒度精细实用**  
     - Token-level 设计对 variable length 非常友好  

  3. **缺少 cluster-level 视角**  
     - 主要单实例优化，可与 disaggregation 互补  

  4. **部署友好度高**  
     - 易集成到现有 continuous batching 引擎中

---
## Paper: Dilu: Enabling GPU Resourcing-on-Demand for Serverless DL Serving via Introspective Elasticity (ASPLOS'25)

---

### 1. 调度机制

- **P/D 复用方式：**  
  提出 **Introspective Elasticity (IE)** 机制：在 serverless 环境下通过 introspective（自省式） profiling 实现 GPU resourcing-on-demand，支持细粒度、adaptive 的 two-dimensional co-scaling（垂直 scaling + 水平 scaling 平滑切换），让 prefill/decode 等 DL serving 任务按需动态分配 GPU 资源。

- **调度粒度：**  
  - 以 **multi-factor profiling + fine-grained resource co-scaling** 为主  
  - 支持 SM-level / memory-level 细粒度调整，以及 instance-level 水平扩展  
  - 属于 introspective + elasticity-driven 的混合资源调度

- **切换策略：**  
  - 采用 **dynamic GPU provisioning 和 smooth transition** between vertical（资源重划分）和 horizontal（实例增减）scaling  
  - 通过 efficient pruning search 和 resourcing-complementary principles 实现低开销自适应调整  
  - 无需预留固定资源，按需 on-demand 分配，支持 serverless 的 cold-start 和 bursty 负载

---

### 2. SLO 建模

- **优化目标：**  
  - 在 serverless 环境下实现 GPU resourcing-on-demand，同时最大化 throughput 并严格满足 latency SLO / QoS。

- **TTFT / TPOT 建模方式：**  
  - 通过 multi-factor profiling（考虑 compute、memory、bandwidth 等）量化 DL 任务需求  
  - 结合运行时 introspection 动态预测和调整资源，避免 over-provisioning

- **约束形式：**  
  - Explicit SLO-aware（强形式）：基于 profiling 和 resourcing-complementary 原则直接优化资源分配  
  - 形成闭环：introspective profiling → dynamic co-scaling → SLO 保障

---

### 3. 关键设计选择 & 理由

- **为什么选这种方案：**  
  - 传统 serverless DL serving 存在资源静态分配、over-provisioning 和 cold-start 问题，无法按需灵活弹性  
  - Dilu 选择 introspective elasticity：通过自省式 profiling + 二维 co-scaling，既实现细粒度资源复用，又支持 serverless 的 on-demand 特性，彻底解决 GPU 资源浪费和 SLO violation

- **主要 trade-off：**  

  1. **Elasticity vs. Profiling overhead**  
     - 细粒度自适应提升利用率，但引入 profiling 开销（通过 pruning search 缓解）  

  2. **Vertical vs. Horizontal scaling**  
     - 平滑切换两者，兼顾快速响应和大规模扩展  

  3. **Isolation vs. Sharing**  
     - 支持高效 co-location sharing，同时保障 QoS  

  4. **Serverless simplicity vs. System complexity**  
     - 极大简化用户侧部署，但底层调度器更复杂

---

### 4. 实验设置

- **硬件：**  
  - Kubernetes + Docker 集群，多 GPU 节点（NVIDIA 数据中心 GPU）

- **模型：**  
  - 多种 DL 模型（包括 LLM inference 和 training 任务）

- **workload：**  
  - Serverless 典型混合负载：bursty 请求、variable batch sizes、不同 SLO 要求

- **baseline：**  
  - INFless 等现有 serverless DL 系统  
  - 静态资源分配 / 非弹性 GPU sharing 系统

---

### 5. 核心实验结论

- 在集群评估中，inference throughput 提升 **1.8×**，training throughput 提升 **1.1×**（vs. INFless）  
- 显著降低 GPU 资源浪费，实现真正 resourcing-on-demand，同时保持低 tail latency 和高 SLO 满足率  
- 在 bursty workload 下表现突出，平滑 scaling 能力强

---

### 6. 局限性 / 未解决问题

- **论文自述：**  
  - Profiling 依赖一定 offline + online 结合，对全新模型适配仍需初始开销  
  - 主要针对 serverless 场景优化，在传统 always-on serving 中优势相对较小  

- **个人判断：**  

  1. **Serverless GPU 弹性代表作**  
     - 首次系统性解决 GPU resourcing-on-demand 问题  

  2. **SLO 控制严格且实用**  
     - Introspective 设计形成良好闭环  

  3. **与 LLM serving 系统互补性强**  
     - 可结合 MuxWise、DistServe 等进一步提升大规模 LLM 服务  

  4. **实际部署价值高**  
     - 开源实现 + Kubernetes 集成友好，未来在云原生 AI 平台中潜力巨大