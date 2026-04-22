# Notes

## vllm-ascend PD 架构现状

- AscendScheduler（`vllm_ascend/core/scheduler.py`）已有 PD phase 机制，但面向多卡 disaggregation，非单卡时分复用
- 上游 vLLM V1 调度器（`vllm/v1/core/sched/scheduler.py`）是统一 token-based 模型，无硬性 P/D 分离
- AscendAttentionState（`vllm_ascend/attention/attention_v1.py`）：PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding
- 现有局限：phase 切换"一刀切"，无动态 P↔D 策略，无单卡时分复用优化

## Baseline 实验（2026-03-31，单卡 910B3）

### 环境
- 2 × Ascend 910B3（64GB HBM），CANN 8.3.RC1，torch 2.7.1（华为镜像源），vllm 0.11.0，vllm-ascend 0.11.0rc1

### 性能数据

| Model | Prompt | TTFT(ms) | TPOT(ms) | Batch OutTPS |
|---|---|---|---|---|
| Qwen3-0.6B | short(~7tok) | 28.2 | 17.97 | 368.1 |
| Qwen3-0.6B | long(~53tok) | 27.6 | 15.59 | 275.0 |
| Qwen3-4B | short(~7tok) | 33.9 | 18.98 | 380.5 |
| Qwen3-4B | long(~53tok) | 30.0 | 19.02 | 229.7 |

### 关键发现
- TTFT 极低（28-34ms），NPU 算力在 prefill 阶段有大量冗余
- TPOT ~18-19ms，decode 是 memory-bandwidth bound，0.6B→4B 差异极小
- Prefill 占总时间 <1%，decode 主导延迟 → **时分复用机会明确**

## 实验 A：ACL Graph 行为分析（2026-04-01，Qwen3-4B，单卡 910B3）

### 关键机制发现（源码分析）
- vLLM **自动 pad** batch size 到最近的已编译 graph shape（`pad_for_cudagraph()`）
- 仅当 total_tokens > max_compiled_size（512）时才 eager fallback
- 因此 Graph-Aware Batch Shaping 的目标是**减少 padding 浪费**，不是避免 eager

### Test 1: Graph 配置
- 48 种已编译 shape，范围 [1, 512]，平均间距 10.9
- 具体 sizes: 1, 2, 8, 16, 32, 40, 48, 64, 72, 88, ..., 504, 512
- 小 batch 区域间距大（1→2→8→16），大 batch 区域间距 ~8

### Padding 浪费分析

| actual_bs | padded_to | waste% | 说明 |
|---|---|---|---|
| 1 | 1 | 0% | 精确匹配 |
| 3 | 8 | 62.5% | 最坏情况之一 |
| 5 | 8 | 37.5% | |
| 8 | 8 | 0% | 精确匹配 |
| 17 | 32 | 46.9% | |
| 32 | 32 | 0% | 精确匹配 |

- Prefill 阶段（bs 1-8）：平均 **23.4%** 浪费
- Decode 阶段（bs 1-64）：平均 **16.5%** 浪费

### Test 2: Padding 对吞吐影响
- 精确匹配 vs pad+1 的 per_token 延迟差异 <5%（都走 graph）
- Padding 主要浪费算力，不改变执行模式

### Test 3: Eager 边界

| batch_size | mode | per_tok(ms) | throughput(tok/s) |
|---|---|---|---|
| 496 | GRAPH | 74.9 | 6620.9 |
| 512 | GRAPH | 81.8 | 6258.2 |
| 513 | EAGER | 98.5 | 5208.6 |
| 528 | EAGER | 96.2 | 5491.0 |
| 544 | EAGER | 101.5 | 5360.1 |

- Eager 退化：**+26%** per_token 延迟（78ms → 99ms）

### Test 4: Graph 编译耗时
- Eager 模式加载：29.7s
- Graph 模式加载：114.1s
- **Graph 编译开销：84.4s**（启动时一次性，48 个 shape）

### 对 TDM 的启示
1. TDM decode batch 通常 <512，不会触发 eager，graph 切换无额外开销
2. Graph-Aware Batch Shaping 核心价值：减少 padding 浪费（尤其 prefill 小 batch 23.4%）
3. TDM 让 P/D batch size 更可预测 → 更好覆盖已编译 shape → 减少浪费
4. 启动编译 84.4s 是固定成本，TDM 不增加额外编译负担

---

## 实验 B：NPU P/D 资源利用率画像（2026-04-02，Qwen3-4B，单卡 910B3）

### 实验设计
- **Prefill-heavy**：200 条唯一长 prompt（~580 tok），max_tokens=1，分 4 批提交保持 NPU 持续忙碌
- **Decode-heavy**：16 条短 prompt（"Hi"），max_tokens=512，最大化 decode 占比
- **Mixed**：5 条混合长度 prompt，max_tokens=150
- **采集方式**：DCMI 后台线程持续采样 HBM BW（~5ms 间隔）和 AICore（~100ms 间隔）
- **注意**：每条 prompt 添加唯一前缀避免 prefix caching 命中

### 结果

| Scenario | AICore% | HBM BW% | In tok | Out tok | Time(s) | AICore samples | HBM BW samples |
|---|---|---|---|---|---|---|---|
| prefill_heavy | **69.8** | **15.8** | 117290 | 200 | 5.49 | 34 | 882 |
| decode_heavy | **42.6** | **25.1** | 16 | 8192 | 10.89 | 67 | 1716 |
| mixed | 33.9 | 26.1 | 45 | 750 | 3.69 | 23 | 603 |

### 关键发现
1. **Prefill = compute-bound**：AICore 69.8%，HBM BW 15.8% → 算力利用率高，带宽利用低
2. **Decode = memory-bandwidth-bound**：AICore 42.6%，HBM BW 25.1% → 带宽利用更高，算力相对空闲
3. **资源互补量化**：Prefill 的 AICore 利用率是 Decode 的 **1.64×**，Decode 的 HBM BW 是 Prefill 的 **1.59×**
4. **TDM 机会确认**：两阶段资源需求互补，时分复用可在时间维度上提升整体资源利用率
5. **Decode 主导时间**：decode_heavy 10.89s vs prefill_heavy 5.49s（2:1），decode 阶段有约 27% AICore 空闲可被利用

### 对 TDM 方案的启示
- Prefill 算力利用率高（69.8%）但带宽有余（15.8%）→ TDM 在 decode 间隙插入 prefill 可充分利用闲置算力
- Decode 带宽利用率更高（25.1%）但算力有余（42.6%）→ P/D 交替执行让两种资源交替饱和
- 混合场景 AICore 33.9%、HBM BW 26.1% → 接近 decode-heavy，说明实际 serving 以 decode 为主

---

## 实验 C：DCMI 在线采样对推理的影响（2026-04-02，Qwen3-4B，单卡 910B3）

### 实验设计
- Baseline：16 条 prompt × 200 tokens，8 轮（取后 7 轮平均）
- 对比：无采样 / HBM BW 2ms / 5ms / 10ms / 20ms / AICore 100ms
- DCMI 采样在 Python 后台 daemon 线程中执行

### 结果

| Config | TPS (tok/s) | Overhead |
|---|---|---|
| no_sampling (baseline) | 670.1 | 0% |
| hbm_bw_2ms | 621.4 | **7.3%** |
| hbm_bw_5ms | 612.2 | **8.6%** |
| hbm_bw_10ms | 615.6 | **8.1%** |
| hbm_bw_20ms | 618.3 | **7.7%** |
| aicore_100ms | 683.1 | ~0% (噪声) |

### 关键发现
1. **HBM BW 采样开销与频率无关**：2ms→20ms 都是 ~7-9%，说明开销是**后台线程存在本身**（GIL 争用 / NPU driver 锁），而非 DCMI 调用次数
2. **AICore 100ms 采样开销可忽略**：354→683 tok/s 范围内波动在噪声内
3. **~8% 开销是否可接受**：如果 TDM 带来 >10% 的吞吐提升，则 DCMI 开销被覆盖

### 对 TDM 方案的启示
1. **AICore 低频采样可直接使用**（~0% 开销）
2. **HBM BW 连续采样需优化**：
   - 方案 A：仅在 phase 切换边界采样（而非每 iteration），大幅降低采样次数
   - 方案 B：用 C 扩展替代 Python ctypes 调用，避免 GIL 争用
   - 方案 C：离线画像替代在线采样，运行时查表
3. **推荐策略**：在线用 AICore 100ms 监测 + 离线画像表覆盖 HBM BW 信息，避免 8% 开销

---

## 实验 D：不同并发负载下的表现（2026-04-02，Qwen3-4B，单卡 910B3）

### 实验设计
- 并发数：1, 2, 4, 8, 16, 32, 64
- 每请求生成 200 tokens，8 轮取后 7 轮平均
- Prompt: "Explain the concept of machine learning in simple terms."（~11 tok）

### 结果

| Conc | TTFT(ms) | TPOT(ms) | TPS (tok/s) | Time(ms) |
|---|---|---|---|---|
| 1 | 27.7 | 21.38 | 46.7 | 4282 |
| 2 | 25.4 | 25.10 | 79.7 | 5021 |
| 4 | 15.0 | 22.19 | 180.5 | 4431 |
| 8 | 6.5 | 24.46 | 328.2 | 4875 |
| 16 | 4.3 | 23.17 | 693.4 | 4615 |
| 32 | 2.3 | 26.63 | 1207.1 | 5302 |
| 64 | 1.9 | 23.56 | 2728.5 | 4691 |

### 关键发现
1. **TPOT 几乎恒定**：21-27ms 范围内（1→64 并发），decode 阶段高度 batching 友好
2. **TTFT 随并发下降**：27.7ms（1 并发）→ 1.9ms（64 并发），因为 batch prefill 效率高
3. **吞吐近线性扩展**：1→64 并发带来 58.4× 吞吐提升，说明 NPU 在低并发下严重 underutilized
4. **无 TPOT 退化拐点**：在 64 并发内 TPOT 仍未翻倍 → 可能需要测试更高并发（128, 256）
5. **总时间几乎恒定**：~4.5s（除 concurrency=2,32 略高），说明 batching 吸收了并发增加

### 对 TDM 方案的启示
1. **低并发（1-4）是 TDM 最佳场景**：NPU 严重 underutilized（46-180 tok/s vs 峰值 2728），有大量空闲算力可用于穿插 prefill
2. **高并发（32-64）TDM 收益递减**：NPU 利用率已较高，但 TPOT 仍稳定 → 说明还有空间
3. **TPOT 恒定特性**：意味着在 decode 间隙插入 prefill 不会严重影响现有 decode 请求的延迟
4. **需要更高并发测试**：找到 TPOT 退化拐点，确定 TDM 在何时需要更谨慎的调度

---

## 实验 E（旧）：PD Unified vs PD Phased 对比（2026-04-02，Qwen3-4B，**单卡** 910B3）

> **注意**：此实验存在公平性问题——仅使用 1 张卡，且 Ascend 默认关闭 chunked prefill。已被实验 E3 取代。

### 实验设计
- **PD Unified**：默认 continuous batching，P/D 混在同一 batch，**chunked prefill OFF**
- **PD Phased**：`enable_pd_transfer=True`，先 prefill 全部请求，再 decode 全部
- 6 种 workload：短 prompt(4/16/32 req) + 长 prompt(4/16 req) + 混合(16 req)，每请求生成 200 tokens

### 结果

| Workload | Unified TPS | Phased TPS | 变化 |
|---|---|---|---|
| short_4req | 157.9 | 173.1 | **+9.6%** |
| short_16req | 759.4 | 647.3 | **-14.8%** |
| short_32req | 1460.5 | 1305.6 | **-10.6%** |
| long_4req | 185.1 | 184.4 | ~0% |
| long_16req | 629.2 | 697.3 | **+10.8%** |
| mixed_16req | 749.1 | 599.4 | **-20.0%** |

---

## 实验 E2：PD Disaggregated 1P1D（2026-04-02，Qwen3-4B，2×910B3）

### 实验设计
- Device 0 = Prefill（kv_producer），Device 1 = Decode（kv_consumer）
- 使用 `LLMDataDistCMgrConnector` + 手工构造 ranktable（同机无 ROCE，填容器 IP 172.17.0.3 成功）
- 8 轮，去首轮，6 种 workload，每请求生成 200 tokens
- **chunked prefill 自动开启**（KV transfer 模式下 vLLM V1 默认行为）

### 结果

| Workload | TPS | TTFT(ms) | TPOT(ms) | P_time(s) | D_time(s) | Total(s) |
|---|---|---|---|---|---|---|
| short_4req | 155.1 | 17.9 | 6.39 | 0.072 | 5.085 | 5.157 |
| short_16req | 623.0 | 4.2 | 1.59 | 0.068 | 5.069 | 5.136 |
| short_32req | 1182.8 | 2.5 | 0.84 | 0.080 | 5.331 | 5.411 |
| long_4req | 151.8 | 15.8 | 6.54 | 0.063 | 5.207 | 5.270 |
| long_16req | 619.0 | 6.8 | 1.59 | 0.108 | 5.061 | 5.169 |
| mixed_16req | 594.5 | 6.5 | 1.66 | 0.104 | 5.278 | 5.383 |

### 关键发现
- Prefill 耗时极短（60-108ms），decode 占 >98% 时间
- ranktable 在同机场景可手工构造，无需 ROCE/hccn_tool

---

## 实验 E3：公平 2 卡对比（2026-04-02，Qwen3-4B，2×910B3，8 轮）

### 实验设计
所有方案统一使用 2 张 910B3，公平对比：
1. **Unified TP=2**：2 卡 tensor parallel，continuous batching，chunked prefill **OFF**（Ascend 默认）
2. **Unified TP=2 + CP**：同上但 chunked prefill **ON**
3. **Phased TP=2**：2 卡 tensor parallel，enable_pd_transfer，chunked prefill OFF
4. **Disagg 1P1D**：1 卡 prefill + 1 卡 decode（来自实验 E2）

### TPS 结果

| Workload | Unified TP2 | Unified TP2+CP | Phased TP2 | Disagg 1P1D | CP vs Unified | Phased vs Unified | Disagg vs Unified |
|---|---|---|---|---|---|---|---|
| short_4req | 174.7 | 177.9 | 184.2 | 155.1 | +1.8% | +5.4% | **-11.2%** |
| short_16req | 671.0 | 707.7 | 715.8 | 623.0 | +5.5% | +6.7% | **-7.2%** |
| short_32req | 1206.6 | 1336.7 | 1249.7 | 1182.8 | **+10.8%** | +3.6% | -2.0% |
| long_4req | 172.6 | 171.9 | 181.5 | 151.8 | -0.4% | +5.2% | **-12.1%** |
| long_16req | 663.4 | 699.4 | 692.2 | 619.0 | +5.4% | +4.3% | **-6.7%** |
| mixed_16req | 631.9 | 705.4 | 646.3 | 594.5 | **+11.6%** | +2.3% | -5.9% |

### TTFT 结果 (ms)

| Workload | Unified TP2 | Unified TP2+CP | Phased TP2 | Disagg 1P1D |
|---|---|---|---|---|
| short_4req | 14.1 | 14.4 | 12.9 | 17.9 |
| short_16req | 10.4 | 3.0 | 3.1 | 4.2 |
| short_32req | 8.8 | 1.7 | 1.8 | 2.5 |
| long_4req | 15.2 | 16.3 | 11.5 | 15.8 |
| long_16req | 12.0 | 5.6 | 6.3 | 6.8 |
| mixed_16req | 6.4 | 5.9 | 6.5 | 6.5 |

### TPOT 结果 (ms)

| Workload | Unified TP2 | Unified TP2+CP | Phased TP2 | Disagg 1P1D |
|---|---|---|---|---|
| short_4req | 5.68 | 5.58 | 5.39 | 6.39 |
| short_16req | 1.45 | 1.41 | 1.39 | 1.59 |
| short_32req | 0.79 | 0.74 | 0.80 | 0.84 |
| long_4req | 5.75 | 5.77 | 5.48 | 6.54 |
| long_16req | 1.45 | 1.41 | 1.42 | 1.59 |
| mixed_16req | 1.56 | 1.40 | 1.52 | 1.66 |

### 关键发现

1. **Chunked Prefill 显著提升混合/高并发场景**：Unified+CP 比 Unified 在 short_32req +10.8%、mixed_16req +11.6%，因为 CP 允许 prefill 和 decode 交错执行，减少 head-of-line blocking
2. **Phased TP=2 全面优于 Unified TP=2（无 CP）**：所有 workload 均 +2.3%~+6.7%，说明在 2 卡 TP 下 P/D 分阶段有一致收益
3. **Disagg 1P1D 全面劣于其他方案**：-2%~-12%，因为：
   - 每张卡只用于单一阶段，decode 卡在 prefill 期间完全空闲（反之亦然）
   - KV cache 跨卡传输开销（虽然同机走 HCCS 但仍有延迟）
   - 小模型（4B）单卡即可高效处理，TP=2 的通信开销 < 分离的空闲开销
4. **TTFT：CP/Phased 大幅优于 Unified（无 CP）**：高并发下 TTFT 从 8-10ms 降到 1.7-3ms
5. **TPOT 差异小**：所有方案 TPOT 接近（decode 阶段行为相似），Disagg 略高 10-15%

### 对 TDM 方案的启示
- **Disagg 1P1D 不适合小模型/同机场景**：分离的资源浪费 > P/D 干扰的代价
- **Chunked Prefill 是强 baseline**：TDM 需要在 CP 基础上论证额外收益，而非与无 CP 的 Unified 比
- **TDM 的定位更清晰**：不是替代 PD 分离，而是在**单卡/TP 并行**场景下，通过智能的时分调度在 CP 基础上进一步优化
- **大模型场景可能不同**：4B 模型计算量小，TP=2 通信占比高；大模型（如 70B+）的 PD 分离收益可能更明显

---

## Core Code Paths
- Scheduler: `vllm_ascend/core/scheduler.py`
- Model Runner: `vllm_ascend/worker/model_runner_v1.py`
- Attention: `vllm_ascend/attention/attention_v1.py`
- ACL Graph: `vllm_ascend/compilation/acl_graph.py`
- 上游 Scheduler: `vllm/v1/core/sched/scheduler.py`

## ACL Graph 核心约束
- 每种 batch size 需单独编译 graph（~1.4s/graph），总数有硬上限（MAX_CAPTURE_SIZE=1800）
- 运行时必须精确匹配预编译 size，未命中 fallback eager mode 性能骤降
- TDM 优势：P/D 分离后 shape 更可预测，graph 高度复用（vs chunked prefill 频繁切换）
- 优化机会：Graph-Aware Batch Shaping — 调度器主动选择匹配已编译 graph 的 batch size

### ACL Graph 主要作用于 Decode 阶段（源码分析，2026-04-02）

**关键机制**：ACL Graph 的 pad/eager 判断基于 `total_num_scheduled_tokens`（一轮 iteration 中所有请求的 token 总数），而非请求数：
- **Decode**：每个请求产出 1 token → `total_tokens = 并发请求数`（如 32 请求 = 32 tokens）→ 通常在 graph 范围内（≤512）
- **Prefill**：处理完整 prompt → `total_tokens = 所有 prompt 长度之和`（如 4×500tok = 2000）→ 大概率超过 max_compiled_size(512) → eager fallback

**源码证据**（`model_runner_v1.py:3488-3545`）：
- 编译分两轮：第一轮 mixed mode（`uniform_decode=False`），第二轮专门为 decode 编译 FULL mode（`uniform_decode=True`）
- Decode 有独立的 `decode_cudagraph_batch_sizes` 集合，体现了 decode 是 graph 的主要受益者

**对 TDM 方案的修正**：
- Graph-Aware Batch Shaping 的**主战场是 decode 阶段**的 padding 浪费优化
- Prefill 阶段：长 prompt 走 eager，graph 约束不相关；仅当少量短 prompt（total_tokens ≤ 512）时 graph 才生效
- 论文叙事应强调 "decode 阶段是主要受益者，prefill 在特定条件下也受益"，而非 "P/D 两阶段都需要 Graph-Aware"

### ACL Graph 相关研究工作（2026-04-02）

- **MuxWise**：使用 CUDA Graphs 但仅用于 decode 执行加速，未将其纳入调度决策
- 其他 11 篇核心论文（DuetServe、Semi-PD、RAPID-Serve、Sarathi、PDM/Drift 等）均不讨论编译图对调度的影响
- **"首次将编译器约束引入调度决策"是本方案的原创贡献**，无直接参考

---

---

> **注：** 相关工作详细分析见 `paper_template.md`；Idea 方案完整版见 `idea_proposal.md`

---

## Dilu 借鉴分析（2026-04-01）

### Dilu → PD TDM 映射
1. **Multi-factor Profiling → 离线 P/D 画像**：扫描 (batch_size, phase) 组合，记录延迟/AICore%/HBM_BW%/graph命中
2. **互补调度 → P/D 互补**：P=compute-bound, D=memory-bound，TDM 让两种资源在时间维度上都不空闲
3. **2D Co-scaling → 自适应**：快调 phase ratio（ms级）+ 慢调 KV cache 预算（需内存重分配）

### NPU 实时利用率获取实测

| 方案 | 延迟 | 结论 |
|---|---|---|
| npu-smi info -t usages | 秒级 | 太慢 |
| acl.rt.get_device_utilization_rate() | ~250ms/call | 太慢 |
| Ascend Profiler (torch_npu.profiler) | trace 模式 | 适合离线建表 |
| **DCMI (libdcmi.so)** | **见下表** | **部分指标可用于在线内省** |

**DCMI 各指标调用延迟实测（ctypes 调用 /usr/local/dcmi/libdcmi.so，需先 dcmi_init()）：**

| 指标 | 延迟 | 可用 | 适合场景 |
|---|---|---|---|
| HBM BW | **~1.1ms** | OK | **每 iteration 可采样，D phase 带宽监测** |
| AICPU | ~1.2ms | OK | 辅助 |
| HBM Usage | ~3.8ms | OK | KV cache 监控 |
| AICore | ~60ms | OK | 每 3-4 iter 采样，P phase 算力监测 |
| VectorCore | ~62ms | OK | 低频采样 |
| NPU Overall | ~62ms | OK | 低频采样 |

### 方案：分层内省 + 离线画像

**在线内省（DCMI）：**
- HBM BW（1ms）：每 iteration 采样 → 实时监测 decode 带宽压力
- AICore（60ms）：异步线程低频采样 → 周期性监测 prefill 算力利用率

**离线画像（Ascend Profiler）：**
- 算子级 AICore/HBM 带宽利用率采集
- 建 (batch_size, phase) → (latency, util%) 画像表
- 在线查表补充 DCMI 无法覆盖的细粒度信息

---

## 安装备忘
- torch/torch-npu 必须从华为镜像源安装：`pip install torch==2.7.1 torch-npu==2.7.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi`
- vllm: `VLLM_TARGET_DEVICE=empty pip install -e .`
- vllm-ascend: `source set_env.sh && pip install -e . --no-build-isolation`
- transformers < 5.0.0
- 推理脚本需在 `if __name__ == '__main__':` 内

## 实验 F：Graph-Aware Batch Shaping vs Naive Batching（2026-04-02，Qwen3-4B，单卡 910B3）

### 实验设计
- 核心问题：调度器主动对齐已编译 Graph shape 能带来多少收益？
- 4 个子测试：padding 浪费全景、精确匹配 vs padding 吞吐对比、TDM 模拟场景、per-token 效率

### Test 1: Padding 浪费全景分析

| 场景 | 平均浪费 | 最坏浪费 | 精确匹配 | 低浪费(<10%)覆盖率 |
|---|---|---|---|---|
| prefill_typical (1-8) | **23.4%** | 62.5% | 3/8 | 38% |
| prefill_burst (1-32) | **23.0%** | 62.5% | 5/32 | 28% |
| decode_light (1-16) | **22.7%** | 62.5% | 4/16 | 31% |
| decode_medium (1-64) | **16.5%** | 62.5% | 8/64 | 41% |
| decode_heavy (1-128) | 11.0% | 62.5% | 14/128 | 62% |
| all (1-512) | 4.3% | 62.5% | 48/512 | 90% |

- **小 batch（prefill 场景）浪费最严重**：平均 23%，低浪费覆盖率仅 28-38%
- 大 batch 区域已编译 shape 间距小（~8），浪费率自然降低

### Test 2: 精确匹配 vs Padding 吞吐对比

| 分类 | avg per-req TPS |
|---|---|
| 精确匹配 | **42.4** |
| 需 padding | **40.0** |
| **精确匹配优势** | **+6.0%** |

- 精确匹配相比需 padding 的 batch，per-request 吞吐高 6%
- padding 的代价不仅是浪费计算位置，还会拖慢整个 batch 的执行

### Test 3: TDM 模拟场景 Naive vs Graph-Aware

| 场景 | Naive TPS | Graph-Aware TPS | Speedup | Waste 减少 |
|---|---|---|---|---|
| prefill_sparse (1-5) | 128.1 | 102.6 | **-19.9%** | 1.8 |
| prefill_moderate (5-20) | 464.8 | 418.2 | **-10.0%** | 4.5 |
| decode_growing (4→40) | 856.4 | 893.5 | **+4.3%** | 1.6 |
| decode_stable (~32) | 1252.0 | 1260.2 | **+0.7%** | 2.5 |

- **Graph-Aware 策略（向下取整）**：Naive batches 中 bs=3 → aware 选 bs=2（精确匹配），减少 padding 但处理更少请求
- **Prefill 小 batch 场景反效果**（-10% ~ -20%）：向下取整减少每轮请求数，代价远大于 padding 浪费
- **Decode 大 batch 场景有正收益**（+0.7% ~ +4.3%）：大 batch 区间 shape 间距小，向下取整损失少

### Test 4: 同一 Graph Shape 下 Per-Token 效率

| Graph Shape | 满载 per_req | 最低载 per_req | 效率差 |
|---|---|---|---|
| 8 | 40.3 | 42.0 | -4.2%（噪声） |
| 16 | 46.7 | 39.2 | **+16.1%** |
| 32 | 40.0 | 39.7 | +0.7% |
| 64 | 38.1 | 29.7 | **+22.0%** |

- 大 graph shape（64）下满载 vs 最低载效率差高达 **22%**
- Padding 的真正代价：拖慢整个 batch 执行时间，而非仅仅浪费 padding 位置的算力

### 核心发现与对 TDM 的启示

1. **简单"向下取整"的 Graph-Aware 策略不可行**：小 batch 下减少请求数的代价 > padding 浪费的代价
2. **正确的策略是"向上凑"而非"向下减"**：在队列中有足够请求时，主动凑到精确匹配 shape
3. **TDM 的 Phase Switching 应感知 Graph shape**：不是"有请求就立刻 prefill"，而是"等凑够一个好 shape 再切到 prefill"
4. **Layer 1 和 Layer 2 耦合设计的必要性被验证**：切换决策不能只看 SLO，还要考虑当前队列深度能否凑出好的 Graph shape
5. **Decode 场景收益稳定但小**（+0.7% ~ +4.3%）：因为大 batch 区域 shape 间距已经很小（~8），padding 本身不严重
6. **Prefill 场景是 Graph-Aware 的主战场**：shape 间距大（1→2→8→16），需要更智能的调度策略

---

## TDM 概念澄清（2026-04-02，讨论后修正）

### 核心修正：TDM 不是逐 iteration 的 P/D 二选一

**旧理解**（有误）：每个 iteration 独立决定 P 还是 D，基于 SLO slack reactive 切换。

**新理解**：TDM 是在**滑动时间窗口内控制 P:D:M 三种模式的执行比例**。核心控制变量是 prefill 插入率 r_p，不是每次的二选一决策。

### 三种执行模式（动作空间扩展为 {P, D, Mixed}）

| 模式 | 行为 | 适用场景 |
|---|---|---|
| PREFILL | 纯 prefill iteration | 队列深、decode 空闲、低并发 |
| DECODE | 纯 decode iteration | TPOT 紧张、无新请求 |
| MIXED | Chunked Prefill（P+D 混合 batch） | 高并发、负载平稳 |

**关键结论**：
- **CP 是 TDM 的特例**（全部 iteration 选 MIXED）→ TDM 是 CP 的超集
- **实验 E 验证了动态选择的必要性**：低并发 Phased +10%，高并发 Phased -20%
- **不是"分离一定好"或"混合一定好"**，而是根据负载动态选择

### Starvation 问题与解决

V1 SLO-reactive 切换的问题：
- 等 SLO 快违约才切换 → 震荡和滞后
- 极端情况可能饿死 prefill 或 decode

V2 AIMD 解决方式：
- 维护 prefill 插入率 r_p（平滑变化，不是逐次决策）
- r_p 有下界（防 prefill 饿死）、连续 decode 有上界（防 decode 饿死）
- TPOT violation → MD（大幅降频），TPOT 达标 → AI（逐步升频）
- 高并发自动切 MIXED 模式

### Iteration-level vs Batch-level 的关系

两个"level"是同一粒度的两个维度：
- **1 iteration = 1 batch = 1 ACL Graph replay**（等价关系）
- **Iteration-level TDM**（Layer 1）：决定这个 iteration 做什么模式 → 时间维度
- **Batch-level Graph**（Layer 2）：决定这个 iteration 装多少请求 → 容量维度
- 串行决策：Layer 1 先决定模式，Layer 2 再决定大小，Layer 2 可反馈 Layer 1 建议延迟

---

## Pending Questions
- ChunkedPrefill 在 NPU 上的支持程度？
- NPU 单卡 P/D 的详细 profiling 数据（用 Ascend Profiler 采集）
- Mixed 模式在当前 AscendScheduler 中的可行性验证

---

## AscendScheduler 源码精读（2026-04-22，为 V1 原型做准备）

文件位置：`vllm-ascend/vllm_ascend/core/scheduler.py`（587 行）、`schedule_config.py`（108 行）、`worker/model_runner_v1.py::_build_attn_state`（1576 行）

### 1. 现有 schedule() 流程（587 行）

```
AscendScheduler.schedule():
  ├─ L64-66  如果 chunked_prefill_enabled → 直接委托 super().schedule()（走上游 V1 统一调度）
  ├─ L93-104 如果 self.phase == "prefill"：
  │    ├─ 把 num_tokens > num_prompt_tokens 的 running 请求移到 finished_prefill_reqs
  │    └─ 若 waiting/running 都空 → phase 翻到 "decode"
  ├─ L114-304 第一轮循环：从 waiting 取请求做 prefill
  │    ├─ L129 跳过 WAITING_FOR_REMOTE_KVS（disagg P→D 传输中）
  │    ├─ L137-144 LoRA max_loras 约束
  │    ├─ L149-170 算 num_computed_tokens（本地 + 远端）
  │    ├─ L181-207 prompt_limit / token_budget 检查
  │    ├─ L222-227 watermark 检查（_check_watermark_for_prefill）
  │    ├─ L229-232 long_prefill_token_threshold 跳过
  │    └─ L263-301 放入 running，记 scheduled_req_ids，分配 KV blocks
  ├─ L306-311 phase == "decode" → 从 finished_prefill_reqs 补入 running
  ├─ L313-426 第二轮循环：len(scheduled_req_ids) == 0 才进来（即当前 step 没挑到 prefill）
  │    └─ 从 running 挑 decode：每个请求 num_new_tokens=1（spec decoding 例外）
  └─ L447-505 组 SchedulerOutput，advance num_computed_tokens
```

**核心观察**：
- "prefill-first" 策略：一个 step 要么全 P 要么全 D，不会混合（CP 除外 — 那是委托给上游的）
- `self.phase` 是 `""`/`"prefill"`/`"decode"` 三态，只在 `enable_pd_transfer=True` 时激活，**面向多卡 disagg**
- `_check_watermark_for_prefill` 只在 prefill 分支做 KV 水位检查
- L315 `len(self.scheduled_req_ids) == 0` 的门禁：**同一 step 内 prefill 和 decode 互斥**（非 CP 路径下）

### 2. _build_attn_state 的 P/D 判定（model_runner_v1.py:1576-1600）

决策来自**实际调度 token 数**，不来自 scheduler 的 phase 字段：

```
if all(num_scheduled_tokens == seq_lens): → PrefillNoCache  # 纯 prefill 首次
elif all(num_scheduled_tokens == 1):       → DecodeOnly      # 纯 decode
elif spec decoding 条件:                    → SpecDecoding
elif CP 开启 / ascend scheduler 关闭:       → ChunkedPrefill   # 混合
else:                                       → PrefillCacheHit  # prefill 但命中了前缀
```

**结论**：如果我们在 scheduler 里把一个 batch 构造成"全 P/全 D/混合"，attn_state 会被自动推断出来，**不需要改 model_runner**。

### 3. Config 扩展点（schedule_config.py）

AscendSchedulerConfig（dataclass）目前有：
- `enable_chunked_prefill`, `max_long_partial_prefills`, `long_prefill_token_threshold`
- `enable_pd_transfer`, `decode_max_num_seqs`
- `policy`, `scheduler_cls`

扩展字段（V1 需要新增）：
```python
enable_tdm: bool = False
tdm_ttft_slo_ms: float = 100.0
tdm_tpot_slo_ms: float = 50.0
tdm_max_consecutive_p: int = 4    # 防 decode 饿死
tdm_max_consecutive_d: int = 20   # 防 prefill 饿死
tdm_high_concurrency_threshold: int = 32   # R > 阈值 → MIXED
```

---

## Phase Switching V1 原型切入点设计（Day 5 实现的直接指导）

### 最小改动面

**不需要改的部分**：上游 Scheduler、model_runner、acl_graph、attention_v1 — 全都通过"scheduler 组出的 batch 组成"被动接收 attn_state。

**只需要改**：
1. `schedule_config.py` — 加 6 个字段（见上）
2. `scheduler.py` — 新增 `_schedule_tdm()` 方法 + 在 `schedule()` 入口分派
3. `scheduler.py` — 在 `update_from_output()` 里记录 decode 请求最近产出 token 的时间戳，供 TPOT slack 计算

### schedule() 分派改动

```python
def schedule(self) -> SchedulerOutput:
    # 新增分派
    if getattr(self.scheduler_config, 'enable_tdm', False):
        return self._schedule_tdm()
    # 原逻辑保留
    if self.scheduler_config.chunked_prefill_enabled:
        return super().schedule()
    # ... 原 phase=prefill/decode 分支
```

### _schedule_tdm 骨架

```python
def _schedule_tdm(self) -> SchedulerOutput:
    # Step 1: 采集状态
    Q = len(self.waiting)
    R = len(self.running)
    now = time.monotonic()
    ttft_slack = self._compute_ttft_slack(self.waiting, now)
    tpot_slack = self._compute_tpot_slack(self.running, now)

    # Step 2: V1 优先级规则（与 idea_proposal §2.3 对齐）
    cfg = self.scheduler_config
    T_tpot = cfg.tdm_tpot_slo_ms * 1e-3
    T_ttft = cfg.tdm_ttft_slo_ms * 1e-3

    if tpot_slack < T_tpot * 0.3:                       # 紧急保 TPOT
        phase = "DECODE"
    elif Q > 0 and ttft_slack < T_ttft * 0.3:           # 紧急保 TTFT
        phase = "PREFILL"
    elif self.consecutive_d >= cfg.tdm_max_consecutive_d:  # 防 prefill 饿死
        phase = "PREFILL"
    elif self.consecutive_p >= cfg.tdm_max_consecutive_p:  # 防 decode 饿死
        phase = "DECODE"
    elif Q > 0 and R == 0:
        phase = "PREFILL"
    elif Q == 0:
        phase = "DECODE"
    elif R > cfg.tdm_high_concurrency_threshold and Q > 0:
        phase = "MIXED"
    else:
        phase = "DECODE"   # 默认保守（TPOT 敏感）

    # Step 3: 按 phase 组 batch
    if phase == "PREFILL":
        self.consecutive_p += 1; self.consecutive_d = 0
        return self._schedule_prefill_only()   # 复用现有 L114-304 逻辑
    elif phase == "DECODE":
        self.consecutive_d += 1; self.consecutive_p = 0
        return self._schedule_decode_only()    # 复用现有 L313-426 逻辑
    else:  # MIXED
        self.consecutive_p = 0; self.consecutive_d = 0
        return super().schedule()              # 直接委托 CP 路径
```

### TTFT/TPOT slack 计算

- **TTFT slack**：`request.arrival_time`（vLLM V1 Request 已有）+ SLO - now
  - 取 waiting queue 里最紧迫请求的 slack
- **TPOT slack**：需要记录每个 running 请求的最近一次产出 token 的时间戳
  - 在 `update_from_output()` 里维护 `self.last_token_time: dict[req_id, float]`
  - TPOT slack = min over running: SLO - (now - last_token_time[req_id])

### 饿死计数器维护

两个计数器 `self.consecutive_p / consecutive_d` 只在 `_schedule_tdm()` 里更新；非 TDM 路径不动。

### CLI 开关

vllm-ascend 的 AscendSchedulerConfig 是通过 `--additional-config` json 传入的，加字段后即可：
```
--additional-config '{"ascend_scheduler_config": {"enabled": true, "enable_tdm": true, "tdm_ttft_slo_ms": 100, "tdm_tpot_slo_ms": 50}}'
```

### 潜在坑点

1. **L65-66 CP 入口**：如果 `chunked_prefill_enabled=True` 就直接走 super，我们 TDM 的 MIXED 分支复用这个路径是 OK 的，但要确保 `enable_tdm` 和 `chunked_prefill_enabled` 不冲突（建议 TDM 模式下内部允许 CP 路径，但对外只暴露 `--enable-tdm`）。
2. **L315 互斥门禁**：我们的 PREFILL/DECODE 分支各复用一块，不能让它们"本 step 没挑到 prefill 就混去挑 decode"。需要把原逻辑拆成独立方法而不是共用一个 schedule()。
3. **防饿死上限需要按 SLO 反推**：如 TPOT SLO=50ms、iteration ~20ms，则 `max_consecutive_p ≤ 2` 才能保 decode 不违约；需要实验校准。
4. **MIXED 路径的饿死**：super().schedule() 走 CP 后 consecutive 计数器清零，但 CP 本身没区分 P/D，要看实验数据是否需要额外保护。

### 下一步（Day 5）

- 先写 `_schedule_prefill_only()` 和 `_schedule_decode_only()`：把现有 L114-304、L306-426 机械摘出成独立方法（不改语义）
- 再写 `_schedule_tdm()` + `_compute_ttft_slack / _compute_tpot_slack`
- 加 config 字段 + unit test：(a) 无请求时不崩；(b) 纯 Q>0 → PREFILL；(c) 连续 D 到上限强制切 P

## 导师讨论纪要 — 2026-04-08

继 2026-04-03 首轮汇报后的反馈：

1. **模型选型**：主测模型改用 **Qwen3-7B**（不是 4B 也不是 30B+）。理由：7B 已能体现 P/D 互补效应，又在 910B3×2 的硬件预算内可行。
2. **不必过度追求大模型与真实负载**：现阶段重点是把 TDM 机制跑通、把"vs Unified+CP"的增量收益讲清楚；30B+ 模型和真实生产 trace 留作后期补充实验，不作为当前阻塞项。这意味着原 idea_proposal/meeting_outline 里"小模型只是验证可行性，大模型才是收益场景"的叙事需要弱化。
3. **相关工作对比缺失（最主要的批评）**：原 meeting_outline 只比较了"算子级 vs iteration 级"两条技术路线，**没有系统性对比已有的单卡 PD 混部 / 时分复用工作**。导师明确要求下次汇报必须能回答："和 Sarathi-Serve / DistServe / Semi-PD / DuetServe / MuxWise / RAPID-Serve / PDM/Drift / Dilu 的差异在哪？为什么这些工作不能搬到 NPU？" 已在 meeting_outline.md 新增 §五"相关工作对比"，内容来自 idea_proposal.md §1.2 + tdm_related_work.md 七维表的浓缩。
4. **后续动作**：(a) 切换 Qwen3-7B 重跑 Exp B/D/E；(b) 把相关工作对比整理成幻灯片级表格融入下次汇报；(c) 精读 PDM/Drift 强化对比。
