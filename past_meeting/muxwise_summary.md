# MuxWise 论文精读笔记

> **论文**: Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
> **会议**: ASPLOS '26
> **作者**: Ziming Mao, Jiaao He, Junkun Peng, Pei Yang, Yongwei Wu (Tsinghua University)
> **关键词**: Intra-GPU PD spatial multiplexing, GreenContext, goodput optimization

---

## 1. 核心思想

在单 GPU 内通过**空间复用（spatial multiplexing）**同时运行 Prefill 和 Decode，利用 NVIDIA GreenContext 实现 SM（Streaming Multiprocessor）级别的动态分区，将空闲的 decode bubble 时间用于 prefill 计算，从而提升 goodput。

**核心洞察**：
- Decode 阶段 SM 利用率极低（memory-bound），大量 SM 空闲
- 传统方案要么耦合（P/D 互相干扰）、要么分离（需要多卡）
- 空间复用可以在保证 decode SLO 的前提下，将空闲 SM 分配给 prefill

---

## 2. 系统架构（三大模块）

### 2.1 Bubble-less Multiplex Engine（无气泡复用引擎）
- **Decode 端**：使用 CUDA Graph 执行（低延迟、确定性）
- **Prefill 端**：使用 layer-wise eager 执行（灵活性高）
- **同步机制**：Query-based synchronization — prefill 在每层结束时检查 decode 是否完成，若完成则让出 SM 给下一轮 decode
- **Decode 优先级**：decode 总是被优先调度，prefill 使用剩余 SM
- **KV cache 管理**：P/D 共享同一 GPU 的 KV cache，无需跨卡传输

### 2.2 Contention-tolerant Estimator（竞争容忍估算器）
- **Solo-run predictor**：基于 batch size / seq len 预测无竞争时的 P/D 延迟
  - Decode：与 batch size 线性相关（CUDA Graph 固定 kernel launch）
  - Prefill：与 batch size × seq len（即 total tokens）线性相关
- **Contention guard**：从离线 profiling 获取最坏情况下的竞争减速比
  - Decode 减速主要来自 **memory bandwidth 竞争**（非 SM 竞争）
  - 最坏减速约 20-30%，用固定系数放大预测值作为保守估计
  - **不主动管理内存带宽竞争**，而是通过保守估计来容忍

### 2.3 SLO-aware Dispatcher（SLO 感知调度器）
- **Decode SM 分配**：Best-fit 策略 — 找到满足 decode SLO 的最小 SM 数量
- **Prefill SM 分配**：使用剩余全部 SM
- **SM 重配置**：通过 GreenContext 动态调整，微秒级切换
- **调度周期**：每个 decode iteration 重新决策 SM 分配比例

---

## 3. GreenContext 技术细节

- NVIDIA 提供的**进程内轻量级 SM 绑定机制**
- 支持 Pascal 及以后架构（P100/V100/A100/H100/H200）
- 与 MPS（Multi-Process Service）不同：GreenContext 是**同一进程内**的资源隔离
- 配置粒度：SM 数量（如 A100 有 108 个 SM）
- 切换延迟：微秒级（远低于 iteration 间隔）
- **AMD 等价物**：`hipExtStreamCreateWithCUMask()`
- **NPU 无等价机制** — 这是我们做 TDM 的根本原因

---

## 4. 关键实验结果

| 指标 | 数值 |
|------|------|
| 平均 goodput 提升 | 2.2× |
| 最大 goodput 提升 | 3.06× |
| 测试模型 | Llama-8B, Llama-70B (TP4), Qwen-235B (TP8) |
| 测试 GPU | A100, H100, H200 |
| Baseline | Sarathi-Serve (chunked prefill) |
| SLO 达成率 | Decode SLO 严格保证; Prefill SLO 作为附带效果 |

**对比 temporal-only variant**：
- 论文实现了增强版 temporal-only 方案（时分复用）
- 结果：**至少比 MuxWise 差 20%**
- 原因：temporal 方案无法在空间上利用 decode 的空闲 SM

---

## 5. 与 NPU-TDM 的关系

### 5.1 可借鉴的思想
1. **Decode 优先调度**：我们的 TDM 同样应保证 decode SLO 优先
2. **轻量级预测器**：线性模型预测 P/D 延迟的思路可直接复用（改为 NPU 上的 profiling）
3. **SLO-aware 动态调整**：每个 iteration 重新决策 P/D 比例的框架
4. **Contention 建模方法**：离线 profiling + 保守估计的思路

### 5.2 无法直接复用的部分
1. **GreenContext / SM 分区**：NPU 无等价的空间分区机制 → 我们的 TDM 本质区别
2. **P/D 同时执行**：MuxWise 是真正的空间并行，TDM 是时间交替
3. **Query-based 同步**：层级同步只在空间并行时有意义
4. **CUDA Graph + eager 混合**：NPU 上使用 ACL Graph，P/D 切换有 graph 重编译开销

### 5.3 NPU-TDM 的差异化优势
1. **无空间分区依赖**：适用于 NPU 及其他缺乏 SM 分区的加速器
2. **ACL Graph 感知**：针对 NPU graph 编译特性做 batch shaping 优化
3. **硬件状态感知**：利用 DCMI 采集 NPU 特有的硬件指标指导调度
4. **时分复用理论保证**：MuxWise 论文承认 temporal-only 差 20%，但那是在有空间分区的 GPU 上；在无空间分区的 NPU 上，TDM 是唯一可行路径

### 5.4 MuxWise 论文中关于泛化性的原文
> "MuxWise generalizes to accelerators that support intra-process spatial sharing with lightweight dynamic adjustment, such as GreenContext on NVIDIA GPUs (supported since Pascal) and hipExtStreamCreateWithCUMask() on AMD GPUs."

**明确排除了不支持空间分区的加速器（如 NPU）** — 这为我们的 TDM 方案提供了清晰的 motivation 和 positioning。

---

## 6. 论文局限性（可作为我们的切入点）

1. **依赖 GreenContext**：无法推广到 NPU 等加速器
2. **内存带宽竞争未管理**：仅靠保守估计容忍，可能浪费资源
3. **Prefill SLO 非显式保证**：只保证 decode SLO，prefill 是附带效果
4. **离线 profiling 依赖**：contention guard 需要预先 profiling，对新硬件/新模型需重做
5. **单卡约束**：空间复用仍受单卡 SM 总数限制，超大 prefill 可能 SM 不足
