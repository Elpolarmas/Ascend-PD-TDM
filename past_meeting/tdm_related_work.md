# 时分复用相关工作梳理 — 与 NPU PD-TDM 的对位分析

> 来源：Dilu 引用链中 5 篇核心 TDM 论文

---

## 一、五篇论文速览

| 论文 | 场景 | 核心思路 |
|---|---|---|
| **FaST-GShare** [19] | Serverless 推理 | 每个 function 分配 token 配额，CUDA Hook 拦截 kernel launch，配额耗尽则阻塞 |
| **GaiaGPU** [20] | 容器云通用 | 将 GPU 虚拟化为 vGPU 单元，API 拦截层做弹性/动态资源分配 |
| **TGS** [47] | DL 训练混部 | AIMD 速率控制（类 TCP 拥塞控制），kernel 级别限流，保护生产任务 |
| **FaaSwap/Torpor** [54] | Serverless 多模型推理 | 模型粒度的 swap in/out + 晚绑定 + 自定义 remoting，减少冷启动 |
| **Dilu** (ASPLOS'25) | Serverless DL 混合 | 内核级 token + KLC 实时自省 + 2D co-scaling（快调时分 + 慢调空间） |

---

## 二、七维度对比

### 维度 1：时分粒度

| 方案 | 粒度 | 机制 |
|---|---|---|
| FaST-GShare | ms 级窗口配额 | 每个时间窗口内分配固定 token 数，耗完阻塞到下一窗口 |
| GaiaGPU | 中粒度（运行时） | vGPU 单元的动态再分配，非 kernel 级 |
| TGS | kernel 级 | AIMD 控制每个 job 的 kernel 发射速率 |
| FaaSwap | 粗粒度（模型级） | 整个模型 swap in/out，切换开销大 |
| Dilu | ~5ms token 周期 | 最细粒度，每 5ms 重新分配 <request, limit> |
| **NPU PD-TDM（ours）** | **iteration 级（~20ms）** | 每个 scheduler iteration 决定 P 或 D phase |

**对我们的启示**：
- 我们的粒度（iteration ~20ms）介于 Dilu（5ms）和 FaaSwap（模型级）之间
- 不追求极细粒度（kernel 级），因为 NPU 的 ACL Graph 机制要求整个 iteration 作为原子执行单元
- **iteration 是 NPU 上最自然的时分边界**：一次 graph 执行 = 一个 iteration，在此粒度切换零额外开销

### 维度 2：切换/抢占机制

| 方案 | 抢占方式 | 隔离机制 |
|---|---|---|
| FaST-GShare | 隐式阻塞（token 拒绝） | MPS 提供空间隔离 |
| GaiaGPU | 软/硬限动态调整 | vGPU 单元边界 |
| TGS | 优先级速率控制（无显式抢占） | AIMD 反馈隐式隔离 |
| FaaSwap | 优先级队列 + 异步 swap | 模型粒度的物理隔离 |
| Dilu | 状态机自适应 + KLC 实时反馈 | EMERGENCY 等状态驱动紧急切换 |
| **NPU PD-TDM** | **SLO-slack 优先级规则** | **Phase 级时间隔离（P/D 不混合在同一 iteration）** |

**对我们的启示**：
- Dilu 的状态机思想可借鉴：NORMAL → THROTTLE → EMERGENCY 的渐进式响应
- 映射到我们的场景：NORMAL（按规则切换）→ URGENT（TTFT 即将违约，强制 PREFILL）→ EMERGENCY（TPOT 即将违约，强制 DECODE）
- 我们的隔离是最干净的——P/D 在不同 iteration 执行，完全无干扰；其他方案都是在**共享执行中限流**

### 维度 3：空间 + 时间复用结合

| 方案 | 时分 | 空间 | 结合方式 |
|---|---|---|---|
| FaST-GShare | token 配额 | MPS 分区 | 时空矩形匹配（static） |
| GaiaGPU | API 拦截限流 | vGPU 单元 | 动态 vGPU 调整 |
| TGS | AIMD 限速 | 无 | 纯时分 + 内存超售 |
| FaaSwap | 模型 swap | 无 | 纯时分 |
| Dilu | token 周期 | collocation | **2D co-scaling（快调时分 + 慢调空间）** |
| **NPU PD-TDM** | **iteration phase** | **无（NPU 无 SM partition）** | **纯时分，Graph-Aware 作为"类空间"优化** |

**对我们的启示**：
- NPU 没有 MPS / SM partition / CU masking → 空间复用不可行 → **时分是唯一路径**
- 但 Graph-Aware Batch Shaping 可类比为"时间维度上的空间优化"——通过控制每个 iteration 的 batch size 来最优化硬件利用
- Dilu 的 2D co-scaling 思想可简化映射：**快调 = phase ratio（ms 级切换），慢调 = KV cache 预算重分配（需内存操作）**

### 维度 4：Profiling 与自适应

| 方案 | Profiling | 在线自适应 |
|---|---|---|
| FaST-GShare | 自动 profiling + 静态 MPS 配置 | 无（静态） |
| GaiaGPU | 简单 vGPU 单元统计 | 有限（周期性调整） |
| TGS | 无 profiling | AIMD 反馈控制（自适应） |
| FaaSwap | RRC 历史估算 + Buddy allocator | 有（基于历史） |
| Dilu | **多因子 profiling + hybrid/binary search** | **KLC 实时自省（最精细）** |
| **NPU PD-TDM** | **离线 Ascend Profiler 画像表** | **DCMI AICore 在线采样（~0% 开销）** |

**对我们的启示**：
- Dilu 的多因子 profiling 最接近我们的思路，但它面向通用 DL 任务，我们专注 LLM P/D 场景
- 我们的 profiling 更轻量：只需要 (batch_size, phase) → latency 的二维画像，不需要 Dilu 那样扫描全部 <request, limit> 组合
- TGS 的 AIMD 思想可借鉴到 Phase Switching V2：将 prefill 插入量看作"发送速率"，SLO violation 作为"拥塞信号"，用 AIMD 自适应调节

### 维度 5：SLO 保障机制

| 方案 | SLO 形式 | 保障强度 |
|---|---|---|
| FaST-GShare | 吞吐目标（隐式） | 弱（静态配额） |
| GaiaGPU | QoS 等级 | 中（软限/硬限） |
| TGS | 生产任务优先级 | 中（AIMD 渐进保护） |
| FaaSwap | 延迟 SLO | 强（优先级队列 + 异步 swap） |
| Dilu | **延迟 SLO + goodput** | **最强（KLC 实时监控 + EMERGENCY 状态）** |
| **NPU PD-TDM** | **TTFT SLO + TPOT SLO** | **强（SLO slack 驱动，双指标约束）** |

**对我们的启示**：
- 我们是**唯一同时建模 TTFT 和 TPOT 双 SLO 的时分方案**
- 其他方案要么只有单一延迟指标（FaaSwap），要么 SLO 是隐式的（TGS 的"不影响生产"）
- Dilu 的 goodput 定义最接近我们（完成率 × SLO 达标率），可以作为评估指标参考

### 维度 6：目标场景差异

| 方案 | 场景 | 共享对象 |
|---|---|---|
| FaST-GShare | Serverless 推理 | 多个 function 共享 GPU |
| GaiaGPU | 容器云 | 多个容器共享 GPU |
| TGS | DL 训练 | 生产训练 + 机会训练 |
| FaaSwap | Serverless 推理 | 多个模型共享 GPU |
| Dilu | Serverless DL | 推理 + 训练混合 |
| **NPU PD-TDM** | **LLM Serving** | **同一模型的 P/D 两阶段共享 NPU** |

**关键区别**：所有 5 篇论文的 TDM 都是**多任务/多模型共享 GPU**，我们的 TDM 是**单模型内 P/D 两阶段共享 NPU**。这是本质性的差异：
- 多任务 TDM：切换涉及上下文切换、可能的模型 swap → 开销大
- P/D TDM：切换只是调度器选择不同的请求集合 → **零切换开销**（共享同一模型权重和 KV cache）

### 维度 7：与编译图的关系

| 方案 | 编译图感知 |
|---|---|
| FaST-GShare | 无 |
| GaiaGPU | 无 |
| TGS | 无 |
| FaaSwap | 无 |
| Dilu | 无 |
| **NPU PD-TDM** | **ACL Graph shape-aware（原创）** |

**所有 5 篇论文均不涉及编译图约束**，这进一步确认了 Graph-Aware Batch Shaping 的原创性。

---

## 三、核心借鉴点总结

### 从 Dilu 借鉴

| Dilu 机制 | 我们的映射 | 适配修改 |
|---|---|---|
| 多因子 Profiling | 离线 (bs, phase) → latency 画像表 | 维度大幅简化（2D vs Dilu 的多维） |
| KLC 实时自省 | DCMI AICore 在线采样 | 采样频率 100ms vs Dilu ~5ms |
| 2D Co-scaling | 快调 phase ratio + 慢调 KV cache 预算 | 慢调维度是 NPU 特有的 |
| EMERGENCY 状态机 | SLO slack 阈值驱动的优先级切换 | 从通用 DL 特化到 TTFT/TPOT 双指标 |
| 互补调度 | P=compute, D=memory → 交替执行 | 最直接的对应 |

### 从 TGS 借鉴

| TGS 机制 | 我们的映射 |
|---|---|
| AIMD 速率控制 | Phase Switching V2：prefill 插入率用 AIMD 调节 |
| "拥塞信号" | TPOT SLO violation = 拥塞，触发 multiplicative decrease（减少 prefill） |

### 从 FaST-GShare 借鉴

| FaST-GShare 机制 | 我们的映射 |
|---|---|
| Token 配额机制 | 每个 phase 的 iteration 配额（如连续最多 N 个 prefill iteration） |
| 时空矩形匹配 | Graph shape × phase 的最优组合选择 |

---

## 四、我们方案的差异化定位

综合对比后，本方法的**独特定位**：

1. **PD 动态混部范畴内首个兼顾硬件普适性与迭代粒度调度灵活性的方法**：空分路径硬件依赖，时分路径现有调度粒度偏粗
2. **在 iteration 粒度上引入 SLO 余量反馈控制**：现有时分方案（含 B 线五篇）多采用跨多个迭代生效的固定比例或启发式策略，缺乏随每次迭代更新的反馈机制
3. **同时建模 TTFT + TPOT 双 SLO**：LLM serving 特有的双延迟指标，B 线五篇均不涉及
4. **对预编译图加速器的落地适配**：配套图识别控制模块，B 线五篇均不涉及编译图约束
5. **最干净的时间隔离**：P/D 在不同 iteration 执行，无同 batch 干扰（vs 其他方案的"限流但仍共享"）；且共享同一模型权重与 KV cache，切换零额外开销

**一句话概括**：B 线现有 TDM 工作解决的是"多租户如何共享 GPU"，本方法解决的是"一个 LLM 服务内的 P/D 两阶段如何在通用加速器（含无空间分区硬件）上兼顾硬件普适性与迭代粒度调度灵活性地共享单卡"。问题不同，但调度思想（配额、自省、自适应）可以借鉴。
