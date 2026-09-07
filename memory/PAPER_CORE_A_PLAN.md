# 论文核心 A：写作与验证路线图（D-016 历史版本）

> 状态：**已被 D-017（2026-08-13）取代，不再作为执行入口。** 当前 ICASSP 2027
> 投稿任务以 `paper/TASKS.md` 为准，四页结构以 `paper/WRITING_PLAN.md` 为准。
> 本文保留 D-016 的代码路径、Attention dispatch、旧 FIA 消融审计和条件性实验设计，
> 供后续任务引用；其中把 Force-FIA、等工作量微基准和 best-tuned CP 全部设为投稿前
> 硬门槛的安排已经失效。

## 1. 当前判断

项目已经具备完整系统实现、丰富的端到端数据和明确的适用区域，足以继续撰写中文
论文草稿；但若要把结果提升为具有较强审稿说服力的算法/系统贡献，还缺少核心因果
闭环：

> 在公平调优并统一 Attention backend 后，bounded phase-pure scheduling 是否仍比
> Sarathi-style mixed Chunked Prefill 获得更高的双 SLO Goodput？

因此，当前项目状态应描述为：

> 已有“完整系统 + 强端到端现象”，尚缺“排除 baseline 调优和旧版执行路径两个替代
> 解释”的关键验证。

论文核心固定为 **A：bounded phase-pure scheduling**。动态 PD ratio 属于后续优化方向，
除非未来得到新的直接证据，否则不承担当前论文第一贡献。

## 2. 术语与 baseline

| Paper 名称 | 项目配置 | 准确定义 |
|---|---|---|
| Traditional P-first Coupling | `c1_baseline` | AscendScheduler 的 admit-driven、Prefill-first、phase-pure 路径 |
| vLLM V1 Chunked Prefill (Sarathi-style) | `c3_cp` | 上游 V1 request-progress Scheduler；Decode/运行请求优先，剩余 token budget 用于切分 Prefill，允许 P/D mixed batch |
| PD-TDM | `c2_tdm_m31_2048_fix` 等 | 保留 Prefill chunk cap，但显式选择 pure-P 或 pure-D iteration |
| PD Disaggregation reference | `c4_pd` | 2 NPU 上的 1P1D 资源受限参考点 |

不得再把 `c3_cp` 直接命名为 **Sarathi-Serve**。准确写法为：

> vLLM V1 Chunked Prefill（采用 Sarathi-style 的 decode-protected mixed scheduling）

vLLM V1 CP 已实现 Sarathi 的核心在线调度思想，但并不是完整 Sarathi-Serve 系统的原样
代码；两者在统一 Scheduler 抽象、token-budget 选择、GPU/NPU backend、PP 支持和工程
功能上存在差别。

## 3. 已确认的代码事实

### 3.1 `c3_cp` 调度路径

1. `c3_cp` 使用 `TDMScheduler`，但 `enable_tdm=False`、`passive_tracker=True`、
   `prefill_chunk_tokens=None`，因此 TDM controller 和自定义 `schedule_chunked()` 不生效。
2. `TDMScheduler.schedule()` 调用 `super().schedule()`。
3. `AscendScheduler.schedule()` 发现 `chunked_prefill_enabled=True` 后立即调用上游 V1
   `Scheduler.schedule()`。
4. V1 Scheduler 先推进 `running` 请求，再使用剩余 budget 接纳/切分 waiting Prefill。
5. 当前纯文本、默认阈值配置通常形成“已有 Decode → 一个 partial Prefill → 新请求”的
   Sarathi-style 行为；运行时仍需记录 batch 组成作为论文证据。

### 3.2 Attention dispatch

当前 CP 和 TDM 都使用 v0.11.0rc1 的 native state dispatch，而不是每个 iteration 固定走
同一个 kernel：

| Scheduler / batch | Attention state | 执行路径 |
|---|---|---|
| CP 或 TDM：初始纯 Prefill | `PrefillNoCache` | Flash Attention 专用路径 |
| CP 或 TDM：纯 Decode | `DecodeOnly` | Paged Attention 专用路径 |
| CP：partial Prefill | `ChunkedPrefill` | `_forward_v1_style` / FIA |
| CP：P/D mixed | `ChunkedPrefill` | `_forward_v1_style` / FIA |
| TDM：后续 pure Prefill chunk | `PrefillCacheHit` | Prefill 专用路径 |

所以“C3 始终 FIA”是不严格的；只有 mixed/partial-prefill iteration 必然进入 FIA。在饱和
负载下它可能占主要执行时间，但必须由运行时 state/time 分布验证。

### 3.3 旧 `m31_fia` 消融的状态

旧实现从 `TDMScheduler.__init__` 安装 monkey patch，但 V1 Engine 先创建 Model Executor / TP
worker，再创建 Scheduler。该 patch 很可能没有传播到已启动的模型 worker。

因此，旧结论“强制 FIA 后性能几乎不变，所以 kernel/path 贡献约等于零”当前视为
**未验证**，不可用于论文归因。必须以 worker 日志、调用计数或 NPU profiler 证明 FIA
真实执行后，才可重新启用该结论。

## 4. 核心 A 的理论表述

当前建议的一句话：

> Chunked Prefill 通过限制每个 iteration 的 Prefill 工作量保护 Decode，但通常仍将 P/D
> 组织为 mixed batch。PD-TDM 保留这一 chunk 上限，同时将 P/D 组织为有界的
> phase-pure slices；当 Decode 具有 TPOT 余量、Prefill 存在服务压力且 mixed execution
> 存在额外代价时，系统可以用 Decode 余量集中推进 Prefill，从而提高联合 SLO Goodput。

设等量工作为 `P` 个 Prefill token 和 `D` 个 Decode token：

```text
T_mix(P,D)              : 一个 mixed iteration 的时间
T_pure(P,D) = T_P(P) + T_D(D) + T_switch
delta_mix = T_mix - T_pure
```

核心假设 H1：在统一 backend 和等工作量下，目标区域存在 `delta_mix > 0`。

这并不意味着 PD-TDM 无条件更好。Pure-P 会暂停 Decode，因此还需要满足：

```text
连续 P slice 造成的 decode gap <= 可用 TPOT 预算
```

ratio 的准确含义是 **控制 P iteration 的长期出现频率/信用**，而不是直接决定单个 P
iteration 的时长：

```text
prefill_chunk_tokens  -> 单次 P iteration 的工作上限
PD ratio/token bucket -> P iteration 的长期频率
urgency/starvation    -> 短期双 SLO 风险保护
```

理论上应导出三个可证伪预测：

1. **H1，等工作量效率：**同 FIA 下，目标区域 `T_P + T_D < T_mix`。
2. **H2，winning region：**Prefill 压力、长 Prompt、TTFT 紧且 TPOT 有余量时，TDM
   优势增大。
3. **H3，负对照：**Decode-heavy、TPOT 极紧、低负载或 mixed cost 很小时，优势缩小或
   消失。

## 5. 论文工作线

### 5.1 服务器不可用期间可完成

1. 固定上述术语、问题定义和条件化核心假设。
2. 建立 claim-evidence 表，区分“已确认事实 / 待验证假设 / 暂不作为贡献”。
3. 撰写不会因 H1 结果改变的章节：
   - 背景与相关工作；
   - 双 SLO Goodput 问题定义；
   - 传统 coupling、vLLM-CP、PD 分离和 PD-TDM 的实现区别；
   - PD-TDM 系统设计与实现；
   - 实验环境、workload、SLO 和指标；
   - 已有数据的客观整理；
   - 局限性与待验证问题。
4. 重整已有实验的论文角色：MaaS 为现实主 workload，balanced 排除极端 Prefill 偏置，
   decode-heavy 为负对照，历史大矩阵提供广度。
5. 准备不同验证结果下的 Motivation/Contribution 分支。

### 5.2 暂不定稿

- 摘要中的核心性能原因；
- 引言最后的贡献列表；
- “mixed-batch tax 已证实”类表述；
- “优于 Sarathi-Serve”类表述；
- 最终标题和 Conclusion。

在关键实验前，统一使用“我们提出并将验证以下假设”的条件化写法。

## 6. 补充实验工作线

### E0. 静态路径审计固化

形成可写入论文的映射表：

```text
实验配置 -> Scheduler -> batch 形态 -> Attention state -> NPU 算子
```

同时为关键调度语义补单元测试，防止后续修改改变 baseline。

### E1. 运行时可观测性

每个 iteration 至少记录：

```text
prefill_tokens
decode_tokens
num_prefill_requests
num_decode_requests
attention_state
batch_total_tokens
iteration_time
waiting_depth
running_depth
```

输出 count fraction、time fraction 和 token fraction。目的：确认 `c3_cp` 中 mixed/FIA、
pure Prefill 和 Decode-only 各占多少，而不是假设 C3 全程 FIA。

### E2. Worker 级 Force-FIA

将 backend 控制放到模型 worker 初始化路径，而不是 Scheduler monkey patch。每个 TP rank
打印 backend，并加入 Attention state/operator 计数；正式运行使用 NPU profiler 复核。

配置命名：

| 名称 | 状态 | 含义 |
|---|---|---|
| `CP-NativeDispatch` | 已有 | 当前 `c3_cp`；mixed/partial 走 FIA，pure P/D 走专用路径 |
| `TDM-NativeDispatch` | 已有 | 当前 PD-TDM；pure P/D 走专用路径 |
| `TDM-ForceFIA` | 需修复并验证 | TDM 的所有 state 强制进入 FIA |
| `CP-ForceFIA` | 条件性新增 | 仅在 CP 的 pure iteration 时间占比显著时需要 |

### E3. 等工作量机制微基准（核心）

最小首轮：

```text
总 token budget B = 2048
Decode batch D = {8, 32, 64}
Prefill tokens P = B - D
固定 Decode context
预热后每点重复约 50 次
```

比较：

```text
Mixed-FIA: P Prefill + D Decode，一次执行
Pure-FIA : P Prefill iteration + D Decode iteration
```

报告 `T_mix`、`T_P + T_D`、P50/P95 和波动。首轮稳定后再扩展
`B={512,1024,2048}` 和多个 context length。

### E4. 公平调优 vLLM-CP

短时单 seed 搜索：

```text
max_num_batched_tokens = {512, 1024, 2048, 4096}
```

优先在 MaaS、balanced、decode-heavy 的接近饱和点搜索；选出每组 SLO 下最佳 budget 后，
只对最佳配置做正式多 seed 运行。最终比较 best-tuned CP，而不是只比较固定 2048。

### E5. 代表性端到端验证

不重跑完整历史矩阵。新实验主要负责因果性，历史数据负责广度。

```text
配置：best-tuned CP / TDM-ForceFIA / TDM-NativeDispatch
workload：MaaS / balanced / decode-heavy
SLO：TTFT紧且TPOT有余量 / TPOT紧
正式点：3 seeds
```

主要指标：联合 SLO Goodput、meet ratio、TTFT/TPOT P50/P99、请求吞吐、iteration time、
preemption 和 Attention state/time distribution。

## 7. 结果判定分支

### 分支 A：核心成立

如果 `TDM-ForceFIA > best-tuned CP`，且等工作量微基准满足 `T_P+T_D<T_mix`：

> 可将 bounded phase-pure scheduling 作为论文核心贡献；Native 路径只是额外工程收益。

### 分支 B：端到端赢，但微基准不快

说明收益可能来自服务时机、队列等待和 TTFT 资源分配，而不是单步 mixed cost。论文仍可
成立，但 Motivation 必须转向双 SLO 下的服务时机控制。

### 分支 C：统一 FIA 后差距消失

现有优势主要来自旧版 Ascend phase-specific dispatch。论文只能收缩为 NPU
scheduler-kernel co-design，并需要更重视版本可迁移性。

### 分支 D：调优 CP 后差距消失

现有核心 A 不成立；需要改进 TDM、寻找更窄的 winning region，或转向 mixed/pure 自适应
选择。不得继续使用“phase-pure 普遍更优”的叙事。

## 8. 非阻塞项

当前不阻塞核心 A：

- 完整移植 Sarathi-Serve；
- 重跑所有历史矩阵；
- 立即迁移最新版 vLLM-Ascend；
- 动态 ratio 重新设计；
- 新增更多数据集、模型或硬件；
- Sarathi 的 Pipeline Parallelism 复现。

新版 vLLM-Ascend 可在核心 A 明确后用一个代表点做 portability 验证。

## 9. 执行顺序与完成标准

### 服务器恢复前

1. 完成论文稳定章节与 claim-evidence 表。
2. 实现 E1 telemetry。
3. 修复 E2 worker 级 Force-FIA。
4. 准备 E3 微基准和 E4 budget sweep 脚本。
5. 不定稿摘要、Motivation、Contribution 和 Conclusion。

### 服务器恢复后

1. Smoke test：验证 state telemetry、正确性和 FIA operator。
2. 跑等工作量微基准。
3. 跑 CP budget 短时搜参。
4. 根据 CP pure iteration 的 time fraction 决定是否需要 `CP-ForceFIA`。
5. 跑代表性端到端三 seed。
6. 将微观时间差、服务率、队列变化和 Goodput 串成因果链。
7. 根据 §7 的分支重写摘要、引言、Motivation、贡献和 Conclusion。

核心 A 只有在以下条件全部满足后才算验证完成：

- CP batch 与 Attention state 得到运行时确认；
- CP 得到公平 token-budget 调优；
- TDM-ForceFIA 在所有 TP worker 中真实生效；
- Mixed-FIA 与 Pure-FIA 完成等工作量对比；
- 同 FIA 后的端到端双 SLO Goodput 得到验证；
- Decode-heavy/TPOT 紧场景提供负对照；
- 正式结果跨 3 seeds 方向稳定。
