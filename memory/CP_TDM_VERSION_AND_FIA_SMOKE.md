# CP/TDM 版本路径分析与 FIA Smoke 验证

> 状态：代码分析已完成；新版 FIA Smoke 尚待服务器实测。表中不得填写推测数据。

## 1. 旧版与新版：CP/TDM 的路径差异

### 1.1 核心差异

| 环境 | 方法与批次 | Attention state | 主要 Attention 路径 |
|---|---|---|---|
| v0.11.0rc1 | CP：初始pure-P | `PrefillNoCache` | Prefill专用Flash Attention |
| v0.11.0rc1 | CP：pure-D | `DecodeOnly` | Paged Attention |
| v0.11.0rc1 | CP：partial Prefill 或 P/D mixed | `ChunkedPrefill` | FIA：`npu_fused_infer_attention_score` |
| v0.11.0rc1 | TDM：首个 pure-P | `PrefillNoCache` | `_npu_flash_attention` |
| v0.11.0rc1 | TDM：后续 pure-P chunk | `PrefillCacheHit` | `_npu_flash_attention_qlens` |
| v0.11.0rc1 | TDM：pure-D | `DecodeOnly` | `_npu_paged_attention` |
| v0.13.0 | CP：mixed/partial Prefill | `ChunkedPrefill` | 默认 FIA |
| v0.13.0 | TDM：pure-P / pure-D | 对应各 state | 默认 FIA；仅特定 Decode 配置可切 PA |

旧版两者都按attention state作native dispatch，并不是CP的每个iteration都固定走FIA。
区别在于：CP中的partial-P和P/D mixed必然进入`ChunkedPrefill/FIA`，而TDM通过
phase-pure组织使pure-P/pure-D主要进入专用路径。旧实验由此同时改变了：

1. **调度组织：**P/D mixed 与 phase-pure；
2. **算子路径：**FIA 与 Prefill/Decode 专用算子。

所以旧数据能够证明 TDM 系统存在性能现象，但不能单独证明收益来自 phase-pure 调度。

![旧版与新版CP/TDM执行路径](../paper/figures/fig_version_path_comparison.png)

上图是**代码路径结构图**而非性能结果图。左侧说明旧实验同时改变调度组织和attention
dispatch mix；右侧给出新版需要完成的受控对照。当前没有图表能够证明旧性能差距由FIA
或phase-pure中的某一个单独造成，这正是统一FIA实验必须补充的原因。

v0.13.0 默认使用上游 V1 Scheduler，Attention 则主要统一为 FIA。移植 TDM 时只改变
每轮的 P/D 组成，即可形成更干净的对照：

```text
CP  = FIA + P/D mixed
TDM = FIA + pure-P / pure-D
```

注意：新版中 pure-P continuation 也可能被标为 `ChunkedPrefill`。因此不能只凭
attention state 判断 mixed，必须同时记录 `prefill_tokens` 和 `decode_tokens`。

### 1.2 为什么必须补实验

- **排除路径混淆：**验证统一 FIA 后，TDM 优势是否仍存在。
- **保证 baseline 公平：**对 CP 的 token budget `{512,1024,2048,4096}` 搜参。
- **验证真实执行：**记录每轮 P/D token、state 和 FIA/PA 调用，不能仅依据配置推断。
- **闭合论文核心：**只有在 best-tuned CP 与同 FIA TDM 之间比较，才能把结果归因于
  mixed 与 phase-pure 的组织差异。

代码依据：旧版 `vllm_ascend/attention/attention_v1.py` 的 state-specific dispatch；新版
`vllm_ascend/worker/model_runner_v1.py::_build_attn_state` 与
`vllm_ascend/attention/attention_v1.py::forward_impl`；上游
`vllm/v1/core/sched/scheduler.py::schedule`。

## 2. vLLM Chunked Prefill 与 Sarathi-Serve

### 2.1 相同的核心思想

二者都采用 token budget，并优先推进运行中的 Decode，再用剩余 budget 接纳或切分
Prefill，形成 P/D mixed batch。因此 vLLM V1 CP 是有效的 **Sarathi-style** 实现，
不是“只切 Prefill、不混合 Decode”的简化版本。

参考：[Sarathi-Serve（OSDI'24）](https://arxiv.org/pdf/2403.02310)、
[vLLM V1 Chunked Prefill 文档](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html#chunked-prefill)。

### 2.2 与本项目相关的差异

| 要点 | Sarathi-Serve | vLLM V1 CP | 对 PD-TDM 的影响 |
|---|---|---|---|
| 调度表达 | 显式 `Decode → partial-P → new-P` | 统一按请求未计算 token 推进，先 `RUNNING` 后 `WAITING` | 常规行为相近，复杂场景不保证逐项一致 |
| Budget 选择 | 根据 TBT SLO、硬件和并行方式 profiling | 使用 `max_num_batched_tokens`，默认不与 SLO 自动绑定 | CP 必须做 budget tuning |
| 执行后端 | NVIDIA + FlashAttention/FlashInfer | Ascend + FIA/可选 PA | 不能直接搬用 Sarathi 的性能结论 |
| 系统范围 | 强调 stall-free 与 PP 均匀微批次 | 通用生产调度，兼容抢占、Prefix Cache、推测解码等 | 本项目 TP=2 时主要比较单轮组批策略 |

![Sarathi-Serve与vLLM V1 Chunked Prefill的范围关系](../paper/figures/fig_vllm_cp_vs_sarathi.png)

这张关系图支持“二者属于同一调度思想家族”，但同时明确：共享decode-first、Prefill
chunk cap和P/D mixed batch，不等于完整系统实现相同。

![vLLM V1 CP中有界Prefill chunk的Ascend实测](../paper/figures/fig_vllm_cp_chunking_evidence.png)

该热图使用现有T6三seed中位数，展示`vLLM V1 CP-2048`相对
`vLLM V1 CP-8192`（在Prompt上限7000下基本不触发切分）的联合SLO达标率差值。code
长Prompt区域的大幅正收益与Sarathi提出的“有界Prefill减少Decode stall”机制一致；
conv严格SLO下的负值也说明chunking不是无条件占优。它能够作为**Sarathi-style核心机制
在Ascend/vLLM上的实证支持**，但不能写成“复现了完整Sarathi-Serve系统”。

论文中的准确名称应为：

> **vLLM V1 Chunked Prefill（Sarathi-style decode-protected mixed batching）**

除非运行或严格移植 Microsoft Sarathi-Serve，不应把该 baseline 直接写成
“Sarathi-Serve”。对 PD-TDM 而言，最重要的不是复刻其全部系统，而是在同一 Ascend
执行栈上公平比较：**best-tuned mixed CP 与 bounded phase-pure TDM**。

## 3. 新版默认 FIA 上的 TDM Smoke

### 3.1 目的与配置

Smoke 只验证新版 TDM 移植的正确性，不用于宣称性能优势。

| 项目 | 配置 |
|---|---|
| 硬件 | 2× Ascend 910B3，TP=2 |
| 软件 | CANN 8.5.0，vLLM/vLLM-Ascend 0.13.0 |
| 模型 | Qwen3-8B，BF16 |
| 长度与预算 | `max_model_len=8192`，`max_num_batched_tokens=2048` |
| TDM | `prefill_chunk_tokens=2048`，静态 ratio=0.3，动态 PID关闭 |
| 干扰项 | Prefix Cache、Spec Decode、Async scheduling关闭 |
| Attention | `pa_shape_list=[]`，保持默认 FIA |

每轮至少记录：

```text
prefill_tokens, decode_tokens, attention_state,
attention_operator, iteration_time_ms
```

### 3.2 最小用例

| 用例 | 请求方式 | 期望时间线 | 主要检查项 |
|---|---|---|---|
| S1 长 Prompt 切分 | 1个请求：Prompt=4096，Output=32 | `P≤2048 → P≤2048 → D...` | chunk上限、无mixed、输出正确 |
| S2 P/D竞争 | 先发16个 `128/512`；进入Decode后注入4个 `4096/32` | pure-D 与 pure-P 交替 | 无mixed、两阶段均获服务、无饥饿 |

### 3.3 Smoke 验收标准

| 用例 | 完成率 | 最大P chunk | Mixed迭代数 | 阶段要求 | FIA确认方式 | 输出要求 |
|---|---:|---:|---:|---|---|---|
| S1 长Prompt | 100% | ≤2048 | 0 | 出现pure-P后进入pure-D | worker日志/Profiler | 与原生vLLM一致 |
| S2 P/D竞争 | 100% | ≤2048 | 0 | pure-P和pure-D均出现且无饥饿 | worker日志/Profiler | 与原生vLLM一致 |

Smoke 通过条件：完成率100%、最大P chunk不超过2048、mixed迭代为0、FIA调用得到
运行时确认、输出与原生 vLLM 一致，并且S2不存在Prefill或Decode饥饿。任一条件失败，
都应先修复移植，不能进入正式 CP/TDM 性能比较。

## 4. Smoke 通过后的最小 CP/TDM 对比

### 4.1 目标与变量控制

该实验只回答一个方向性问题：

> 在 v0.13.0 默认 FIA、相同 2048 token cap 下，phase-pure TDM 是否仍可能比 mixed CP
> 获得更好的 TTFT/TPOT 联合表现？

| 配置 | CP-2048 | TDM-2048 |
|---|---|---|
| Scheduler | 上游 V1 CP | 新版 TDM Scheduler |
| P/D组批 | 允许 mixed | 每轮只能 pure-P 或 pure-D |
| Token cap | `max_num_batched_tokens=2048` | `prefill_chunk_tokens=2048` |
| Attention | 默认 FIA，`pa_shape_list=[]` | 默认 FIA，`pa_shape_list=[]` |
| 其他配置 | 与第3.1节相同 | 与第3.1节相同；静态 ratio=0.3 |

除 Scheduler 和 P/D 组批方式外，模型、请求序列、随机种子、KV配置和图模式必须完全
相同。该小实验暂不做 CP budget sweep，也不用于最终论文结论。

### 4.2 负载与运行方式

使用一个确定性的“Decode背景流 + 长Prompt突发”负载：

1. `t=0s`：发送16个背景请求，Prompt=256、Output=1024；
2. 观察到背景请求进入 Decode 后，记为 `t_inject`；
3. `t_inject`：同时注入8个长请求，Prompt=4096、Output=128；
4. 所有请求完成后结束；每种方法独立运行3次，先做1次不计入结果的warmup。

该负载直接构造“Decode已有服务压力，同时Prefill需要推进”的目标区域，无需先估计
服务器饱和QPS。若单轮总时长过短，可把上述注入重复3波，每波间隔10秒。

建议临时双SLO仅用于比较，不作为论文最终SLO：

```text
TTFT SLO = 1000 ms
TPOT SLO = 200 ms
联合达标 = 同一请求同时满足 TTFT 和 TPOT SLO
```

如果两种方法联合达标率都接近0%或100%，说明SLO没有区分度，应根据warmup分布调整，
而不是据此判断方法优劣。

### 4.3 必须采集的数据

- 请求级：TTFT、TPOT mean/p99、E2E、是否联合达标、完成状态；
- 迭代级：P/D token、mixed/pure类型、attention state/operator、iteration time；
- 系统级：总完成时间、输出token吞吐、preemption和错误数。

### 4.4 历史实测数据：新版实验的量级参考

> 以下数值来自 **v0.11.0rc1 + CANN 8.3.rc1** 的历史实验，不是 v0.13.0 FIA-TDM
> 新结果。它们用于展示结果格式和合理量级，不能作为新版实验的验收阈值。

#### T6 的 workload 与系统配置

“Azure burst”不是直接回放 Azure 原始时间戳，而是：

1. 从 Azure conv/code CSV 的有效行中，**联合采样**一组 `(Prompt tokens, Output tokens)`；
2. 忽略该行原始时间戳，使用周期性高/低峰到达过程生成请求时间；
3. 每次采样有放回，同一行的 Prompt/Output 配对关系得到保留；
4. 服务端使用 `ignore_eos`，使请求生成到 trace 指定的 Output token 数。

原始 conv/code CSV 分别包含19,365/8,818条请求；实验跳过 Prompt 大于7000或
Output大于600的行。过滤后分别剩19,164/8,314条有效请求，长度分布如下（单位均为
token）：

| Trace | 有效行数 | Prompt mean/p50/p90/p99 | Output mean/p50/p90/p99 | 平均P:O |
|---|---:|---:|---:|---:|
| conv | 19,164 | 1,153 / 1,018 / 2,743 / 4,141 | 206 / 127 / 420 / 540 | 约5.6:1 |
| code | 8,314 | 1,737 / 1,352 / 3,932 / 6,419 | 26 / 13 / 55 / 223 | 约67:1 |

因此，conv 是“中等Prompt、较长Output”，仍会产生较持续的Decode压力；code 是“长
Prompt、很短Output”，明显Prefill-heavy。两者测试的是不同的P/D压力组合，code尤其
不代表深Decode工作负载。

#### `k` 的含义

`k` 是无量纲的**到达强度缩放系数**，不是“千个请求”、chunk大小或PD ratio。它同时
乘在高峰和低峰QPS上，不改变Prompt/Output长度分布或SLO。理论平均到达率为
`k × [high × high_frac + low × (1-high_frac)]`。每个周期为10秒：

| Trace | 一个周期 | k=1基础QPS | 理论平均QPS | k取值 |
|---|---|---|---:|---|
| conv | 前1秒高峰，后9秒低峰 | high=12，low=5 | `5.7 × k` | 0.5、1.0、1.4、1.8、2.2、2.6、3.0 |
| code | 前2秒高峰，后8秒低峰 | high=10，low=1 | `2.8 × k` | 0.7、1.4、2.1、2.8、3.5、4.2、4.9 |

例如：

```text
conv k=3.0: high=36 QPS, low=15 QPS, 理论平均=17.1 QPS
code k=4.9: high=49 QPS, low=4.9 QPS, 理论平均=13.72 QPS
```

表中报告的是实际生成请求的三seed中位到达率，因此分别为16.833和13.733 req/s，
与理论值存在随机采样误差。每轮总注入时长为90秒，前30秒到达的请求仅作warmup，
请求级指标统计窗口为 `[30s, 90s)`；到达流总共覆盖9个burst周期，seed为0、1、2。

| 系统项 | T6配置 |
|---|---|
| 硬件/模型 | 2× Ascend 910B3，TP=2，Qwen3-8B |
| 软件 | CANN 8.3.rc1，vLLM-Ascend v0.11.0rc1 |
| 公共引擎参数 | `max_model_len=8192`，`gpu_memory_utilization=0.85` |
| vLLM CP（下表的c3-fair） | 补充实验中`max_num_batched_tokens=2048`，允许P/D mixed |
| PD-TDM（下表的m31-fix） | 服务器budget=8192，`prefill_chunk_tokens=2048`，phase-pure；`slo_pid`控制器，初始ratio=0.3，ratio范围[0.05, 0.8]，每个slice为2--8个iteration |
| 严格SLO | conv=TTFT/TPOT 200/120ms；code=500/200ms |

最初的T6主脚本其实还以budget=8192运行过一次CP；由于Prompt上限7000，此配置通常不
触发chunk，后续被作为`vanilla_cb`保留。下表采用的是后来单独补跑的CP-2048
（`c3-fair`）与修复chunk调度问题后重跑的TDM-2048（`m31-fix`），不能和最初的
CP-8192结果混用。

原始脚本的完整SLO网格为：conv `{200/120, 300/150, 500/200, 1000/200}` ms，
code `{500/200, 500/700, 2000/400, 3000/1500}` ms；最终点对点汇总保留前三档。
CP调度不读取SLO，因此只运行一次，再用请求级结果对各阈值作post-hoc重分类；PD-TDM
控制器读取双SLO，所以每档SLO均单独运行。下面选取各自严格SLO下的高负载代表点。

#### T6 历史结果

Azure长度采样 + 周期burst、2048-token bound、三seed中位数：

| Workload/严格SLO | 方法 | 到达率(req/s) | 联合达标率 | Goodput(tok/s) | TTFT mean/p99(ms) | TPOT mean/p99(ms) |
|---|---|---:|---:|---:|---:|---:|
| conv k=3.0，200/120ms | vLLM CP | 16.833 | 29.0% | 1,525.43 | 291.55 / 700.29 | 92.22 / 195.78 |
| conv k=3.0，200/120ms | PD-TDM | 16.833 | 63.8% | 3,395.85 | 212.94 / 632.52 | 87.76 / 172.75 |
| code k=4.9，500/200ms | vLLM CP | 13.733 | 6.5% | 87.10 | 406.98 / 913.31 | 207.54 / 228.19 |
| code k=4.9，500/200ms | PD-TDM | 13.733 | 80.4% | 438.03 | 315.06 / 725.76 | 168.88 / 190.12 |

历史 code k=2.8 调度遥测：

| 方法 | 代表性执行单元 | 工作组成 | 平均时长 | 补充信息 |
|---|---|---|---:|---|
| vLLM CP | chunk-cap mixed iteration | 约2030 P + 30 D token | 220.6ms | chunk-cap迭代占57.4% |
| PD-TDM | 1个pure-P + 1个pure-D cycle | 约2048 P + 29 D token | 189ms | pure-P约150ms，pure-D约39ms |

该遥测不是严格等工作量微基准，而且旧版两种方法的 Attention 路径不同，因此只能说明
历史执行形态，不能据此声称 mixed 本身固定产生31.6ms开销。新版最小实验应沿用上述
指标格式，并检查统一 FIA 后差距是否仍存在。

MaaS v12 长输入/短输出 replay 也可作为量级参考：

| 指标 | vLLM CP | PD-TDM |
|---|---:|---:|
| Wall clock/s | 25,554 | 23,841 |
| TTFT mean/P99(ms) | 952 / 1,316 | 808 / 1,180 |
| TPOT mean/P99(ms) | 236 / 257 | 222 / 252 |
| 原始3×SLO达标率 | 99.997% | 100.0% |
| 原始3×SLO Goodput(tok/s) | 164.2 | 164.3 |

数据来源：`results/phase_2_post/m31fix_phase1_pointwise.json`、
`results/c3_telemetry/`、`results/m31fix_validate/` 和
`results/maas_replay/V12_REPORT.md`。

### 4.5 可复用的历史图表

![历史T6严格联合SLO结果](../paper/figures/fig_t6_strict_slo.png)

上图适合说明：旧环境下 PD-TDM 在严格联合 SLO 中出现了明显端到端优势，但仍需新版
统一 FIA 实验排除路径差异。

![历史合成工作负载适用边界](../paper/figures/fig_f5_workload_boundary.png)

下图适合说明：PD-TDM 并非无条件占优；进入深 Decode 区域后，收益会缩小或反转。
两图中的 `Sarathi-Serve` 是旧稿简称，准确含义是项目中的 **vLLM V1 CP
（Sarathi-style）**，正式论文应重新生成图例。

### 4.6 新版小实验结果如何解释

| 实测现象 | 初步结论 | 下一步 |
|---|---|---|
| 同FIA下TDM联合达标率更高，且长请求TTFT改善、背景TPOT仍满足SLO | phase-pure核心假设获得初步支持 | 开展CP budget sweep和正式trace实验 |
| TDM改善TTFT但明显破坏背景TPOT | pure-P slice过长或ratio过高 | 调小chunk/ratio，再验证适用区域 |
| TDM与CP基本相同 | 旧版优势可能依赖特定负载或旧执行路径 | 做等工作量Mixed-FIA vs Pure-FIA微基准 |
| TDM稳定更差 | 当前核心A不成立或移植策略不合适 | 停止扩展实验，先分析iteration timeline |

该小实验只能作为“是否值得继续”的门槛。只有后续完成 CP budget tuning、代表性trace、
多seed和双SLO Goodput对比后，才能形成论文级结论。
