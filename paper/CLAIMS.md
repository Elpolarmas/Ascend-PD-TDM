# PD-TDM Claim–Evidence Ledger

> D-020审查更新（2026-09-07）：本表已有证据评级不等于本轮完成全部raw复算。
> 进入R0的数字必须通过TASKS T1；C7为探索性边界。执行日期以TASKS为准。

状态定义：`SUPPORTED` 可按限定范围写入论文；`PROVISIONAL` 仍需分析或补实验；
`INTERNAL` 只描述实现；`REJECTED` 不得作为论文主张。

## 1. 核心主张

| ID | 状态 | 可写主张 | 主要证据 | 限定与风险 | 章节 |
|---|---|---|---|---|---|
| C1 | SUPPORTED | PD-TDM 在单个共享 TP 副本内保留 bounded chunking，并以 phase-pure iteration 显式分配 P/D 服务机会 | 当前 TDM 代码、T6 config、iteration telemetry | 不声称首次提出时间分相；贡献是有限资源场景下的细粒度设计与验证 | Design |
| C2 | SUPPORTED | PD-TDM 是“状态耦合、执行解耦”：权重和 KV 保持本地，P/D 不在同一 iteration 执行 | 代码路径、遥测 phase invariant | 不属于空间并发，也不等于完整 Prompt 后再 Decode 的粗粒度串行 | Motivation/Design |
| C3 | PROVISIONAL | 代表性 cell 中 mixed iteration 约 220ms，pure-P/pure-D 观测呈现不同执行形态 | `c3_telemetry/`、m31 iter JSONL | 不是等工作量对照，不得推导固定 15% mixed tax 或 kernel 根因 | Evaluation |
| C4 | SUPPORTED | 在 Azure 长输入/偏 Prefill burst workload 的严格双 SLO 下，PD-TDM 相对 vLLM-Ascend CP 提高 attainment/goodput | T6 m31-fix 3-seed matrix | 只覆盖 Qwen3-8B、2×910B3、chunk=2048 和给定 trace/SLO | Evaluation |
| C5 | SUPPORTED | 可见收益主要表现为 TTFT/TPOT 分布相对联合 SLO 的移动，而不是无条件提升 raw throughput | F5、T6、MaaS v12 | raw throughput 接近只按对应实验陈述 | Motivation/Evaluation |
| C6 | PROVISIONAL | 适用边界由 Prefill 压力、Decode 压力、到达负载和双 SLO 余量共同决定，不是 Prompt/Output ratio 的单调函数 | F5b 反例、F5c/F5d 数据 | F5c/F5d 尚未统一 post-hoc；主要为单 seed | Motivation/Evaluation |
| C7 | PROVISIONAL | 在 deep-decode 饱和区，PD-TDM 优势消失并可能低于 vLLM-Ascend CP（仅探索性信号） | F5b A2 | 单 seed且高负载存在 PoolTimeout；只作为边界信号 | Evaluation/Limitations |
| C8 | SUPPORTED | MaaS v12 上 PD-TDM 的 TTFT mean/p99 为 808/1180ms，vLLM-Ascend CP 为 952/1316ms | MaaS v12，两次大规模 replay 方向一致 | 聚合 trace、log-normal 重建、单 seed；只称流量形态接近 Agent 服务 | Evaluation |
| C9 | SUPPORTED | MaaS 原始 ×3 SLO 近 ceiling；在 ×1.2 post-hoc SLO 下 PD-TDM goodput 高约 50% | V12 report/raw requests | 必须明确是 sensitivity analysis，不是原始预设 SLO 结果 | Evaluation |
| C10 | SUPPORTED | 两 NPU 预算下 1P1D reference 受到 TP=1 Prefill capacity 和 KV/proxy 开销限制 | c4_pd burst + steady | 不代表多节点 P/D 分离系统的一般表现 | Evaluation/Limitations |
| C11 | SUPPORTED | ratio 是 P/D 竞争时的阶段调度倾向，提供双 SLO 资源交换接口 | selector 实现、T6/MaaS 配置 | 不等于 wall-clock 份额；当前在线控制器尚未证明能找到最优 ratio | Motivation/Design/Evaluation |

## 2. 方法贡献

1. **Problem formulation**：指出在额外设备和设备内空间分区都不可得时，仍需在单个
   TP 模型副本上满足 TTFT/TPOT 双 SLO；chunk bound 只约束单次 Prefill 工作量，不能
   独立表达跨 iteration 的阶段服务份额。
2. **System design**：实现“状态耦合、执行解耦”的细粒度 P/D 时分复用，以 bounded
   pure-P/pure-D iteration、显式 ratio、队列 fallback 和本地 KV 状态交换阶段机会。
3. **Boundary-oriented evaluation**：通过 Azure、synthetic matrix 和 MaaS trace 展示
   联合 Goodput 收益、SLO ceiling、长输入短输出胜区及 deep-decode 反例。

不使用“首次提出 P/D 时分复用”或等价措辞。EcoServe 已采用同实例时间分离；PD-TDM
聚焦单 TP 副本、iteration 级细粒度调度、双 SLO 资源交换和适用边界。

## 3. 机制因果链

```text
资源仅够一个完整 TP 模型副本，且无法可靠进行设备内空间分区
        ↓
mixed Chunked Prefill 以 batch token budget 隐式决定 P/D 服务量
        ↓
PD-TDM 将 chunk 大小和跨 iteration 阶段份额解耦
        ↓
当 Prefill 压力占主导且 Decode 仍有 TPOT 余量时，增加 pure-P 服务机会
        ↓
等待 Prompt 更快推进，TTFT 分布左移，同时 TPOT 保持在阈值内
        ↓
更多请求跨过 TTFT/TPOT 联合 SLO
        ↓
联合 SLO Goodput 提升
```

当 Decode 队列深或 TPOT 无余量时，上述条件不成立，pure-P 引入的 Decode silence 可能
降低 Goodput。mixed/pure kernel、tiling、KV layout 等只可作为候选解释，除非完成等工作量
受控实验。

## 4. 明确不能声称的内容

| ID | 状态 | 禁止表述 | 原因 |
|---|---|---|---|
| N1 | REJECTED | 串行 P/D 或 temporal disaggregation 本身是新机制 | EcoServe 已明确采用同实例 temporal disaggregation；MuxServe 也讨论阶段感知时空复用 |
| N2 | REJECTED | PD-TDM 总是优于 vLLM-Ascend CP | F5b deep-decode 区存在反例 |
| N3 | REJECTED | PID/双回路是当前性能来源或能在线求最优 ratio | P1.9d/e 与 T6 telemetry 不支持 |
| N4 | REJECTED | PD-TDM 用 tail latency 换 mean latency | m31-fix 数据中 mean/tail 可同步改善 |
| N5 | REJECTED | dedicated kernel 速度解释主要收益 | 同 kernel ablation 不支持 |
| N6 | REJECTED | mixed execution 具有已证实的固定约 15% tax | pure/mixed 不是等工作量受控对照 |
| N7 | REJECTED | PD-TDM 是空分或并行 P/D | 当前实现是共享 TP 设备上的时间分相 |
| N8 | REJECTED | 1P1D 结果证明 disaggregation 普遍较差 | 当前两卡预算不在典型多节点应用域 |
| N9 | REJECTED | ratio 是严格 wall-clock P:D 占比 | P/D iteration 时长不同且队列 fallback 会改变实际份额 |
| N10 | REJECTED | MaaS trace 证明请求来自 Agent | 源数据只有分钟级聚合长度与到达统计 |

## 5. 关键证据缺口

| Gap | 影响的 claim | 最小关闭方式 |
|---|---|---|
| F5c/F5d 没有统一分析 | C6 | 确定统一 metric/SLO，生成脚本、表和 heatmap |
| chunk=2048 选择攻击面 | C1/C4/C6 | vLLM-Ascend CP 与 PD-TDM 共享 chunk-budget sensitivity |
| mixed/pure 因果隔离不足 | C3 | 等 Prefill/Decode 工作量 controlled microbenchmark |
| synthetic matrix 单 seed | C6/C7 | 胜/边界/负三个无 timeout 代表 cell 各补 3 seed |
| 单模型/单平台 | C4/C6 | 服务器恢复后补小模型或另一硬件平台 |
| 静态 ratio 的实证交换价值未验证 | C11 | 静态 sensitivity 或将 claim 限为配置接口；动态控制不进入 R0 |

## 6. 当前无补实验条件下的 claim 策略

- **Primary evidence**：T6 3-seed m31-fix 主结果，可承载定量 headline。
- **Mechanism evidence**：phase telemetry 用于证明阶段纯净和 Decode interval，不做 kernel
  根因或固定 mixed tax 断言。
- **Exploratory boundary evidence**：F5--F5d 用于展示趋势和反例，精确百分比不强泛化。
- **External-validity evidence**：MaaS v12 用于支持长输入短输出形态中的 TTFT 方向一致性，
  原始 SLO 与 post-hoc sensitivity 分开报告。

9/9英文R0以已审计证据完成，并集中披露局限；设备现状未核实，不将新NPU实验设为
初稿前置。公平性与数字审计仍是硬门槛，补实验按TASKS决策。
