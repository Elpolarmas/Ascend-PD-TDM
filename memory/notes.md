# Notes

> 项目：单卡 NPU 上的 PD 时分复用调度（vllm-ascend 插件）
> 最近一次大改：2026-05-07，按用户澄清重写框架视图 + 锁定 vllm-ascend v0.11.0rc1

---

## 0. 版本锁定（2026-05-07 决定）

**vllm-ascend = v0.11.0rc1（2025-11-10），CANN 8.3.rc1，torch-npu 2.7.1**——本项目**全部**实验和论文实现锁在这个版本。

### 为什么锁定（不升级）

1. **PR #4623 在 v0.13.0 删除了 AscendScheduler**——上游官方公告原文：*"Ascend Scheduler has been dropped"*。这是我们 TDMScheduler 的父类，整个插件架构的依附点。**v0.13.0+ 没有这个类**，TDMScheduler 没法继承，注入路径 `additional_config.ascend_scheduler_config.scheduler_cls` 也失效。
2. **没有迁移路径**——PR #4623 的描述就是"删代码 + 让上游 vLLM Scheduler 接管"，没有等价的 phase-aware 替代接口。
3. **升级到 v0.13+ 会让 C1 baseline 消失**——v0.13.0+ 所有 workload 强制走上游 V1 Scheduler + chunked prefill + FIA kernel，等价于我们当前的 C3 路径。
4. **我们环境受限**，无法做并行 v0.18.0 supplementary 验证。

### 这个决定的后果（论文叙事必须明写）

- **优势**：v0.11.0rc1 是**最后一个保有 phase-aware AscendScheduler 的版本**，恰好是研究 phase-aware 调度价值的最后窗口
- **劣势**：审稿人会注意到这是 6 个 release 前的版本，必须在 §implementation 主动写明锁定理由
- **重新定位**：这反而把 TDM 的故事从"在 NPU 上做调度优化"升级为"vllm-ascend 上游放弃了 phase-aware（PR #4623），我们论证它的价值并提出再引入方案"——systems paper 标准 motivation

### Kernel 性能事实（micro-bench 实测，2026-05-07）

| 内核 op | 路径 | mean / p99 (mixed 1P+7D, q=519) | 相对 |
|---|---|---|---|
| `_npu_flash_attention_qlens` | C1 / TDM phase-pure → PrefillCacheHit | 0.088 / 0.119 ms | **基准** |
| `npu_fused_infer_attention_score` (sparse=3, TND) | C3 / mixed P+D → ChunkedPrefill | 0.234 / 0.705 ms | **慢 2.66×** |

- 同 query/KV/block_table 输入；warmup 20 + 200 iters；两路径需在独立进程跑（同进程 ATB context 冲突）
- pure_decode bs=64：FIA 慢 **3.43×**；pure_prefill q=1024：FIA 慢 **3.10×**
- 端到端反映：qps=8 R1 regime 下 C3 TTFT min 67ms vs C1 23ms（**2.9×**），与 36 层 transformer × attention kernel 占比 30-40% 的换算一致
- vllm-ascend 自己的源码 TODO 标注："*The npu_fused_infer_attention_score op is planned to be utilized in a wider range in upcoming versions*" —— 官方承认此路径仍在过渡

### Kernel 数据的正确解读（重要修正 2026-05-07）

| 路径 | attn_state | 内核 | 速度 |
|---|---|---|---|
| **C1 mixed P+D** | `PrefillCacheHit` | `_npu_flash_attention_qlens` (V0) | 快 |
| **TDM phase-pure P** | `PrefillNoCache` | `_forward_prefill_no_cache` (V0) | 快 |
| **TDM phase-pure D** | `DecodeOnly` | `_forward_decode_only` (V0) | 快 |
| **C3 mixed P+D (CP=True)** | `ChunkedPrefill` | `npu_fused_infer_attention_score` (V1 FIA) | 慢 2.7-3.4× |

**关键**：C1 和 TDM 都走 V0 dedicated 快路径——FIA kernel 慢这件事**只解释 C3 < C1，不解释 TDM 输 C1**。

→ TDM 在 8K canonical (短 prompt) sweep 上输 C1，根因不是 kernel，是 **C1 hybrid 在短 prompt 下利用率天然高**（mixed iter 让 decode 顺路搭车）。TDM 的真正赢点必须从 **workload 维度** 找，详见 `current_task.md §1.3`。

### M3 设计原则（基于 kernel 现实）

**phase-pure 严格遵守**——TDM 内部 prefill chunking 必须保证：
- 单个 iter **要么 pure-prefill 要么 pure-decode**，绝不让 mixed P+D 出现在同一 iter
- 这样 attn_state 走 `PrefillNoCache` / `PrefillCacheHit` / `DecodeOnly`（dedicated kernel），不退化到 `ChunkedPrefill` 的 FIA 慢路径
- **重要**：phase-pure 不是为了让 TDM "比 C1 走更快的 kernel"（两者一样快），而是**避免 TDM 不小心退化到 C3 的慢路径**——是守底线，不是抢上限

---

## 1. 框架视图（修订版 2026-05-07）

### 1.1 两个模块的分工

```
┌──────────────────────────────────────────────────────────────┐
│  SLO 自适应控制模块  ★ 核心                                     │
│  输入：                                                        │
│    · SLO 违例率 (ttft / tpot)        ← 主因素                   │
│    · 请求队列状态 (waiting_age, queue_depth, kv_pressure)        │
│    · 硬件利用率 (NPU AICore busy%, HBM BW%)                      │
│  输出（两个时间维度）：                                          │
│    · iteration 维度：本 iter 走 P / 走 D / 切多大 prefill chunk    │
│    · 滑动窗口维度：未来 N iter 的 P:D 比例 / 切片策略 target       │
└──────▲────────────────────────────────────────────────────────┘
       │ batch_size_for_decode (绑 ACL graph capture buckets)
       │ chunk_size_for_prefill (匹配 graph 边界 / 控制单次 iter 时长)
┌──────┴────────────────────────────────────────────────────────┐
│  Graph 自适应控制模块  ── 辅助（喂硬件知识）                       │
│  · 主要面向 Decode：选 ACL Graph 的最优 batch size bucket          │
│  · vllm-ascend 静态预编译一组 graph sizes（如 {1,2,8,16,…,512}）  │
│  · 给 SLO 控制器一张"哪个 size 跑得最快"的字典                     │
└──────────────────────────────────────────────────────────────┘
```

**关键澄清**：graph-aware 不是独立目标，是 SLO 控制器的下游算力字典。

### 1.2 Prefill chunking：与 C3 chunked prefill 的本质区别

| 维度 | vLLM Chunked Prefill (C3) | 本项目的 TDM-内部 chunked prefill |
|---|---|---|
| **P/D 是否共享 forward** | **是**（in-iter 混合 batch） | **否**（pure-phase per iter） |
| 切分粒度 | 单 iter 内 prefill 占多少 token | 单 prefill 占多少**连续独占 iter**，何时让位 D |
| 切分依据 | 静态 `max_num_batched_tokens` 预算 | 动态：SLO + queue + HW 反馈 |
| Decode batch size | 不可控（被 prefill 挤剩多少算多少） | **graph-aware 锁定**到 bucket |
| Kernel 形状 | 不规整（prefill chunk + decode tokens 拼） | 纯净（prefill 满帧 / decode bucket） |
| 哲学 | 空间混叠 | 时间分片 |

**当前设计选择**（2026-05-07 用户拍板）：
- prefill chunk **按 token 数切**（与 decode graph bucket 对应更直接）
- graph buckets **静态**（vllm-ascend 编译期固定）
- sliding window **静态长度**，8 vs 几百待实测

### 1.3 当前问题

C3 在业界是 SOTA，但我们 8K canonical 实测 C3 全谱输 C1（goodput -14%~-22%, ttft +54%~+125%）—— **数据反常**。可能是 vllm-ascend 上的 chunked prefill 实现 / Ascend kernel 特性问题，需要后续核查。这件事影响"主对比是 C3"的论文叙事根基。

---

## 2. 项目背景 + 代码路径

### 2.1 vllm-ascend PD 架构现状

- AscendScheduler（`vllm_ascend/core/scheduler.py`）已有 PD phase 字段，但原本面向多卡 disaggregation
- 上游 vLLM V1（`vllm/v1/core/sched/scheduler.py`）是 token-based 统一调度，无硬性 P/D 分离
- AscendAttentionState（`vllm_ascend/attention/attention_v1.py`）：PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding

### 2.2 核心代码路径

| 用途 | 路径 |
|---|---|
| AscendScheduler | `vllm-ascend/vllm_ascend/core/scheduler.py` |
| AscendSchedulerConfig | `vllm-ascend/vllm_ascend/core/schedule_config.py` |
| NPU Model Runner | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` |
| Attention | `vllm-ascend/vllm_ascend/attention/attention_v1.py` |
| ACL Graph 编译 | `vllm-ascend/vllm_ascend/compilation/acl_graph.py` |
| 上游 Scheduler | `vllm/v1/core/sched/scheduler.py` |
| **TDM 插件根** | `vllm-ascend/vllm_ascend/core/tdm/` |
| TDM 注入键 | `additional_config.ascend_scheduler_config.scheduler_cls = "vllm_ascend.core.tdm.scheduler.TDMScheduler"` |

### 2.3 ACL Graph 关键约束

- 每种 batch size 单独编译 graph（~1.4s/graph），总数硬上限 `MAX_CAPTURE_SIZE=1800`
- 运行时未命中 → fallback eager（per_token 延迟 +26%）
- vllm-ascend 默认 48 种已编译 shape ∈ [1, 512]，间距 1→2→8→16→32→…
- pad/eager 判断基于 `total_num_scheduled_tokens`（一 iter 全部 token 之和）：
  - **Decode**：每 req 1 token → total = 并发数 → 通常 ≤ 512 → graph 命中
  - **Prefill**：长 prompt → total = Σ prompt 长度 → 大概率 > 512 → eager
- ACL Graph 主战场是 **decode 阶段的 padding 浪费优化**（源码：`model_runner_v1.py:3488-3545`，decode 有独立 `decode_cudagraph_batch_sizes`）

---

## 3. 实验事实速查（早期可行性论据）

> Qwen3-4B + 单/双 910B3。详细数字保留在 git 历史的旧 notes.md，下面是结论。

### 3.1 P/D 资源画像（Exp B）
- Prefill = **compute-bound**（AICore 69.8%，HBM BW 15.8%）
- Decode = **memory-bandwidth-bound**（AICore 42.6%，HBM BW 25.1%）
- → P/D 在硬件上互补，时分复用有理论收益

### 3.2 ACL Graph padding 浪费（Exp A/F）
- Prefill 小 batch（1-8）平均 23.4% 浪费；decode 中等 batch 16.5%
- 简单"向下取整"凑 graph shape **不可行**（小 batch 下减请求代价 > padding 代价）
- 正确策略是**"向上凑"**：队列足够时主动凑到精确匹配 shape

### 3.3 在线利用率采集（DCMI 实测）
- **HBM BW 1.1ms / AICore 60ms** 单次调用延迟
- HBM BW 后台采样恒定 7-9% 开销（来自 GIL/driver 锁，与频率无关）
- AICore 100ms 采样 ~0% 开销
- 推荐：在线 AICore 100ms + 离线画像表覆盖 HBM BW

### 3.4 拓扑对比（Exp E3，Qwen3-4B 2×910B3）
- Disagg 1P1D **全面输 -2%~-12%**（小模型同机场景，TP=2 通信开销 < 分离的空闲开销）
- Chunked Prefill 在高并发提升 +10.8%~+11.6%（在 GPU/小模型上 SOTA 验证）
- → C4 PD-disagg 在我们 setup 不是真实威胁；C3 才是

---

## 4. TDM 概念定稿（2026-04-02）

### 4.1 不是逐 iter 二选一
TDM = 在**滑动时间窗口内控制 P:D:M 三种模式的执行比例**。控制变量是 prefill 插入率 `r_p`，不是每次硬切。

### 4.2 三种执行模式（V2 完整动作空间）
| 模式 | 行为 | 适用 |
|---|---|---|
| PREFILL | 纯 prefill iter | 队列深、decode 空闲 |
| DECODE | 纯 decode iter | TPOT 紧、无新请求 |
| MIXED | Chunked Prefill (P+D 同 batch) | 高并发、负载平稳 |

**CP 是 TDM 的特例**（全 iter 选 MIXED）→ TDM 是 CP 的超集。

### 4.3 Iteration-level vs Window-level
- 1 iteration = 1 batch = 1 ACL Graph replay（等价）
- **iter 维度（Layer 1）**：本 iter 走什么模式 → 时间维度
- **window 维度（Layer 2）**：未来 N iter 的 P:D 比例 target → 长程规划
- 串行决策：window 决定 ratio → iter 按 ratio + 即时状态选 phase

---

## 5. TDM 插件代码架构

### 5.1 文件布局

```
vllm-ascend/vllm_ascend/core/tdm/
├── __init__.py                 导出 TDMScheduler, TDMConfig
├── scheduler.py                TDMScheduler（继承 AscendScheduler，注入入口）
├── config.py                   TDMConfig（所有超参 + 校验）
├── controller.py               StaticRatioController + SLOReactiveController + make_controller
├── selector.py                 TokenBucketSelector（peek/commit 两阶段）
├── engine.py                   PhaseEngine（执行 phase 切换）
├── constraints.py              HardConstraints（min/max_slice_iters）
├── boundary.py                 BoundaryGuard（KV pressure freeze）
├── monitor.py                  QueueMonitor（QueueSnapshot 采集）
├── tracker.py                  RequestTracker（passive per-req TTFT/TPOT）
├── telemetry.py                Telemetry（hot ring + cold JSONL）
├── timing.py                   TimeAccountant
├── types.py                    QueueSnapshot / RequestRecord / PhaseDecision
└── tests/                      单测 60/60 全过
    └── _runner.py              `python3 -m vllm_ascend.core.tdm.tests._runner`
```

### 5.2 Option W（零上游改动）
`AscendScheduler.schedule()` L103-104 的 "waiting+running 都空 → 切 decode" 自动翻 phase 用作"物理兜底"，TDM 以 peek/commit 模式观察实际执行 phase 后 reconcile 计数。

### 5.3 schedule() 调用时序

```
schedule():
  snap = monitor.snapshot()
  ratio = controller.get_target_ratio(snap, iter_id)         ← L3 慢变量（每 update_interval iter）
  planned = selector.peek(phase, phase_iters, ratio, snap)    ← L3 快变量
  after_c = constraints.enforce(planned, ...)
  candidate = boundary.override(after_c, snap)
  engine.apply(candidate)                                     ← 设 self.phase
  out = super().schedule()                                    ← 父类可能自动翻 phase
  actual = self.phase
  selector.commit(actual)                                     ← 按实际消耗 token
  engine.reconcile(actual, candidate)
  telemetry.record_iter(...)
  return out
```

### 5.4 共享数据契约

```python
@dataclass(frozen=True)
class QueueSnapshot:
    waiting_depth: int
    waiting_oldest_age_ms: float
    finished_prefill_depth: int
    running_depth: int
    kv_free_blocks: int
    kv_total_blocks: int
    @property
    def kv_free_ratio(self) -> float: ...

@dataclass(frozen=True)
class PhaseDecision:
    phase: Literal["prefill", "decode"]
    source: Literal["controller", "constraint_min/max_slice",
                    "boundary_kv_pressure", "parent_auto_flip", "fallback"]
    target_ratio: float

@dataclass
class RequestRecord:
    request_id, prompt_tokens, output_tokens
    admission_ts_ms, first_token_ts_ms, finish_ts_ms
    decode_intervals_ms: list[float]
    @property ttft_ms / tpot_ms_mean / tpot_ms_p99
```

### 5.5 消融开关矩阵

| 方案 | enable_tdm | controller_kind | 说明 |
|---|---|---|---|
| Baseline (C1) | False (passive_tracker=True) | — | AscendScheduler 原版 + tracker |
| TDM static (M1=C2) | True | static | 固定 ratio=0.30 |
| TDM SLO-PID (M2) | True | slo_pid | + SLO 自适应 |

---

## 6. 实验 driver 架构

### 6.1 文件布局

```
Ascend-PD-TDM/experiments/
├── lib/
│   ├── workload.py        # WorkloadSource ABC + SyntheticPoisson；扩展点 AzureTraceReplay
│   └── metrics.py         # tracker JSONL 读 + 精确 ID join + 窗口聚合 + Goodput
├── qps_sweep.py           # 单点 driver（httpx + asyncio + tracker join）
├── run_qps_sweep_all.py   # 编排（一 config 一 server，QPS 列表共享）
├── compare_c1_c2_c3.py    # 多方对照图（支持 --c12 --c3 --c4 --m2..--m27）
├── plot_qps_sweep.py      # 单 sweep 4 子图（system python3）
└── plot_m2_diag.py        # ratio over time 诊断图
```

### 6.2 关键基础设施约定

- **passive_tracker 模式**：baseline / 各对照配置都走 TDMScheduler 但 `enable_tdm=False, passive_tracker=True`，仅跑 tracker → C1/C2/C3/C4 测量口径完全一致
- **Goodput 定义**：`status==200 ∧ ttft_ms<SLO_ttft ∧ tpot_ms_mean<SLO_tpot` 的 output_tokens 总和 / 稳态窗口
- **精确 ID 匹配**：tracker `cmpl-xxx-0` ↔ driver `cmpl-xxx`，`_strip_sample_suffix` 去末尾索引
- **SLO 默认**：TTFT < 500ms，TPOT < 50ms

### 6.3 Config 矩阵（`build_additional_config`）

| 配置 | 关键设置 |
|---|---|
| `c1_baseline` | `enable_tdm=False, passive_tracker=True` |
| `c2_tdm` (M1) | `enable_tdm=True, static_ratio=0.3, controller_kind=static` |
| `c2_tdm_m2` ~ `c2_tdm_m27` | M2.x 各变体（见 §8） |
| `c3_cp` | vLLM 默认 chunked prefill |
| `c4_pd` | PD-disagg 1P1D + ranktable + driver `--stream` 拿 client-side metric |

---

## 7. 论文对照矩阵 + 关键发现

### 7.1 四方对照

| # | 配置 | scheduler | 角色 |
|---|---|---|---|
| **C1** | hybrid | `AscendScheduler` (phase="") | sanity baseline / 参考线 |
| **C2** (M1) | TDM static | `TDMScheduler` static_ratio=0.3 | ablation 起点 |
| **C3** | Chunked Prefill | vLLM 默认 (AscendScheduler 短路退回) | **真正 SOTA on-device 主对比** |
| **C4** | PD disagg 1P1D | 双 server + proxy + LLMDataDist KV connector | 架构竞品 |

### 7.2 模块化 ablation

```
M0  hybrid (C1 vanilla)              ← sanity
M1  basic TDM (C2)                   ← 切片本身（已实现）
M2  + SLO-adaptive ratio             ← PID + 各种 guard（已实现，未达预期）
M3  + graph-aware + prefill chunking ← 见 §1.1-1.2（待实现）
M4  + 监控反馈闭环                     ← 在线（基础设施部分就绪）
```

### 7.3 8K canonical regime 数据（论文核心）

```
budget        max_model_len = max_num_batched_tokens = 8192
sweep         duration=60s warmup=20s
QPS           收敛到 {16, 32}（R2 / R3 endpoint）
SLO           ttft_p99<500ms ∧ tpot_p99<50ms
硬件天花板     tpot p99 ≈ 57ms ⇒ 50ms SLO 物理不可达，是 M2 困局根源
```

| qps | regime | C1 gp | M1 gp | M2 gp | C3 gp | C1 SLO% | M1 SLO% | M2 SLO% | C3 SLO% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8  | R1 | 1519 | 1513 | 1508 | 1189 | 98.6 | 96.8 | 95.7 | 72.3 |
| 16 | R2 | **1906** | 1624 | **1823** | 1508 | **65.0** | 53.5 | 61.3 | 50.8 |
| 24 | R2 | **3104** | 2621 | 2913 | 2668 | **70.8** | 61.8 | 67.8 | 61.7 |
| 32 | R3 | 4681 | 4331 | 4146 | 3762 | **78.8** | 72.6 | 70.3 | 63.9 |

**核心 finding**：
1. C1 hybrid 在 8K canonical 全谱占优，TDM 仅在 R3 ttft_p99 维度赢（116 vs 215ms, -46%）
2. C3 chunked prefill 全谱输 C1（业界 SOTA，但在我们 setup 反常）—— **待核查 vllm-ascend CP 实现**
3. **裸 TDM 输 hybrid 不是设计错，是「单纯切片 ≠ 完整系统价值」** —— 这是 ablation M1 起点

### 7.4 Regime 反转
- **2K regime**（紧 budget）：qps=16 c2 ttft 比 c1 好 21% → TDM 占优
- **8K regime**（松 budget）：qps=16 c1 ttft 比 c2 好 42% → hybrid 占优
- 解释：紧 budget → slot 稀缺 → 时分调度有用；松 budget → hybrid 自由打包更高效

---

## 8. M2 SLO-adaptive 实验（2026-05-02 ~ 2026-05-06）

### 8.1 M2 系列变体

每个变体 = PID-lite controller + 一个新机制。所有用 `controller_kind=slo_pid`。

| 版本 | config | 新增机制 | 关键参数 |
|---|---|---|---|
| **M2** | `c2_tdm_m2` | 纯 PID-lite | `kp=0.5, ema_alpha=0.3, deadband=0.02` |
| **M2.1** | `c2_tdm_m21` | + starvation guard | `starvation_min_ticks=3` |
| **M2.2** | `c2_tdm_m22` | + hysteresis release | `release_margin=0.10` |
| **M2.3** | `c2_tdm_m23` | + backlog 加和 | `kp_q=0.5, backlog_target=0.5` |
| **M2.4** | `c2_tdm_m24` | + tpot 饱和检测 | `saturation_min_ticks=5` |
| **M2.5** | `c2_tdm_m25` | + ReLU err clip | `relu_err=True` |
| **M2.6** | `c2_tdm_m26` | M2.5 + saturation_min_ticks=1 | 立即锁饱和 |
| **M2.7** | `c2_tdm_m27` | M2.5 + saturation_min_ticks=2 | 2 tick 锁饱和 |

### 8.2 各机制效果

| 机制 | 设计初衷 | 实测效果 |
|---|---|---|
| starvation guard | ratio 撞底关 err_tpot 让 ttft 救场 | 没用，guard 释放后立刻拉回 floor |
| hysteresis release | 解振荡 | 没用，振荡不是主因 |
| backlog-aware | "队首 age / SLO budget" 当 leading indicator | 没用，oldest_age 90% 时间 = 0 |
| tpot 饱和检测 | tpot 物理不可达时屏蔽 err_tpot | 部分有用，但激活前 ratio 已被推到底 |
| ReLU err clip | 满足的 SLO 不该反推 ratio（修 err 公式） | 关键修复 |
| saturation_min_ticks=1 | 立即锁 saturation | 锁住 0.30，但 R3 反差 |

### 8.3 综合排名（R2 + R3 SLO% 几何均值）

| 方案 | R2 SLO% | R3 SLO% | geomean | 备注 |
|---|---:|---:|---:|---|
| C1 hybrid | 65.0 | 78.8 | **71.6** | 全局最优（参考线） |
| **M2.7** | 62.6 | 70.1 | **66.2** ⭐ | M 家族综合最优 |
| M2 | 61.3 | 70.3 | 65.6 | 纯 PID 居然次优 |
| M2.3 | 55.9 | 71.9 | 63.4 | |
| M2.5 | 53.5 | 74.5 | 63.1 | R3 单点最优 |
| M1 (static) | 53.5 | 72.6 | 62.3 | |
| C3 chunked | 50.8 | 63.9 | 57.0 | 全谱负向 |

### 8.4 关键 finding：regime-dependent 最优 ratio

| Regime | 实测最优 ratio | 数据点 |
|---|---|---|
| R2 (qps=16) | ≈ 0.27 | M2.7 锁 0.27 → 62.6（M2.6 锁 0.30 → 49.8） |
| R3 (qps=32) | ≈ 0.05 (floor) | M2.5 ratio 在 floor 97.8% → 74.5 |

**含义**：单一全局最优 ratio 不存在 → 真正的 adaptive 必须 regime-aware → motivate M3。

### 8.5 PID 控制器超参（`SLOReactiveController` 完整接口）

| 参数 | TDMConfig 字段 | 默认 | 含义 |
|---|---|---:|---|
| `kp` | `slo_pid_kp` | 0.5 | `delta = kp × (err_ttft - err_tpot)` |
| `ema_alpha` | `slo_pid_ema_alpha` | 0.3 | EMA 平滑 |
| `deadband` | `slo_pid_deadband` | 0.02 | 误差死区 |
| `min_samples` | `slo_pid_min_samples` | 16 | 冷启动门槛 |
| `window_size` | `window_size` | 128 | 滑窗大小 |
| `update_interval` | `controller_update_interval` | 8 | 每 N iter 重算 |
| `target_violation_rate` | `slo_target_violation_rate` | 0.05 | 允许 5% 违例 |
| `initial_ratio` | `static_ratio` | 0.30 | 冷启动 |
| `ratio_min/max` | `slo_ratio_min/max` | 0.05/0.80 | clip 范围 |
| `slo_ttft_ms / slo_tpot_ms` | 同名 | 500/50 | SLO 阈值 |
| `slo_starvation_min_ticks` | M2.1 | 3 | 撞底 N tick 进 guard |
| `slo_starvation_release_margin` | M2.2 | 0.10 | guard 释放门槛 |
| `slo_pid_kp_q` | M2.3 | 0.5 | backlog 加和系数 |
| `slo_pid_backlog_target` | M2.3 | 0.5 | backlog 中性点 |
| `slo_tpot_saturation_enabled` | M2.4 | True | 饱和检测开关 |
| `slo_tpot_saturation_min_ticks` | M2.4 | 5 | 进入饱和 |
| `slo_tpot_saturation_release_ticks` | M2.4 | 5 | 释放饱和 |
| `slo_pid_relu_err` | M2.5 | True | err ReLU 截断 |

---

## 9. 复现实验

### 9.1 数据落盘位置

```
Ascend-PD-TDM/results/
├── qps_sweep_8k/              C1 + C2 (M1) 8K canonical
├── qps_sweep_8k_c3/           C3 chunked prefill 8K
├── qps_sweep_2k_y/            2K Y plan C1+C2+C3
├── qps_sweep_2k_y_c4/         2K Y plan C4 PD-disagg
├── m2_slo_adaptive/           M2 sweep + diag 图
├── m2{1..5}_slo_adaptive/     M2.1-M2.5
├── m26_m27_slo_adaptive/      M2.6 + M2.7（同 sweep）
└── tdm_trace/                 passive tracker JSONL（每 config 一对 req+iter）
```

### 9.2 跑 sweep（vllm 的 python，必须从 /tmp 启）

```bash
cd /vllm-workspace/Ascend-PD-TDM/experiments

# 单变体（例：M2.7 R2/R3 endpoint）
/usr/local/python3.11.13/bin/python3 run_qps_sweep_all.py \
  --configs c2_tdm_m27 \
  --qps 16,32 --duration 60 --warmup 20 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --outdir /vllm-workspace/Ascend-PD-TDM/results/m27_slo_adaptive

# 全 M 家族（~1.5h）
python run_qps_sweep_all.py \
  --configs c2_tdm_m2,c2_tdm_m21,c2_tdm_m22,c2_tdm_m23,c2_tdm_m24,c2_tdm_m25,c2_tdm_m26,c2_tdm_m27 \
  --qps 16,32 --duration 60 --warmup 20 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --outdir /vllm-workspace/Ascend-PD-TDM/results/m2_full_family

# baseline（C1+C2 / C3 / C4）
python run_qps_sweep_all.py --configs c1_baseline,c2_tdm \
  --qps 16,32 --duration 60 --warmup 20 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --outdir /vllm-workspace/Ascend-PD-TDM/results/qps_sweep_8k
# C3: --configs c3_cp，C4: --configs c4_pd（自动加 --stream）
```

`--configs` 可选：`c1_baseline, c2_tdm, c2_tdm_m2..c2_tdm_m27, c3_cp, c4_pd`。

### 9.3 出综合对比图（system python3，要 matplotlib）

```bash
/usr/bin/python3 compare_c1_c2_c3.py \
  --c12 .../qps_sweep_8k/qps_sweep_summary.json \
  --c3  .../qps_sweep_8k_c3/qps_sweep_summary.json \
  --m2  .../m2_slo_adaptive/qps_sweep_summary.json \
  --m21 .../m21_slo_adaptive/qps_sweep_summary.json \
  ...
  --m27 .../m26_m27_slo_adaptive/qps_sweep_summary.json \
  --outdir <out> --out-name compare.png \
  --title-prefix "8K canonical: full M-family"
```

CLI: `--c12 --c3 --c4 --m2..--m27`，每个 arg 接一个 sweep summary JSON。

### 9.4 单测（不依赖 NPU）

```bash
cd /tmp && /usr/local/python3.11.13/bin/python3 -m vllm_ascend.core.tdm.tests._runner
# 60 个 controller + scheduler 测试，~5s
```

### 9.5 ratio 轨迹诊断

```bash
/usr/bin/python3 -c "
import json, statistics
from collections import Counter
path = '<sweep>/tdm_trace/qps_sweep_<config>_iter.jsonl'
ratios=[]; phases=[]
for line in open(path):
    rec = json.loads(line)
    if 'target_ratio' in rec: ratios.append(rec['target_ratio'])
    if 'phase' in rec: phases.append(rec['phase'])
print(f'iters={len(ratios)} ratio min={min(ratios):.3f} max={max(ratios):.3f} mean={statistics.mean(ratios):.3f}')
print(f'phases: {dict(Counter(phases))}')
"
```

### 9.6 环境陷阱（必读）

| 陷阱 | 解决 |
|---|---|
| 从 `/vllm-workspace` 启 vllm 撞 namespace（`vllm.__file__=None`） | server / driver 必须 `cd /tmp` 启（编排脚本已自动） |
| 本机 `http_proxy=localhost:7890` 不可改 | curl `--noproxy '*'`，httpx `trust_env=False` |
| vLLM V1 `RequestOutput.metrics=None` | 用 TDM passive_tracker；C3/C4 加 driver `--stream` SSE 时间戳 fallback |
| httpx 默认连接池 `max_connections=256, pool_timeout=10s` 在 qps≥32 滚雪球 | 已修：`max_connections=2048, pool_timeout=300s` |
| C4 PD-disagg `LLM_LINK_FAILED` | ranktable.json 给两 NPU 不同占位 IP（10.0.0.1/10.0.0.2），HCCS 物理直连不查真实 IP |
| `compare_c1_c2_c3.py / plot_qps_sweep.py` import matplotlib | 用 system `/usr/bin/python3` |
| `run_qps_sweep_all.py::run_one_qps()` rc≠0 时丢 summary | 修法：rc 标志只打印，summary 始终从 out_path 读回（已知 bug，未修） |

---

## 10. 安装备忘

```
torch / torch-npu       华为镜像源装：pip install torch==2.7.1 torch-npu==2.7.1 \
                        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
vllm                    VLLM_TARGET_DEVICE=empty pip install -e .
vllm-ascend             source set_env.sh && pip install -e . --no-build-isolation
transformers            < 5.0.0
```

- 推理脚本必须在 `if __name__ == '__main__':` 内
- vllm 的 python：`/usr/local/python3.11.13/bin/python3`
- 出图用 system python：`/usr/bin/python3`（带 matplotlib）

---

## 11. Pending Questions（开放）

1. **C3 反常**：vllm-ascend chunked prefill 全谱输 C1 的根因？kernel/scheduler 实现层核查
2. **prefill chunk 粒度**：按 token 切（已定），具体值范围（512 / 1024 / 1500 / 2048…）需扫
3. **sliding window 长度**：8 vs 几百 iter，需扫参（更长 → 更稳但响应慢；更短 → 响应快但抖）
4. **graph bucket 列表**：vllm-ascend 实际 capture 哪些 size，对应每个 size 的 latency 画像（M3 待建表）
5. **AICore 利用率信号**：DCMI 100ms 采样接入 SLO controller（M4 工作）

---

> 相关工作详细分析见 `paper_template.md`（如未生成）；Idea 方案完整版见 `idea_proposal.md`；M2 SLO-adaptive 设计文档见 `slo_adaptive_design.md`。
