# TDM System Architecture

> 写于 2026-05-07（Sprint 1 / Step 1.5）。本文档定义 TDM 完整系统的设计：
> 三大控制模块 + 核心 mechanism (M3 chunking) + 集成点 + 数据流。
> 是 `idea_proposal.md`（论文 narrative）和 `slo_adaptive_design.md`（M2 单模块）
> 之上的统一系统视图，新增模块的设计文档（`slo_sampling_module_design.md`,
> `m3_chunking_design.md` 等）作为子节扩展。

---

## 0. 一句话总览

```
                  ┌─────────────────────────────────────────────────────┐
                  │  TDM Plugin (vllm-ascend/vllm_ascend/core/tdm/)     │
                  │                                                     │
   trace ──► [Sampling]──► SLO 档 ┐                                    │
                                  ├─► [SLO-PID] ──► ratio + chunk ──► [Scheduler] ──► AscendScheduler
   capture ──► [Graph] ──► buckets ┘            ▲                       │  (V0 dedicated kernels)
                                                │                       │
   AscendRuntime ──► [Telemetry/Tracker] ────────┘                      │
                  └─────────────────────────────────────────────────────┘
```

三个控制模块（① 采样 / ② SLO 自适应 / ③ Graph-aware）共同决定调度器每个 update tick 的两个输出维度：**phase ratio**（P:D 比例）和 **chunk_tokens**（每个 P iter 的 token cap）。M3 prefill chunking 是把后者从 `+∞`（不切）变成可调的核心 mechanism。

---

## 1. 三大控制模块的角色与边界

### 1.1 采样模块（Sampling）— 离线一次性

**职责**：从真实 trace（ShareGPT / Azure LLM Inference Trace / 内部生产 trace）按工作负载类别 + 模型 + 硬件配置校准固定 SLO 档。

**输入**：trace 文件（per-req 的 prompt_tokens / output_tokens / arrival pattern）；硬件信息（NPU 型号 + TP 数）；模型信息。

**输出**：每个 (workload_class, model, hw_config) 三元组对应的固定 SLO 档：
```python
{"ttft_budget_ms": 1500.0, "tpot_budget_ms": 200.0,
 "rationale": "long-context Q&A; aligned with industry workload-aware
              SLO conventions and reading-speed bound"}
```

**关键属性**：
- 一次性（offline）：SLO 档不在 inference 路径上动态变化
- 类别级而非 per-request：与 Sarathi-Serve 等工业实践一致
- 与硬件可迁移：换 TP 数 / 换模型时重新校准

**当前状态**：基本空白。仅有 `experiments/lib/workload.PROMPT_PROFILES` 的 short/long/mixed profile 注册表。完整设计见 `slo_sampling_module_design.md`（待写）。

### 1.2 SLO 自适应控制器（SLO-PID）— 在线闭环

**职责**：根据在线观察到的 SLO 违例率反馈调整 phase ratio（M2 已上线）和 chunk_tokens（M3.4 待实装）。

**实装位置**：`vllm-ascend/vllm_ascend/core/tdm/controller.py::SLOReactiveController`

**输入**：每个 update tick（默认 8 次 schedule call）从 telemetry 读 ttft / tpot violation rate + backlog age。

**输出**：phase ratio（M1: static 0.30；M2.x: PID 调出来的 dynamic [0.05, 0.80]）；M3.4 后还会输出 chunk_tokens。

**关键演进**（详见 `slo_adaptive_design.md` + commit 历史）：
- M2: 基础 PID
- M2.1: starvation guard
- M2.2: hysteresis
- M2.3: backlog-aware additive term (`kp_q × (age/ttft_budget - backlog_target)`)
- M2.4: tpot saturation detector（防 unreachable tpot 把 ratio 拖到地板）
- M2.5: ReLU clipping on err（已满足的 SLO 不再反向拖 ratio）
- M2.6/M2.7: saturation-tick 调优

**当前状态**：M2.7 已上线，长 prompt mid-intensity 区分带内胜出实证（见 `current_task.md §3.1.3`）。

### 1.3 Graph-aware 模块 — 设计辅助

**职责**：让 ratio / chunk 决策**对齐** vllm-ascend 实际 capture 的 graph batch sizes（默认 cudagraph_capture_sizes），让每个 D iter 命中 ACL FULL graph、每个 P iter 也尽量命中 small graph buckets。

**输入**：vllm-ascend 启动时的 capture 参数 + 实际 capture 完成后的 size list。

**输出**：建议的 chunk_tokens 候选（与 graph buckets 对齐，如 {128, 256, 512}）+ 建议的 decode batch 凑齐策略。

**实装位置**：当前散布在 `tdm/scheduler.py` 的部分逻辑里；M3.3 会做离线建表（actual capture sizes + 各 size 单 iter latency）后正式独立。

**当前状态**：模块存在但实证信号弱；M3.3 完成后才有完整数据。

---

## 2. 核心 mechanism: M3 prefill chunking

不是模块（无控制逻辑），是被三个模块共同驱动的 **底层调度 primitive**——一个 prefill iter 的 token cap。

**chunk 语义（已拍板，见 `current_task.md §4`）**：
- 切的对象：iter 内**总** prefill token cap，不限单 req
- 跨 iter 续：复用 vLLM V1 chunked-prefill 状态机，但走 AscendScheduler 路径（保 V0 快 kernel）
- phase-pure 严格保证：单 iter 要么 pure-P 要么 pure-D，**不混 D token**
- 候选值 {128, 256, 512, 1024, 2048, 8192}：前 3 与 graph capture sizes 对齐
- 静态扫参（M3.1-M3.2）→ SLO 控制器接 chunk 输出维度（M3.4）

**与三模块的耦合**：
- 采样模块 → 决定 chunk 候选范围（参考 trace 中 prompt 长度分布）
- SLO 控制器 → 在线选 chunk 值
- Graph-aware → chunk 值优先选与 capture buckets 对齐的

详见 `m3_chunking_design.md`（M3.1 启动时写）。

---

## 3. 与 vllm-ascend 的集成点

**入口**：`additional_config.tdm = {...}` 的 dataclass `TDMConfig`（`tdm/config.py`），TDMScheduler 在 `__init__` 读取，**不修改任何已有 vllm-ascend 文件**。

**继承关系**：
```
vllm.v1.core.sched.scheduler.Scheduler
  └─ vllm_ascend.core.scheduler.AscendScheduler        (v0.11.0rc1 phase-aware)
       └─ vllm_ascend.core.tdm.scheduler.TDMScheduler  (本插件)
```

**激活**：`additional_config.ascend_scheduler_config = {"enabled": True, "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler"}`

**数据流（per schedule() call）**：
1. AscendScheduler 决定可调度的 running + waiting 集合（这部分不动）
2. TDMScheduler.`_decide_phase()` 用当前 ratio 决定 P 或 D
3. 选择 req 集合（P: waiting；D: running）+ M3 后还要选 chunk_tokens 上限
4. 落地为 `SchedulerOutput`（vLLM 标准接口）
5. Tracker（`tracker.py`）记录 per-iter / per-req metrics → telemetry
6. 每 N 次 schedule call，Controller `update()` 读 telemetry 更新 ratio + chunk

---

## 4. 模块文件地图

| 文件 | 职责 |
|---|---|
| `tdm/config.py` | `TDMConfig` dataclass（所有 knob） |
| `tdm/scheduler.py` | `TDMScheduler`：phase 决定 + req 选择 + chunk cap（M3 后）|
| `tdm/controller.py` | `StaticRatioController` (M1) / `SLOReactiveController` (M2.x) |
| `tdm/selector.py` | req 选择策略（已抽离，便于单测）|
| `tdm/boundary.py` + `constraints.py` | phase 切换的边界条件、KV/budget 约束 |
| `tdm/tracker.py` | per-req TTFT/TPOT 抓取（passive_tracker 模式也用）|
| `tdm/telemetry.py` | iter / req level 数据落盘（jsonl）|
| `tdm/monitor.py` | DCMI AICore 实时信号接入（M4 hooks，未启用）|
| `tdm/timing.py` | iter 耗时归因 |
| `tdm/engine.py` | engine-side 钩子（与 v1 worker 联动）|
| `tdm/types.py` | 共享数据类型 |
| `tdm/tests/` | 60/60 单测 |

---

## 5. Future modules / hooks

| 名字 | 角色 | 当前状态 |
|---|---|---|
| **Sampling module** | 离线 SLO 校准（§1.1） | 设计中 |
| **M3 prefill chunking** | core mechanism (§2) | M3.0 done; M3.1- 待开 |
| **M4 DCMI feedback loop** | online AICore 利用率 → controller 第二维输入 | hooks 在 `monitor.py`，未启用 |
| **Admission control** | qps>32 saturated regime；queue gate 防 overload | 未规划 |
| **Phase 3: Azure trace replay** | 真实 trace driver，验证采样模块校准的 SLO 档 | backlog |

---

## 6. 评估方法（与 `evaluation_methodology.md` 对接）

**workload 维度**：`PROMPT_PROFILES` 注册的 `short` / `long` / `mixed`（`experiments/lib/workload.py`），未来加 Azure replay。

**SLO 维度**：post-hoc grid（`experiments/posthoc_slo_grid.py`）+ heatmap（`plot_slo_grid_heatmap.py`），区分带定义 = ttft∈[500,1500] × tpot∈[100,200]（mid-intensity 长 prompt regime）。

**intensity 维度**：默认 qps ∈ {8, 16, 32}，主战场 qps=16。

**对照配置**：
- `c1_baseline`：vllm-ascend default（hybrid prefill-first）
- `c2_tdm`：M1 static ratio=0.30
- `c2_tdm_m27`：M2.7 SLO-PID
- `c3_cp`：vLLM chunked prefill（mixed P+D 走 FIA 慢路径）
- `c4_pd`：PD 分离 1P1D（备用，长 prompt 还没跑）
- 未来：`c2_tdm_m3`（M3 chunking 完整）

**核心指标**：post-hoc SLO grid 上多档 meet_slo% + 区分带内的相对差距。

---

## 7. 论文 narrative 锚点

> "TDM is a three-module control framework over a phase-pure scheduler with prefill chunking, designed for Ascend NPU's V0 dedicated-kernel fast path. The offline sampling module calibrates workload-class-specific SLO budgets; the online SLO-PID controller adapts P:D ratio + chunk size to meet those budgets; the graph-aware module aligns scheduling choices with the vllm-ascend graph-capture bucket grid. Empirically, on Qwen3-8B / 2×910B3 TP=2 in long-prompt mid-intensity regime, TDM (M2.7) meets calibrated SLO at 100% vs hybrid baseline (C1) at 99.7% and chunked prefill (C3) which loses 3-4pp due to FIA-on-NPU being 2.7-3.4× slower than dedicated kernels."
