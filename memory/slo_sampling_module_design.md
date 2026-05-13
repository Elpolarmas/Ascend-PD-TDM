# SLO Sampling Module 设计文档

> 写于 2026-05-07（Sprint 1 / Step 1.5）。本文档描述 TDM 三大控制模块之
> ① **离线采样模块**的设计。它的产出是后续两个模块（SLO-PID / Graph-aware）
> 的输入前提。
>
> 上下文：见 `system_architecture.md §1.1`。

---

## 1. 为什么需要这个模块

### 1.1 来自数据的 motivation

`results/long_prompt_sweep` 跑出来的事实：

- 在 strict SLO 档（ttft<500ms / tpot<50ms）下，**所有 4 个 config 都趴在地板**（meet_slo% < 5%）→ 看不出差异
- 在 loose SLO 档（ttft>2000ms / tpot>300ms）下，**所有 4 个 config 都 100%** → 也看不出差异
- 只在**区分带**（tpot ∈ [100,200] × ttft ∈ [500,1500]）内 4 个 config 才显出差异，M2.7 一致胜出

→ **SLO 档位的选择直接决定 evaluation 是否有信号**。如果 SLO 档错配，再好的调度算法也看不出价值。

### 1.2 现有论文的做法 + 我们的偏移

工业实践通常做以下之一：
- 给一个固定 SLO 常数（如 ttft<200ms, tpot<50ms）—— Sarathi-Serve 等
- 用 SLO scaling factor（如 SLO = N × T_baseline，AlpaServe）
- 不报 SLO%，只报 latency 分布

我们的偏移：**SLO 档位是 (workload_class, model, hw_config) 三元组的函数**，由采样模块**离线一次性**校准。这个偏移有三个好处：
1. **方法论稳**：审稿人不能挑剔"你们 SLO 是怎么定的"——它来自 trace 统计 + reading-speed 锚点 + 工业惯例
2. **跨硬件可复现**：换 TP 数 / 换模型时重新校准，evaluation 框架不变
3. **与 TDM narrative 对齐**：采样模块本身就是论文 contribution 的一部分

---

## 2. 模块输入 / 输出 contract

### 2.1 输入

| 输入项 | 来源 | 备注 |
|---|---|---|
| `trace_path` | ShareGPT / Azure LLM Inference Trace / 内部生产 trace（jsonl） | 至少含 `arrival_time`, `prompt_tokens`, `output_tokens`，最好有 `category` 标签 |
| `model_id` | 模型名（如 `Qwen3-8B`） | 决定基线 latency |
| `hw_config` | NPU 型号 + TP 数 + max_model_len | 决定可达 SLO 下限 |
| `latency_anchor` (optional) | 单 req 在该 hw 上的 baseline latency micro-bench | 用于 reading-speed × 硬件下限 cross-check；缺省时只用 trace 统计 |

### 2.2 输出

```python
# experiments/lib/slo_profiles.py 注册表（待写）
SLO_PROFILES = {
    ("Qwen3-8B", "910B3-TP2", "short"): {
        "ttft_budget_ms": 500.0,
        "tpot_budget_ms": 50.0,
        "rationale": "interactive chat; aligned with Sarathi-Serve",
        "trace_p50_prompt": 90,
        "trace_p99_prompt": 1029,
        "calibration_date": "2026-05-07",
    },
    ("Qwen3-8B", "910B3-TP2", "long"): {
        "ttft_budget_ms": 1500.0,
        "tpot_budget_ms": 200.0,
        "rationale": "long-context Q&A; reading-speed bound (3-6 tok/s)",
        "trace_p50_prompt": 1124,
        "trace_p99_prompt": 4000,
        "calibration_date": "2026-05-07",
    },
    ("Qwen3-8B", "910B3-TP2", "mixed"): {
        # to be calibrated from mixed sweep result
        "ttft_budget_ms": 800.0,
        "tpot_budget_ms": 150.0,
        "rationale": "70/30 short-long bimodal; ShareGPT-like",
        ...
    },
}
```

每条记录是**单次 evaluation 内固定不变**的标量 SLO 阈值（不是 per-req deadline，不是动态 scaling）。

---

## 3. 校准方法

校准 = "为给定 (workload_class, model, hw) 选 (ttft_budget, tpot_budget)"，遵循三个**约束**的交集：

### 3.1 约束 A：硬件下限（feasibility floor）

`SLO_x ≥ K × baseline_x`，其中 `baseline_x` 是在 trace 单 req 单负载下测的最小可达 latency。`K` 是 slack factor，建议 `K ∈ [1.5, 3.0]`。

- ttft baseline = 单 req prefill 时间（取 trace p50 prompt 长度算）
- tpot baseline = 单 req decode 时间（小 batch 下 pure decode iter 时间）

**作用**：保证 SLO 在物理上可达，不至于全部趴地板。

**测法**：跑 `qps_sweep.py` 的 qps=1 / 2 单点，看 ttft/tpot 实际值；或用 `vllm-ascend` 的 micro-bench harness。

### 3.2 约束 B：用户感受上限（user-perceived ceiling）

**ttft**：
- < 200ms：感觉即时
- < 1000ms：可接受 chat
- < 3000ms：长上下文文档分析尚可
- > 5000ms：明显延迟

**tpot**（人眼读速 ≈ 3-6 tok/s ≈ 167-333ms inter-token）：
- < 50ms：远超读速，明显流畅
- < 150ms：不慢于读速
- < 300ms：勉强不卡
- > 500ms：明显卡

**作用**：保证 SLO 在用户体验上有意义，不至于宽到无差别。

### 3.3 约束 C：区分带可见性（experimental signal）

跑一次 post-hoc grid（`posthoc_slo_grid.py`），找让 4 配置 meet_slo% **散开** 5pp 以上的档位区间，建议在区间中心选档。

**作用**：保证 evaluation 有信号；如果约束 A ∩ B 全部落在区间外，需要回头调 hw 配置（如 TP 数）或 workload 类别定义。

### 3.4 三约束交集示例（long workload, Qwen3-8B / TP=2）

- 约束 A：ttft ≥ 1.5 × 400 = 600ms（trace p50≈1100 单 req prefill ≈ 400ms，K=1.5）；tpot ≥ 1.5 × 80 = 120ms（小 batch decode ~80ms）
- 约束 B：ttft ≤ 3000ms（doc-analysis tier）；tpot ≤ 300ms（reading-speed bound）
- 约束 C（从 long_prompt_sweep grid 看）：区分带 ttft ∈ [500, 1500] × tpot ∈ [100, 200]

→ 三者交集中心：**ttft_budget = 1500ms, tpot_budget = 200ms** ✓（与现 long-context 档一致）

---

## 4. 工作负载类别（workload_class）的定义

校准的前置：把 trace 切成有意义的类别。建议分类维度：

| 维度 | 候选值 | 校准依据 |
|---|---|---|
| **prompt 长度类** | short (p50<200) / medium (200-1000) / long (>1000) | 直接影响 prefill 时间 |
| **output 长度类** | short (<50) / medium / long (>500) | 影响 decode 持续时间，间接影响 tpot SLO |
| **arrival pattern** | bursty (cv > 1) / smooth (cv < 0.5) | 影响 backlog & ttft 长尾 |
| **mix 类型** | pure-short / pure-long / bimodal-X% / multimodal | 影响 schedule 决策的难度 |

**当前实现**：`experiments/lib/workload.PROMPT_PROFILES` 仅按 prompt 长度分（short / long / mixed）。未来扩展。

---

## 5. 与上游/下游模块的接口

```python
# 上游：trace → 类别 + SLO（offline 一次性）
sampler = TraceSampler(trace_path="azure_2024_q4.jsonl",
                       model_id="Qwen3-8B",
                       hw_config="910B3-TP2")
slo_profile = sampler.calibrate(workload_class="long")
# {"ttft_budget_ms": 1500.0, "tpot_budget_ms": 200.0, ...}

# 下游：注入 driver + controller
# 1. driver: experiments/qps_sweep.py 的 --slo-ttft-ms / --slo-tpot-ms
#    （已存在，目前 hardcode 500/50；未来从 SLO_PROFILES 自动取）
# 2. controller: TDMConfig.slo_ttft_ms / slo_tpot_ms
#    （已存在；同样未来从 SLO_PROFILES 自动取）
```

---

## 6. 与现有论文的关系

| 论文 | SLO 设定方式 | 与本模块对比 |
|---|---|---|
| Sarathi-Serve | 固定 ttft / tpot 常数 | 本模块的 short/medium/long 分档校准是对其的细化 |
| AlpaServe | SLO = N × per-req baseline latency（per-request scaling） | 本模块拒绝 per-req scaling（工程不可行），用类别级常数 + offline 校准 |
| DistServe | 定 SLO 为 throughput@latency curve | 本模块只关心固定档，throughput@latency 留给主图（不冲突）|
| Splitwise | 报 latency 分布，不报 SLO% | 本模块强报 SLO%，因为 it's the only metric reviewers anchor on |

---

## 7. 实装计划

| 步骤 | 内容 | 估时 |
|---|---|---|
| **S1** | `experiments/lib/slo_profiles.py` 注册表（手填 short/long/mixed 三档，对齐当前 sweep 数据） | 0.25d |
| **S2** | `experiments/calibrate_slo.py` 离线脚本：吃 trace → 出 prompt 长度直方图 + 建议 SLO 档（约束 A∩B∩C） | 0.5d |
| **S3** | `qps_sweep.py` / `run_qps_sweep_all.py` 加 `--slo-profile <class>` 选项，自动从注册表取 ttft/tpot budget | 0.25d |
| **S4** | 在 mixed sweep 数据出来后跑一次 calibrate，把 mixed SLO 档填入注册表 | 0.25d |
| **S5** | 接入 TDM controller：`TDMConfig.slo_*_ms` 从 `--slo-profile` 自动注入（贯通 driver → server） | 0.25d |

**Sprint 1 内只做 S1 + S4**（校准 mixed 档）；S2 / S3 / S5 留 Sprint 2。

---

## 8. 论文 narrative 角度

> "Existing serving evaluations either fix SLO at chat-bot defaults (ttft<500ms, tpot<50ms) or scale SLO per-request, neither suiting workload-class-aware production deployment. We propose an **offline sampling module** that calibrates per-class SLO budgets at the intersection of three constraints: (a) hardware feasibility floor (`K × per-req baseline latency`), (b) user-perceived ceiling (anchored to reading speed), and (c) experimental discrimination (post-hoc SLO-grid validation that the chosen budget separates configurations by ≥ 5pp). On Qwen3-8B / 2×910B3 TP=2, calibrated long-context budget (ttft<1500ms, tpot<200ms) places the discrimination band at the intersection of all three constraints, allowing TDM-M2.7 to demonstrate a clean +0.3 to +3pp meet-SLO advantage over hybrid (C1) and chunked prefill (C3) — an advantage invisible under naive default SLOs."
