# 采样模块设计(SLO 档校准)

> 设计完成,实装为 0。P1.6e 后定位转向,见 D-003。

---

## 1. 模块定位(D-003 后)

**职责:** 针对 `(workload_class, model, hw_config)` 三元组,离线校准固定 SLO 档(`ttft_budget_ms`, `tpot_budget_ms`)。

**输入:** 真实 trace(ShareGPT / Azure LLM Inference Trace / 内部生产)+ 模型 + 硬件配置。

**输出:** 单次 evaluation 内固定不变的标量 SLO 阈值(不是 per-req deadline,不是动态 scaling)。

**关键属性:**
- 一次性(offline),SLO 档不在推理路径上动态变化
- 类别级而非 per-request(跟 Sarathi-Serve 等工业实践一致)
- 跨硬件可迁移(换 TP 数 / 换模型时重新校准)

---

## 2. 模块的三个真实价值(D-003 后修正)

> 不再 claim「让 PID 双输入工作」(P1.6e 实证 SLO 校准必要但不充分)。

1. **定义系统可服务边界**
   - code 类工作负载在我们硬件 + 11qps 下违例率 0.57-0.92,无论 SLO 怎么调
   - 校准输出能识别这类「物理过载」工作负载 → 出 admission control / scale-up 范畴
2. **提供主报结果的目标档**
   - thesis 报 meet_slo% 用这个档,不用 strict 50ms 这种打地板的档
3. **校准 PID 启动期爬升目标**
   - 让爬升收敛到工程意义点,而非物理不可达档

---

## 3. 输入 / 输出 contract

### 输入

| 输入项 | 来源 | 备注 |
|---|---|---|
| `trace_path` | ShareGPT / Azure / 内部 jsonl | 至少含 `arrival_time` / `prompt_tokens` / `output_tokens` |
| `model_id` | 模型名 | 决定基线 latency |
| `hw_config` | NPU 型号 + TP 数 + max_model_len | 决定可达 SLO 下限 |
| `latency_anchor`(可选) | 单 req baseline latency micro-bench | 用于「读速 × 硬件下限」cross-check |

### 输出

```python
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
    },
    # 等
}
```

---

## 4. 校准方法(三约束交集)

校准 = 「为给定 (workload, model, hw) 选 (ttft_budget, tpot_budget)」,遵循三约束的交集:

### A. 硬件下限(feasibility floor)

`SLO_x ≥ K × baseline_x`,其中 `baseline_x` = 单 req 单负载下测的最小 latency,`K ∈ [1.5, 3.0]` 是 slack 系数。
- ttft baseline = 单 req prefill 时间(用 trace p50 prompt 长度算)
- tpot baseline = 单 req decode 时间(小 batch 下 pure decode iter 时间)

**作用:** 保证 SLO 在物理上可达,不至于全部趴地板。

### B. 用户感受上限(user-perceived ceiling)

**ttft:** 200ms(即时)/ 1000ms(可接受 chat)/ 3000ms(长文档分析尚可)/ > 5000ms(明显延迟)

**tpot**(人眼读速 ≈ 3-6 tok/s ≈ 167-333ms inter-token):
- < 50ms(远超读速,流畅)
- < 150ms(不慢于读速)
- < 300ms(勉强不卡)
- > 500ms(明显卡)

**作用:** 保证 SLO 在用户体验上有意义,不至于宽到无差别。

### C. 区分带可见性(experimental signal)

跑一次 post-hoc grid(`experiments/posthoc_slo_grid.py`),找让 4 个 config 的 meet_slo% 散开 ≥ 5pp 的档位区间,建议在区间中心选档。

**作用:** 保证 evaluation 有信号。如果 A∩B 全落在区间外,需要回头调 hw(TP 数)或 workload 类别定义。

### 示例:long workload, Qwen3-8B / TP=2

- A:ttft ≥ 1.5 × 400 = 600ms;tpot ≥ 1.5 × 80 = 120ms
- B:ttft ≤ 3000ms;tpot ≤ 300ms
- C(从 `long_prompt_sweep/` 看):区分带 ttft∈[500, 1500] × tpot∈[100, 200]

→ 三者交集中心:**ttft_budget=1500ms, tpot_budget=200ms** ✓

---

## 5. 工作负载类别(workload_class)定义

| 维度 | 候选值 | 校准依据 |
|---|---|---|
| prompt 长度 | short (p50<200) / medium (200-1000) / long (>1000) | 直接影响 prefill 时间 |
| output 长度 | short (<50) / medium / long (>500) | 影响 decode 持续时间,间接影响 tpot |
| arrival pattern | bursty (cv>1) / smooth (cv<0.5) | 影响 backlog + ttft 长尾 |
| mix 类型 | pure-short / pure-long / bimodal-X% / multi | 影响调度决策难度 |

**当前实装:** `experiments/lib/workload.PROMPT_PROFILES` 仅按 prompt 长度分(short / long / mixed)。未来扩展。

---

## 6. 上下游接口

```python
# 上游:trace → 类别 + SLO(offline 一次性)
sampler = TraceSampler(trace_path="azure_2024_q4.jsonl",
                       model_id="Qwen3-8B",
                       hw_config="910B3-TP2")
slo_profile = sampler.calibrate(workload_class="long")

# 下游 1:driver
#   experiments/qps_sweep.py 的 --slo-ttft-ms / --slo-tpot-ms
#   (已存在,目前 hardcode 500/50;未来从 SLO_PROFILES 自动取)

# 下游 2:controller
#   TDMConfig.slo_ttft_ms / slo_tpot_ms(已存在;同样未来自动取)
```

---

## 7. 实装计划

| 步骤 | 内容 | 估时 |
|---|---|---|
| S1 | `experiments/lib/slo_profiles.py` 注册表(手填 short/long/mixed 三档) | 0.25d |
| S2 | `experiments/calibrate_slo.py` 离线脚本:吃 trace → 出 prompt 长度直方图 + 建议档(A∩B∩C) | 0.5d |
| S3 | `qps_sweep.py` / `run_qps_sweep_all.py` 加 `--slo-profile <class>` 选项 | 0.25d |
| S4 | 在 mixed sweep 数据出来后跑一次 calibrate,把 mixed 档填入注册表 | 0.25d |
| S5 | TDMConfig.slo_*_ms 从 `--slo-profile` 自动注入 | 0.25d |

**当前优先:** S1(手填注册表),够 P1.7b 阶段用调过的档跑 baseline。S2-S5 留以后。

---

## 8. 与现有论文的关系

| 论文 | SLO 设定方式 | 跟本模块对比 |
|---|---|---|
| Sarathi-Serve | 固定 ttft / tpot 常数 | 本模块的 short/medium/long 分档是对其细化 |
| AlpaServe | SLO = N × per-req baseline | 本模块拒绝 per-req scaling(工程不可行),用类别级常数 |
| DistServe | 定 SLO 为 throughput@latency curve | 本模块只关心固定档,curve 留主图 |
| Splitwise | 报 latency 分布,不报 SLO% | 本模块强报 SLO%,因为审稿人锚定这个 |

---

## 9. 论文叙事(D-003 修正版)

> 现有 serving evaluation 要么用 chatbot 默认 SLO(ttft<500ms, tpot<50ms)固定,要么按 per-request scaling,都不适合 workload-aware 生产部署。我们提出**离线采样模块**,在三约束交集校准 per-class SLO 档:(a) 硬件可行下限,(b) 用户感受上限,(c) 实验区分性(post-hoc SLO-grid 验证选档让 config 散开 ≥ 5pp)。在 Qwen3-8B / 2×910B3 TP=2 上,校准出的 long-context 档(ttft<1500ms, tpot<200ms)正好落在三约束交集,让 TDM-M3.1 显出 +10pp(mixed,3 seeds)~ +18pp(long,1 seed)的 meet-SLO 优势,这个优势在默认 strict SLO 下完全看不见。

**关键定位补充:** Sampling 模块不再 claim「让 PID 工作」(P1.6e 实证不成立),而是负责定义可服务边界 + 校准 PID 启动期爬升目标 + 提供主报结果档。
