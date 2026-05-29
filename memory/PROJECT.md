# 项目核心

> 这个文档说的是不轻易变的事情。任何 thesis 级修改要走 `DECISIONS.md` 记录。
> **当前 thesis 版本: D-013(2026-05-25)** — 在 D-011 + D-012 框架内,Phase 1 数据后 framing 修正:**"phase-pure batching 让 cycle 时间短于 c3 mixed iter,mean/tail 同步改善"**(不是原 "trade tail for mean")。历次 thesis 修订见 `DECISIONS.md`。

---

## 1. 项目一句话

在 single-node multi-NPU 上(2 张 Ascend 910B3,TP=2),基于 vllm-ascend v0.11.0rc1 实现 PD-TDM(Prefill-Decode Temporal Multiplexing)调度 paradigm,提供 goodput-centric improvement under SLO constraints。论文目标 CCF-B/C。

---

## 2. 不能偏离的几条原则

任何工作方向跟这几条冲突 → 必须先走 `DECISIONS.md` 记录,不允许默默改。

1. **thesis 是「PD-TDM paradigm + SLO-aware fast-loop selector(单维度,SLO-aware)」**(D-011 修订)
   - **PD-TDM paradigm:** temporal P/D multiplexing(prefill iter 与 decode iter 分相,空间上 TP=N 共享权重)
   - **SLO-aware fast-loop selector:** token bucket(cap=4)+ urgency_ttft per-request sorting
   - **保留但 paper 不提的设计:** 慢回路 PID(P1.9d/e 实证伪)、Starvation tpot override(P1.7c 触发率 0.5%)、Graph-aware(D-005 F only)
   - **不能**再 claim 「双维度协同」或「闭环反馈控制器」——P1.9d/e 已证伪。任何回归这些叙事的建议必须先写新 decision

2. **主报 metric = goodput at SLO(max sustainable QPS at SLO target ≥ X%)**(D-011 修订)
   - 跟 DistServe / Sarathi-Serve 等 prior work 同框架,可量级对比
   - SLO meet rate at fixed QPS = secondary metric(saturated regime quality view)
   - Pareto frontier(x=QPS,y=SLO meet%)作为主图

3. **Paper scope = Framing C hybrid**(D-011)
   - **Conceptual scope:** "Single-node multi-device accelerators **lacking SM partitioning support**"(包括 NPU / older GPUs / 未来 ASIC)
   - **Evaluation scope:** Ascend 910B3 NPU 作为 representative platform
   - 这个 scope 是 MuxWise(ASPLOS'26)"supports intra-process spatial sharing" scope 的 explicit complement set
   - Title 不锁 NPU,Section 1 明确 scope statement

4. **对手 = 4-way paradigm comparison**(D-014 baseline 重定位,2026-05-26)
   - **Vanilla CB(主流 LLM serving 文献 baseline):** vllm v1 chunked prefill 路径 + `max_num_batched_tokens=8192`,Azure trace prompt cap=7000 < 8192 → chunk 实际不触发 = mixed batch + 长 prompt 整 iter 一次跑完。数据 = 老 c3 chunk=8192(`phase_2_t6_burst_goodput/*_nonpid/c3_cp_qps0.0.json`)。
   - **Sarathi chunked prefill(主流 LLM serving 文献 baseline):** Sarathi-Serve paradigm 的 open-source impl(vllm v1 chunked prefill + chunk=2048)。数据 = `c3_chunk2048_supplement/`。
   - **c4_pd PD-disagg 1P1D(reference):** budget-equivalent reference point(不 claim 普遍反优势,1P:1D 是资源受限被迫配置)。
   - **PD-TDM (M3.1, ours):** chunked prefill 框架内的 phase-pure variant(同样 chunk_budget=2048)。数据 = `m31fix_validate/`。
   - ~~C1 (admit-driven phase-pure)~~ **从 paper 主线移除**(D-014):NPU-specific niche design,文献无对应命名;代码 / 数据保留 internal use。
   - 不直接对比 MuxWise(spatial,需要 SM partition,scope 不交)/ Semi-PD(angle 不同)/ DistServe family(multi-node)

   **Baseline 命名原则(D-014 新加)**:必须跟主流 LLM serving 文献对应。Vanilla CB / Sarathi chunked prefill / PD-TDM 都是文献术语。Reviewer 一眼可 map。

5. **vllm-ascend 锁 v0.11.0rc1**(D-007)
   - PR #4623 在 v0.13.0 删了 AscendScheduler,无迁移路径
   - v0.13.0+ 强制走 V1 Scheduler + chunked prefill + FIA kernel(等价 C3)

6. **kernel 速度不作为论文核心论据**(D-009)
   - 同 kernel ablation 已实证 kernel 贡献 ≈ 0
   - 只用来解释「为什么 C3 在 NPU 上反常输 C1」

7. **实验聚焦 burst trace sweep + 校准 SLO grid + cross-model**(D-013 修订)
   - **T6 burst goodput sweep** = paper Section 5 主图:2 wl × 7 k(burst intensity)× 3 SLO × 3 seed = 126 cells
   - Arrival = `trace_sampled_burst`(Azure trace,period=10s,high/low QPS 按 k scale)
   - SLO grid = Sarathi 风格 × {5,10,15}× × micro-benchmark ideal
   - Cross-model:Qwen3-4B + Qwen3-8B(Phase 2 待跑)

8. **任何 cite m31 数据前必须确认是 fix 版**(D-013)
   - 检查 outdir 是 `m31fix_validate/`(fix 版)
   - 旧版 `phase_2_t6_burst_goodput/*_pid/` 数据已删除(5/25 cleanup)
   - chunked_schedule.py 必须含 Diff #6(waiting loop gate `self.phase == "prefill"`)

9. **c3 vs c1 不普适胜出**(D-013)
   - code 上 c3 大幅胜 c1(strict SLO Δmeet +14~+75pp)
   - **conv strict SLO 上 c3 反输 c1 -4~-19pp**(chunked prefill 短 prompt 高频负载弱点)
   - paper Section 5 须 explicit 讲此 caveat,不能假设 c3 > c1

---

## 3. 系统架构

```
trace ──┐
        ├─► Sampling(offline)─► calibrated SLO grid
hw+model┘                          (evaluation methodology, 不进 contribution)
                                          
State probe(QueueMonitor/Tracker/Telemetry)
       │
       ▼
SLO-aware fast-loop selector(token bucket + urgency_ttft)
       │
       ▼
PD-TDM phase-pure scheduling ─► AscendScheduler
       │
       ▼
Static target_ratio config(per-workload offline tuning)
```

### Paper 主 mechanism(进 contribution)

**PD-TDM paradigm**(`tdm/scheduler.py` + `chunked_schedule.py`)
- Phase-pure iter dispatch:prefill iter 100% prefill,decode iter 100% decode
- TP-shared weights(空间上 2 NPU 共享),temporal P/D 分相(时间上分相)
- Iter trace 物理 fingerprint:decode silence p99 800-940ms,99.7% prefill 时 decode queue 非空

**SLO-aware fast-loop selector**(`tdm/selector.py`)
- Token bucket:每 iter `tokens += target_ratio`,cap=4 累积上限
- Urgency-based per-request priority:`urgency_ttft = 0.7 × SLO_TTFT`,紧急请求 prefill 排序优先
- P1.9d 实证:conv tight-TTFT regime 提供 +1-2.5pp 加成

**Static target_ratio**(Section 5.6 sensitivity sweep)
- 不在线动态调,recommend per-workload(conv: r=0.8 / code: r=0.7)
- 不 claim contribution,只是 design choice

### Paper 不进 contribution 的(代码保留 internal,paper 不写)

**Slow-loop PID controller**(`tdm/controller.py::SLOReactiveController`)
- P1.9d 在 saturated 上 B-A ≈ 0(action space ratio_max 太窄)
- P1.9e 在 transition 上 -2.68pp(conv2code,反优势)
- D-011 决定:代码保留 internal,paper 主体不提

**Starvation tpot override**(`tdm/selector.py::starvation_decouple_bucket` flag)
- P1.7b/c 触发率 0.5-0.8%,Δ ±1pp noise
- 跟 urgency_ttft 在 selector 优先级互相 shadow

**Graph-aware module**(D-005 F only)
- vllm-ascend 已内置 cudagraph_capture_sizes
- 只在 Implementation 节一句话 fairness setup

### Infrastructure(Section 4 Implementation 描述,不进 contribution)

**State probe 三件套**(`tdm/monitor.py` / `tracker.py` / `telemetry.py`)
- QueueMonitor:每 iter snapshot scheduler 内部状态
- RequestTracker:per-request lifecycle(admit / first-token / per-token / finish)
- Telemetry:4 类 jsonl 结构化 log,evaluation 用

### Evaluation methodology(Section 5.1 Setup 描述,不进 contribution)

**Sampling 模块**(offline SLO grid 校准)
- 离线 sweep calibrate 校准 SLO grid(tpot×ttft 多档)
- 不进 contribution,只作为 evaluation methodology

---

## 4. 版本锁(2026-05-07,见 D-007)

vllm-ascend = **v0.11.0rc1**,CANN 8.3.rc1,torch-npu 2.7.1。所有实验/论文锁这个版本。

### kernel 事实(D-009 后 paper 不报为 contribution)

| 路径 | 实际 kernel | 速度 |
|---|---|---|
| C1 mixed P+D | `_npu_flash_attention_qlens` (V0) | 快(基准) |
| TDM phase-pure P | `_forward_prefill_no_cache` (V0) | 快 |
| TDM phase-pure D | `_forward_decode_only` (V0) | 快 |
| **C3 mixed (CP=True)** | `npu_fused_infer_attention_score` (FIA V1) | **慢 2.7-3.4×** |

注:同 kernel ablation(P1.8 m31_fia)已实证 kernel 贡献 ≈ 0。M3.1 vs C3 paradigm-level Δ 来自 paradigm bundle 总差(D-010 立场),不归因到单变量。

---

## 5. 代码地图

| 用途 | 路径 |
|---|---|
| TDM 插件根 | `vllm-ascend/vllm_ascend/core/tdm/` |
| 注入入口 | `additional_config.ascend_scheduler_config.scheduler_cls = "vllm_ascend.core.tdm.scheduler.TDMScheduler"` |
| **Paper 主 mechanism: Selector** | `tdm/selector.py`(`TokenBucketSelector` + urgency_ttft)|
| **Paper 主 mechanism: Scheduler** | `tdm/scheduler.py` + `chunked_schedule.py`(phase-pure dispatch)|
| Phase 引擎(internal) | `tdm/engine.py` |
| 控制器(internal,paper 不写) | `tdm/controller.py`(`SLOReactiveController` 慢回路 PID)|
| Infrastructure(Section 4 Implementation)| `tdm/monitor.py` / `tracker.py` / `telemetry.py` |
| Chunking | `tdm/chunking.py`(planner)+ `chunked_schedule.py`(fork 自父类) |
| 共享类型 | `tdm/types.py`(`QueueSnapshot` / `RequestRecord` / `PhaseDecision`) |
| 单测 | `tdm/tests/`(104/104 pass,2026-05-18) |
| 实验 driver | `Ascend-PD-TDM/experiments/` |
| 实验数据 | `Ascend-PD-TDM/results/` |
| Paper outline 起草 | `Ascend-PD-TDM/memory/design/paper.md`(W2 起重写)|

---

## 6. 禁区清单

| 禁区 | 解禁条件 |
|---|---|
| 用 strict 默认档(50/500)跑主实验 | 永久(物理不可达,主报 calibrated SLO grid)|
| 设计 dynamic chunk_tokens 控制器 | 永久(chunk pareto 单调,D-006)|
| 把 kernel 速度作为论文主图 | 永久(D-009)|
| Stationary workload 反复跑 PID 调参 | 永久(P0-2 / P1.9d 都实证不显效)|
| 加 KV cache 调度 / SpecDecode / 多优先级扩 scope | 永久(scope 锁定)|
| 把 reactive 控制本身抬为范式贡献 | 永久(D-002 → D-011 thesis 转向 paradigm 视角)|
| Paper 写慢回路 PID 作为 contribution | 永久(D-011,P1.9d/e 证伪)|
| Paper 写 Starvation tpot override 作为 contribution | 永久(D-011,P1.7c 0.5% 触发率)|
| 不 cite MuxWise(ASPLOS'26)和 Semi-PD | 永久(D-011 D2 defense 关键 anchor)|
| 实装 mixed_mode 开关跑 phase-pure 单变量 ablation | 永久(D-010 立场:paradigm-level Δ 不归因单机制)|
| Title 锁 NPU 关键字 | 永久(D-011 Framing C,scope 不锁 NPU)|

---

## 7. 不可逆约束 / 外部依赖

- **硬件:** 2 × Ascend 910B3 (64GB HBM each),TP=2 tensor parallel
- **模型:** Qwen3-4B + Qwen3-8B(主 evaluation);Qwen3-0.6B 系统中有但 paper 不用
- **软件:** CANN 8.3.rc1 + PyTorch 2.7.1 + torch-npu 2.7.1 + vllm-ascend v0.11.0rc1
- **Trace:** Azure conv + Azure code(已用)+ BurstGPT(待加,W1)+ mixed(conv+code 同时到达,待生成 + sweep,W1)
- 没有 GPU(H100 / A100 / MI300)access,跨 hardware 实证不可行 → Limitations 诚实承认
- 没有第二台同型号设备 → 不能跑跨版本对照
- 工作流:vllm 必须 `cd /tmp` 启(namespace 陷阱);出图用 system `/usr/bin/python3`(带 matplotlib);server/driver 用 `/usr/local/python3.11.13/bin/python3`

---

## 8. Paper structure 速览(D-011)

| Section | 内容要点 |
|---|---|
| S1 Intro | NPU 视野 + LLM serving 现状 + paradigm landscape + PD-TDM 定位 + scope statement(Framing C)|
| S2 Background+Motivation | 4-way paradigm landscape(mixed / temporal / spatial / disagg)+ cite Sarathi / DistServe / MuxWise / Semi-PD + design gap motivation |
| S3 Design | 3.1 PD-TDM paradigm overview;3.2 phase-pure on TP-shared NPUs;3.3 SLO-aware fast-loop selector;3.4 static ratio config |
| S4 Implementation | vllm-ascend integration + state probe infrastructure + cudagraph fairness setup + open-source artifact 声明 |
| S5 Evaluation | 5.1 Setup(model / trace / SLO grid / 4 baselines)+ 5.2 Pareto frontier(goodput-centric)+ 5.3 SLO meet% at fixed QPS(secondary)+ 5.4 mechanism ablation(fast-loop)+ 5.5 cross-model + cross-trace + 5.6 ratio sensitivity + 5.7 physical evidence(iter trace fingerprint)|
| S6 Discussion | Interference shifting unified lens(4-paradigm)+ regime boundary 物理解释 + intra-node disagg discussion |
| S7 Related Work | MuxWise(spatial multiplexing,scope 互补)+ Sarathi-Serve(mixed batching)+ DistServe family(disagg)+ Semi-PD + vLLM/TGI |
| S8 Limitations | single-platform / single-family / no online switching / 1P:1D budget reference |
| S9 Conclusion | — |

---

## 9. 后续工作时间线(W1-W4,D-011 决定的 3-4 周 timeline)

| W | 任务 |
|---|---|
| **W1** | D-011(本次)+ PROJECT.md 重写(本次)+ EXPERIMENTS.md/FINDINGS.md 更新 + 准备实验脚本 + 启动 QPS sweep + c4_pd 补跑 + 跨模型 sweep + BurstGPT trace + mixed workload trace 生成 |
| W2 | Paper draft S1-S4(Intro / Background / Design / Implementation)+ 实验数据 ready 时整合到 S5 |
| W3 | Paper draft S5-S9(Evaluation / Discussion / Related / Limitations / Conclusion)+ polish 第 1 轮 |
| W4 | Polish 第 2 轮 + 选 venue + 投递 + buffer |

**Venue 投递顺序:**
- 优先冲 B 中下:IPDPS / DSN / Middleware / SoCC(B-top 风险大但顺手投)
- Fallback CCF-C:IISWC / ICPADS / HPCC / ICCD

---

**给下次 session 的关键 hand-off:**
- thesis 在 D-011,所有 reviewer attack defense framework 完整记录在 D-011 内
- 实验 W1 的 5 个补跑(QPS sweep / c4_pd / Qwen3-4B sweep / BurstGPT / mixed workload)是 paper 完整 evaluation 的前置
- Paper writing 在 W2 起,Section 1-4 先写,实验数据 ready 后写 Section 5
