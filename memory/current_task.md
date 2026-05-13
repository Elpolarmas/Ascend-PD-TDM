# 当前任务

> 更新日期:2026-05-13
> 上下文文档:`notes.md`(代码地图/复现)、`idea_proposal.md`(论文叙事)、`slo_adaptive_design.md`(控制器细节)、`slo_sampling_module_design.md`(SLO 校准)、`p20_interface_audit.md`(状态采集接口)
> **核心目标**:系统实装到位 + 跑实测;论文是结果不是目标

---

## §0. NEW SESSION 速读

**当前进行**:`P1.6e` SLO target sweep(M3.1 在 slo_tpot ∈ {150, 200, 300} 下的行为对照),后台跑(PID 750628,~6h wall,数据进 `results/azure_p16e/`)。

**最近 finding 链**:
1. P1.6a 屏蔽审计 → 99.4% M2.4 屏蔽是设计合理 + backlog kp_q=0 装饰
2. P1.6b 反馈链修复(F1+F2 实装,85 单测过)→ ttft warm 从 16.5s 降到 2.0s,**但 tpot 信号一上线就 100% 违例**
3. P1.6e SLO target sweep(完成):**SLO 校准对反馈链有巨大影响**(conv M2.4 屏蔽 99.5% → 5.4%),**但 PID ratio 仍钉 max 80%+** —— ReLU + M2.4 联合让 PID 在边界几乎必然单输入,**SLO 校准必要但不充分**

**核心未决问题**:见 §4

---

## §0.5 核心创新模块当前真实工作模式(2026-05-13 P1.6e 之后)

> **核心创新声明** vs **实际表现** 的精确差距,thesis-critical。

**Thesis claim**:SLO-aware 闭环反馈控制 P/D 比例,tpot/ttft 双 SLO 作 runtime 受控变量。

**P1.6 实测**:控制器实际工作模式是**开环单输入 + 边界保护**,不是闭环反馈。

**证据**:跨 4 个 SLO 档(50/150/200/300ms),conv `ratio 钉 max` 都 80%+,`err_tpot_eff 非零率` 最高 3.6%(PASS 标准 >20%)。tpot 这条腿从未真正参与 ratio 调节。

**根因**:ReLU + M2.4 联合屏蔽机制:
- viol > target:M2.4 屏蔽 tpot 信号(防 ratio 拉地板)
- viol < target:ReLU clip(防余量推 ratio 反向)
- 双输入工作区是极窄中间带,稳态系统几乎不可能稳定停在这里

**PID 真实价值定位**:
- 启动 ramp 加速(F1+F2 修后从 27s 缩到 19s)
- 自动找到 ratio_max 这个边界(避免人工 tune)
- saturated 物理饱和检测(M2.4 按设计屏蔽,避免误调度)

**三条出路**(详见 §4):
- A. **iter 维度补强 P1.7b**(selector 旁路 PID 屏蔽,在 iter 粒度做双向 leading indicator)— **推荐**
- B. 改 PID 控制律(去 ReLU / 调 target / 换 MPC)— 风险大
- C. 承认问题,改 thesis 定位为"启动 ramp + 边界保护"— 创新度下降

**推荐 A + C 组合**:thesis 叙事变"两维度协同"——慢回路负责 ramp+边界保护,快回路负责 in-flight 反馈。

---

## §1. 系统定位与架构

### 1.1 场景与对手
- **场景**:Serverless AI 推理,真实非平稳/突发负载
- **对手**:vLLM Chunked Prefill (C3, SOTA fusion 方案)
- **MuxWise**(GPU 空分):同类问题不同 substrate(我们 NPU 走时分)
- **不 claim**:范式级创新;**claim**:Ascend 工程方案 + Azure trace 实证 + 设计选择

### 1.2 系统架构(三模块,Graph 已降级)

```
┌──────────────────────────────────────────────────┐
│ ① Sampling 模块(workload-conditional SLO 校准)│ ← 升格中,P1.6e/f 验证
├──────────────────────────────────────────────────┤
│ ② SLO 自适应控制器(window 维度 PID + iter 决策器)│ ← 核心创新
│   - A 速率调整器(慢回路 PID,8 iter 更新 ratio)│
│   - B Phase 决策器(快回路 token bucket + leading indicator) │
├──────────────────────────────────────────────────┤
│ ③ 状态采集模块(monitor / tracker / telemetry)│ ← 基础设施,P1.6b F1+F2 修后实时
└──────────────────────────────────────────────────┘

底层执行: TDM 切片 + Chunk 机制(均已实装,not main contribution)

已撤下:Graph 自适应模块(vllm 已内置 + P1.0b 证伪;论文写作前定夺 G/F/D)
```

### 1.3 核心创新声明

> **"Ascend NPU 上 SLO-aware 闭环 P/D 时分复用调度"**——把 SLO 违例率作为 runtime 受控变量,通过反馈控制器在线调节 P:D 比例,适配非平稳负载。Sampling 模块提供 workload-conditional SLO 目标。

---

## §2. 实装现状

| 模块 | 状态 | 关键文件 |
|---|---|---|
| ① Sampling 模块 | 🔴 设计文档存在,实装为 0(P1.6e 手工版本中) | `slo_sampling_module_design.md` |
| ② A 速率调整器 PID | ✅ 实装,P0/P1.6 已深度审计 | `controller.py` |
| ② B Phase 决策器 | ✅ 实装,P1.7 单向 urgency FAIL,P1.7b 双向未做 | `selector.py` |
| ③ 状态采集 | ✅ tracker/monitor/telemetry 实装,**P1.6b F1+F2 修反馈链滞后** | `tracker.py / monitor.py / telemetry.py` |
| TDM 切片 | ✅ 实装 | `scheduler.py` |
| Chunk 机制 | ✅ 实装,pareto 单调 | `chunking.py` |
| 单测 | ✅ 85/85 pass | `tests/` |

---

## §3. 关键 finding 速查

### 3.1 Azure trace 主结果(`results/azure_main/`)

2 traces × 4 configs × 3 seeds × 60s。**conv meet_slo%**:
| tier | C1 | M1+chunk | M3.1 | C3 |
|---|---|---|---|---|
| ttft500/tpot150 | 55.6 | 60.7 | **64.1** | **68.6** ⚠️ |
| ttft500/tpot200 | 86.4 | 83.5 | **88.6** | 87.2 |

**code meet_slo% (ttft500/tpot200)**:M3.1 **12.2 vs C3 2.8 (+9.4pp)**。

正向:conv tpot200 +5.2pp vs M1+chunk,code tpot200 +9.4pp vs C3。负向:conv tpot150 输 C3 4.5pp(P/D 分开结构性代价,P0-3 归因)。

### 3.2 跨窗扩展(`results/azure_p15/`,P1.5,2026-05-12)

2 traces × 3 windows × 4 configs × 3 seeds:
- conv 4/4 tier 跨窗 Δ(M3.1, M1+chunk) PASS(mean ≥ 0,min ≥ -0.6pp)
- **code 严档 tpot100/150 跨窗 Δ ≈ 0**——M3.1 vs M1+chunk 在 saturated 无增量,给 P1.7 提供动机

### 3.3 PID 行为深度归因(P0-2 + P1.6a/b/e)

| Finding | 来源 | 含义 |
|---|---|---|
| ratio 启动 5-10s 从 0.3 推到 ratio_max=0.8,之后钉死 | P0-2 | PID 稳态等效"启动 ramp + 静态 ratio_max" |
| M2.4 saturation 屏蔽占 99.4% warm tick | P1.6a | 设计合理(tpot 违例 mean 97%) |
| **backlog 项 kp_q 实测=0,设计装饰** | P1.6a | M2.3 leading indicator 未启用 |
| record_request 只在 on_finish 调用 → ttft/tpot 都滞后 req 生命周期 | P1.6a | 反馈链是 bug,F1+F2 已修(P1.6b) |
| F1+F2 修后 warm 13-17s → 2-3s,**但 tpot 信号 100% 违例不变** | P1.6b | 反馈链不是根因,**SLO target 错才是根因** |
| SLO 校准让 conv M2.4 屏蔽从 99.5% 砍到 5.4%(50→300ms) | P1.6e | 反馈链可达性救回 |
| **但 ratio 仍钉 max 80%+ 跨所有 SLO 档** | P1.6e | ReLU+M2.4 联合屏蔽让 PID 在边界几乎必然单输入 |
| code saturated:任何 SLO 档下 M2.4 永恒 94.5%,viol_rate≥0.57 | P1.6e | code 在我们硬件 + 11 qps 下不可服务(物理过载) |

### 3.3b SLO sweep 完整数据(P1.6e,2026-05-13)

`results/azure_p16e/tpot{50,150,200,300}/{conv,code}_w1_seed{0,1,2}/`,1 config × 4 SLO 档 × 2 windows × 3 seeds × 60s = 24 calls(50 档复用 azure_p16b)。

**conv 跨 seed mean**:
| slo_tpot | tpot_viol | M2.4 屏蔽 | err_tpot_eff 非零 | ratio 钉 max |
|---|---|---|---|---|
| 50  | 0.94 | **99.5%** | 0.5% | 82.9% |
| 150 | 0.38 | 94.1% | 1.7% | 81.6% |
| **200** ⭐ | 0.09 | **28.2%** | **3.6%** | 83.1% |
| 300 | 0.02 | **5.4%** | 1.4% | 84.0% |

**code 跨 seed mean**:几乎完全不变(M2.4 永恒 94.5%,viol_rate 0.57-0.92)。

**PID 单输入的两类失败模式**(F5):

| 失败模式 | 触发条件 | 含义 |
|---|---|---|
| M2.4 屏蔽 | viol > target 持续 | SLO 过严或物理过载 |
| ReLU clip | viol < target | SLO 过松,系统有余量 |
| 持续双输入 | viol ≈ target 稳定 | 极窄区,实际系统难稳定停在这里 |

### 3.4 P1.7 单向 urgency 反向伤(`results/azure_p17/`)

8/8 严档 tier 中 7 个 Δ(M3.2 - M3.1) < 0。**finding 加固 thesis**:selector 需要**双向 leading indicator**(TTFT urgency + TPOT starvation guard),单向不够。P1.7b 待做。

### 3.5 Graph 模块降级(`results/baseline_c3_audit_*/`,P1.0/P1.0b)

- vllm-ascend 已内置 `cudagraph_capture_sizes` + decode 自动 pad
- P1.0b:AIV(23 sizes) vs FFTS+(15 sizes),capture 数量非单调(AIV 反而 C3 -1.5~-23.8pp)
- 结论:**没有算法空间**,撤下转 Platform Finding(论文写作前定 G+F / D / F only)

### 3.6 其它

- **stationary long prompt**(`long_5way_q1632/`):M3.1 vs M1 在 stationary +0.9/+0.3pp 弱(PID 不显效)
- **chunk pareto 单调**(`p1_chunk_scan/`):dynamic chunk_tokens 控制器路径作废
- **合成 burst**(`burst_*/`):PID Δ ≈ ±0.7pp(合成 burst 撬不动 PID)

---

## §4. 核心未决问题

### 4.1 SLO 设置:静态 vs 动态(2026-05-13 用户提出)

**问题来源**:P1.6e 暴露了 PID 用 `slo_tpot_ms=50` 失效——这个值是 evaluation strict tier,不是 workload-realistic SLO。

**关键事实**:
- 物理可达 tpot:conv@1860 p50=148ms,code@570 p50=453ms,min=30ms
- 50ms 不只让 PID 失效,evaluation 看也无信号(conv 4.2% 达标,code 2.1%)
- evaluation 用多档 tier(50/100/150/200/250)是行业惯例,但 thesis main result 应**只报 calibrated 档**

**三种 SLO 策略候选**:

| 策略 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| **A. 固定 strict** | 跑 50/500 (现状) | 工业默认值 | PID/evaluation 都失效,thesis 主表无信号 |
| **B. 离线 calibrated**(推荐) | sampling 模块离线校准 (workload, model, hw) → 单一 SLO,thesis main result 报这个档 | 跟 idea_proposal §1.1 sampling 模块对接;PID 真闭环反馈 | 静态校准对非平稳负载有局限 |
| **C. 在线动态** | workload class 在线检测 → 动态切换 SLO target | 真正自适应 | 工作量大;workload class detection 是独立问题;P1.6e/f 验证后才能评估必要性 |

**P1.6e 后的修正理解**:SLO 校准**必要但不充分**——
- 必要:让 M2.4 屏蔽机制不再钉死反馈链
- 不充分:ReLU + M2.4 联合在边界仍让 PID 单输入,ratio 仍钉 max

**新定位**:Sampling 模块的真实价值不是"让 PID 双输入工作",而是:
1. **定义系统可服务边界**(code 那种 viol 无论 SLO 怎么调都极高 → workload 不可服务,出 admission control / scale-up 范畴)
2. **提供 evaluation main result 的目标档**(thesis 报这个档下 meet_slo%,不是 strict 50ms 这种打地板的档)
3. **校准 PID 的"启动 ramp 目标"**——PID 仍是分段控制器(ramp + 边界保护),但 ramp 目标合理

**走向 β+(更精确)**:thesis 措辞改为
> "M3.1 是分段控制器,启动期闭环反馈快速逼近 SLO-aware 平衡点,饱和稳态期边界保护。Sampling 模块定义合理的 SLO 目标,让启动 ramp 收敛到工程意义点,而非物理不可达的 strict 档。Iter 维度 leading indicator (P1.7b) 旁路 PID 屏蔽机制,在饱和场景仍提供 in-flight 反馈。"

**当前路径**:走 B(离线 calibrated)+ P1.7b 配合。C(动态)作为未来扩展。

### 4.2 P1.7b 双向 leading indicator

P1.7 单向 urgency FAIL → 需要 TTFT urgency + TPOT starvation 双向。前置:实装 `QueueSnapshot.decode_oldest_silence_ms`(P2.0 缺口 #1,1-2h)。等 P1.6e/f 结果定夺前不启动。

### 4.3 Graph 模块定夺(论文写作前)

候选 G+F(撤下转 finding)/ D(合并 chunking)/ F(只留诊断)。讨论材料完整,论文写作时三选一。

---

## §5. 任务表(精简)

| ID | 阶段 | 状态 | 工作量 | 备注 |
|---|---|---|---|---|
| #6 | P1.6a 屏蔽审计 + 信号滞后量化 | ✅ done | — | §3.3 |
| #7 | P1.6b 反馈链 F1+F2 实装 + 实验 | ✅ done | — | §3.3 |
| #10 | P1.6e SLO target sweep | ✅ done | — | §3.3b |
| — | **P1.6f** calibrated SLO 重跑 baseline(conv=200ms) | pending | 0.5d | 4 configs × 2 windows × 3 seeds |
| #8 | P1.6c thesis 措辞 + Threats to validity | pending | 1-2d | 等 P1.6f + P1.7b 完整 |
| #9 | P1.6d TDM 系统设计代码审计 | pending | 0.5-1d | kp_q=0 等错配清查 |
| #1-4 | **P1.7b** 双向 leading indicator(iter 维度旁路) | pending | 1-1.5d | 救 saturated 场景的唯一路径 |
| #5 | P2.5 Graph 模块定夺 | pending | ~1h | 论文写作前 |

**论文短板补全**(冲 CCF-B 候选):Sampling 模块实装 + ShareGPT trace + 跨模型对照 + Threats to validity。

---

## §6. Done 标准(关键任务)

- **P1.6e**:某 SLO 档下 conv `pct_at_ratio_max < 80%` 且 `err_tpot_eff 非零率 > 20%` → PID 双输入复活(走向 α);全档钉边界 → 走向 β
- **P1.6f**:calibrated SLO 下 M3.1 在 conv/code 双 trace meet_slo% 显著优于 baseline,跟 azure_main 主结果可比
- **P1.7b**:code 严档 Δ(M3.3-M3.1) ≥ 0(回正 P1.7 FAIL)且 conv 不退步(min > -1pp)
- **3 seeds ±std 描述性**;临界结果(±1pp 内)扩到 5 seeds

---

## §7. 关键 artifact 路径

```
代码:
  vllm-ascend/vllm_ascend/core/tdm/     插件 (85/85 单测)
  Ascend-PD-TDM/experiments/            driver + posthoc + runner
  Ascend-PD-TDM/memory/                 设计/状态文档

实验数据:
  results/azure_main/                   论文 main 候选(2 traces × 4 cfg × 3 seeds)
  results/azure_p15/                    跨窗 (P1.5)
  results/azure_p17/                    P1.7 urgency FAIL
  results/azure_p16b/                   F1+F2 反馈链修
  results/azure_p16e/                   SLO target sweep (in_progress)
  results/baseline_c3_audit{,_aiv}/     P1.0/P1.0b graph audit
  results/static_scan_3a/               P0-1
  data/azure_trace/                     原始 trace
```

---

## §8. 不要再做的

- ❌ 把 kernel 速度作为论文核心论据
- ❌ 设计 dynamic chunk_tokens 控制器(chunk pareto 单调)
- ❌ 把 TDM 或 reactive 作为范式贡献抬高
- ❌ stationary workload 反复跑 PID 调参
- ❌ 系统完整前画 paper figures
- ❌ **用 fixed strict SLO(50/500)做 PID target 跑主实验**——必须 workload-conditional
- ❌ 急着加新模块(KV/SpecDecode/multi-priority)扩 scope
