# Finding 链 + 设计与实测的差距

> 时间倒序。新 finding 闭环时追加一段。

---

## P1.7b(2026-05-15)双向 selector smoke + threshold sweep — 触发率太低 + code 上净退步

**实装:** TDMScheduler 加 `starvation_tpot` 双向预警(P1.7 单向 urgency 的补全)。selector.peek() 在 tokens≥1 + decode_oldest_silence_ms ≥ threshold 时强切 decode,不扣 bucket。设计上跟 urgency_ttft 互斥(tokens<1 时 urgency,tokens≥1 时 starvation),单 iter 至多触发一个。完整 16 单测全过。

**配置:** 5 cfg(M3.1 / M3.2 / M3.3 default starv=3.0 / M3.3 starv2=2.0 / M3.3 starv5=5.0)× 2 traces(conv@1860 / code@570)× 3 seeds × 60s。`results/azure_p17b/`。Wall ~2.5h。

**核心数据(meet_slo% tpot<200,3 seeds 平均):**

| cfg | conv | vs M3.1 | code | vs M3.1 |
|---|---|---|---|---|
| M3.1 | 86.79 | — | 14.51 | — |
| M3.2 (urgency only) | 87.93 | +1.14 | 13.04 | -1.47 |
| M3.3 default (starv=3.0) | 88.65 | +1.86 | 11.67 | **-2.84** |
| M3.3 starv2 (=2.0) | 85.96 | -0.83 | 13.77 | -0.74 |
| M3.3 starv5 (=5.0) | 89.27 | +2.48 | 13.25 | -1.26 |

**urgency_tpot 触发率(对 1200-1450 iter/cfg 窗口):**

| cfg | conv 触发 | code 触发 |
|---|---|---|
| M3.3 default (starv 3.0) | 0.2%(3 次/3 seeds) | **0%** |
| M3.3 starv2 (2.0) | 0.7%(10 次) | 1.0%(12 次) |
| M3.3 starv5 (5.0) | **0%** | **0%** |

**三条不容回避的发现:**

1. **触发率太低撑不起"双向预警"thesis claim** — 设计目的是 iter 粒度瞬时响应,实际 0.2-1% 频率近乎空跑;starv5 在两 trace 上都 0% 触发
2. **conv 上 starv5 (0% 触发) 反而最好(+2.48pp)** — M3.3 default 触发 3 次 vs starv5 触发 0 次,starv5 反而赢 0.6pp。说明 conv 上 +2.48pp 不是 starvation 在做事,是 noise/variance。**反相关:** starv2 触发最频繁(0.7%) → 反而退步 -0.83pp,提示 starvation 触发当下其实 hurt
3. **code 上 M3.3 default 全方位输 M3.1(-2.84pp tpot<200,-5.37pp tpot<300)** — 不是 starvation 直接触发造成(default 0% 触发),但 source 字段 `constraint_max_slice` 比例从 25% → 50%。推测:starvation logic 代码路径(即使没触发)间接改变了 selector 状态机或 phase 切换 pattern,在 saturated workload 上放大成 -3 ~ -5pp 损失

**对 thesis 的影响:**
- P1.7b 当前实装**不能直接落地** — conv 上"赢"不可归因到机制,code 上净损失
- "iter 粒度双向预警" thesis claim 缺数据支撑
- D-010 ablation 矩阵 claim B2/B3(快回路 + 双维度协同)目前没有正向证据

**三条待诊断方向(新 session 接):**
- (B) **诊断 code 上 default 0% 触发根因** — saturated workload decode 永远在 produce token(虽然慢),`decode_oldest_silence_ms` 实际分布需要 post-hoc 看,理解为什么从未超 150ms
- (C) **诊断 code 上 max_slice 飙升根因** — selector 状态机里 starvation check 路径是否改变了 token bucket crediting 时机/顺序(可能 implementation bug)
- (D) **诊断后决定 P1.7b 落地策略** — 调更激进 threshold / 修实装 / 接受弱 finding 转向其他 thesis 方向

**未推动新决策**,等诊断结果(任务 #24/#25/#26)。

---

## P1.8(2026-05-14)m31_fia ablation — kernel 贡献 ≈ 0,M3.1 优势全来自 phase-pure 调度

> **2026-05-15 注解(D-010 修正):** 下表 "phase Δ" 和 "phase 占比" 严格说**不是** "phase-pure 单机制贡献",而是 **TDM-paradigm bundle vs CP-paradigm bundle 总差**。bundle 内部机制(调度器实现 stack / chunk_tokens / queue+batching 行为 / max_num_seqs 等)互相耦合,不可单变量归因。
>
> 论文写法不再写「phase-pure 贡献 X%」,改写「TDM-paradigm vs CP-paradigm bundle Δ = X%,内部机制不可单变量归因」。phase-pure 单机制贡献需要 **同 stack ablation**(TDMScheduler 加 `mixed_mode` 开关)才能 isolate,见 D-010。
>
> launch 参数只做两条 sanity check:c1/c3/m31 都开 graph capture(`enforce_eager` 一致);c3 默认 `chunk_tokens` 写清楚作为 paradigm 配置交代。其他 paradigm-specific 参数不强制对齐。

**配置:** 4 configs × 2 traces(conv@1860s / code@570s)× 3 seeds × 60s, warmup 20s。新加 `c2_tdm_m31_fia` config:M3.1 同样 phase-pure + chunk=2048,但 monkey-patch 让 attention 强制走 FIA(等价 vllm-ascend v0.13+ 行为),绕开 v0.11.0rc1 dedicated kernels。`results/azure_m31_fia_ablation/`。

**结果(跨 3 seeds 平均):**

| trace | metric | m31_2048 | m31_fia | c3_cp | kernel Δ | phase Δ | phase 占比 |
|---|---|---|---|---|---|---|---|
| conv | ttft_mean | 219.2 | 226.9 | 311.8 | +7.7 | +85.0 | **92%** |
| conv | tpot_p99  | 175.7 | 175.0 | 170.8 | -0.7 | -4.2 | **86%** |
| conv | e2e_mean  | 36483 | 35796 | 34728 | -687 | -1069 | **61%** |
| code | ttft_mean | 387.4 | 385.8 | 708.9 | -1.6 | +323.1 | **101%** |
| code | tpot_p99  | 1667.5 | 1654.4 | 662.2 | -13.1 | -992.2 | **99%** |
| code | e2e_mean  | 43931 | 43691 | 36682 | -240 | -7009 | **97%** |

(kernel Δ = m31_fia - m31_2048;phase Δ = c3 - m31_fia;phase 占比 = phase Δ / 总差距)

**核心结论(thesis 级):**

1. **dedicated kernel 不是 M3.1 的优势源** — kernel Δ 在所有指标上 ≤ 13ms 或 < 1%,统计上等于 0。强制 FIA 后 M3.1 几乎不掉性能。
2. **M3.1 vs C3 全部差距来自 phase-pure 调度** — 86-101% 跨指标 / trace。
3. **phase-pure 是双刃剑:**
   - **TTFT 上 M3.1 大胜 C3** — conv -30% (219 vs 312),code -45% (387 vs 709)
   - **TPOT_p99 / e2e 上 C3 反胜 M3.1** — code tpot_p99 M3.1 是 C3 的 2.5×(1668 vs 662),e2e 输 7.2s (-16%)
   - 因为 phase-pure 让 prefill 集中跑 → TTFT 降;但巨型 prefill iter 期间 decode 完全停 → TPOT_p99 升。

**对 thesis 的影响:**
- **kernel argument 死了** — vllm-ascend v0.13+ 移除 dedicated kernels 不会让我们掉性能,论文不需要这条论点(原 PROJECT.md「不能偏离 §5 kernel 速度只解释 C3 反常」反而高估了 kernel 重要性)
- **phase-pure 在 saturated 长 prompt 下输 TPOT_p99 是真实物理代价**,不能靠"kernel 更快"遮掩
- **SLO-adaptive TDM 的论点更聚焦了:** 核心卖点 = 「根据 workload regime 动态选择 phase-pure / mixed」,而不是「phase-pure 一定赢」。M3.1 适合 TTFT 紧 + 短 prompt,C3-style 适合 TPOT 紧 + 长 prompt → 这正好 motivate 分段控制器 + 双向 selector(P1.7b)。

推动 Gap-5 + D-009。

---

## P1.6e(2026-05-13)SLO 档 sweep — thesis 调整核心证据

**配置:** M3.1 × 4 个 SLO 档(tpot=50/150/200/300ms)× 2 windows × 3 seeds × 60s。`results/azure_p16e/`。

**conv 跨 seed 平均:**

| slo_tpot | tpot_viol | M2.4 屏蔽 | err_tpot_eff 非零 | ratio 钉 max |
|---|---|---|---|---|
| 50  | 0.94 | 99.5% | 0.5% | 82.9% |
| 150 | 0.38 | 94.1% | 1.7% | 81.6% |
| 200 ⭐ | 0.09 | 28.2% | 3.6% | 83.1% |
| 300 | 0.02 | 5.4%  | 1.4% | 84.0% |

**code 跨 seed 平均:** 几乎完全不变(M2.4 永恒 94.5%,viol 0.57-0.92)。

**核心结论(thesis 级):**
- SLO 档校准对反馈链有巨大影响(conv M2.4 屏蔽从 99.5% 降到 5.4%)
- **但 PID ratio 仍钉 max 80%+ 跨所有 SLO 档** ——ReLU + M2.4 联合让 PID 在边界几乎必然单输入
- **SLO 校准必要但不充分**
- code 类工作负载在我们硬件 + 11qps 下不可服务(物理过载),无论 SLO 怎么调
- 推动 D-002(thesis 改「分段控制器 + 双维度协同」)、D-003(Sampling 模块定位转向)

**PID 单输入的两类失败模式:**

| 模式 | 触发条件 | 含义 |
|---|---|---|
| M2.4 屏蔽 | viol > target 持续 | SLO 过严或物理过载 |
| ReLU clip | viol < target | SLO 过松,系统有余量 |
| 持续双输入 | viol ≈ target 稳定 | 极窄区,稳态系统几乎不可能停在这里 |

---

## P1.6b(2026-05-13)反馈链 F1+F2 修复

**问题:** P1.6a 发现 `record_request` 只在 `on_finish` 调用 → ttft/tpot 信号有「请求 lifetime 量级」延迟。

**实装:** F1(ttft on first token)+ F2(tpot on each token interval),85 单测过。

**实测:**
- ttft warm-up 从 16.5s 降到 2.0s
- **但 tpot 信号一上线就 100% 违例不变**

**结论:** 反馈链滞后不是 PID 失效根因,**SLO 档目标错才是根因** → 直接 motivate P1.6e SLO 档 sweep。

---

## P1.6a(2026-05-13)PID 屏蔽审计 + 信号滞后量化

`results/p16a_audit_summary.txt`。

| Finding | 含义 |
|---|---|
| ratio 启动 5-10s 从 0.3 推到 ratio_max=0.8 之后钉死 | PID 稳态等效「启动爬升 + 静态 ratio_max」 |
| M2.4 saturation 屏蔽占 warm 期 99.4% tick | 设计合理(tpot 违例 mean 97%) |
| **backlog 项 `kp_q` 实测 = 0,设计装饰** | M2.3 leading indicator 未启用 |
| `record_request` 只在 `on_finish` 调用 | 反馈链 bug,F1+F2 已修(P1.6b) |

---

## P1.7(2026-05-12)单向 urgency selector — FAIL

**实验:** M3.2(M3.1 + selector 加 TTFT urgency 单向预警)vs M3.1。`results/azure_p17/`。

**结果:** 8/8 严档 tier 中 7 个 Δ(M3.2 - M3.1) < 0。

**根因:** 单向 urgency 只在 TTFT 紧迫时强切 prefill,没有反向(TPOT 沉默时强切 decode)。在 saturated 场景下,TPOT 端持续违例 → 单向 urgency 反而把 ratio 错误地往 prefill 偏。

**结论:** selector 必须**双向**——TTFT 紧迫 + TPOT 沉默 → motivate P1.7b。

---

## P1.5(2026-05-12)跨窗扩展

**配置:** `results/azure_p15/`,2 traces × 3 windows × 4 configs × 3 seeds。

**结果:**
- conv 4/4 tier 跨窗 Δ(M3.1, M1+chunk) PASS(mean ≥ 0, min ≥ -0.6pp)
- **code 严档 tpot=100/150 跨窗 Δ ≈ 0** ——M3.1 在 saturated 无增量

**结论:** code 在我们硬件 + 当前 qps 下饱和,慢回路救不了 → motivate P1.7b 救饱和场景。

---

## P1.4(2026-05-11)Azure trace 4-way 主结果

**配置:** `results/azure_main/`,2 traces × 4 configs × 3 seeds × 60s。

**conv meet_slo%:**

| tier | C1 | M1+chunk | M3.1 | C3 |
|---|---|---|---|---|
| ttft=500/tpot=150 | 55.6 | 60.7 | **64.1** | **68.6** ⚠️ |
| ttft=500/tpot=200 | 86.4 | 83.5 | **88.6** | 87.2 |

**code meet_slo% (ttft=500/tpot=200):** M3.1 **12.2 vs C3 2.8(+9.4pp)**。

**正向:** conv tpot=200 +5.2pp vs M1+chunk, code tpot=200 +9.4pp vs C3。
**负向:** conv tpot=150 输 C3 4.5pp(P/D 分开结构性代价,P0-3 归因)。

**注意:** 这是 strict SLO 档跑的,不符合「主报只在调过的档下报」原则,P1.7b 阶段会用调过的档重跑替代。

---

## P1.0 / P1.0b(2026-05-12)Graph 模块降级

- vllm-ascend 已内置 `cudagraph_capture_sizes` + decode 自动 pad
- P1.0b 实验:AIV(23 sizes)vs FFTS+(15 sizes)
- **capture 数量非单调** ——AIV 反而比 FFTS+ 输 -1.5 ~ -23.8pp

**结论:** Graph 模块**没有算法空间**,降级。论文写作前三选一:写进 paper(G+F)/ 合并 chunking(D)/ 只留 finding(F only)。见 D-005。

---

## P0-3(2026-05-07)长 prompt sweep — thesis 双 pivot 起点

**实验:** 4 configs × post-hoc SLO grid sweep on long prompt workload。`results/long_prompt_sweep/`。

**Finding:**
- strict 档(ttft<500ms / tpot<50ms):所有 4 个 config 趴在地板(meet_slo% < 5%),看不出差异
- loose 档(ttft>2000ms / tpot>300ms):所有 4 个 config 100%,也看不出差异
- **区分带** ttft∈[500, 1500] × tpot∈[100, 200]:M2.7 一致胜 C1/C2/C3

**双 pivot:**
1. Sampling 模块从设计稿升格为实装目标(workload-conditional SLO 框架)
2. M3 chunking 从「在线 controller 调 chunk_tokens」收敛为「静态 chunk=2048,推进区分带边界」

---

## P0-2(2026-05-08)8K canonical M2 家族 sweep — M2.X 各 guard 实测

数据已删,关键数字归档:

**R2+R3 SLO% 几何均值排名:**

| 方案 | R2 SLO% | R3 SLO% | geomean |
|---|---|---|---|
| C1 hybrid | 65.0 | 78.8 | **71.6**(全局最优参考线) |
| M2.7 ⭐ | 62.6 | 70.1 | **66.2** |
| M2 | 61.3 | 70.3 | 65.6 |
| M2.5 | 53.5 | 74.5 | 63.1(R3 单点最优) |
| M1 static | 53.5 | 72.6 | 62.3 |
| C3 chunked | 50.8 | 63.9 | 57.0 |

**M2.X 各 guard 实测:**

| 机制 | 设计初衷 | 实测 |
|---|---|---|
| starvation guard | ratio 撞底关 err_tpot 让 ttft 救场 | 没用,guard 释放后立刻拉回 floor |
| hysteresis release | 解振荡 | 没用,振荡不是主因 |
| backlog-aware kp_q | queue age 当 leading indicator | 没用,`oldest_age` 90% 时间 = 0 |
| tpot 饱和检测 | tpot 物理不可达时屏蔽 err_tpot | 部分有用,但激活前 ratio 已被推到底 |
| ReLU err clip | 满足的 SLO 不该反推 ratio | **关键修复** |
| saturation_min_ticks=1 立即锁 | 立即锁 saturation | 锁住 ratio=0.30 但 R3 反差 |

---

## P0-1(2026-05-06)static ratio scan — regime-dependent 最优 ratio

**实验:** M1 static ratio ∈ {0.05, 0.1, ..., 0.8}。`results/static_scan_3a/`。

**Finding:**

| Regime | 实测最优 ratio |
|---|---|
| R2 (qps=16) | ≈ 0.27 |
| R3 (qps=32) | ≈ 0.05 (floor) |

**结论:** 单一全局最优 ratio 不存在 → 真正 adaptive 必须 regime-aware → motivate M3。

---

## P0(~4 月)早期 profile

`figures/fig{1-7}_exp_*.png` 保留。原始数据 `results_exp_*.json` 已删。

| Exp | 结论 |
|---|---|
| A | ACL Graph padding 浪费(prefill 23.4%, decode 16.5%);eager fallback per-token +26%;48 种 capture shape ∈ [1,512] |
| B | Prefill compute-bound(AICore 69.8%, HBM 15.8%);Decode memory-bound(42.6% / 25.1%)→ 资源互补 |
| C | DCMI 采样开销:HBM BW ~8%(频率无关,GIL 瓶颈);AICore 100ms ≈ 0% |
| D | TPOT 恒定 ~21-27ms,吞吐 1→64 并发 58.4× 近线性 → 低并发 NPU underutilized |
| E | Unified vs Phased:short_4req +9.6%, short_16req -14.8%, mixed_16req -20% → **动态选择必要** |
| F | 简单「向下取整」Graph-aware 在 prefill 小 batch 反效果 -10~-20%;decode 大 batch +0.7~+4.3% → 正确方向「向上凑」 |
| E3 | Disagg 1P1D 在 Qwen3-4B 2×910B3 -2~-12%(同机 TP=2 通信 < 分离空闲)→ C4 不是真实威胁 |

---

# 设计与实测的差距(thesis-critical gap)

> 这部分专门记录设计文档假设 vs 实测推翻的偏差。每条带影响范围。

## Gap-1:thesis「闭环反馈」实际是「分段控制器」

- **设计期 claim**(`idea_proposal.md` 原版):SLO-aware 闭环反馈控制 P/D 比例
- **P1.6e 实测:** 跨 4 档 SLO conv ratio 钉 max 80%+,PID 几乎从不在双输入工作区
- **真实工作模式:** 启动期闭环爬升(0.3→0.8 用 5-10s)+ 饱和稳态边界保护
- **处理:** D-002 改 thesis 为「分段控制器 + 双维度协同」

## Gap-2:Sampling 模块原 claim「让 PID 双输入工作」

- **设计期 claim**(`slo_sampling_module_design.md` 原版):提供 SLO target 让 PID 在双输入工作区
- **P1.6e 实测:** SLO 校准让 M2.4 屏蔽从 99.5% 降到 5.4%,但 ratio 仍钉 max → 必要但不充分
- **处理:** D-003 转向「定义可服务边界 + 校准爬升目标 + 主报结果档」

## Gap-3:M3 chunking 原计划「在线 controller 调 chunk」

- **设计期 claim**(`m3_chunking_design.md` 原版):M3.4 SLOReactiveController 加 chunk_tokens 输出维度
- **p1_chunk_scan 实测:** chunk pareto 单调,chunk=2048 是 sweet spot,无 dynamic 设计空间
- **处理:** D-006 chunking 收敛为静态 chunk=2048,不做 dynamic controller

## Gap-4:Graph 模块原作为「三模块之一」

- **设计期 claim**(`system_architecture.md` 原版):Graph-aware 是 TDM 三大控制模块之一
- **P1.0/P1.0b 实测:** vllm-ascend 已内置 capture + AIV 23 sizes 反而 -1.5~-23.8pp,capture 数量非单调
- **处理:** D-005 降级为论文 finding,三模块变两模块(Sampling + SLO 控制器)

## Gap-5:dedicated kernel 速度被假设为 M3.1 优势组件之一

- **设计期 claim**(`PROJECT.md` 不能偏离 §5 + paper.md threats):M3.1 优势 = phase-pure 调度 + dedicated kernel 速度(2.7-3.4× 比 FIA 快),kernel 只是不报主图但作为 contribution 一部分
- **P1.8 实测:** 强制 M3.1 走 FIA 后所有指标变化 ≤ 13ms 或 < 1%,跟 m31_2048 统计等价。M3.1 vs C3 差距 86-101% 来自 phase-pure 调度本身,kernel 贡献 ≈ 0
- **处理:** D-009 — kernel 完全移出 contribution。phase-pure 是真正卖点,但要承认双向(TTFT 赢 / TPOT_p99 输)。这反过来强化「需要 SLO-adaptive 切换 regime」论点

---

# 已废弃的设计路径(防反复探索)

| 路径 | 试过的实验 | 为什么作废 | 永久 / 临时 |
|---|---|---|---|
| 单向 urgency selector(只看 TTFT) | P1.7 (azure_p17) | 8/8 严档 7 个反向 | 永久(被 P1.7b 双向取代) |
| dynamic chunk_tokens 控制器 | p1_chunk_scan | chunk pareto 单调,无双向 trade-off | 永久 |
| M2.X 各种 guard(starvation / hysteresis / backlog kp_q) | P0-2 M 家族 sweep | 实测无效或装饰 | 永久(只留 ReLU clip + 饱和检测) |
| 把 strict SLO(50/500)作为 main 实验目标 | azure_main | 物理不可达,所有 config 趴地板 | 解禁条件:Sampling 模块实装 |
| 升级 vllm-ascend > v0.11.0rc1 | — | PR #4623 删 AscendScheduler,无迁移路径 | 永久 |
| 把 reactive 控制 / TDM 抬为范式贡献 | — | thesis 已转「分段控制器」 | 永久 |
| 在 stationary workload 反复跑 PID 调参 | long_5way_q1632 | PID 在 stationary 不显效(+0.9/+0.3pp) | 永久 |
| dynamic Graph-aware 调度 | P1.0/P1.0b | vllm-ascend 已内置 + 非单调 | 永久(可能整合进 chunking 或只留 finding) |
| 合成 burst 撬 PID | burst_seed* | Δ ≈ ±0.7pp,撬不动 | 临时(真实 trace 突发已被 azure_p15 验证) |

---

# 未决核心问题

1. **P1.7b 双向 selector 具体设计**:绕开慢回路屏蔽的语义/双向 threshold/跟 PID 的协调,见 `design/slo_pid.md` 快回路节
2. **P1.6h calibrated baseline 何时跑**:D-001 推迟到 P1.7b 阶段,届时四 configs 都用调过的档跑
3. **Graph 模块论文形态**:G+F / D / F-only 三选一,论文写作前定(D-005)
4. **C3 在 NPU 上反常的根因**:vllm-ascend chunked prefill 实现层核查(`notes.md` 原版 §11 遗留),论文 threats to validity 需要
