# T6 burst goodput sweep — findings(2026-05-25 finalized,m31-fix 全数据版)

> 完整数据落地:`results/m31fix_validate/` + `results/c3_chunk2048_supplement/` + `results/phase_2_t6_burst_goodput/*_nonpid/`
> 聚合 JSON:`results/phase_2_post/m31fix_phase1_pointwise.json`
> Post-hoc 脚本:`experiments/posthoc_m31fix_phase1.py`
> 决策链:`DECISIONS.md` D-013

## 实验 matrix

| 维度 | 取值 |
|---|---|
| Workload | conv(短 prompt 长 output)+ code(长 prompt 短 output)|
| Burst intensity k | conv: {0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0};code: {0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9} |
| SLO tier | conv: s1=200/120, s2=300/150, s3=500/200;code: s1=500/200, s2=500/700, s3=2000/400 |
| Seeds | 0, 1, 2 |
| Total | 126 cells × 3-seed median(0 failed run) |

**Arrival**:`trace_sampled_burst`(period=10s,high/low QPS 按 k scale,frac=0.1 conv / 0.2 code)
**Trace**:Azure conv + code,prompt cap=7000 token,output cap=600
**Duration**:90s,warmup=30s,window=60s
**Hardware**:2× Ascend 910B3 TP=2,Qwen3-8B

## Configs

| config | scheduler | server max_num_batched_tokens | chunk 行为 |
|---|---|---|---|
| **c1** | AscendScheduler (unified, no chunked prefill) | 8192 | 无 chunk,整 prompt 一 iter |
| **c3-fair** | vllm v1 chunked prefill | **2048** | chunk = budget,prefill + decode 共享 |
| **m31-fix** | PD-TDM phase-pure | 8192(无效)+ chunk_budget=2048 | prefill iter 多 reqs 并行,共享 2048 budget |

c3 / m31 chunk size 公平(都 2048)。c1=8192 是 unified 唯一合理配置(改 2048 会 reject 长 prompt,c1 没 chunked prefill 路径)。

## Winning region — m31-fix vs c3-fair(3-seed median Δmeet%)

**CONV(21/21 全胜,无 tie 无 loss)**

| SLO | k=0.5 | k=1.0 | k=1.4 | k=1.8 | k=2.2 | k=2.6 | k=3.0 |
|---|---|---|---|---|---|---|---|
| s1 200/120 | +9.9 | +23.8 | +26.7 | +28.3 | +30.5 | +30.9 | **+34.8** |
| s2 300/150 | +2.8 | +7.2 | +3.0 | +7.0 | +6.4 | +5.3 | +6.3 |
| s3 500/200 | +4.0 | +2.3 | +3.3 | +2.6 | +2.2 | +2.5 | +1.9 |

**CODE(9 decisive / 11 tied / 1 noise loss)**

| SLO | k=0.7 | k=1.4 | k=2.1 | k=2.8 | k=3.5 | k=4.2 | k=4.9 |
|---|---|---|---|---|---|---|---|
| s1 500/200 | +12.3 | +27.9 | **+71.5** | **+77.8** | **+77.2** | **+77.4** | **+73.9** |
| s2 500/700 | +2.5 | +0.6 | +2.1 | 0.0 | +0.6 | +0.6 | +0.9 |
| s3 2000/400 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0(都 100%) |

## Goodput 提升(tok/s)

- conv s1 k=3.0:m31 3396 vs c3 1525,**Δ +1870**(+123%)
- code s1 k=4.9:m31 438 vs c3 87,**Δ +351**(+403%)

## Latency 维度 winner count(3-seed median,21 cells × 6 metric)

| Workload | metric | c1 win | c3 win | **m31 win** | tied |
|---|---|---|---|---|---|
| conv | ttft_mean | 0 | 0 | **20** | 1 |
| conv | ttft_p99 | 2 | 0 | **17** | 2 |
| conv | tpot_p99 | 0 | 0 | **20** | 1 |
| code | ttft_mean | 0 | 0 | **21** | 0 |
| code | ttft_p99 | 0 | 0 | **21** | 0 |
| code | tpot_mean | 0 | 0 | **21** | 0 |
| code | tpot_p99 | 0 | 0 | **21** | 0 |

**c3-fair 在任何 wl/latency metric 上 0 wins**;c1 仅 conv ttft_p99 偶 2 cell。

## Telemetry 机制验证(code k2.8 s1 cell)

**m31 phase 行为(window 5-90s,898 iters)**:
- prefill iter:n=449,mean 150ms,batch_num_tokens 恒 2048,batch_num_reqs mean 2.16 max 6
- decode iter:n=449,mean 39ms,batch_num_reqs mean 29 max 40
- 严格 1:1 alternation(burst 满负载下)
- decode-iter inter-arrival p50=189 / p99=203 ms ≈ 实测 tpot p99=190ms

**Cycle 时间对比**:
- c3 单 mixed iter:~225 ms(≈ c3 实测 tpot p99)
- m31 cycle (prefill + decode):150 + 39 = **189 ms**
- 差 36ms(16%)→ 直接解释 tpot 优势

**机制根因**:
- c3 mixed batch 跑 varlen attention(prefill chunk 长 query + decode 单 token query 混合)→ kernel 路径慢
- m31 prefill iter 跑纯 prefill kernel;decode iter 跑纯 decode kernel → 每种 kernel 在最擅长形状下跑
- m31 prefill iter 多 reqs 并行(mean 2.16, max 6 reqs/iter)→ burst 队列消化更快(c3 一 iter 只 1 prefill chunk)

## PID 不是优势来源(D-011 立场确认)

- target_ratio 推 0.8 ratio_max 撞顶,实际 prefill iter 占比 12-50% 由 waiting queue 决定
- 8 cell verify:7 cell 都见 target_ratio 0.80 max,实际 ratio 12% (conv k0.5 轻负载) ~ 50% (code k2.8 burst 满)
- PID 实际为 "SLO 违反诊断指示灯",paper 报 m31 = static ratio per workload

## ~~c3 vs c1 baseline 复现~~ → Sarathi vs Vanilla CB 基准对照(D-014 重定位,2026-05-26)

**说明**:原 §"c3 vs c1 复现" 把 c1(vllm-ascend AscendScheduler default,admit-driven phase-pure)当作 vanilla CB 的代表 — 但 c1 在 LLM serving 文献里没有标准命名,跟主流 mixed-batch vanilla CB 不是同一 design。D-014 把 baseline 重定位到主流文献:**Vanilla CB = 老 c3 chunk=8192**(`phase_2_t6_burst_goodput/*_nonpid/c3_cp_qps0.0.json`,Azure trace prompt cap=7000 < 8192 → chunk 不触发 = 真 mixed batch),**Sarathi chunked prefill = c3-fair chunk=2048**(`c3_chunk2048_supplement/`)。**c1 数据 / 代码保留 internal,paper 不主报**。

**NPU 上首次复现 Sarathi 论文 OSDI'24 finding(F2c heatmap 实测,3-seed median Δ pp = Sarathi − Vanilla CB)**:
```
conv s1:  -6.2  -6.5  -6.2  -3.5  -1.6  -0.5  +1.1
conv s2:  +1.2  -3.0  +2.4 +10.1 +14.6 +17.8 +22.6
conv s3:  -5.2  -2.0  +1.2 +12.9 +24.9 +26.2 +34.9
code s1: +13.0 +22.4  -1.1  +1.7  +0.9  +1.1  +5.5
code s2: +16.3 +43.6 +61.6 +70.6 +74.5 +75.2 +74.8
code s3:  +0.6  +7.1 +32.5 +87.5 +80.0 +80.6 +80.6
```

**code workload(长 prompt)上 Sarathi 大胜 Vanilla CB**(NPU 复现 Sarathi OSDI'24 核心 finding):
- code 4.9 s3 cell **Sarathi meet 100% vs Vanilla CB 19.4%(+80.6pp)**
- code 2.8 s1 cell **Sarathi TPOT p99 228ms vs Vanilla CB 670ms(−65%)**
- max +88pp(code 2.8 s3) — chunked prefill 把长 prompt 灾难救活的 dramatic 量化

**conv workload(短 prompt)上 Vanilla CB 反输 Sarathi**(fresh finding,Sarathi 论文没明示):
- conv 1.0 s1 cell:Vanilla CB meet 55.7% vs Sarathi 49.2%(+6.5pp 反向)
- conv 2.2 s1 cell:Vanilla CB 35.4% vs Sarathi 33.8%
- 物理原因:短 prompt 在 chunk=8192 整 iter 一次跑完 ttft 更快,chunk=2048 切多份反向拖慢首 token ~30-40ms
- **paper §5 caveat:NPU 首次实测的 Sarathi chunked prefill 在短 prompt regime 反向**
- **对 PD-TDM 不构成威胁** — PD-TDM 在两边都赢(长 prompt 拿 Sarathi 的好处,短 prompt 拿 phase-pure 避开 mix overhead)

**PD-TDM 跟两个 baseline 关系(D-014 final framing)**:
- **vs Sarathi(F2 heatmap,unique contribution)**:phase-pure variant of chunked prefill,strict SLO 上 +30~+77pp 增益
- **vs Vanilla CB(F2b heatmap,总优势)**:NPU 复现 Sarathi 部分 + phase-pure 额外增益的总和,全谱完胜,max +87.5pp(code 2.8 s3)
- **Paper 须明确归 credit**:vs Vanilla CB 的胜利绝大部分来自 chunked prefill(跟 Sarathi 共享),PD-TDM 独占 contribution 在 F2 那张图

## Thesis framing 修正(对应 D-013)

**原 D-012 framing 不成立**:"deliberately trades tail latency for mean latency to maximize goodput"
- 数据显示 m31 mean 和 tail 同时改善,不是 trade-off

**修正后 framing**:
> Phase-pure temporal multiplexing 把 mixed-batch 的"持续低幅 prefill-decode 干扰"替换为"phase-specific 高效 batch"。一个 cycle (prefill_iter + decode_iter) 时间(实测 189ms)短于 c3 单 mixed iter (~225ms),消除 mixed-attention overhead。**mean 和 tail 同步改善,不是 trade-off**。

## Caveat

- 结论限定 Azure burst trace + Qwen3-8B + 2-NPU + chunk_budget=2048
- 极重 prefill demand regime(如长 prompt + 高频 burst)下,m31 cycle 退化为"多 prefill iter + 1 decode iter",cycle 时间变长,trade-off 此时会重新暴露 — 当前 burst regime 没触发
- ~~c3 vs c1 不普适胜出(conv strict SLO 反例)~~ → D-014 重定位为 Sarathi vs Vanilla CB:**conv 短 prompt 上 Sarathi 反输 Vanilla CB**(-2~-6.5pp);仍是 paper §5 caveat,只是改了 baseline 命名

## c3 iter telemetry — 直接量化 mechanism #1(2026-05-25)

**背景**:之前 c3 iter 时长从未实测(c3_cp 是 TDMScheduler passive 模式,passive 不 record_iter)。机制论证只靠物理推理 + outcome。
**Patch**:`scheduler.py` passive 分支加 IterRecord(see `vllm-ascend/vllm_ascend/core/tdm/scheduler.py` __init__ + schedule + update_from_output)。Unit test 104/104 通过。
**Run**:`results/c3_telemetry/code_k{2.8,4.9}_seed0`,聚合 `experiments/posthoc_c3_telemetry.py`。

### c3 mixed iter 时长分布(code k=2.8 cell,与 m31 telemetry verify cell 对齐)

| metric | mean | median | p99 |
|---|---|---|---|
| 全 c3 iter | 137.5ms | **208.0ms** | 286ms |
| chunk-cap iters(57% 占比,prefill-dominant)| **220.6ms** | 215.5ms | 358ms |
| decode-dominated iters(42% 占比) | 25.0ms | 24.8ms | 33.8ms |

(mean << median 因为分布 bimodal — 慢的 chunk-cap iter 和快的 decode iter 混在一起,mean 被快 iter 拉低)

### 直接对比 m31(code k=2.8 cell)

| paradigm | iter 类型 | 时长 (mean) | batch token | reqs/iter |
|---|---|---|---|---|
| m31 | prefill iter | **150ms** | 2048 | 2.16 |
| m31 | decode iter | **39ms** | ~29 | 29 |
| m31 | cycle (P+D) | **189ms** | — | — |
| c3-fair | mixed iter (chunk-cap) | **220.6ms** | 2048 | 31.5 (1-2 prefill + ~30 decode) |
| c3-fair | mixed iter (decode-dom) | 25ms | <10 | 3.7 |
| c3-fair | mixed iter (median) | **208ms** | — | — |

### Mechanism #1(batch composition)直接验证

**同样 2048 token budget,m31 prefill iter = 150ms,c3 chunk-cap mixed iter = 220.6ms → mixed-attention overhead = +70.6ms / iter(+47%)**。

物理解释:
- m31 prefill iter:2.16 prompts 全 prefill,attention 计算同质(全 prompt growing)
- c3 chunk-cap mixed iter:1-2 prompts 在 prefill chunk + ~30 reqs 在 decode 一个 token,attention 计算变长 seq lengths + 混合 partial-prefill 和 already-decoded,kernel pattern + memory access 不友好

### Cycle 对比

- m31 cycle (1 prefill iter + 1 decode iter):**189ms** 完成 2048 prefill token + ~29 decode token = 2077 token / 189ms = 11.0 tok/ms
- c3 chunk-cap iter:**220.6ms** 完成 ~2030 prefill token + ~30 decode token = 2060 token / 220.6ms = 9.3 tok/ms
- **m31 token throughput +18%**(chunk-cap regime)

### 修正 framing(对 README D-013 + 旧 memory 的 cleanup)

**旧 framing(错)**:"c3 mixed iter ~225ms"(实际是 c3 tpot p99)
**实测 framing(对)**:c3 median mixed iter = **208ms**;chunk-cap mean = **220.6ms**;p99 = 286ms。m31 cycle 189ms,Δ = +19ms(+10%)按 median 算,+31ms(+16%)按 chunk-cap mean 算。

**结论稳定**:m31 cycle 仍短于 c3 mixed iter,只是 magnitude 比之前 memory 估的略小。Phase-pure batching 优势仍来自机制 #1(mixed-attention overhead 70ms/iter)+ #2(scheduler 路径)。

### code k=4.9 也验证(高负载下同样 pattern)

- chunk-cap iter mean = 217.5ms(vs k=2.8 的 220.6,基本不变,负载只影响 iter 占比不影响单 iter 时长)
- chunk-cap 占比 65.5%(k=2.8 是 57.4%) — 负载越高 prefill 压力越大,mixed batch 越饱和

## c4_pd 4-way 补完(2026-05-25,D-011 commitment debt 还清)

**Sweep**:c4_pd × 2 wl × 7 k × 3 seed = 42 cells,outdir `results/c4_pd_supplement/`,~87 min wall。c4_pd 不读 runtime SLO,post-hoc 4 档 reclassify。聚合 + figures 4-way / 3-way 两套已生成(`paper_F1{a-f}_*.png` + `_3way.png`)。

### c4_pd 表现:全谱 dominated,但 TPOT 异常好
- **TTFT 灾难**:conv s1 全 0% meet;TTFT mean 从 low load(k=0.5)的 472ms 单调爆到 high load(k=4.9)的 63,663ms
- **TPOT 全员最好**:全 sweep TPOT mean 稳定 **~70ms**,p99 ~80ms,完爆 m31 (~170ms) / c3 (~210ms) / c1 (~2000+ ms)
- → 这个对比模式本身就是诊断 signal

### TTFT 烂的 Layer 1-4 诊断(非 bug,是 1P1D fundamental cost)

**不是 bug 的证据**:smoke 通过、TPOT 健康 → KV transfer + decode 链路 working;output throughput 在 high load 仍达 353 tok/s;ranktable.json working version 已验证(memory)。

**4 层叠加导致 TTFT 爆炸**:

1. **TP=1 vs TP=2(架构层)**:c4_pd prefill 只在 NPU0 TP=1,c1/c3/m31 prefill 用 TP=2 跨 2 NPU。7000-token long prompt prefill compute time 差 ~2×(c4 ~2.5-3s vs others ~1.5s)
2. **无法 batch long prompt(scheduler 层)**:prefill server `max_num_batched_tokens=8192`,一个 7000-token prompt 吃 86% budget → effective prefill rate ≈ 0.3-0.5 QPS sustainable on long prompts。c3 chunked / m31 chunk_budget=2048 把 long prompt 切 chunks 跟 short prompt 拼 batch,所以不撞这个上限
3. **Burst trace 放大(workload 层)**:burst_high = 5-10× base QPS,1P1D 静态分区**没法借资源**;c1/c3/m31 在 burst 期都能用满 2 NPU 做 prefill
4. **Static partitioning 资源不对称(paradigm 层)**:TPOT 70ms 全程 idle → decode NPU 整 sweep 没打满,但 c4_pd 没办法把闲置 decode 借给 prefill。c4_pd 的 decode 隔离好处被 prefill halving 代价吃掉

**TTFT 数学**:code k=2.8 arrival 7.89 QPS / n_ok per 60s = 7.67 → service rate 略低于 arrival → 队列**线性堆积**,60s 测试窗内积累 ~120 reqs backlog,TTFT mean = backlog × per-req prefill ≈ 15s。high load 下 backlog 累积更猛,TTFT 推到 60+ 秒。

**Low load 也比 m31 差 3×(base overhead 估算)**:
- conv k=0.5 c4_pd TTFT mean 472ms vs m31 146ms → **+326ms 固定 overhead**
  - ~150-250ms:TP=1 vs TP=2 prefill compute 时间差
  - ~50-150ms:KV transfer over HCCS(LLMDataDistCMgrConnector round-trip)
  - ~30-80ms:proxy 路由 + 两个 server 间 HTTP round-trip

### Paper framing(避免 reviewer strawman 攻击)

c4_pd 不是被 cherry-pick 设置打死的,**而是 1P1D + burst + single-node + long-prompt 这个 setup 数学上不可能赢**。Paper Section 5 caveat 须 explicit:

1. **Setup 限制**:1P1D 是 2-NPU 上唯一的 disagg config;DistServe/Splitwise 原 paper 在多 NPU(2P2D/4P2D)+ steady arrival 下成立
2. **Workload 限制**:burst trace 是 disagg static partitioning 的最坏场景
3. **Hardware 限制**:KV-over-HCCS 在 m31/c1/c3 路径上完全不存在,c4_pd 多 ~50-150ms 固定 overhead
4. **Positive framing**:c4_pd 反而验证 m31 的核心 claim — phase-pure 在同样 2-NPU 上拿到了 disagg 的部分 decode 隔离好处(TPOT 170ms 比 c1/c3 的 ~210ms 好,虽不及 c4 的 70ms),又没付 prefill halving 的代价

### Paradigm-level 4-way 结论(更新)

| paradigm | conv winning regime | code winning regime | TPOT 优势 | TTFT 优势 |
|---|---|---|---|---|
| c1 (hybrid) | none(strict/mid 部分 win c3) | none | × | × |
| c3-fair (chunked) | s3 loose | s2/s3 mid+loose tied with m31 | × | × |
| **m31 (PD-TDM)** | **全谱 dominant** | **strict s1 dominant,s2/s3 tied 或 leading** | **vs c1/c3 ✓** | **全员 ✓** |
| c4_pd (1P1D disagg) | dominated everywhere | dominated everywhere | **vs c1/c3/m31 ✓**(but TTFT 烂死) | ×

## c4_pd steady (Poisson) vs burst — 1P1D 真凶是 capacity,不是 burst(2026-05-25)

**Hypothesis 入场**:burst trace 是 disagg static partitioning 的最坏场景,steady 下 c4_pd 应该好得多。
**实测推翻**:steady 在所有 load 下仍灾难,burst 只占 ~10-20% 额外 overhead。

### 3 cell smoke 对比(`results/c4_pd_steady_smoke/`,3 conv cells matched 到 burst 平均 QPS)

| Load level | m31 TTFT | c4_pd burst TTFT | c4_pd steady TTFT | burst/steady | steady vs m31 |
|---|---|---|---|---|---|
| QPS 2.85 (low) | 146 | 472 | **432** | 1.1× | **3.0× 慢** |
| QPS 5.70 (mid) | 191 | 1118 | **916** | 1.2× | **4.8× 慢** |
| QPS 7.98 (high) | 191 | 16266 | **17423** | 0.9× | **91× 慢** |

(high load QPS 7.98 steady **略差于** burst — burst 完全不是放大器)

### Framing 修正(D-013 caveat update — 不再用 "burst 是放大器")

**之前 framing(基于 burst-only data 推测)**:
> "burst trace 是 disagg static partitioning 的最坏场景;steady arrival 下差距会显著缩小"

**实测推翻,实际故事**:
1. **Base overhead(KV transfer + proxy + TP=1 vs TP=2 prefill)给 c4_pd 加约 300ms TTFT,无论 load 如何** — low load 2.85 QPS 下 c4_pd 仍 3× 慢于 m31
2. **TP=1 prefill capacity 在 ~QPS 5-6 撞顶** — 不是 burst 触发的瞬时 spike,steady QPS 5.7 已经开始队列堆积(TTFT 916ms)
3. **Steady QPS 7.98 直接 TTFT 17 秒** → service rate < arrival rate,无 burst 也线性堆积
4. **Burst 只贡献 0-20% TTFT overhead** — 远不是首要因子

### Paper Section 5 caveat 重写(替换之前的 "burst 是最坏场景" framing)

**新 caveat**(更 honest,且对 paper 更有利 — 关上"setup 是 strawman"攻击):
- 1P1D 在 2-NPU 上 prefill capacity 只能用 TP=1,fundamental capacity 约 5-6 QPS(long-prompt code workload)
- KV-over-HCCS + proxy round-trip 加约 300ms 固定 TTFT overhead vs unified/multiplex paradigm
- → **c4_pd 在 2-NPU single-node 上数学层面就不可能赢 m31**,无论 trace 是 burst 还是 steady
- → DistServe / Splitwise 原 paper 在 ≥4 NPU(2P2D / 4P2D)+ multi-node + RoCE 下成立,**2-NPU 不在它们的应用域**
- m31 反而是 **2-NPU regime 的合适 paradigm** — 既拿到了 disagg 的部分 decode 隔离好处(TPOT 比 c1/c3 强),又没付 prefill 减半 + KV transfer overhead

### Positive 重新 framing(给 paper 主图 narrative)

**old**:m31 击败 c4_pd 因为 c4_pd 不适合 burst → reviewer 可能反问 "那 steady 下呢"
**new**:m31 击败 c4_pd 因为 2-NPU regime 下 1P1D 数学 capacity 就不够 → reviewer 没得攻击

### TPOT 观察(supporting 数据点)

c4_pd TPOT 在 burst 和 steady 下都稳定在 73-79ms(全员最好),证明:
- decode NPU 在任何 c4_pd load 下都没打满
- KV transfer 链路完全 working(不是 setup bug)
- **资源浪费的根源就是 TP=1 prefill / TP=1 decode 的不对称分配 — 任何分配比例固定的 disagg 都吃这个亏**

## 已废弃 / 清理(5/25)

- `results/phase_2_t6_burst_goodput/*_pid/`(168 个 m31-bug cells)已删除
- `results/azure_m31_fia_ablation/`(kernel ablation 带 bug)已删除
- 中间 long-prompt 探索数据已删除
- 旧 post-hoc 脚本 `posthoc_phase_2_t6{,_fair}.py` 已删除

## 跟进事项

详 `DECISIONS.md` D-013 「跟进事项」:
1. 修 phase_iters semantic bug
2. m31@chunk_budget ∈ {2048, 4096, 8192} ablation
3. 跨 workload telemetry verify
4. 决定 Phase 2 优先级
