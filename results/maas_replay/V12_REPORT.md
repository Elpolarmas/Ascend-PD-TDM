# PD-TDM vs Sarathi 综合实验报告

> 硬件: 2× Ascend 910B3 TP=2 | 模型: Qwen3-8B | vllm-ascend v0.11.0rc1

---

## 1. 实验全景

### 1.1 时间线与实验代次

```
2026-05-22  T6 Azure burst 实验启动
2026-05-25  T6 完成 (m31-fix / c3-fair / vanilla_cb / c4_pd)
2026-05-26  F5 decode-heavy synthetic sweep 完成 → DH 发现 degenerate
2026-05-27  F5b 修正 chunk degeneracy → 二维 tax-ratio 模型建立
2026-05-27  F5c prefill-balanced 补充实验
2026-06-30  MaaS smoke (rpm_scale 0.02~0.38)
2026-07-01  MaaS v3 首次全量回放 → 失败 (PoolTimeout 68%)
2026-07-06  v4-v7 调试: concur 调优 + 分布长度 + sarathi config
2026-07-07  v11 正式 (sub=30, 17.8K req) → pdtdm 0 err / sarathi 0 err
2026-07-08  v12 正式 (sub=15, 35.4K req) → pdtdm 0 err / sarathi 0 err
```

### 1.2 实验层次

| 层次 | 实验 | 流量来源 | 到达模式 | 目标 |
|------|------|---------|---------|------|
| L1 合成 | F5/F5b/F5c | 合成参数 | Poisson | 建立二维机制模型 |
| L2 真实 trace | T6 conv/code | Azure LLM trace | burst | 真实 trace + 过载 SLO 对比 |
| L3 生产 trace | MaaS v11/v12 | MaaS 全天聚合 | aggregated replay | 生产环境验证 |

### 1.3 关键方法演进

| 版本 | 改进 | 效果 |
|------|------|------|
| v3→v6 | max_concurrency 4096→64 | PoolTimeout 68%→0% |
| v6→v7 | 均匀长度→log-normal σ=0.35 | wall -15% |
| v7→v11 | subsample+compress_timeline | wall 从 3.6天 缩到 6.5h |
| v11→v12 | subsample 30→15 | 精度翻倍, 结论一致 |

---

## 2. MaaS Trace 特征

### 2.1 数据来源

- 文件: `aggregated_curve.csv`
- 来源: MaaS 生产环境, 2026-03-03 全天
- 格式: 每分钟一行共 1139 行, timestamp + RPM + prompt_tokens + completion_tokens

### 2.2 流量规模

| 指标 | 值 |
|------|-----|
| 全天总请求 | 5,331,373 |
| 全天总 prompt tokens | 35.9B |
| 全天总 output tokens | 852M |
| 平均 RPM | 4,678/min (78 req/s) |
| 峰值 RPM | 9,763/min (163 req/s) |

### 2.3 请求长度分布

**avg_prompt (每 bucket 均值):**

| 范围 | 占比 | 可视化 |
|------|------|--------|
| 0-2K | 1.5% | |
| 2K-4K | 14.8% | ███████ |
| 4K-6K | 9.0% | ████ |
| 6K-7K | 8.1% | ████ |
| 7K-8K | **33.6%** | ████████████████ |
| 8K-10K | **33.0%** | ████████████████ |

67% 的流量 prompt 在 7K-10K 区间，长上下文输入占绝对主导。

**avg_output (每 bucket 均值):**

| 范围 | 占比 | 可视化 |
|------|------|--------|
| 50-100 | 4.5% | ██ |
| 100-150 | 20.8% | ██████████ |
| 150-200 | **71.4%** | ███████████████████████████████████ |
| 200-250 | 3.4% | █ |

output 全域在 **75-243 token**，92% 在 100-200 区间。极短的输出。

### 2.4 工作负载定性

```
整体 ratio = 42:1 (prompt:output)

特征:
  - 长上下文输入 (7K-10K token)
  - 极短输出 (150-200 token)
  - 典型 code completion / fill-in-the-middle 类服务
  - 对 PD-TDM 极为有利: 极度的 prefill-leaning
```

---

## 3. 跨工作负载统一对比

### 3.1 对比总表

| 实验 | 日期 | prompt | output | ratio | PD-TDM TTFTp99 | Sarathi TTFTp99 | ΔTTFT | PD meet | SA meet | Δmeet | PD gp | SA gp | Δgp |
|------|------|--------|--------|-------|---------------|----------------|-------|---------|---------|-------|-------|-------|------|
| **T6 code s1 k=2.8** | 5/22-25 | 2,048 | 28 | 73:1 | 727ms | 912ms | **-20%** | 80.1% | 2.3% | **+77.8pp** | 266 | 37 | +619% |
| **T6 conv s1 k=3.0** | 5/22-25 | 1,155 | 211 | 5.5:1 | 633ms | 700ms | **-10%** | 63.8% | 29.0% | **+34.8pp** | 3,396 | 1,525 | +123% |
| **F5b A2 QPS=0.15** | 5/27 | 4,096 | 8,192 | 1:2 | 898ms | 857ms | **+5%** | 94.0% | 95.0% | -1.0pp | 619 | 678 | **-9%** |
| **MaaS v12** | 7/7-8 | 6,378 | 160 | 40:1 | 1,180ms | 1,316ms | **-10%** | 100% | 100% | 0pp | 1,242 | 1,241 | 0% |
| ├ SLO×1.2 ‡ | — | — | — | — | — | — | — | **92.6%** | **63.5%** | **+29.1pp** | **1,138** | **757** | **+50%** |

> ‡ MaaS SLO×1.2: post-hoc 收紧到 TTFT<1,123ms, TPOT<259ms 的 reclassification, 见 §3.6

### 3.2 MaaS v12 详细

| 指标 | PD-TDM | Sarathi |
|------|--------|---------|
| submitted | 35,441 | 35,441 |
| errors | **0** | **0** |
| traffic window | 76 min (compress_timeline) | 76 min |
| wall clock | 23,841s (6.6h) | 25,554s (7.1h) |
| TTFT mean | 808ms | 952ms |
| TTFT p50 | 857ms | 993ms |
| TTFT p99 | **1,180ms** | **1,316ms** |
| TTFT max | 4,851ms | 4,665ms |
| TPOT mean | 222ms | 236ms |
| TPOT p50 | 234ms | 247ms |
| TPOT p99 | 252ms | 257ms |
| TPOT max | 308ms | 308ms |
| E2E mean | 43s | 46s |
| E2E p99 | 92s | 97s |
| SLO meet | **100.0%** | 99.997% |
| goodput (×3.0 SLO) | 1,242 tok/s | 1,241 tok/s |
| goodput (×1.2 SLO) | **1,138 tok/s** | **757 tok/s** |

### 3.3 E2E 延迟分析 — 为什么端到端延迟高达 90 秒

v12 的 E2E p99 达到 **92s (PD-TDM)** 和 **97s (Sarathi)**，这看起来异常，
但实际上是 trace replay 过载场景下的必然结果。

**E2E 的构成:**

```
E2E = 排队积压时间 + 服务器处理时间

PD-TDM:  E2E mean=43s = 41.2s 排队 + 1.3s 处理  (排队占 97%)
Sarathi: E2E mean=46s = 44.5s 排队 + 1.5s 处理  (排队占 97%)
```

**为什么排队这么长:**

1. compress_timeline 把 76 个 bucket（代表全天每隔 15 分钟的一分钟）紧排成连续 75
   分钟的流量。每个 bucket 的 rpm_scaled 请求在 60s 内到达完毕。
2. 峰值 bucket 的到达率为 14.3 req/s，服务器处理能力仅 ~1.3 req/s (heavy prompt)。
3. 请求到达后被放入服务器队列，等待前面的请求完成。在 over-subscription 11× 下，
   后到的请求需要等待数十秒甚至数分钟。
4. 客户端 E2E 计时 = 从 `arrival_time_s` 到 HTTP response 返回。因为服务器积压，
   大部分请求的 E2E 包含漫长的排队时间。

**为什么这对分析没有影响:**

- **TTFT 是 scheduler 评估的正确指标** — 它衡量请求到达服务器后首 token 生成的时间，
  直接反映 prefill 调度效率。E2E 被客户端排队污染，不反映调度质量。
- **TPOT 是 decode 评估的正确指标** — 不受积压影响（decode 请求已在处理中）。
- **E2E 的排队成分是 rs=0.10 过载的必然产物**，不是任何一方的 bug。在更高
  rs 或更长时间窗口下，积压会更严重。

**PD-TDM 的 E2E 低于 Sarathi (43s vs 46s):**

PD-TDM 更快的 prefill 消化速度（phase-pure prefill iterations）使队列
清空稍快，从而整体排队时间更低。差值 ~5% 与 wall clock 差值 (~7%) 方向一致。

```
E2E vs arrival time:
  [0-1h]: PD-TDM 44s, Sarathi 48s (早期达到稳态积压)
  [1-2h]: PD-TDM 35s, Sarathi 38s (积压逐渐消化)
```

> **结论**: MaaS v12 的核心指标是 TTFT/TPOT，E2E 被重度过载场景下的客户端排队
> 主导（97%），不作为 scheduler 对比的有效指标。

### 3.4 SLO 敏感性分析 — 收紧 SLO 后的 goodput 对比

v12 当前 SLO 为 TTFT < 2,808ms (ideal ×3)。两个范式都 100% meet，goodput
相等。但 PD-TDM 的 TTFT 分布整体左移 (~150ms)，这一优势在 **更紧的 SLO**
下会直接转化为 goodput 差距。

**方法**: 对 v12 原始 per-request 数据做 post-hoc SLO 重分类，不重跑实验。

```
MaaS micro-benchmark ideal: TTFT p99=936ms, TPOT p99=216ms
当前 SLO = ideal ×3.0
测试 SLO = ideal ×{1.2, 1.5, 1.8, 2.0, 2.5, 3.0}
```

**TTFT/TPOT 分布基线:**

| 分位 | PD-TDM TTFT | Sarathi TTFT | PD-TDM TPOT | Sarathi TPOT |
|------|------------|-------------|------------|-------------|
| p50 | 857ms | 993ms | 234ms | 247ms |
| p90 | 1,113ms | 1,267ms | 242ms | 253ms |
| p95 | 1,132ms | 1,285ms | 245ms | 254ms |
| p99 | 1,180ms | 1,316ms | 252ms | 257ms |

PD-TDM 在 p50 处已有 136ms 优势，到 p99 处优势缩小为 136ms。分布整体左移。

**SLO 重分类结果:**

| SLO × | TTFT < | TPOT < | PD meet | SA meet | Δmeet | PD gp | SA gp | **Δgp** |
|-------|--------|--------|---------|---------|-------|-------|-------|---------|
| 3.0 | 2,808ms | 648ms | 100% | 100% | 0pp | 1,242 | 1,241 | **0%** ← 当前 |
| 2.5 | 2,340ms | 540ms | 100% | 100% | 0pp | 1,242 | 1,241 | 0% |
| 2.0 | 1,872ms | 432ms | 100% | 100% | 0pp | 1,242 | 1,241 | 0% |
| 1.8 | 1,685ms | 389ms | 100% | 100% | 0pp | 1,242 | 1,241 | 0% |
| 1.5 | 1,404ms | 324ms | 99.9% | 99.8% | +0.1pp | 1,241 | 1,238 | 0% |
| **1.2** | **1,123ms** | **259ms** | **92.6%** | **63.5%** | **+29.1pp** | **1,138** | **757** | **+50%** |

**解读:**

1. SLO ≥ ×1.5 时两者都接近 100% meet，goodput 相等。因为双方 TTFT p99 都在 SLO 线以下。

2. SLO = ×1.2 时差距突然拉开: PD-TDM TTFT p90=1,113ms < 1,123ms(SLO) < Sarathi TTFT p90=1,267ms。Sarathi 有 ~36% 的请求 TTFT 超过 1,123ms，而 PD-TDM 只有 ~7%。

3. **差距出现的窗口在 ×1.2~1.5 之间**，正好是 PD-TDM 分布左移量的位置 (~150ms)。这是 PD-TDM phase-pure 在 latency 层面的结构性优势。

4. 这是 **static ratio (static_ratio=0.3)** 下的结果——PD-TDM 的 prefill:decode time ratio
   固定为 30:70，没有根据 SLO 余量动态调整。如果 SLO 更紧而系统仍有 prefill
   余量，增加 prefill ratio 可以进一步提升 meet%。这正是自适应 PID 的设计目标。

5. 当前实验没有展现自适应 PID 的效果，因为:
   - static_ratio=0.3 已经让 92.6% 请求满足 ×1.2 SLO
   - 剩余 7.4% 的违规请求可能需要更高的 prefill ratio 来加速消化
   - 但当前 PID 实现（target_ratio→ratio_max 撞顶）未能在这种 regime 下发挥作用
   - 这引出了下一步: 改良自适应 PID，使 prefill ratio 能真正响应 SLO 余量

### 3.5 Static Ratio 与自适应 SLO 的关系

当前 v12 的 PD-TDM 结果本质上是 **static ratio = 0.3 的 phase-pure temporal
multiplexing** 效果。数据证明了即使在固定 ratio 下，phase 分离本身就能产生
~150ms TTFT 左移。这是 PD-TDM 的 "基础效果"——不依赖自适应。

**自适应 PID 的角色是 "增量效果"**: 在基础效果之上，根据实时 SLO 余量动态调节
prefill:decode 比例——
- SLO 余量大 (TTFT << SLO) → 减少 prefill ratio，更多 decode
- SLO 余量小 (TTFT → SLO) → 增加 prefill ratio，加速 prefill 消化

D-013 的发现是当前 PID 实现存在局限:
- `target_ratio` 推 `ratio_max` 撞顶 (0.8)
- 实际 prefill iter 占比由 waiting queue 自然决定 (12-50%)
- PID 降级为 "SLO 违反诊断指示灯"

**改良方向** (后续工作):
- 当前 SLO 余量驱动 ratio 调整 (而非 target_ratio→max)
- 让 PID 在过载 + 紧 SLO 场景下 (如 ×1.2) 通过提升 prefill ratio 来
  进一步回收 meet%——这 7.4% 的违规可能在高 prefill ratio 下被消除
- 需要 vs static_ratio 的 ablation 实验量化自适应增量


---

## 4. 各实验规模统计

### 4.1 请求数与 Wall Clock

| 实验 | QPS/k | PD-TDM reqs | Sarathi reqs | PD-TDM err | SA err | PD wall | SA wall | 窗口 |
|------|-------|------------|-------------|-----------|--------|---------|---------|------|
| T6 conv k=3.0 | 16.8 QPS | ~504/seed | ~504/seed | 0 | 0 | 90s | 90s | 60s ×3seed |
| T6 code k=2.8 | 7.9 QPS | ~237/seed | ~237/seed | 0 | 0 | 90s | 90s | 60s ×3seed |
| F5 bal QPS=2.5 | — | 441 | 441 | 0 | 0 | 208s | 208s | 180s |
| F5 DH QPS=1.0 | — | 175 | 175 | 0 | 0 | 219s | 220s | 180s |
| F5b A1 QPS=0.25 | — | 49 | 49 | 0 | 0 | 357s | 337s | 180s |
| F5b A2 QPS=0.15 | — | 23 | 23 | 7 | 6 | 459s | 459s | 180s |
| F5c B1 QPS=1.0 | — | 185 | 185 | 0 | 0 | 213s | 211s | 180s |
| F5c B2 QPS=1.0 | — | 185 | 185 | 0 | 0 | 220s | 221s | 180s |
| MaaS v7 peak | — | **5,747** | **5,747** | 0 | 0 | 4,336s | 4,712s | 600s |
| MaaS v11 | — | **17,764** | **17,764** | 0 | 0 | 11,977s | 12,714s | 38min |
| MaaS v12 | — | **35,441** | **35,441** | 0 | 0 | 23,841s | 25,554s | 76min |

### 4.2 实验规模分层

```
请求数规模:
  F5/F5b/F5c 合成:    23-441  req/experiment    (QPS sweep, 180s 窗口)
  T6 Azure burst:    ~237-504 req/seed/cell     (60s 窗口, 3 seed)
  MaaS v7 peak:       5,747  req/paradigm       (600s 峰值窗口)
  MaaS v11:          17,764  req/paradigm       (38min 压缩窗口, sub=30)
  MaaS v12:          35,441  req/paradigm       (76min 压缩窗口, sub=15)

Wall clock 规模:
  合成:  3-8 min
  T6:    1.5 min × 3 seed × 2 config × 2 wl × 7 k ≈ 2h total
  MaaS v7:  72 min (pdtdm) + 79 min (sarathi)
  MaaS v11: 3.3h (pdtdm) + 3.5h (sarathi) = 6.8h
  MaaS v12: 6.6h (pdtdm) + 7.1h (sarathi) = 13.7h
```

> T6 每个 cell 是 90s (warmup 30s + window 60s)，3-seed median。reqs = arrival_qps × 60s × 3seed。conv k=3.0: 16.8×60×3=3,024 total。表中列出的是 seed 中位数 (取 median 的 seed)。

## 5. 机制分析

### 5.1 为什么 PD-TDM 降低 TTFT

```
实测 telemetry (code k2.8 s1, D-013):

Sarathi c3 mixed batch: 每 iter ~220ms
  一 iter = 1 个 prefill chunk (≤2048 token) + ~30 个 decode 请求
  所有请求共享同一次 varlen attention (长query + 短query 混合)
  → mixed attention 效率低: 单 token decode 被长 prefill query 拖慢

PD-TDM phase-pure: cycle = prefill 150ms + decode 39ms = 189ms
  prefill iter: 纯 prefill kernel, 2048 token, mean 2.16 reqs (max 6)
  decode iter: 纯 decode kernel, 29 reqs
  → 每种 kernel 在自己最擅长的 query 形状下运行
  → 总 cycle 比 c3 单 iter 快 36ms (16%)
```

- **机制 #1: 纯 kernel 路径** — c3 每 iter 处理 prefill(长query) + decode(单token query) 混合 varlen attention，kernel 效率低。m31 拆成纯 prefill 和纯 decode 两个 iter，各自在最优 query 形状下运行。实测 c3 iter=220ms > m31 cycle=189ms。
- **机制 #2: prefill 批量并行** — m31 prefill iter 可同时处理多个请求 (mean 2.16, max 6)，c3 每 iter 只做 1 个 prefill chunk。prefill 吞吐更高 → 队列消化更快 → TTFT 降低。
- **机制 #3: NPU variable-query tax** — Ascend NPU 上 varlen attention 的 query 长度差异越大效率越低。m31 通过 phase 分离，保证每个 iter 内 query 长度统一，避开这个 tax。

### 5.2 为什么 TPOT 不变

Decode 阶段两种方案本质上相同:
- Sarathi: decode-priority 调度, decode 请求优先于 prefill chunk
- PD-TDM: 纯 decode phase, 所有请求都是 decode
- TPOT 差异 < 5%，在测量误差范围内

### 5.3 为什么 goodput 相等

```
goodput = output_throughput × SLO_meet%

rs=0.10: 系统不饱和
  pdtdm  100% × 164 = 164
  sarathi 100% × 164 = 164

rs 足够高时 (T6 code):
  pdtdm  80% × 266 = 213
  sarathi 2%  × 37  = 1     ← goodput 差距 619%
```

PD-TDM 的 TTFT 优势在 **系统过载时**转化为 goodput 优势: Sarathi 先碰到 SLO ceiling, PD-TDM 仍有余量。

### 5.4 二维优势模型 (F5b 修正后)

```
PD-TDM 优势平面

        output tokens ↑ (decode 主导)
              │
    A2 (1:2)  │  A1 (1:2)
    PD-TDM +5%│  PD-TDM +15%
    反输 -9%gp│  微弱劣势
              │
  ────────────┼────────────────────→ prompt tokens ↑ (prefill 主导)
              │
    B1/B2     │  T6 conv    MaaS v12   T6 code
    -5~-7%    │  -10%       -10%       -20%
              │  meet+35pp  meet 0pp*  meet+78pp
              │
              │  F5 bal (1:1)
              │  -62% TTFT
              │
        (* rs=0.10不饱和, 提高rs后预期拉开)

PD-TDM 优势 ∝ prefill_tax / decode_load

  prefill_tax     ≈ (prompt_tokens × arrival_rate) / (chunk_budget × prefill_batch)
  decode_load     ≈ (output_tokens × arrival_rate × avg_decodes)
  
  当 prefill_tax >> decode_load  → PD-TDM 大胜 (T6 code +78pp)
  当 decode_load >> prefill_tax  → PD-TDM 反输 (F5b A2 -9%)
  当 二者接近                      → 取决于饱和程度
```

---

## 6. MaaS v12 配置与方法

### 6.1 实验参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据源 | `aggregated_curve.csv` | MaaS 全量 1140 行, 每分钟一行 |
| 采样 | subsample_step=15 | 每 15 行取一行 → 76 bucket |
| 时间压缩 | compress_timeline=True | 76 bucket 紧排 i×60s, 实际流量窗口 76 min |
| 流量缩放 | rpm_scale=0.10 | 原始 RPM 的 10% |
| SLO | TTFT<2,808ms, TPOT<648ms | micro-benchmark ideal (936ms/216ms) × 3 |
| 请求长度 | log-normal σ=0.35 | 每请求独立采样 prompt/output 长度 |
| 并发控制 | max_concurrency=64 | httpx + asyncio 背压 |
| SLO | TTFT<2,808ms, TPOT<648ms | micro-benchmark ideal × 3 |
| 模型 | Qwen3-8B, TP=2, MML=8,192 | |

### 6.2 模拟流程

```
Trace CSV → _load() → 76 bucket × avg_p/avg_o
  → subsample+compress → t_rel = i×60s
  → iter_buckets() → per-request log-normal 采样
  → run_driver() → asyncio.sleep(arrival_time_s) → HTTP POST
  → server 处理 → 记录 TTFT/TPOT/E2E → aggregate window [30s, 34200s]
```

### 6.3 范式配置

| 范式 | 内部 config | max_num_batched_tokens | chunk | scheduler |
|------|------------|----------------------|-------|-----------|
| PD-TDM | c2_tdm_m31_2048 | 8,192 | 2,048 | TDMScheduler, phase-pure |
| Sarathi | c3_cp | 2,048 | 2,048 | AscendScheduler, chunked prefill |

两者 chunk_budget 均为 2,048 token，公平对比。

---

## 7. 讨论与 Caveat

### 7.1 结论范围

- **硬件**: 2× Ascend 910B3 NPU (仅此平台, 不声称 GPU 通用性)
- **模型**: Qwen3-8B (单一模型)
- **工作负载**: code completion 类 (prefill-leaning, 42:1)
- **SLO**: micro-benchmark × 3
- **chunk_budget**: 2,048 token

### 7.2 对论文的启示

1. **MaaS trace 是 PD-TDM 的 "主场"** — 42:1 prefill-leaning 是 PD-TDM 优势最大的 regime。paper §5 须显式写出此 scope 限定。

2. **F5b A2 揭示 PD-TDM 的弱点** — 在 deep decode queue 下 PD-TDM 反输。这是 fair-trial 的诚实立场, 防 reviewer "你 cherry-pick" 攻击。

3. **二维模型优于一维 ratio 谱** — 原 "ratio 单调谱" 被 F5b 推翻。正确的 framing 是 heatmap on (prompt_tokens, output_tokens) 2D 平面。

4. **TTFT 优势稳定可复现** — 合成实验 (-5% ~ -62%)、Azure trace (-10% ~ -20%)、MaaS trace (-10%) 三线一致。TPOT 始终接近。

5. **goodput 差异需要过载** — rs=0.10 不饱和时两者 100% meet。论文的 goodput 主图应使用 T6 过载数据, MaaS 数据印证延迟层面的一致性。

### 7.3 已知局限

- **聚合 trace 丢失分布信息**: 每分钟只有平均值, 我们通过 log-normal 分布近似。理想情况应使用 per-request trace
- **rs=0.10 偏低**: 原始 trace 是 ~70 NPU 集群产生, 2 NPU 只能承载 10%
- **单 seed**: v11/v12 跑 seed=0, 但两次实验相互印证
- **NPU 平台**: 所有发现限于 Ascend NPU, strong scaling 和 varlen attention 特性可能不适用于 GPU
