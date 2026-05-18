# Chunking 机制(M3 prefill chunking)

> 已实装(M3.1)。**不做** dynamic controller(D-006),静态 `chunk_tokens=2048`。

---

## 1. 关键 insight

vllm-ascend v0.11.0rc1 上**保留 V0 快 kernel + 实现 prefill chunking 是兼容的**:

| 路径 | attn_state | kernel | 速度 |
|---|---|---|---|
| C1 mixed P+D | `PrefillCacheHit` | `_npu_flash_attention_qlens` (V0) | 快 |
| TDM phase-pure P 新 req | `PrefillNoCache` | `_forward_prefill_no_cache` (V0) | 快 |
| **M3 chunked prefill 续 iter** | `PrefillCacheHit` | `_npu_flash_attention_qlens` (V0) | **快** |
| TDM phase-pure D | `DecodeOnly` | `_forward_decode_only` (V0) | 快 |
| **C3 mixed (CP=True)** | `ChunkedPrefill` | `npu_fused_infer_attention_score` (FIA V1) | **慢 2.7-3.4×** |

→ **关键:只要不开 `chunked_prefill_enabled` 启动配置**(保持 V0 路径),**手动切碎的 prefill 续 iter 会落到 PrefillCacheHit (qlens),仍走 V0 快路径**。这是 M3 跟 C3 的本质区别。

---

## 2. chunk 语义(已拍板)

- 切的对象:iter 内**总** prefill token cap,不限单 req
- 跨 iter 续:复用 vLLM V1 chunked-prefill 状态机(`request.num_computed_tokens` 单调累加),但走 AscendScheduler 路径(保 V0 快 kernel)
- phase-pure 严格保证:单 iter 要么 pure-P 要么 pure-D,**不混 D token**
- 静态值:`chunk_tokens=2048`(sweet spot,p1_chunk_scan 实证)

---

## 3. 实装

### 3.1 文件布局

| 文件 | 内容 |
|---|---|
| `tdm/chunking.py` | 91 行 ChunkPlanner(resume pass + fresh pass + phase-pure filter)+ 6 单测 |
| `tdm/chunked_schedule.py` | ~330 行,fork 自 `AscendScheduler.schedule()`,5 处 Diff |
| `tdm/scheduler.py` | 路由按 `prefill_chunk_tokens is not None` 分流 |
| `tdm/config.py` | `prefill_chunk_tokens: Optional[int] = None` |

### 3.2 5 处 Diff(vs parent.schedule())

| # | Diff | 说明 |
|---|---|---|
| 1 | Running pre-pass | 扫 `self.running` 找 partial-prefill req,via `plan_chunks()`,gated `phase=="prefill"` |
| 2 | chunk_budget tracked | 跟 token_budget 并列追踪 |
| 3 | skip → truncate | parent line 204 `skip_cur_request()` → `num_new_tokens = min(num_new_tokens, chunk_budget, token_budget)` |
| 4 | Shared num_computed advance | iter 末单点 advance(共享 resume 与 waiting 两 pass) |
| 5 | Decode-loop partial-prefill filter | `if num_computed < num_prompt: continue`——没有这个 filter,phase=decode 时 parent assert 会爆 |

### 3.3 单测

`tests/test_chunking.py`:

| 测试 | 设置 | 期望 |
|---|---|---|
| chunked_prefill_no_cache | 1 个 req prompt=2048, chunk=512 | 4 个连续 iter,每 iter 512 tokens;第 5 iter 进 decode |
| chunked_prefill_multiple_reqs | 2 个 req prompts=[2048,1024], chunk=1024 | 3 iter 完成全部 prefill |
| chunk_off_back_compat | chunk=None | 行为与 M2.7 完全一致(regression guard) |
| budget_smaller_than_chunk | chunk=2048, max_num_batched_tokens=1024 | 实际生效 = min(2048, 1024) = 1024 |
| phase_pure_invariant | chunk=512 + decode 队列非空 | P iter 不混 D token,D iter 不混 P token |

---

## 4. 实测结果(P0-3 / p1_chunk_scan)

### 4.1 mixed regime,qps=16(same-sweep 4-way)

`results/m31_samesweep_validation/`(已删,数据归档):

| Config | strict | ttft<500/tpot<100 | ttft<500/tpot<150 | goodput |
|---|---|---|---|---|
| c2_tdm_m27 | 11.8% | 79.4% | 79.5% | 427 tok/s |
| c3_cp | 12.2% | 82.4% | 82.9% | 419 tok/s |
| **c2_tdm_m31_2048** ⭐ | **15.2%** | **96.2%** | **97.6%** | **525 tok/s** |
| c2_tdm_m31_4096 | 13.3% | 95.5% | 96.2% | 439 tok/s |

### 4.2 chunk size pareto(p1_chunk_scan)

`{512, 1024, 2048, 4096, 8192}` strict meet_slo% 单调:10.2 / 11.8 / 15.2 / 13.3 / 11.0%。
- **chunk=2048 sweet spot**
- chunk=512 太激进(4000-token prompt 切 8 块,ttft_p99 飙到 3078ms,反输 4-5×)
- chunk=2048 active chunking + ttft 代价小 + tpot 改善
- chunk=4096 弱 chunking(max prompt 4000 但 chunk_budget iter-wide 仍能截短 prompt)

### 4.3 5-way Ablation 干净因果分解(`m31_5way_ablation/`,已删归档)

通过 `c2_tdm_m31_disabled`(chunk_tokens=100000,chunk_budget 永不绑紧)隔离 fork-path 效应:

| Tier | TDM (m27-c1) | fork-path (m31_dis-m27) | **chunking** (m31_2048-m31_dis) |
|---|---|---|---|
| strict | -1.8 | +1.4 | -0.8 |
| ttft<500/tpot<100 | -1.0 | -1.0 | **+13.4** |
| ttft<500/tpot<150 | -1.0 | -1.0 | **+14.4** |
| ttft<500/tpot<200 | -0.6 | -1.1 | **+14.7** |

**三个结论:**
1. fork-path 效应 ≈ 0(disabled 全谱在 M2.7 ±1.4pp 内)
2. 纯 chunking 效应 = +13-15pp 在 ttft<500/tpot∈[100,200]
3. M2.7 controller 在 mixed regime 反输 C1(-1.0~-1.8pp)——**chunking 才是主角,不是 TDM controller**

### 4.4 multi-seed 收口(3 seeds × mixed)

`ttft<500/tpot∈[100,200]`:

| Δ | mean ± std | σ above 0 |
|---|---|---|
| chunking (m31_2048 - m31_disabled) | +10.2~10.8 ± 3.0-3.5 | ~3.4σ |
| fork-path | -1.7 ± 3.8 | ≈ 0 |
| TDM controller (m27 - c1) | -0.3 ± 1.9 | 0(mixed 无效应) |
| **M3.1 vs C1** | +8.2~8.9 ± 2.4-2.9 | ~3.4σ |
| **M3.1 vs C3** | +13.1~13.9 ± 1.5-2.1 | **~9σ 铁稳** |

---

## 5. 论文叙事(D-006 修正版)

> On Ascend NPU at qps=16, M3.1 prefill chunking (chunk_tokens=2048) reliably improves meet_slo% by **+10pp (mixed, 3 seeds) ~ +18pp (long, 1 seed) over the M2.7 baseline within the workload-conditional SLO band ttft<500ms × tpot∈[100,200]ms**, simultaneously beating C1 hybrid by +8-16pp and C3 by +13-23pp.
>
> Ablation: gain is fully chunking (m31_2048 − m31_disabled = +13-15pp); fork-code-path effect ≈ 0. **TDM controller alone (M2.7) does not significantly improve over C1 hybrid in either regime** — chunking on V0 dedicated kernels is the protagonist.
>
> chunking on V0 path avoids both M2.7's "long-prompt-blocks-decode" tail-latency mode and C3's FIA-on-NPU 2.7-3.4× kernel penalty. The SLO band itself is part of the contribution: at strict tpot<50ms or at loose ttft>1000ms, meaningful comparisons collapse.

---

## 6. 跟其他模块的耦合

- **Sampling 模块** → 决定 chunk 候选范围(参考 trace 中 prompt 长度分布)。**chunk_tokens=2048 是当前 sweet spot**
- **SLO 控制器**(慢回路)→ 不再调 chunk_tokens(D-006);留 `prefill_chunk_tokens` 配置静态注入
- **Graph-aware 模块**(已降级)→ 原计划 chunk 值优先选与 capture buckets 对齐的(`{128, 256, 512}`),实测 capture 数量非单调,放弃

---

## 7. 实测修正(D-006)

| 原设计 | 实测修正 |
|---|---|
| M3.4 SLOReactiveController 加 chunk_tokens 输出维度(运行时调) | **不做**。chunk pareto 单调,无 dynamic 设计空间(p1_chunk_scan) |
| M3.5 滑窗长度扫参 | **不做**。慢回路已有 update_interval=8,够 |
| M3.6 完整 vs M2.7 vs C1 vs C3 两 regime 对比 | 已做(`azure_main` + `long_5way_q1632`) |
| M3 chunking 主要面向 decode graph 对齐 | 实际 V0 路径不依赖 graph,chunk=2048 与 graph buckets 对齐是 bonus 不是目的 |

---

## 8. 已知遗留

- C3 在 NPU 上反常输 C1 的根因还未核查(`vllm-ascend` chunked prefill 实现层),论文 threats to validity 需要
- chunking 在 long-prompt 1 seed 给 +18pp,需补 multi-seed 验证(P1.7b 阶段顺带)
