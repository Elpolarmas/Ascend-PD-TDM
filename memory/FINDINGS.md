# Finding 链 + 设计与实测的差距

> 时间倒序。新 finding 闭环时追加一段。

---

## D-015 续 (2026-05-27) — F5b: 修正 F5 DH chunk degeneracy + 推翻"单调谱"framing,改为二维 tax-ratio 谱

**触发**:F5 DH cell (prompt~64) 被指出是 chunk degenerate — prompt=64 远小于 chunk_budget=2048,三 paradigm 在 prefill 路径全部 short-circuit 到"近似纯 decode iter",所谓的"Δgoodput≈0"不是 decode-heavy 工况属性,而是 prefill 缺失。"workload prefill-leaning 单调谱"叙事因此有 confound:F5 谱表 4 个点既变了 ratio 又变了 prompt 绝对长度。

**F5b 设计**:固定 ratio 1:2,只扫 prompt 绝对长度。
- **A1 = dh_chunk1**:prompt~2048 (1 chunk) / output~4096
- **A2 = dh_chunk2**:prompt~4096 (2 chunk) / output~8192

**Matrix**:2 wl × 2 paradigm(sarathi c3 chunk=2048 / pdtdm m31-fix)× 5 QPS = 20 cell。3 SLO post-hoc reclassify。**数据源**:`results/f5b_decode_heavy_long_prompt/{dh1,dh2}_{sarathi,pdtdm}_seed0/`。脚本 `experiments/run_f5b_{a1,a2}_only.sh`。Wall A2 ~36min + A1 ~28min ≈ 1.1h。

### A1 (prompt~2048 / output~4096)

err=0 整套(系统未饱和),actual_qps 跟 target 吻合 0.05-0.27。

| QPS | ttft_mean m31/c3 | tpot_p99 m31/c3 | Δgp_strict (ttft<200 ∧ tpot<40) |
|---|---|---|---|
| 0.05 | 199 / 235 | 25.4 / 25.4 | 0 (100% meet 二者 56%) |
| 0.10 | 126 / 156 | 26.8 / 26.8 | 0 (86% meet 平手) |
| 0.15 | 98 / 135 | 29.7 / 27.8 | **+18 (+3%)** |
| 0.20 | 65 / 110 | 33.3 / 29.6 | **+68 (+8%)** |
| 0.25 | 84 / 125 | 39.2 / 36.3 | **+59 (+6%)** |

→ m31 TTFT mean systematic 快 30-45ms,TPOT p99 高 QPS 时**反输** 2-4ms,goodput 微胜 6-8%。**远不如 F5 balanced (+45%)**。loose SLO (ttft<500 ∧ tpot<100) 全员 100% 完全无区分。

### A2 (prompt~4096 / output~8192)

非饱和区(QPS 0.025-0.075)err=0;饱和区(QPS 0.1-0.15)err 25-30%(client PoolTimeout),m31/c3 对称受影响。

| QPS | sat? | ttft_p99 m31/c3 | tpot_p99 m31/c3 | Δgp_strict (ttft<500 ∧ tpot<50) |
|---|---|---|---|---|
| 0.05 | no | 453 / 527 | 26.6 / 26.4 | +66 (+30%) — 小样本 7 req |
| 0.075 | borderline | 509 / 562 | 29.2 / 28.1 | **0** |
| 0.10 | sat | 530 / 594 | 33.0 / 30.3 | **−66 (−15%)** |
| 0.15 | sat | 561 / 563 | 38.2 / 34.4 | **−58 (−9%)** |

→ 非饱和区:**wash** (TTFT 几乎同,TPOT m31 略慢)。饱和区:**m31 略输 c3 9-15%**(deep-decode-queue 拉胯,见 mechanism)。

### Mechanism — 推翻"prefill-leaning ratio 单调"叙事,改为"mixed-iter tax 占比"二维谱

**A1 / A2 / F5 balanced 三组同 ratio 1:2~1:1 但绝对量级不同,m31 优势量级差一个数量级**:

| workload | prompt | output | m31 Δgp strict | 解读 |
|---|---|---|---|---|
| F5 balanced | 512 | 512 | **+16~+89%** | tax 高占比(短 prompt 频繁 + 短 decode batch) |
| **F5b A1** | 2048 | 4096 | **+3~+8%** | tax 摊薄一半(prompt 1 chunk,decode 队列中等深) |
| F5b A2 非饱和 | 4096 | 8192 | **0%** | tax 完全摊薄(prompt 2 chunks,decode 极深) |
| F5b A2 饱和 | 4096 | 8192 | **−9~−15%** | m31 cycle 在 deep queue 下反劣 |
| F5 DH(degenerate) | 64 | 1024 | **0%** | prefill 远小于 chunk,tax 不存在 |

**新 first-principles 模型**:

```
PD-TDM_advantage ∝ tax_ratio
                ≈ (prefill_iter_per_req × tax_per_prefill_iter)
                  / (decode_iter_per_req × decode_batch_size)
```

- prefill 那边大(prompt 长 + 来得勤,T6 code 或 F5 balanced 短 prompt 高 QPS) → m31 大胜
- decode 那边大(output 长 + active decodes 多,A2)→ tax 摊薄 → m31 优势消失甚至反劣
- prefill 太小(F5 DH prompt=64) → tax 不存在 → 三 paradigm 退化

**老 framing "ratio 单调谱" confounded**:A1 ratio 1:2 比 balanced 1:1 更 decode-leaning 但绝对量级大 4 倍,呈现完全不同优势。**真实是二维(prompt 绝对量级 × output 绝对量级)平面**。

### 对 D-015 三柱辩护的影响

| 柱 | F5b 后状态 |
|---|---|
| 1(chunk_budget 继承) | ✓ 不变 |
| 2(NPU variable-query 税) | ✓ **进一步加强** — F5b A2 实证了 F5 hypothesis 中"tax 在大 decode batch 上 amortize"的正面方向,c3 单 mixed iter 在 deep queue 下反而高效 |
| 3(selector + kernel 满血) | ✓ 不变;A1 上 m31 TTFT mean 仍快 30-45ms 印证 selector |

### 对 paper §3 / §5 narrative 的影响

**§3 主图**:
- 旧 narrative:"PD-TDM Δ vs prompt:output ratio 单调上升"(1 维曲线)
- 新 narrative:**heatmap on (prompt_tokens, output_tokens) 2D 平面**,色阶 = m31 strict gp Δ
- 现有 6 个数据点(F5 DH / F5 bal / A1 / A2 / T6 conv / T6 code)够 sparse 画热图,**够 paper §3 用**

**§5 caveats**:
- (新)**m31 在 deep-decode-queue 饱和区略输 c3 ~10%** — 必写,fair-trial framing 关 reviewer "你 cherry-pick" 攻击
- (新)**output 长 + decode 队列深时优势消失** — 收紧 scope claim,只在 mixed-iter tax 高占比 regime 主张 m31 优势
- (修)**Azure conv/code prefill-leaning 偏置**仍是 valid caveat,但**不是优势唯一来源** — F5 balanced(1:1)+ A1(1:2)上 m31 仍胜

### F5b Caveats

- **单 seed**,A1 全 sweep err=0 数据干净,但 QPS=0.05 只 9 req 统计噪声大
- A2 QPS≥0.1 饱和(actual_qps 压不到 target,client PoolTimeout 7/23 = 30%),饱和区 paradigm 对比 dubious — 干净对照在 0.05-0.075
- **QPS 数值跟 T6 / F5 不可比**:A1/A2 per-req decode wall 极长(135s / 287s),Little's Law 决定饱和 QPS 上限 ~0.26 / ~0.11。0.15 在 A2 上 ≡ 2.5 在 F5 balanced 上的负载强度
- **未做 chunk_budget ablation** — 仍是 T6 / F5b 共同未关闭的 reviewer 攻击面

### 跟进事项

- **P0**:`MEMORY.md` 加 feedback 条 — "decode-heavy 不能用 prompt:output ratio 单变量定义,需看绝对 prompt × output 量级,prompt < chunk_budget 则 paradigm 在 prefill 路径坍缩"
- **P1**:paper §3 主图从单调曲线改 heatmap(2D)
- **P1**:T6 m31@chunk_budget ablation(D-013 跟进事项一直挂着)— 越快越好
- **P2**:A1/A2 跑 3-seed 复跑 confirm 量级
- **P3**:A2 饱和区 "m31 deep-queue 略输" 机制详查 — 拿 iter telemetry 看 cycle 时间 vs c3 mixed iter

### 相关

- 决策:[[D-015 三柱辩护]](`DECISIONS.md`)+ F5 finding(上下文)
- 数据:`results/f5b_decode_heavy_long_prompt/`(20 JSON)
- 脚本:`experiments/run_f5b_a{1,2}_only.sh`
- 链接:[[pd-tdm-first-principles-defense]] 柱 2 强化

---

## D-015 续 (2026-05-26) — F5: decode-heavy + balanced 合成 workload sweep,first-principles 三柱模型主要验证

**触发**:导师 challenge #2 — Azure conv/code 实测都 prefill-leaning(prompt:output = 5.5×/73×),要求验证 PD-TDM 优势是否 workload-specific。F0+ Option A 实证柱 2 后,启动 F5 跑合成 workload sweep。

**Matrix**:2 wl(decode-heavy prompt~64/output~1024;balanced prompt~512/output~512)× 3 paradigm(vanilla_cb / sarathi / pdtdm)× 5 QPS × 1 seed = 30 cell。3 SLO post-hoc reclassify。

**数据源**:`results/f5_decode_heavy_synth/{wl}_{paradigm}_seed0/{cfg}_qps{Q}.json`。脚本:`experiments/run_f5_decode_heavy_synth.sh`。Wall ~2h。

### Decode-heavy(prompt~64 / output~1024,1:16)

**Goodput 三 paradigm 几乎完全相同**(差 < 1%):

| QPS | vanilla_cb gp | sarathi gp | **pdtdm gp** | meet% (s1 strict) |
|---|---|---|---|---|
| 0.2 | 244.3 | 244.3 | 244.3 | 100% / 100% / 100% |
| 0.4 | 502.0 | 502.0 | 502.0 | 100% / 100% / 100% |
| 0.6 | 769.1 | 769.1 | 769.1 | 100% / 100% / 100% |
| 0.8 | 961.9 | 961.9 | **970.0** | 99.3% / 99.3% / **100%** |
| 1.0 | 1219.6 | 1217.0 | 1219.6 | 100% / 100% / 100% |

**但 raw latency 显示 PD-TDM TTFT 砍半**(SLO 100ms 全员都过所以 goodput 不分):

| QPS | paradigm | ttft mean | ttft p99 | tpot mean | tpot p99 |
|---|---|---|---|---|---|
| 0.2 | vanilla_cb | 66.1 | 90.4 | 24.5 | 26.2 |
| 0.2 | sarathi | 69.1 | 81.1 | 25.3 | 26.3 |
| 0.2 | **pdtdm** | **29.5** | **73.9** | 24.5 | 25.8 |
| 0.6 | vanilla_cb | 69.4 | 88.7 | 26.6 | 28.1 |
| 0.6 | sarathi | 67.3 | 89.2 | 26.1 | 27.7 |
| 0.6 | **pdtdm** | **28.0** | **38.1** | 26.9 | 28.1 |
| 1.0 | vanilla_cb | 71.2 | 86.7 | 28.6 | 30.6 |
| 1.0 | sarathi | 71.8 | 88.6 | 29.3 | 32.0 |
| 1.0 | **pdtdm** | **28.2** | **42.0** | 29.7 | 32.4 |

→ **细化结论**:DH 上 PD-TDM 仍把 TTFT mean 砍半(28 vs 70ms)+ TTFT p99 降一半(38-74ms vs 81-90ms);但因为 SLO 100ms 太松,**所有 paradigm 全员过 → goodput 不区分**。PD-TDM 的 latency 优势在 DH 上"锦上添花,无决定性",不是"完全没用"。

### Balanced(prompt~512 / output~512,1:1)

**PD-TDM dramatic 胜出**(strict SLO s1,ttft<100ms ∧ tpot<50ms):

| QPS | Sarathi gp / meet% | **PD-TDM gp / meet%** | Vanilla CB gp / meet% |
|---|---|---|---|
| 0.5 | 121.9 / 46.8% | **230.1 / 89.6%(+43pp)** | 110.1 / 42.9% |
| 1.0 | 374.6 / 67.1% | **554.4 / 95.8%(+29pp)** | 394.6 / 70.1% |
| 1.5 | 683.5 / 78.7% | **861.3 / 96.0%(+17pp)** | 697.3 / 79.4% |
| 2.0 | 991.4 / 82.1% | **1157.2 / 97.7%(+16pp)** | 1013.9 / 84.5% |
| 2.5 | 1231.1 / 79.4% | **1543.7 / 98.2%(+19pp)** | 1324.9 / 85.0% |

Δ goodput vs Sarathi:**+16% ~ +89%**,且 vanilla_cb 跟 sarathi 类似(短 prompt 反例方向不明显)

→ **PD-TDM 在 balanced 上仍有强势优势,不是 Azure 偏置带来的假象**

**Raw latency**(进一步证明优势来源):

| QPS | paradigm | ttft mean | ttft p99 | tpot mean | tpot p99 |
|---|---|---|---|---|---|
| 0.5 | vanilla_cb | 110.9 | 191.3 | 26.2 | 28.6 |
| 0.5 | sarathi | 111.1 | 188.9 | 26.4 | 32.7 |
| 0.5 | **pdtdm** | **68.6** | **164.2** | 25.9 | 30.0 |
| 1.0 | vanilla_cb | 89.0 | 153.7 | 26.8 | 29.5 |
| 1.0 | sarathi | 91.3 | 159.1 | 27.7 | 32.1 |
| 1.0 | **pdtdm** | **50.7** | **135.1** | 28.1 | 33.4 |
| 1.5 | vanilla_cb | 85.3 | 155.8 | 28.2 | 31.0 |
| 1.5 | sarathi | 86.0 | 155.0 | 29.1 | 31.6 |
| 1.5 | **pdtdm** | **45.0** | **126.9** | 29.9 | 33.8 |
| 2.0 | vanilla_cb | 82.9 | 152.4 | 30.8 | 35.3 |
| 2.0 | sarathi | 85.2 | **192.1** | 31.2 | 36.3 |
| 2.0 | **pdtdm** | **39.4** | **116.6** | 31.2 | 36.2 |
| 2.5 | vanilla_cb | 84.0 | 196.7 | 32.6 | 37.7 |
| 2.5 | sarathi | 92.7 | **290.8** | 34.0 | 40.9 |
| 2.5 | **pdtdm** | **38.4** | **111.5** | 32.6 | 37.6 |

### 核心机制观察:总吞吐相等,goodput Δ 全部来自 latency 分布

**所有 (wl, QPS) 上三 paradigm 的 raw tput tok/s 完全相等**(给同样 QPS / 同样 workload,系统处理同样多 reqs)。Goodput Δ 100% 来自 latency 分布差异:
- PD-TDM 在所有 workload 上把 **TTFT mean 砍半 + TTFT p99 显著降低**(DH 38ms vs 80-90ms,bal QPS=2.5 时 111ms vs 290ms)
- TPOT 三 paradigm 几乎相等(差 < 5ms)
- → **PD-TDM 不让系统处理更多 req,只让现有 req 更早开始 decode**
- → **paradigm 级 goodput 优势 = TTFT 改善让更多 req 跨过 strict SLO 门槛**;DH 上 SLO 100ms 太松,TTFT 改善白送
- 论文 §3 narrative:**"PD-TDM 优势 = 缩短 new-req-to-first-token 路径,而不是提高 system throughput"**

**Sarathi TTFT p99 在高负载 + 长 prompt 上极速恶化**(2.0 QPS = 192ms,2.5 QPS = 291ms vs PD-TDM 117 / 112ms);vanilla_cb 居中(150-200ms);**PD-TDM 是唯一在高负载下 TTFT p99 仍保持稳定的 paradigm**。这是 paper §3 / §5 极强卖点。

### Workload → PD-TDM 优势单调谱(整合 D-014 + D-015)

| Workload | prompt:output | PD-TDM Δ vs Sarathi(strict SLO) |
|---|---|---|
| code(Azure) | 73:1 | +70~+80pp |
| conv(Azure) | 5:1 | +10~+30pp |
| balanced(synth) | 1:1 | +16~+89% goodput,+17~+43pp meet |
| decode-heavy(synth) | 1:16 | ~0%(完全消失) |

**清晰的 monotonic regression**:prefill-leaning 越极端,PD-TDM 优势越大;decode 占主导时优势归零。

### 对导师 challenge #2 的回答

**导师部分正确**:极端 decode-heavy 上 PD-TDM **goodput 优势** = 0,确认 Azure workload prefill-leaning 偏置贡献了**部分**优势量级。

**但 thesis 没被推翻 + 两个反驳要点**:
1. DH 上 PD-TDM **TTFT 仍砍半**(28ms vs 70ms),只是 SLO 100ms 太松,所有 paradigm 全过 → goodput 不区分。**latency 优势在,只是不显著到推动 goodput 数字**。如果场景需要 strict TTFT(< 50ms),DH 上 PD-TDM 仍胜出
2. Balanced workload(1:1,chatbot 常见场景)PD-TDM 仍 +45% goodput / +29pp meet 在 QPS=1.0 strict SLO 下,**说明优势不是 Azure 数据集 artifact**

**Mechanism self-consistent**:
- DH 上 prefill chunk ≈ 64 tokens,mixed iter 只是 "64 prefill + ~20 decode" 几乎 = 纯 decode iter → mixed batch tax ≈ 0 → 三 paradigm 等价 ✓
- Balanced 上 prefill chunk ≈ 512 tokens(每 req 1 chunk),mixed iter ≈ 200ms vs phase-pure iter (40ms prefill + 33ms decode = 73ms) → 大幅省时 ✓

### Paper 写作影响

- **§5.3 Scope claim 收紧 + 放宽**:
  - 收紧:"在 prompt:output ≤ 1:16 的极端 decode-heavy regime,PD-TDM 优势消失,所有 paradigm 等价"
  - 放宽:"在 prompt:output ≥ 1:1 的 chatbot 常见 regime(含 conv / balanced / code),PD-TDM 有 +16~+89% goodput / +17~+43pp meet 优势"
- **§3 motivation**:从 "Azure 偏置 prefill-leaning 上 PD-TDM 大胜" 改为 "PD-TDM 优势随 workload prefill-leaning 程度单调上升,balanced 仍 +45% gp"
- **§F5 figure**:画 PD-TDM Δ vs Sarathi 关于 prompt:output ratio 的曲线(横轴 ratio,纵轴 Δ goodput pp at QPS=1 strict),把 4 个 workload 串成连续 narrative

### 三柱辩护实证状态更新([[pd-tdm-first-principles-defense]])

| 柱 | 实证 |
|---|---|
| 1(chunk_budget 继承) | ✓ 引 Sarathi 论文 Fig.7 + chunk=2048 锁定 |
| 2(NPU variable-query 税) | ✓ qualitative(F0+ Option A);F5 数据进一步验证:DH 上 prefill ≈ 0 → tax ≈ 0 → 优势 ≈ 0,**模型预测 vs 实测完全 self-consistent** |
| 3(selector + kernel 满血)| ✓ 5/25 telemetry + F5 balanced 上 PD-TDM 高 meet% 印证 |

### F5 Caveats

1. **单 seed**:30 cell × 1 seed,stat significance 弱;balanced 上 PD-TDM 优势数量级大(+45%),seed 噪声不会反转结论,但 ±5% 的精确值有不确定性
2. **loose SLO (s3) 全 100%**:s3 ttft<500/tpot<200 太松,所有 paradigm 满足;只有 strict s1 / mid s2 才区分
3. **DH 上 mid s2 也全 100%**:DH 系统本来就 idle(per-req 慢但 batch 满),都过 mid SLO;**只在 s1 strict 区分得出微弱差异**
4. **没跑 decode-heavy 高 QPS 饱和点**:DH QPS 上限 1.0 时三 paradigm 都 100% meet → 真正的饱和点没扫到,**可能更高 QPS 下 PD-TDM 仍有 latent 优势**(future work)

### 跟进事项

- (可选 P2)DH 高 QPS 补扫:QPS=[1.5, 2.0, 2.5] DH 看是否会暴露 PD-TDM 边际优势
- (可选 P2)bal 3 seed 补强:确认 +45% 数量级不变
- (P3)落 `experiments/posthoc_f5_decode_heavy.py` 把上述 inline python 持久化

### 相关

- 决策:D-015(三柱辩护提出)+ F0+ Option A finding(柱 2 实证)+ F5 finding(本块)
- 数据:`results/f5_decode_heavy_synth/`(30 JSON)
- 脚本:`experiments/run_f5_decode_heavy_synth.sh`

---

## D-015 续 (2026-05-26) — F0+ Option A: NPU mixed batch 开销 ~15% 实证(first-principles 柱 2 ✓)

**触发**:D-015 first-principles 三柱辩护柱 2(NPU variable-query 税)需要从现有 c3 / m31 telemetry 拆出 mixed iter 的额外开销,实证 "NPU 混合 batch 比 serial 跑 P + D 还慢"。

**方法**:用 m31 iter.jsonl 数据拟合纯 phase 成本模型,代入 c3 mixed iter 的 (prefill_tokens, decode_tokens) 算 expected serial cost,差额 = "mixed batch tax"。

### 拟合 cost model(m31,code,k2.8 + k4.9 seed0,2 cells 合并)

- **纯 prefill iter**:chunk-cap 满 2048 tokens → **150.0ms**(n=1679,2.12 reqs avg)
- **纯 decode iter linear fit**:`iter_time_ms = 23.3 + 0.495 × num_reqs`(n=2550,R² 视 buckets stable)
  - 23.3ms 固定 = 每 iter kernel launch + scheduler tick + ...
  - 0.495ms / req = 边际 decode-per-req(memory-bound,KV cache 访问主导)

### c3 mixed iter 开销(同 cell 同 seed,2048-token chunk-cap iters only)

| cell | n_mixed | c3 actual mean | serial expected | overhead | % |
|---|---|---|---|---|---|
| code k2.8 | 641 | **220.0ms** | 185.7ms | **+34.3ms** | **15.6%** |
| code k4.9 | 1062 | **217.0ms** | 185.3ms | **+31.7ms** | **14.6%** |

### 按 reqs 桶细分(code k2.8,验证开销不只是 outlier)

| reqs | n | actual | serial exp | overhead | % |
|---|---|---|---|---|---|
| <10 | 7 | 210.1 | 174.7 | +35.4 | 16.9% |
| 10-19 | 17 | 245.5 | 178.9 | +66.6 | 27.1% |
| 20-29 | 207 | 221.4 | 183.7 | +37.7 | 17.0% |
| 30-39 | 397 | 218.5 | 187.0 | +31.5 | 14.4% |
| 40+ | 13 | 214.7 | 190.0 | +24.8 | 11.5% |

→ 开销随 decode 批量增长缓慢下降(11-17%,排除 r=10-19 outlier),稳定存在,**不是 amortize 掉的固定 launch**

### Finding 结论

- **NPU 910B3 上 mixed batch iter 比 serial 跑 P + D 慢 ~15%(~33ms / 220ms)**,**不是 ≤ serial(Sarathi GPU 假设)**
- Sarathi 论文核心假设 "mixed batching hides decode latency cost-free" 在 NPU 不成立;PD-TDM 通过 phase-pure batching 省掉这部分
- **解释 PD-TDM paradigm 级优势**:m31 cycle 189ms vs c3 mixed iter 220ms 的 31ms gap = **这就是 mixed batch tax**,跟分析得到的 ~33ms 几乎完全吻合(self-consistent check ✓)

### 实证状态更新([[pd-tdm-first-principles-defense]] 柱 2)

- ⏳ → **✓ qualitatively validated**(NPU mixed batch 确实有正开销 vs serial,paper §3 可以用)
- 仍 ⏳:精确归因到 "variable-query length 税" vs "KV layout interference" vs "kernel launch" — 当前 ~33ms 是 **all-in mixed batch overhead**,不是纯 attention kernel variable-query 一项;若要严拆需 Option B microbenchmark(同 req 数 + 同 token 数,仅变 query length 分布)

### Caveats

1. **单 seed**,仅 code workload(c3 telemetry 只 2 cell)
2. 假设 c3 mixed iter 每次 1 个 prefill chunk(若 2 个短 prefill 合一 iter,prefill_tokens 估计有误,但 chunk_budget=2048 限制下大多是 1 chunk)
3. 线性 cost model 在 chunk-bounded prefill / 边际 decode-per-req 上合理,极端 batch 可能偏
4. ~15% 是 **upper bound of variable-query overhead**(实际可能 10-12%,因为含其它 mixed 特有开销)

### Paper 写作影响

- §3 性能分析可以正面写:**"NPU mixed batch tax 实测 ~15%,从根本上推翻 Sarathi 在 NPU 上的 free decode hiding 假设"**
- §3.1 motivation 重排:承认 Sarathi GPU 上的成功 → 转到 "NPU 硬件 attention kernel variable-query 支持弱 → 让 phase-pure 反超"
- §5 Caveat 仍需说明:开销精确归因到具体 NPU kernel 行为待 Option B 微基准(future work)

### F5 启动判断

- first-principles 柱 2 已经 qualitative validated → **F5 可以启动**(decode-heavy synthetic sweep)
- F5 现在的 hypothesis:**decode-heavy 工况下 PD-TDM 优势会缩小**(因为 mixed batch 开销在大 decode 批量上 amortize,c3 mixed iter overhead 从 17% 降到 11%)
- 但优势不会消失(11% 仍是正开销)
- 这个预测如果 F5 验证 → first-principles 模型完全建立
- 如果 F5 反例(decode-heavy 上 PD-TDM 反输)→ 三柱模型有 missing 项,重新找

### 相关决策 / 代码

- 决策:D-015(三柱辩护提出)+ 本 finding(柱 2 实证)
- 数据源:`results/c3_telemetry/code_k{2.8,4.9}_seed0/tdm_trace/qps_sweep_c3_cp_iter.jsonl` + `results/m31fix_validate/t6_code_k{2.8,4.9}_s1_s0_m31/tdm_trace/qps_sweep_c2_tdm_m31_2048_iter.jsonl`
- 分析脚本:**未落盘**(inline python,见 conversation log);如需复现写到 `experiments/posthoc_variable_query_overhead.py`

---

## D-015 (2026-05-26) — Azure trace token 分布实测(conv/code 都 prefill-leaning,反驳 "balanced workload" 自我描述)

**触发**:导师 5/26 challenge #2 质疑实验优势是否特定于 prefill-长 / decode-短 数据集;之前底稿口头说 "conv 输入输出差不多" 自圆其说,实测后被推翻。

**实测**(`AzureLLMInferenceTrace_{conv,code}.csv`,5/26):
| Trace | ContextTokens mean / median / p10 / p90 / max | GeneratedTokens mean / median / p10 / p90 / max |
|---|---|---|
| conv | 1155 / 1020 / 207 / 2735 / 14050 | **211 / 129 / 54 / 424 / 1000** |
| code | 2048 / 1469 / 147 / 5194 / 7437 | **28 / 13 / 7 / 55 / 1899** |

**finding**:
- conv prompt:output ratio ≈ **5.5×**,code ≈ **73×** — **两个 trace 都是 prefill-leaning**
- 不存在文献意义上的 decode-heavy regime,所有 paradigm-level 优势 claim 都只在 prefill-leaning 上 valid
- 之前所有 "conv balanced 而 code prefill-heavy" 的表述都是 over-claim;实际 conv 也偏 prefill,只是没 code 极端

**影响**:
- paper §5 Caveat 必须明写 prefill-leaning scope 限定(D-015 已加原则)
- 需要 F5(decode-heavy synthetic sweep,P1)正面证伪 / 验证 PD-TDM 优势是否 workload-specific;但**必须先 F0+(kernel breakdown,P0)实证 first-principles 模型再跑**
- `AzureLLMInferenceTrace_{conv,code}_long3000.csv` + `_hybrid_conv2code.csv` + `_hybrid_code2conv.csv` 在 trace 目录里存在,未来需要确认是否提供 decode-heavy 补充(估计 long3000 是长 context 非长 output,不能直接复用)

**相关决策**:D-015 first-principles 三柱辩护 / F5 降级 / paper scope 收紧

---

## D-014 (2026-05-26) — Baseline 重定位 + NPU 首次复现 Sarathi finding + 短 prompt 反例 fresh finding

**触发**:5/26 准备 mentor 汇报时 reframe baseline 命名,发现老 c3 chunk=8192 数据(原标 "c3 unfair 版本",D-013 时被换成 c3-fair)实际上就是文献意义 Vanilla CB equivalent(`chunked_prefill_enabled=True` + `max_num_batched_tokens=8192`,Azure trace prompt cap=7000 < 8192 → chunk 不触发 = mixed batch + 长 prompt 整 iter 一次跑完)。

**Baseline 重定位**:
- ~~c1 (admit-driven phase-pure)~~ → **从 paper 主线移除**(NPU-specific niche,文献无对应命名)
- 老 c3 chunk=8192 → **Vanilla CB**(主流 LLM serving 文献 baseline,Sarathi 论文比较的就是这个)
- c3-fair chunk=2048 → **Sarathi chunked prefill**(主流文献 baseline)
- m31-fix → **PD-TDM**(本工作)
- c4_pd 1P1D → 不变(disagg reference)

### NPU 上首次直接量化 Sarathi 论文 OSDI'24 核心 finding

**code workload(长 prompt)上 Sarathi 大胜 Vanilla CB**(`Sarathi − Vanilla CB` Δ pp,3-seed median):
| SLO | k=0.7 | k=1.4 | k=2.1 | k=2.8 | k=3.5 | k=4.2 | k=4.9 |
|---|---|---|---|---|---|---|---|
| s1 | +13.0 | +22.4 | -1.1 | +1.7 | +0.9 | +1.1 | +5.5 |
| s2 | +16.3 | +43.6 | +61.6 | +70.6 | +74.5 | +75.2 | +74.8 |
| s3 | +0.6 | +7.1 | +32.5 | **+87.5** | +80.0 | +80.6 | +80.6 |

→ 长 prompt 上 chunked prefill 把灾难救活的 dramatic 量化,跟 Sarathi 论文 GPU 上 finding 同方向同量级。

### Fresh finding(NPU 上 Sarathi 论文没明示的反例)

**conv workload(短 prompt)低 k 上 Sarathi 反输 Vanilla CB**:
| SLO | k=0.5 | k=1.0 | k=1.4 | k=1.8 | k=2.2 |
|---|---|---|---|---|---|
| s1 | -6.2 | -6.5 | -6.2 | -3.5 | -1.6 |
| s3 | -5.2 | -2.0 | +1.2 | +12.9 | +24.9 |

物理:短 prompt 在 chunk=8192 整 iter 一次跑完 ttft 更快;chunk=2048 切多份反向拖慢首 token ~30-40ms。Sarathi 论文当时只 evaluate 长 prompt benchmark,**没观察到短 prompt + 低 k 上 chunked prefill 的反向损失**。

### PD-TDM 跟两 baseline 的总账(F2 / F2b heatmap 实证)

- **vs Sarathi(unique contribution)**:strict SLO 上 +30~+77pp(phase-pure 在 chunked prefill 内部额外增益)
- **vs Vanilla CB(总优势)**:全谱完胜,max +87.5pp(code 2.8 s3) — 这是 "NPU 复现 Sarathi finding + phase-pure 额外增益" 的总和

**Paper 须明确归 credit**:PD-TDM vs Vanilla CB 的胜利绝大部分来自 chunked prefill 设计(跟 Sarathi 共享),**不能 claim 是 PD-TDM 独占**;PD-TDM 真正独占的 contribution 仅是 F2 那张 heatmap 的数字。

### Artifact 状态

- ✅ Aggregate JSON 已含 vanilla_cb 列(`m31fix_phase1_pointwise.json`)
- ✅ 18 张 PNG 重画完成(`paper_F1{a-f}_*_pareto{,_3way}.png` × 12 + `paper_F2{,b,c}_*_heatmap{,_3way}.png` × 6)
- ❌ **Vanilla CB iter telemetry 缺**(老 c3 5/22 跑时 D-013 patch 还没加),F0 跟进任务 ~30min wall

详 `DECISIONS.md` D-014 / `T6_FINDINGS.md` § Vanilla CB vs Sarathi 基准对照 / `MENTOR_DISCUSSION.md` §3-4 完整 framing。

---

## Phase 1 T6 m31-fix (2026-05-25) — D-013 主结果:m31 paradigm 全胜 c1/c3-fair,thesis framing 修正

**Matrix**:2 wl × 7 k × 3 SLO × 3 seed = 126 cells。详 `T6_FINDINGS.md`。
**数据**:`results/m31fix_validate/`(m31)+ `results/c3_chunk2048_supplement/`(c3-fair)+ `results/phase_2_t6_burst_goodput/*_nonpid/`(c1/c3 nonpid)
**聚合**:`results/phase_2_post/m31fix_phase1_pointwise.json`
**分析脚本**:`experiments/posthoc_m31fix_phase1.py`

### 核心结论

- **paradigm-level 全胜 c3-fair**:conv 21/21(12 decisive +5pp↑/9 small win/0 tie/0 loss);code 9 decisive /11 tied / 1 noise loss
- **latency 全维度**:c3-fair 在任何 wl × ttft/tpot mean/p99 上 0 wins(c1 仅 conv ttft_p99 2 cell)
- **goodput 提升**:conv s1 k=3.0 Δ +1870 tok/s(+123%);code s1 k=4.9 Δ +351 tok/s(+403%)

### Telemetry 机制 verify(code k2.8 s1)

- m31 prefill iter mean 150ms,batch 永远 2048 token,1-6 reqs 并行(mean 2.16)
- m31 decode iter mean 39ms,29 reqs / iter
- m31 cycle = 150 + 39 = **189 ms** < c3 单 mixed iter **~225 ms**(-36ms / -16%)
- 直接解释 tpot p99 优势(m31=190 vs c3=228)

### Thesis framing 修正

**原 D-012**:"trade tail latency for mean latency"
**实测推翻**:m31 mean **和** tail 同时改善(不是 trade-off)

**新 framing**(Gap-8):
> Phase-pure temporal multiplexing 把 mixed-batch 持续低幅 prefill-decode 干扰替换为 phase-specific 高效 batch。cycle 时间(189ms)短于 c3 mixed iter(225ms),消除 mixed-attention overhead → mean/tail 同步改善。

### Caveat

- 限定 Azure burst trace + Qwen3-8B + 2-NPU + chunk_budget=2048
- **极重 prefill demand 未测**(长 prompt + 极高频 burst),m31 cycle 退化时 trade-off 会暴露
- **c3 vs c1 不普适胜出**:conv strict SLO 上 c3 反输 c1 -4~-19pp(chunked prefill 短 prompt 高频负载弱点)

---

## chunked_schedule bug + 修正 (2026-05-24 / D-013) — Gap-6

**Bug**:`vllm-ascend/vllm_ascend/core/tdm/chunked_schedule.py` waiting loop 没 gate `phase=="prefill"`,decode iter 偷塞 prefill chunk 并 skip decode loop(L396 判断)→ 双输

**修复**:waiting loop while 加 `self.phase == "prefill" and ...` gate(Diff #6)

**影响**:T6 旧版 168 个 m31 cells 数据全废,已清理删除。Phase 1 用 fix 版 m31 重跑 126 cells(`m31fix_validate/`)。

---

## phase_iters semantic bug (2026-05-25 / D-013) — Gap-7

**Bug**:`vllm-ascend/vllm_ascend/core/tdm/engine.py:30-40` `reconcile()` 比较失效。`apply()` 提前更新 `self._phase = candidate`,导致 `reconcile()` 内 `actual == self._phase` 退化成 `actual == candidate`。

**后果**:
- `phase_iters` 实际记录"controller decision obey 连续 iter 数",非"phase 持续 iter 数"
- `HardConstraints` `constraint_min_slice` 永远 0% 触发(因 phase_iters 永远 ≥ 2)
- `constraint_max_slice` 误触发 46%(无害,因 controller planned 通常本来就想切)
- **实际 phase 切换 100% 由 controller selector + parent_auto_flip 驱动,HardConstraints guard 实际不起作用**

**影响**:不影响 Phase 1 数据(guard 本来就 effectively 不存在);修复后预计行为不变,但代码正确性需要修。Follow-up 跟进项。

---

## Phase 0 (2026-05-20) — D-012 实验基础设施 + H2 verified + defining figure 数据

**Phase 0 = T3 ideal + T4 H2 verify + T2 defining figure(本 session 5/20 全做完)。**

### T3 micro-benchmark ideal (Sarathi-Serve protocol)

**实装:** `lib/workload.SequentialSampler` + `qps_sweep.py --arrival-mode sequential`。concurrent=1 random-sample N=50 rows from trace,前 5 warmup drop。

**配置:** Qwen3-8B × {conv, code} × N=50 × 3 seed。`results/phase_a_micro_ideal/slo_grid.json`。Wall ~10min。

**结果:**

| workload | prompt mean/p99 | output mean/p50 | ideal_ttft_p99 | ideal_tpot_p99 |
|---|---|---|---|---|
| conv | 1029 / 4202 | 208 / 124 | **415 ms** | **23.4 ms** |
| code | 1764 / 6275 | 30 / 14 | **684 ms** | **24.9 ms** |

SLO 4 档 = ideal_p99 × {5, 10, 15, 25}×。conv 5×=2076/117ms,code 5×=3418/124ms。

### T4 fallback H2 diagnostic

**实装:** `run_phase_a_h2_ratiomax.sh`(`--slo-ratio-max 0.95`)+ post-hoc 比较 ctrl.jsonl ratio 时序 vs v8_matrix(原 ratio_max=0.8)。

**配置:** m31_2048 × conv off1860/off2160 × 3 seed × 60s。`results/phase_a_h2_ratiomax/`。Wall ~1h。

**核心数据(off1860 peak,3 seed pool):**

| ceiling | ratio mean | ratio p50/p99 | ttft_p99 | tpot_p99 | goodput |
|---|---|---|---|---|---|
| 0.8 (v8) | 0.75 | 0.800 | 769 | 174 | 155 |
| 0.95 (H2) | 0.88 | 0.950 | 824 (+7%) | 171 (-2%) | 165 (+6%) |

**结论:**
1. **H2 mechanism-saturated 在 saturation regime 确认** — PID 钉新 ceiling
2. **off2160 (reverse) PID 自然收敛 0.17** — workload-adaptive 在 unsaturated regime 成立
3. **0.8 不是真最优**:0.95 时 goodput +6%,但 ttft_p99 +7% → 真最优 0.85-0.9 之间
4. **L3 narrative 保住**(D-012 措辞已避开 "SLO-Aware"),M3.1=static ratio per workload 更合理

详 [[project-pdtdm-h2-open]]。

### T2 defining figure 数据(post-hoc azure_p15)

**实装:** `posthoc_phase_0_t2_interference.py`。从 azure_p15 已有 req.jsonl 取 per-req ttft_ms / tpot_ms_p99,pool 4 cfg × 2 workload × 3 seed。

**结果(X axis = TTFT_mean / Y axis = TPOT_p99 mean):**

| workload | config | X (mean interference) | Y (tail interference) | quadrant |
|---|---|---|---|---|
| conv | c1_baseline | 306 | 1095 | 右上(双输) |
| conv | m1_chunk2048 | 257 | 864 | 左下(双赢 ref) |
| conv | **m31** | **242** | **908** | **左上 ← thesis** |
| conv | **c3** | **356** | **571** | **右下 ← chunked prefill** |
| code | c1_baseline | 620 | 6226 | 右上 |
| code | m1_chunk2048 | 443 | 5198 | 左下 |
| code | **m31** | **440** | **5269** | **左上 ← thesis** |
| code | **c3** | **738** | **680** | **右下** |

**完美 4-quadrant split** — D-012 thesis "trades tail latency for mean latency" 实证。

特别:**code workload c3 vs m31 tail 比 7.7×**(680 vs 5269 ms)→ 解释 P1.8 finding (code m31 输 c3 tpot 2.5×)。

数据:`results/azure_p15/interference_coords.json`。

### 对 thesis 影响

- D-012 三层 framework + goal-first thesis 完整数据支撑
- L3 "Per-Workload Static Ratio"(避开 "SLO-Aware")保住
- M3.1 = static ratio per workload + PID 代码 internal 路径合理
- Paper Section 5.3 defining figure 数据 ready

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

> **⚠️ 2026-05-25 注解(D-013 修正):** 本节"phase-pure 双刃 — TTFT 赢但 code TPOT_p99 输 C3 2.5×(992ms)"结论**已被 Phase 1 m31-fix 数据推翻**。原因:本实验用的是 m31-bug 版本(chunked_schedule waiting-loop bug),m31 在 code 上 tpot_p99 飙到 1668ms 是 bug 假象。fix 后实测 m31 code tpot_p99 = 190ms,**比 c3-fair 228ms 还低 38ms**。
> 但本节 kernel 贡献 ≈ 0 的 finding 仍 valid(kernel ablation 路径独立于 chunked_schedule bug)。
> 完整修正数据见 `T6_FINDINGS.md`。

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
- **处理:** D-009 — kernel 完全移出 contribution。phase-pure 是真正卖点。

## Gap-6:`chunked_schedule.py` waiting-loop 没 gate phase 导致 m31 在 long-prompt 上数据全错

- **设计期 claim**(注释 L13, L120-125):waiting loop 在 phase-pure 模式下应只在 prefill iter admit 新 req
- **5/24 实测**:long_prompt code smoke 1073 iter 里 277 个 "decode" iter 实际偷做 prefill(reqs=1, tokens=2048)占 26%,然后因 L396 `len(scheduled_req_ids)==0` 跳过 decode loop → decode reqs 0 token 生成
- **处理**:Diff #6 加 `self.phase == "prefill" and ...` gate(D-013)。T6 旧版 168 个 m31 cells 数据全废,重跑 126 cells

## Gap-7:`engine.py` phase_iters 字段语义 bug 导致 HardConstraints guard 实际不起作用

- **设计期 claim**:`HardConstraints.min_slice_iters=2 / max_slice_iters=8` 防 phase 抖动 + 防 starvation
- **5/25 实测**:`apply()` 提前更新 `self._phase = candidate`,导致 `reconcile()` 内 `actual == self._phase` 退化成 `actual == candidate`。phase_iters 实际是"controller decision obey 连续 iter 数",constraint_min_slice 永远 0% 触发,constraint_max_slice 误触发 46% 但无害
- **处理**:不影响 Phase 1 数据;follow-up 修代码(可能不影响数据,sanity verify)

## Gap-8:原 thesis "trade tail for mean" framing 被 D-013 数据推翻

- **设计期 claim**(D-012):"deliberately trades tail latency for mean latency to maximize goodput"
- **Phase 1 m31-fix 实测**:m31 在 conv/code 全 cell 上 ttft mean/p99 + tpot mean/p99 全维度低于 c3-fair,**mean 和 tail 同时改善,不是 trade-off**
- **机制 verify**(telemetry):m31 cycle (prefill+decode) 189ms < c3 mixed iter 225ms,消除 mixed-attention overhead
- **处理**:D-013 新 framing — "phase-pure batching 让 cycle 时间短于 c3 mixed iter",mean/tail 同步改善
- **caveat**:极重 prefill demand regime(长 prompt + 极高频 burst)未测;若 cycle 退化为多 prefill iter + 1 decode iter,原 trade-off 会重新暴露

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
