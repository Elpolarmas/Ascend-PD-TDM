# PD-TDM 项目记忆

> 新 session 从这里开始读。其它文档按需读。

## 项目一句话

在 single-node multi-NPU(2 张 Ascend 910B3,TP=2,锁 vllm-ascend v0.11.0rc1)上实现 PD-TDM(P/D Temporal Multiplexing)调度 paradigm,提供 goodput-centric improvement under SLO constraints。目标 CCF-B/C 论文。

**当前 thesis 版本:D-014(2026-05-26 baseline 重定位)+ D-015(2026-05-26 导师 challenges 应对 + F0+/F5 实证)**。在 D-013 数据基础上,paper baseline 命名重定位为 Vanilla CB / Sarathi / PD-TDM / c4_pd,NPU 首次复现 Sarathi finding + 短 prompt 反例 fresh finding。D-015 应对导师 3 challenges:trace token 分布实测确认 conv/code 都 prefill-leaning;first-principles 三柱辩护(chunk_budget 继承 + NPU variable-query 税 + selector 节奏);**F0+ Option A 实证柱 2:NPU mixed batch tax ~15%(34ms / 220ms)**;**F5 实证 workload → 优势单调谱:DH(1:16)goodput Δ=0(但 TTFT 仍砍半)、balanced(1:1)PD-TDM +45% gp / +29pp meet @ QPS=1.0 strict、conv(5:1)+10~30pp、code(73:1)+70~80pp**。**核心 mechanism finding**:三 paradigm 总吞吐完全相等,goodput Δ 100% 来自 latency 分布(PD-TDM 在所有 workload 上把 TTFT mean 砍半)。

## 当前状态(2026-05-26 / D-014 baseline 重定位)

### Paper baseline 重定位(D-014)

| Paper 名 | 内部 config | 数据源 |
|---|---|---|
| **Vanilla CB** | `vanilla_cb`(老 c3 chunk=8192) | `phase_2_t6_burst_goodput/*_nonpid/c3_cp_qps0.0.json` |
| **Sarathi chunked prefill** | `c3_cp`(c3-fair @ chunk=2048) | `c3_chunk2048_supplement/` |
| **PD-TDM** | `c2_tdm_m31_2048_fix` | `m31fix_validate/` |
| c4_pd 1P1D disagg(reference) | `c4_pd` | `c4_pd_supplement/` |
| ~~c1 (vllm-ascend default phase-pure)~~ | 不进 paper(NPU-specific niche) | (代码 / 数据保留 internal) |

- **NPU 复现 Sarathi finding**:code workload 上 Sarathi 大胜 Vanilla CB(max +88pp,code 2.8 s3)
- **Fresh finding(短 prompt 反例)**:conv s1 低 k 上 Sarathi 反输 Vanilla CB(-2 ~ -6.5pp)— Sarathi 论文没明示
- **PD-TDM 总优势(vs Vanilla CB)**:全谱完胜,max +87.5pp;**vs Sarathi(独占)**:strict SLO 上 +30~+77pp(phase-pure 在 chunked prefill 框架内额外增益)
- **Aggregate JSON**:`results/phase_2_post/m31fix_phase1_pointwise.json`(D-014 已含 vanilla_cb 列)
- **Figures**:18 张 PNG(`figures/paper_F1{a-f}_*_pareto{,_3way}.png` + `paper_F2{,b,c}_*_heatmap{,_3way}.png`)— F2 = vs Sarathi,F2b = vs Vanilla CB,F2c = NPU 复现 + 反例

### Phase 1 m31-fix sweep 完成(D-013 数据,D-014 baseline 重命名后仍 valid)

- **完成时间**:2026-05-25 01:32 UTC
- **Matrix**:2 wl × 7 k × 3 SLO × 3 seed = 126 cells,0 failed run
- **Outdir**:`results/m31fix_validate/`(m31)+ `results/c3_chunk2048_supplement/`(c3-fair)+ `results/phase_2_t6_burst_goodput/*_nonpid/`(c1/c3 nonpid)
- **聚合**:`results/phase_2_post/m31fix_phase1_pointwise.json`
- **分析**:`experiments/posthoc_m31fix_phase1.py`
- **详细 findings**:`T6_FINDINGS.md`

### Paradigm-level winning region(m31-fix vs c3-fair)

- **conv**: **21/21 全胜**(12 decisive ≥+5pp / 9 small win / 0 tie / 0 loss)
  - strict s1: +9.9~+34.8pp;mid s2: +2.8~+7.2pp;loose s3: +1.9~+4.0pp
- **code**: **9 decisive / 11 tied / 1 noise loss**
  - strict s1 大胜:k≥2.1 时 +71~+77pp;loose s3 都 100% tied(SLO 太松)
- **latency 全维度**:c3 在任何 wl × ttft/tpot mean/p99 上 0 wins;c1 仅 conv ttft_p99 偶 2 cell

### Thesis framing 修正(D-013)

**原 D-012**:"deliberately trades tail latency for mean latency to maximize goodput"

**Phase 1 数据推翻 "trade-off" 表述**:m31 mean 和 tail 同时改善。

**修正后 framing**:
> Phase-pure temporal multiplexing 把 mixed-batch 的"持续低幅 prefill-decode 干扰"替换为"phase-specific 高效 batch"。一个 cycle (prefill_iter 150ms + decode_iter 39ms) = **189ms** 短于 c3 chunk-cap mixed iter **220ms**(5/25 c3 telemetry 实测,same 2048 token budget,c3 batches 1-2 prefill + ~30 decode → mixed-attention overhead +70ms/iter)。mean 和 tail 同步改善。

### Telemetry 机制 verify(code k2.8 s1)

- m31 prefill iter mean 150ms,batch 永远 2048 token,1-6 reqs 并行(mean 2.16)
- m31 decode iter mean 39ms,29 reqs / iter
- m31 cycle = prefill + decode = 189ms < c3 mixed iter ~225ms = -36ms (-16%)
- 直接解释 tpot p99 优势(m31=190 vs c3=228)

### PID 不是优势来源

- target_ratio 推 0.8 ratio_max 撞顶,实际 prefill iter 占比 12-50% 由 waiting queue 决定
- PID 实际为 SLO 违反诊断指示灯,跟 D-011 "PID 退出 paper contribution" 一致
- paper 报 m31 = static ratio per workload

### Caveat(paper Section 5 须 explicit)

- **结论限定**:Azure burst trace + Qwen3-8B + 2-NPU + chunk_budget=2048
- **极重 prefill demand regime 未测**:长 prompt + 极高频 burst 下 m31 cycle 退化,trade-off 此时会暴露
- **c3 vs c1 不普适胜出**:conv strict SLO 上 c3 反输 c1 -4~-19pp(chunked prefill 短 prompt 高频负载弱点)

### 已知 code bug(不影响数据,follow-up)

**phase_iters semantic bug**(`engine.py:30-40`):
- `apply()` 提前更新 `self._phase = candidate`,导致 `reconcile()` 比较失效
- `phase_iters` 实际记录"controller decision obey 连续 iter 数",非"phase 持续 iter 数"
- 后果:`HardConstraints` `constraint_min_slice` 永远 0%,`constraint_max_slice` 误触发 46%(无害)
- 实际 phase 切换 100% 由 controller selector + `parent_auto_flip` 驱动

### 5/25 项目整理(本 session)

- 删除中间废弃数据 ~720M:`phase_2_t6_burst_goodput/*_pid/` (168 m31-bug cells)、`long_prompt_smoke{,_fix,_sweep}/`、`long_5way_q1632/`、`azure_m31_fia_ablation/`、`c3_chunk_smoke/`
- 删除中间 script:`posthoc_phase_2_t6{,_fair}.py`、`run_long_5way_q1632.sh`、`posthoc_long_5way_q1632.py`、`posthoc_long_chunk.py`
- 删除 memory:`LONG_PROMPT_SMOKE.md`
- 删除 auto-memory 12 个过期文档(bug fix 中间状态 / 实验 progress snapshot 等)
- rewrite `README.md` / `T6_FINDINGS.md` / 部分 auto-memory 反映当前 finalize 状态

## 5/25 二次更新(in D-013 scope)

完成补强实验:
- **c4_pd 4-way sweep**(`results/c4_pd_supplement/`):42 cells,paper 4-way 数据齐
- **c3 iter telemetry**(`results/c3_telemetry/`):patch TDMScheduler passive 模式后实测,c3 chunk-cap mean **220ms**,m31 prefill **150ms** → 机制 #1 直接量化 **+70ms mixed-attention overhead**
- **c4_pd steady (Poisson) smoke**(`results/c4_pd_steady_smoke/`):3 cells **推翻 "burst 是 c4_pd 放大器"** 假设,真凶是 1P1D 在 2-NPU 上 prefill capacity ~5-6 QPS

Framing 净变化:
- ~~"burst trace 是 disagg 最坏场景"~~ → "2-NPU 不在 DistServe/Splitwise 应用域,1P1D capacity 数学上不够"(防 reviewer strawman 攻击)
- c3 iter 时长从物理推理升级到 telemetry 直接证据

详 `T6_FINDINGS.md` § c3 iter telemetry / § c4_pd 4-way 补完 / § c4_pd steady vs burst。

## 跟进事项(D-015 续 / F5b 更新)

按优先级排序:

- ~~P0 — F0+ kernel/iter cost breakdown~~ ✓ **完成 5/26**
- ~~P1 — F5 decode-heavy synthetic sweep~~ ✓ **完成 5/26**(但 DH cell prompt=64 后来发现 chunk degenerate,见 F5b)
- ~~P1 — F5b 修正 chunk degeneracy~~ ✓ **完成 5/27**(2 wl × 2 paradigm × 5 QPS,A1 prompt~2048 / A2 prompt~4096,**结论:推翻"prefill-leaning ratio 单调谱",改二维 tax-ratio 谱**;详 `FINDINGS.md` § F5b)
- **P0 — paper §3 figure 重设计**:从单调曲线 → heatmap on (prompt, output) 2D 平面,色阶 = m31 strict gp Δ;现有 6 数据点(F5 DH / F5 bal / A1 / A2 / T6 conv / T6 code)够 sparse 画
- **P0 — T6 m31@chunk_budget ablation**(D-013 跟进事项一直挂着):F5b 暴露二维谱后,reviewer "chunk 选得巧" 攻击面更紧,优先级从 P3 升 P0
- **P1 — prior art 调研**(D-015 challenge 3):用户自行 web search;反馈后定 paper §2 / §1
- **P1 — paper §5 caveats 补**:(a) m31 在 deep-decode-queue 饱和区略输 c3 ~10%;(b) scope claim 收紧到"mixed-iter tax 高占比 regime"
- **P2 — bal / A1 / A2 补 3-seed**:确认量级,paper 主图能用
- **P2 — Phase 2 决定**:Qwen3-4B 跨模型 / BurstGPT 跨 trace / L3 real Azure trace
- **P3 — F0+ Option B**(future work):microbenchmark 严拆 variable-query overhead
- **P3 — 残留**:phase_iters semantic bug fix;落 `posthoc_f5_decode_heavy.py` + `posthoc_f5b.py` 持久化

## 历史决策

- **D-015 续**(5/26)F0+ Option A:NPU mixed batch tax 实证 ~15%(34ms/220ms,两 cell 一致);F5 sweep:DH(1:16)goodput Δ=0(但 TTFT 仍砍半)+ balanced(1:1)PD-TDM +45% gp / +29pp meet @ QPS=1.0 strict;workload → 优势单调谱(code 73:1 → conv 5:1 → bal 1:1 → DH 1:16 单调下降)
- **D-015**(5/26)导师 3 challenges 应对:trace token 分布实测(conv/code 都 prefill-leaning)+ first-principles 三柱辩护(chunk_budget 继承 / NPU variable-query 税 / selector 节奏)+ F0+ kernel breakdown 升 P0、F5 decode-heavy 降 P1
- **D-014**(5/26)Baseline 重定位:c1 → Vanilla CB(老 c3 chunk=8192)+ NPU 复现 Sarathi finding(code workload max +88pp)+ 短 prompt 反例 fresh finding(conv s1 低 k 上 Sarathi 反输 -2~-6.5pp)+ 18 张 PNG 重画
- **D-013**(5/24-5/25)chunked_schedule waiting-loop bug fix + Phase 1 完成 + 机制 telemetry verify + thesis framing 修正(详 DECISIONS.md)
- **D-012**(5/20)thesis 三层结构化 + goal-first framing + 双 SLO goodput + 三层实验 hierarchy
- **D-011**(5/18)thesis 完整重定位:PD-TDM paradigm + SLO-aware fast-loop selector + Framing C scope + goodput-centric metric + 4-way paradigm comparison
- **D-010**(5/15)phase Δ 归因层级修正

完整决策史见 `DECISIONS.md`。

## 不能偏离的几条原则(D-015 修订)

**D-015 新加**:
- **不能否定 Sarathi 在 GPU 上的成功** — paper §3.1 motivation 须承认 "Sarathi 混合 batch + decode 优先是 GPU 上的最优解";PD-TDM 的贡献是 "NPU 上 variable-query 税 + cycle 长度差异让 phase-pure 反超",不是 "Sarathi 错了"
- **结论 scope 须 explicit prefill-leaning** — Azure conv/code 实测 prompt:output = 5×/73×,都不是 decode-heavy;§5 Caveat 须明写,F5 跑完后再据结果收紧或放宽
- **mechanism 实证优先于 workload 验证** — first-principles 模型(F0+ kernel breakdown)未实证前,不要花成本跑 scope 验证(F5),否则可能验证错的预测

**D-014 新加 / 修订**:
- **Paper baseline 命名必须跟主流 LLM serving 文献对应** — Vanilla CB(mixed batch + 无 chunk 上限)/ Sarathi chunked prefill(mixed batch + chunk_budget)/ PD-TDM(phase-pure + chunk_budget) 都是文献术语;c1 (NPU-specific phase-pure) 不再进 paper 主线
- **NPU 复现 Sarathi finding + 短 prompt 反例 finding** 作 paper §3.1 motivation 实证 + §5.3 caveat
- **PD-TDM 跟 baseline 的关系明确归 credit**:vs Vanilla CB 的胜利绝大部分来自 chunked prefill(Sarathi 也共享),PD-TDM 独占 contribution 仅是 chunked prefill 内部 phase-pure 的额外增益(F2 heatmap 数字)

---

## 不能偏离的几条原则(D-013 修订)

1. **thesis = PD-TDM paradigm + phase-pure batching 让 cycle 短于 c3 mixed iter**(mean/tail 同步改善);**不再用 "trade tail for mean" framing**
2. **主报 metric = goodput at SLO**(max sustainable QPS at SLO ≥ X%),Pareto frontier 为主图;SLO meet% secondary
3. **Paper scope = Framing C hybrid**:conceptual = "accelerators lacking SM partitioning support";evaluation = Ascend 910B3 NPU;Title 不锁 NPU
4. **对手 = 4-way paradigm**:C1 / C3 / M3.1(=m31)/ c4_pd;MuxWise 在 Related Work cite,scope explicit complementary
5. **vllm-ascend 锁 v0.11.0rc1**(D-007)
6. **kernel 速度不作为论文主图**(D-009)
7. **PID 不进 contribution**(D-011);paper 报 m31 = static ratio per workload
8. **任何 cite m31 数据前必须确认是 fix 版**(检查 outdir 是 `m31fix_validate/`);旧 `*_pid/` 已删除
9. **c3 vs c1 不普适胜出**:paper Section 5 须 explicit 讲 conv strict SLO 反例

## 文档导航

| 想知道 | 读哪里 |
|---|---|
| 当前完整结论 + 机制 + caveat | `T6_FINDINGS.md` |
| 项目核心 / 架构 / Paper outline / 禁区 | `PROJECT.md` |
| 代码地图 / TDM 插件位置 / 关键文件 | `PROJECT.md` §5 |
| config 缩写(C1/C3/c4_pd/M3.1)+ M 家族演化 | `EXPERIMENTS.md` §2、§4 |
| `results/` 目录用途 / 实验代次时间线 | `EXPERIMENTS.md` |
| Finding 链 / 设计与实测的差距 / 已废弃路径 | `FINDINGS.md` |
| Thesis 调整历史 | `DECISIONS.md` |
| 慢回路 PID 细节(internal,paper 不写)| `design/slo_pid.md` |
| 采样模块(降级为 evaluation methodology)| `design/sampling.md` |
| Chunking 机制 | `design/chunking.md` |
| 状态采集接口 | `design/interfaces.md` |

## 做完事要更新哪些文档

- **跑完实验:** `EXPERIMENTS.md` §2 主表加一行(目录 / 主 finding / thesis 用途)
- **实测推翻原设计:** `FINDINGS.md` § 设计与实测的差距 加一条 Gap-X,并写一条 `DECISIONS.md` D-XXX
- **改 thesis 方向 / 不能偏离的原则:** 必须先写 `DECISIONS.md` D-XXX,才能动其他文档
- **删 / 废弃实验目录:** `EXPERIMENTS.md` § 已废弃实验加一行
- **接口 / 代码地图变化:** `PROJECT.md` §5 + 相关 `design/*.md`

---

**给新 session 的关键 hand-off:**

1. 读 `T6_FINDINGS.md` 看 Phase 1 完整数据 + 机制 verify
2. 读 `DECISIONS.md` D-013 看 thesis framing 修正 + caveat
3. 跟进事项见上文「跟进事项」section
4. **不要回退 D-013 的几个核心结论**:
   - thesis framing = "phase-pure cycle 短于 c3 mixed iter"(不是 trade tail for mean)
   - PID 不是优势来源
   - c3 vs c1 不普适胜出(conv strict SLO 反例)
   - 任何 m31 数据来自 `m31fix_validate/`,不是已删除的 `*_pid/`
