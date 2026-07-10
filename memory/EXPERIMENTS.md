# 实验台账

> 每个 `results/` 子目录的用途 + 主 finding + thesis 用途。跑完新实验追加一行。

---

## 1. 实验代次时间线

| 代次 | 时间 | 内容 | 输出目录 |
|---|---|---|---|
| P0 | ~4 月 | exp_a-f 早期 profile(ACL Graph / P-D 资源画像 / DCMI 开销 / 并发扩展 / Unified vs Phased / Graph-aware) | `figures/fig{1-7}_exp_*.png`(数据已删) |
| P0-1 | 5/6 | static ratio scan | `static_scan_3a/` |
| P1.0 / P1.0b | 5/12 | Graph capture audit(FFTS+ vs AIV) | `baseline_c3_audit{,_aiv}/` |
| P1.4 | 5/11 | Azure trace 4-way 主结果 | `azure_main/` |
| P1.5 | 5/12 | 跨窗扩展 | `azure_p15/` |
| P1.6e | 5/13 | SLO 档 sweep(4 档) | `azure_p16e/` |
| **Phase 0 T3** | 5/20 | Micro-benchmark ideal latency(D-012) | `phase_a_micro_ideal/` |
| **Phase 0 T4** | 5/20 | H2 PID saturation diagnostic | `phase_a_h2_ratiomax/` |
| **Phase 0 T2** | 5/20 | Post-hoc B 4-way interference coords | `azure_p15/interference_coords.json` |
| **Phase 1 T5** | 5/20 | Trace-sampled QPS sweep(已完成,被 T6 取代) | `phase_1_t5_qps_sweep/` |
| **Phase 1 T6** | 5/22 启动 / 5/24 bug 发现 / 5/25 m31-fix 重跑完成 | T6 burst goodput sweep(c1/c3/m31) | `phase_2_t6_burst_goodput/*_nonpid/`(c1/c3)+ `c3_chunk2048_supplement/`(c3-fair)+ `m31fix_validate/`(m31-fix) |
| **5/24 bug fix** | 5/24 | `chunked_schedule.py` waiting-loop bug fix(Diff #6) | D-013 |
| **5/25 cleanup** | 5/25 | 删除中间 / 废弃数据 ~720M | — |
| **MaaS smoke** | 6/30 | rpm_scale 0.02~0.40 sweep(20 档),pdtdm, Qwen3-0.6B, 180s window | `smoke_rpm_scale/` |
| **MaaS 正式回放** | 6/30 23:14 启动 / 7/6 重做 | v3: full trace × rs=0.12 × pdtdm (17.4h, 68% err, PoolTimeout+ContextLength) / sarathi 未完成 | `maas_replay/` |
| **MaaS v4-v6 调试** | 7/6 | 峰值窗口 (rs=0.15, start_offset=21600), Qwen3-8B, concur=4096→1024→64 | v4/v5: 高并发卡死; **v6: concur=64, pdtdm 5747 ok / 0 err / 100% meet / wall 84min** | `maas_replay/run_v{4,5,6}.log` |
| **MaaS v7 分布长度** | 7/6 | log-normal σ=0.35 替代均匀 avg; rs=0.15, concur=64, peak 600s, pdtdm+sarathi | ✅ pdtdm 4336s / sarathi 4712s, 0 err, pdtdm TTFT -13% vs sarathi, wall -15% vs v6 | `maas_replay/run_v7_nohup.log` |
| **MaaS v8-v10 配置迭代** | 7/7 | rs→0.10, subsample, compress_timeline | v8: 无压缩 trace 间隙; v9: 压缩但用 sampled CSV; v10: 全量 CSV subsample=10 | — |
| **MaaS v11 (正式)** | 7/7 12:23 | **rs=0.10, concur=64, subsample=30, compress, 全量 CSV→38 buckets, 17.8K reqs** | pdtdm ✅ (11977s, 0 err, TTFT p99=1182ms, SLO=100%) / **sarathi running** | paper MaaS 评估主数据 |

---

## 2. 当前 active results/ 目录(paper-relevant)

| 目录 | 代次 | 配置 / 规模 | 主 finding | thesis 用途 |
|---|---|---|---|---|
| `phase_a_micro_ideal/` | T3 (D-012) | Qwen3-8B × 2 wld × 3 seed × N=50,concurrent=1 sequential | conv ideal_ttft_p99=415ms / tpot_p99=23.4ms;code 684/24.9ms。SLO 4 档 = ideal × {5,10,15,25}× | **paper SLO grid 数据源**,Sarathi-Serve protocol 同款 |
| `phase_a_h2_ratiomax/` | T4 fallback (D-012) | m31_2048 × slo-ratio-max=0.95 × conv off1860/2160 × 3 seed | off1860 PID 钉新 ceiling 0.95 → H2 mechanism-saturated 确认 | H2 verified;L3 narrative 保住 |
| `azure_p15/interference_coords.json` | T2 (D-012) | post-hoc 4 cfg × 2 wld × 3 seed pool | 4-quadrant defining figure: c1=右上 / m1+chunk=左下 / m31=左上 / c3=右下 | paper S5.3 defining figure 数据 |
| **`m31fix_validate/`** | **Phase 1 T6 m31-fix (D-013)** | **m31 × 2 wld × 7 k × 3 SLO × 3 seed = 126 cells** | **paradigm-level 全胜 c3-fair**(conv 21/21,code 9 decisive + 11 tied + 0 loss);详 `T6_FINDINGS.md` | **paper Section 5 主图数据源**(winning region heatmap + Pareto frontier) |
| **`c3_chunk2048_supplement/`** | T6 Sarathi (D-013) | c3 × chunk=2048 × 2 wld × 7 k × 3 seed = 42 cells | **= Sarathi chunked prefill baseline**(D-014 重命名),post-hoc reclassify 多 SLO | **paper Section 5 主对手 baseline** |
| **`phase_2_t6_burst_goodput/*_nonpid/`** | T6 c1 + Vanilla CB (D-013/014) | c1 + c3 × 2 wld × 7 k × 3 seed = 42 dirs(每个含 c1 + c3 两 cfg) | **`c3_cp_qps0.0.json` 文件 = Vanilla CB baseline**(D-014 重命名;chunk=8192 + Azure prompt cap=7000 → chunk 不触发 = mixed batch 真 vanilla CB);**`c1_baseline_qps0.0.json` = c1 phase-pure(D-014 后 paper 不主报)** | **paper Section 5 Vanilla CB baseline**(D-014);c1 数据保留 internal |
| **`c4_pd_supplement/`** | **T6 c4_pd 4-way (5/25)** | **c4_pd × 2 wld × 7 k × 3 seed = 42 cells**(1P1D, prefill@NPU0 TP=1 + decode@NPU1 TP=1 + proxy@8000) | **全谱 dominated**(TTFT 灾难,low load 472ms → high load 63,663ms);**TPOT 70ms 全员最好** → 1P1D static partitioning fundamental cost,详 `T6_FINDINGS.md` § c4_pd 4-way 补完 | paper 4-way 补完(D-011 commitment),c4_pd 紫线 |
| **`c4_pd_steady_smoke/`** | **c4_pd Poisson 对照 (5/25)** | c4_pd × conv × 3 QPS(2.85/5.70/7.98) × seed 0,arrival_mode=trace_sampled(Poisson IID) | **推翻 "burst 是 c4_pd 放大器" 假设**:steady 在 high load 反而比 burst TTFT 略差。证明 1P1D fundamental capacity 才是真凶,burst 仅 10-20% overhead | paper Section 5 caveat:c4_pd 在 2-NPU 数学上 capacity 不够,不是 burst-driven |
| **`c3_telemetry/`** | **c3 iter telemetry (5/25)** | c3-fair × code k=2.8/4.9 seed 0,scheduler.py passive 模式 patch 后 record_iter | **直接量化 mechanism #1**:c3 chunk-cap mixed iter mean 220ms vs m31 prefill iter 150ms(same 2048 budget,2.16 vs 31.5 reqs)→ **mixed-attention overhead +70ms/iter**;c3 median mixed iter 208ms vs m31 cycle 189ms(+10%) | paper Section 5 mechanism evidence(物理推理 → telemetry 直接证据) |
| `phase_2_post/m31fix_phase1_pointwise.json` | Phase 1 aggregate | 3-seed median,4 cfg × 2 wld × 3 SLO × 7 k | 主聚合数据(已含 c4_pd) | post-hoc 出图 / cite 数据用此 |
| `azure_main/` | P1.4 | 2 traces × 4 cfg × 3 seeds × 60s | 早期 strict SLO 主结果 | 历史(SLO 档已升级,T6 取代);保留作 reference |
| `azure_p15/` | P1.5 | 2 traces × 3 windows × 4 cfg × 3 seeds | 跨窗稳定性 + interference_coords | T2 defining figure 数据源(`interference_coords.json`) |
| `phase_1_t5_qps_sweep/` | Phase 1 T5 | 3-way × QPS{2..16} × {conv,code} × 3 seed × m31×4 SLO = 288 run | Poisson IID + 5× SLO 双重 dilute paradigm Δ | 不进主图,被 T6 burst 取代;保留作 sanity reference |
| `static_scan_3a/` | P0-1 | M1 static ratio ∈ {0.05..0.8} | regime-dependent 最优 ratio | thesis motivate:单一全局最优 ratio 不存在 |
| `tdm_trace/` | 基础设施 | 空目录,每次 run 重新填中间 jsonl | 不汇总 finding | 工具目录 |
| **`smoke_rpm_scale/`** | **MaaS smoke (6/30)** | pdtdm × 20 档 rpm_scale(0.02~0.40),Qwen3-0.6B,180s | **找 PD-TDM 在真实流量下的饱和边界** | 正式 MaaS 实验 rpm_scale 定值 |
| **`maas_replay/`** | **MaaS 系列 (7/1-7/7)** | v3: rs=0.12 失败 (68% err); v6: concur=64 稳定但 wall 8.4x; v7: 分布长度 -15% wall; **v11: rs=0.10 + subsample=30 + compress, pdtdm ✅ (3.3h, 0 err, 100% meet), sarathi running** | 真实 trace goodput 对比; v11 结果用于 paper MaaS 评估 |

---

## 3. 已废弃实验(2026-05-25 cleanup 已删除)

| 目录 / 类别 | 废弃理由 | 替代 |
|---|---|---|
| `phase_2_t6_burst_goodput/*_pid/`(168 m31 cells) | **chunked_schedule waiting-loop bug**;数据全部不可信(D-013) | `m31fix_validate/` |
| `azure_m31_fia_ablation/` | 带 bug + kernel ablation 已从 contribution 移除(D-009) | 若 future 需 kernel ablation,fix 后重跑 |
| `long_prompt_smoke/` + `long_prompt_smoke_fix/` | 5/23 bug 假象 smoke + 5/24 fix 验证;关键数字已落 D-013 | — |
| `long_prompt_sweep/` | 早期 long-prompt 探索 | — |
| `long_5way_q1632/` + log | 长 prompt 5-way 探索(基于错误前提) | — |
| `c3_chunk_smoke/` | 早期 c3 smoke | `c3_chunk2048_supplement/` |
| 早期 profile `exp_a-f` | 跟 PD-TDM 主线无直接关系 | `figures/fig{1-7}_exp_*.png` 保留 |
| M2.X PID 演化(`m2_slo_adaptive`、`m21-m27_slo_adaptive`) | 已被 azure_main 4-way 替代 | `azure_main/` |
| M3.1 早期(`m31_5way_*`、`m31_chunksize_scan` 等) | 单 seed 探索,已被替代 | `azure_main/` + `p1_chunk_scan/`(已删) |
| chunk 早期(`m1_chunk_seed*`、`long_chunk_seed*`) | 早期 smoke | `p1_chunk_scan/`(已删)/ `long_5way_q1632/`(已删) |
| qps sweep 早期(11 个变体) | 早期版本迭代 | `azure_main/` |
| smoke(`azure_smoke`、`m2_smoke`、`m3_smoke` 等) | 开发期 smoke | — |
| P1.6 sensitivity sweep(P1.6f/g/i) | sensitivity 价值降级 | D-004 |
| `baseline_c3_audit{,_aiv}` | Graph 模块降级证据,findings 已 fold | `FINDINGS.md` |
| `burst_seed{0,1,2}/` + `burst_slow_seed{0,1,2}/` | 合成 burst 撬不动 PID,negative finding | — |
| `azure_p16{a,b,e}/` + `azure_p17{,b}/` | M2/M3 演化中间数据 | 已 fold 进 D-007/D-008/D-009/D-010 |

---

## 4. 实验启动 / 数据落盘约定

### 启动方式(T6 burst goodput sweep,paper main)

```bash
cd /vllm-workspace/Ascend-PD-TDM/experiments
# c1/c3 nonpid(原 T6 driver)
bash run_phase_2_t6_burst_goodput.sh
# c3-fair chunk=2048 单独跑
bash run_c3_chunk2048_supplement.sh
# m31-fix(Phase 1 sweep,需要 fix 版 chunked_schedule.py)
bash /tmp/run_m31fix_phase1.sh
# c4_pd 4-way 补完(1P1D disagg sweep,支持断点续跑)
bash experiments/run_c4_pd_supplement.sh
# c4_pd Poisson 对照(3 cells,~12 min)
bash experiments/run_c4_pd_steady_smoke.sh
# c3 iter telemetry(2 cells,~10 min,需要 scheduler.py passive 模式 patch 已 in place)
bash experiments/run_c3_telemetry.sh
```

### 关键 config(`run_qps_sweep_all.py` 内定义)

| 内部 config 名 | Paper 名(D-014 后) | 含义 | server max_num_batched_tokens |
|---|---|---|---|
| `c1_baseline` | ~~(不进 paper)~~ | AscendScheduler unified(NPU-specific admit-driven phase-pure) | 8192 |
| `c2_tdm_m31_2048` (M3.1) | **PD-TDM (ours)** | PD-TDM phase-pure + prefill_chunk_tokens=2048 + PID | 8192(实际死代码) |
| `c3_cp` + chunk=2048 | **Sarathi chunked prefill** | vLLM 默认 chunked prefill + chunk=2048 = mixed batch + 触发 chunking | 2048 |
| `c3_cp` + chunk=8192 | **Vanilla CB** | vLLM chunked prefill 路径 + chunk=8192;Azure prompt cap=7000 < 8192 → chunk 不触发 = 真 mixed batch vanilla CB(D-014) | 8192 |
| `c4_pd` | c4_pd 1P1D disagg reference | PD 分离 1P1D | — |

**Aggregate JSON 列名**(`m31fix_phase1_pointwise.json` 内,2026-05-26 D-014 加 vanilla_cb 后):
| 列名 | Paper 名 |
|---|---|
| `vanilla_cb` | Vanilla CB(老 c3 chunk=8192) |
| `c3_cp` | Sarathi chunked prefill(c3-fair chunk=2048) |
| `c2_tdm_m31_2048_fix` | PD-TDM |
| `c4_pd` | c4_pd 1P1D disagg reference |
| `c1_baseline` | (D-014 后 paper 不主报,数据保留) |
| `c2_tdm_m31_2048_bug` | (D-013 已废弃 m31 PID bug 版,数据已删,列全 -) |

### Post-hoc

```bash
/usr/local/python3.11.13/bin/python3 experiments/posthoc_m31fix_phase1.py
# 输出 winning region + bug vs fix + per-cell 全 metric 详细
```

### 单测(不依赖 NPU)

```bash
cd /tmp && /usr/local/python3.11.13/bin/python3 -m vllm_ascend.core.tdm.tests._runner
```

---

## 5. 评估口径约定(D-012/D-013 finalize)

- **主指标**:max sustainable QPS at meet ≥ 90% under (TTFT_p99 < SLO_ttft) AND (TPOT_p99 < SLO_tpot)
- **次指标**:SLO meet% at fixed QPS;latency mean/p99
- **SLO grid**:Sarathi 风格 × {5, 10, 15, 25}× × micro-benchmark ideal(T6 实际用 {5, 10, 15}×,s4=25× 已弃)
- **passive_tracker 模式**:c1/c3 都走 TDMScheduler 但 `enable_tdm=False`,只跑 tracker → 测量口径完全一致
- **3 seeds median**(临界结果扩 5 seeds)
- **窗口**:T6 用 `[30s, 90s]`(warmup 30 + measurement 60)

---

## 6. 跑实验前的检查清单

1. 假设是什么?跑这个实验想验证哪句话?
2. PASS / FAIL 标准?数值条件
3. 不做什么?边界
4. 预期 wall 时间?预算
5. 数据落盘在 `results/<run_name>/`,新增一行到本台账
6. **跑 m31 前先确认 `chunked_schedule.py` 是 fix 版**(Diff #6 in place)

跑完回填:**结果** / **影响 finding** / **thesis 用途**。
