# 实验台账

> 每个 `results/` 子目录的用途 + 主 finding + thesis 用途。跑完新实验追加一行。

---

## 1. 实验代次时间线

| 代次 | 时间 | 内容 | 输出 |
|---|---|---|---|
| P0 | ~4 月 | exp_a-f 早期 profile(ACL Graph / P-D 资源画像 / DCMI 开销 / 并发扩展 / Unified vs Phased / Graph-aware) | `figures/fig{1-7}_exp_*.png`(数据已删) |
| P0-1 | 5/6 | static ratio scan | `static_scan_3a/` |
| P0-2 | 5/8 | 8K canonical M 家族 sweep | 数据已删,finding 进入 `FINDINGS.md` |
| P0-3 | 5/7 | 长 prompt SLO grid sweep(thesis 双 pivot 起点) | `long_prompt_sweep/` |
| P1.0 / P1.0b | 5/12 | Graph capture audit(FFTS+ vs AIV) | `baseline_c3_audit{,_aiv}/` |
| P1.4 | 5/11 | Azure trace 4-way 主结果 | `azure_main/` |
| P1.5 | 5/12 | 跨窗扩展 | `azure_p15/` |
| P1.6a | 5/13 | PID 屏蔽审计 + 信号滞后量化 | `p16a_audit_summary.txt` |
| P1.6b | 5/13 | 反馈链 F1+F2 修复 | `azure_p16b/`, `p16b_summary.txt` |
| P1.6e | 5/13 | SLO 档 sweep(4 档) | `azure_p16e/`, `p16e_summary.txt` |
| P1.7 | 5/12 | 单向 urgency 实验(FAIL) | `azure_p17/` |
| **P1.8** | 5/14 | m31_fia ablation:M3.1 强制走 FIA vs 原 m31_2048 vs C3,拆 kernel 贡献 vs phase-pure 贡献 | `azure_m31_fia_ablation/` |
| stationary 长 prompt | 5/11 | 5-way stationary(C1/M1+chunk/M27/M31/C3) | `long_5way_q1632/` |
| chunk pareto | 5/9 | chunk_tokens 静态扫参 | `p1_chunk_scan/` |
| burst 合成 | 5/8 | 合成 fast/slow burst | `burst_seed{0,1,2}/`, `burst_slow_seed{0,1,2}/` |
| **P1.6h** (pending) | — | calibrated SLO 重跑主结果 baseline | 推迟到 P1.7b 阶段合并跑(D-001) |
| **P1.7b** | 5/15 | 双向 selector smoke + threshold sweep(starv ∈ {2.0,3.0,5.0})| `azure_p17b/` — **触发率 0-1%,conv 上微赢不归因到机制,code 上 M3.3 default -2.84pp 输 M3.1。当前实装不能直接落地,待诊断** |
| **P1.9a** (pending,D-010) | — | M1 静态 ratio∈{0.05..ratio_max} 扫描 vs M3.1 | **可立即跑**,不依赖新代码。支撑慢回路角色重定位(claim B1)。两种结果都对 thesis 有用 |
| **P1.9b** (pending,D-010) | — | mixed_mode ablation:TDM mixed_mode on/off,同 stack 验证 phase-pure 机制级贡献 | 依赖 `TDMConfig.mixed_mode` 开关实装(interfaces.md 缺口 #5)。支撑 claim A |
| **P1.9c** (pending,D-010) | — | 双维度协同:M3.1(只慢)vs M3.1+P1.7b(慢+快)vs 只快回路 | 依赖 P1.7b 实装。支撑 claim B3(双维度协同) |

---

## 2. 主线 results/ 目录(17 个)

| 目录 | 代次 | 配置 / 规模 | 主 finding | thesis 用途 |
|---|---|---|---|---|
| `azure_main/` | P1.4 | 2 traces × 4 cfg × 3 seeds × 60s, strict SLO | M3.1 在 tpot=200ms 档 conv +5.2pp vs M1+chunk,code +9.4pp vs C3;但 conv tpot=150ms 档输 C3 4.5pp | **主结果(临时)**——SLO 档不符合「不能偏离 §2」,P1.7b 阶段用调过的档重跑 |
| `azure_p15/` | P1.5 | 2 traces × 3 windows × 4 cfg × 3 seeds | conv 4/4 tier 跨窗 Δ(M3.1, M1+chunk) PASS;code 严档 tpot=100/150 跨窗 Δ≈0 | 跨窗稳定性 ablation,motivate P1.7b 救饱和场景 |
| `azure_p16b/` | P1.6b | 1 cfg × 短 sweep,F1+F2 修后 | ttft warm 从 16.5s 降到 2.0s,但 tpot 信号一上线 100% 违例 | finding 来源(反馈链不是根因,SLO 档错才是),不进主表 |
| `azure_p16e/` | P1.6e | M3.1 × 4 SLO 档 × 2 windows × 3 seeds | conv M2.4 屏蔽 99.5% → 5.4% 跨 4 档,但 ratio 仍钉 max 80%+;**SLO 校准必要但不充分** | thesis 调整核心证据(支撑 D-002 / D-003) |
| `azure_p17/` | P1.7 | M3.2 单向 urgency vs M3.1 | 8/8 严档 7 个 Δ<0 | 反例,motivate P1.7b 双向 selector |
| `azure_m31_fia_ablation/` | P1.8 | 4 cfg × 2 traces × 3 seeds × 60s,新加 `c2_tdm_m31_fia`(M3.1 强制走 FIA) | kernel Δ ≈ 0(≤ 13ms / < 1%);M3.1 vs C3 差距 86-101% 是 **TDM-paradigm vs CP-paradigm bundle 总差**(D-010 修正:不归因到 phase-pure 单变量);phase-pure 双刃 — TTFT 赢但 code TPOT_p99 输 C3 992ms (2.5×) | thesis narrative 重写(D-009 + D-010):kernel 移出 contribution,phase Δ 改 paradigm-level 表述,phase-pure 单变量验证待 mixed_mode ablation(P1.9b) |
| `azure_p17b/` | P1.7b | 5 cfg(M3.1/M3.2/M3.3 default/starv2/starv5)× 2 traces × 3 seeds × 60s。M3.3 = M3.1 + 双向 urgency_ttft + starvation_tpot | starvation_tpot 触发率 0-1%(0% on code default);conv +1.86pp 但归因不实(starv5 0% 触发反而 +2.48pp);code default -2.84pp 净退步;source 字段 max_slice 25%→50% 暗示间接副作用 | **claim B2/B3 当前缺数据支撑**,待诊断三条方向(任务 #24/#25/#26) |
| `baseline_c3_audit/` | P1.0 | C3 default FFTS+ 15 sizes | C3 baseline 数据 | Graph 模块降级证据之一(D-005) |
| `baseline_c3_audit_aiv/` | P1.0b | C3 AIV 23 sizes vs FFTS+ 15 | 23 sizes 反而 -1.5~-23.8pp,capture 数量非单调 | 同上,**没有算法空间**结论的关键证据 |
| `static_scan_3a/` | P0-1 | M1 static ratio ∈ {0.05..0.8} | regime-dependent 最优 ratio:R2(qps=16)≈0.27,R3(qps=32)≈0.05 | thesis motivate:单一全局最优 ratio 不存在 |
| `long_5way_q1632/` | 5/11 | 5-way × q∈{16,32} × stationary 长 prompt | M3.1 vs M1 stationary +0.9/+0.3pp(弱) | negative finding:PID 在 stationary 不显效,thesis 写作要避谈 stationary 主胜 |
| `long_prompt_sweep/` | P0-3 | 4 configs × post-hoc SLO grid | strict/loose 都看不出差异;区分带 ttft∈[500,1500] × tpot∈[100,200] M2.7 一致胜 | **Sampling 模块的核心 motivation** + thesis 双 pivot 起点 |
| `p1_chunk_scan/` | 5/9 | chunk_tokens ∈ {512..8192} | chunk pareto 单调,chunk=2048 是 sweet spot | confirm 不做 dynamic chunk controller(D-006) |
| `burst_seed{0,1,2}/` | 5/8 | 合成 fast burst × 3 seeds | PID Δ ≈ ±0.7pp | negative finding:合成 burst 撬不动 PID,thesis 不报 |
| `burst_slow_seed{0,1,2}/` | 5/8 | 合成 slow burst × 3 seeds | 同上 | 同上 |
| `tdm_trace/` | 基础设施 | 空目录,每次 run 重新填中间 jsonl | 不汇总 finding | 工具目录 |

### 摘要文件
- `p16a_audit_summary.txt` — P1.6a 屏蔽审计的离线分析摘要
- `p16b_summary.txt` — P1.6b 反馈链修复摘要
- `p16e_summary.txt` — P1.6e SLO 档 sweep 摘要

### 各代次 orchestrator log
- `azure_p15_orchestrator.log` / `azure_p16b_orchestrator.log` / `azure_p16e_orchestrator.log` / `azure_p17_orchestrator.log`
- `long_5way_q1632.log` / `p1_chunk_scan_run.log`

---

## 3. 已废弃实验(不再跑,数据已删)

| 类别 | 名称 | 废弃理由 | 替代 |
|---|---|---|---|
| 早期 profile | exp_a-f(`results_exp_*.json`) | 跟 PD-TDM 主线无直接关系 | figures/fig{1-7}_exp_*.png 保留 |
| M2.X PID 演化 | `m2_slo_adaptive`、`m21-m25_slo_adaptive`、`m26_m27_slo_adaptive` | 已被 azure_main 4-way 替代;M2 各 guard 实测无效或装饰 | `azure_main/` + `FINDINGS.md` M2.X 节 |
| M3.1 早期 | `m31_5way_*`、`m31_chunksize_scan`、`m31_mixed_qps16`、`m31_samesweep_validation`、`mixed_workload_sweep`、`azure_l2_conv60s` | 早期单 seed 探索,已被 azure_main + p1_chunk_scan 替代 | `azure_main/` + `p1_chunk_scan/` |
| chunk 早期 | `m1_chunk_seed*`、`long_chunk_seed*` | 早期 smoke,被 p1_chunk_scan + long_5way_q1632 替代 | `p1_chunk_scan/` + `long_5way_q1632/` |
| qps sweep 早期 | `qps_sweep`、`_v2`、`_v3`、`_2k_y`、`_2k_y_c4`、`_8k`、`_8k_c3`、各 `_dryrun`、`_streamcheck`(11 个) | 早期版本迭代 | `azure_main/` |
| smoke | `azure_smoke`、`m2_smoke`、`m3_smoke`、`m2_y_plan_2k` | 开发期 smoke | — |
| P1.6 sensitivity sweep | P1.6f(target_violation)、P1.6g(ratio_max)、P1.6i(chunk on azure)| 方向已定 P1.7b,sensitivity 价值降级 | D-004 |

---

## 4. 实验启动 / 数据落盘约定

### 启动方式

```bash
# 跑 sweep(必须从 /tmp 启 vllm,避开 namespace 陷阱)
cd /vllm-workspace/Ascend-PD-TDM/experiments
/usr/local/python3.11.13/bin/python3 run_qps_sweep_all.py \
  --configs <config_list> \
  --qps 16,32 --duration 60 --warmup 20 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --outdir /vllm-workspace/Ascend-PD-TDM/results/<run_name>
```

### 关键 config

| 名字 | 含义 |
|---|---|
| `c1_baseline` | AscendScheduler 原版 + tracker(passive 模式) |
| `c2_tdm` (M1) | 静态 ratio=0.30 |
| `c2_tdm_m27` (M2.7) | SLO 反应式 PID + 全部 guard |
| `c2_tdm_m31_2048` (M3.1) | M2.7 + prefill_chunk_tokens=2048 |
| `c2_tdm_m31_fia` | M3.1 + `force_fia_attention=True`(monkey-patch 让 attention 走 FIA,P1.8 ablation 用,等价 vllm-ascend v0.13+) |
| `c3_cp` | vLLM 默认 chunked prefill(C3 对手) |
| `c4_pd` | PD 分离 1P1D(备用,2 卡) |

### 出图 / posthoc

```bash
/usr/bin/python3 experiments/posthoc_<name>.py ...
```

各 posthoc 脚本对应一次 sweep 或一次 finding 链节点。

### 单测(不依赖 NPU)

```bash
cd /tmp && /usr/local/python3.11.13/bin/python3 -m vllm_ascend.core.tdm.tests._runner
```

---

## 5. 评估口径约定

- **SLO 默认:** TTFT < 500ms, TPOT < 50ms(strict,**不用于主报**);conv 调过的档:tpot=200ms / ttft=1500ms
- **Goodput:** `status==200 ∧ ttft_ms<SLO_ttft ∧ tpot_ms_mean<SLO_tpot` 的 output_tokens 总和 / 稳态窗口
- **meet_slo%:** 命中 SLO 的请求数 / 窗口内总请求数
- **passive_tracker 模式:** 对照 config 都走 TDMScheduler 但 `enable_tdm=False`,只跑 tracker → 测量口径完全一致
- **3 seeds ± std 描述性;临界结果(±1pp 内)扩 5 seeds**
- **窗口:** `[20s, 60s]`(warmup 后)

---

## 6. 跑实验前的检查清单

1. 假设是什么?跑这个实验想验证哪句话?
2. PASS / FAIL 标准?数值条件
3. 不做什么?边界
4. 预期 wall 时间?预算
5. 数据落盘在 `results/<run_name>/`,新增一行到本台账

跑完回填:**结果** / **影响 finding** / **thesis 用途**。
