# SLO 控制器设计(慢回路 + 快回路)

> 慢回路 PID 已实装(M2.7),快回路 selector 待 P1.7b 实装。

---

## 1. 双回路定位

| 回路 | 模块 | 更新频率 | 输出 | 状态 |
|---|---|---|---|---|
| 慢回路 | `SLOReactiveController` | 每 8 iter(约 1s) | `ratio`(P:D 比例) + `chunk_tokens` | 已实装,M2.7 + M3.1 |
| 快回路 | `TokenBucketSelector` | 每 iter | `phase`(本 iter 走 P 还是 D) | 已实装但单向 urgency 失败(P1.7),P1.7b 双向待做 |

**协调:** 慢回路决定「这段时间 P:D 应该是什么比例」(预算),快回路在预算内决定「这个 iter 具体哪个 phase」+ 双向预警触发时绕过预算。

---

## 2. 慢回路(已实装)

### 2.1 输入信号

来自 `QueueSnapshot`(每 iter O(1) 读)+ `Telemetry.recent_requests`(请求级 SLO 历史滑窗):

| 字段 | 来源 | 用途 |
|---|---|---|
| `waiting_depth` | snapshot | prefill backlog,高 → 倾向 prefill |
| `running_depth` | snapshot | 正 decode 数 |
| `kv_free_ratio` | snapshot | KV pressure,低 → BoundaryGuard 触发 |
| `waiting_oldest_age_ms` | snapshot | 队头 TTFT 风险 |
| `ttft_ms / tpot_ms_mean` | RequestRecord 滑窗 | SLO 违例率 |

### 2.2 控制律(PID-lite + 一系列 guard)

```
err_ttft = ttft_violation_rate - target_violation_rate    # default 0.05
err_tpot = tpot_violation_rate - target_violation_rate
err_ttft = ReLU(err_ttft)    # M2.5:已满足的 SLO 不反推 ratio
err_tpot = ReLU(err_tpot)
if tpot_saturated:           # M2.4:viol > target 持续 → 物理饱和,屏蔽 tpot 信号
    err_tpot = 0

delta = kp * (err_ttft - err_tpot)
ratio = clamp(last_ratio + delta, ratio_min=0.05, ratio_max=0.80)
```

**关键超参:**

| 参数 | 默认 | 用途 |
|---|---|---|
| `kp` | 0.5 | PID 增益 |
| `ema_alpha` | 0.3 | EMA 平滑 |
| `deadband` | 0.02 | 误差死区 |
| `update_interval` | 8 | 每 N iter 重算 |
| `window_size` | 128 | 违例率滑窗大小 |
| `min_samples` | 16 | 冷启动门槛(< 16 用 `static_ratio=0.3`) |
| `target_violation_rate` | 0.05 | 允许 5% 违例 |
| `ratio_min / max` | 0.05 / 0.80 | clip 范围 |
| `slo_ttft_ms / slo_tpot_ms` | 500 / 50 | SLO 阈值 |
| `tpot_saturation_min_ticks` | 2(M2.7) | 进入饱和的连续 tick 数 |

### 2.3 实测修正(P1.6e 后)

**真实工作模式不是「连续闭环反馈」,而是「分段控制器」:**
- **启动期(5-10s):** PID 把 ratio 从 0.3 爬升到 ratio_max=0.8
- **饱和稳态期:** ratio 钉在 ratio_max 不动,M2.4 屏蔽 tpot 信号

**两类失败模式让双输入工作区极窄:**

| 模式 | 触发条件 |
|---|---|
| M2.4 屏蔽 | `viol > target` 持续(饱和)→ 屏蔽 tpot 信号防 ratio 拉地板 |
| ReLU clip | `viol < target` → clip 防余量推 ratio 反向 |
| 持续双输入 | `viol ≈ target` 稳定 → 极窄区,稳态系统几乎不可能稳定停在这里 |

**M2.X 演化里实测无效的 guard**(都在代码里保留但效果是 0):
- `starvation guard`(M2.1)
- `hysteresis release`(M2.2)
- `backlog kp_q`(M2.3 装饰,实测 kp_q=0)

**关键修复(M2.5 ReLU clip + M2.4 饱和检测):** 联合让 PID 在边界几乎必然单输入。这是 P1.6e 的 thesis 调整证据。

### 2.4 真实价值定位

| 任务 | 慢回路是否胜任 |
|---|---|
| 启动期把 ratio 爬升到合理值 | ✅ 主要价值 |
| 自动找到 `ratio_max` 边界(免人工 tune) | ✅ |
| 物理饱和检测(M2.4 屏蔽,避免误调度) | ✅ |
| 连续闭环反馈调 ratio | ❌(原 thesis claim,P1.6e 实证不成立) |
| 在 saturated 场景救 SLO | ❌(预算内全是 prefill,救不了 TPOT) |

### 2.5 慢回路角色重定位(D-010,2026-05-15)

慢回路在 thesis 里**不是 SLO 自适应主力**,只承担两件事:
- **启动爬升:** 把 ratio 从冷启动值推到 ratio_max
- **边界保护:** ratio_min / ratio_max clip + M2.4 物理饱和检测

真正的 SLO 自适应在**快回路(P1.7b,见 §3)** —— iter 粒度双向预警绕过慢回路屏蔽,在饱和场景仍提供反馈。

**ablation 计划:** M1 静态 ratio∈{0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80=ratio_max} 扫描 vs M3.1。两种结果都对 thesis 有用:
- 若 M1@ratio_max ≈ M3.1 → 慢回路自适应贡献=0,慢回路 = 单纯启动爬升 + 边界保护。论文按这条诚实写
- 若 M1@ratio_max << M3.1 → 慢回路有自适应贡献。但 P1.6e 数据(ratio 钉 max 80%+)倾向前者

这条扫描不依赖新代码,只是 config 改动 + 重跑,**可立即启动**。结果用来支撑慢回路的角色划分,避免论文写成「PID 在线 SLO 自适应」被审稿翻 P1.6e 戳穿。

---

## 3. 快回路(P1.7b 待实装)

### 3.1 为什么需要

**P1.7 单向 urgency 实验 FAIL**(`results/azure_p17/`):8/8 严档 7 个 Δ(M3.2 - M3.1) < 0。

**根因:** P1.7 只在 TTFT 紧迫时强切 prefill,没有反向(TPOT 沉默时强切 decode)。saturated 场景下 TPOT 端持续违例 → 单向 urgency 反而把 ratio 错偏。

**慢回路救不了 saturated:** 慢回路在边界单输入,即使 TPOT 紧也输出 `ratio = max`,不会让位 decode。

**思路:** 快回路在 iter 粒度看**原始**信号(不经过 ReLU / M2.4 屏蔽),做 TTFT 紧迫 + TPOT 沉默**双向预警**,触发时**绕过**慢回路给的预算。

### 3.2 设计候选(待 P1.7b 实装前敲定)

**双向预警:**
- TTFT 紧迫:`snap.waiting_oldest_age_ms > urgency_threshold_ms` → 本 iter 强切 prefill
- TPOT 沉默:`snap.decode_oldest_silence_ms > starvation_threshold_ms` → 本 iter 强切 decode
- `decode_oldest_silence_ms` 需要新加(见 `design/interfaces.md` 缺口 #1)

**优先级:**
- 没活儿(waiting + running 都空)→ 慢回路兜底
- 仅有 prefill / 仅有 decode → 自动决定
- TPOT 沉默触发 → 强切 decode(优先级最高,救饱和)
- TTFT 紧迫触发 → 强切 prefill
- 都不触发 → 按慢回路的 token bucket 走

**threshold 候选(待扫):**
- `urgency_threshold_ms ∈ {0.5 × ttft_budget, 0.7 × ttft_budget, 0.9 × ttft_budget}`
- `starvation_threshold_ms ∈ {2 × tpot_budget, 3 × tpot_budget, 5 × tpot_budget}`

### 3.3 开干前必须回答的设计问题

1. **绕过语义的精确边界**:快回路绕过慢回路的什么?是 token bucket 配额还是 phase 决定?
2. **不引入震荡**:快回路 vs 慢回路在「TTFT 紧 + TPOT 也紧」时的协调
3. **跟 BoundaryGuard(KV pressure)的优先级**
4. **是否需要冷却期**:快回路触发后是否锁 N iter,防止抖动

### 3.4 验证标准(P1.7b done 判定)

- code 严档 Δ(M3.3 - M3.1) ≥ 0(回正 P1.7 FAIL)
- conv 不退步,min > -1pp(临界 ±1pp 扩 5 seeds)
- 3 seeds ± std 描述性

### 3.5 前置接口缺口

`QueueSnapshot.decode_oldest_silence_ms` —— running 中最久未出 token 的 decode 距今多久。数据已在 `tracker._records._last_token_ts_ms`,只是没暴露到 selector 决策路径。1-2h 实装,见 `design/interfaces.md` 缺口 #1。

---

## 3b. phase-pure 机制级 ablation:`mixed_mode` 开关(D-010,2026-05-15)

> 论文 claim "P/D 时分复用(phase-pure)有用" 需要同 stack ablation 支撑,不能靠 vs C3 跨 stack 对比。

**对照设计:**
- TDM `mixed_mode=False`(默认,phase-pure):每个 iter 要么 pure-P 要么 pure-D
- TDM `mixed_mode=True`(新增):一个 iter 内 prefill + decode 同跑(类似 CP 风格,但走 TDMScheduler 底盘)
- 两组都用 chunk=2048 + M2.7 PID + 同 launch 参数,差异只在 phase 是否分段

**实装位置(初步):**
- `vllm-ascend/vllm_ascend/core/tdm/config.py`:`TDMConfig.mixed_mode: bool = False`
- `vllm-ascend/vllm_ascend/core/tdm/selector.py`:`TokenBucketSelector.peek()` 当 `mixed_mode=True` 时直接 return MIXED phase,绕过 P/D 二选一
- 还要核 scheduler.py 是否真的让 schedule() 在 MIXED phase 下混合 admit prefill + decode(可能需要补一小段逻辑)

**工程量评估:** 中等(几天)。比 P1.7b 简单,因为不涉及预警 / 协调机制,只是给 selector 加一个 short-circuit 分支。

**优先级:** 跟 P1.7b 并列 P0,但工程上可以串行(先做 mixed_mode 收获 claim A,再做 P1.7b 收获 claim B2/B3)。

---

## 4. 实装位置

| 文件 | 内容 |
|---|---|
| `vllm-ascend/vllm_ascend/core/tdm/controller.py` | `StaticRatioController`(M1)/ `SLOReactiveController`(M2.7)|
| `vllm-ascend/vllm_ascend/core/tdm/selector.py` | `TokenBucketSelector` peek/commit;P1.7b 在这里改 |
| `vllm-ascend/vllm_ascend/core/tdm/monitor.py` | `QueueMonitor.snapshot()`;P1.7b 前要加 `decode_oldest_silence_ms` |
| `vllm-ascend/vllm_ascend/core/tdm/types.py` | `QueueSnapshot` / `PhaseDecision`;P1.7b 前要扩 |
| `vllm-ascend/vllm_ascend/core/tdm/config.py` | `TDMConfig`,新增 starvation_threshold_ms 字段(P1.7b)+ `mixed_mode` 字段(D-010 ablation) |

---

## 5. 副作用 / 已知风险

| 风险 | 缓解 |
|---|---|
| Oscillation(PID kp 过大震荡) | 已有 EMA + deadband + update_interval=8 |
| Starvation 加剧 | HardConstraints `max_slice_iters` 兜底 + ratio_max=0.80 |
| 跟 BoundaryGuard 耦合(KV pressure 强切 decode 时 PID 还在加 ratio)| controller 入口检查 `kv_free_ratio < watermark + margin` → 不更新 ratio |
| tracker 数据延迟(F1+F2 修后已大幅缓解,见 P1.6b finding) | 慢回路本就是分钟级,可接受 |
| 冷启动:短 sweep(<60s)window 填不满 | duration ≥ 60s + qps ≥ 中等 |
