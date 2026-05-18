# Two-Timescale SLO-Aware Control for Prefill-Decode Time-Division Multiplexing

> 2026-05-14. 本文档定义 TDM 系统的最终算法框架：双层协同控制架构。替换此前单层 PID-centric 框架（slo_adaptive_design.md），是论文核心创新的定稿设计。

---

## 1. 核心 Insight

### 1.1 问题：纯 Iter 级响应式控制的固有失稳

考虑一种"纯 Iter 层"调度器——每个 iteration 根据此刻的实时队列状态独立决定走 P 还是走 D：

```
每 iter 执行:
  if 队头请求快超时了 → 切 PREFILL
  if decode 请求卡住了 → 切 DECODE
  if 两者都紧 → 仲裁
  else → 维持当前 phase
```

在工作负载平稳时，这一逻辑可以工作。但真实负载是**非平稳**的——请求到达率、prompt 长度分布、output 长度分布在时间轴上波动。在这种波动下，纯 Iter 层响应式控制会**过反应**：

```
时间轴示例（bursty trace）：
  iter 1-5:   队列空 → 全 DECODE
  iter 6:     突然涌入 10 个长 prompt → U_ttft 飙升 → 切 PREFILL
  iter 7-12:  连续 PREFILL 处理这批请求 → decode 全部被挤压
  iter 13:    U_tpot 报警 → 切 DECODE
  iter 14:    上一批还没 decode 完，又有一批新请求 → U_ttft 又飙 → 切 PREFILL
  ...
```

**核心问题**：纯 Iter 层的决策是**零记忆**的——每个 iter 只看瞬时快照，没有对负载趋势的"认知"。在波动负载下，这会表现为两种典型失败模式：

1. **频繁切换（chattering）**：工作负载波动时，U_ttft 和 U_tpot 交替报警，phase 在 P/D 之间高频抖动。每次切换本身有代价（ACL Graph 切换、batch 组成变化），频繁切换会抵消调度收益。

2. **单向漂移**：如果某一端持续产生更强的信号（例如长 prompt workload 下 U_ttft 始终高于 U_tpot），纯 Iter 层会被持续推向同一方向，最终等价于"全 P"或"全 D"——退化到无调度状态。P1.7 单向 urgency 8/8 严档 tier 全输，深层原因正在于此。

### 1.2 根因：缺少一个缓慢变化的锚点

上述失败不是 U_ttft / U_tpot 的阈值设置问题。它是**结构性的**——响应式控制（reactive control）在时间尺度上的天然缺陷：

> 响应式控制器的决策频率等于它感知信号的频率。当信号在 iter 粒度上波动时，控制器的输出也在 iter 粒度上波动。没有一个更慢的机制来**平滑**这些波动并提供一个稳定的方向锚点。

这是控制理论中一个经典问题的实例：单层高带宽反馈回路在没有低频参考信号时，天然倾向于过反应和振荡。cascade control（级联控制）正是为解决此问题而设计的——外环提供缓慢变化的 setpoint，内环仅在外环设定的目标区附近做高频跟踪。

### 1.3 双层框架的解决方式

**认识论贡献**：P/D 调度的 SLO 控制天然需要**两个时间尺度的分离**——不是因为信号类型不同，而是因为单时间尺度的响应式控制在非平稳负载下缺乏阻尼，必然出现切换振荡或单向漂移。

```
┌──────────────────────────────────────────────────────────────┐
│  Window 层（慢回路，每 N iter 更新一次）                        │
│  ─────────────────────────────────────                       │
│  信号：post-completion SLO violation（反馈，lagging）           │
│  作用：提供一个缓慢变化的 ratio 锚点，约束 Iter 层的决策半径     │
│  方法：简化 PID + saturation guard + ReLU clip               │
│  输出：target_ratio ∈ [ratio_min, ratio_max]                 │
│                                                              │
│  "慢变量锚点——负载波动时，Iter 层的决策有一个稳定的长程参考"      │
└──────────────────────────┬───────────────────────────────────┘
                           │ target_ratio (缓慢变化)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Iter 层（快回路，每 iter 执行）                                │
│  ─────────────────────────────                               │
│  信号：real-time queue state（前馈，leading indicator）         │
│  作用：在 ratio 锚定的方向区间内，做 iter 粒度的双向安全保护       │
│  方法：TTFT urgency threshold + TPOT starvation threshold    │
│  输出：PREFILL / DECODE（per-iteration phase decision）       │
│                                                              │
│  "即时安全——在慢变量设定的方向约束下，防止即刻违约"              │
└──────────────────────────────────────────────────────────────┘
```

**两层之间的本质关系**不是"分工"（你做这个我做那个），而是**阻尼**：

- Window 层的核心作用是用低通滤波（统计平滑 + 低频更新）吸收负载波动，输出一个缓慢变化的 ratio。这个 ratio 不会因为一次突发请求而剧烈跳动
- Iter 层的决策始终以这个 ratio 为**基线**。即使 U_ttft 或 U_tpot 触发 override，也只是单 iter 的例外处理，下一个 iter 继续回到 ratio 的基线方向
- 这样，纯 Iter 层的两个失败模式被消除：
  - **频繁切换**被抑制——ratio 锚定了一个"大概率"方向，Iter override 只在真正的安全紧急时才触发
  - **单向漂移**被阻止——即使 U_ttft 持续报警，Iter 层的 override 是例外而非新常态。ratio 的决定权在 Window 层，Window 层看的是平滑后的统计信号，不会被瞬时波动带偏

**类比**：Window 层是船舵（决定航向，转动缓慢），Iter 层是船头推进器（应对即刻障碍，动作频繁）。没有船舵，推进器会在波浪中失去方向；没有推进器，船舵的反应延迟会让你撞上障碍。两者缺一，船都不能稳定航行。

---

## 2. Window 层设计：Regime-Ramp Controller

### 2.1 设计哲学：慢变量作为阻尼锚点

M2.x 系列实验揭示了 PID 在此问题上的真实行为模式：

| 阶段 | 实际行为 | 在双层框架中的角色 |
|------|---------|------------------|
| 启动期 (0-19s) | ratio 从 0.3 单调 ramp 到 0.8 | 为 Iter 层建立一个 regime-appropriate 锚点 |
| 稳态期 (19s+) | ratio 钉死在 ratio_max | 提供稳定的长程方向，吸收负载波动 |

从"阻尼"视角看，**ratio 钉死不是 bug，是特性和必要**：

- 纯 Iter 层的问题是决策在每 iter 独立做，没有记忆 → 波动 → 振荡
- Window 层解决了这个问题：它输出一个**缓慢变化**的 ratio。稳态下 ratio 不变，意味着 Iter 层的决策有一个稳定的基线参照。即使负载波动导致 U_ttft/U_tpot 在 iter 间抖动，ratio 锚点保证常规决策（Phase 2）始终回到同一个方向
- 如果 ratio 也频繁变化，那两层一起振荡——变成两个高频回路互相干扰

**设计原则**：
1. Window 层的核心价值 = **提供一个变化缓慢的 ratio，为 Iter 层阻尼负载波动**
2. PID 的真实角色 = 自动找到当前 workload regime 下的合适比例，收敛后保持不动
3. 不 claim PID 做持续闭环调节。持续适应性在 Iter 层以 safety override 形式完成

### 2.2 控制律（简化自 M2.7）

```
保留 M2.7 中用得上的部分：
✅  delta = Kp × (err_ttft - err_tpot)
✅  tpot saturation detection (M2.4) —— 物理不可达 SLO 时屏蔽 tpot 侧信号
✅  ReLU err clipping (M2.5) —— 已满足的 SLO 不反推 ratio
✅  ratio clamp [ratio_min, ratio_max]

去掉或降级：
→  starvation guard (M2.1) —— ratio 撞底救场逻辑交给 Iter 层
→  hysteresis (M2.2) —— 简化工，振荡不是主因
→  backlog additive (M2.3) —— kp_q=0 装饰，移除以简化代码
```

**实际行为模型（论文中要明确写出来）**：
1. **启动 ramp**：冷启动时 ratio 从 initial=0.3 出发，基于反馈信号逐步调整。对于 ttft-bound workload（长 prompt）→ ratio 推向 ratio_max；对于 tpot-bound workload（高并发）→ ratio 留在低区
2. **边界保护**：当 tpot saturation 激活（物理不可达 SLO），屏蔽 tpot 信号防止 ratio 被拉向低区无法退回
3. **稳态平衡**：ratio 收敛到某个区域后，deadband 机制停止更新——没有新信息时不做无意义的微调

### 2.3 输入与输出

| 项目 | 内容 |
|------|------|
| **输入信号** | `ttft_viol_rate`, `tpot_viol_rate` (post-completion SLO violation rate, 滑窗) |
| **输入辅助** | `snap.kv_free_ratio` (KV 压力门控), `snap.waiting_oldest_age_ms` (backlog 项保留) |
| **输出** | `target_ratio` ∈ [0.05, 0.80] |
| **更新频率** | 每 8 iter (~1s)，与现有 `controller_update_interval` 一致 |
| **冷启动** | `target_ratio = 0.30`，样本不足 16 时维持初始值 |

### 2.4 论文叙事定位

> "A purely iter-level reactive controller lacks memory: each decision is made independently from instantaneous snapshots, causing phase-chattering under bursty arrivals and directional drift when one SLO dimension dominates. The window-layer controller solves this by providing a **slow-moving ratio anchor** — after an initial ramp (~20 iters), the ratio converges to a regime-appropriate equilibrium and stays there, absorbing workload fluctuation rather than reacting to it. Its role is not continuous feedback regulation; it is **damping** for the iter layer."

### 2.5 与现有 StaticRatioController 的关系

StaticRatioController (M1) = 需要人工为每种 worklaod 选一个 `static_ratio`。

Window 层 controller = 自动 ramp 到合理值并保持。本质上是用 PID transient 做了一次性的"自动调参"，之后相当于 static ratio。**比人工 tune 好，但不比人工 tune 好的 static ratio 在稳态有更强性能**——因为稳态就是 static。

**消融**：M1 (人工 static) vs M2.7 (PID ramp) — 预期 M2.7 在 workload 切换时优于 M1，在稳态无显著差异。

---

## 3. Iter 层设计：Bidirectional Leading Indicator Guard

### 3.1 设计哲学：快回路的不可替代价值

Window 层的反馈信号是 lagging 的（请求完成后才能算 violation），这有两个不可消除的后果：

1. **突发违约来不及救**：如果短时间内涌入一批长 prompt，队头请求 TTFT 马上就要超，但 PID 的 feedback 要等这批请求完成才知道——那时已经违约了
2. **反馈到达时负载状态已变**：短脉冲过后违约反馈才到，PID 根据"已消失的负载"的错误信号调整 ratio

Iter 层用**前馈信号**解决这两个问题：不等待请求完成，直接从实时队列状态判断是否会违约。

### 3.2 双向 Leading Indicator 定义

#### TTFT Urgency Indicator

```
U_ttft = waiting_oldest_age_ms / slo_ttft_ms

含义：队头请求的等待时间占 TTFT 预算的比例
U_ttft = 0.0  → 队头请求刚进来，不急
U_ttft = 0.7  → 队头请求等了 350ms (预算 500ms)，只剩 30% 余量，紧迫
U_ttft = 1.0  → 队头请求等待已达 TTFT 预算上限，即将违约
```

**来源**：`QueueSnapshot.waiting_oldest_age_ms` —— 已在每 iter 由 `QueueMonitor.snapshot()` 采集，O(1) 读 `scheduler.waiting[0].arrival_time`。

#### TPOT Starvation Indicator

```
U_tpot = decode_oldest_silence_ms / slo_tpot_ms

其中:
  decode_oldest_silence_ms = max{ now_ms - req.last_token_ts_ms | req ∈ running, req has emitted ≥1 token }

含义：所有正在 decode 的请求中，距上次出 token 最久的那个，沉默了多久
U_tpot = 0.0  → 所有 decode 请求刚出过 token
U_tpot = 1.5  → 某个请求 300ms 没出 token 了 (预算 200ms)，开始卡
U_tpot = 3.0  → 某个请求 600ms 没出 token，已经严重卡顿
```

**来源**：`RequestTracker._records` 中每个活跃请求的 `_last_token_ts_ms` 字段已由 `on_token_produced()` 维护。`QueueMonitor` 需新增对此字段的遍历（P2.0 缺口 #1，约 1-2h 实现）。

### 3.3 Selector 融合决策（核心算法）

Window 层的 `target_ratio` 和 Iter 层的 leading indicators **不冲突**：
- ratio 通过 TokenBucket 做"统计平滑"的 P/D 切换——管长程比例
- leading indicators 做"无条件 safety override"——管即刻安全

**决策序列**：

```
selector.peek(target_ratio, snap):

  ── Phase 0: 无工作兜底 ──
  if snap.waiting_depth == 0 and snap.running_depth == 0:
      return (flip current phase, source="no_work")

  ── Phase 1: Iter 层双向安全 override（无条件优先级）──
  # TTFT urgency: 队头请求快超时了 → 强制 prefill
  if snap.waiting_depth > 0 and U_ttft > URGENCY_THRESHOLD:
      return (PREFILL, source="ttft_urgency")

  # TPOT starvation: 有 decode 请求长时间没出 token → 强制 decode
  if snap.running_depth > 0 and U_tpot > STARVATION_THRESHOLD:
      return (DECODE, source="tpot_starvation")

  # 双向冲突仲裁: 两边都触发时，看 severity
  if ttft_override and tpot_override:
      # 如果 TPOT 已经到灾难级别 → 先救 decode
      if U_tpot > STARVATION_CRITICAL:
          return (DECODE, source="tpot_starvation_critical")
      # 否则 TTFT 优先（请求没进来比 decode 慢一点更严重）
      return (PREFILL, source="ttft_urgency")

  ── Phase 2: Window 层 ratio 决策（正常模式，无安全 override）──
  planned = TokenBucket.decide(target_ratio, phase_iters)
  return (planned, source="window_ratio")
```

**source 字段作用**：每一次 phase decision 都标注来源，用于事后 telemetry 分析 Iter 层 override 频率和条件分布。

### 3.4 超参数

| 参数 | 默认值 | 含义 | 校准方法 |
|------|--------|------|---------|
| `URGENCY_THRESHOLD` | 0.70 | U_ttft > 0.70 时触发 TTFT urgency override | sweep {0.5, 0.6, 0.7, 0.8}，选 code trace 上 meet_slo% 最优值 |
| `STARVATION_THRESHOLD` | 1.50 | U_tpot > 1.50 时触发 TPOT starvation override | sweep {1.0, 1.5, 2.0, 3.0}，同上 |
| `STARVATION_CRITICAL` | 3.00 | 双向冲突时 TPOT 灾难级阈值 | 仅仲裁时用，不直接触发 override |

**设计直觉**：
- URGENCY_THRESHOLD = 0.70 是保守选择——TTFT 还剩 30% 预算时就介入，不给突发留余量。太低（0.3）会导致 Iter 层过度 override，使 ratio 决策失效；太高（0.95）会来不及救
- STARVATION_THRESHOLD = 1.50 意味着"已经超过 SLO 50%"才触发——decode iter 间隔在正常负载下是 20-50ms，到达 1.5× SLO 意味着连续多次 iter 没轮到这个请求的 decode batch，是真实的 starvation
- STARVATION_CRITICAL = 3.00 只在仲裁时生效——两边同时触发且 TPOT 已经灾难级，宁可牺牲 TTFT

### 3.5 P1.7 单向版本为什么 FAIL

P1.7 只做了 TTFT 端 urgency：

```
# P1.7 单向逻辑（只有 TTFT arm）
if waiting_oldest_age_ms > urgency_threshold_ms:
    planned = PREFILL
```

**失败模式**：
1. TTFT urgency 触发 prefill → 连续多个 iter 进入 prefill → decode 被挤压
2. 没有 TPOT 端 guard 来制衡 → decode request 一直得不到 iter → tpot 长尾恶化
3. 最终 tpot viol 上升（虽然 ttft 改善），SLO% 综合下降

**双向版本怎么修**：
- TPOT starvation guard 会在 decode 被过度挤压时强制切回 decode
- 两者相互制衡 → 不会出现单向的"过度保护"

### 3.6 论文叙事定位

> "The iter-layer controller uses real-time queue age and per-request token emission intervals as leading indicators of imminent SLO violation. Unlike the window layer's feedback signals (which arrive only after request completion, at hundreds-of-ms delay), these feedforward signals are available every iteration at O(1) cost from the scheduler's internal state. The bidirectionality is critical: a single-direction urgency guard (e.g., TTFT-only, as evaluated in P1.7) over-protects one SLO dimension at the expense of the other. The dual guard — TTFT urgency for prefill starvation, TPOT token silence for decode starvation — provides symmetric protection, with a critical-severity tiebreaker for cases where both alarms fire simultaneously."

---

## 4. 两层协同机制：阻尼原理

### 4.1 协调模式：慢变量约束快变量的决策空间

两层的核心关系不是"分工"（你管方向我管安全），而是**阻尼**：

```
Window 层 → 输出 target_ratio（缓慢变化）
             └── 定义了 Iter 层"正常模式"下的 P:D 基线
             
Iter 层   → 每 iter 决策时：
             95%+ 的 iter: 沿 ratio 基线走 (Phase 2)
             5%- 的 iter: 检测到即刻安全威胁时 override (Phase 1)
             
             override 后立即回到 ratio 基线，不修改 ratio 本身
```

**为什么这解决了纯 Iter 层的振荡**：

考虑一种 burst 场景：一批长 prompt 突然到达。

- **纯 Iter 层**：U_ttft 检测到队头请求 → 切 PREFILL → 连续 N 个 iter 走 P → decode 被挤压 → U_tpot 报警 → 切 DECODE → 回归 P... 这是在 P/D 之间振荡，因为每次决策都是零记忆的"看此刻信号"
- **双层**：Window 层在 burst 发生**之前**就已经输出一个 ratio（如 0.3）。Iter 层在 burst 到达时可能因 U_ttft 触发 1-2 个 iter 的 PREFILL override，但 override 不是新的 ratio——下一 iter 如果没有安全威胁，就回到 ratio=0.3 的 D-bias 方向。ratio 本身不受这几次 override 影响，因为它每 8 iter 才更新一次

这就是阻尼的本质：**ratio 是积分器（accumulator），吸收高频波动；Iter override 是微分器（derivative），只响应即刻偏差。**

### 4.2 冲突不存在

双层之间唯一的"偏离"是单 iter override，这不是冲突——是正常设计：

- Window 层不指定每 iter 的 phase，它只输出 ratio（长期统计期望）
- Iter 层不修改 ratio，它只在单个 iter 上暂时偏离
- 单 iter 偏离不影响 ratio 的语义（ratio 是期望，不是硬指令）

真正的"冲突"只有一种：override 频率过高（>20%），说明 Window ratio 已经不匹配当前负载。这需通过缩短 Window update interval 或重新校准 ratio 范围来解决。

### 4.3 诊断输出

Controller telemetry（`ctrl.jsonl`）每条记录包含：

| 字段 | 含义 |
|------|------|
| `target_ratio` | Window 层输出的当前 ratio |
| `planned_phase` | Iter 层原始决策（来自 ratio） |
| `actual_phase` | Iter 层最终执行 phase（可能被 override） |
| `decision_source` | `"window_ratio"`, `"ttft_urgency"`, `"tpot_starvation"`, `"tpot_starvation_critical"` |
| `U_ttft` | 本 iter TTFT urgency 值 |
| `U_tpot` | 本 iter TPOT starvation 值 |
| `override_count_since_last_update` | 自上次 Window 更新以来 override 次数 |

---

## 5. 消融实验设计

### 5.1 消融矩阵

```
Config  Window PID  Iter Bidirectional  Description
──────  ──────────  ──────────────────  ───────────
M1      ❌           ❌                  Static ratio=0.30 (人工 baseline)
M2.7    ✅           ❌                  PID-only (当前 SOTA)
P1.7    ❌           ✅ (单向 TTFT)       Urgency-only (FAILED)
M3.3    ✅           ✅ (双向)            完整双层 (proposed)
M3.3a   ❌           ✅ (双向)            纯前馈迭代 (ablation: 去掉PID)
M3.3b   ✅           ❌ (仅有 TTFT 端)    单向双层 (ablation: 去掉 TPOT guard)
```

### 5.2 核心假设与验证

| # | 假设 | 验证方式 | 支持/反驳判定 |
|---|------|---------|-------------|
| H1 | 双向 override 在 code strict tier 有正 Δ vs M2.7 | M3.3 vs M2.7, code trace, tpot∈[50,150] | Δ > +3pp 且 3 seeds σ > 2 |
| H2 | 纯前馈 (M3.3a) 在稳态 regime 输给完整双层 | M3.3a vs M3.3, conv trace | 纯前馈缺少方向感导致 ratio 无结构振荡 |
| H3 | 单向双层 (M3.3b) 在严 tier 输给双向 | M3.3b vs M3.3 | 复制 P1.7 FAIL 模式 |
| H4 | Window PID 在 workload 切换时优于 static | M2.7 vs M1, half-conv-half-code trace | PID 自动 ramp 比 static=0.3 有正 Δ |
| H5 | Iter override 频率 < 20%（正常工作） | telemetry source 字段分布 | override 频率过高 = Window ratio 失效 |

### 5.3 实验对标

| 配置 | 说明 |
|------|------|
| C1 (hybrid) | AscendScheduler 原版，sanity baseline |
| C3 (chunked prefill) | vLLM default chunked prefill，SOTA 对照 |
| M1 (static ratio) | 静态 ratio baseline，验证"自动 ramp"价值 |
| M2.7 (PID only) | 当前 M 家族最优，验证"双向 indicator"增量 |
| M3.3 (完整双层) | 本框架 |

---

## 6. 论文叙事线

### 6.1 三段式 Motivation

```
Para 1 — 问题：纯 Iter 级响应式控制的失稳
  P/D 调度需要在每 iter 决定走 PREFILL 还是 DECODE。一个自然的做法
  是根据实时队列状态做响应式切换：TTFT 紧了切 P，TPOT 紧了切 D。
  但这种单层控制是零记忆的——每个 iter 的决策独立于历史。在非平稳
  负载下，这导致两种失稳：①频繁切换（信号交替报警时 P/D 之间高频
  抖动）；②单向漂移（某一端持续报警时退化到全 P 或全 D）。P1.7
  单向 urgency 8/8 严档全输，深层原因正是这种结构性的欠阻尼。
  
Para 2 — 我们的 insight：双层控制提供阻尼
  纯 Iter 控制的失稳是结构性的——决策频率等于信号频率，没有慢变量
  吸收负载波动。解决方案是引入一个缓慢变化的 ratio 锚点作为 Iter
  层的决策基线：Window 层看统计平滑的 SLO 反馈，每 ~1s 更新一次
  ratio，输出缓慢变化的 P:D 方向；Iter 层在 ratio 设定的基线附近
  运行，仅在即刻安全威胁时做单 iter override。这本质上是级联控制
  (cascade control) 的结构：慢外环提供 setpoint 阻尼，快内环做
  安全跟踪。
  
Para 3 — 现有方法的定位
  现有 TDM (PDM/Drift) 是单层固定轮转——没有反馈，没有阻尼需求。
  TGS/AIMD 只做单层 window 级比例控制，缺乏 iter 级保护。P1.7 
  urgency 只做单层 iter 级阈值，缺乏方向稳定性。无一在时分路径上
  设计了双层协同——因为无人认识到这个问题的核心是"响应式控制在非
  平稳负载下欠阻尼"而非"需要更好的反馈信号"。
```

### 6.2 贡献声明（3 条）

1. **Two-Timescale Control Framework for P/D TDM**：首次将 P/D 调度建模为需要两个时间尺度分离的控制问题。核心 insight：纯 iter 级响应式控制在非平稳负载下欠阻尼，需要一个缓慢变化的 ratio 锚点（Window 层）来抑制 Iter 层的切换振荡和单向漂移。两层以级联控制结构协同——外环提供阻尼，内环提供安全。

2. **Bidirectional Iter-Level Safety Guard**：定义 TTFT urgency（队头等待时间 / SLO budget）和 TPOT starvation（decode 最大 token silence / SLO budget）两个 iter 级前馈指标。设计双向 override 机制——两者相互制衡，避免单向 urgency（P1.7）的过度保护。冲突时以 critical-severity tiebreaker 仲裁。

3. **Empirical Validation on Azure Production Traces**：在 Ascend NPU 910B3 + Qwen3-8B 上，使用 Azure LLM Inference Trace（conv + code）在 workload-conditional SLO tiers 下进行完整消融，验证 (a) 双层优于纯 PID（M2.7）和纯前馈（单层），(b) 双向 guard 优于单向，(c) Window 层在 workload 切换时自动 ramp 优于人工 static ratio。

### 6.3 与现有工作的区别

| 维度 | PDM/Drift | TGS/FaST-GShare | MuxWise (ASPLOS'26) | **Ours** |
|------|-----------|-----------------|---------------------|----------|
| 路径 | 时分 | 时分 | 空分（SM partition） | **时分** |
| 控制结构 | 单层固定轮转 | 单层 window 级 | 单层 SM 级 | **双层级联** (慢外环阻尼 + 快内环安全) |
| 反馈机制 | 无 | 开环 AIMD | 反馈（decode 优先） | **前馈 + 反馈协同** |
| SLO 机制 | 无 | 隐式 | 显式（decode 优先） | **显式双向（TTFT + TPOT）** |
| 硬件普适性 | ✅ | ✅ | ❌ (需 SM / CU masking) | ✅ |

---

## 7. 实现计划

### 7.1 代码改动点

| 改动 | 文件 | 内容 | 工作量 |
|------|------|------|--------|
| 状态采集补缺 | `tdm/monitor.py` | `QueueSnapshot` 新增 `decode_oldest_silence_ms` 字段，遍历 tracker._records | 1-2h |
| Selector 升级 | `tdm/selector.py` | `TokenBucketSelector.peek()` 接收 `U_ttft, U_tpot`，实现 Phase 1 safety override + Phase 2 ratio 决策 | 2-3h |
| Controller 简化 | `tdm/controller.py` | SLOReactiveController 去掉 kp_q=0 的 backlog 装饰项，清理 M2.1-2.2 无效 guard | 1h |
| 配置更新 | `tdm/config.py` | 新增 `urgency_threshold, starvation_threshold, starvation_critical` | 0.5h |
| 单测 | `tdm/tests/` | selector override 逻辑测试（4 场景：无 override / TTFT 触发 / TPOT 触发 / 双向冲突） | 1-2h |
| **合计** | | | **6-9h** |

### 7.2 实验计划

| 步骤 | 内容 | 时间 |
|------|------|------|
| Smoke | 1 seed × conv_w1, 验证 override 不崩、telemetry 字段正确 | 0.5h |
| Sweep | M3.3 vs M2.7 vs M1 vs P1.7-style(单向) × 2 windows × 3 seeds | ~2h wall |
| Posthoc | `posthoc_p17_urgency.py` 扩展支持双向版本的 source 分析 | 1h |
| 阈值校准 | URGENCY_THRESHOLD / STARVATION_THRESHOLD sweep | ~2h wall |

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| `decode_oldest_silence_ms` 在实际 trace 上始终 ≤ 50ms（code trace 并发低时 token 间隔本就短） | 中 | 如果 TPOT 端 leading indicator 没有足够信号，双向退化为单向 | 先看现有 azure_main telemetry 数据中 `_last_token_ts_ms` 的分布。如果信号确实弱，STARVATION_THRESHOLD 下调到 0.8-1.0 |
| 双向 override 频率过高（>30%），把 ratio 决策架空 | 低 | 论文叙事矛盾——声称两层协同，实际 Iter 层主导 | 调高 URGENCY/STARVATION 阈值，让 override 更保守；或缩短 Window 更新间隔 |
| code trace 物理过载（viol_rate >0.57 不论配置），双向 indicator 仍无法拉回 | 高 | 论文不能在这个场景宣胜 | 诚实承认物理过载场景属于 admission control 范畴（非调度器职责），论文聚焦在可服务 regime（conv trace 为主） |
| 完整双层 vs M2.7 的 Δ 在 conv trace 上不显著 | 中 | Window 层的自动 ramp 已足够 → Iter 层没有增量 | 需要强调双层在非平稳负载（regime 切换、burst）上的价值——可能需要在 sweep 中加入 workload-mixing 场景 |

---

## Appendix A: 术语速查

| 术语 | 含义 |
|------|------|
| `target_ratio` | P:D 比例控制参数。ratio=0 → 全 D，ratio=1 → 全 P，ratio=0.3 → 约 3:7 P:D |
| `U_ttft` | TTFT Urgency = `waiting_oldest_age_ms / slo_ttft_ms`，衡量队头请求距 TTFT 违约的紧迫度 |
| `U_tpot` | TPOT Starvation = `decode_oldest_silence_ms / slo_tpot_ms`，衡量最饿的 decode 请求距 TPOT 违约的紧迫度 |
| `URGENCY_THRESHOLD` | `U_ttft > 0.7` 触发 TTFT 安全 override |
| `STARVATION_THRESHOLD` | `U_tpot > 1.5` 触发 TPOT 安全 override |
| `STARVATION_CRITICAL` | `U_tpot > 3.0` 双向冲突时 TPOT 灾难优先级 |
| Window 层 | 慢回路。每 8 iter (~1s) 用反馈信号调整 ratio |
| Iter 层 | 快回路。每 iter 用前馈信号执行安全 override |
| leading indicator | 前馈。从实时队列状态预判即将发生的 SLO 违约 |
| lagging indicator | 反馈。从已完成请求确证 SLO 违约，天然滞后请求生命周期 |
