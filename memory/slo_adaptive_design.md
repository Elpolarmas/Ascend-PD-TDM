# SLO-Adaptive Controller 设计文档（M2 模块）

> 写于 2026-05-01。Plan A 期间产出，待用户回来 review 后再写代码。
> 本文档**只描述设计**，不写实现。代码落地在 review 通过后另起一轮。

## 目标

把 TDM 的 prefill/decode 切片比例从「静态常数 `static_ratio=0.3`」升级为「按在线 SLO 违例率反馈调整的闭环控制」。

论文叙事位置：M2 = M1 (basic TDM) + SLO-adaptive ratio。期望相对 M1 在中-高 QPS 区段把 SLO meet 率拉回到 ≥ C1 (hybrid) 水平，同时保持 M1 的尾延迟优势。

非目标（留给 M3 / M4）：
- graph-aware batch sizing（phase=decode 时按 `cudagraph_capture_sizes` 选档位装 batch，命中 ACL FULL graph）→ M3
- 监控反馈闭环（passive_tracker → controller 走集中 telemetry bus）→ M4，本控制器仅消费 in-process Telemetry，不出 socket

---

## 1. 输入信号清单

控制器在每个 `get_target_ratio(snap, iter_id)` tick 上能拿到的信号分两类：

### 1a. 实时 Queue 状态（每 iter 同步可读）

来自 `QueueSnapshot`（`tdm/types.py`），由 `QueueMonitor.snapshot()` 在 `schedule()` 入口现采，O(1) 读：

| 字段 | 类型 | 含义 | 用途 |
|---|---|---|---|
| `waiting_depth` | int | 等待 admission 的请求数（prefill backlog） | 高 → 倾向 prefill |
| `running_depth` | int | 正在 decode 的请求数 | 高 → 已经在产 token，可适度让位 prefill |
| `finished_prefill_depth` | int | 已 prefill 完待 decode 的请求 | 高 → decode 待消化，prefill 可缓 |
| `kv_free_ratio` | float | KV cache 空闲比例 | 低 → BoundaryGuard 已会强切 decode |
| `waiting_oldest_age_ms` | float | 队头请求等了多久 | 高 → TTFT SLO 风险 |

### 1b. Per-Request SLO 历史（滑窗，仅已完成请求）

来自 `Telemetry.recent_requests(n)`，返回 `RequestRecord` deque（默认 cap 256）：

| 字段 | 类型 | 计算方式 |
|---|---|---|
| `ttft_ms` | float | `first_token_ts_ms - admission_ts_ms` |
| `tpot_ms_mean` | float | mean of `decode_intervals_ms` |
| `tpot_ms_p99` | float | p99 of `decode_intervals_ms` |
| `prompt_tokens` | int | 输入长度 |
| `output_tokens` | int | 输出长度 |
| `finish_ts_ms` | float | 完成时刻（用来按时间窗筛） |

**关键约束**：`record_request()` 仅在 `on_finish()` 被调用——也就是说控制器看到的 SLO 信号有「请求 lifetime」量级的延迟（Qwen3-8B 典型 e2e 几百 ms 到几秒）。短脉冲过载在请求完成前不会反映到这里，必须用 1a 的实时 queue 状态做前馈，1b 仅做反馈。

### 1c. 不打算用的信号（明确排除）

- `Telemetry.recent_iters(n)` 里的 `iter_duration_ms` / `phase_iters`——这些是 actuation history，控制器自己产生的，再喂回去会形成 bias。
- vllm 主仓的 prometheus metrics——本 controller 完全 in-process，不接 vllm metrics bus（M4 才接）。

---

## 2. 状态变量

控制器内部维护的滑动统计（每 `update_interval=8` iter 刷新一次，与现有 `StaticRatioController` 节奏一致，避免高频抖动）：

| 状态 | 数据结构 | 维护方式 |
|---|---|---|
| `ttft_window` | `deque[float]`，cap=128 | 每 update tick 把 `recent_requests(n)` 中**新出现**的 ttft 追加 |
| `tpot_window` | `deque[float]`，cap=128 | 同上 |
| `last_seen_req_ids` | `set[str]`，cap=512 | 防止重复计入；用 `RequestRecord.request_id` 去重 |
| `prefill_backlog_window` | `deque[int]`，cap=32 | 每 tick append `snap.waiting_depth` |
| `last_ratio` | float | 上 tick 输出，给增量律用 |
| `iters_since_last_update` | int | 控制更新节奏 |

派生量（每 update tick 计算）：

| 量 | 公式 |
|---|---|
| `ttft_violation_rate` | `count(ttft > slo_ttft_ms) / len(ttft_window)`，window 空 → 0 |
| `tpot_violation_rate` | 同上，针对 `tpot_ms_mean`（mean 而非 p99，与 metrics.py 的 SLO 定义对齐） |
| `prefill_pressure` | `mean(prefill_backlog_window)` 标准化到 `[0,1]`，分母用 `max_num_running_reqs` |

**冷启动**：window 样本数 < `min_samples=16` 时退化为 `static_ratio`（默认值 0.3），避免 0/0。

---

## 3. 控制律候选

三选一，倾向 **C1 (PID-lite)** 作为首选，理由见每条尾段。

### Option A — PID-lite on `static_ratio`

```
err_ttft = ttft_violation_rate - target_violation_rate     # default target=0.05
err_tpot = tpot_violation_rate - target_violation_rate
err = max(err_ttft, err_tpot)                              # 取较违的那一边

# 注意符号：err > 0 即 SLO 违例率太高
#   - 若 ttft 违例占主：要更多 prefill → ratio↑
#   - 若 tpot 违例占主：要更多 decode → ratio↓
# 所以分维度更新：
delta = Kp * (err_ttft - err_tpot)
ratio = clamp(last_ratio + delta, ratio_min=0.05, ratio_max=0.8)
```

**优点**：单旋钮、状态量小、行为可解释、收敛性容易论证。
**缺点**：Kp 要调，过大震荡；对突发 spike 反应慢（请求完成才看得到违例）。
**默认增益**：`Kp=0.05`，意味着「违例率每差 100% 调 5pp ratio」，配合 `update_interval=8` 大约 1s 一次更新。

> ⭐ **首选**。简单，稳，论文里好画 attractor 图（ratio 随违例率收敛到平衡点）。

### Option B — Bang-bang

```
if ttft_violation_rate > 0.10:   ratio = 0.6   # prefill-bias
elif tpot_violation_rate > 0.10: ratio = 0.15  # decode-bias
else:                             ratio = 0.3   # neutral
```

**优点**：实现极简，决策边界清晰。
**缺点**：在阈值附近震荡（典型 chatter problem），需要加 hysteresis；论文不好讲（容易被审稿人贴「ad-hoc」标签）。
**适用场景**：作为 ablation 用——证明 PID 比 bang-bang 好。

### Option C — Lookup table (`backlog × phase → ratio`)

```
table = {
    (low_backlog,  prefill): 0.2,
    (low_backlog,  decode):  0.2,
    (mid_backlog,  prefill): 0.4,
    (mid_backlog,  decode):  0.5,
    (high_backlog, prefill): 0.6,
    (high_backlog, decode):  0.7,
}
```

**优点**：无在线状态，纯查表；可由离线 sweep 数据校准。
**缺点**：跟 SLO 信号脱钩（仅看 backlog），不真闭环；要在论文里讲清「这是开环优化版 static」而非「SLO-adaptive」。
**适用场景**：作 baseline 对比，不作为主方案。

---

## 4. 输出空间

控制器可以调三个旋钮，本设计**仅动 `static_ratio`**：

| 旋钮 | 作用 | 本设计动否 | 理由 |
|---|---|---|---|
| `static_ratio` | TokenBucketSelector 的 prefill 信用速率 | ✅ | 主要执行器，最直接 |
| `min_slice_iters` | HardConstraints debounce 下限 | ❌ | 切片粒度跟 graph 档位/SLO 都不在同一时间尺度 |
| `max_slice_iters` | HardConstraints 切换上限 | ❌ | 同上 |
| decode batch 装多少 reqs | 命中 ACL FULL graph 档位 | ❌ | 这是 M3 (graph-aware batch sizing) 的事，前馈模块 |

**为什么不动 slice_iters / batch 大小**：把三个维度（ratio / slice 粒度 / batch 装填）分开能让 ablation 干净——M2 单独跑 SLO-adaptive ratio，M3 单独跑 graph-aware batch sizing，M2+M3 看叠加增益。如果 M2 同时动多个，实验上撇不清贡献。

**M3 (graph-aware batch sizing) 的形态——澄清**：

之前误写成"按编译期 prefill graph cost 调 min/max_slice_iters"。实际上 vllm-ascend 的 ACL Graph 是**对 decode 阶段**做 NPUGraph capture 的（`cudagraph_mode=FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE`），运行时按 `BatchDescriptor(num_tokens, uniform_decode)` 查表，要求 batch 全是 decode 才能命中 FULL graph（`vllm_ascend/worker/model_runner_v1.py:1920` 的 `uniform_decode` 判定）。

预设 size 集合在 `compilation_config.cudagraph_capture_sizes`，`utils.py:322 update_aclgraph_sizes` 受单 NPU 1800 graph 上限约束，会从原 sizes 里均匀采样保留代表性子集。

所以 M3 真正该做的事是：当 phase=decode 时，从 `cudagraph_capture_sizes` 里挑一个最贴近当前 running queue 可装 reqs 数的档位，避免 padding 浪费。模块**纯前馈，不反向影响 selector**。phase=prefill 时 graph 用不上，M3 不介入。

**TDM 切片 vs chunked prefill 在 graph 模式下的硬件级差异**（值得入论文）：
- C3 (chunked prefill) 永远 mixed batch（partial prefill + decode 同 iter）→ 永远 `uniform_decode=False` → 永远命中不了 FULL graph
- C2 (TDM) decode phase 是纯 decode batch → uniform_decode=True → 命中 FULL graph 加速

这是 Y 方案 c3 全面输 c1/c2 的**底层硬件解释**之一（之前只归因于 partial prefill kernel 路径未优化）。

**ratio 的物理含义**：每 iter 给 prefill 桶充 `ratio` 个 token；桶满 1 个 token 就允许一次 prefill peek。`ratio=0` → 永不主动 prefill（仅 BoundaryGuard 反向兜底）；`ratio=1` → 每 iter 都尝试 prefill；`ratio=0.5` → 平均每 2 iter 一次 prefill。

---

## 5. 集成接口

### 5a. 替换点

`tdm/scheduler.py:71-76`，把：
```python
self._tdm_controller = StaticRatioController(
    static_ratio=cfg.static_ratio,
    update_interval=cfg.controller_update_interval,
    telemetry=self._tdm_telemetry,
)
```
换成：
```python
if cfg.disable_controller or cfg.controller_kind == "static":
    self._tdm_controller = StaticRatioController(...)
elif cfg.controller_kind == "slo_pid":
    self._tdm_controller = SLOReactiveController(
        target_violation_rate=cfg.slo_target_violation_rate,
        kp=cfg.slo_pid_kp,
        ratio_min=0.05, ratio_max=0.8,
        slo_ttft_ms=cfg.slo_ttft_ms,
        slo_tpot_ms=cfg.slo_tpot_ms,
        update_interval=cfg.controller_update_interval,
        telemetry=self._tdm_telemetry,
        initial_ratio=cfg.static_ratio,
    )
```

调用入口不变：`scheduler.py:94` 仍然是 `self._tdm_controller.get_target_ratio(snap, iter_id)`，签名兼容。

### 5b. 新 TDMConfig 字段

加到 `tdm/config.py`：
```python
controller_kind: str = "static"          # "static" | "slo_pid" | "bang_bang" | "lookup"
slo_target_violation_rate: float = 0.05  # PID 目标违例率
slo_pid_kp: float = 0.05                 # PID 增益
```
保留 `slo_ttft_ms / slo_tpot_ms`，已有字段。

### 5c. SLOReactiveController 接口骨架

```python
class SLOReactiveController:
    def __init__(self, *, target_violation_rate, kp, ratio_min, ratio_max,
                 slo_ttft_ms, slo_tpot_ms, update_interval, telemetry,
                 initial_ratio):
        ...

    def get_target_ratio(self, snap: QueueSnapshot, iter_id: int) -> float:
        # 1. 节流：iter_id - last_iter_id < update_interval → 返回 cached
        # 2. 从 telemetry.recent_requests(N) 增量提取新完成请求的 ttft/tpot
        # 3. 计算 ttft_violation_rate / tpot_violation_rate
        # 4. PID 更新 last_ratio
        # 5. clamp & cache & 返回
```

为方便 unit test：所有「读 telemetry 增量」逻辑封装成 `_pull_recent_violations(now_iter_id)`，单测时直接喂 fake 数据。

### 5d. 选择 hook 入口的备选

任务文档问「在哪个 hook 接入」，候选：
- ✅ **`_decide_phase()` 之前**（即 `get_target_ratio()` 处）——本设计选这里。控制器只输出 ratio，不直接干预 phase；selector 把 ratio 转 phase。这条路径已经存在，零侵入。
- ❌ `update_from_output()` 之后——这里只能影响**下一**个 schedule tick，对反馈量来说一样（反正 controller 是 sticky 的），但代码上要新加 hook 点。
- ❌ 加 callback 让 telemetry 主动 push 到 controller——过度耦合，控制循环应该是 controller pull 而非 telemetry push。

---

## 6. 副作用 / 风险

### 6a. Oscillation（震荡）

PID Kp 过大、update_interval 过小、min_samples 过小都会导致 ratio 在 0.2 ↔ 0.6 间抖。

**缓解**：
- 加 EMA：`smoothed_ratio = 0.7*last + 0.3*new_ratio`
- min_samples=16，update_interval=8（≈ 1s），保证每次决策基于稳定窗口
- 加 deadband：`|delta| < 0.02` 时不动 ratio

### 6b. Starvation 加剧

当 `ttft_violation_rate >> tpot_violation_rate`，PID 会一路把 ratio 推到 0.8。HardConstraints 的 `max_slice_iters` 仍兜底（强制每 8 iter 至少切一次），所以 decode 不会饿死，但 tpot p99 可能会被拉高。

**缓解**：
- ratio_max=0.8（不允许 1.0）
- HardConstraints 已存在的 max_slice_iters=8 兜底
- 监测：在 telemetry 落 `controller_diag` event（决策时点的 ttft/tpot 违例率 + 输出 ratio），事后画图验证没有 starvation

### 6c. 与 BoundaryGuard 的耦合

KV pressure 高时 BoundaryGuard 强切 decode。如果此时 PID 因 ttft 违例正在加 ratio，会出现「想 prefill 却被压下来 → ttft 持续违例 → ratio 继续涨 → 压力释放后 ratio 已饱和 → 突然爆 prefill → KV 又紧」的振荡。

**缓解**：
- 在 controller 入口检查 `snap.kv_free_ratio < kv_free_watermark + margin`（margin=0.05）→ 直接返回 last_ratio，不更新（避免在 KV 紧时学错梯度）
- 或者把 BoundaryGuard 的次数也作为信号，多次触发 boundary 后衰减 ratio

### 6d. 与 graph-aware batch sizing (M3) 的耦合

M3 决定 phase=decode 时本 iter 装几个 reqs（贴 ACL graph 档位），不动 phase 也不动 ratio。M2 只动 ratio。两者**正交**，但有一条间接耦合：M3 选档位时如果 padding gap 大，单 iter 时间被拉长 → tpot 抬高 → 进 telemetry → controller 看到 tpot 违例率上升会推 ratio 往 decode 偏。要避免这个误学，M3 自身应该限定「padding gap 超阈值就降级 eager」，而不是依赖 controller 介入。

### 6e. 冷启动 / 短跑实验

短 sweep（duration=20s）window 永远填不满 → controller 永远退化为 static。这点要在实验设计上保证 `duration ≥ 60s` 且 QPS 中等以上。

### 6f. tracker 数据延迟

如 §1b 所述，请求完成才进 telemetry。短输出请求（output_tokens=16）不会有 tpot p99，长输出请求（output_tokens=512）反馈滞后到几秒后。**控制器的反应速度是 request lifetime 量级，不是 iter 量级。**

**缓解**：
- 接受这个延迟，把 controller 定位为「分钟级负载特征调节」而非「秒级 spike 抑制」
- spike 由 BoundaryGuard 兜底（KV pressure）和 HardConstraints 兜底（max_slice_iters）

---

## 7. 实验验证计划（review 后落地用）

最小验证集（review 通过后顺道安排）：

| 实验 | 配置 | 期望 |
|---|---|---|
| Unit test | fake telemetry / fake snap，覆盖冷启动 / 单边违例 / oscillation 防护 | controller 行为可预期 |
| 8K sweep + qps=24 | M2 vs M1 (c2_tdm) | M2 SLO meet 率 ≥ M1，goodput 不显著下降 |
| 8K sweep + qps=32 | M2 vs C3 (chunked prefill) | M2 ttft p99 仍优于 C3，SLO 接近 C3 |
| Diag 图 | M2 跑 60s，画 ratio over time | 看 ratio 是否收敛到合理值，无明显震荡 |

实验数据落 `results/m2_slo_adaptive/`，画图脚本复用 `plot_qps_sweep.py` 加一条 M2 曲线。

---

## 8. 待用户决策点（review 时关注）

1. **PID 还是 bang-bang 起步？**——本设计推 PID-lite（Option A）。
2. **target_violation_rate=0.05 合适吗？**——意思「允许 5% 请求超 SLO」。从 8K sweep 数据看 c1 在 qps=24 也只有 70.8% meet，调到 0.30 可能更现实。
3. **是否需要分维度 PID（ttft 一根、tpot 一根）？**——本设计用 max(err_ttft, err_tpot) 的 trick 把维度合并到 delta，保留可分性。
4. **Telemetry 是否要新增控制器自身的诊断输出？**——建议加，方便事后画 ratio over time。
5. **是否上 Kalman / 自适应 Kp？**——本设计先不上，留作后续优化。
