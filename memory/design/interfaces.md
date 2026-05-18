# 状态采集接口 + 缺口表

> 厘清现有 tracker / monitor / telemetry 接口,列各上层模块的数据需求,产出缺口表。P1.7b 前置工作。完成日期 2026-05-12。

---

## 1. 现有接口

### 1.1 `QueueMonitor.snapshot(scheduler, now_ms) → QueueSnapshot`

- 调用点:`scheduler.py:90`,每 iter 一次
- 实现:`monitor.py`,O(1) 读 scheduler.{waiting, running, finished_prefill_reqs, kv_cache_*}

**QueueSnapshot**(immutable dataclass,`types.py:16`):

| 字段 | 类型 | 来源 |
|---|---|---|
| `waiting_depth` | int | `len(scheduler.waiting)` |
| `waiting_oldest_age_ms` | float | `now_ms - waiting[0].arrival_time*1000` |
| `finished_prefill_depth` | int | `len(scheduler.finished_prefill_reqs)` |
| `running_depth` | int | `len(scheduler.running)` |
| `kv_free_blocks / kv_total_blocks` | int | `kv_cache_manager.block_pool.get_num_free_blocks()` + `kv_cache_config.num_blocks` |
| `kv_free_ratio` (property) | float | 派生 |

### 1.2 `RequestTracker`(`tracker.py`)

**API:**
- `on_admit(req_id, prompt_tokens, ts_ms)` ← scheduler.schedule() admit 时
- `on_token_produced(req_id, num_new_tokens, ts_ms)` ← update_from_output() 出 token 时
- `on_finish(req_id, ts_ms) → Optional[RequestRecord]` ← parent 释放 req 时

**RequestRecord**(`types.py:168`):

| 字段 | 含义 |
|---|---|
| `request_id, prompt_tokens` | 标识 + 长度 |
| `admission_ts_ms, first_token_ts_ms, finish_ts_ms` | 关键时点 |
| `output_tokens` | 累计 |
| `decode_intervals_ms: list` | 每 token 间隔 |
| `_last_token_ts_ms` | private,**最后出 token 时刻**(P1.7b 关键) |
| `ttft_ms / tpot_ms_mean / tpot_ms_p99`(property) | 派生 |

生命周期:on_admit 加入 `_records` dict;on_finish 弹出。窗口期内 dict 持有所有活跃 + 已完成 req。

### 1.3 `Telemetry`(`telemetry.py`)

**热环**(in-memory deque,for controller hot read):
- `record_iter(IterRecord)` + `recent_iters(n)`,window=32
- `record_request(RequestRecord)` + `recent_requests(n)`,buffer=256

**冷日志**(JSONL,for posthoc):
- 4 路:`{run_id}_{iter|req|ctrl|chunk}.jsonl`
- 异步 writer thread(不阻塞 schedule)

### 1.4 P1.6b 反馈链修复(F1+F2)

修复前:`record_request` 只在 `on_finish` 调用 → ttft/tpot 信号有「请求 lifetime 量级」延迟。

修复后:
- F1:出第一个 token 时把 ttft 实时算出,直接送热环
- F2:每个 token interval 实时更新 tpot 滑动平均

效果:ttft warm-up 从 16.5s 降到 2.0s(见 `FINDINGS.md` P1.6b)。

---

## 2. 各上层模块的数据需求

### 2.1 慢回路 `SLOReactiveController`

读什么:
- `telemetry.recent_requests(window_size)` → 滑窗 ttft_ms / tpot_ms_mean → SLO 违例率(`controller.py:376`)
- `snap.kv_free_ratio` → KV freeze 触发(`controller.py:174`)
- `snap.waiting_oldest_age_ms / slo_ttft_ms` → backlog_norm 项(`controller.py:296`)

**状态:** ✅ 全满足。

### 2.2 快回路 `TokenBucketSelector`

读什么(现状):
- `target_ratio`(来自慢回路)
- `snap.waiting_depth / running_depth / finished_prefill_depth`
- P1.7 新增:`snap.waiting_oldest_age_ms ≥ urgency_threshold_ms`(单向 urgency,FAIL)

**P1.7b 双向预警缺口:**
- ⚠️ **TPOT 沉默信号:**「running 中最久未出 token 的 decode 距今多久」
  - 数据其实已在 tracker(每个 RequestRecord 的 `_last_token_ts_ms`)
  - 缺口:没暴露成 snapshot 字段或便捷查询

### 2.3 Graph 自适应模块(已降级,D-005)

如果未来重启需要:
- ACL Graph capture sizes 列表(model + TP 配置相关)→ 当前缺
- 当前 `prefill_chunk_tokens` 配置(从 cfg 可读)→ 有
- Per-size latency 画像(离线刻画产物)→ 缺

**状态:** ⚠️ 缺,但模块已降级。等 D-005 三选一拍板再决定补不补。

---

## 3. 缺口表

| # | 缺口 | 给谁用 | 阻塞什么 | 实装方案 | 代价 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | `QueueSnapshot.decode_oldest_silence_ms` | P1.7b 快回路 | P1.7b 启动 | `QueueMonitor` 持有 tracker 引用,snapshot 时遍历 `tracker._records`(running 子集)取 `max(now_ms - _last_token_ts_ms)`。注意 `_last_token_ts_ms is None`(prefill 未出首 token)的 req 应排除或单独计 | 1-2h(含单测) | 中(P1.7b 决定做才需要) |
| 2 | ACL Graph capture sizes 暴露 | Graph 模块(候选) | Graph 模块实装 | 排查 vllm-ascend `acl_graph.py` / piecewise_capture 接口,在 TDMScheduler 启动时记录一份 | 0.5d 排查 | 低(等 D-005) |
| 3 | Per-size latency 画像(离线) | Graph 模块(候选) | Graph 决策算法 | 离线 microbench,1 次 run,~30min wall + 数据落盘 | 0.5d | 低(等 D-005) |
| 4 | StateProbe 门面重构 | 全模块解耦 | 架构清洁(非阻塞) | 新增 `state_probe.py`,scheduler 只持有一个 StateProbe;现有三件套不动,加 facade | 1-2d | 低(非阻塞,推迟到论文写完) |
| 5 | `TDMConfig.mixed_mode` 开关 + selector 短路分支 | phase-pure 机制级 ablation(D-010 claim A) | 论文 claim「P/D 时分复用有用」缺同 stack 对照证据 | `config.py` 加 bool 字段;`selector.py.peek()` 当 mixed_mode=True 时直接 return MIXED;`scheduler.py` 核对 MIXED phase 下能正常 admit 混合 batch | 中等(几天,含单测) | **P0**(与 P1.7b 并列,可串行) |

---

## 4. 关键观察

1. **缺口 #1 是 P1.7b 唯一短期阻塞**:数据其实已在 tracker(`_last_token_ts_ms`),只是没暴露到 selector 决策路径。补这个便宜(1-2h),其他模块无依赖
2. **缺口 #2/#3 跟 D-005 强耦合**:Graph 模块如果选 F-only(只留 finding),这两个缺口直接消失
3. **缺口 #4 是架构层面**:现在 scheduler.py L51-58 手工 wire 三个对象,长期会脏,但功能上不阻塞
4. **缺口 #5(D-010)是 phase-pure claim 的同 stack 对照阻塞**:不补则论文 claim A 只能靠 vs C3 paradigm 对比立,代码 stack 差异 confound 清不掉。设计简单(selector short-circuit),工程量主要在确认 scheduler MIXED phase 路径

---

## 5. P1.7b 接口设计预案

```python
# 1. QueueMonitor 拿 tracker 引用(scheduler init 时传入)
class QueueMonitor:
    def __init__(self, tracker: RequestTracker | None = None):
        self._tracker = tracker

    def snapshot(self, scheduler, now_ms):
        decode_silence_ms = 0.0
        if self._tracker is not None:
            for req in scheduler.running:
                rec = self._tracker._records.get(req.request_id)
                if rec is None or rec._last_token_ts_ms is None:
                    continue  # prefill 未出首 token → 不算沉默
                age = now_ms - rec._last_token_ts_ms
                if age > decode_silence_ms:
                    decode_silence_ms = age
        return QueueSnapshot(..., decode_oldest_silence_ms=decode_silence_ms)

# 2. selector.peek() 接收 starvation_threshold_ms
class TokenBucketSelector:
    def peek(self, target_ratio, snap,
             urgency_threshold_ms=0.0,
             starvation_threshold_ms=0.0,
             ...):
        # 顺序:no-work → no-prefill → no-decode → starvation → urgency → bucket
        # 新增 starvation:if planned == prefill and decode 沉默 → 翻 decode
```

---

## 6. Done 自检

- ✅ 现有 tracker / monitor / telemetry 接口清单(§1)
- ✅ 慢回路 / 快回路 / Graph 三模块的数据需求(§2)
- ✅ 缺口表 4 条,代价/优先级明确(§3)
- ✅ P1.7b 接口预案(§5)
