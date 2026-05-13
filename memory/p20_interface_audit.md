# P2.0 状态采集模块接口审计

> 完成日期:2026-05-12
> 目的:厘清现有 tracker/monitor/telemetry 接口,列各上层模块的数据需求,产出缺口表
> Done 标准(current_task.md §6):接口清单 + 各模块数据清单 + 缺口表(数量明确,代价清楚)

## 1. 现有接口清单

### 1.1 `QueueMonitor.snapshot(scheduler, now_ms) → QueueSnapshot`
- 调用点:scheduler.py L90,每 iter 一次
- 实现:`monitor.py`,O(1) 读 scheduler.{waiting, running, finished_prefill_reqs, kv_cache_*}
- 产物 **QueueSnapshot**(immutable dataclass,`types.py:16`):
  | 字段 | 类型 | 来源 |
  |---|---|---|
  | `waiting_depth` | int | `len(scheduler.waiting)` |
  | `waiting_oldest_age_ms` | float | `time.time()*1000 - waiting[0].arrival_time*1000`(best-effort) |
  | `finished_prefill_depth` | int | `len(scheduler.finished_prefill_reqs)` |
  | `running_depth` | int | `len(scheduler.running)` |
  | `kv_free_blocks / kv_total_blocks` | int | `kv_cache_manager.block_pool.get_num_free_blocks()` + `kv_cache_config.num_blocks` |
  | `kv_free_ratio` (property) | float | 派生 |

### 1.2 `RequestTracker`(`tracker.py`)
**API**:
- `on_admit(req_id, prompt_tokens, ts_ms)` ← scheduler.schedule() admit 时
- `on_token_produced(req_id, num_new_tokens, ts_ms)` ← update_from_output() 出 token 时
- `on_finish(req_id, ts_ms) → Optional[RequestRecord]` ← parent 释放 req 时

**RequestRecord**(`types.py:168`):
| 字段 | 含义 |
|---|---|
| `request_id, prompt_tokens` | 标识 + 长度 |
| `admission_ts_ms, first_token_ts_ms, finish_ts_ms` | 关键时点 |
| `output_tokens` | 累计 |
| `decode_intervals_ms: list` | 每个 token 间隔(分摊到 num_new_tokens) |
| `_last_token_ts_ms` | private,**最后出 token 时刻**(P1.7b 关键字段) |
| `ttft_ms, tpot_ms_mean, tpot_ms_p99`(property) | 派生 |

**当前生命周期**:on_admit 加入 `_records` dict;on_finish 从 dict 弹出。**在窗口期内 dict 持有所有活跃 + 已完成 req**(直到 finish)。

### 1.3 `Telemetry`(`telemetry.py`)
**热环**(in-memory deque,for controller hot read):
- `record_iter(IterRecord)` + `recent_iters(n)` — window_size=32 默认
- `record_request(RequestRecord)` + `recent_requests(n)` — request_buffer=256

**冷日志**(JSONL,for posthoc):
- 4 路:`{run_id}_{iter|req|ctrl|chunk}.jsonl`
- 异步 writer thread(不阻塞 schedule)
- API:`record_iter/record_request/record_controller_tick/record_chunk_plan`

## 2. 各上层模块的数据需求

### 2.1 A 速率调整器(SLOReactiveController)
读什么:
- `telemetry.recent_requests(window_size)` → 滑窗 ttft_ms / tpot_ms_mean → SLO 违例率(`controller.py:376`)
- `snap.kv_free_ratio` → KV freeze 触发(`controller.py:174`)
- `snap.waiting_oldest_age_ms / slo_ttft_ms` → backlog_norm 项(`controller.py:296`)

**状态**:✅ 全满足。

### 2.2 B 决策器(TokenBucketSelector)
现有读什么:
- `target_ratio`(来自 A,via scheduler)
- `snap.waiting_depth / running_depth / finished_prefill_depth`
- P1.7 新增:`snap.waiting_oldest_age_ms ≥ urgency_threshold_ms`

**当前状态**:✅ 满足。

**P1.7b 双向 indicator 缺口**:
- ⚠️ **TPOT starvation 信号**:"running 中最久未出 token 的 decode 距今多久"
  - 数据其实已经在 tracker:每个 RequestRecord 的 `_last_token_ts_ms`
  - 缺口:没暴露成 snapshot 字段或便捷查询

### 2.3 Graph 自适应模块(候选 C/D,若保留)
按 §4.4:推荐 chunk size 上限边界 / 跟 chunking 合并

读什么:
- ACL Graph capture sizes 列表(model + TP 配置相关)
- 当前 `prefill_chunk_tokens` 配置(从 cfg 可读)
- Per-size latency 画像(离线刻画产物)

**状态**:⚠️ 全缺。但 P1.0b 已让模块价值存疑(§4.4),先看 P2.5 角色定夺再决定是否补缺。

## 3. 缺口表(按优先级)

| # | 缺口 | 给谁用 | 阻塞了什么 | 实装方案 | 代价 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | `QueueSnapshot.decode_oldest_silence_ms` — running 中最久未出 token 的 decode 距今多久 | P1.7b 双向 indicator | P1.7b 启动 | QueueMonitor 持有 tracker 引用,snapshot 时遍历 `tracker._records`(self.scheduler.running 子集)取 max(now - _last_token_ts_ms)。注意 `_last_token_ts_ms is None`(prefill 未出首 token)的 req 应排除或单独计 | 1-2h(含单测) | 中(P1.7b 决定做才需要) |
| 2 | ACL Graph capture sizes 暴露(从 vllm runner 拿) | Graph 模块(候选 C) | Graph 模块实装 | 排查 vllm-ascend 的 acl_graph.py / piecewise_capture 接口,在 TDMScheduler 启动时记录一份 | 0.5d 排查 | 低(等 P2.5 定夺) |
| 3 | Per-size latency 画像表(离线) | Graph 模块(候选 C) | Graph 模块决策算法 | 离线 microbench,1 次 run,~30min wall + 数据落盘 | 0.5d | 低(等 P2.5) |
| 4 | StateProbe 门面重构 — 把 monitor/tracker/telemetry 三个对象包成一个状态采集服务,上层不再直接 import 这三个 | 全模块解耦 / P4 | 架构清洁(非阻塞) | 新增 `state_probe.py`,scheduler.py 只持有一个 StateProbe;现有 monitor/tracker/telemetry 不动,只加 facade | 1-2d | 低(P4,非阻塞) |

## 4. 关键发现

1. **缺口 #1 是唯一短期阻塞**:P1.7 FAIL 的根因(单向 urgency)要靠双向 indicator 解决,而 TPOT 端的数据其实**已经在 tracker 里**(`_last_token_ts_ms`),只是没暴露到 selector 决策路径。补这个缺口便宜(1-2h),且其他模块无依赖。

2. **缺口 #2/#3 高度耦合于 P2.5 决定**:Graph 模块如果撤下(选项 A from §4.4),这两个缺口直接消失。

3. **缺口 #4 是架构层面**:现在 scheduler.py L51-58 手工 wire 三个对象,长期会脏。但功能上没阻塞,可以推到 P4。

## 5. 给 P1.7b 的接口设计预案

如果用户决定做 P1.7b,推荐接口形态:

```python
# 1. QueueMonitor 拿 tracker 引用(scheduler init 时传入)
class QueueMonitor:
    def __init__(self, tracker: RequestTracker | None = None): ...
    def snapshot(self, scheduler, now_ms):
        decode_silence_ms = 0.0
        if self._tracker is not None:
            for req in scheduler.running:
                rec = self._tracker._records.get(req.request_id)
                if rec is None or rec._last_token_ts_ms is None:
                    continue  # prefill not yet emitting → not starvation
                age = now_ms - rec._last_token_ts_ms
                if age > decode_silence_ms:
                    decode_silence_ms = age
        return QueueSnapshot(..., decode_oldest_silence_ms=decode_silence_ms)

# 2. selector.peek() 接受 starvation_threshold_ms
class TokenBucketSelector:
    def peek(self, target_ratio, snap,
             urgency_threshold_ms=0.0,
             starvation_threshold_ms=0.0,
             ...):
        # 当前 peek 顺序:no-work → no-prefill → no-decode → urgency → bucket
        # 新增 starvation:if planned == prefill and decode starvation → flip decode
```

## 6. Done 判定(自检)

- ✅ 现有 tracker / monitor / telemetry 接口清单完成(§1)
- ✅ A / B / Graph 三模块的数据需求清单(§2)
- ✅ 缺口表 4 条,代价/优先级明确(§3)
- ✅ 给 P1.7b 的接口预案(§5,可选 deliverable)
