---
name: maas-full-replay-experiment
description: Full MaaS trace replay pdtdm vs sarathi — pdtdm done, sarathi running as of 2026-07-01
metadata:
  type: project
  experiment_id: maas_full_rs0.12_v3
  status: pdtdm_complete_sarathi_running
  started: 2026-07-01T14:16:01
  pdtdm_done: 2026-07-01T07:40 (wall 17.4h)
  sarathi_eta: unknown (progress very slow, ~18K POSTs as of ~14:00)
---

# MaaS 全量 Trace Replay 对比实验

## 最终实验配置

| 参数 | 值 |
|---|---|
| Trace | `aggregated_curve_sampled.csv` (569行, 隔分钟采样, 9.5h) |
| rpm_scale | **0.12** |
| SLO | TTFT < 2808ms, TPOT < 648ms (MaaS ideal ×3) |
| SLO 来源 | micro-benchmark: TTFT p99=936ms, TPOT p99=216ms |
| 模型 | Qwen3-8B, TP=2 |

## 代码修复 (vs 原始代码)

1. `qps_sweep.py` run_driver: 新增 `iter_buckets()` 分支——按每分钟 batch gather, 避免 320K asyncio Task
2. `workload.py` AggregatedTraceReplay: 新增 `iter_buckets()` → yield `list[Request]` per minute
3. `__iter__` 改为 delegate 到 `iter_buckets()` (`yield from`)
4. `cd /tmp` 避免 namespace package 冲突
5. CSV 空行清理

## 已知问题

1. **per-minute batching 超载**: 峰值分钟 1172 请求全部并发 → PoolTimeout 80K
2. **长 prompt 溢出**: 14,444 请求超过 max_model_len=8192
3. **wall time 膨胀**: 预期 9.5h, 实际 17.4h (1.8x)——超载分钟 +300s timeout drain
4. **sarathi 极慢**: 7h 仅 18K POSTs, 原因待查

## pdtdm 结果

| 指标 | 值 |
|---|---|
| submitted | 138,529 |
| ok | 43,554 (31.4%) |
| errors | 94,975 (68.6%) |
| wall | 62,536s (17.4h) |
| TTFT p99 | 1,261ms |
| TPOT p99 | 291ms |
| SLO meet | 100.0% |
| goodput | 166 tok/s |
| e2e p99 | 386s |

错误: PoolTimeout 80,528 / ContextLength 14,444 / ReadError 2 / ConnectTimeout 1

## 检查命令

```bash
# 进程
ps aux | grep -E "run_maas|qps_sweep|EngineCore" | grep -v grep

# 日志进度
grep "\[driver\] done\|sarathi\|### MaaS done" /vllm-workspace/Ascend-PD-TDM/results/maas_replay/run_full.log

# pdtdm 结果
ls -la /vllm-workspace/Ascend-PD-TDM/results/maas_replay/maas_pdtdm_rs0.12.json

# sarathi POSTs
grep -c "POST /v1/completions" /vllm-workspace/Ascend-PD-TDM/results/maas_replay/server_sarathi.log
```
