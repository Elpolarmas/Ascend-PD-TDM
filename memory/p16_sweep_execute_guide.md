# P1.6 后续 sensitivity sweep 执行指南

> 2026-05-13 准备。4 个独立实验,均可后台跑。

## 全部 4 个实验一览

| 实验 | 目的 | wall | calls | 关键产出 |
|---|---|---|---|---|
| **P1.6f** target_violation_rate sweep | attack ReLU+M2.4 联合屏蔽 | ~2h | 5×2×3=30 | 双输入是否复活 |
| **P1.6g** ratio_max sweep | 验证 ratio_max=0.8 天花板假设 | ~2.4h | 6×2×3=36 | ratio 上界是真物理还是设计 |
| **P1.6h** calibrated baseline | 200ms 档下重测全套 baseline | ~2.2h | 6 calls × 22min | thesis main result candidate |
| **P1.6i** chunk sweep on Azure | chunk 在 mild burst 上的最优值 | ~2.2h | 6 calls × 22min | P0-3 长 prompt 救场可能性 |

**所有 4 个连续跑**:~9h(一夜)

---

## 启动方式(任选 1-4 个独立跑或串联跑)

### 单独跑一个

```bash
cd /vllm-workspace/Ascend-PD-TDM
# 选一个:
nohup bash experiments/run_p16f_target_viol_sweep.sh \
    > results/azure_p16f_orchestrator.log 2>&1 &
echo "PID=$!"
```

把 `f` 换成 `g` / `h` / `i` 跑其它实验。

### 4 个串联跑(一夜搞定)

```bash
cd /vllm-workspace/Ascend-PD-TDM
nohup bash -c '
  bash experiments/run_p16f_target_viol_sweep.sh > results/azure_p16f_orchestrator.log 2>&1
  bash experiments/run_p16g_ratio_max_sweep.sh   > results/azure_p16g_orchestrator.log 2>&1
  bash experiments/run_p16h_calibrated_baseline.sh > results/azure_p16h_orchestrator.log 2>&1
  bash experiments/run_p16i_chunk_sweep.sh        > results/azure_p16i_orchestrator.log 2>&1
  echo "ALL DONE $(date)"
' > results/azure_p16_full_pipeline.log 2>&1 &
echo "Pipeline PID=$!"
```

## 进度监控

### 查当前在跑哪一个

```bash
pgrep -af "run_p16[fghi]" | head -5
```

### 查每个实验的进度

```bash
# P1.6f 进度(目标 30 calls)
find /vllm-workspace/Ascend-PD-TDM/results/azure_p16f -name "qps_sweep_summary.json" | wc -l

# P1.6g(36)/ P1.6h(6)/ P1.6i(6) 同理
find /vllm-workspace/Ascend-PD-TDM/results/azure_p16g -name "qps_sweep_summary.json" | wc -l
find /vllm-workspace/Ascend-PD-TDM/results/azure_p16h -name "qps_sweep_summary.json" | wc -l
find /vllm-workspace/Ascend-PD-TDM/results/azure_p16i -name "qps_sweep_summary.json" | wc -l
```

### 查 orchestrator 最新输出

```bash
tail -20 /vllm-workspace/Ascend-PD-TDM/results/azure_p16f_orchestrator.log
```

### 监控失败

```bash
grep -rE "Traceback|FAILED|OOM|RuntimeError" \
    /vllm-workspace/Ascend-PD-TDM/results/azure_p16{f,g,h,i}/*/run.log 2>/dev/null | head -5
```

## 实验完成后看结果

```bash
cd /vllm-workspace/Ascend-PD-TDM

# 各个实验单独 posthoc
python3 experiments/posthoc_p16f_target_viol.py    | tee results/p16f_summary.txt
python3 experiments/posthoc_p16g_ratio_max.py      | tee results/p16g_summary.txt
python3 experiments/posthoc_p16h_calibrated_baseline.py | tee results/p16h_summary.txt
python3 experiments/posthoc_p16i_chunk.py          | tee results/p16i_summary.txt
```

## 关键判读标准

### P1.6f target_viol — 是否救回闭环
- 某 target 下 `err_tpot_eff 非零率 > 20%` 且 `ratio 钉 max < 80%` → **PASS,核心问题有 fix**
- 全档失败 → ReLU+M2.4 联合不可解,**走 P1.7b iter 维度旁路**

### P1.6g ratio_max — 天花板真假
- ratio_max=0.5/0.7 下 `ratio 钉 max > 50%` → 确认是 PID 单向推力,不是物理边界
- ratio_max=0.95/1.0 下 meet_slo% 退步 → 0.8 是合理 decode 余量
- ratio_max=0.95/1.0 下 meet_slo% 反而升 → 0.8 偏保守可放宽

### P1.6h calibrated baseline — thesis main result
- M3.1 vs C3 / M1+chunk 在 ttft500/tpot200 档下的 Δ
- 跟 azure_main(50ms 档)对比,区分度应当显著更高
- 这一档下 conv/code 的 Δ 是 thesis 主图候选

### P1.6i chunk — pareto 单调还是有 sweet spot
- 如果 chunk_tokens=512 在 conv 长 prompt 上反而好 → 救 P0-3 22pp 输 C3 的 finding
- 如果仍单调 chunk=2048/4096 最好 → 跟 P1 chunk scan 结论一致

## 重启/续跑

所有 runner 都有 `skip 已完成 driver` 逻辑,直接重跑就行:

```bash
bash experiments/run_p16f_target_viol_sweep.sh
# 会自动 skip 已有 summary.json 的目录,从断点继续
```

## 数据落点速查

```
results/azure_p16f/tgt{005,010,020,030,050}/{conv,code}_w1_seed{0,1,2}/
results/azure_p16g/rmax{05,07,08,09,095,10}/{conv,code}_w1_seed{0,1,2}/
results/azure_p16h/{conv,code}_w1_seed{0,1,2}/
results/azure_p16i/{conv,code}_w1_seed{0,1,2}/
```

每个 seed 目录下:
- `qps_sweep_summary.json` — 完成标志 + 主要 metrics
- `tdm_trace/*_ctrl.jsonl` — PID 内部状态时序(P1.6f/g 关键)
- `tdm_trace/*_req.jsonl` — 单 req latency(P1.6h/i posthoc 重算 meet_slo)
- `run.log` — driver 日志

## 注意事项

1. **NPU 资源独占**:这些实验串行跑(每个内部启停 vllm server),不要同时跑两个
2. **磁盘空间**:每个 driver call 落 ~10MB jsonl,4 个实验合计 ~700MB
3. **续跑安全**:中断后直接重跑 runner,会跳过已完成
4. **server 进程清理**:如果 watcher 显示有 orphan server 进程,`pkill -f vllm` 后重跑

## 实验完成后建议

跑完 4 个 sensitivity 后,落 finding 到 `memory/current_task.md §3` 各自子节,然后:
- 如果 P1.6f / P1.6g 给出"PID 设计本身可修"信号 → 不必走 P1.7b
- 否则 → 走 P1.7b iter 维度补强 + thesis 措辞改 β+
