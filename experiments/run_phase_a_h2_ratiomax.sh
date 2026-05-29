#!/usr/bin/env bash
# Phase A H2 diagnostic (D-012, 2026-05-20):
# PID ratio_max=0.95 重跑,直接判 H1 vs H2。
#
# 背景 (memory project_pdtdm_h2_open):
#   M3.1 PID 在跨 4 档 SLO 都钉 ratio_max=0.8 (FINDINGS line 286)。
#   H1 (workload-adaptive):0.8 是 workload 决定的真实最优;PID 找到 peak
#   H2 (mechanism-saturated):0.8 是 PID 的 ceiling;PID 单向 urgency 撞 ceiling,实际最优更高
#
# T4 post-hoc 用 static_scan_3a 弱微倾向 H1 (conv goodput r07=89 peak, r09=85 微退),
# 但 sweep 太稀,踩不到 PID 实际收敛点。本实验直接证:
#   把 ratio_max 抬到 0.95,看 PID 是否还钉新 ceiling (H2) 还是回落到 r07-r08 (H1)。
#
# 配置:
#   - config: c2_tdm_m31_2048 (PID 主路径)
#   - SLO: strict 500/50 (跟 v8_matrix 同档,PID 在此 SLO 已知钉 0.8)
#   - traces: conv off1860 (peak / M3.1 winning regime) + off2160 (HIGH-short / reverse case)
#   - seeds: 3
#   - duration: 60s,warmup 20s (跟 v8 / azure_main 一致)
#
# 规模: 2 offset × 3 seed × 1 config = 6 runs ≈ 1h wall。

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/phase_a_h2_ratiomax
TRACE_DIR=$ROOT/results/tdm_trace
TRACE_FILE=$ROOT/data/azure_trace/AzureLLMInferenceTrace_conv.csv
CONFIGS="c2_tdm_m31_2048"
RATIO_MAX=0.95

mkdir -p "$OUT"

OFFSETS=(1860 2160)
SEEDS=(0 1 2)

for OFF in "${OFFSETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    D=$OUT/off${OFF}_seed${SEED}
    if [[ -f "$D/qps_sweep_summary.json" ]]; then
      echo "[skip] $D already complete"
      continue
    fi
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== H2 diag offset=${OFF}s seed=${SEED} ratio_max=${RATIO_MAX}"
    echo "==================================================================="
    /usr/local/python3.11.13/bin/python3 \
      "$ROOT/experiments/run_qps_sweep_all.py" \
      --configs "$CONFIGS" \
      --qps 0 \
      --duration 60 --warmup 20 \
      --max-model-len 8192 --max-num-batched-tokens 8192 \
      --slo-ttft-ms 500 \
      --slo-tpot-ms 50 \
      --slo-ratio-max "$RATIO_MAX" \
      --arrival-mode trace \
      --trace-file "$TRACE_FILE" \
      --trace-start-offset "$OFF" \
      --outdir "$D" \
      --seed "$SEED" \
      2>&1 | tee "$D/run.log"
    for ST in iter req ctrl chunk; do
      F=$TRACE_DIR/qps_sweep_${CONFIGS}_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CONFIGS}_${ST}.jsonl" || true
    done
  done
done

echo "=== Phase A H2 ratio_max=${RATIO_MAX} done ==="
echo "下一步:"
echo "  1. 比较 ctrl.jsonl 的 ratio 时序 vs azure_main / v8_matrix (ratio_max=0.8) 同 offset"
echo "  2. 如果 ratio 仍钉 ~0.95 → H2 (mechanism saturation) 确认"
echo "  3. 如果 ratio 收敛 r07-r085 → H1 (workload-adaptive) 确认"
