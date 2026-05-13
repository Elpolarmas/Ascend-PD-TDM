#!/usr/bin/env bash
# 5-way long-prompt sweep at qps∈{16,32}, 3 seeds. Fills §8.4 backlog:
# - qps=32 saturated regime (PID hypothesis: static_ratio can't keep up)
# - C3 chunked-prefill baseline at long prompt (never multi-seed before)
# Preserves per-seed traces (TRACE_DIR is global, would otherwise overwrite).
set -euo pipefail
ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/long_5way_q1632
TRACE_DIR=$ROOT/results/tdm_trace
CONFIGS="c1_baseline,c2_tdm,c2_tdm_m1_chunk2048,c2_tdm_m31_2048,c3_cp"
mkdir -p "$OUT"
for SEED in 0 1 2; do
  D=$OUT/seed$SEED
  mkdir -p "$D/tdm_trace"
  echo "===== seed=$SEED ====="
  /usr/local/python3.11.13/bin/python3 \
    "$ROOT/experiments/run_qps_sweep_all.py" \
    --configs "$CONFIGS" --qps 16,32 --duration 60 --warmup 20 \
    --prompt-profile long --seed "$SEED" --outdir "$D" \
    2>&1 | tee "$D/run.log"
  for CFG in c1_baseline c2_tdm c2_tdm_m1_chunk2048 c2_tdm_m31_2048 c3_cp; do
    for ST in ctrl chunk iter req; do
      F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
    done
  done
done
echo "=== done ==="
