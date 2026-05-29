#!/bin/bash
# c4_pd steady-arrival (Poisson IID, non-burst) smoke (2026-05-25)
# Goal: prove burst trace is the c4_pd failure amplifier (vs 1P1D fundamental cost)
# Compare with `c4_pd_supplement/conv_k{0.5,1.0,1.4}_seed0/` burst cells.
#
# Match avg arrival rate: burst conv = 0.1*12k + 0.9*5k = 5.7k QPS
#   k=0.5: 2.85 QPS  | k=1.0: 5.70 QPS  | k=1.4: 7.98 QPS
# Use trace_sampled = Poisson IID at target QPS (same trace, no burst overlay).
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/c4_pd_steady_smoke
CONV_CSV=$ROOT/data/azure_trace/AzureLLMInferenceTrace_conv.csv

DURATION=90
WARMUP=30
SLO_TTFT=500
SLO_TPOT=200

run_steady_cell() {
  local d=$1 qps=$2 sd=$3
  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi
  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d"
  echo "===== c4_pd steady  qps=$qps  seed=$sd"
  echo "==================================================================="
  $PY "$SCRIPT" \
    --configs c4_pd \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len 8192 --max-num-batched-tokens 8192 \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode trace_sampled \
    --trace-file "$CONV_CSV" \
    --trace-max-prompt-tokens 7000 --trace-max-output-tokens 600 \
    --qps "$qps" \
    --outdir "$d" --seed "$sd" 2>&1 | tail -8
}

# 3 cells matching burst c4_pd conv k=0.5/1.0/1.4 avg arrival rates
run_steady_cell "$OUT/conv_qps2.85_seed0" 2.85 0
run_steady_cell "$OUT/conv_qps5.70_seed0" 5.70 0
run_steady_cell "$OUT/conv_qps7.98_seed0" 7.98 0

echo "==================================================================="
echo "=== $(date '+%H:%M:%S') c4_pd steady smoke done ==="
echo "=== outdir: $OUT"
echo "==================================================================="
