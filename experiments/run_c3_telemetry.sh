#!/bin/bash
# c3-fair iter telemetry capture (2026-05-25)
# Goal: quantify c3 mixed iter time to verify mechanism #1+#2 framing
# (currently c3 iter time only has physical reasoning, no telemetry)
#
# Prereq: scheduler.py patched to enable iter recording in passive mode
# (TDMScheduler now records IterRecord on c3_cp passive path)
#
# Cells: code k=2.8 / k=4.9 seed 0 (matches m31 telemetry verify cell at k=2.8)
# Wall: ~10 min total
#
# Per cell:
#   1. clean global tdm_trace/ for run_id
#   2. run sweep (writes to results/tdm_trace/qps_sweep_c3_cp_iter.jsonl)
#   3. copy iter file into cell-local tdm_trace/ to prevent next cell overwrite
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/c3_telemetry
GLOBAL_TRACE=$ROOT/results/tdm_trace
RUN_ID=qps_sweep_c3_cp

DURATION=90
WARMUP=30
PERIOD=10
CODE_CSV=$ROOT/data/azure_trace/AzureLLMInferenceTrace_code.csv
SLO_TTFT=500
SLO_TPOT=200

run_telemetry_cell() {
  local d=$1 hi=$2 lo=$3 fr=$4 sd=$5 k=$6
  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d"
  echo "===== c3 telemetry  burst high=$hi low=$lo frac=$fr  seed=$sd  k=$k"
  echo "==================================================================="
  # clean previous run iter file so we capture only this cell's data
  rm -f "$GLOBAL_TRACE/${RUN_ID}_iter.jsonl" \
        "$GLOBAL_TRACE/${RUN_ID}_chunk.jsonl" \
        "$GLOBAL_TRACE/${RUN_ID}_req.jsonl"
  $PY "$SCRIPT" \
    --configs c3_cp \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len 8192 --max-num-batched-tokens 2048 \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode trace_sampled_burst \
    --trace-file "$CODE_CSV" \
    --trace-max-prompt-tokens 7000 --trace-max-output-tokens 600 \
    --burst-period $PERIOD \
    --burst-high-qps "$hi" --burst-low-qps "$lo" --burst-high-frac "$fr" \
    --outdir "$d" --seed "$sd" 2>&1 | tail -8
  # capture this cell's telemetry before next cell overwrites
  cp -v "$GLOBAL_TRACE/${RUN_ID}_iter.jsonl"  "$d/tdm_trace/" 2>&1 | tail -1 || true
  cp -v "$GLOBAL_TRACE/${RUN_ID}_chunk.jsonl" "$d/tdm_trace/" 2>&1 | tail -1 || true
  cp -v "$GLOBAL_TRACE/${RUN_ID}_req.jsonl"   "$d/tdm_trace/" 2>&1 | tail -1 || true
}

# code k=2.8: matches m31 telemetry verify cell (k2.8 s1, prefill iter 150ms / decode 39ms / cycle 189ms)
# code k=4.9: heaviest load, mechanism #1 most visible (chunk_budget 2048 split most extreme)
run_telemetry_cell "$OUT/code_k2.8_seed0" 28.0 2.8 0.2 0 2.8
run_telemetry_cell "$OUT/code_k4.9_seed0" 49.0 4.9 0.2 0 4.9

echo "==================================================================="
echo "=== $(date '+%H:%M:%S') c3 telemetry capture done ==="
echo "=== outdir: $OUT"
echo "==================================================================="
