#!/bin/bash
# c4_pd online smoke test — verify 8B + ranktable + KV transfer + proxy 全链路。
# 单 cell:conv k=0.5 seed=0,低负载,跑通即可。
# 用 short duration 加快迭代。
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/c4_pd_smoke/conv_k0.5_seed0
CONV_CSV=$ROOT/data/azure_trace/AzureLLMInferenceTrace_conv.csv

DURATION=60
WARMUP=20
PERIOD=10
# conv k=0.5: hi=12*0.5=6, lo=5*0.5=2.5, frac=0.1
HI=6
LO=2.5
FR=0.1

mkdir -p "$OUT/tdm_trace"
echo "==================================================================="
echo "===== $(date '+%H:%M:%S') c4_pd smoke @ conv k=0.5 seed=0 ====="
echo "===== burst high=$HI low=$LO frac=$FR  duration=${DURATION}s"
echo "===== outdir=$OUT"
echo "==================================================================="

$PY "$SCRIPT" \
  --configs c4_pd \
  --duration $DURATION --warmup $WARMUP \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --slo-ttft-ms 500 --slo-tpot-ms 200 \
  --arrival-mode trace_sampled_burst \
  --trace-file "$CONV_CSV" \
  --trace-max-prompt-tokens 7000 --trace-max-output-tokens 600 \
  --burst-period $PERIOD \
  --burst-high-qps "$HI" --burst-low-qps "$LO" --burst-high-frac "$FR" \
  --outdir "$OUT" --seed 0 2>&1 | tee "$OUT/smoke.log"

echo "==================================================================="
echo "=== $(date '+%H:%M:%S') c4_pd smoke done ==="
echo "=== summary: $OUT/qps_sweep_summary.json"
echo "=== server logs: $ROOT/results/server_logs/server_*_{prefill,decode,proxy}.log"
echo "==================================================================="
