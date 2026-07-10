#!/bin/bash
# Smoke test: sweep rpm_scale on MaaS aggregated trace to find saturation boundary.
#
# Runs one short window (3 min = 180s data + warmup) per rpm_scale value,
# pdtdm (m31) paradigm only, single seed.
# Outputs a JSON summary per scale for quick inspection.
#
# FIX (2026-06-30): cd /tmp to avoid sys.path namespace-package conflict
#   when running from /vllm-workspace (which contains vllm/ without __init__.py).
#   Also added engine-core fatal error detection during health check.
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
PY=/usr/local/python3.11.13/bin/python3
SCRIPT=$ROOT/experiments/qps_sweep.py
OUT=$ROOT/results/smoke_rpm_scale
TRACE=$ROOT/data/maas/aggregated_curve.csv
MODEL=/models/Qwen/Qwen3-8B

DURATION=120
WARMUP=20
MML=8192
MB=8192
CONFIG="c2_tdm_m31_2048"
SEED=0

SCALES="0.02 0.06 0.10 0.14 0.18 0.22 0.26 0.30 0.34 0.38"

# CRITICAL: avoid sys.path namespace-package conflict.
# Running from /vllm-workspace causes Python's PathFinder to resolve
# /vllm-workspace/vllm/ (repo root without __init__.py) as a namespace
# package before the editable-install _EditableFinder can handle it.
cd /tmp

# NPU environment
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

BASE_URL="http://127.0.0.1:8000"
TRACE_DIR=/tmp/tdm_smoke_trace
RUN_ID="smoke_rpm"

mkdir -p "$OUT" "$TRACE_DIR"
echo "### rpm_scale smoke started $(date '+%H:%M:%S')"
echo "### window=${DURATION}s config=$CONFIG model=$MODEL cwd=$(pwd)"

for RS in $SCALES; do
  OUTFILE=$OUT/smoke_rpm${RS}.json
  TRACKER=$TRACE_DIR/${RUN_ID}_rpm${RS}_req.jsonl
  LOGFILE=$OUT/server_rpm${RS}.log

  echo "===== $(date '+%H:%M:%S') rpm_scale=$RS ====="

  # Ensure port 8000 is free before starting
  fuser -k 8000/tcp 2>/dev/null || true
  sleep 2

  # Start vLLM server with TDMScheduler
  echo "  Starting server..."
  ASCEND_SCHEDULER_CLS="vllm_ascend.core.tdm.scheduler.TDMScheduler"
  ADDITIONAL_CONFIG="{\"ascend_scheduler_config\":{\"enabled\":true,\"scheduler_cls\":\"${ASCEND_SCHEDULER_CLS}\"},\"tdm\":{\"enable_tdm\":true,\"static_ratio\":0.3,\"min_slice_iters\":2,\"max_slice_iters\":8,\"kv_free_watermark\":0.05,\"telemetry_enabled\":true,\"telemetry_dir\":\"${TRACE_DIR}\",\"run_id\":\"${RUN_ID}_rpm${RS}\",\"initial_phase\":\"prefill\",\"prefill_chunk_tokens\":2048}}"

  $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --max-model-len $MML \
    --max-num-batched-tokens $MB \
    --gpu-memory-utilization 0.85 \
    --additional-config "$ADDITIONAL_CONFIG" \
    --port 8000 \
    > "$LOGFILE" 2>&1 &

  SERVER_PID=$!
  echo "  Server PID=$SERVER_PID"

  # Wait for server to be ready (health endpoint + engine core initialized)
  echo "  Waiting for server..."
  READY=0
  for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
      # Verify engine core didn't fail silently
      if grep -q "Bind_Failed\|Timed out waiting for engines\|ImportError" "$LOGFILE" 2>/dev/null; then
        echo "  ERROR: engine core fatal error detected at ${i}s"
        grep -E "Bind_Failed|Timed out waiting for engines|ImportError" "$LOGFILE" | head -5
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        exit 1
      fi
      READY=1
      echo "  Server ready after ${i}s"
      break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "  ERROR: server died at ${i}s! Last 30 lines:"
      tail -30 "$LOGFILE"
      exit 1
    fi
    # Progress every 30s
    if [ $((i % 30)) -eq 0 ]; then
      echo "    ...${i}s elapsed, still waiting..."
    fi
    sleep 1
  done

  if [ $READY -eq 0 ]; then
    echo "  ERROR: server not ready after 180s"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    exit 1
  fi

  # Run driver
  echo "  Running driver..."
  $PY "$SCRIPT" \
    --arrival-mode aggregated_trace \
    --trace-file "$TRACE" \
    --rpm-scale "$RS" \
    --duration $DURATION --warmup $WARMUP \
    --base-url "$BASE_URL" --model "$MODEL" \
    --config-name "$CONFIG" \
    --out "$OUTFILE" \
    --tracker-jsonl "$TRACKER" \
    --seed $SEED \
    --slo-ttft-ms 5000 --slo-tpot-ms 500 \
    2>&1 | tail -5

  # Kill server
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  echo "  Server stopped."

  # Aggressively clean up any leftover NPU processes
  sleep 3
done

# Quick summary
echo ""
echo "===== Summary ====="
for RS in $SCALES; do
  F=$OUT/smoke_rpm${RS}.json
  if [[ -f "$F" ]]; then
    $PY -c "
import json
d=json.load(open('$F'))
n=d.get('n_submitted',0)
ok=d.get('n_ok',0)
err=d.get('n_err',0)
w=d.get('window',{})
dur=d.get('duration_s',0)
ttft_m=w.get('ttft_mean_ms','-')
ttft_p99=w.get('ttft_p99_ms','-')
tpot_m=w.get('tpot_mean_ms','-')
tpot_p99=w.get('tpot_p99_ms','-')
qps=round(n/dur,1) if dur else '-'
print(f'rpm_scale=$RS | submitted={n} ok={ok} err={err} qps≈{qps} | TTFT mean={ttft_m} p99={ttft_p99} | TPOT mean={tpot_m} p99={tpot_p99}')
"
  else
    echo "rpm_scale=$RS | MISSING"
  fi
done
echo "### smoke done $(date '+%H:%M:%S')"
