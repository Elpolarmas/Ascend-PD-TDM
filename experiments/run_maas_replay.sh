#!/bin/bash
# MaaS comparison: pdtdm vs sarathi, aggregated trace replay.
# Short-window test mode → full 9.5h trace.
# rs=0.15, Qwen3-8B, TP=2.
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
PY=/usr/local/python3.11.13/bin/python3
SCRIPT=$ROOT/experiments/qps_sweep.py
OUT=$ROOT/results/maas_replay
TRACE=$ROOT/data/maas/aggregated_curve.csv        # full trace, 1140 rows, 1-min spacing
MODEL=/models/Qwen/Qwen3-8B

# ── Experiment parameters ──
RS=0.10
SEED=0
MML=8192
MAX_CONCURRENCY=64
WARMUP=30
SUBSAMPLE_STEP=15     # every 15th row → ~76 rows, 15-min bucket interval
COMPRESS_TIMELINE="--trace-compress-timeline"  # remove original trace gaps

# ── Window selection ──
# Set TEST_DURATION to run a short validation window (e.g. 600s).
# Leave empty ("") for the full 9.5h trace.
TEST_DURATION=""       # empty = full trace (34200s, subsampled → 1/10 = 1.9h)
if [ -n "$TEST_DURATION" ]; then
  DURATION=$TEST_DURATION
  START_OFFSET=${TEST_START_OFFSET:-0}
  RUN_ID="maas_test_rs${RS}"
else
  DURATION=34200      # 9.5h full trace (subsampled to 1/10)
  START_OFFSET=0
  RUN_ID="maas_full_rs${RS}_ss${SUBSAMPLE_STEP}"
fi

# CRITICAL: avoid sys.path namespace-package conflict
cd /tmp

# NPU environment
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

BASE_URL="http://127.0.0.1:8000"
TRACE_DIR=/tmp/tdm_maas_trace

mkdir -p "$OUT" "$TRACE_DIR"
echo "### MaaS replay started $(date '+%H:%M:%S')"
echo "### scale=$RS window=${DURATION}s start_offset=${START_OFFSET}s model=$MODEL max_concurrency=$MAX_CONCURRENCY"

# ── pdtdm (m31) ──
echo ""
echo "===== $(date '+%H:%M:%S') pdtdm (c2_tdm_m31_2048) ====="

fuser -k 8000/tcp 2>/dev/null || true
sleep 2

ASCEND_SCHEDULER_CLS="vllm_ascend.core.tdm.scheduler.TDMScheduler"
ADDITIONAL_CONFIG="{\"ascend_scheduler_config\":{\"enabled\":true,\"scheduler_cls\":\"${ASCEND_SCHEDULER_CLS}\"},\"tdm\":{\"enable_tdm\":true,\"static_ratio\":0.3,\"min_slice_iters\":2,\"max_slice_iters\":8,\"kv_free_watermark\":0.05,\"telemetry_enabled\":true,\"telemetry_dir\":\"${TRACE_DIR}\",\"run_id\":\"${RUN_ID}_pdtdm\",\"initial_phase\":\"prefill\",\"prefill_chunk_tokens\":2048}}"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len $MML \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --additional-config "$ADDITIONAL_CONFIG" \
  --port 8000 \
  > "$OUT/server_pdtdm.log" 2>&1 &

SERVER_PID=$!
echo "  Server PID=$SERVER_PID"

READY=0
for i in $(seq 1 300); do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    if grep -q "Bind_Failed\|Timed out waiting for engines\|ImportError" "$OUT/server_pdtdm.log" 2>/dev/null; then
      echo "  ERROR: engine core fatal error at ${i}s"
      grep -E "Bind_Failed|Timed out waiting for engines|ImportError" "$OUT/server_pdtdm.log" | head -5
      kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
      exit 1
    fi
    READY=1; echo "  Server ready after ${i}s"; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ERROR: server died at ${i}s"; tail -30 "$OUT/server_pdtdm.log"; exit 1
  fi
  sleep 1
done
[ $READY -eq 0 ] && { echo "  ERROR: not ready after 300s"; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit 1; }

echo "  Running driver..."
$PY "$SCRIPT" \
  --arrival-mode aggregated_trace --trace-file "$TRACE" --rpm-scale "$RS" \
  --trace-subsample-step $SUBSAMPLE_STEP $COMPRESS_TIMELINE \
  --duration $DURATION --warmup $WARMUP \
  --trace-start-offset $START_OFFSET \
  --base-url "$BASE_URL" --model "$MODEL" \
  --config-name "c2_tdm_m31_2048" \
  --out "$OUT/maas_pdtdm_rs${RS}.json" \
  --tracker-jsonl "$TRACE_DIR/${RUN_ID}_pdtdm_req.jsonl" \
  --seed $SEED \
  --max-concurrency $MAX_CONCURRENCY \
  --max-model-len $MML \
  --slo-ttft-ms 2808 --slo-tpot-ms 648 \
  2>&1 | tail -5

kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
echo "  pdtdm done."
sleep 3

# ── sarathi (c3 chunk=2048, passive tracker) ──
echo ""
echo "===== $(date '+%H:%M:%S') sarathi (c3_cp chunk=2048) ====="

fuser -k 8000/tcp 2>/dev/null || true
sleep 2

ADDITIONAL_CONFIG="{\"ascend_scheduler_config\":{\"enabled\":true,\"scheduler_cls\":\"${ASCEND_SCHEDULER_CLS}\",\"enable_chunked_prefill\":true},\"tdm\":{\"enable_tdm\":false,\"passive_tracker\":true,\"telemetry_enabled\":true,\"telemetry_dir\":\"${TRACE_DIR}\",\"run_id\":\"${RUN_ID}_sarathi\"}}"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len $MML \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --additional-config "$ADDITIONAL_CONFIG" \
  --port 8000 \
  > "$OUT/server_sarathi.log" 2>&1 &

SERVER_PID=$!
echo "  Server PID=$SERVER_PID"

READY=0
for i in $(seq 1 300); do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    if grep -q "Bind_Failed\|Timed out waiting for engines\|ImportError" "$OUT/server_sarathi.log" 2>/dev/null; then
      echo "  ERROR: engine core fatal error at ${i}s"
      grep -E "Bind_Failed|Timed out waiting for engines|ImportError" "$OUT/server_sarathi.log" | head -5
      kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
      exit 1
    fi
    READY=1; echo "  Server ready after ${i}s"; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ERROR: server died at ${i}s"; tail -30 "$OUT/server_sarathi.log"; exit 1
  fi
  sleep 1
done
[ $READY -eq 0 ] && { echo "  ERROR: not ready after 300s"; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit 1; }

echo "  Running driver..."
$PY "$SCRIPT" \
  --arrival-mode aggregated_trace --trace-file "$TRACE" --rpm-scale "$RS" \
  --trace-subsample-step $SUBSAMPLE_STEP $COMPRESS_TIMELINE \
  --duration $DURATION --warmup $WARMUP \
  --base-url "$BASE_URL" --model "$MODEL" \
  --config-name "c3_cp" \
  --trace-start-offset $START_OFFSET \
  --out "$OUT/maas_sarathi_rs${RS}.json" \
  --tracker-jsonl "$TRACE_DIR/${RUN_ID}_sarathi_req.jsonl" \
  --seed $SEED \
  --max-concurrency $MAX_CONCURRENCY \
  --max-model-len $MML \
  --slo-ttft-ms 2808 --slo-tpot-ms 648 \
  2>&1 | tail -5

kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
echo "  sarathi done."

# ── Summary ──
echo ""
echo "===== Results ====="
for TAG in pdtdm sarathi; do
  F=$OUT/maas_${TAG}_rs${RS}.json
  if [[ -f "$F" ]]; then
    $PY -c "
import json
d=json.load(open('$F'))
s=d['summary']
n,ok,err=s['n_submitted'],s['n_ok'],s['n_err']
w=s['window']
mch,meet=w['n_matched'],w['n_meet_slo']
mp=round(100*meet/mch,1) if mch else 0
tt99=w['ttft_ms']['p99']; tp99=w['tpot_ms_mean']['p99']
e99=w['e2e_ms']['p99']/1000
gp=w['output_throughput_tok_s']
print(f'${TAG}: n={n} ok={ok} err={err} | TTFT_p99={tt99:.0f}ms TPOT_p99={tp99:.0f}ms | e2e_p99={e99:.0f}s | meet={mp}% goodput={gp:.0f}tok/s')
"
  else
    echo "${TAG}: MISSING"
  fi
done
echo "### MaaS replay done $(date '+%H:%M:%S')"
