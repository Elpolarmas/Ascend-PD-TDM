#!/bin/bash
# MaaS comparison: pdtdm vs sarathi using Poisson arrival + trace-sampled lengths.
# Paper-standard method: sweep QPS points, measure goodput at SLO.
# Estimated: 8 runs x 300s + server startup ≈ 45min.
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
PY=/usr/local/python3.11.13/bin/python3
SCRIPT=$ROOT/experiments/qps_sweep.py
OUT=$ROOT/results/maas_replay
TRACE=$ROOT/data/maas/aggregated_curve_sampled.csv
MODEL=/models/Qwen/Qwen3-8B

DURATION=300
WARMUP=30
MML=8192; MB=8192
SEED=0
SLO_TTFT=2808; SLO_TPOT=648

cd /tmp
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
BASE_URL="http://127.0.0.1:8000"
ASCEND_CLS="vllm_ascend.core.tdm.scheduler.TDMScheduler"
TRACE_DIR=/tmp/tdm_maas_poisson

mkdir -p "$OUT" "$TRACE_DIR"
echo "### Poisson MaaS comparison $(date '+%H:%M:%S')"
echo "### QPS sweep: 6 8 10 12 × 2 paradigms, 300s each"

for PARADIGM in pdtdm sarathi; do
  if [ "$PARADIGM" = "pdtdm" ]; then
    CONFIG="c2_tdm_m31_2048"
    TDM_ENABLE="true"
    CHUNK_PREFILL=""
    BATCHED=8192
    ADDITIONAL="{\"ascend_scheduler_config\":{\"enabled\":true,\"scheduler_cls\":\"${ASCEND_CLS}\"},\"tdm\":{\"enable_tdm\":true,\"static_ratio\":0.3,\"min_slice_iters\":2,\"max_slice_iters\":8,\"kv_free_watermark\":0.05,\"telemetry_enabled\":true,\"telemetry_dir\":\"${TRACE_DIR}\",\"run_id\":\"maas_poisson_pdtdm\",\"initial_phase\":\"prefill\",\"prefill_chunk_tokens\":2048}}"
  else
    CONFIG="c3_cp"
    TDM_ENABLE="false"
    CHUNK_PREFILL=',"enable_chunked_prefill":true'
    BATCHED=2048
    ADDITIONAL="{\"ascend_scheduler_config\":{\"enabled\":true,\"scheduler_cls\":\"${ASCEND_CLS}\"${CHUNK_PREFILL}},\"tdm\":{\"enable_tdm\":false,\"passive_tracker\":true,\"telemetry_enabled\":true,\"telemetry_dir\":\"${TRACE_DIR}\",\"run_id\":\"maas_poisson_sarathi\"}}"
  fi

  echo ""
  echo "===== $(date '+%H:%M:%S') $PARADIGM ($CONFIG) ====="
  fuser -k 8000/tcp 2>/dev/null || true; sleep 3

  $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --tensor-parallel-size 2 \
    --max-model-len $MML --max-num-batched-tokens $BATCHED \
    --gpu-memory-utilization 0.85 \
    --additional-config "$ADDITIONAL" --port 8000 \
    > "$OUT/server_${PARADIGM}.log" 2>&1 &
  SRV=$!
  echo "  Server PID=$SRV"

  for i in $(seq 1 300); do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
      if grep -q "Bind_Failed\|Timed out\|ImportError" "$OUT/server_${PARADIGM}.log" 2>/dev/null; then
        echo "  FATAL: engine error"; grep -E "Bind|Timed out|ImportErr" "$OUT/server_${PARADIGM}.log" | head -3; kill $SRV 2>/dev/null; exit 1
      fi
      echo "  Ready after ${i}s"; break
    fi
    if ! kill -0 $SRV 2>/dev/null; then echo "  Server died"; tail -5 "$OUT/server_${PARADIGM}.log"; exit 1; fi
    sleep 1
  done

  for QPS in 6 8 10 12; do
    echo "  QPS=$QPS..."
    $PY "$SCRIPT" \
      --arrival-mode trace_sampled \
      --trace-file "$TRACE" --qps "$QPS" \
      --duration $DURATION --warmup $WARMUP \
      --base-url "$BASE_URL" --model "$MODEL" \
      --config-name "$CONFIG" \
      --out "$OUT/maas_${PARADIGM}_qps${QPS}.json" \
      --seed $SEED \
      --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
      2>&1 | tail -1
  done

  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
  echo "  $PARADIGM done."
  sleep 5
done

echo ""
echo "===== Results ====="
for QPS in 6 8 10 12; do
  for P in pdtdm sarathi; do
    F="$OUT/maas_${P}_qps${QPS}.json"
    if [ -f "$F" ]; then
      $PY -c "
import json; d=json.load(open('$F')); s=d['summary']; w=s['window']
n,ok,err=s['n_submitted'],s['n_ok'],s['n_err']
mch,meet=w['n_matched'],w['n_meet_slo']
mp=round(100*meet/mch,1) if mch else 0
tt99=w['ttft_ms']['p99']; tp99=w['tpot_ms_mean']['p99']
print(f'${P} qps=${QPS}: n={n} ok={ok} err={err} meet={mp}% TTFTp99={tt99:.0f}ms TPOTp99={tp99:.0f}ms gp={w[\"output_throughput_tok_s\"]:.0f}tok/s')"
    fi
  done
done
echo "### Poisson sweep done $(date '+%H:%M:%S')"