#!/bin/bash
# F5b A2 only — dh_chunk2 (prompt~4096 / output~8192)
# 2 paradigm × 5 QPS = 10 cell,wall ~50min
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/f5b_decode_heavy_long_prompt
TRACE_DIR=$ROOT/results/tdm_trace

DURATION=180
WARMUP=30
SEED=0
MML=24576

DH2_OPTS="--prompt-mu 8.27 --prompt-sigma 0.3 --prompt-min 2048 --prompt-max 8192 \
          --output-mu 8.96 --output-sigma 0.3 --output-min 4096 --output-max 16384"
DH2_QPS="0.025,0.05,0.075,0.10,0.15"

run_tuple() {
  local paradigm=$1
  local d=$OUT/dh2_${paradigm}_seed${SEED}
  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $d"
    return 0
  fi

  local cfg mb
  case "$paradigm" in
    sarathi) cfg="c3_cp";           mb=2048 ;;
    pdtdm)   cfg="c2_tdm_m31_2048"; mb=$MML ;;
  esac

  mkdir -p "$d/tdm_trace"
  echo "===== $(date '+%H:%M:%S') $d  cfg=$cfg mb=$mb ====="
  $PY "$SCRIPT" \
    --configs "$cfg" --qps "$DH2_QPS" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len $MML --max-num-batched-tokens "$mb" \
    --slo-ttft-ms 500 --slo-tpot-ms 200 \
    --arrival-mode poisson \
    $DH2_OPTS \
    --outdir "$d" --seed $SEED 2>&1 | tail -20

  for ST in iter req ctrl chunk; do
    F=$TRACE_DIR/qps_sweep_${cfg}_${ST}.jsonl
    [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${cfg}_${ST}.jsonl" || true
  done
}

mkdir -p "$OUT"
echo "### F5b A2 (dh_chunk2) started $(date '+%H:%M:%S')"
for P in sarathi pdtdm; do
  run_tuple "$P"
done
echo "### F5b A2 done $(date '+%H:%M:%S')"
