#!/bin/bash
# F5c — T6 补充:prefill-heavy MEDIUM output + balanced MEDIUM(2026-05-27)
#
# 触发:F5b 后发现 T6 现有 cell(conv / code)都在 output 极小区域,
#       paper §3 heatmap 中央"prefill 频繁 + decode batch 中等"格子缺数据。
#
# Workloads(lognormal sigma=0.3):
#   - B1 prefill-heavy: prompt~2048 / output~1024(ratio 2:1,prompt = 1 chunk)
#   - B2 balanced:      prompt~1024 / output~1024(ratio 1:1,prompt < 1 chunk)
#
# Paradigms(F5b 同款,drop vanilla_cb):
#   - sarathi: c3_cp + max_num_batched_tokens=2048(chunk=2048)
#   - pdtdm:   c2_tdm_m31_2048 + max_num_batched_tokens=8192(m31-fix)
#
# Matrix:2 wl × 2 paradigm × 5 QPS = 20 cell × ~25min lifecycle ≈ 1.5-1.7h
# Outdir:results/f5c_prefill_balanced_supp/{wl}_{paradigm}_seed0/

set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/f5c_prefill_balanced_supp
TRACE_DIR=$ROOT/results/tdm_trace

DURATION=180
WARMUP=30
SEED=0
SLO_TTFT=500
SLO_TPOT=200
MML=8192

# B1 prefill-heavy: prompt 2048 / output 1024(2:1)
# per-req decode wall = 1024 × ~30ms ≈ 30s,sat ~1.0 req/s
B1_OPTS="--prompt-mu 7.58 --prompt-sigma 0.3 --prompt-min 1024 --prompt-max 4096 \
         --output-mu 6.89 --output-sigma 0.3 --output-min 512 --output-max 2048"
B1_QPS="0.2,0.4,0.6,0.8,1.0"

# B2 balanced: prompt 1024 / output 1024(1:1)
B2_OPTS="--prompt-mu 6.89 --prompt-sigma 0.3 --prompt-min 512 --prompt-max 2048 \
         --output-mu 6.89 --output-sigma 0.3 --output-min 512 --output-max 2048"
B2_QPS="0.2,0.4,0.6,0.8,1.0"

run_tuple() {
  # $1 = wl (b1|b2)  $2 = paradigm (sarathi|pdtdm)
  local wl=$1 paradigm=$2
  local d=$OUT/${wl}_${paradigm}_seed${SEED}

  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi

  local qps_list wl_opts
  case "$wl" in
    b1) qps_list=$B1_QPS; wl_opts="$B1_OPTS" ;;
    b2) qps_list=$B2_QPS; wl_opts="$B2_OPTS" ;;
    *) echo "Unknown wl: $wl"; exit 1 ;;
  esac

  local cfg mb
  case "$paradigm" in
    sarathi) cfg="c3_cp";           mb=2048 ;;
    pdtdm)   cfg="c2_tdm_m31_2048"; mb=$MML ;;
    *) echo "Unknown paradigm: $paradigm"; exit 1 ;;
  esac

  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d  cfg=$cfg mb=$mb qps=$qps_list"
  echo "==================================================================="

  $PY "$SCRIPT" \
    --configs "$cfg" --qps "$qps_list" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len $MML --max-num-batched-tokens "$mb" \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode poisson \
    $wl_opts \
    --outdir "$d" --seed $SEED 2>&1 | tail -20 || echo "[warn] tuple $wl $paradigm rc=$?"

  for ST in iter req ctrl chunk; do
    F=$TRACE_DIR/qps_sweep_${cfg}_${ST}.jsonl
    [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${cfg}_${ST}.jsonl" || true
  done
}

mkdir -p "$OUT"
echo "###################################################################"
echo "### F5c prefill+balanced supp started $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "### Matrix: 2 wl × 2 paradigm × 5 QPS = 20 cells, ~1.5-1.7h"
echo "###################################################################"

for WL in b1 b2; do
  for P in sarathi pdtdm; do
    run_tuple "$WL" "$P"
  done
done

echo "###################################################################"
echo "### F5c done $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "###################################################################"
