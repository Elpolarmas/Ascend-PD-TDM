#!/bin/bash
# F5 — decode-heavy + balanced 合成 workload sweep(D-015 / 2026-05-26)
#
# 目的:回答导师 challenge #2 — Azure conv/code 实测都 prefill-leaning
#       (conv prompt:output=5.5×,code=73×),用合成 workload 验证
#       PD-TDM 优势是否 workload-specific。
#
# Hypothesis(D-015 first-principles 柱 2 实证后):
#   - decode-heavy 上 PD-TDM vs Sarathi 优势缩小(mixed batch tax 在大 decode
#     批量上 amortize 到 ~11%),但不会消失
#   - balanced 上 PD-TDM 优势位于 conv 和 decode-heavy 之间
#
# Workload:
#   - dh (decode-heavy):prompt ~64 (1:16) / output ~1024
#   - bal (balanced):  prompt ~512 / output ~512
#
# Paradigms(orchestrator 一次 server lifecycle 内 sweep 所有 QPS):
#   - vanilla_cb: c3_cp + max_num_batched_tokens=8192(老 c3 等价)
#   - sarathi:    c3_cp + max_num_batched_tokens=2048
#   - pdtdm:      c2_tdm_m31_2048 + max_num_batched_tokens=8192
#
# Matrix:2 wl × 3 paradigm = 6 server lifecycles,each sweeps 5 QPS
#         = 30 client measurements
# Wall:~6 × ~18min = ~1.8h(180s × 5 + warmup + server startup)
# SLO grid:s1/s2/s3 post-hoc reclassify(server 不读 SLO)
#
# Outdir 结构:
#   results/f5_decode_heavy_synth/{wl}_{paradigm}_seed0/
#     ├── {cfg}_qps0.2.json
#     ├── {cfg}_qps0.4.json
#     ├── ...
#     ├── qps_sweep_summary.json
#     └── tdm_trace/ ...

set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/f5_decode_heavy_synth
TRACE_DIR=$ROOT/results/tdm_trace

DURATION=180
WARMUP=30
SEED=0
# SLO 走任意值(post-hoc reclassify)
SLO_TTFT=500
SLO_TPOT=200

# ============================================================
# Workload 形状(lognormal mu,sigma)
# ============================================================
# dh:  prompt ~64,output ~1024(95% prompt [50,80],95% output [800,1300])
# bal: prompt ~512,output ~512(95% [310, 850] both)
DH_OPTS="--prompt-mu 4.16 --prompt-sigma 0.3 --prompt-min 32 --prompt-max 128 \
         --output-mu 6.93 --output-sigma 0.3 --output-min 512 --output-max 2048"
BAL_OPTS="--prompt-mu 6.24 --prompt-sigma 0.5 --prompt-min 128 --prompt-max 1024 \
          --output-mu 6.24 --output-sigma 0.5 --output-min 128 --output-max 1024"

# ============================================================
# QPS 扫描(覆盖 idle → saturate)
# ============================================================
# dh:  per-req decode = 1024 × 33ms ≈ 33s,batch=20 → sat ~0.6 req/s
# bal: per-req decode = 512 × 33ms ≈ 17s,batch=20 → sat ~1.2 req/s
DH_QPS="0.2,0.4,0.6,0.8,1.0"
BAL_QPS="0.5,1.0,1.5,2.0,2.5"

# ============================================================
# Run one (workload, paradigm) tuple: server lifecycle × QPS sweep
# ============================================================
run_tuple() {
  # $1=workload (dh|bal)  $2=paradigm (vanilla_cb|sarathi|pdtdm)
  local wl=$1 paradigm=$2
  local d=$OUT/${wl}_${paradigm}_seed${SEED}

  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi

  # paradigm → config + max_num_batched_tokens
  local cfg mb
  case "$paradigm" in
    vanilla_cb) cfg="c3_cp";              mb=8192 ;;
    sarathi)    cfg="c3_cp";              mb=2048 ;;
    pdtdm)      cfg="c2_tdm_m31_2048";    mb=8192 ;;
    *) echo "Unknown paradigm: $paradigm"; exit 1 ;;
  esac

  # workload → QPS list + prompt/output args
  local qps_list wl_opts
  case "$wl" in
    dh)  qps_list=$DH_QPS;  wl_opts=$DH_OPTS ;;
    bal) qps_list=$BAL_QPS; wl_opts=$BAL_OPTS ;;
    *) echo "Unknown workload: $wl"; exit 1 ;;
  esac

  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d"
  echo "===== paradigm=$paradigm cfg=$cfg mb=$mb qps=$qps_list"
  echo "==================================================================="

  $PY "$SCRIPT" \
    --configs "$cfg" \
    --qps "$qps_list" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len 8192 --max-num-batched-tokens "$mb" \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode poisson \
    $wl_opts \
    --outdir "$d" --seed $SEED 2>&1 | tail -12

  # copy server-side tdm_trace
  for ST in iter req ctrl chunk; do
    F=$TRACE_DIR/qps_sweep_${cfg}_${ST}.jsonl
    [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${cfg}_${ST}.jsonl" || true
  done
}

# ============================================================
# Main
# ============================================================
mkdir -p "$OUT"
echo "###################################################################"
echo "### F5 decode-heavy + balanced sweep started $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "### Matrix: 2 wl × 3 paradigm = 6 server lifecycles × 5 QPS"
echo "###################################################################"

for WL in dh bal; do
  for PARADIGM in vanilla_cb sarathi pdtdm; do
    run_tuple "$WL" "$PARADIGM"
  done
done

echo ""
echo "==================================================================="
echo "=== $(date '+%H:%M:%S') F5 sweep done ==="
echo "=== Out: $OUT"
echo "==================================================================="
