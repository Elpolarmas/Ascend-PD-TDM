#!/bin/bash
# F5b — decode-heavy with non-degenerate prefill (2026-05-27)
#
# 背景:F5 原 dh cell (prompt~64) prefill 远低于 chunk_budget=2048,
#       三 paradigm 在 prefill 路径行为坍缩 → "Δgoodput≈0" 无说服力。
#
# F5b 设计:prompt ≥ chunk_budget,output 仍主导总 compute(ratio 1:2)。
#
# Workloads(都 lognormal,sigma=0.3):
#   - dh_chunk1: prompt ~2048 (1 chunk) / output ~4096
#   - dh_chunk2: prompt ~4096 (2 chunk) / output ~8192
#
# Paradigms(F5b only,drop vanilla_cb):
#   - sarathi: c3_cp + max_num_batched_tokens=2048(chunk=2048)
#   - pdtdm:   c2_tdm_m31_2048 + max_num_batched_tokens=8192(m31-fix)
#
# Matrix:2 wl × 2 paradigm = 4 server lifecycles × 5 QPS = 20 cells
# Wall:~4 × ~25min = ~1.7h
#
# Outdir: results/f5b_decode_heavy_long_prompt/{wl}_{paradigm}_seed0/

set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/f5b_decode_heavy_long_prompt
TRACE_DIR=$ROOT/results/tdm_trace

DURATION=180
WARMUP=30
SEED=0
SLO_TTFT=500
SLO_TPOT=200

# ============================================================
# Workload params(lognormal mu,sigma=0.3 → 95%CI ≈ mean × [0.55, 1.82])
# ============================================================
# dh_chunk1: prompt mean=2048(mu=ln(2048)-0.045=7.58),output mean=4096(mu=8.27)
DH1_OPTS="--prompt-mu 7.58 --prompt-sigma 0.3 --prompt-min 1024 --prompt-max 4096 \
          --output-mu 8.27 --output-sigma 0.3 --output-min 2048 --output-max 8192"

# dh_chunk2: prompt mean=4096(mu=8.27),output mean=8192(mu=8.96)
DH2_OPTS="--prompt-mu 8.27 --prompt-sigma 0.3 --prompt-min 2048 --prompt-max 8192 \
          --output-mu 8.96 --output-sigma 0.3 --output-min 4096 --output-max 16384"

# ============================================================
# QPS sweep(per-req decode 长 → sat QPS 很低)
# ============================================================
# dh_chunk1: per-req decode = 4096 × 33ms ≈ 135s,batch=20 → sat ~0.15 req/s
# dh_chunk2: per-req decode = 8192 × 33ms ≈ 270s,batch=20 → sat ~0.07 req/s
DH1_QPS="0.05,0.10,0.15,0.20,0.25"
DH2_QPS="0.025,0.05,0.075,0.10,0.15"

# ============================================================
# max-model-len per wl(prompt_max + output_max)
# ============================================================
# dh_chunk1: 4096 + 8192 = 12288 → 取 16384 留 slack
# dh_chunk2: 8192 + 16384 = 24576 → 取 24576
DH1_MML=16384
DH2_MML=24576

# Note: max-num-seqs 用 vLLM 默认(sweep script 没暴露),OOM 时再 patch
run_tuple() {
  # $1=workload (dh1|dh2)  $2=paradigm (sarathi|pdtdm)
  local wl=$1 paradigm=$2
  local d=$OUT/${wl}_${paradigm}_seed${SEED}

  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi

  local qps_list wl_opts mml
  case "$wl" in
    dh1) qps_list=$DH1_QPS; wl_opts=$DH1_OPTS; mml=$DH1_MML ;;
    dh2) qps_list=$DH2_QPS; wl_opts=$DH2_OPTS; mml=$DH2_MML ;;
    *) echo "Unknown workload: $wl"; exit 1 ;;
  esac

  # cfg + max_num_batched_tokens
  # - sarathi: chunked prefill enabled,mb=2048 即 chunk size
  # - pdtdm:   Ascend scheduler 要求 mb >= max_model_len,内部用 prefill_chunk_tokens=2048 做真正切 chunk
  local cfg mb
  case "$paradigm" in
    sarathi) cfg="c3_cp";           mb=2048 ;;
    pdtdm)   cfg="c2_tdm_m31_2048"; mb=$mml ;;
    *) echo "Unknown paradigm: $paradigm"; exit 1 ;;
  esac

  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d"
  echo "===== paradigm=$paradigm cfg=$cfg mb=$mb mml=$mml qps=$qps_list"
  echo "==================================================================="

  $PY "$SCRIPT" \
    --configs "$cfg" \
    --qps "$qps_list" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len $mml --max-num-batched-tokens "$mb" \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode poisson \
    $wl_opts \
    --outdir "$d" --seed $SEED 2>&1 | tail -12

  for ST in iter req ctrl chunk; do
    F=$TRACE_DIR/qps_sweep_${cfg}_${ST}.jsonl
    [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${cfg}_${ST}.jsonl" || true
  done
}

# ============================================================
mkdir -p "$OUT"
echo "###################################################################"
echo "### F5b decode-heavy long-prompt sweep started $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "### Matrix: 2 wl × 2 paradigm = 4 lifecycles × 5 QPS = 20 cells"
echo "###################################################################"

for WL in dh1 dh2; do
  for PARADIGM in sarathi pdtdm; do
    run_tuple "$WL" "$PARADIGM"
  done
done

echo ""
echo "==================================================================="
echo "=== $(date '+%H:%M:%S') F5b sweep done ==="
echo "=== Out: $OUT"
echo "==================================================================="
