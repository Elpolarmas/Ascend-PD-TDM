#!/bin/bash
# c3@chunk=2048 fair Sarathi-Serve baseline 补跑(2026-05-23)
# 只补 c3,c1 + m31 复用 T6 数据。
# Matrix: 2 wl × 7 k × 3 seed = 42 server lifecycles ≈ 3h wall。
# c3 不读 runtime SLO,joined 数据 post-hoc reclassify 4 档 SLO。
# 支持断点续跑(若 qps_sweep_summary.json 存在则 skip)。
set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/c3_chunk2048_supplement

DURATION=90
WARMUP=30
PERIOD=10
CONV_CSV=$ROOT/data/azure_trace/AzureLLMInferenceTrace_conv.csv
CODE_CSV=$ROOT/data/azure_trace/AzureLLMInferenceTrace_code.csv
CONV_KS=(0.5 1.0 1.4 1.8 2.2 2.6 3.0)
CODE_KS=(0.7 1.4 2.1 2.8 3.5 4.2 4.9)
# c3 不读 SLO,用任意值即可(post-hoc 会 reclassify)
SLO_TTFT=500
SLO_TPOT=200

# Burst calibration(同 T6)
compute_burst() {
  local wl=$1 k=$2
  if [[ "$wl" == "conv" ]]; then
    awk -v k="$k" 'BEGIN { printf "%.4f %.4f 0.1\n", 12*k, 5*k }'
  else
    awk -v k="$k" 'BEGIN { printf "%.4f %.4f 0.2\n", 10*k, 1*k }'
  fi
}

run_one() {
  local d=$1 csv=$2 hi=$3 lo=$4 fr=$5 sd=$6
  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi
  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d"
  echo "===== c3@chunk=2048  burst high=$hi low=$lo frac=$fr  seed=$sd"
  echo "==================================================================="
  $PY "$SCRIPT" \
    --configs c3_cp \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len 8192 --max-num-batched-tokens 2048 \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode trace_sampled_burst \
    --trace-file "$csv" \
    --trace-max-prompt-tokens 7000 --trace-max-output-tokens 600 \
    --burst-period $PERIOD \
    --burst-high-qps "$hi" --burst-low-qps "$lo" --burst-high-frac "$fr" \
    --outdir "$d" --seed "$sd" 2>&1 | tail -6
}

# 复用 smoke test 已跑数据(避免重复)
copy_if_exists() {
  local src=$1 dst=$2
  if [[ -f "$src/qps_sweep_summary.json" && ! -f "$dst/qps_sweep_summary.json" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    echo "[reuse] $src → $dst"
  fi
}

SMOKE=$ROOT/results/c3_chunk_smoke
copy_if_exists "$SMOKE/conv_k2.2_chunk2048_seed0" "$OUT/conv_k2.2_seed0"
copy_if_exists "$SMOKE/conv_k2.2_chunk2048_seed1" "$OUT/conv_k2.2_seed1"
copy_if_exists "$SMOKE/conv_k2.2_chunk2048_seed2" "$OUT/conv_k2.2_seed2"
copy_if_exists "$SMOKE/code_k2.1_chunk2048_seed0" "$OUT/code_k2.1_seed0"
copy_if_exists "$SMOKE/code_k4.9_chunk2048_seed0" "$OUT/code_k4.9_seed0"

for SEED in 0 1 2; do
  for k in "${CONV_KS[@]}"; do
    read hi lo fr <<< "$(compute_burst conv $k)"
    run_one "$OUT/conv_k${k}_seed${SEED}" "$CONV_CSV" "$hi" "$lo" "$fr" "$SEED"
  done
  for k in "${CODE_KS[@]}"; do
    read hi lo fr <<< "$(compute_burst code $k)"
    run_one "$OUT/code_k${k}_seed${SEED}" "$CODE_CSV" "$hi" "$lo" "$fr" "$SEED"
  done
done

echo "==================================================================="
echo "=== $(date '+%H:%M:%S') c3@chunk=2048 supplement done ==="
echo "=== outdir: $OUT"
echo "=== next: 改 posthoc_phase_2_t6.py 让 c3 从这里读 ==="
echo "==================================================================="
