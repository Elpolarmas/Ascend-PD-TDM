#!/usr/bin/env bash
# Phase 2 T6 — goodput-at-SLO Pareto main figure
# Paper Section 5.2 主图数据源。
#
# Arrival = trace_sampled_burst (固定 burst 形状 calibrated from real trace,
#   k 倍 scale (high,low) 同步以扫 avg_qps,保持 burst structure)
# Configs: c1_baseline + c3_cp (1 SLO,post-hoc reclassify) + c2_tdm_m31_2048 (4 SLO)
# Workload: conv (短 prompt 长 output) + code (长 prompt 短 output)
# SLO grid: T2 attainment heatmap 数据驱动选档
# Duration 90s + warmup 30s = 9 个 burst 周期(period=10s)
# Seeds: 0, 1, 2
#
# 规模: nonpid 42 inv × 2 cfg + pid 168 inv = 252 server lifecycles × ~255s ≈ 17.8h wall
#
# c1/c3 在 nonpid pass 跑 conv 用 (200,120) / code 用 (500,200) 一档;
# post-hoc 用 per-req tracker 在其他 3 档 SLO 上 reclassify。

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/phase_2_t6_burst_goodput
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
DURATION=90
WARMUP=30
PERIOD=10

mkdir -p "$OUT"

# ============================================================
# Calibration (from TraceSampledBurst docstring)
# ============================================================
# conv base: high=12  low=5  frac=0.1  (avg ≈ 5.7,real conv avg 5.5)
# code base: high=10  low=1  frac=0.2  (avg ≈ 2.8,real code avg 2.6)

# k-scale on (high, low) to sweep avg_qps while holding burst shape
CONV_KS=(0.5 1.0 1.4 1.8 2.2 2.6 3.0)
CODE_KS=(0.7 1.4 2.1 2.8 3.5 4.2 4.9)

# ============================================================
# SLO grid (data-driven from T2 attainment heatmap)
# ============================================================
# conv: 4 档 from 严 (paradigm Δ 最大) 到 松 (优势衰减)
# Format: "ttft_ms tpot_ms tier_label"
declare -a CONV_SLOS=(
  "200 120 s1"   # 严档,m31 vs c1 +6-13pp
  "300 150 s2"   # 中紧
  "500 200 s3"   # 中松
  "1000 200 s4"  # 松档
)
declare -a CODE_SLOS=(
  "500 200 s1"   # m31 sweet spot (+8pp)
  "500 700 s2"   # 大 m31 优势 (+29pp)
  "2000 400 s3"  # 中等档 (+3.7pp)
  "3000 1500 s4" # c3 winning regime (m31 -8pp,诚实降级)
)

# nonpid 一档 SLO(不影响 c1/c3 调度,post-hoc reclassify 其他档)
CONV_NONPID_TTFT=200
CONV_NONPID_TPOT=120
CODE_NONPID_TTFT=500
CODE_NONPID_TPOT=200

# Workload trace files
CONV_CSV=$DATA/AzureLLMInferenceTrace_conv.csv
CODE_CSV=$DATA/AzureLLMInferenceTrace_code.csv

# ============================================================
# Helpers
# ============================================================
compute_burst() {
  # $1=workload $2=k → echo "high low frac"
  local wl=$1 k=$2
  if [[ "$wl" == "conv" ]]; then
    awk -v k="$k" 'BEGIN { printf "%.4f %.4f 0.1\n", 12*k, 5*k }'
  else
    awk -v k="$k" 'BEGIN { printf "%.4f %.4f 0.2\n", 10*k, 1*k }'
  fi
}

run_one() {
  # $1=outdir $2=configs $3=workload-csv $4=high $5=low $6=frac $7=slo_ttft $8=slo_tpot $9=seed
  local d=$1 cfgs=$2 csv=$3 hi=$4 lo=$5 fr=$6 stt=$7 spt=$8 sd=$9
  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $d already complete"
    return 0
  fi
  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $d"
  echo "===== cfgs=$cfgs  burst high=$hi low=$lo frac=$fr  SLO=($stt/$spt)  seed=$sd"
  echo "==================================================================="
  /usr/local/python3.11.13/bin/python3 \
    "$ROOT/experiments/run_qps_sweep_all.py" \
    --configs "$cfgs" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len 8192 --max-num-batched-tokens 8192 \
    --slo-ttft-ms "$stt" --slo-tpot-ms "$spt" \
    --arrival-mode trace_sampled_burst \
    --trace-file "$csv" \
    --trace-max-prompt-tokens 7000 \
    --trace-max-output-tokens 600 \
    --burst-period $PERIOD \
    --burst-high-qps "$hi" \
    --burst-low-qps "$lo" \
    --burst-high-frac "$fr" \
    --outdir "$d" \
    --seed "$sd" \
    2>&1 | tee "$d/run.log"
  for CFG in c1_baseline c3_cp c2_tdm_m31_2048; do
    for ST in iter req ctrl chunk; do
      F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
    done
  done
}

# ============================================================
# Pass 1: nonpid (c1 + c3),1 SLO each,7 k × 3 seed × 2 wl = 42 invocations
# ============================================================
echo ""
echo "###################################################################"
echo "### Pass 1: c1+c3 nonpid (42 invocations × 2 cfgs each)"
echo "###################################################################"

for SEED in 0 1 2; do
  # conv
  for k in "${CONV_KS[@]}"; do
    read hi lo fr <<< "$(compute_burst conv $k)"
    d=$OUT/conv_k${k}_seed${SEED}_nonpid
    run_one "$d" "c1_baseline,c3_cp" "$CONV_CSV" "$hi" "$lo" "$fr" \
            "$CONV_NONPID_TTFT" "$CONV_NONPID_TPOT" "$SEED"
  done
  # code
  for k in "${CODE_KS[@]}"; do
    read hi lo fr <<< "$(compute_burst code $k)"
    d=$OUT/code_k${k}_seed${SEED}_nonpid
    run_one "$d" "c1_baseline,c3_cp" "$CODE_CSV" "$hi" "$lo" "$fr" \
            "$CODE_NONPID_TTFT" "$CODE_NONPID_TPOT" "$SEED"
  done
done

# ============================================================
# Pass 2: m31 PID,4 SLO × 7 k × 3 seed × 2 wl = 168 invocations
# ============================================================
echo ""
echo "###################################################################"
echo "### Pass 2: m31_2048 PID (168 invocations)"
echo "###################################################################"

for SEED in 0 1 2; do
  for SLO_ENTRY in "${CONV_SLOS[@]}"; do
    read STT SPT LABEL <<< "$SLO_ENTRY"
    for k in "${CONV_KS[@]}"; do
      read hi lo fr <<< "$(compute_burst conv $k)"
      d=$OUT/conv_k${k}_${LABEL}_seed${SEED}_pid
      run_one "$d" "c2_tdm_m31_2048" "$CONV_CSV" "$hi" "$lo" "$fr" \
              "$STT" "$SPT" "$SEED"
    done
  done
  for SLO_ENTRY in "${CODE_SLOS[@]}"; do
    read STT SPT LABEL <<< "$SLO_ENTRY"
    for k in "${CODE_KS[@]}"; do
      read hi lo fr <<< "$(compute_burst code $k)"
      d=$OUT/code_k${k}_${LABEL}_seed${SEED}_pid
      run_one "$d" "c2_tdm_m31_2048" "$CODE_CSV" "$hi" "$lo" "$fr" \
              "$STT" "$SPT" "$SEED"
    done
  done
done

echo ""
echo "==================================================================="
echo "=== Phase 2 T6 burst goodput sweep done ==="
echo "Run: experiments/posthoc_phase_2_t6.py 出 goodput-at-SLO Pareto 主图"
echo "==================================================================="
