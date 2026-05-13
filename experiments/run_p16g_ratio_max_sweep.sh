#!/usr/bin/env bash
# P1.6g — ratio_max sweep
#
# 目的:验证 ratio_max=0.8 是真物理天花板还是设计选择。放宽到 0.9/0.95/1.0
# 看 ratio 是否飞出去(让 PID 真"动态"但代价是 ratio 经常 100% 全 prefill)。
# 收紧到 0.5/0.7 看 ratio 是否仍钉新边界。
#
# Sweep ratio_max ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 1.0}
# 用 P1.6e 校准的 slo_tpot_ms=200,target=0.05(默认)。
#
# 6 档 × 2 windows × 3 seeds × 60s = 36 driver calls × ~4min = ~2.4h wall
#
# 关键观察:
#   - 不同 ratio_max 下 ratio 时序形态
#   - 0.5/0.7 是否仍钉新边界(确认 PID 单向推力)
#   - 0.95/1.0 时 meet_slo% 跟 0.8 比是否退步(过度 prefill 伤 decode)
#
# 后台启动:
#   nohup bash experiments/run_p16g_ratio_max_sweep.sh \
#       > results/azure_p16g_orchestrator.log 2>&1 &
# Posthoc:
#   python3 experiments/posthoc_p16g_ratio_max.py

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16g
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048"
SLO_TPOT=200

mkdir -p "$OUT"

declare -a RATIO_MAXES=(0.5 0.7 0.8 0.9 0.95 1.0)
declare -a WINDOWS=(
  "conv_w1:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "code_w1:$DATA/AzureLLMInferenceTrace_code.csv:570"
)

for RMAX in "${RATIO_MAXES[@]}"; do
  R_TAG=$(echo "$RMAX" | sed 's/\.//')
  for ENTRY in "${WINDOWS[@]}"; do
    IFS=':' read -r WLABEL TCSV TOFFSET <<< "$ENTRY"
    for SEED in 0 1 2; do
      D=$OUT/rmax${R_TAG}/${WLABEL}_seed${SEED}
      if [[ -f "$D/qps_sweep_summary.json" ]]; then
        echo "[skip] $D already complete"
        continue
      fi
      mkdir -p "$D/tdm_trace"
      echo "==================================================================="
      echo "===== ratio_max=${RMAX}  window=$WLABEL  seed=$SEED"
      echo "==================================================================="
      /usr/local/python3.11.13/bin/python3 \
        "$ROOT/experiments/run_qps_sweep_all.py" \
        --configs "$CONFIGS" \
        --qps 0 \
        --duration 60 --warmup 20 \
        --max-model-len 8192 --max-num-batched-tokens 8192 \
        --slo-ttft-ms 500 \
        --slo-tpot-ms "$SLO_TPOT" \
        --slo-ratio-max "$RMAX" \
        --arrival-mode trace \
        --trace-file "$TCSV" \
        --trace-start-offset "$TOFFSET" \
        --trace-max-prompt-tokens 7000 \
        --trace-max-output-tokens 600 \
        --outdir "$D" \
        --seed "$SEED" \
        2>&1 | tee "$D/run.log"
      for CFG in $CONFIGS; do
        for ST in iter req ctrl chunk; do
          F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
          [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
        done
      done
    done
  done
done

echo "=== P1.6g done ==="
