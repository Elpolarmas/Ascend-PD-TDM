#!/usr/bin/env bash
# P1.6e — SLO target sweep
#
# 验证 workload-conditional SLO 救回 PID 闭环反馈。
# 当前 slo_tpot_ms=50ms 物理不可达(conv p50=148ms, code p50=453ms),
# PID 看 100% 违例 → M2.4 屏蔽 → 单输入钉 ratio_max。
#
# Sweep slo_tpot_ms ∈ {150, 200, 300},复用 azure_p16b 当 50 档。
# 3 档 × 2 windows × 3 seeds × 60s = 18 driver calls,wall ~6h。
#
# 关键观察(ctrl.jsonl + req.jsonl):
#   1. ratio 是否真动态(不再钉 ratio_max)
#   2. tpot_sat 屏蔽占比 < 80%
#   3. err_tpot_eff 非零率上升(双输入复活)
#   4. evaluation 多档 meet_slo% 在不同 SLO target 下的变化

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16e
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048"

mkdir -p "$OUT"

declare -a SLO_TPOTS=(150 200 300)
declare -a WINDOWS=(
  "conv_w1:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "code_w1:$DATA/AzureLLMInferenceTrace_code.csv:570"
)

for SLO_TPOT in "${SLO_TPOTS[@]}"; do
  for ENTRY in "${WINDOWS[@]}"; do
    IFS=':' read -r WLABEL TCSV TOFFSET <<< "$ENTRY"
    for SEED in 0 1 2; do
      D=$OUT/tpot${SLO_TPOT}/${WLABEL}_seed${SEED}
      if [[ -f "$D/qps_sweep_summary.json" ]]; then
        echo "[skip] $D already complete"
        continue
      fi
      mkdir -p "$D/tdm_trace"
      echo "==================================================================="
      echo "===== slo_tpot=${SLO_TPOT}ms  window=$WLABEL  seed=$SEED"
      echo "==================================================================="
      /usr/local/python3.11.13/bin/python3 \
        "$ROOT/experiments/run_qps_sweep_all.py" \
        --configs "$CONFIGS" \
        --qps 0 \
        --duration 60 --warmup 20 \
        --max-model-len 8192 --max-num-batched-tokens 8192 \
        --slo-ttft-ms 500 \
        --slo-tpot-ms "$SLO_TPOT" \
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

echo "=== P1.6e done ==="
