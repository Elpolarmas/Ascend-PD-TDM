#!/usr/bin/env bash
# P1.6f — target_violation_rate sweep
#
# 目的:直接 attack PID ReLU+M2.4 联合屏蔽。当前 target=5% 让"双输入工作区"
# 是极窄中间带,调高 target 让窄区变宽,看 PID 是否真双输入复活。
#
# Sweep target_violation_rate ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
# 用 P1.6e 校准的 slo_tpot_ms=200(conv sweet spot),保持其它参数默认。
#
# 5 档 × 2 windows × 3 seeds × 60s = 30 driver calls × ~4min = ~2h wall
#
# 关键观察:
#   - err_tpot_eff 非零率 是否随 target 上升而上升(双输入复活的关键指标)
#   - ratio 钉 max 占比是否下降(>20% 非零率 + <80% 钉 max → PASS)
#
# 后台启动:
#   nohup bash experiments/run_p16f_target_viol_sweep.sh \
#       > results/azure_p16f_orchestrator.log 2>&1 &
# Posthoc:
#   python3 experiments/posthoc_p16f_target_viol.py

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16f
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048"
SLO_TPOT=200  # P1.6e 校准的 conv sweet spot

mkdir -p "$OUT"

declare -a TARGETS=(0.05 0.10 0.20 0.30 0.50)
declare -a WINDOWS=(
  "conv_w1:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "code_w1:$DATA/AzureLLMInferenceTrace_code.csv:570"
)

for TARGET in "${TARGETS[@]}"; do
  # 文件名安全:0.05 → 005
  T_TAG=$(echo "$TARGET" | sed 's/\.//')
  for ENTRY in "${WINDOWS[@]}"; do
    IFS=':' read -r WLABEL TCSV TOFFSET <<< "$ENTRY"
    for SEED in 0 1 2; do
      D=$OUT/tgt${T_TAG}/${WLABEL}_seed${SEED}
      if [[ -f "$D/qps_sweep_summary.json" ]]; then
        echo "[skip] $D already complete"
        continue
      fi
      mkdir -p "$D/tdm_trace"
      echo "==================================================================="
      echo "===== target_viol=${TARGET}  window=$WLABEL  seed=$SEED"
      echo "==================================================================="
      /usr/local/python3.11.13/bin/python3 \
        "$ROOT/experiments/run_qps_sweep_all.py" \
        --configs "$CONFIGS" \
        --qps 0 \
        --duration 60 --warmup 20 \
        --max-model-len 8192 --max-num-batched-tokens 8192 \
        --slo-ttft-ms 500 \
        --slo-tpot-ms "$SLO_TPOT" \
        --slo-target-violation-rate "$TARGET" \
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

echo "=== P1.6f done ==="
