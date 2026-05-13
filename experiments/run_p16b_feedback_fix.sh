#!/usr/bin/env bash
# P1.6b — 反馈链滞后修复(F1 ttft 实时 emit + F2 in-flight tpot)
#
# 方案 A(最小验证):只跑 M3.1(已含 F1+F2)单配置,验证 PID 修反馈链后
# 是否真动态。跟 azure_main 历史的 M3.1 ctrl.jsonl 直接 diff,**无需重测
# baseline**——这是 M3.1 自身内部行为对照,关键看:
#   1. n_ttft_window warm 时刻(应从 13-17s 降到 <1s)
#   2. ratio 时序(不再 27s 才到 ratio_max,可能根本不到上界)
#   3. tpot_saturated 屏蔽占比(从 99.4% 降到合理值)
#   4. |err_tpot_effective| 非零占比(双输入是否复活)
#
# 规模:2 windows × 1 config × 3 seeds × 60s = 6 driver calls
#      每 call wall ~22min → 全跑完 wall ~2.2h
#
# done 标准:
#   PASS — ratio 不再钉 0.8 持续 50s+;tpot 屏蔽占比 < 80%;PID delta 有
#          双向 oscillation 形态 → 反馈链是根因,救回闭环反馈叙事
#   FAIL — ratio 仍 90%+ 时间钉 ratio_max → 控制律本身不适配,P1.6c 走 β

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16b
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048"

mkdir -p "$OUT"

declare -a WINDOWS=(
  "conv_w1:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "code_w1:$DATA/AzureLLMInferenceTrace_code.csv:570"
)

for ENTRY in "${WINDOWS[@]}"; do
  IFS=':' read -r WLABEL TCSV TOFFSET <<< "$ENTRY"
  for SEED in 0 1 2; do
    D=$OUT/${WLABEL}_seed${SEED}
    if [[ -f "$D/qps_sweep_summary.json" ]]; then
      echo "[skip] $D already complete"
      continue
    fi
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== window=$WLABEL seed=$SEED start_offset=${TOFFSET}s"
    echo "==================================================================="
    /usr/local/python3.11.13/bin/python3 \
      "$ROOT/experiments/run_qps_sweep_all.py" \
      --configs "$CONFIGS" \
      --qps 0 \
      --duration 60 --warmup 20 \
      --max-model-len 8192 --max-num-batched-tokens 8192 \
      --arrival-mode trace \
      --trace-file "$TCSV" \
      --trace-start-offset "$TOFFSET" \
      --trace-max-prompt-tokens 7000 \
      --trace-max-output-tokens 600 \
      --outdir "$D" \
      --seed "$SEED" \
      2>&1 | tee "$D/run.log"
    for CFG in c2_tdm_m31_2048; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.6b done ==="
