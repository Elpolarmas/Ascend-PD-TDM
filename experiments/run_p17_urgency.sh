#!/usr/bin/env bash
# P1.7 leading indicator(TTFT urgency)初探。
#
# 在 P1.5 锚点窗位(conv@1860, code@570)上对比 M3.1 vs M3.2(=M3.1 + urgency)。
# 同时保留 c1_baseline / c2_tdm_m1_chunk2048 / c3_cp 让 posthoc 可复用
# P1.5 的对比格式。
#
# 规模:2 windows × 5 configs × 3 seeds × 60s = 6 driver calls
#      每 call wall ~22min(同 P1.5 节奏)→ 全跑完 wall ~2.5h
#
# done 标准(参考 current_task.md §6 P1.7):
#   1. sanity:Δ(M3.2 - M3.1) meet_slo% ≥ -1pp 跨 seed(基线不输)
#   2. 真实考验:code@570 严档 tpot100/150 上 Δ > 0(撬动 P1.5 的 FAIL)

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p17
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline,c2_tdm_m1_chunk2048,c2_tdm_m31_2048,c2_tdm_m32_2048,c3_cp"

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
    for CFG in c1_baseline c2_tdm_m1_chunk2048 c2_tdm_m31_2048 c2_tdm_m32_2048 c3_cp; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.7 done ==="
