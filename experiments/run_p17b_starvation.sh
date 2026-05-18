#!/usr/bin/env bash
# P1.7b smoke + threshold scan(bidirectional urgency_ttft + starvation_tpot)。
#
# 目的:
#   1. smoke:确认 starvation_tpot 真的会被触发(不是又一次 P1.7 那种 0% 触发率)
#   2. scan:扫 starvation_tpot_threshold ∈ {2.0, 3.0, 5.0} × slo_tpot_ms,
#      找触发率有意义且不过激的 sweet spot
#
# 对照:
#   - c2_tdm_m31_2048    M3.1,只慢回路(P/D 时分复用 + PID + chunk)
#   - c2_tdm_m32_2048    M3.2,M3.1 + 单向 urgency_ttft(P1.7 留下)
#   - c2_tdm_m33_2048    M3.3 default,M3.1 + 双向 (urgency_ttft + starvation_tpot=3.0)
#   - c2_tdm_m33_2048_starv2  M3.3 + starvation_tpot=2.0 (更敏感)
#   - c2_tdm_m33_2048_starv5  M3.3 + starvation_tpot=5.0 (更保守)
#
# 规模:
#   - 2 windows (conv@1860, code@570) × 5 configs × 3 seeds × 60s
#   - 同 P1.7 节奏,wall ~2.5h
#
# done 标准:
#   1. M3.3 default starvation_tpot trigger 频率 > 0.5%(说明双向逻辑活了,
#      不是 P1.7 那种死代码)。两 trace 至少一个有触发
#   2. M3.3 vs M3.1 meet_slo% 不退步 -1pp 以内(基线安全)
#   3. M3.3 vs M3.2 看是否有正向 Δ(双向 > 单向,P1.7 FAIL 翻盘)
#   4. starv2 / starv5 看 threshold sensitivity:trigger 频率单调,goodput 单调
#
# 出图 / 分析:posthoc_p17b_starvation.py(touchstone: source 字段分布)

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p17b
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048,c2_tdm_m32_2048,c2_tdm_m33_2048,c2_tdm_m33_2048_starv2,c2_tdm_m33_2048_starv5"

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
    echo "===== P1.7b window=$WLABEL seed=$SEED start_offset=${TOFFSET}s"
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
    for CFG in c2_tdm_m31_2048 c2_tdm_m32_2048 c2_tdm_m33_2048 c2_tdm_m33_2048_starv2 c2_tdm_m33_2048_starv5; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.7b done ==="
