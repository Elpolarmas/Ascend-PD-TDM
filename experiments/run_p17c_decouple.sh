#!/usr/bin/env bash
# P1.7c decouple smoke(2026-05-17)。诊断结论:P1.7b starvation_tpot 触发率
# 0.5% 不是 silence 不够,是 selector 里 `tokens >= 1.0` AND-门把"双向预警"
# pin 在 bucket 反向的窄窗。本实验验证去掉 AND-门后:
#   1. 触发率是否真的上来(silence 在 code 上 20% iter > 150ms,物理上有空间)
#   2. meet_slo% 是否净改善(thesis B2/B3 的正向证据)
#   3. P/D budget anchor 是否仍然守得住(bucket 不被 charged,只是窄窗放开)
#
# 对照:
#   - c2_tdm_m31_2048             只慢回路(锚点)
#   - c2_tdm_m33_2048             M3.3 default,旧 AND-门(P1.7b 失败现场)
#   - c2_tdm_m33_2048_decouple    M3.3 + decouple_bucket=True,starv=3.0
#   - c2_tdm_m33_2048_decouple_starv2  decouple + starv=2.0(更敏感)
#
# 规模:2 windows × 4 configs × 3 seeds × 60s,wall ~2 h
#
# done 标准:
#   1. M3.3 decouple starvation_tpot 触发率 > 5%(thesis"双向预警活了"实证)
#   2. M3.3 decouple vs M3.3 default code meet_slo% Δ >= -1pp(不显著退步)
#      或更激进:Δ > 0(机制本身正贡献)
#   3. ratio mean / std 跨配置稳定(慢回路锚点没被快回路扰动)
#
# 后处理:复用 posthoc_p17b_starvation.py + 新加 source 分布对比图

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p17c
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31_2048,c2_tdm_m33_2048,c2_tdm_m33_2048_decouple,c2_tdm_m33_2048_decouple_starv2"

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
    echo "===== P1.7c window=$WLABEL seed=$SEED start_offset=${TOFFSET}s"
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
    for CFG in c2_tdm_m31_2048 c2_tdm_m33_2048 c2_tdm_m33_2048_decouple c2_tdm_m33_2048_decouple_starv2; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.7c done ==="
