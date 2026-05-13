#!/usr/bin/env bash
# P1.6i — chunk_tokens sweep on Azure trace
#
# 目的:P0-3 暴露 conv 长 prompt 输 C3 22pp;chunk 在 azure mild-burst 上没扫过。
# 不同 chunk_tokens(512/1024/2048/4096) 在 azure trace 上看是否有更好的点。
#
# 复用现有 c2_tdm_m31_{512,1024,2048,4096} configs(已在 run_qps_sweep_all.py
# 的 M31_CHUNK_SIZES 表中)。每个 driver call 串行 4 configs。
#
# 2 windows × 3 seeds × 60s = 6 driver calls × ~22min/call = ~2.2h wall
#
# 关键观察:
#   - chunk_tokens 在 Azure 上是否仍 pareto 单调(P1 chunk scan 是 stationary)
#   - 不同 chunk 在 conv 长 prompt(p99~4000 tokens)上的表现差异
#
# 后台启动:
#   nohup bash experiments/run_p16i_chunk_sweep.sh \
#       > results/azure_p16i_orchestrator.log 2>&1 &
# Posthoc:
#   python3 experiments/posthoc_p16i_chunk.py

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16i
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m31,c2_tdm_m31_1024,c2_tdm_m31_2048,c2_tdm_m31_4096"

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
    echo "===== window=$WLABEL  seed=$SEED  chunk sweep"
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
    for CFG in c2_tdm_m31 c2_tdm_m31_1024 c2_tdm_m31_2048 c2_tdm_m31_4096; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.6i done ==="
