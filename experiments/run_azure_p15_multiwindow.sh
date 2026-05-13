#!/usr/bin/env bash
# P1.5 跨窗扩展实验:Azure trace 多窗位回放
#
# 窗位由 select_p15_windows.py 按"QPS≥p75 + 贪心 ≥120s 间距 + top-3"挑出。
# 锚点 1860s (conv) / 570s (code) 自然落入 top-3。
#
# 规模:2 traces × 3 windows × 4 configs × 3 seeds × 60s
#       18 driver calls,wall ≈ 4-5h(对照 azure_main 6 calls ≈ 1.5h)
#
# Per-(window, seed) 快照 tdm_trace/ 避免被覆盖。
set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p15
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline,c2_tdm_m1_chunk2048,c2_tdm_m31_2048,c3_cp"

mkdir -p "$OUT"

# (window_label, csv_path, start_offset_s)
# 窗位由 select_p15_windows.py 计算;w1 是 azure_main 锚点。
declare -a WINDOWS=(
  "conv_w0:$DATA/AzureLLMInferenceTrace_conv.csv:1650"
  "conv_w1:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "conv_w2:$DATA/AzureLLMInferenceTrace_conv.csv:2130"
  "code_w0:$DATA/AzureLLMInferenceTrace_code.csv:180"
  "code_w1:$DATA/AzureLLMInferenceTrace_code.csv:570"
  "code_w2:$DATA/AzureLLMInferenceTrace_code.csv:840"
)

for ENTRY in "${WINDOWS[@]}"; do
  IFS=':' read -r WLABEL TCSV TOFFSET <<< "$ENTRY"
  for SEED in 0 1 2; do
    D=$OUT/${WLABEL}_seed${SEED}
    # 跳过已有结果(支持断点续跑)
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
    # 快照 per-(window, seed) trace 文件(server uses global TRACE_DIR)
    for CFG in c1_baseline c2_tdm_m1_chunk2048 c2_tdm_m31_2048 c3_cp; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.5 done ==="
