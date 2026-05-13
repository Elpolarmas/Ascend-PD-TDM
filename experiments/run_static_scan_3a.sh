#!/usr/bin/env bash
# P0-3a: target_ratio static scan on conv@1860s.
#
# 目的: 验证 target_ratio 这个 PID actuator 的 trade-off 形状(U 型 / 单调 / 平台)。
# 是 PID 设计成不成立的前提实验 — current_task.md §4 问题 3a。
#
# 设计:
# - trace: AzureLLMInferenceTrace_conv.csv, start_offset=1860s (与 azure_main 同窗)
# - configs: 4 个新静态 ratio (r01/r05/r07/r09) + 已有 r03 锚点不重跑
# - seeds: 0, 1, 2
# - duration: 60s
#
# 对比 baseline (已存在,不重跑):
# - M1@r03 = results/azure_main/conv_seed*/c2_tdm_m1_chunk2048_qps0.0.json
# - M3.1   = results/azure_main/conv_seed*/c2_tdm_m31_2048_qps0.0.json
#
# Total: 3 seeds × 1 driver call (4 configs serial) ≈ 40-50 min wall.

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/static_scan_3a
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m1_chunk2048_r01,c2_tdm_m1_chunk2048_r05,c2_tdm_m1_chunk2048_r07,c2_tdm_m1_chunk2048_r09"

mkdir -p "$OUT"

TLABEL=conv
TCSV=$DATA/AzureLLMInferenceTrace_conv.csv
TOFFSET=1860

for SEED in 0 1 2; do
  D=$OUT/${TLABEL}_seed${SEED}
  mkdir -p "$D/tdm_trace"
  echo "==================================================================="
  echo "===== 3a static_scan trace=$TLABEL seed=$SEED start_offset=${TOFFSET}s"
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
  # Snapshot per-seed trace files (server uses global TRACE_DIR)
  for CFG in c2_tdm_m1_chunk2048_r01 c2_tdm_m1_chunk2048_r05 c2_tdm_m1_chunk2048_r07 c2_tdm_m1_chunk2048_r09; do
    for ST in iter req ctrl chunk; do
      F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
    done
  done
done

echo "=== 3a static_scan done ==="
