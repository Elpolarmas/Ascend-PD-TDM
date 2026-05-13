#!/usr/bin/env bash
# Level 3 论文 main experiment: Azure LLM Inference Trace 回放
#
# - 2 traces (conv at burst-rich start=1860s, code at start=570s)
# - 4 configs (c1_baseline / c2_tdm_m1_chunk2048 / c2_tdm_m31_2048 / c3_cp)
# - 3 seeds (0, 1, 2)
# - 60s duration each
#
# Total: 2 × 3 = 6 driver calls, each ~14min (4 configs serial).
# Estimated wall ≈ 1.5h.
#
# Per-seed snapshots tdm_trace/ to avoid global TRACE_DIR overwrite.
set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_main
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline,c2_tdm_m1_chunk2048,c2_tdm_m31_2048,c3_cp"

mkdir -p "$OUT"

# (trace_label, csv_path, start_offset_s)
declare -a TRACES=(
  "conv:$DATA/AzureLLMInferenceTrace_conv.csv:1860"
  "code:$DATA/AzureLLMInferenceTrace_code.csv:570"
)

for TRACE_ENTRY in "${TRACES[@]}"; do
  IFS=':' read -r TLABEL TCSV TOFFSET <<< "$TRACE_ENTRY"
  for SEED in 0 1 2; do
    D=$OUT/${TLABEL}_seed${SEED}
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== trace=$TLABEL seed=$SEED start_offset=${TOFFSET}s"
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
    for CFG in c1_baseline c2_tdm_m1_chunk2048 c2_tdm_m31_2048 c3_cp; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== L3 done ==="
