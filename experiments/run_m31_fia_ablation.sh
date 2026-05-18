#!/usr/bin/env bash
# m31_fia ablation:Azure trace 上的 4-way 对比 sweep
#
# 目的:拆分 phase-pure 调度策略贡献 vs dedicated kernel 速度贡献。
#   delta(m31_2048 - m31_fia)  = dedicated kernel 速度的贡献
#   delta(m31_fia  - c3_cp)    = phase-pure 调度策略本身的贡献
#
# 4 configs:
#   c1_baseline       hybrid + dedicated kernel(sanity)
#   c2_tdm_m31_2048   phase-pure + chunking + dedicated kernel(M3.1 baseline)
#   c2_tdm_m31_fia    phase-pure + chunking + FIA(本 ablation 新加)
#   c3_cp             mixed batch + FIA(SOTA 对照)
#
# 跟 azure_main 同 trace 参数(conv@1860s / code@570s,60s duration,3 seeds)
# 完整重跑保证 vllm 进程版本一致,数据可比。
#
# Total: 2 traces × 3 seeds = 6 driver calls,4 configs serial per call,
# 每次 ~14min,合计 wall ≈ 1.5-2h。

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_m31_fia_ablation
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline,c2_tdm_m31_2048,c2_tdm_m31_fia,c3_cp"

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
    # 每 seed 快照 trace 文件(server 用全局 TRACE_DIR)
    for CFG in c1_baseline c2_tdm_m31_2048 c2_tdm_m31_fia c3_cp; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== m31_fia ablation done ==="
