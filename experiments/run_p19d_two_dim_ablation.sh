#!/usr/bin/env bash
# P1.9d 双维度协同 ablation(2026-05-17)。直接回答 D-002 thesis 核心问题:
# 慢回路 + 快回路在 saturated workload 上是不是真的协同?
#
# 4-way matrix:
#   A 都没有:    c2_tdm_m1_r08_cap1     (static r=0.8 + bucket_cap=1)
#   B 只慢:      c2_tdm_m31_2048_cap1   (PID + bucket_cap=1)
#   C 只快:      c2_tdm_m1_chunk2048_r08 (static r=0.8 + bucket_cap=4)
#   D 慢+快:     c2_tdm_m31_2048        (PID + bucket_cap=4 = M3.1 原版)
#
# 协同 Δ = D - max(B, C)。P1.9a 实证 PID 在 saturated 钉 ratio_max=0.8,所以
# 预期 D ≈ C(慢回路 dynamic 调整无 Δ),协同 Δ ≈ 0。本实验是把这个结构性结
# 论从间接推断变成直接证据。
#
# 子贡献:
#   只慢贡献 = B - A(PID 在 cap=1 上贡献,如果有的话)
#   只快贡献 = C - A(token bucket burst 在 static ratio 上贡献)
#   协同效应 = D - B - C + A (interaction term)
#
# 规模:2 windows × 4 configs × 3 seeds × 60s,wall ~2h

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p19d
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m1_r08_cap1,c2_tdm_m31_2048_cap1,c2_tdm_m1_chunk2048_r08,c2_tdm_m31_2048"

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
    echo "===== P1.9d window=$WLABEL seed=$SEED start_offset=${TOFFSET}s"
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
    for CFG in c2_tdm_m1_r08_cap1 c2_tdm_m31_2048_cap1 c2_tdm_m1_chunk2048_r08 c2_tdm_m31_2048; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.9d done ==="
