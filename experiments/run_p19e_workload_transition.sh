#!/usr/bin/env bash
# P1.9e workload-transition ablation(2026-05-17)。P1.9d 验证稳态 saturated
# 上慢回路无 Δ(预期),这里验证 transition 期慢回路 vs 静态 best-ratio 有无
# Δ — 也就是 thesis 真正的"双维度协同"应该发生的场景。
#
# 实验设计:60s 拼接 trace
#   conv→code: 0-30s 用 conv(轻载),30-60s 用 code(重载)— PID 该 ramp ratio UP
#   code→conv: 0-30s 用 code(重载),30-60s 用 conv(轻载)— PID 该 ramp ratio DOWN
#
# 对照(简化版,只比"只快" vs "慢+快"):
#   c2_tdm_m1_chunk2048_r08    static r=0.8(P1.9a 实测 saturated steady-state ratio)
#   c2_tdm_m31_2048            M3.1 (PID 在 transition 期动态调整)
#
# 预期 finding:
#   - conv→code 起始期:M1@0.8 在 conv 上 over-allocated to prefill → TTFT 也许差点 / TPOT 退步
#     M3.1 PID 从 0.3 启动 → TTFT 在 conv 段更稳
#   - code→conv 起始期:M1@0.8 还是 0.8(code 上正好)
#     M3.1 在 code 段也跑到 0.8,但切到 conv 后 ratio 该往下走 → 看 PID 是否真的 ramp down
#   - 协同 Δ 应该出现在 transition 边界附近的 SLO 满足率
#
# 规模:2 directions × 2 configs × 3 seeds × 60s,wall ~50min

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p19e
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c2_tdm_m1_chunk2048_r08,c2_tdm_m31_2048"

mkdir -p "$OUT"

declare -a WINDOWS=(
  "conv2code:$DATA/AzureLLMInferenceTrace_hybrid_conv2code.csv:0"
  "code2conv:$DATA/AzureLLMInferenceTrace_hybrid_code2conv.csv:0"
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
    echo "===== P1.9e direction=$WLABEL seed=$SEED"
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
    for CFG in c2_tdm_m1_chunk2048_r08 c2_tdm_m31_2048; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.9e done ==="
