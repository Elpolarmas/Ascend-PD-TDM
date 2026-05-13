#!/usr/bin/env bash
# P1.0b: baseline C3 audit with HCCL_OP_EXPANSION_MODE=AIV.
#
# 验证假设: vllm-ascend log 建议设 HCCL_OP_EXPANSION_MODE=AIV 以提升通信性能
# 并增加 ACL Graph 可 capture 的 shape 数量(默认 FFTS+ 限制到 15 种)。
# 如果设了 AIV 后 c3@2048 vs c3@8192 趋势反转,说明 P1.0 的反常源于通信模式
# + graph capture 限制,而非 chunked prefill 本身。
#
# 实验: 同 P1.0,但 export HCCL_OP_EXPANSION_MODE=AIV
set -euo pipefail

export HCCL_OP_EXPANSION_MODE=AIV

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/baseline_c3_audit_aiv
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace

mkdir -p "$OUT"

TLABEL=conv
TCSV=$DATA/AzureLLMInferenceTrace_conv.csv
TOFFSET=1860

# Two batched values: same as P1.0 (skip 512, since P1.0 showed it's worse)
# Include 8192 to get same-env baseline for fair comparison
for BATCHED in 2048 8192; do
  for SEED in 0 1 2; do
    D=$OUT/${TLABEL}_b${BATCHED}_seed${SEED}
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== baseline_audit_aiv HCCL=AIV batched=${BATCHED} seed=${SEED}"
    echo "==================================================================="
    /usr/local/python3.11.13/bin/python3 \
      "$ROOT/experiments/run_qps_sweep_all.py" \
      --configs c3_cp \
      --qps 0 \
      --duration 60 --warmup 20 \
      --max-model-len 8192 \
      --max-num-batched-tokens "$BATCHED" \
      --arrival-mode trace \
      --trace-file "$TCSV" \
      --trace-start-offset "$TOFFSET" \
      --trace-max-prompt-tokens 7000 \
      --trace-max-output-tokens 600 \
      --outdir "$D" \
      --seed "$SEED" \
      2>&1 | tee "$D/run.log"
    for ST in iter req ctrl chunk; do
      F=$TRACE_DIR/qps_sweep_c3_cp_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_c3_cp_${ST}.jsonl" || true
    done
  done
done

echo "=== baseline_c3_audit_aiv done ==="
