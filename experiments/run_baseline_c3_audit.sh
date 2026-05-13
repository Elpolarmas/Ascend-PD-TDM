#!/usr/bin/env bash
# P1.0: baseline C3 公平性 audit.
#
# 验证假设: azure_main C3 baseline 用 max_num_batched_tokens=8192,导致
# chunked prefill 几乎不切 — 失去 Sarathi-Serve 的"小 chunk + decode fusion"
# 核心优势。等价于"挂着 cp 牌子的 fusion 单 batch"。
#
# 实验设计: 在 conv@1860s 上跑 c3_cp 在两个更激进的 max_num_batched_tokens:
#   - batched=2048: 等同 M3.1 chunk_tokens,真正切 chunk
#   - batched=512:  Sarathi-Serve 标准,激进切
# 复用 azure_main 现有 c3_cp@8192 数据作 baseline。
#
# 判定: 若新配置下 C3 显著 (>2pp on strict tier) 好于 c3_cp@8192,
# 候选 A 成立,需要重做 main result。
set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/baseline_c3_audit
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace

mkdir -p "$OUT"

TLABEL=conv
TCSV=$DATA/AzureLLMInferenceTrace_conv.csv
TOFFSET=1860

# Two batched-tokens values to scan
for BATCHED in 2048 512; do
  for SEED in 0 1 2; do
    D=$OUT/${TLABEL}_b${BATCHED}_seed${SEED}
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== baseline_audit batched=${BATCHED} seed=${SEED} trace=${TLABEL}"
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

echo "=== baseline_c3_audit done ==="
