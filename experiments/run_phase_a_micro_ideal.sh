#!/usr/bin/env bash
# Phase A micro-benchmark ideal (D-012, 2026-05-20):
# concurrent=1 sequential per (workload) ideal latency,Sarathi-Serve / Semi-PD 同款。
#
# 替换 run_phase_a_ref_ideal.sh (D-012 拒绝 trace-based 低 qps 测 ideal,
# 因为 code 端 prompt 分布 10× workload bias + 实证 conv qps=3.18 / code qps=1.05 受 queueing 污染)。
#
# 出 ideal_ttft_p99 / ideal_tpot_p99 → 推导 Sarathi 风格 SLO 4 档 = {5×, 10×, 15×, 25×} × ideal。
#
# 配置选择:c1_baseline 因为 c1 不读 slo_tpot,SLO 设值对 ideal 测量中立。
# 模型:Qwen3-8B(主 model;Qwen3-4B 留到 Phase 2 跨模型 generalization)。
# 规模:2 trace × 3 seed × 50 sample = 300 req,预计 wall 15-20min。

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/phase_a_micro_ideal
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline"
N_SAMPLES=50
# c1 不读 SLO,这些值只影响 aggregate_window 内 meet_slo 计数(我们看 raw p99,不用 meet_slo)
SLO_TTFT=10000
SLO_TPOT=1000

mkdir -p "$OUT"

SEEDS=(0 1 2)
TRACES=(conv code)

for TRACE in "${TRACES[@]}"; do
  TRACE_FILE=$DATA/AzureLLMInferenceTrace_${TRACE}.csv
  for SEED in "${SEEDS[@]}"; do
    D=$OUT/${TRACE}_seed${SEED}
    if [[ -f "$D/qps_sweep_summary.json" ]]; then
      echo "[skip] $D already complete"
      continue
    fi
    mkdir -p "$D/tdm_trace"
    echo "==================================================================="
    echo "===== Phase A micro-ideal trace=${TRACE} seed=${SEED} N=${N_SAMPLES}"
    echo "==================================================================="
    /usr/local/python3.11.13/bin/python3 \
      "$ROOT/experiments/run_qps_sweep_all.py" \
      --configs "$CONFIGS" \
      --qps 0 \
      --duration 0 --warmup 5 \
      --max-model-len 8192 --max-num-batched-tokens 8192 \
      --slo-ttft-ms "$SLO_TTFT" \
      --slo-tpot-ms "$SLO_TPOT" \
      --arrival-mode sequential \
      --num-samples "$N_SAMPLES" \
      --trace-file "$TRACE_FILE" \
      --trace-max-prompt-tokens 7000 \
      --trace-max-output-tokens 600 \
      --outdir "$D" \
      --seed "$SEED" \
      2>&1 | tee "$D/run.log"
    for ST in iter req ctrl chunk; do
      F=$TRACE_DIR/qps_sweep_${CONFIGS}_${ST}.jsonl
      [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CONFIGS}_${ST}.jsonl" || true
    done
  done
done

echo "=== Phase A micro-ideal done ==="
echo "Run: experiments/posthoc_phase_a_micro_ideal.py to derive SLO grid"
