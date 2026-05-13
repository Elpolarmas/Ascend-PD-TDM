#!/usr/bin/env bash
# P1.6h — calibrated SLO 下重测 baseline
#
# 目的:azure_main 主表用的是 strict SLO(50/500),那个档下 evaluation 信号
# 弱(都打地板)。在 P1.6e 校准的 slo_tpot=200ms 下重测完整 baseline,产出
# 新 thesis main result candidate 数据。
#
# 4 configs × 2 windows × 3 seeds × 60s = 6 driver calls × ~22min/call = ~2.2h
# (每个 driver call 串行跑 4 configs,每个 config 启停 vllm server)
#
# 关键观察:
#   - M3.1 vs C3 / M1+chunk / C1 在 200ms 档下的 meet_slo% 区分度
#   - 跟 azure_main 主表(50ms 档)对比 — 区分度应当更高
#
# 后台启动:
#   nohup bash experiments/run_p16h_calibrated_baseline.sh \
#       > results/azure_p16h_orchestrator.log 2>&1 &
# Posthoc(复用 P1.5 思路):
#   python3 experiments/posthoc_p16h_calibrated_baseline.py

set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
OUT=$ROOT/results/azure_p16h
TRACE_DIR=$ROOT/results/tdm_trace
DATA=$ROOT/data/azure_trace
CONFIGS="c1_baseline,c2_tdm_m1_chunk2048,c2_tdm_m31_2048,c3_cp"
SLO_TPOT=200  # P1.6e calibrated conv sweet spot

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
    echo "===== window=$WLABEL  seed=$SEED  slo_tpot=${SLO_TPOT}"
    echo "==================================================================="
    /usr/local/python3.11.13/bin/python3 \
      "$ROOT/experiments/run_qps_sweep_all.py" \
      --configs "$CONFIGS" \
      --qps 0 \
      --duration 60 --warmup 20 \
      --max-model-len 8192 --max-num-batched-tokens 8192 \
      --slo-ttft-ms 500 \
      --slo-tpot-ms "$SLO_TPOT" \
      --arrival-mode trace \
      --trace-file "$TCSV" \
      --trace-start-offset "$TOFFSET" \
      --trace-max-prompt-tokens 7000 \
      --trace-max-output-tokens 600 \
      --outdir "$D" \
      --seed "$SEED" \
      2>&1 | tee "$D/run.log"
    for CFG in c1_baseline c2_tdm_m1_chunk2048 c2_tdm_m31_2048 c3_cp; do
      for ST in iter req ctrl chunk; do
        F=$TRACE_DIR/qps_sweep_${CFG}_${ST}.jsonl
        [[ -f $F ]] && cp "$F" "$D/tdm_trace/qps_sweep_${CFG}_${ST}.jsonl" || true
      done
    done
  done
done

echo "=== P1.6h done ==="
