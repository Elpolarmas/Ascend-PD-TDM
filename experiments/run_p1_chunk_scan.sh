#!/usr/bin/env bash
# P1 chunk scan driver: runs {chunk=512, 1024, 2048} × {seed=0,1,2} on long
# qps=16 and preserves the per-seed ctrl/chunk JSONL traces (which the global
# tdm_trace/ dir would otherwise overwrite between seeds).
#
# Layout produced:
#   results/p1_chunk_scan/
#     seed{0,1,2}/                       <- run_qps_sweep_all.py outdir (summaries)
#     seed{0,1,2}/tdm_trace/             <- per-seed snapshot of ctrl/chunk JSONL
#
# Wall budget: 3 chunk × 3 seed × ~5min ≈ 45-75min (server start dominates).
set -euo pipefail

ROOT=/vllm-workspace/Ascend-PD-TDM
SCAN_DIR=$ROOT/results/p1_chunk_scan
TRACE_DIR=$ROOT/results/tdm_trace
CONFIGS="c2_tdm_m31,c2_tdm_m31_1024,c2_tdm_m31_2048"

mkdir -p "$SCAN_DIR"

for SEED in 0 1 2; do
  SEED_DIR=$SCAN_DIR/seed$SEED
  mkdir -p "$SEED_DIR/tdm_trace"

  echo "=========================================="
  echo "=== P1 chunk scan: seed=$SEED"
  echo "=========================================="

  /usr/local/python3.11.13/bin/python3 \
    "$ROOT/experiments/run_qps_sweep_all.py" \
    --configs "$CONFIGS" \
    --qps 16 --duration 60 --warmup 20 \
    --prompt-profile long --seed "$SEED" \
    --outdir "$SEED_DIR" \
    2>&1 | tee "$SEED_DIR/run.log"

  # Snapshot per-seed traces. The next seed iteration will wipe TRACE_DIR
  # entries via cleanup_tracker, so we copy now.
  for CFG in c2_tdm_m31 c2_tdm_m31_1024 c2_tdm_m31_2048; do
    for STREAM in ctrl chunk iter req; do
      SRC=$TRACE_DIR/qps_sweep_${CFG}_${STREAM}.jsonl
      if [[ -f "$SRC" ]]; then
        cp "$SRC" "$SEED_DIR/tdm_trace/qps_sweep_${CFG}_${STREAM}.jsonl"
      fi
    done
  done
done

echo
echo "=== P1 chunk scan complete. ==="
echo "Per-seed traces preserved in $SCAN_DIR/seed{0,1,2}/tdm_trace/"
