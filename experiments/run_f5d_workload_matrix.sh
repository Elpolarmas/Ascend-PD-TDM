#!/bin/bash
# F5d - workload 量级矩阵填充 (2026-05-27)
#
# 触发: F5/F5b/F5c 累积 6 cell 沿对角线散布,paper §3 heatmap 需要 3x3
#       prompt 量级 x output 量级矩阵。F5d 补 8 个新 cell,完成 4x4 中等覆盖。
#
# 矩阵: prompt {256, 1024, 4096} x output {256, 1024, 4096} = 9 格,
#       去掉 F5cB2 (1024x1024) 已覆盖,新增 8 cell。
#
# Cells (lognormal sigma=0.3):
#   C1: P~256  / O~256   (1:1 短 chat balanced)
#   C2: P~256  / O~1024  (1:4 chatbot decode-leaning)
#   C3: P~256  / O~4096  (1:16 extreme decode-heavy short P)
#   C4: P~1024 / O~256   (4:1 RAG mild prefill-heavy)
#   C5: P~1024 / O~4096  (1:4 long-gen mid P)
#   C6: P~4096 / O~256   (16:1 code-like extreme prefill-heavy)
#   C7: P~4096 / O~1024  (4:1 code-like prefill-heavy)
#   C8: P~4096 / O~4096  (1:1 long balanced - cap=4096 防 MML 溢出)
#
# Paradigms: sarathi (c3_cp, mb=2048) + pdtdm (c2_tdm_m31_2048, mb=8192)
# Matrix: 8 wl x 2 paradigm x 5 QPS = 80 cells / 16 server lifecycles
# Wall: ~16 x 25min = ~6.7h
# Outdir: results/f5d_workload_matrix/{c1..c8}_{sarathi|pdtdm}_seed0/

set -e

ROOT=/vllm-workspace/Ascend-PD-TDM
SCRIPT=$ROOT/experiments/run_qps_sweep_all.py
PY=/usr/local/python3.11.13/bin/python3
OUT=$ROOT/results/f5d_workload_matrix
TRACE_DIR=$ROOT/results/tdm_trace

DURATION=180
WARMUP=30
SEED=0
SLO_TTFT=500
SLO_TPOT=200

# Per-cell workload + QPS + MML
# Order: P=prompt-mu/min/max, O=output-mu/min/max, Q=qps-list, MML=max-model-len
C1_OPTS="--prompt-mu 5.50 --prompt-sigma 0.3 --prompt-min 128 --prompt-max 512 \
         --output-mu 5.50 --output-sigma 0.3 --output-min 128 --output-max 512"
C1_QPS="0.5,1.0,1.5,2.0,2.5"; C1_MML=4096

C2_OPTS="--prompt-mu 5.50 --prompt-sigma 0.3 --prompt-min 128 --prompt-max 512 \
         --output-mu 6.89 --output-sigma 0.3 --output-min 512 --output-max 2048"
C2_QPS="0.1,0.2,0.3,0.4,0.5"; C2_MML=8192

C3_OPTS="--prompt-mu 5.50 --prompt-sigma 0.3 --prompt-min 128 --prompt-max 512 \
         --output-mu 8.27 --output-sigma 0.3 --output-min 2048 --output-max 8192"
C3_QPS="0.04,0.08,0.12,0.16,0.20"; C3_MML=8192

C4_OPTS="--prompt-mu 6.89 --prompt-sigma 0.3 --prompt-min 512 --prompt-max 2048 \
         --output-mu 5.50 --output-sigma 0.3 --output-min 128 --output-max 512"
C4_QPS="0.5,1.0,1.5,2.0,2.5"; C4_MML=4096

C5_OPTS="--prompt-mu 6.89 --prompt-sigma 0.3 --prompt-min 512 --prompt-max 2048 \
         --output-mu 8.27 --output-sigma 0.3 --output-min 2048 --output-max 8192"
C5_QPS="0.04,0.08,0.12,0.16,0.20"; C5_MML=8192

C6_OPTS="--prompt-mu 8.27 --prompt-sigma 0.3 --prompt-min 2048 --prompt-max 8192 \
         --output-mu 5.50 --output-sigma 0.3 --output-min 128 --output-max 512"
C6_QPS="0.5,1.0,1.5,2.0,2.5"; C6_MML=8192

C7_OPTS="--prompt-mu 8.27 --prompt-sigma 0.3 --prompt-min 2048 --prompt-max 8192 \
         --output-mu 6.89 --output-sigma 0.3 --output-min 512 --output-max 2048"
C7_QPS="0.1,0.2,0.3,0.4,0.5"; C7_MML=8192

# C8: P+O 都 cap 在 4096 防 MML=8192 溢出 (max P+O = 8192 = MML)
C8_OPTS="--prompt-mu 8.27 --prompt-sigma 0.3 --prompt-min 2048 --prompt-max 4096 \
         --output-mu 8.27 --output-sigma 0.3 --output-min 2048 --output-max 4096"
C8_QPS="0.04,0.08,0.12,0.16,0.20"; C8_MML=8192

run_tuple() {
  # $1 = cell (c1..c8)  $2 = paradigm (sarathi|pdtdm)
  local cell=$1 paradigm=$2
  local d=$OUT/${cell}_${paradigm}_seed${SEED}

  if [[ -f "$d/qps_sweep_summary.json" ]]; then
    echo "[skip] $(date '+%H:%M:%S') $d"
    return 0
  fi

  local upper="${cell^^}"
  local opts_var="${upper}_OPTS"
  local qps_var="${upper}_QPS"
  local mml_var="${upper}_MML"
  local wl_opts="${!opts_var}"
  local qps_list="${!qps_var}"
  local mml="${!mml_var}"

  local cfg mb
  case "$paradigm" in
    sarathi) cfg="c3_cp";           mb=2048 ;;
    pdtdm)   cfg="c2_tdm_m31_2048"; mb=$mml ;;
    *) echo "Unknown paradigm: $paradigm"; exit 1 ;;
  esac

  mkdir -p "$d/tdm_trace"
  echo "==================================================================="
  echo "===== $(date '+%H:%M:%S') $d  cfg=$cfg mml=$mml mb=$mb qps=$qps_list"
  echo "==================================================================="

  $PY "$SCRIPT" \
    --configs "$cfg" --qps "$qps_list" \
    --duration $DURATION --warmup $WARMUP \
    --max-model-len $mml --max-num-batched-tokens "$mb" \
    --slo-ttft-ms $SLO_TTFT --slo-tpot-ms $SLO_TPOT \
    --arrival-mode poisson \
    $wl_opts \
    --outdir "$d" --seed $SEED 2>&1 | tail -20 || echo "[warn] tuple $cell $paradigm rc=$?"

  for ST in iter req ctrl chunk; do
    F=$TRACE_DIR/qps_sweep_${cfg}_${ST}.jsonl
    [[ -f $F ]] && cp "$F" "$d/tdm_trace/qps_sweep_${cfg}_${ST}.jsonl" || true
  done
}

mkdir -p "$OUT"
echo "###################################################################"
echo "### F5d workload matrix started $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "### Matrix: 8 cell x 2 paradigm x 5 QPS = 80 cells, ~6.7h wall"
echo "###################################################################"

for CELL in c1 c2 c3 c4 c5 c6 c7 c8; do
  for P in sarathi pdtdm; do
    run_tuple "$CELL" "$P"
  done
done

echo "###################################################################"
echo "### F5d done $(date '+%Y-%m-%d %H:%M:%S')"
echo "### Out: $OUT"
echo "###################################################################"
