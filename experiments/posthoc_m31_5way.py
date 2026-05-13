"""Post-hoc SLO grid for M3.1 5-way ablation (mixed regime, qps=16)."""
from __future__ import annotations

import json
from pathlib import Path

SWEEP_DIR = Path("/vllm-workspace/Ascend-PD-TDM/results/m31_5way_ablation")
CONFIGS = ["c1_baseline", "c2_tdm_m27", "c2_tdm_m31_disabled",
           "c2_tdm_m31_2048", "c3_cp"]
QPS = 16.0

TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load(cfg):
    d = json.loads((SWEEP_DIR / f"{cfg}_qps{QPS}.json").read_text())
    s = d["summary"]
    return d["joined"], s["warmup_s"], s["warmup_s"] + s["duration_s"]


def meet(joined, w, we, tt, tp):
    in_win = [j for j in joined
              if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    total = len(in_win)
    n = sum(
        1 for j in in_win
        if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
        and j["ttft_ms"] < tt and j["tpot_ms_mean"] < tp
        and j.get("output_tokens"))
    return n, total


def main():
    grids = {}
    for cfg in CONFIGS:
        joined, w, we = load(cfg)
        cell = {}
        for tt in TTFT_GRID:
            for tp in TPOT_GRID:
                m, t = meet(joined, w, we, tt, tp)
                cell[(tt, tp)] = 100.0 * m / t if t else 0.0
        grids[cfg] = cell

    for cfg in CONFIGS:
        print(f"\n--- {cfg} ---  (rows=tpot, cols=ttft)")
        print(f"  tpot \\ ttft  | "
              + "  ".join(f"{tt:>6d}" for tt in TTFT_GRID))
        print(f"  {'-'*13} | " + "  ".join("-"*6 for _ in TTFT_GRID))
        for tp in TPOT_GRID:
            row = [grids[cfg][(tt, tp)] for tt in TTFT_GRID]
            print(f"  {tp:>5d}        | "
                  + "  ".join(f"{v:>6.1f}" for v in row))

    print("\n\n=== Cross-config ranking (meet_slo %) ===")
    key = [
        ("strict", 500, 50),
        ("ttft500/tpot100", 500, 100),
        ("ttft500/tpot150", 500, 150),
        ("ttft500/tpot200", 500, 200),
        ("ttft1000/tpot100", 1000, 100),
        ("ttft1000/tpot150", 1000, 150),
        ("ttft1500/tpot150", 1500, 150),
        ("ttft1500/tpot200", 1500, 200),
    ]
    print(f"  {'tier':<18s} {'tt':>5s} {'tp':>4s}  | "
          + "  ".join(f"{c:>20s}" for c in CONFIGS))
    print(f"  {'-'*18} {'-'*5} {'-'*4}  | "
          + "  ".join("-"*20 for _ in CONFIGS))
    for label, tt, tp in key:
        row = [grids[c][(tt, tp)] for c in CONFIGS]
        print(f"  {label:<18s} {tt:>5d} {tp:>4d}  | "
              + "  ".join(f"{v:>20.1f}" for v in row))

    print("\n\n=== Ablation deltas at each tier ===")
    print("  tier               | "
          "TDM-controller         fork-path           chunking            "
          "vs C1            vs C3")
    print("  (m27-c1)           | "
          "(m27-c1)               (m31_dis-m27)       (m31_2048-m31_dis)  "
          "(m31_2048-c1)    (m31_2048-c3)")
    for label, tt, tp in key:
        c1 = grids["c1_baseline"][(tt, tp)]
        m27 = grids["c2_tdm_m27"][(tt, tp)]
        mdis = grids["c2_tdm_m31_disabled"][(tt, tp)]
        m2048 = grids["c2_tdm_m31_2048"][(tt, tp)]
        c3 = grids["c3_cp"][(tt, tp)]
        d_tdm = m27 - c1
        d_fork = mdis - m27
        d_chunk = m2048 - mdis
        d_vs_c1 = m2048 - c1
        d_vs_c3 = m2048 - c3
        print(f"  {label:<18s} | "
              f"  TDM:{d_tdm:+5.1f}  fork:{d_fork:+5.1f}  "
              f"chunk:{d_chunk:+5.1f}  C1:{d_vs_c1:+5.1f}  C3:{d_vs_c3:+5.1f}")


if __name__ == "__main__":
    main()
