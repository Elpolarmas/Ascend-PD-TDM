"""Post-hoc SLO grid for the M3.1 same-sweep validation (mixed, qps=16).

All 4 configs run under one seed in one sweep (m31_samesweep_validation/).
This rules out cross-sweep timing variance that confused the chunk-size scan.
"""
from __future__ import annotations

import json
from pathlib import Path

SWEEP_DIR = Path("/vllm-workspace/Ascend-PD-TDM/results/m31_samesweep_validation")
CONFIGS = ["c2_tdm_m27", "c3_cp", "c2_tdm_m31_2048", "c2_tdm_m31_4096"]
QPS = 16.0

TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load_joined(config):
    d = json.loads((SWEEP_DIR / f"{config}_qps{QPS}.json").read_text())
    s = d["summary"]
    return d["joined"], s["warmup_s"], s["warmup_s"] + s["duration_s"]


def meet_slo_count(joined, w, we, tt, tp):
    in_win = [j for j in joined
              if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    total = len(in_win)
    meet = sum(
        1 for j in in_win
        if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
        and j["ttft_ms"] < tt and j["tpot_ms_mean"] < tp
        and j.get("output_tokens"))
    return meet, total


def main():
    grids = {}
    for cfg in CONFIGS:
        joined, w, we = load_joined(cfg)
        cell = {}
        for tt in TTFT_GRID:
            for tp in TPOT_GRID:
                m, t = meet_slo_count(joined, w, we, tt, tp)
                cell[(tt, tp)] = 100.0 * m / t if t else 0.0
        grids[cfg] = cell

    for cfg in CONFIGS:
        print(f"\n--- {cfg} ---  (rows=tpot, cols=ttft)")
        print(f"  tpot \\ ttft  | "
              + "  ".join(f"{tt:>6d}" for tt in TTFT_GRID))
        print(f"  {'-'*13} | " + "  ".join("-" * 6 for _ in TTFT_GRID))
        for tp in TPOT_GRID:
            row = [grids[cfg][(tt, tp)] for tt in TTFT_GRID]
            print(f"  {tp:>5d}        | "
                  + "  ".join(f"{v:>6.1f}" for v in row))

    print("\n\n=== Cross-config ranking on key SLO points (meet_slo %) ===")
    key_points = [
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
          + "  ".join(f"{c:>16s}" for c in CONFIGS))
    print(f"  {'-'*18} {'-'*5} {'-'*4}  | "
          + "  ".join("-" * 16 for _ in CONFIGS))
    for label, tt, tp in key_points:
        row = [grids[c][(tt, tp)] for c in CONFIGS]
        print(f"  {label:<18s} {tt:>5d} {tp:>4d}  | "
              + "  ".join(f"{v:>16.1f}" for v in row))

    print("\n\n=== Δ vs M2.7 baseline (M3.1_2048 - M2.7) ===")
    for label, tt, tp in key_points:
        d = grids["c2_tdm_m31_2048"][(tt, tp)] - grids["c2_tdm_m27"][(tt, tp)]
        sign = "+" if d > 0 else ""
        print(f"  {label:<18s} {tt:>5d} {tp:>4d}  |  "
              f"{grids['c2_tdm_m27'][(tt, tp)]:>5.1f} -> "
              f"{grids['c2_tdm_m31_2048'][(tt, tp)]:>5.1f}  ({sign}{d:.1f}pp)")


if __name__ == "__main__":
    main()
