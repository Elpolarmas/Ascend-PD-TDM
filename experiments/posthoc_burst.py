"""Burst regime ablation analysis.

Compares C1 / M1+chunk / M2.7+chunk under burst arrival across 3 seeds.
Burst pattern: 10s period, 2s peak at qps=64 / 8s quiet at qps=4, mean qps=16.

Hypothesis: closed-loop M2.7 PID adapts to burst-induced SLO spikes; open-loop
M1 static_ratio doesn't. If true, M2.7 - M1 delta should be positive in burst
regime (vs ≈0 in stationary regime).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
QPS = 16.0
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]

RUNS = {
    "c1_baseline": [
        (0, ROOT / "burst_seed0", "c1_baseline"),
        (1, ROOT / "burst_seed1", "c1_baseline"),
        (2, ROOT / "burst_seed2", "c1_baseline"),
    ],
    "M1+chunk2048": [
        (0, ROOT / "burst_seed0", "c2_tdm_m1_chunk2048"),
        (1, ROOT / "burst_seed1", "c2_tdm_m1_chunk2048"),
        (2, ROOT / "burst_seed2", "c2_tdm_m1_chunk2048"),
    ],
    "M2.7+chunk2048": [
        (0, ROOT / "burst_seed0", "c2_tdm_m31_2048"),
        (1, ROOT / "burst_seed1", "c2_tdm_m31_2048"),
        (2, ROOT / "burst_seed2", "c2_tdm_m31_2048"),
    ],
}


def load(sweep_dir: Path, config: str):
    path = sweep_dir / f"{config}_qps{QPS}.json"
    d = json.loads(path.read_text())
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


def grid_for(sweep_dir: Path, config: str):
    joined, w, we = load(sweep_dir, config)
    cell = {}
    for tt in TTFT_GRID:
        for tp in TPOT_GRID:
            m, t = meet(joined, w, we, tt, tp)
            cell[(tt, tp)] = 100.0 * m / t if t else 0.0
    return cell


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu)**2 for x in xs) / n if n > 1 else 0.0
    return mu, math.sqrt(var)


def main():
    grids = {label: [grid_for(sd, cfg) for _, sd, cfg in runs]
             for label, runs in RUNS.items()}

    tiers = [
        ("strict", 500, 50),
        ("ttft500/tpot100", 500, 100),
        ("ttft500/tpot150", 500, 150),
        ("ttft500/tpot200", 500, 200),
        ("ttft1000/tpot100", 1000, 100),
        ("ttft1000/tpot150", 1000, 150),
        ("ttft1500/tpot200", 1500, 200),
    ]

    print(f"\n=== BURST regime (3 seeds, mixed prompts, mean qps=16) ===")
    print(f"  {'tier':<18s} {'tt':>5s} {'tp':>4s}  | "
          + "  ".join(f"{label:>22s}" for label in RUNS))
    print(f"  {'-'*18} {'-'*5} {'-'*4}  | "
          + "  ".join("-"*22 for _ in RUNS))
    for label_t, tt, tp in tiers:
        cells = []
        for label in RUNS:
            xs = [g[(tt, tp)] for g in grids[label]]
            mu, sd = mean_std(xs)
            cells.append(f"{mu:>5.1f}±{sd:>4.1f}              ")
        print(f"  {label_t:<18s} {tt:>5d} {tp:>4d}  | " + "  ".join(cells))

    # Paired deltas
    print(f"\n=== Δ tables (paired, mean ± std across 3 seeds, pp) ===")
    print(f"  {'tier':<18s} | {'M1+chunk - C1':>20s}  "
          f"{'M2.7+chunk - M1+chunk':>26s}  {'M2.7+chunk - C1':>22s}")
    print(f"  {'-'*18} | " + "  ".join("-"*max(len(s), 20) for s in
                                        ["M1+chunk - C1",
                                         "M2.7+chunk - M1+chunk",
                                         "M2.7+chunk - C1"]))
    for label_t, tt, tp in tiers:
        c1_xs = [g[(tt, tp)] for g in grids["c1_baseline"]]
        m1c_xs = [g[(tt, tp)] for g in grids["M1+chunk2048"]]
        m27c_xs = [g[(tt, tp)] for g in grids["M2.7+chunk2048"]]

        d_m1c_c1 = [m1c_xs[i] - c1_xs[i] for i in range(3)]
        d_m27c_m1c = [m27c_xs[i] - m1c_xs[i] for i in range(3)]
        d_m27c_c1 = [m27c_xs[i] - c1_xs[i] for i in range(3)]

        def fmt(xs):
            mu, sd = mean_std(xs)
            return f"{mu:+5.1f} ± {sd:>4.1f} pp"

        print(f"  {label_t:<18s} | "
              f"{fmt(d_m1c_c1):>20s}  "
              f"{fmt(d_m27c_m1c):>26s}  "
              f"{fmt(d_m27c_c1):>22s}")


if __name__ == "__main__":
    main()
