"""Slow-burst regime ablation (period=20s peak=10s high=28 low=4)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
QPS = 16.0
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]

RUNS = {
    "c1_baseline": [(s, ROOT / f"burst_slow_seed{s}", "c1_baseline")
                    for s in (0, 1, 2)],
    "M1+chunk2048": [(s, ROOT / f"burst_slow_seed{s}", "c2_tdm_m1_chunk2048")
                     for s in (0, 1, 2)],
    "M2.7+chunk2048": [(s, ROOT / f"burst_slow_seed{s}", "c2_tdm_m31_2048")
                       for s in (0, 1, 2)],
}


def load(sweep_dir: Path, config: str):
    d = json.loads((sweep_dir / f"{config}_qps{QPS}.json").read_text())
    s = d["summary"]
    return d["joined"], s["warmup_s"], s["warmup_s"] + s["duration_s"]


def meet(joined, w, we, tt, tp):
    in_win = [j for j in joined
              if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    total = len(in_win)
    n = sum(1 for j in in_win
            if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
            and j["ttft_ms"] < tt and j["tpot_ms_mean"] < tp
            and j.get("output_tokens"))
    return n, total


def grid_for(sweep_dir, config):
    joined, w, we = load(sweep_dir, config)
    return {(tt, tp): (lambda x: 100.0 * x[0] / x[1] if x[1] else 0.0)(
            meet(joined, w, we, tt, tp))
            for tt in TTFT_GRID for tp in TPOT_GRID}


def mean_std(xs):
    n = len(xs); mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs)/n if n>1 else 0.0
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
    print(f"\n=== SLOW-BURST regime (3 seeds, period=20s peak=10s) ===")
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

    print(f"\n=== Δ tables (paired, 3 seeds, pp) ===")
    print(f"  {'tier':<18s} | {'M1+chunk - C1':>20s}  "
          f"{'M2.7+chunk - M1+chunk':>26s}  {'M2.7+chunk - C1':>22s}")
    for label_t, tt, tp in tiers:
        c1_xs = [g[(tt, tp)] for g in grids["c1_baseline"]]
        m1c_xs = [g[(tt, tp)] for g in grids["M1+chunk2048"]]
        m27c_xs = [g[(tt, tp)] for g in grids["M2.7+chunk2048"]]
        d1 = [m1c_xs[i] - c1_xs[i] for i in range(3)]
        d2 = [m27c_xs[i] - m1c_xs[i] for i in range(3)]
        d3 = [m27c_xs[i] - c1_xs[i] for i in range(3)]
        def fmt(xs):
            mu, sd = mean_std(xs)
            return f"{mu:+5.1f} ± {sd:>4.1f} pp"
        print(f"  {label_t:<18s} | {fmt(d1):>20s}  {fmt(d2):>26s}  {fmt(d3):>22s}")


if __name__ == "__main__":
    main()
