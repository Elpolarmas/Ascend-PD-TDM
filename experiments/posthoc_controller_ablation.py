"""Controller ablation: does M2.7's SLO PID add value on top of M1+chunk?

Compares 3 configurations across 3 seeds (mixed, qps=16):

  Config           | controller        | chunking | sweep dir
  -----------------|-------------------|----------|--------------------
  c1_baseline      | none (hybrid)     | no       | m31_5way_{ablation,seed1,seed2}
  c2_tdm_m1_chunk2048 | M1 static_ratio | yes (2048) | m1_chunk_seed{0,1,2}
  c2_tdm_m31_2048  | M2.7 SLO PID      | yes (2048) | m31_5way_{ablation,seed1,seed2}

Key deltas:
  m1_chunk - c1   = (phase-pure toggling + chunking) total value
  m31_2048 - m1_chunk = SLO PID's incremental value above static-ratio + chunk
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
QPS = 16.0
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]

# (label, list of (seed, sweep_dir, config_name))
RUNS = {
    "c1_baseline": [
        (0, ROOT / "m31_5way_ablation", "c1_baseline"),
        (1, ROOT / "m31_5way_seed1", "c1_baseline"),
        (2, ROOT / "m31_5way_seed2", "c1_baseline"),
    ],
    "M1+chunk2048": [
        (0, ROOT / "m1_chunk_seed0", "c2_tdm_m1_chunk2048"),
        (1, ROOT / "m1_chunk_seed1", "c2_tdm_m1_chunk2048"),
        (2, ROOT / "m1_chunk_seed2", "c2_tdm_m1_chunk2048"),
    ],
    "M2.7+chunk2048": [
        (0, ROOT / "m31_5way_ablation", "c2_tdm_m31_2048"),
        (1, ROOT / "m31_5way_seed1", "c2_tdm_m31_2048"),
        (2, ROOT / "m31_5way_seed2", "c2_tdm_m31_2048"),
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
    # Per-tier mean ± std for each label
    tiers = [
        ("strict", 500, 50),
        ("ttft500/tpot100", 500, 100),
        ("ttft500/tpot150", 500, 150),
        ("ttft500/tpot200", 500, 200),
        ("ttft1000/tpot100", 1000, 100),
        ("ttft1000/tpot150", 1000, 150),
        ("ttft1500/tpot200", 1500, 200),
    ]
    grids = {}
    for label, runs in RUNS.items():
        per_seed = [grid_for(sd, cfg) for _, sd, cfg in runs]
        grids[label] = per_seed

    # Cross-config table
    print(f"\n=== Controller ablation (3 seeds, mixed, qps=16) ===")
    print(f"  {'tier':<18s} {'tt':>5s} {'tp':>4s}  | "
          + "  ".join(f"{label:>20s}" for label in RUNS))
    print(f"  {'-'*18} {'-'*5} {'-'*4}  | "
          + "  ".join("-"*20 for _ in RUNS))
    for label_t, tt, tp in tiers:
        cells = []
        for label in RUNS:
            xs = [g[(tt, tp)] for g in grids[label]]
            mu, sd = mean_std(xs)
            cells.append(f"{mu:>5.1f}±{sd:>4.1f}        ")
        print(f"  {label_t:<18s} {tt:>5d} {tp:>4d}  | " + "  ".join(cells))

    # Delta tables
    print(f"\n=== Δ tables (mean ± std across 3 seeds, pp) ===")
    print(f"  {'tier':<18s} | {'M1+chunk - C1':>22s}  "
          f"{'M2.7+chunk - M1+chunk':>26s}  {'M2.7+chunk - C1':>22s}")
    print(f"  {'-'*18} | " + "  ".join("-"*max(len(s), 22) for s in
                                        ["M1+chunk - C1",
                                         "M2.7+chunk - M1+chunk",
                                         "M2.7+chunk - C1"]))
    for label_t, tt, tp in tiers:
        c1_xs = [g[(tt, tp)] for g in grids["c1_baseline"]]
        m1c_xs = [g[(tt, tp)] for g in grids["M1+chunk2048"]]
        m27c_xs = [g[(tt, tp)] for g in grids["M2.7+chunk2048"]]

        # Per-seed deltas (paired comparison)
        d_m1c_c1 = [m1c_xs[i] - c1_xs[i] for i in range(3)]
        d_m27c_m1c = [m27c_xs[i] - m1c_xs[i] for i in range(3)]
        d_m27c_c1 = [m27c_xs[i] - c1_xs[i] for i in range(3)]

        def fmt(xs):
            mu, sd = mean_std(xs)
            return f"{mu:+5.1f} ± {sd:>4.1f} pp"

        print(f"  {label_t:<18s} | "
              f"{fmt(d_m1c_c1):>22s}  "
              f"{fmt(d_m27c_m1c):>26s}  "
              f"{fmt(d_m27c_c1):>22s}")


if __name__ == "__main__":
    main()
