"""Multi-seed + multi-regime aggregator for M3.1 5-way ablation.

Reads:
  results/m31_5way_ablation/         (mixed, seed=0)
  results/m31_5way_seed1/            (mixed, seed=1)
  results/m31_5way_seed2/            (mixed, seed=2)
  results/m31_5way_long_seed0/       (long, seed=0)

Computes meet_slo% on the same 6×6 SLO grid for each (regime, config) and
reports mean ± stddev across seeds for the mixed regime. Long regime is
single-seed for now — reported as point estimate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")

MIXED_SEEDS = [
    (0, ROOT / "m31_5way_ablation"),
    (1, ROOT / "m31_5way_seed1"),
    (2, ROOT / "m31_5way_seed2"),
]
M1_CHUNK_SEEDS = [
    (0, ROOT / "m1_chunk_seed0"),
    (1, ROOT / "m1_chunk_seed1"),
    (2, ROOT / "m1_chunk_seed2"),
]
LONG_SEEDS = [
    (0, ROOT / "m31_5way_long_seed0"),
]
CONFIGS = ["c1_baseline", "c2_tdm_m27", "c2_tdm_m31_disabled",
           "c2_tdm_m31_2048", "c3_cp"]
# Per-config sweep dir overrides (when a config's data lives in a
# different sweep dir than the default).
SWEEP_OVERRIDES = {
    "c2_tdm_m1_chunk2048": M1_CHUNK_SEEDS,
}
QPS = 16.0
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load(sweep_dir: Path, cfg: str):
    path = sweep_dir / f"{cfg}_qps{QPS}.json"
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


def grid_for(sweep_dir: Path, cfg: str):
    joined, w, we = load(sweep_dir, cfg)
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


def aggregate(sweeps, label):
    print(f"\n{'='*78}\n=== {label}  (rows=tpot, cols=ttft, mean±std over "
          f"{len(sweeps)} seeds)\n{'='*78}")
    for cfg in CONFIGS:
        # Collect grids per seed
        per_seed = [grid_for(sd, cfg) for _, sd in sweeps]
        print(f"\n--- {cfg} ---")
        if len(sweeps) > 1:
            print(f"  tpot \\ ttft  | "
                  + "  ".join(f"{tt:>11d}" for tt in TTFT_GRID))
            print(f"  {'-'*13} | " + "  ".join("-"*11 for _ in TTFT_GRID))
            for tp in TPOT_GRID:
                row_vals = []
                for tt in TTFT_GRID:
                    xs = [g[(tt, tp)] for g in per_seed]
                    mu, sd = mean_std(xs)
                    row_vals.append(f"{mu:>5.1f}±{sd:>4.1f}")
                print(f"  {tp:>5d}        | " + "  ".join(row_vals))
        else:
            g = per_seed[0]
            print(f"  tpot \\ ttft  | "
                  + "  ".join(f"{tt:>6d}" for tt in TTFT_GRID))
            print(f"  {'-'*13} | " + "  ".join("-"*6 for _ in TTFT_GRID))
            for tp in TPOT_GRID:
                row = [g[(tt, tp)] for tt in TTFT_GRID]
                print(f"  {tp:>5d}        | "
                      + "  ".join(f"{v:>6.1f}" for v in row))


def cross_config_table(sweeps, label):
    print(f"\n=== {label}: cross-config ranking (mean±std on key tiers) ===")
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
          + "  ".join(f"{c:>16s}" for c in CONFIGS))
    print(f"  {'-'*18} {'-'*5} {'-'*4}  | "
          + "  ".join("-"*16 for _ in CONFIGS))
    for label_t, tt, tp in key:
        cells = []
        for cfg in CONFIGS:
            xs = [grid_for(sd, cfg)[(tt, tp)] for _, sd in sweeps]
            if len(xs) > 1:
                mu, sd = mean_std(xs)
                cells.append(f"{mu:>5.1f}±{sd:>4.1f}    ")
            else:
                cells.append(f"{xs[0]:>16.1f}")
        print(f"  {label_t:<18s} {tt:>5d} {tp:>4d}  | "
              + "  ".join(cells))


def ablation_deltas(sweeps, label):
    print(f"\n=== {label}: ablation deltas (mean ± std across seeds, pp) ===")
    print("  tier               | "
          "TDM(m27-c1)        fork(disable-m27)    chunking(2048-disable)  vs C1            vs C3")
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
    for label_t, tt, tp in key:
        d_tdm, d_fork, d_chunk, d_c1, d_c3 = [], [], [], [], []
        for _, sd in sweeps:
            c1 = grid_for(sd, "c1_baseline")[(tt, tp)]
            m27 = grid_for(sd, "c2_tdm_m27")[(tt, tp)]
            mdis = grid_for(sd, "c2_tdm_m31_disabled")[(tt, tp)]
            m2048 = grid_for(sd, "c2_tdm_m31_2048")[(tt, tp)]
            c3 = grid_for(sd, "c3_cp")[(tt, tp)]
            d_tdm.append(m27 - c1)
            d_fork.append(mdis - m27)
            d_chunk.append(m2048 - mdis)
            d_c1.append(m2048 - c1)
            d_c3.append(m2048 - c3)

        def fmt(xs):
            mu, sd = mean_std(xs)
            return f"{mu:+5.1f}±{sd:>4.1f}" if len(xs) > 1 else f"{mu:+5.1f}     "

        print(f"  {label_t:<18s} | "
              f"{fmt(d_tdm)}        {fmt(d_fork)}        "
              f"{fmt(d_chunk)}            {fmt(d_c1)}    {fmt(d_c3)}")


if __name__ == "__main__":
    aggregate(MIXED_SEEDS, "MIXED REGIME (3 seeds)")
    cross_config_table(MIXED_SEEDS, "MIXED REGIME")
    ablation_deltas(MIXED_SEEDS, "MIXED REGIME")

    aggregate(LONG_SEEDS, "LONG-PROMPT REGIME (1 seed)")
    cross_config_table(LONG_SEEDS, "LONG-PROMPT REGIME")
    ablation_deltas(LONG_SEEDS, "LONG-PROMPT REGIME")
