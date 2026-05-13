"""Posthoc analysis for Azure trace replay main experiment (L3).

results/azure_main/{conv,code}_seed{0,1,2}/ — 2 traces × 4 configs × 3 seeds × 60s.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACES = ["conv", "code"]
CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
SHORT = {"c1_baseline": "C1",
         "c2_tdm_m1_chunk2048": "M1+chunk",
         "c2_tdm_m31_2048": "M3.1",
         "c3_cp": "C3"}
SEEDS = [0, 1, 2]

TIERS = [
    ("strict",          500,  50),
    ("ttft500_tpot100", 500, 100),
    ("ttft500_tpot150", 500, 150),
    ("ttft500_tpot200", 500, 200),
    ("ttft1000_tpot200", 1000, 200),
    ("ttft1500_tpot300", 1500, 300),
    ("ttft2000_tpot500", 2000, 500),
]


def meet(joined, w, we, t_lim, p_lim):
    in_win = [j for j in joined if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n = len(in_win)
    nm = sum(1 for j in in_win
             if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
             and j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim
             and j.get("output_tokens"))
    return nm, n


def load(trace, seed, cfg):
    p = ROOT / f"{trace}_seed{seed}" / f"{cfg}_qps0.0.json"
    return json.loads(p.read_text())


def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def collect(trace, cfg, t_lim, p_lim):
    out = []
    for s in SEEDS:
        d = load(trace, s, cfg)
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def window_stat(trace, cfg, key, sub=None):
    out = []
    for s in SEEDS:
        w = load(trace, s, cfg)["summary"]["window"]
        v = w.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        out.append(v if v is not None else float("nan"))
    return out


def main():
    print("=" * 100)
    print("Azure trace replay main (L3): 2 traces × 4 configs × 3 seeds × 60s")
    print(f"  conv at start_offset=1860s (peak 8.4 qps avg, ~506 reqs/60s)")
    print(f"  code at start_offset=570s  (peak 11.1 qps avg, ~668 reqs/60s)")
    print("=" * 100)

    # ---- 1. SLO grid per trace ----
    for trace in TRACES:
        print(f"\n## {trace}  meet_slo% (mean±std over {len(SEEDS)} seeds)\n")
        header = f"{'tier':<22}" + "".join(f"{SHORT[c]:>14}" for c in CONFIGS)
        print(header)
        print("-" * len(header))
        for name, t_lim, p_lim in TIERS:
            row = f"{name:<22}"
            for cfg in CONFIGS:
                pcts = collect(trace, cfg, t_lim, p_lim)
                row += f"{fmt(pcts):>14}"
            print(row)

    # ---- 2. Window stats ----
    print("\n" + "=" * 100)
    print("Window stats (mean across seeds)")
    print("=" * 100)
    metrics = [
        ("ttft_p50 (ms)",   "ttft_ms",       "p50"),
        ("ttft_p99 (ms)",   "ttft_ms",       "p99"),
        ("tpot_p50",        "tpot_ms_mean",  "p50"),
        ("tpot_p99",        "tpot_ms_mean",  "p99"),
        ("e2e_p99 (s)",     "e2e_ms",        "p99"),
        ("goodput (tok/s)", "goodput_tok_s", None),
        ("out_tput (tok/s)","output_throughput_tok_s", None),
    ]
    for trace in TRACES:
        print(f"\n## {trace}\n")
        header = f"{'metric':<22}" + "".join(f"{SHORT[c]:>14}" for c in CONFIGS)
        print(header)
        print("-" * len(header))
        for label, key, sub in metrics:
            row = f"{label:<22}"
            for cfg in CONFIGS:
                vals = window_stat(trace, cfg, key, sub)
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                if "e2e" in label:
                    row += f"{m/1000:>9.1f}±{sd/1000:>3.1f}"
                else:
                    row += f"{m:>9.1f}±{sd:>3.1f}"
            print(row)

    # ---- 3. Headline contrasts ----
    print("\n" + "=" * 100)
    print("Headline Δ (mean±std pp, 3 seeds, paired by seed)")
    print("=" * 100)

    def delta(trace, a, b, t_lim, p_lim):
        A = collect(trace, a, t_lim, p_lim)
        B = collect(trace, b, t_lim, p_lim)
        d = [x - y for x, y in zip(A, B)]
        m = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        return f"{m:+5.1f}±{sd:4.1f}"

    pairs = [
        ("M3.1 - C1 (closed-loop TDM vs hybrid)", "c2_tdm_m31_2048", "c1_baseline"),
        ("M3.1 - M1 (PID earn vs static_ratio)", "c2_tdm_m31_2048", "c2_tdm_m1_chunk2048"),
        ("M3.1 - C3 (TDM vs chunked-prefill)",   "c2_tdm_m31_2048", "c3_cp"),
        ("M1 - C1 (chunk effect on TDM)",        "c2_tdm_m1_chunk2048", "c1_baseline"),
        ("C3 - C1 (chunked-prefill vs hybrid)",  "c3_cp", "c1_baseline"),
    ]
    for trace in TRACES:
        print(f"\n## {trace}\n")
        for tier_name, t_lim, p_lim in TIERS:
            print(f"  tier={tier_name:<22}", end="")
            for label, a, b in pairs:
                print(f"  {label[:25]:<25}={delta(trace, a, b, t_lim, p_lim):>11}pp", end="")
            print()
        print()


if __name__ == "__main__":
    main()
