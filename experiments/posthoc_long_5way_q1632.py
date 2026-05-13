"""Post-hoc SLO-grid analysis for long_5way_q1632 sweep (resumed after week-limit).

5 configs × qps∈{16,32} × 3 seeds × 60s long prompt.
Re-uses meet_slo judging logic from posthoc_slo_grid.py (ttft<SLO_ttft AND tpot_mean<SLO_tpot).

Key questions:
  Q1. qps=32 saturated regime: does M2.7 (PID) beat M1 (static_ratio)?  (§9.3 / P2' question)
  Q2. M3.1 long-regime gap (was +15-22pp at seed0 single-seed) — does it survive 3-seed mean±std?
  Q3. C3 at qps=32 long: does it become competitive in saturated regime?
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/long_5way_q1632")
CONFIGS = ["c1_baseline", "c2_tdm", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
QPS_POINTS = [16.0, 32.0]
SEEDS = [0, 1, 2]

# SLO tiers we care about for narrative
TIERS = [
    ("strict",          500,  50),
    ("ttft500_tpot100", 500,  100),
    ("ttft500_tpot150", 500,  150),
    ("ttft500_tpot200", 500,  200),
    ("ttft1000_tpot100", 1000, 100),
    ("ttft1000_tpot150", 1000, 150),
    ("ttft1500_tpot200", 1500, 200),
    ("long_ctx",        1500, 300),
]


def meet_slo_count(joined, warmup_s, window_end_s, slo_ttft_ms, slo_tpot_ms):
    in_win = [j for j in joined
              if warmup_s <= j["arrival_time_s"] < window_end_s
              and j["status"] == 200 and j.get("matched")]
    n_matched = len(in_win)
    n_meet = sum(
        1 for j in in_win
        if j.get("ttft_ms") is not None
        and j.get("tpot_ms_mean") is not None
        and j["ttft_ms"] < slo_ttft_ms
        and j["tpot_ms_mean"] < slo_tpot_ms
        and j.get("output_tokens")
    )
    return n_meet, n_matched


def load_run(seed, cfg, qps):
    p = ROOT / f"seed{seed}" / f"{cfg}_qps{qps}.json"
    d = json.loads(p.read_text())
    s = d["summary"]
    warmup_s = s["warmup_s"]
    window_end_s = warmup_s + s["duration_s"]
    return d["joined"], warmup_s, window_end_s, s


def collect_meet_pct(cfg, qps, slo_ttft, slo_tpot):
    pcts = []
    for seed in SEEDS:
        joined, w, we, _ = load_run(seed, cfg, qps)
        n_meet, n_matched = meet_slo_count(joined, w, we, slo_ttft, slo_tpot)
        pcts.append(100.0 * n_meet / max(1, n_matched))
    return pcts


def fmt_mean_std(pcts):
    m = statistics.mean(pcts)
    sd = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def collect_window_stat(cfg, qps, key, sub="p99"):
    """e.g. key='ttft_ms', sub='p99' → list[3] of the window p99."""
    out = []
    for seed in SEEDS:
        _, _, _, s = load_run(seed, cfg, qps)
        win = s["window"]
        v = win.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        out.append(v if v is not None else float("nan"))
    return out


def main():
    print("=" * 110)
    print("LONG 5-way sweep (qps∈{16,32}, 3 seeds × 60s, long prompt)")
    print("=" * 110)

    # 1) SLO-tier grid: mean±std across seeds, for each (config, qps, tier)
    for qps in QPS_POINTS:
        print(f"\n## qps = {qps}  meet_slo% (mean±std over {len(SEEDS)} seeds)\n")
        header = f"{'tier':<20}" + "".join(f"{c:>20}" for c in CONFIGS)
        print(header)
        print("-" * len(header))
        for name, t_ms, p_ms in TIERS:
            row = f"{name:<20}"
            for cfg in CONFIGS:
                pcts = collect_meet_pct(cfg, qps, t_ms, p_ms)
                row += f"{fmt_mean_std(pcts):>20}"
            print(row)

    # 2) Window stats (ttft p99 / tpot p99 / goodput)
    print("\n" + "=" * 110)
    print("Window stats (mean across seeds)")
    print("=" * 110)
    for qps in QPS_POINTS:
        print(f"\n## qps = {qps}\n")
        header = f"{'metric':<22}" + "".join(f"{c:>20}" for c in CONFIGS)
        print(header)
        print("-" * len(header))
        for label, key, sub in [
            ("ttft_p50 (ms)",   "ttft_ms",       "p50"),
            ("ttft_p99 (ms)",   "ttft_ms",       "p99"),
            ("tpot_mean_p50",   "tpot_ms_mean",  "p50"),
            ("tpot_mean_p99",   "tpot_ms_mean",  "p99"),
            ("goodput (tok/s)", "goodput_tok_s", None),
            ("out_tput (tok/s)","output_throughput_tok_s", None),
        ]:
            row = f"{label:<22}"
            for cfg in CONFIGS:
                vals = collect_window_stat(cfg, qps, key, sub)
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"{m:>14.1f}±{sd:4.1f}"
            print(row)

    # 3) Headline contrasts for the 3 narrative questions
    print("\n" + "=" * 110)
    print("Headline contrasts (Δpp, mean ± std)")
    print("=" * 110)

    def delta(cfg_a, cfg_b, qps, tier_ttft, tier_tpot):
        a = collect_meet_pct(cfg_a, qps, tier_ttft, tier_tpot)
        b = collect_meet_pct(cfg_b, qps, tier_ttft, tier_tpot)
        diffs = [x - y for x, y in zip(a, b)]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        return f"{m:+5.1f}±{sd:4.1f}pp"

    contrasts = [
        # Q1. PID vs static_ratio (M2.7 vs M1)  → if positive at qps=32 saturated, P2' has hope
        ("M2.7 - M1 (PID vs static)", "c2_tdm",            "c2_tdm_m1_chunk2048"),
        # Q1'. M2.7 vs C2 (TDM controller raw effect)
        ("M2.7 - C2 (controller add)", "c2_tdm",           "c2_tdm"),  # placeholder; c2_tdm itself is M2.7-enabled? clarify in print
        # Q2. M3.1 gap vs C1 / C3
        ("M3.1 - C1 (chunking-on-TDM vs hybrid)", "c2_tdm_m31_2048", "c1_baseline"),
        ("M3.1 - C3 (chunking-on-TDM vs chunked-prefill)", "c2_tdm_m31_2048", "c3_cp"),
        # Q3. C3 sat regime
        ("C3 - C1 (chunked prefill vs hybrid)", "c3_cp", "c1_baseline"),
    ]

    for qps in QPS_POINTS:
        print(f"\n## qps = {qps}\n")
        for tier_name, t_ms, p_ms in TIERS:
            print(f"  -- tier={tier_name} (ttft<{t_ms} tpot<{p_ms})")
            for label, a, b in contrasts:
                if a == b:
                    continue
                print(f"    {label:<48} {delta(a, b, qps, t_ms, p_ms)}")


if __name__ == "__main__":
    main()
