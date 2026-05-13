"""Posthoc analysis for P0-3a: target_ratio static scan on conv@1860s.

Goal: 判定 target_ratio 这个 PID actuator 的 trade-off 形状。三种可能结局
对应不同 thesis 后果:
  - U 型,M3.1 (PID) 落在最优区附近    → controller 设计成立
  - U 型,但有静态点优于 M3.1         → actuator 对,PID 没调好(P2')
  - 单调 / 平台                        → actuator 选错,thesis 重构

Data sources:
  results/static_scan_3a/conv_seed{0,1,2}/  — 4 new ratios (r01, r05, r07, r09)
  results/azure_main/conv_seed{0,1,2}/      — r03 anchor (c2_tdm_m1_chunk2048)
                                              + M3.1 anchor (c2_tdm_m31_2048)
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

SCAN_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/static_scan_3a")
AZURE_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACE = "conv"
SEEDS = [0, 1, 2]

# (label, ratio_value, config_name, source_root)
POINTS = [
    ("r=0.1", 0.1, "c2_tdm_m1_chunk2048_r01", SCAN_ROOT),
    ("r=0.3", 0.3, "c2_tdm_m1_chunk2048",     AZURE_ROOT),  # anchor
    ("r=0.5", 0.5, "c2_tdm_m1_chunk2048_r05", SCAN_ROOT),
    ("r=0.7", 0.7, "c2_tdm_m1_chunk2048_r07", SCAN_ROOT),
    ("r=0.9", 0.9, "c2_tdm_m1_chunk2048_r09", SCAN_ROOT),
    ("M3.1",  None, "c2_tdm_m31_2048",        AZURE_ROOT),  # PID, for comparison
]

TIERS = [
    ("strict          (ttft500/tpot50)",  500,  50),
    ("ttft500_tpot100",                   500, 100),
    ("ttft500_tpot150",                   500, 150),
    ("ttft500_tpot200",                   500, 200),
    ("ttft1000_tpot200",                 1000, 200),
    ("ttft1500_tpot300",                 1500, 300),
    ("ttft2000_tpot500",                 2000, 500),
]


def load(root, trace, seed, cfg):
    p = root / f"{trace}_seed{seed}" / f"{cfg}_qps0.0.json"
    return json.loads(p.read_text())


def meet(joined, w, we, t_lim, p_lim):
    in_win = [j for j in joined if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n = len(in_win)
    nm = sum(1 for j in in_win
             if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
             and j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim
             and j.get("output_tokens"))
    return nm, n


def collect(root, trace, cfg, t_lim, p_lim):
    out = []
    for s in SEEDS:
        d = load(root, trace, s, cfg)
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def window_stat(root, trace, cfg, key, sub=None):
    out = []
    for s in SEEDS:
        w = load(root, trace, s, cfg)["summary"]["window"]
        v = w.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        out.append(v if v is not None else float("nan"))
    return out


def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def shape_diagnose(ratios, means):
    """Quick heuristic to label the trade-off shape per tier."""
    if max(means) - min(means) < 1.0:
        return "FLAT (<1pp range)"
    # check monotone
    diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
    if all(d >= -0.5 for d in diffs):
        return "MONOTONE↑ (ratio↑ → meet%↑)"
    if all(d <= 0.5 for d in diffs):
        return "MONOTONE↓ (ratio↑ → meet%↓)"
    # find argmax
    am = means.index(max(means))
    if am == 0:
        return "MONOTONE↓-ish (max at r=0.1)"
    if am == len(means) - 1:
        return "MONOTONE↑-ish (max at r=0.9)"
    return f"U-SHAPED (peak at r={ratios[am]})"


def main():
    print("=" * 100)
    print("P0-3a: target_ratio static scan on conv@1860s")
    print("  — verifies whether target_ratio has bidirectional trade-off (PID prerequisite)")
    print(f"  5 static ratios (M1+chunk=2048, controller_kind=static) + M3.1 (PID closed-loop) × 3 seeds × 60s")
    print("=" * 100)

    # ---- 1. meet_slo% vs ratio table ----
    print("\n## meet_slo% (mean±std, 3 seeds) — rows: SLO tier, cols: target_ratio + M3.1\n")
    header = f"{'tier':<35}" + "".join(f"{p[0]:>12}" for p in POINTS)
    print(header)
    print("-" * len(header))
    static_ratios_for_shape = [p[1] for p in POINTS if p[1] is not None]
    for tier_name, t_lim, p_lim in TIERS:
        row = f"{tier_name:<35}"
        static_means = []
        for label, ratio, cfg, root in POINTS:
            pcts = collect(root, TRACE, cfg, t_lim, p_lim)
            row += f"{fmt(pcts):>12}"
            if ratio is not None:
                static_means.append(statistics.mean(pcts))
        shape = shape_diagnose(static_ratios_for_shape, static_means)
        row += f"  | shape: {shape}"
        print(row)

    # ---- 2. Best static vs PID — the decisive comparison ----
    print("\n" + "=" * 100)
    print("Best static ratio vs M3.1 PID — decides controller-design fate")
    print("=" * 100)
    print(f"\n{'tier':<35}{'best static':>18}{'M3.1 PID':>12}{'PID - best_static':>20}")
    print("-" * 85)
    for tier_name, t_lim, p_lim in TIERS:
        static_means_by_label = []
        for label, ratio, cfg, root in POINTS:
            if ratio is None:
                continue
            pcts = collect(root, TRACE, cfg, t_lim, p_lim)
            static_means_by_label.append((label, statistics.mean(pcts), pcts))
        # best static
        best_label, best_mean, best_pcts = max(static_means_by_label, key=lambda x: x[1])
        # PID
        pid_pcts = collect(AZURE_ROOT, TRACE, "c2_tdm_m31_2048", t_lim, p_lim)
        pid_mean = statistics.mean(pid_pcts)
        # paired Δ
        d = [p - b for p, b in zip(pid_pcts, best_pcts)]
        d_mean = statistics.mean(d)
        d_sd = statistics.stdev(d) if len(d) > 1 else 0.0
        print(f"{tier_name:<35}"
              f"{best_label+' '+f'{best_mean:5.1f}':>18}"
              f"{pid_mean:>12.1f}"
              f"{d_mean:>+10.1f}±{d_sd:<4.1f}pp")

    # ---- 3. Window stats per ratio ----
    print("\n" + "=" * 100)
    print("Window stats (mean across seeds) — diagnose ttft / tpot trade-off mechanism")
    print("=" * 100)
    metrics = [
        ("ttft_p50 (ms)",   "ttft_ms",      "p50"),
        ("ttft_p99 (ms)",   "ttft_ms",      "p99"),
        ("tpot_p50 (ms)",   "tpot_ms_mean", "p50"),
        ("tpot_p99 (ms)",   "tpot_ms_mean", "p99"),
        ("e2e_p99 (s)",     "e2e_ms",       "p99"),
        ("goodput (tok/s)", "goodput_tok_s", None),
    ]
    header = f"{'metric':<22}" + "".join(f"{p[0]:>12}" for p in POINTS)
    print(header)
    print("-" * len(header))
    for label, key, sub in metrics:
        row = f"{label:<22}"
        for plabel, ratio, cfg, root in POINTS:
            vals = window_stat(root, TRACE, cfg, key, sub)
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if "e2e" in label:
                row += f"{m/1000:>8.1f}±{sd/1000:>3.1f}"
            else:
                row += f"{m:>8.1f}±{sd:>3.1f}"
        print(row)

    # ---- 4. Verdict scaffold ----
    print("\n" + "=" * 100)
    print("Verdict (manual interpretation per tier):")
    print("  - If shape=U-SHAPED and PID-best_static ≥ 0:  controller design stands.")
    print("  - If shape=U-SHAPED and PID-best_static < 0:  P2' (tune PID params) required.")
    print("  - If shape=MONOTONE / FLAT:                   actuator misdesigned, thesis must be revised.")
    print("=" * 100)


if __name__ == "__main__":
    main()
