"""Post-hoc B: Interference decomposition for the defining figure (data version).

D-012 Phase 0 T2. Reads azure_p15 telemetry to project 4 paradigms onto the
mean/tail interference 2D space.

Operational definition:
  - mean_latency  = median of per-request tpot_ms_mean   (where the bulk sits)
  - tail_excess   = median of per-request (tpot_ms_p99 - tpot_ms_mean)
                    (how much the *worst* decode interval in a request
                     exceeds that request's average — within-request tail)
  - ttft_mean     = mean of per-request ttft_ms          (prefill side proxy)

We also extract:
  - prefill_iter_dur_mean / p99 = decode silence per prefill iter
  - prefill_iter_fraction = how much wallclock is spent in prefill iters

Outputs:
  results/_posthoc_b_interference/
    summary.csv               -- one row per (workload, window, config)
    defining_figure.png       -- 2D scatter, mean vs tail, 4 paradigm points
    silence_table.csv         -- prefill silence diagnostic
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p15")
OUT = Path("/vllm-workspace/Ascend-PD-TDM/results/_posthoc_b_interference")
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
CONFIG_LABEL = {
    "c1_baseline":         "C1 Unified",
    "c2_tdm_m1_chunk2048": "M1+chunk",
    "c2_tdm_m31_2048":     "M3.1 (ours)",
    "c3_cp":               "C3 Chunked Prefill",
}
CONFIG_COLOR = {
    "c1_baseline":         "#444444",
    "c2_tdm_m1_chunk2048": "#888888",
    "c2_tdm_m31_2048":     "#b73232",
    "c3_cp":               "#1f3a5f",
}
CONFIG_MARKER = {
    "c1_baseline":         "o",
    "c2_tdm_m1_chunk2048": "s",
    "c2_tdm_m31_2048":     "*",
    "c3_cp":               "D",
}

WORKLOADS = ["conv", "code"]
WINDOWS   = ["w0", "w1", "w2"]
SEEDS     = [0, 1, 2]


def load_reqs(d: Path, cfg: str) -> list[dict]:
    p = d / "tdm_trace" / f"qps_sweep_{cfg}_req.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_iters(d: Path, cfg: str) -> list[dict]:
    p = d / "tdm_trace" / f"qps_sweep_{cfg}_iter.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def safe_mean(xs):
    return statistics.mean(xs) if xs else float("nan")

def safe_median(xs):
    return statistics.median(xs) if xs else float("nan")

def percentile(xs, p):
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(round((p / 100) * (len(xs2) - 1)))))
    return xs2[k]


def summarize(workload: str, window: str, cfg: str) -> dict:
    """Aggregate over 3 seeds for one (workload, window, cfg)."""
    per_seed_metrics = []
    for seed in SEEDS:
        d = ROOT / f"{workload}_{window}_seed{seed}"
        reqs = load_reqs(d, cfg)
        iters = load_iters(d, cfg)
        if not reqs:
            continue

        # Per-request distributions
        ttfts     = [r["ttft_ms"]      for r in reqs if r.get("ttft_ms") is not None]
        tpot_means = [r["tpot_ms_mean"] for r in reqs if r.get("tpot_ms_mean") is not None]
        tpot_p99s  = [r["tpot_ms_p99"]  for r in reqs if r.get("tpot_ms_p99")  is not None]

        # Within-request tail excess (per-request tail vs per-request mean)
        tail_excess = [
            r["tpot_ms_p99"] - r["tpot_ms_mean"]
            for r in reqs
            if (r.get("tpot_ms_p99") is not None) and (r.get("tpot_ms_mean") is not None)
        ]

        # Prefill-iter silence
        prefill_iters = [it for it in iters if it.get("actual_phase") == "prefill"]
        decode_iters  = [it for it in iters if it.get("actual_phase") == "decode"]
        prefill_durs  = [it["iter_duration_ms"] for it in prefill_iters
                         if it.get("iter_duration_ms") is not None]
        decode_durs   = [it["iter_duration_ms"] for it in decode_iters
                         if it.get("iter_duration_ms") is not None]
        total_dur     = sum(prefill_durs) + sum(decode_durs)
        prefill_frac  = sum(prefill_durs) / total_dur if total_dur > 0 else 0.0

        per_seed_metrics.append({
            "ttft_mean":           safe_mean(ttfts),
            "ttft_p99":            percentile(ttfts, 99),
            "tpot_mean_median":    safe_median(tpot_means),    # bulk of decode latency
            "tpot_mean_p99":       percentile(tpot_means, 99), # ACROSS-request tail
                                                               # (= paper main metric TPOT side)
            "tail_excess_median":  safe_median(tail_excess),   # WITHIN-request tail uplift
            "tail_excess_p99":     percentile(tail_excess, 99),
            "prefill_iter_dur_mean": safe_mean(prefill_durs),
            "prefill_iter_dur_p99":  percentile(prefill_durs, 99),
            "prefill_frac":        prefill_frac,
            "n_reqs":              len(reqs),
            "n_prefill_iters":     len(prefill_iters),
        })

    if not per_seed_metrics:
        return None

    # Average across seeds
    keys = per_seed_metrics[0].keys()
    out = {"workload": workload, "window": window, "config": cfg}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if not (isinstance(m[k], float) and m[k] != m[k])]
        out[k] = safe_mean(vals) if vals else float("nan")
    return out


def main():
    rows = []
    for workload in WORKLOADS:
        for window in WINDOWS:
            for cfg in CONFIGS:
                r = summarize(workload, window, cfg)
                if r:
                    rows.append(r)

    if not rows:
        print("[fatal] no data loaded")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)

    # Print key table
    print("\n=== mean/tail interference per (workload, window, config) ===")
    keep = ["workload", "window", "config",
            "ttft_mean", "tpot_mean_median", "tail_excess_median",
            "prefill_iter_dur_mean", "prefill_frac"]
    pretty = df[keep].copy()
    pretty.columns = ["wl", "win", "cfg",
                      "ttft_mean", "tpot_med", "tail_excess_med",
                      "prefill_dur", "prefill_frac"]
    print(pretty.to_string(index=False))

    # === Defining figure (data version) ===
    # 4 subplots: 2 workload × 2 tail-view (within-req / across-req)
    # within-req view  = (tpot_mean_median, tail_excess_median)
    # across-req view  = (tpot_mean_median, tpot_mean_p99) — aligns with paper main metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for col_idx, workload in enumerate(WORKLOADS):
        sub = df[df["workload"] == workload]
        if sub.empty:
            continue

        for row_idx, view in enumerate(["within", "across"]):
            ax = axes[row_idx, col_idx]
            ykey = "tail_excess_median" if view == "within" else "tpot_mean_p99"
            ylabel = ("Within-request tail uplift\n(per-req tpot_p99 − tpot_mean, ms)"
                      if view == "within"
                      else "Across-request TPOT tail\n(p99 of per-req tpot_mean, ms)\n[paper main metric]")
            title_suffix = ("within-request tail (per-req silence imprint)"
                            if view == "within"
                            else "across-request tail (paper main metric)")

            for cfg in CONFIGS:
                cfg_sub = sub[sub["config"] == cfg]
                if cfg_sub.empty:
                    continue
                for _, r in cfg_sub.iterrows():
                    ax.scatter(
                        r["tpot_mean_median"], r[ykey],
                        marker=CONFIG_MARKER[cfg],
                        color=CONFIG_COLOR[cfg],
                        s=180 if cfg == "c2_tdm_m31_2048" else 110,
                        edgecolor="black", linewidth=0.7,
                        label=CONFIG_LABEL[cfg] if (r["window"] == "w1" and row_idx == 0) else None,
                        zorder=3,
                    )
                    ax.annotate(r["window"],
                                (r["tpot_mean_median"], r[ykey]),
                                xytext=(5, 4), textcoords="offset points",
                                fontsize=7.5, color=CONFIG_COLOR[cfg])
            ax.set_xlabel("Mean-domain interference (per-request tpot_median, ms)",
                          fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{workload.upper()} — {title_suffix}",
                         fontsize=10.5)
            ax.grid(alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle("Defining figure (data version): 4 paradigms in interference plane\n"
                 "[Azure trace, w0/w1/w2 windows, 3 seeds mean]", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "defining_figure.png", dpi=140)
    plt.close(fig)

    # Per-paradigm verbal summary (does it match doc's T_* predictions?)
    print("\n=== Direction check vs doc framing prediction ===")
    print(f"{'workload':<8}{'window':<6}{'config':<22}"
          f"{'tpot_med':>10}{'tail_exc':>10}{'pf_frac':>9}")
    for _, r in df.iterrows():
        print(f"{r['workload']:<8}{r['window']:<6}{r['config']:<22}"
              f"{r['tpot_mean_median']:>10.1f}{r['tail_excess_median']:>10.1f}"
              f"{r['prefill_frac']:>9.3f}")

    print(f"\nArtifacts → {OUT}/")
    print("  summary.csv,  defining_figure.png")


if __name__ == "__main__":
    main()
