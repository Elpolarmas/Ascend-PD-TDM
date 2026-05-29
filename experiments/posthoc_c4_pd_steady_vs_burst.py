"""Compare c4_pd steady (Poisson IID) vs burst arrival (2026-05-25).

Goal: separate "1P1D fundamental cost" from "burst amplifier".
- If steady c4_pd ≈ m31/c1/c3 → burst is the killer
- If steady c4_pd still bad → 1P1D fundamental
- Likely somewhere between

Input:
  burst:  results/c4_pd_supplement/conv_k{0.5,1.0,1.4}_seed0/qps_sweep_summary.json
  steady: results/c4_pd_steady_smoke/conv_qps{2.85,5.70,7.98}_seed0/qps_sweep_summary.json

m31 baseline (T6_FINDINGS conv s1):
  k=0.5: TTFT mean 146  | k=1.0: 191  | k=1.4: 191
"""
import json
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
BURST = ROOT / "c4_pd_supplement"
STEADY = ROOT / "c4_pd_steady_smoke"


def load_window(cell_dir: Path):
    p = cell_dir / "qps_sweep_summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    cfgs = d.get("configs", {})
    rec = cfgs.get("c4_pd", [None])[0]
    if rec is None:
        return None
    return rec.get("window"), rec.get("qps_actual_arrival")


def fmt(w, qps):
    if w is None:
        return "  (missing)"
    t = w["ttft_ms"]
    p = w["tpot_ms_mean"]
    return (f"  QPS={qps:.2f}  "
            f"meet={w['n_meet_slo']}/{w['n_matched']}  "
            f"TTFT mean={t['mean']:.0f}  p99={t['p99']:.0f}  "
            f"TPOT mean={p['mean']:.1f}  p99={p['p99']:.1f}  "
            f"out_thr={w['output_throughput_tok_s']:.0f}")


def main():
    pairs = [
        ("k=0.5 (avg QPS 2.85)", "conv_k0.5_seed0", "conv_qps2.85_seed0", 146),
        ("k=1.0 (avg QPS 5.70)", "conv_k1.0_seed0", "conv_qps5.70_seed0", 191),
        ("k=1.4 (avg QPS 7.98)", "conv_k1.4_seed0", "conv_qps7.98_seed0", 191),
    ]
    print(f"\n{'='*100}\nc4_pd: burst vs steady arrival  (conv workload)\n{'='*100}")
    print(f"m31 baseline TTFT mean (T6_FINDINGS conv s1) shown as reference.\n")
    for label, burst_cell, steady_cell, m31_ttft in pairs:
        print(f"\n--- {label} ---")
        bw, bq = load_window(BURST / burst_cell) or (None, None)
        sw, sq = load_window(STEADY / steady_cell) or (None, None)
        print(f"  m31 TTFT mean reference: {m31_ttft}ms")
        print(f"  burst :{fmt(bw, bq) if bw else '  (missing)'}")
        print(f"  steady:{fmt(sw, sq) if sw else '  (missing)'}")
        if bw and sw:
            r_ttft = bw["ttft_ms"]["mean"] / max(sw["ttft_ms"]["mean"], 1.0)
            print(f"  → burst TTFT is {r_ttft:.1f}× steady TTFT")
            print(f"  → steady c4_pd TTFT is {sw['ttft_ms']['mean']/m31_ttft:.1f}× m31 baseline")


if __name__ == "__main__":
    main()
