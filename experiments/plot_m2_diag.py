"""M2 SLO-adaptive controller diagnostic plot.

Reads {run_id}_iter.jsonl (per-iter telemetry, includes target_ratio + phase
+ decision source) and {run_id}_req.jsonl (per-request, includes ttft_ms /
tpot_ms_mean / finish_ts_ms). Produces a 2x2 panel:

  (0,0) target_ratio over time
  (0,1) rolling SLO violation rate (ttft, tpot) over completed-request stream
  (1,0) rolling phase fraction (prefill share over a sliding iter window)
  (1,1) decision source breakdown (bar chart)

Run with system python3 (matplotlib is not in the vllm venv):

  python3 plot_m2_diag.py \\
      --iter-log /vllm-workspace/Ascend-PD-TDM/results/tdm_trace/qps_sweep_c2_tdm_m2_iter.jsonl \\
      --req-log  /vllm-workspace/Ascend-PD-TDM/results/tdm_trace/qps_sweep_c2_tdm_m2_req.jsonl \\
      --out /vllm-workspace/Ascend-PD-TDM/results/m2_slo_adaptive/diag.png
"""
import argparse
import json
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def rolling_fraction(values, window: int) -> list[float]:
    """For a binary stream, return rolling mean over `window` items."""
    buf = deque(maxlen=window)
    out = []
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-log", required=True, type=Path)
    ap.add_argument("--req-log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slo-ttft-ms", type=float, default=500.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=50.0)
    ap.add_argument("--phase-window-iters", type=int, default=200,
                    help="rolling window (in iters) for phase fraction")
    ap.add_argument("--viol-window-reqs", type=int, default=64,
                    help="rolling window (in completed reqs) for violation rate")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    iters = load_jsonl(args.iter_log)
    reqs = load_jsonl(args.req_log)
    if not iters:
        raise SystemExit(f"empty iter log: {args.iter_log}")
    if not reqs:
        raise SystemExit(f"empty req log: {args.req_log}")

    # Normalize timestamps to seconds since first iter.
    t0 = iters[0]["ts_ms"]
    iter_t = [(r["ts_ms"] - t0) / 1000.0 for r in iters]
    iter_ratio = [r["target_ratio"] for r in iters]
    iter_actual_phase = [r["actual_phase"] for r in iters]
    iter_source = [r["source"] for r in iters]

    reqs_sorted = sorted(
        (r for r in reqs if r.get("finish_ts_ms") is not None),
        key=lambda r: r["finish_ts_ms"])
    req_t = [(r["finish_ts_ms"] - t0) / 1000.0 for r in reqs_sorted]
    req_ttft = [r.get("ttft_ms") for r in reqs_sorted]
    req_tpot = [r.get("tpot_ms_mean") for r in reqs_sorted]

    ttft_viol = [(1 if (v is not None and v > args.slo_ttft_ms) else 0)
                 for v in req_ttft]
    tpot_viol = [(1 if (v is not None and v > args.slo_tpot_ms) else 0)
                 for v in req_tpot]

    ttft_rate = rolling_fraction(ttft_viol, args.viol_window_reqs)
    tpot_rate = rolling_fraction(tpot_viol, args.viol_window_reqs)

    # Phase fraction: 1 if prefill, 0 otherwise.
    phase_bin = [1 if p == "prefill" else 0 for p in iter_actual_phase]
    prefill_frac = rolling_fraction(phase_bin, args.phase_window_iters)

    src_counts = Counter(iter_source)
    src_keys = sorted(src_counts.keys())
    src_vals = [src_counts[k] for k in src_keys]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_r, ax_v, ax_p, ax_s = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    ax_r.plot(iter_t, iter_ratio, lw=0.8, color="#1f77b4")
    ax_r.set_xlabel("time (s, since first iter)")
    ax_r.set_ylabel("target_ratio")
    ax_r.set_title("Controller output: target_ratio over time")
    ax_r.set_ylim(-0.02, 1.02)
    ax_r.grid(alpha=0.3)

    ax_v.plot(req_t, ttft_rate, label=f"ttft > {args.slo_ttft_ms:.0f}ms",
              color="#d62728", lw=1.2)
    ax_v.plot(req_t, tpot_rate, label=f"tpot > {args.slo_tpot_ms:.0f}ms",
              color="#1f77b4", lw=1.2)
    ax_v.axhline(0.05, ls="--", color="grey", lw=0.8,
                 label="target violation rate=0.05")
    ax_v.set_xlabel("time (s, request finish_ts)")
    ax_v.set_ylabel(f"violation rate (rolling, last {args.viol_window_reqs} reqs)")
    ax_v.set_title("SLO violation rate (controller's input signal)")
    ax_v.set_ylim(-0.02, 1.02)
    ax_v.legend(loc="upper right", fontsize=9)
    ax_v.grid(alpha=0.3)

    ax_p.plot(iter_t, prefill_frac, color="#2ca02c", lw=1.0)
    ax_p.set_xlabel("time (s)")
    ax_p.set_ylabel(f"prefill fraction (rolling {args.phase_window_iters} iters)")
    ax_p.set_title("Realized phase mix (actual_phase from telemetry)")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.grid(alpha=0.3)

    bars = ax_s.bar(range(len(src_keys)), src_vals, color="#9467bd")
    ax_s.set_xticks(range(len(src_keys)))
    ax_s.set_xticklabels(src_keys, rotation=20, ha="right", fontsize=9)
    ax_s.set_ylabel("iter count")
    ax_s.set_title("Decision source breakdown (who picked the phase)")
    total = sum(src_vals) or 1
    for b, v in zip(bars, src_vals):
        ax_s.text(b.get_x() + b.get_width() / 2, b.get_height(),
                  f"{v/total:.0%}", ha="center", va="bottom", fontsize=8)

    title = args.title or f"M2 diag — {args.iter_log.stem}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # Console summary so users can sanity-check without the PNG.
    print(f"\n[summary] {len(iters)} iters / {len(reqs_sorted)} finished reqs")
    print(f"  target_ratio: min={min(iter_ratio):.3f} "
          f"max={max(iter_ratio):.3f} "
          f"mean={sum(iter_ratio)/len(iter_ratio):.3f}")
    print(f"  prefill iters: {sum(phase_bin)}/{len(phase_bin)} "
          f"({sum(phase_bin)/len(phase_bin):.1%})")
    print(f"  ttft viol total: {sum(ttft_viol)}/{len(ttft_viol)}")
    print(f"  tpot viol total: {sum(tpot_viol)}/{len(tpot_viol)}")
    print(f"  source counts: {dict(src_counts)}")


if __name__ == "__main__":
    main()
