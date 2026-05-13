"""P1 telemetry diagnostic plots: PID controller behaviour + chunk budget
utilisation, across the chunk-size scan {512, 1024, 2048} × 3 seeds in the
long-prompt regime at qps=16.

Reads `tdm_trace/{run_id}_ctrl.jsonl` (per-PID-tick) and
`tdm_trace/{run_id}_chunk.jsonl` (per-P-iter chunk plan). Produces 5 figures
(saved to OUTDIR), each answering one diagnostic question:

 1. ratio_trajectory.png — ratio_after over wall time, by chunk size and seed.
    Annotated with saturation/clamp activations. Answers "is PID actuating
    at all, or stuck?"

 2. saturation_share.png — fraction of update ticks where tpot_saturated /
    tpot_clamp_active was true, by chunk size. Answers "is the saturation
    state machine permanently latched, hiding the err_tpot signal?"

 3. chunk_budget_utilization.png — histogram of chunk_budget_used /
    chunk_budget_in per P-iter, by chunk size. Answers "if I shrink chunk,
    does it cost overhead (= we're already filling the budget) or is it free
    (= long prompts don't fill 2048-token budgets)?"

 4. truncated_share.png — share of P-iters with at least one truncated
    request, by chunk size. Answers "is `prefill_chunk_tokens` actually
    binding scheduler decisions, or just nominally present?"

 5. err_vs_ratio.png — scatter of (err_ttft, err_tpot_effective) vs the
    delta_slo / ratio_after movement, by chunk size. Answers "when the
    controller sees a real err signal, does the ratio actually move?"

Usage:
  python3 posthoc_p1_ctrl_chunk.py
  # outputs to /vllm-workspace/Ascend-PD-TDM/results/p1_chunk_scan/diag/
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------- config -------------------

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
TRACE_DIR = ROOT / "tdm_trace"
SCAN_DIR = ROOT / "p1_chunk_scan"
OUTDIR = SCAN_DIR / "diag"
SEEDS = (0, 1, 2)
# Config name → chunk size. Sweep produces qps_sweep_{config}_{ctrl,chunk}.jsonl
# in TRACE_DIR (the trace dir is global; sweep dirs only carry summary jsons).
# We tag each chunk size with its run_id for trace lookup.
CHUNK_CONFIGS = [
    ("c2_tdm_m31", 512),       # canonical 512
    ("c2_tdm_m31_1024", 1024),
    ("c2_tdm_m31_2048", 2048),
]

# Seed-aware run_id template. Sweep writes run_id = qps_sweep_{config} (no seed
# embedded by default — seed goes into the seed-folder OUTDIR but the global
# trace dir overwrites between seeds). To keep all 3 seeds available for
# diagnosis we re-symlink/copy ctrl.jsonl/chunk.jsonl into per-seed slots.
# See _seed_trace_files() — supports both layouts (per-seed or last-seed-only).


# ------------------- IO helpers -------------------


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _seed_trace_files(config: str, seed: int,
                      stream: str) -> Optional[Path]:
    """Find the {ctrl,chunk}.jsonl for this (config, seed) tuple.

    Two supported layouts:
      A) per-seed: results/p1_chunk_scan/seed{N}/tdm_trace/qps_sweep_{cfg}_{stream}.jsonl
      B) global last-only: tdm_trace/qps_sweep_{cfg}_{stream}.jsonl
         (last seed wins; only useful when SEEDS=(0,))
    """
    # Layout A
    a = SCAN_DIR / f"seed{seed}" / "tdm_trace" / \
        f"qps_sweep_{config}_{stream}.jsonl"
    if a.exists():
        return a
    # Layout B (only seed 0 makes sense)
    if seed == 0:
        b = TRACE_DIR / f"qps_sweep_{config}_{stream}.jsonl"
        if b.exists():
            return b
    return None


# ------------------- Plot 1: ratio trajectory -------------------


def plot_ratio_trajectory():
    fig, axes = plt.subplots(len(SEEDS), 1, figsize=(10, 8), sharex=True)
    if len(SEEDS) == 1:
        axes = [axes]
    for ax, seed in zip(axes, SEEDS):
        for config, size in CHUNK_CONFIGS:
            p = _seed_trace_files(config, seed, "ctrl")
            ticks = _load_jsonl(p) if p else []
            if not ticks:
                continue
            t0 = ticks[0]["ts_ms"]
            xs = [(t["ts_ms"] - t0) / 1000.0 for t in ticks]
            ys = [t["ratio_after"] for t in ticks]
            ax.plot(xs, ys, label=f"chunk={size}", alpha=0.8)
            # Mark saturation activations.
            sat_xs = [
                x for x, t in zip(xs, ticks) if t["tpot_saturated"]
            ]
            if sat_xs:
                ax.scatter(sat_xs, [0.05] * len(sat_xs), s=8,
                           marker="x", alpha=0.5)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel(f"ratio (seed={seed})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time since first tick (s)")
    fig.suptitle("PID ratio trajectory (× = tpot_saturated tick)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "ratio_trajectory.png", dpi=130)
    plt.close(fig)


# ------------------- Plot 2: saturation share -------------------


def plot_saturation_share():
    rows: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for config, size in CHUNK_CONFIGS:
        for seed in SEEDS:
            p = _seed_trace_files(config, seed, "ctrl")
            ticks = _load_jsonl(p) if p else []
            if not ticks:
                continue
            n = len(ticks)
            sat = sum(1 for t in ticks if t["tpot_saturated"]) / n
            clamp = sum(1 for t in ticks if t["tpot_clamp_active"]) / n
            cold = sum(1 for t in ticks if t["cold_start"]) / n
            kvf = sum(1 for t in ticks if t["kv_freeze_triggered"]) / n
            dead = sum(1 for t in ticks if t["deadband_dropped"]) / n
            rows[size]["sat"].append(sat)
            rows[size]["clamp"].append(clamp)
            rows[size]["cold"].append(cold)
            rows[size]["kvfreeze"].append(kvf)
            rows[size]["deadband"].append(dead)
    sizes = sorted(rows.keys())
    flags = ["sat", "clamp", "cold", "kvfreeze", "deadband"]
    flag_labels = {
        "sat": "tpot_saturated",
        "clamp": "tpot_clamp_active",
        "cold": "cold_start",
        "kvfreeze": "kv_freeze",
        "deadband": "deadband_dropped",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sizes))
    w = 0.16
    for i, fl in enumerate(flags):
        means = [
            np.mean(rows[s][fl]) if rows[s][fl] else 0.0 for s in sizes
        ]
        stds = [np.std(rows[s][fl]) if rows[s][fl] else 0.0 for s in sizes]
        ax.bar(x + (i - 2) * w, means, w, yerr=stds, capsize=2,
               label=flag_labels[fl])
    ax.set_xticks(x)
    ax.set_xticklabels([f"chunk={s}" for s in sizes])
    ax.set_ylabel("share of update ticks")
    ax.set_title("PID branch-flag activation share (mean ± std over seeds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "saturation_share.png", dpi=130)
    plt.close(fig)


# ------------------- Plot 3: chunk budget utilization -------------------


def plot_chunk_budget_utilization():
    fig, axes = plt.subplots(1, len(CHUNK_CONFIGS), figsize=(12, 4),
                             sharey=True)
    for ax, (config, size) in zip(axes, CHUNK_CONFIGS):
        utils: list[float] = []
        for seed in SEEDS:
            p = _seed_trace_files(config, seed, "chunk")
            recs = _load_jsonl(p) if p else []
            for r in recs:
                if r["chunk_budget_in"] > 0:
                    utils.append(r["chunk_budget_used"] /
                                 r["chunk_budget_in"])
        if not utils:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.hist(utils, bins=20, range=(0, 1.0), alpha=0.7)
            ax.axvline(np.median(utils), color="red", linestyle="--",
                       label=f"median={np.median(utils):.2f}")
            ax.legend(fontsize=8)
        ax.set_title(f"chunk={size} (n={len(utils)} P-iters)")
        ax.set_xlabel("chunk_budget_used / chunk_budget_in")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("chunk budget utilization per P-iter")
    fig.tight_layout()
    fig.savefig(OUTDIR / "chunk_budget_utilization.png", dpi=130)
    plt.close(fig)


# ------------------- Plot 4: truncated share -------------------


def plot_truncated_share():
    rows: dict[int, list[float]] = defaultdict(list)
    abs_rows: dict[int, list[int]] = defaultdict(list)
    for config, size in CHUNK_CONFIGS:
        for seed in SEEDS:
            p = _seed_trace_files(config, seed, "chunk")
            recs = _load_jsonl(p) if p else []
            if not recs:
                continue
            n = len(recs)
            tr = sum(1 for r in recs if r["truncated_count"] > 0)
            rows[size].append(tr / n)
            abs_rows[size].append(tr)
    sizes = sorted(rows.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    means = [np.mean(rows[s]) if rows[s] else 0.0 for s in sizes]
    stds = [np.std(rows[s]) if rows[s] else 0.0 for s in sizes]
    ax1.bar([f"chunk={s}" for s in sizes], means, yerr=stds, capsize=4,
            color="steelblue")
    ax1.set_ylabel("share of P-iters with ≥1 truncated request")
    ax1.set_title("truncation share (mean ± std over seeds)")
    ax1.grid(alpha=0.3, axis="y")

    abs_means = [
        np.mean(abs_rows[s]) if abs_rows[s] else 0.0 for s in sizes
    ]
    ax2.bar([f"chunk={s}" for s in sizes], abs_means, color="indianred")
    ax2.set_ylabel("absolute count of P-iters with truncation")
    ax2.set_title("truncation count (mean over seeds)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "truncated_share.png", dpi=130)
    plt.close(fig)


# ------------------- Plot 5: err vs ratio movement -------------------


def plot_err_vs_ratio():
    fig, axes = plt.subplots(1, len(CHUNK_CONFIGS), figsize=(13, 4.2))
    for ax, (config, size) in zip(axes, CHUNK_CONFIGS):
        all_dr: list[float] = []
        all_err_diff: list[float] = []
        sat: list[bool] = []
        for seed in SEEDS:
            p = _seed_trace_files(config, seed, "ctrl")
            ticks = _load_jsonl(p) if p else []
            for t in ticks:
                if t["cold_start"] or t["kv_freeze_triggered"]:
                    continue
                all_dr.append(t["ratio_after"] - t["ratio_before"])
                all_err_diff.append(
                    t["err_ttft"] - t["err_tpot_effective"])
                sat.append(t["tpot_saturated"])
        if all_dr:
            colors = ["red" if s else "steelblue" for s in sat]
            ax.scatter(all_err_diff, all_dr, c=colors, alpha=0.5, s=12)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("err_ttft - err_tpot_effective")
        ax.set_ylabel("Δ ratio per tick")
        ax.set_title(f"chunk={size}  (red=saturated, blue=normal)")
        ax.grid(alpha=0.3)
    fig.suptitle("err signal → ratio movement, gradient-step ticks only")
    fig.tight_layout()
    fig.savefig(OUTDIR / "err_vs_ratio.png", dpi=130)
    plt.close(fig)


# ------------------- summary table -------------------


def print_summary_table():
    print("\n=== P1 diagnostic summary (per chunk size, mean over seeds) ===")
    print(f"{'chunk':>7} {'n_ticks':>8} {'sat%':>6} {'clamp%':>7} "
          f"{'cold%':>6} {'kvf%':>6} {'dead%':>6} "
          f"{'P-iters':>8} {'trunc%':>7} {'budget%':>8}")
    for config, size in CHUNK_CONFIGS:
        n_ticks_l: list[int] = []
        sat_l: list[float] = []
        clamp_l: list[float] = []
        cold_l: list[float] = []
        kvf_l: list[float] = []
        dead_l: list[float] = []
        n_pit_l: list[int] = []
        trunc_l: list[float] = []
        util_l: list[float] = []
        for seed in SEEDS:
            ctrl = _load_jsonl(_seed_trace_files(config, seed, "ctrl")
                               or Path("/dev/null"))
            chunk = _load_jsonl(_seed_trace_files(config, seed, "chunk")
                                or Path("/dev/null"))
            if ctrl:
                n = len(ctrl)
                n_ticks_l.append(n)
                sat_l.append(
                    sum(1 for t in ctrl if t["tpot_saturated"]) / n)
                clamp_l.append(
                    sum(1 for t in ctrl if t["tpot_clamp_active"]) / n)
                cold_l.append(
                    sum(1 for t in ctrl if t["cold_start"]) / n)
                kvf_l.append(
                    sum(1 for t in ctrl if t["kv_freeze_triggered"]) / n)
                dead_l.append(
                    sum(1 for t in ctrl if t["deadband_dropped"]) / n)
            if chunk:
                m = len(chunk)
                n_pit_l.append(m)
                trunc_l.append(
                    sum(1 for r in chunk
                        if r["truncated_count"] > 0) / m)
                util_l.append(
                    np.mean([
                        r["chunk_budget_used"] / r["chunk_budget_in"]
                        for r in chunk if r["chunk_budget_in"] > 0
                    ]))
        if not n_ticks_l:
            print(f"{size:>7} (no data)")
            continue
        print(f"{size:>7} {int(np.mean(n_ticks_l)):>8} "
              f"{np.mean(sat_l) * 100:>5.1f} {np.mean(clamp_l) * 100:>6.1f} "
              f"{np.mean(cold_l) * 100:>5.1f} {np.mean(kvf_l) * 100:>5.1f} "
              f"{np.mean(dead_l) * 100:>5.1f} "
              f"{int(np.mean(n_pit_l)) if n_pit_l else 0:>8} "
              f"{(np.mean(trunc_l) * 100) if trunc_l else 0:>6.1f} "
              f"{(np.mean(util_l) * 100) if util_l else 0:>7.1f}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print_summary_table()
    plot_ratio_trajectory()
    plot_saturation_share()
    plot_chunk_budget_utilization()
    plot_truncated_share()
    plot_err_vs_ratio()
    print(f"\nplots → {OUTDIR}")


if __name__ == "__main__":
    main()
