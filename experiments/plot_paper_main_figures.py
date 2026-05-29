"""Paper 主图组:F1a-f (Pareto frontiers) + F2 (PD-TDM vs Sarathi heatmap) +
F2b (PD-TDM vs Vanilla CB heatmap, 2026-05-26 新加,NPU 复现 Sarathi 论文核心
finding + PD-TDM 在 chunked prefill 框架内额外增益的视觉证据).

数据源:results/phase_2_post/m31fix_phase1_pointwise.json
输出:figures/paper_F1{a-f}_*_pareto.png + paper_F2_advantage_heatmap.png
      + paper_F2b_advantage_vs_vanilla_heatmap.png

2026-05-26 baseline 重构:c1_baseline 移除 → vanilla_cb 取代(老 c3 chunk=8192,
Azure trace prompt cap=7000 < 8192,chunk 实际不触发 = 文献意义 vanilla CB)。
3-way 主线 = Vanilla CB / Sarathi (c3-fair @ chunk=2048) / PD-TDM (m31-fix);
4-way 加 c4_pd 作 disagg reference。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vllm-workspace/Ascend-PD-TDM")
DATA = ROOT / "results/phase_2_post/m31fix_phase1_pointwise.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

PARADIGMS_4WAY = [
    ("vanilla_cb", "Vanilla CB (chunk=8192, no effective chunk)", "tab:gray", "o"),
    ("c3_cp", "Sarathi chunked prefill (chunk=2048)", "tab:orange", "s"),
    ("c4_pd", "1P1D disagg (reference)", "tab:purple", "^"),
    ("c2_tdm_m31_2048_fix", "PD-TDM (ours)", "tab:red", "D"),
]
PARADIGMS = PARADIGMS_4WAY  # mutable at runtime via --variant
WORKLOADS = ["conv", "code"]
SLOS = ["s1", "s2", "s3"]
SLO_LABELS = {
    "conv": {"s1": "s1: 200/120 ms (strict)",
             "s2": "s2: 300/150 ms (mid)",
             "s3": "s3: 500/200 ms (loose)"},
    "code": {"s1": "s1: 500/200 ms (strict)",
             "s2": "s2: 500/700 ms (mid)",
             "s3": "s3: 2000/400 ms (loose)"},
}


def load():
    return json.loads(DATA.read_text())


def cells_sorted(rows):
    return sorted(rows, key=lambda r: r["k"])


# ============================================================
# F1a / F1b — Pareto frontier
# ============================================================
def plot_pareto(metric: str, ylabel: str, title_suffix: str, outpath: Path):
    d = load()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for i, wl in enumerate(WORKLOADS):
        for j, slo in enumerate(SLOS):
            ax = axes[i, j]
            for cfg, label, color, marker in PARADIGMS:
                rows = cells_sorted(d[wl][slo][cfg])
                xs = [r["arrival_qps_median"] for r in rows]
                ys = [r[metric] for r in rows]
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=1.8, markersize=6, alpha=0.9)
            ax.set_title(f"{wl}  |  {SLO_LABELS[wl][slo]}", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("arrival QPS (median over 3 seeds)")
    axes[0, 0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Pareto frontier: paradigm comparison ({title_suffix})",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


# ============================================================
# F2 — PD-TDM vs Sarathi (c3-fair) advantage heatmap
# (Δ meet_frac pp,展示 PD-TDM 在 chunked prefill 框架内对 Sarathi 的额外增益)
# ============================================================
def plot_advantage_heatmap(outpath: Path):
    _plot_pairwise_heatmap(
        outpath=outpath,
        baseline_cfg="c3_cp",
        baseline_label="Sarathi (c3-fair)",
        target_cfg="c2_tdm_m31_2048_fix",
        target_label="PD-TDM",
        suptitle="Paradigm advantage: PD-TDM vs Sarathi chunked prefill (Δ SLO-meet fraction)",
    )


# ============================================================
# F2b — PD-TDM vs Vanilla CB advantage heatmap (2026-05-26)
# 展示 PD-TDM 相对 Vanilla CB 的总 Δ:含 NPU 复现 Sarathi finding 的部分
# (chunked prefill 把长 prompt 灾难救活)+ phase-pure 额外增益
# ============================================================
def plot_advantage_heatmap_vs_vanilla(outpath: Path):
    _plot_pairwise_heatmap(
        outpath=outpath,
        baseline_cfg="vanilla_cb",
        baseline_label="Vanilla CB",
        target_cfg="c2_tdm_m31_2048_fix",
        target_label="PD-TDM",
        suptitle="Paradigm advantage: PD-TDM vs Vanilla CB (Δ SLO-meet fraction; "
                 "includes NPU reproduction of Sarathi finding)",
    )


# ============================================================
# F2c (bonus) — Sarathi vs Vanilla CB advantage heatmap
# NPU 复现 Sarathi 论文 OSDI'24 核心 finding 的直接视觉证据,
# 同时暴露短 prompt 上的反例(conv strict 上 Sarathi 输 Vanilla CB)
# ============================================================
def plot_sarathi_vs_vanilla_heatmap(outpath: Path):
    _plot_pairwise_heatmap(
        outpath=outpath,
        baseline_cfg="vanilla_cb",
        baseline_label="Vanilla CB",
        target_cfg="c3_cp",
        target_label="Sarathi chunked prefill",
        suptitle="NPU reproduction of Sarathi finding: chunked prefill vs Vanilla CB "
                 "(Δ SLO-meet fraction; positive = Sarathi wins; conv short-prompt = "
                 "fresh reverse finding)",
    )


def _plot_pairwise_heatmap(*, outpath, baseline_cfg, baseline_label,
                           target_cfg, target_label, suptitle):
    """Generic helper: per-workload heatmap of (target − baseline) meet_frac, in pp."""
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Single shared vmax across both workloads for color comparability
    all_vals = []
    per_wl_mats = {}
    ks_per_wl = {}
    for wl in WORKLOADS:
        ks = sorted({r["k"] for r in d[wl]["s1"][baseline_cfg]})
        ks_per_wl[wl] = ks
        mat = np.zeros((len(SLOS), len(ks)))
        for sj, slo in enumerate(SLOS):
            base = {r["k"]: r["meet_frac_median"]
                    for r in d[wl][slo][baseline_cfg]}
            tgt = {r["k"]: r["meet_frac_median"]
                   for r in d[wl][slo][target_cfg]}
            for ki, k in enumerate(ks):
                b = base.get(k); t = tgt.get(k)
                if b is None or t is None:
                    mat[sj, ki] = np.nan
                else:
                    mat[sj, ki] = (t - b) * 100.0  # → pp
        per_wl_mats[wl] = mat
        all_vals.extend(mat[~np.isnan(mat)].tolist())
    vmax = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 1.0

    for i, wl in enumerate(WORKLOADS):
        ax = axes[i]
        mat = per_wl_mats[wl]
        ks = ks_per_wl[wl]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k:.1f}" for k in ks])
        ax.set_yticks(range(len(SLOS)))
        ax.set_yticklabels(SLOS)
        ax.set_xlabel("k (QPS scale)")
        ax.set_ylabel("SLO tier")
        ax.set_title(f"{wl}: Δ SLO-meet ({target_label} − {baseline_label}), pp")
        for sj in range(len(SLOS)):
            for ki in range(len(ks)):
                v = mat[sj, ki]
                if np.isnan(v):
                    txt = "n/a"
                else:
                    txt = f"{v:+.1f}"
                ax.text(ki, sj, txt, ha="center", va="center",
                        fontsize=8.5,
                        color="white" if (not np.isnan(v) and abs(v) > vmax * 0.55) else "black")
        plt.colorbar(im, ax=ax, label="Δ pp")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["4way", "3way"], default="4way",
                        help="4way includes c4_pd; 3way drops it (outputs _3way suffix)")
    args = parser.parse_args()

    suffix = ""
    if args.variant == "3way":
        PARADIGMS = [p for p in PARADIGMS_4WAY if p[0] != "c4_pd"]
        suffix = "_3way"

    plot_pareto("goodput_median",
                "goodput (SLO-meeting req/s)",
                "goodput @ SLO",
                OUT / f"paper_F1a_goodput_pareto{suffix}.png")
    plot_pareto("meet_frac_median",
                "SLO attainment (fraction)",
                "SLO attainment",
                OUT / f"paper_F1b_attainment_pareto{suffix}.png")
    plot_pareto("ttft_mean_median",
                "TTFT mean (ms)",
                "TTFT mean",
                OUT / f"paper_F1c_ttft_mean_pareto{suffix}.png")
    plot_pareto("ttft_p99_median",
                "TTFT p99 (ms)",
                "TTFT p99",
                OUT / f"paper_F1d_ttft_p99_pareto{suffix}.png")
    plot_pareto("tpot_mean_median",
                "TPOT mean (ms)",
                "TPOT mean",
                OUT / f"paper_F1e_tpot_mean_pareto{suffix}.png")
    plot_pareto("tpot_p99_median",
                "TPOT p99 (ms)",
                "TPOT p99",
                OUT / f"paper_F1f_tpot_p99_pareto{suffix}.png")
    plot_advantage_heatmap(OUT / f"paper_F2_advantage_heatmap{suffix}.png")
    plot_advantage_heatmap_vs_vanilla(
        OUT / f"paper_F2b_advantage_vs_vanilla_heatmap{suffix}.png")
    plot_sarathi_vs_vanilla_heatmap(
        OUT / f"paper_F2c_sarathi_vs_vanilla_heatmap{suffix}.png")
