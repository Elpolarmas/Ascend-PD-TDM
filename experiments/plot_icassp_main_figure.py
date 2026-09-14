"""Regenerate the only figure used by the ICASSP 2027 manuscript.

Input: results/phase_2_post/m31fix_phase1_pointwise.json
Output: paper/figures/fig_t6_strict_slo.pdf
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "phase_2_post" / "m31fix_phase1_pointwise.json"
OUTPUT = ROOT / "paper" / "figures" / "fig_t6_strict_slo.pdf"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "paper" / "build" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SERIES = (
    ("vanilla_cb", "CP (B=8192; equal-B ref.)", "#7A7A7A", "o"),
    ("c3_cp", "CP (B=2048)", "#B45F06", "s"),
    ("c2_tdm_m31_2048_fix", "PD-TDM (B=8192)", "#1261A0", "D"),
)
WORKLOADS = (
    ("conv", "Conversation: strict joint SLO\nTTFT < 200 ms, TPOT < 120 ms"),
    ("code", "Code: strict joint SLO\nTTFT < 500 ms, TPOT < 200 ms"),
)


def main() -> None:
    data = json.loads(DATA.read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 3.95), sharex="col")
    for column, (workload, title) in enumerate(WORKLOADS):
        for config, label, color, marker in SERIES:
            rows = sorted(data[workload]["s1"][config], key=lambda row: row["k"])
            x = [row["arrival_qps_median"] for row in rows]
            style = dict(label=label, color=color, marker=marker)
            axes[0, column].plot(
                x, [100 * row["meet_frac_median"] for row in rows], **style
            )
            axes[1, column].plot(
                x, [row["goodput_median"] for row in rows], **style
            )
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(0, 105)
        axes[1, column].set_xlabel("Measured arrival rate (request/s)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25, linewidth=0.6)
            axes[row, column].text(
                0.01, 1.03, f"({chr(97 + 2 * row + column)})",
                transform=axes[row, column].transAxes, fontweight="bold",
                ha="left"
            )
    axes[0, 0].set_ylabel("Joint-SLO attainment (%)")
    axes[1, 0].set_ylabel("SLO goodput (output token/s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
