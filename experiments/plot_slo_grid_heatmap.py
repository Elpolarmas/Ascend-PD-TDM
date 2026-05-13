"""SLO grid heatmap：4 config × 3 qps × (ttft × tpot) → meet_slo% 三色热图。

读 long_prompt_sweep 的 12 个 JSON，post-hoc 在 SLO grid 上重算 meet_slo%（与
posthoc_slo_grid.py 同款判定），画 4×3 子图：行=config, 列=qps, 每张是 ttft×tpot
heatmap。RdYlGn colormap，区分带（中间值）醒目。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = Path("/vllm-workspace/Ascend-PD-TDM/results/long_prompt_sweep")

CONFIGS = ["c1_baseline", "c2_tdm", "c2_tdm_m27", "c3_cp"]
LABELS = {
    "c1_baseline": "C1 hybrid",
    "c2_tdm":      "C2 TDM (M1, ratio=0.30)",
    "c2_tdm_m27":  "M2.7 TDM SLO-PID",
    "c3_cp":       "C3 chunked prefill",
}
QPS_POINTS = [8.0, 16.0, 32.0]

TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load_joined(config: str, qps: float):
    path = SWEEP_DIR / f"{config}_qps{qps}.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    return d["joined"], s["warmup_s"], s["warmup_s"] + s["duration_s"]


def grid(joined, w, we) -> np.ndarray:
    """[len(TPOT_GRID), len(TTFT_GRID)] meet_slo% 矩阵。"""
    in_win = [j for j in joined
              if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n_total = len(in_win)
    out = np.zeros((len(TPOT_GRID), len(TTFT_GRID)), dtype=float)
    if n_total == 0:
        return out
    for i, tp in enumerate(TPOT_GRID):
        for k, tt in enumerate(TTFT_GRID):
            n = sum(1 for j in in_win
                    if j.get("ttft_ms") is not None
                    and j.get("tpot_ms_mean") is not None
                    and j["ttft_ms"] < tt and j["tpot_ms_mean"] < tp
                    and j.get("output_tokens"))
            out[i, k] = 100.0 * n / n_total
    return out


def main():
    fig, axes = plt.subplots(len(CONFIGS), len(QPS_POINTS),
                             figsize=(13.5, 14), sharex=True, sharey=True)
    cmap = plt.get_cmap("RdYlGn")
    last_im = None
    for r, cfg in enumerate(CONFIGS):
        for c, qps in enumerate(QPS_POINTS):
            joined, w, we = load_joined(cfg, qps)
            mat = grid(joined, w, we)
            ax = axes[r, c]
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100,
                           aspect="auto", origin="lower")
            last_im = im
            # 在每格标注百分比
            for i in range(mat.shape[0]):
                for k in range(mat.shape[1]):
                    val = mat[i, k]
                    color = "black" if 25 < val < 80 else "white"
                    ax.text(k, i, f"{val:.0f}", ha="center", va="center",
                            color=color, fontsize=8)
            if r == 0:
                ax.set_title(f"qps={int(qps)}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{LABELS[cfg]}\ntpot SLO (ms)", fontsize=9)
            if r == len(CONFIGS) - 1:
                ax.set_xlabel("ttft SLO (ms)", fontsize=9)
            ax.set_xticks(range(len(TTFT_GRID)))
            ax.set_xticklabels([str(t) for t in TTFT_GRID], fontsize=8)
            ax.set_yticks(range(len(TPOT_GRID)))
            ax.set_yticklabels([str(t) for t in TPOT_GRID], fontsize=8)
    fig.suptitle(
        "Long-prompt regime: meet_slo% under workload-conditional SLO grid\n"
        "(post-hoc on long_prompt_sweep; "
        "discrimination band: tpot in [100,200] x ttft in [500,1500])",
        fontsize=13, y=0.995,
    )
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, pad=0.02,
                        label="meet_slo (%)")
    out = SWEEP_DIR / "slo_grid_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
