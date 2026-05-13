"""Posthoc analysis for P1.5 跨窗扩展实验.

results/azure_p15/{conv,code}_w{0,1,2}_seed{0,1,2}/ — 2 traces × 3 windows × 4 configs × 3 seeds × 60s.

Done 标准检查项(current_task.md §6 P1.5):
- M3.1 - M1+chunk Δ 跨窗 mean / std / min / max
- 1860s/570s 在新标准下的位次
- Δ mean > 0 且 min > -1pp → main result 立住
- 1860s/570s 不能是 outlier
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p15")
TRACES = ["conv", "code"]
WINDOWS = [0, 1, 2]  # w1 是 azure_main 锚点(conv:1860s, code:570s)
ANCHOR_W = 1  # w1 是锚点
WINDOW_OFFSETS = {
    "conv": [1650, 1860, 2130],
    "code": [180, 570, 840],
}
WINDOW_QPS = {
    "conv": [8.45, 8.43, 7.83],
    "code": [8.85, 12.03, 10.37],
}
CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
SHORT = {"c1_baseline": "C1",
         "c2_tdm_m1_chunk2048": "M1+chunk",
         "c2_tdm_m31_2048": "M3.1",
         "c3_cp": "C3"}
SEEDS = [0, 1, 2]

TIERS = [
    ("ttft500_tpot100", 500, 100),
    ("ttft500_tpot150", 500, 150),
    ("ttft500_tpot200", 500, 200),
    ("ttft1000_tpot200", 1000, 200),
]

# done 标准:Δ(M3.1, M1+chunk)
HEADLINE_PAIR = ("c2_tdm_m31_2048", "c2_tdm_m1_chunk2048")


def meet(joined, w, we, t_lim, p_lim):
    in_win = [j for j in joined if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n = len(in_win)
    nm = sum(1 for j in in_win
             if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
             and j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim
             and j.get("output_tokens"))
    return nm, n


def load(trace, w_idx, seed, cfg):
    p = ROOT / f"{trace}_w{w_idx}_seed{seed}" / f"{cfg}_qps0.0.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def meet_pct(trace, w_idx, cfg, t_lim, p_lim):
    """Return list of meet% across SEEDS for one (trace, window, config)."""
    out = []
    for s in SEEDS:
        d = load(trace, w_idx, s, cfg)
        if d is None:
            out.append(float("nan"))
            continue
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def fmt(vals):
    clean = [v for v in vals if v == v]  # NaN check
    if not clean:
        return "  --  "
    m = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def fmt_d(d):
    return f"{d:+5.1f}"


def paired_delta(trace, w_idx, a, b, t_lim, p_lim):
    """Per-seed paired Δ; returns list (one per seed)."""
    A = meet_pct(trace, w_idx, a, t_lim, p_lim)
    B = meet_pct(trace, w_idx, b, t_lim, p_lim)
    return [x - y for x, y in zip(A, B) if x == x and y == y]


def main():
    print("=" * 120)
    print("P1.5 跨窗扩展:2 traces × 3 windows × 4 configs × 3 seeds × 60s")
    print("窗位选择:60s 窗、30s 步进、QPS≥p75、贪心 ≥120s 间距、top-3")
    print("=" * 120)
    for tr in TRACES:
        offs = WINDOW_OFFSETS[tr]
        qps = WINDOW_QPS[tr]
        line = f"  {tr}: " + ", ".join(
            f"w{i}@{offs[i]}s({qps[i]:.1f}qps)" + (" [anchor=azure_main]" if i == ANCHOR_W else "")
            for i in WINDOWS
        )
        print(line)

    # ---- 1. meet% per (trace, window, config) ----
    for tr in TRACES:
        for tier_name, t_lim, p_lim in TIERS:
            print(f"\n## {tr} | tier={tier_name}  meet_slo% (mean±std over {len(SEEDS)} seeds)\n")
            header = f"{'window':<22}" + "".join(f"{SHORT[c]:>14}" for c in CONFIGS)
            print(header)
            print("-" * len(header))
            for w_idx in WINDOWS:
                tag = " (anchor)" if w_idx == ANCHOR_W else ""
                row = f"w{w_idx}@{WINDOW_OFFSETS[tr][w_idx]}s{tag:<10}"[:22].ljust(22)
                for cfg in CONFIGS:
                    pcts = meet_pct(tr, w_idx, cfg, t_lim, p_lim)
                    row += f"{fmt(pcts):>14}"
                print(row)

    # ---- 2. Headline Δ(M3.1 - M1+chunk) per window + cross-window stats ----
    a, b = HEADLINE_PAIR
    print("\n" + "=" * 120)
    print(f"Headline Δ = {SHORT[a]} - {SHORT[b]}  (paired-by-seed, pp)")
    print(f"Done criterion: cross-window mean > 0 AND min > -1pp; anchor 不能是 outlier")
    print("=" * 120)

    for tr in TRACES:
        print(f"\n## {tr}\n")
        header = f"{'tier':<22}" + "".join(
            f"{'w'+str(w)+'@'+str(WINDOW_OFFSETS[tr][w])+'s':>16}" for w in WINDOWS
        ) + f"{'mean':>10}{'std':>8}{'min':>8}{'max':>8}{'verdict':>14}"
        print(header)
        print("-" * len(header))
        for tier_name, t_lim, p_lim in TIERS:
            row = f"{tier_name:<22}"
            per_window_mean = []
            per_seed_all = []  # for cross-window stats: mean across all 9 (window,seed) pairs
            for w_idx in WINDOWS:
                ds = paired_delta(tr, w_idx, a, b, t_lim, p_lim)
                if ds:
                    m = statistics.mean(ds)
                    sd = statistics.stdev(ds) if len(ds) > 1 else 0.0
                    per_window_mean.append(m)
                    per_seed_all.extend(ds)
                    row += f"{m:+6.1f}±{sd:4.1f}".rjust(16)
                else:
                    row += "  --  ".rjust(16)
            if per_seed_all:
                cm = statistics.mean(per_seed_all)
                csd = statistics.stdev(per_seed_all) if len(per_seed_all) > 1 else 0.0
                cmn = min(per_window_mean)
                cmx = max(per_window_mean)
                ok = "PASS" if cm > 0 and cmn > -1.0 else "FAIL"
                anchor_m = per_window_mean[ANCHOR_W] if ANCHOR_W < len(per_window_mean) else float("nan")
                # anchor outlier 判定:超过其它窗口 mean 的 ±2σ?简化为 |anchor - other_mean| > 2σ
                others = [per_window_mean[i] for i in range(len(per_window_mean)) if i != ANCHOR_W]
                if others:
                    other_m = statistics.mean(others)
                    other_sd = statistics.stdev(others) if len(others) > 1 else 0.0
                    is_outlier = abs(anchor_m - other_m) > 2 * other_sd if other_sd > 1e-6 else abs(anchor_m - other_m) > 2.0
                    if is_outlier:
                        ok += "*"  # 标记 anchor 可能是 outlier
                row += f"{cm:+6.1f}".rjust(10) + f"{csd:6.1f}".rjust(8) + f"{cmn:+6.1f}".rjust(8) + f"{cmx:+6.1f}".rjust(8) + f"{ok:>14}"
            print(row)

    # ---- 3. M3.1 vs C3 跨窗(thesis 主对手) ----
    a2, b2 = "c2_tdm_m31_2048", "c3_cp"
    print("\n" + "=" * 120)
    print(f"Secondary Δ = {SHORT[a2]} - {SHORT[b2]} (paired-by-seed, pp)")
    print("(thesis: M3.1 用每字延迟换首字延迟 — 看每个窗口的输/赢面)")
    print("=" * 120)
    for tr in TRACES:
        print(f"\n## {tr}\n")
        header = f"{'tier':<22}" + "".join(
            f"{'w'+str(w)+'@'+str(WINDOW_OFFSETS[tr][w])+'s':>16}" for w in WINDOWS
        ) + f"{'mean':>10}{'min':>8}{'max':>8}"
        print(header)
        print("-" * len(header))
        for tier_name, t_lim, p_lim in TIERS:
            row = f"{tier_name:<22}"
            per_window_mean = []
            for w_idx in WINDOWS:
                ds = paired_delta(tr, w_idx, a2, b2, t_lim, p_lim)
                if ds:
                    m = statistics.mean(ds)
                    sd = statistics.stdev(ds) if len(ds) > 1 else 0.0
                    per_window_mean.append(m)
                    row += f"{m:+6.1f}±{sd:4.1f}".rjust(16)
                else:
                    row += "  --  ".rjust(16)
            if per_window_mean:
                row += f"{statistics.mean(per_window_mean):+6.1f}".rjust(10)
                row += f"{min(per_window_mean):+6.1f}".rjust(8)
                row += f"{max(per_window_mean):+6.1f}".rjust(8)
            print(row)


if __name__ == "__main__":
    main()
