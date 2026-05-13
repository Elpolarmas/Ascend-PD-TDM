"""Posthoc for P1.7 leading indicator (TTFT urgency) probe.

results/azure_p17/{conv,code}_w1_seed{0,1,2}/ — 2 windows × 5 configs × 3 seeds × 60s.

Done criterion (current_task.md §6 P1.7):
1. Sanity:Δ(M3.2 - M3.1) meet_slo% ≥ -1pp cross-seed (基线立住)
2. 真实考验:code@570 严档 tpot100/150 是否 Δ > 0(撬动 P1.5 的 FAIL)

Secondary:
- Δ(M3.2 - M1+chunk) — 是否把 P1.5 的 mean ≈ 0 撬正
- urgency 触发率(诊断:M3.2 真的被调用了多少次)
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p17")
TRACES = ["conv", "code"]
W_IDX = 1  # 只有 anchor 窗位
WINDOW_OFFSETS = {"conv": 1860, "code": 570}
WINDOW_QPS = {"conv": 8.43, "code": 12.03}
CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048",
           "c2_tdm_m31_2048", "c2_tdm_m32_2048", "c3_cp"]
SHORT = {"c1_baseline": "C1",
         "c2_tdm_m1_chunk2048": "M1+chunk",
         "c2_tdm_m31_2048": "M3.1",
         "c2_tdm_m32_2048": "M3.2",
         "c3_cp": "C3"}
SEEDS = [0, 1, 2]

TIERS = [
    ("ttft500_tpot100", 500, 100),
    ("ttft500_tpot150", 500, 150),
    ("ttft500_tpot200", 500, 200),
    ("ttft1000_tpot200", 1000, 200),
]


def meet(joined, w, we, t_lim, p_lim):
    in_win = [j for j in joined if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n = len(in_win)
    nm = sum(1 for j in in_win
             if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
             and j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim
             and j.get("output_tokens"))
    return nm, n


def load(trace, seed, cfg):
    p = ROOT / f"{trace}_w{W_IDX}_seed{seed}" / f"{cfg}_qps0.0.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def meet_pct(trace, cfg, t_lim, p_lim):
    out = []
    for s in SEEDS:
        d = load(trace, s, cfg)
        if d is None:
            out.append(float("nan"))
            continue
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def fmt(vals):
    clean = [v for v in vals if v == v]
    if not clean:
        return "  --  "
    m = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def paired_delta(trace, a, b, t_lim, p_lim):
    A = meet_pct(trace, a, t_lim, p_lim)
    B = meet_pct(trace, b, t_lim, p_lim)
    return [x - y for x, y in zip(A, B) if x == x and y == y]


def urgency_stats(trace):
    """Count urgency_ttft fires per seed in the M3.2 iter trace."""
    rows = []
    for s in SEEDS:
        f = ROOT / f"{trace}_w{W_IDX}_seed{s}" / "tdm_trace" / \
            "qps_sweep_c2_tdm_m32_2048_iter.jsonl"
        if not f.exists():
            rows.append((s, 0, 0))
            continue
        total = 0
        urg = 0
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                total += 1
                if r.get("source") == "urgency_ttft":
                    urg += 1
        rows.append((s, urg, total))
    return rows


def main():
    print("=" * 110)
    print(f"P1.7 leading-indicator (TTFT urgency) probe")
    print(f"  2 windows × 5 configs × 3 seeds × 60s   (anchor windows only)")
    for tr in TRACES:
        print(f"  {tr}@{WINDOW_OFFSETS[tr]}s ({WINDOW_QPS[tr]:.1f} qps)")
    print("=" * 110)

    # --- 1. urgency fire rate (sanity: did the new path actually engage?) ---
    print("\n## Urgency fire rate (M3.2 c2_tdm_m32_2048)\n")
    print(f"{'window':<14}{'seed':>6}{'urgency':>10}{'total':>10}{'pct':>8}")
    print("-" * 48)
    for tr in TRACES:
        for s, urg, total in urgency_stats(tr):
            pct = 100.0 * urg / max(1, total)
            print(f"{tr}@w{W_IDX:<8}{s:>6}{urg:>10}{total:>10}{pct:>7.2f}%")

    # --- 2. meet% table per (trace, config, tier) ---
    for tr in TRACES:
        print(f"\n## {tr}@{WINDOW_OFFSETS[tr]}s  meet_slo% (mean±std over {len(SEEDS)} seeds)\n")
        header = f"{'tier':<22}" + "".join(f"{SHORT[c]:>14}" for c in CONFIGS)
        print(header)
        print("-" * len(header))
        for tier_name, t_lim, p_lim in TIERS:
            row = f"{tier_name:<22}"
            for cfg in CONFIGS:
                pcts = meet_pct(tr, cfg, t_lim, p_lim)
                row += f"{fmt(pcts):>14}"
            print(row)

    # --- 3. Headline Δ(M3.2 - M3.1) paired-by-seed ---
    print("\n" + "=" * 110)
    print("Headline Δ = M3.2 - M3.1  (paired-by-seed, pp)")
    print("Done sanity: Δ ≥ -1pp (基线立住)")
    print("Real test (code 严档): Δ > 0 撬动 P1.5 的 FAIL")
    print("=" * 110)
    print(f"\n{'trace':<10}{'tier':<22}{'per-seed Δ':<30}{'mean':>10}{'min':>10}{'verdict':>14}")
    print("-" * 96)
    for tr in TRACES:
        for tier_name, t_lim, p_lim in TIERS:
            ds = paired_delta(tr, "c2_tdm_m32_2048", "c2_tdm_m31_2048",
                              t_lim, p_lim)
            if not ds:
                continue
            mean = statistics.mean(ds)
            mn = min(ds)
            per = ", ".join(f"{d:+5.1f}" for d in ds)
            # sanity verdict per pair; real test is tagged where applicable
            is_strict_code = (tr == "code" and tier_name in
                              ("ttft500_tpot100", "ttft500_tpot150"))
            if is_strict_code:
                v = "REAL_PASS" if mean > 0 else "REAL_FAIL"
            else:
                v = "SANITY_PASS" if mn >= -1.0 else "SANITY_FAIL"
            print(f"{tr:<10}{tier_name:<22}{per:<30}{mean:>+10.1f}{mn:>+10.1f}{v:>14}")

    # --- 4. Secondary: Δ(M3.2 - M1+chunk) (P1.5 main gap) ---
    print("\n" + "=" * 110)
    print("Secondary Δ = M3.2 - M1+chunk  (the P1.5 headline pair)")
    print("Was P1.5 mean ≈ 0 撬正?")
    print("=" * 110)
    print(f"\n{'trace':<10}{'tier':<22}{'per-seed Δ':<30}{'mean':>10}{'min':>10}")
    print("-" * 82)
    for tr in TRACES:
        for tier_name, t_lim, p_lim in TIERS:
            ds = paired_delta(tr, "c2_tdm_m32_2048", "c2_tdm_m1_chunk2048",
                              t_lim, p_lim)
            if not ds:
                continue
            mean = statistics.mean(ds)
            mn = min(ds)
            per = ", ".join(f"{d:+5.1f}" for d in ds)
            print(f"{tr:<10}{tier_name:<22}{per:<30}{mean:>+10.1f}{mn:>+10.1f}")

    # --- 5. Reference: Δ(M3.2 - C3) (thesis 主对手) ---
    print("\n" + "=" * 110)
    print("Reference Δ = M3.2 - C3  (thesis 主对手,看 urgency 加上后 vs C3 怎样)")
    print("=" * 110)
    print(f"\n{'trace':<10}{'tier':<22}{'per-seed Δ':<30}{'mean':>10}{'min':>10}")
    print("-" * 82)
    for tr in TRACES:
        for tier_name, t_lim, p_lim in TIERS:
            ds = paired_delta(tr, "c2_tdm_m32_2048", "c3_cp", t_lim, p_lim)
            if not ds:
                continue
            mean = statistics.mean(ds)
            mn = min(ds)
            per = ", ".join(f"{d:+5.1f}" for d in ds)
            print(f"{tr:<10}{tier_name:<22}{per:<30}{mean:>+10.1f}{mn:>+10.1f}")


if __name__ == "__main__":
    main()
