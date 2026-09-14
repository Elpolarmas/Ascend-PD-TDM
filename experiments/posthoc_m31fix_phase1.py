"""Phase 1 m31-fix sweep post-hoc (2026-05-25,2026-05-26 加 vanilla_cb)。

跟 posthoc_phase_2_t6_fair.py 同结构,但 m31 从 results/m31fix_validate/ 读
(chunked_schedule waiting-loop bug fixed 后重跑的全 matrix)。
c1 / c3-fair 复用原目录。额外加 m31-bug 列做 before/after 对比。

2026-05-26 update:加 vanilla_cb config(老 c3 chunk=8192)从 nonpid 目录读
— Azure trace prompt cap=7000 < 8192,chunk 实际不触发 = mixed batch +
长 prompt 整 iter 一次跑完,文献意义的 vanilla continuous batching baseline。

目录命名:
    m31-fix:    results/m31fix_validate/t6_{wl}_{k}_{slo}_s{seed}_m31/c2_tdm_m31_2048_qps0.0.json
    m31-bug:    results/phase_2_t6_burst_goodput/{wl}_{k}_{slo}_seed{seed}_pid/c2_tdm_m31_2048_qps0.0.json
    c3-fair:    results/c3_chunk2048_supplement/{wl}_{k}_seed{seed}/c3_cp_qps0.0.json
    c1:         results/phase_2_t6_burst_goodput/{wl}_{k}_seed{seed}_nonpid/c1_baseline_qps0.0.json
    vanilla_cb: results/phase_2_t6_burst_goodput/{wl}_{k}_seed{seed}_nonpid/c3_cp_qps0.0.json
                (老 c3 chunk=8192 文件名仍是 c3_cp,逻辑名 vanilla_cb 区分 chunk=2048 的 c3-fair)

Matrix: 2 wl × 7 k × 3 SLO × 3 seed = 126(s4 已去)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T6_DIR = ROOT / "results/phase_2_t6_burst_goodput"
C3_FAIR_DIR = ROOT / "results/c3_chunk2048_supplement"
M31_FIX_DIR = ROOT / "results/m31fix_validate"
C4_PD_DIR = ROOT / "results/c4_pd_supplement"
OUT_DIR = ROOT / "results/phase_2_post"

WORKLOADS = ["conv", "code"]
SLO_TIERS = ["s1", "s2", "s3"]
SEEDS = [0, 1, 2]
K_GRID = {
    "conv": [0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
    "code": [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9],
}
CONFIG_C1 = "c1_baseline"
CONFIG_C3 = "c3_cp"
CONFIG_C4 = "c4_pd"
CONFIG_VCB = "vanilla_cb"  # 老 c3 chunk=8192 = vanilla CB equivalent (2026-05-26)
CONFIG_M31_FIX = "c2_tdm_m31_2048_fix"
CONFIG_M31_BUG = "c2_tdm_m31_2048_bug"
ALL_CONFIGS = [CONFIG_C1, CONFIG_VCB, CONFIG_C3, CONFIG_C4, CONFIG_M31_BUG, CONFIG_M31_FIX]

SLO_GRID = {
    "conv": {"s1": (200.0, 120.0), "s2": (300.0, 150.0), "s3": (500.0, 200.0)},
    "code": {"s1": (500.0, 200.0), "s2": (500.0, 700.0), "s3": (2000.0, 400.0)},
}

WINDOW_START_S = 30.0
WINDOW_END_S = 90.0
WINDOW_DURATION_S = WINDOW_END_S - WINDOW_START_S
DECISIVE_PP = 5.0
TIE_PP = 1.0


def _pct(xs, q):
    if not xs: return None
    s = sorted(xs)
    k = q * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _recompute(joined, ttft_slo, tpot_slo):
    # Match the online metric definition: select requests by arrival time in
    # the steady-state window before filtering successful telemetry records.
    win = [r for r in joined
           if WINDOW_START_S <= r.get("arrival_time_s", -1) < WINDOW_END_S]
    ok = [r for r in win
          if r.get("matched") and r.get("status") == 200
          and r.get("ttft_ms") is not None and r.get("tpot_ms_mean") is not None]
    if not ok:
        return None
    n_meet = 0; gp = 0
    for r in ok:
        if r["ttft_ms"] < ttft_slo and r["tpot_ms_mean"] < tpot_slo:
            n_meet += 1
            gp += r.get("output_tokens", 0) or 0
    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms_mean"] for r in ok]
    return {
        # Count all arrivals in the denominator. Requests with client errors
        # or missing telemetry are therefore conservative non-attainments.
        "n_in_window": len(win),
        "n_matched": len(ok),
        "n_meet": n_meet,
        "meet_frac": n_meet / len(win),
        "ttft_mean": round(sum(ttfts)/len(ttfts), 2),
        "ttft_p99": round(_pct(ttfts, 0.99), 2),
        "tpot_mean": round(sum(tpots)/len(tpots), 2),
        "tpot_p99": round(_pct(tpots, 0.99), 2),
        "goodput_tok_s": round(gp / WINDOW_DURATION_S, 2),
    }


def _load_json(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError): return None


def _path_for(cfg, workload, slo, k, seed):
    if cfg == CONFIG_C1:
        return T6_DIR / f"{workload}_k{k}_seed{seed}_nonpid" / "c1_baseline_qps0.0.json"
    if cfg == CONFIG_VCB:
        # 老 c3 chunk=8192 文件名仍是 c3_cp_qps0.0.json,跟 c1 同 nonpid 目录里
        return T6_DIR / f"{workload}_k{k}_seed{seed}_nonpid" / "c3_cp_qps0.0.json"
    if cfg == CONFIG_C3:
        return C3_FAIR_DIR / f"{workload}_k{k}_seed{seed}" / "c3_cp_qps0.0.json"
    if cfg == CONFIG_C4:
        return C4_PD_DIR / f"{workload}_k{k}_seed{seed}" / "c4_pd_qps0.0.json"
    if cfg == CONFIG_M31_BUG:
        return T6_DIR / f"{workload}_k{k}_{slo}_seed{seed}_pid" / "c2_tdm_m31_2048_qps0.0.json"
    if cfg == CONFIG_M31_FIX:
        return M31_FIX_DIR / f"t6_{workload}_k{k}_{slo}_s{seed}_m31" / "c2_tdm_m31_2048_qps0.0.json"
    raise ValueError(cfg)


def _load_cell(cfg, workload, slo, k, seed):
    ttft_slo, tpot_slo = SLO_GRID[workload][slo]
    path = _path_for(cfg, workload, slo, k, seed)
    d = _load_json(path)
    if d is None:
        return {"_missing": str(path)}
    m = _recompute(d.get("joined", []), ttft_slo, tpot_slo)
    if m is None:
        return {"_missing": str(path)}
    summary = d.get("summary", {})
    m["arrival_qps"] = summary.get("qps_actual_arrival")
    return m


def _med(xs):
    xs2 = [x for x in xs if x is not None]
    if not xs2: return None
    return round(statistics.median(xs2), 3)


def aggregate_pointwise():
    out = {}
    missing = []
    for workload in WORKLOADS:
        out[workload] = {}
        for slo in SLO_TIERS:
            out[workload][slo] = {}
            for cfg in ALL_CONFIGS:
                out[workload][slo][cfg] = []
                for k in K_GRID[workload]:
                    cells = [_load_cell(cfg, workload, slo, k, s) for s in SEEDS]
                    for c in cells:
                        if "_missing" in c:
                            missing.append(c["_missing"])
                    valid = [c for c in cells if "_missing" not in c]
                    if not valid:
                        out[workload][slo][cfg].append({"k": k, "_no_data": True})
                        continue
                    out[workload][slo][cfg].append({
                        "k": k,
                        "arrival_qps_median": _med([c.get("arrival_qps") for c in valid]),
                        "meet_frac_median": _med([c.get("meet_frac") for c in valid]),
                        "goodput_median": _med([c.get("goodput_tok_s") for c in valid]),
                        "ttft_mean_median": _med([c.get("ttft_mean") for c in valid]),
                        "ttft_p99_median": _med([c.get("ttft_p99") for c in valid]),
                        "tpot_mean_median": _med([c.get("tpot_mean") for c in valid]),
                        "tpot_p99_median": _med([c.get("tpot_p99") for c in valid]),
                        "n_seeds": len(valid),
                    })
    return out, missing


SHORT = {
    CONFIG_C1: "c1",
    CONFIG_VCB: "vanilla-cb",
    CONFIG_C3: "c3-fair",
    CONFIG_C4: "c4-pd",
    CONFIG_M31_BUG: "m31-bug",
    CONFIG_M31_FIX: "m31-fix",
}


def print_winning_region(pw):
    """Per-paradigm winning region: Δmeet, Δgoodput vs c3-fair (m31-fix focus)."""
    print("\n" + "=" * 100)
    print("WINNING REGION — m31-fix vs c3-fair (Δmeet% , Δgoodput, 3-seed median)")
    print("=" * 100)
    win_total = {wl: {"win": 0, "tie": 0, "loss": 0} for wl in WORKLOADS}
    for workload in WORKLOADS:
        ks = K_GRID[workload]
        print(f"\n--- {workload.upper()} ---")
        hdr = f"{'SLO (ttft/tpot)':<18}" + "".join(f"{'k='+str(k):>14}" for k in ks)
        print(hdr)
        for slo in SLO_TIERS:
            tt, tp = SLO_GRID[workload][slo]
            slo_lbl = f"{slo} {int(tt)}/{int(tp)}"
            row = f"{slo_lbl:<18}"
            for i, k in enumerate(ks):
                fix = pw[workload][slo][CONFIG_M31_FIX][i]
                c3 = pw[workload][slo][CONFIG_C3][i]
                mf = fix.get("meet_frac_median")
                mc = c3.get("meet_frac_median")
                gf = fix.get("goodput_median")
                gc = c3.get("goodput_median")
                if mf is None or mc is None:
                    row += f"{'n/a':>14}"; continue
                d_meet = (mf - mc) * 100
                d_gp = (gf - gc) if (gf is not None and gc is not None) else None
                if d_meet >= DECISIVE_PP: tag = "WIN"; win_total[workload]["win"] += 1
                elif abs(d_meet) < TIE_PP: tag = "tie"; win_total[workload]["tie"] += 1
                elif d_meet <= -TIE_PP:    tag = "LOSS"; win_total[workload]["loss"] += 1
                else:                       tag = "+"; win_total[workload]["win"] += 1  # small win
                cell_str = f"{tag} {d_meet:+.1f}/{d_gp:+.0f}" if d_gp is not None else f"{tag} {d_meet:+.1f}/-"
                row += f"{cell_str:>14}"
            print(row)
    print("\nWIN-tally (vs c3-fair, decisive ≥+5pp / small-win ≥+1pp / tie / loss):")
    for wl in WORKLOADS:
        n = len(K_GRID[wl]) * len(SLO_TIERS)
        t = win_total[wl]
        loss = t["loss"]
        win = t["win"]
        tie = t["tie"]
        print(f"  {wl}: WIN/small-win {win}/{n}, TIE {tie}/{n}, LOSS {loss}/{n}")


def print_bug_vs_fix(pw):
    """Bug fix effect: Δmeet of (m31-fix vs m31-bug)."""
    print("\n" + "=" * 100)
    print("BUG-FIX EFFECT — Δmeet% (m31-fix - m31-bug), 3-seed median")
    print("=" * 100)
    for workload in WORKLOADS:
        ks = K_GRID[workload]
        print(f"\n--- {workload.upper()} ---")
        hdr = f"{'SLO':<14}" + "".join(f"{'k='+str(k):>10}" for k in ks)
        print(hdr)
        for slo in SLO_TIERS:
            tt, tp = SLO_GRID[workload][slo]
            slo_lbl = f"{slo} {int(tt)}/{int(tp)}"
            row = f"{slo_lbl:<14}"
            for i, k in enumerate(ks):
                fix = pw[workload][slo][CONFIG_M31_FIX][i]
                bug = pw[workload][slo][CONFIG_M31_BUG][i]
                mf = fix.get("meet_frac_median")
                mb = bug.get("meet_frac_median")
                if mf is None or mb is None:
                    row += f"{'-':>10}"; continue
                d = (mf - mb) * 100
                row += f"{d:>+9.1f}"
            print(row)


def print_detail_per_cell(pw):
    print("\n" + "=" * 110)
    print("详细数据 — per (workload, slo) × 4 cfg × {meet%, goodput, ttft m/p99, tpot m/p99} × 7 k")
    print("=" * 110)
    for workload in WORKLOADS:
        for slo in SLO_TIERS:
            tt, tp = SLO_GRID[workload][slo]
            ks = K_GRID[workload]
            print(f"\n=== {workload} {slo} (TTFT {int(tt)} / TPOT {int(tp)}) ===")
            hdr = f"{'metric':<22}" + "".join(f"{('k='+str(k)):>11}" for k in ks)
            print(hdr)
            arr_row = f"{'arrival_qps':<22}"
            for i, k in enumerate(ks):
                e = pw[workload][slo][CONFIG_M31_FIX][i]
                aq = e.get("arrival_qps_median")
                arr_row += f"{aq if aq is not None else '-':>11}"
            print(arr_row)
            print("-" * len(hdr))
            for cfg in ALL_CONFIGS:
                short = SHORT[cfg]
                for label, key, fmt in [
                    ("meet%", "meet_frac_median", lambda v: f"{round(v*100,1)}" if v is not None else "-"),
                    ("goodput", "goodput_median", lambda v: f"{v}" if v is not None else "-"),
                    ("ttft mean", "ttft_mean_median", lambda v: f"{v}" if v is not None else "-"),
                    ("ttft p99", "ttft_p99_median", lambda v: f"{v}" if v is not None else "-"),
                    ("tpot mean", "tpot_mean_median", lambda v: f"{v}" if v is not None else "-"),
                    ("tpot p99", "tpot_p99_median", lambda v: f"{v}" if v is not None else "-"),
                ]:
                    row = f"{short+' '+label:<22}"
                    for i, k in enumerate(ks):
                        e = pw[workload][slo][cfg][i]
                        v = e.get(key)
                        row += f"{fmt(v):>11}"
                    print(row)
                print()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[m31fix-phase1] 加载 + reclassify per-cell …")
    pw, missing = aggregate_pointwise()
    if missing:
        uniq = sorted(set(missing))
        print(f"  缺失 {len(uniq)} 路径:")
        for m in uniq[:15]:
            print(f"    {m}")
        if len(uniq) > 15:
            print(f"    ... ({len(uniq)-15} more)")
    out_path = OUT_DIR / "m31fix_phase1_pointwise.json"
    out_path.write_text(json.dumps(pw, indent=2, default=str))
    print(f"  wrote {out_path}")

    print_winning_region(pw)
    print_bug_vs_fix(pw)
    print_detail_per_cell(pw)


if __name__ == "__main__":
    main()
