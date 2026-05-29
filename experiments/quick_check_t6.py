#!/usr/bin/env python3
"""Quick-check T6 in-progress status + paradigm Δ on data landed so far.

Usage:
    python3 experiments/quick_check_t6.py
    python3 experiments/quick_check_t6.py --verbose   # 多打几行 per-cell 细节
    python3 experiments/quick_check_t6.py --kill      # 报告完后 kill T6 wrapper

输出:
1. 进度计数 (done / 252) + ETA
2. 当前正在跑的 sub-dir (从 /tmp/phase_2_t6.log 解析)
3. 已落地的 paradigm Δ 表 — 按 (workload, SLO_label, seed) pool,
   只对 nonpid+pid 都已落地的 cell 显示 m31 vs c1/c3 attainment%
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/phase_2_t6_burst_goodput")
LOG = Path("/tmp/phase_2_t6.log")

CONV_KS = [0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
CODE_KS = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9]
CONV_SLOS = {"s1": (200, 120), "s2": (300, 150), "s3": (500, 200), "s4": (1000, 200)}
CODE_SLOS = {"s1": (500, 200), "s2": (500, 700), "s3": (2000, 400), "s4": (3000, 1500)}
SEEDS = [0, 1, 2]

# Expected total: nonpid 84 + pid 168 = 252 sub-dirs (BUT nonpid has 2 cfgs per sub-dir,
# we count 42 nonpid invocations + 168 pid = 210 sub-dirs)
N_NONPID = len(SEEDS) * (len(CONV_KS) + len(CODE_KS))   # 42
N_PID = len(SEEDS) * (len(CONV_SLOS) + len(CODE_SLOS)) * len(CONV_KS)  # 168
N_TOTAL = N_NONPID + N_PID  # 210


def _load_summary(d: Path) -> dict | None:
    p = d / "qps_sweep_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_reqs(d: Path, cfg: str) -> list[tuple[float, float, int]]:
    """Return [(ttft_ms, tpot_mean_ms, output_tokens)] inside steady window (admission >= 30s)."""
    p = d / "tdm_trace" / f"qps_sweep_{cfg}_req.jsonl"
    if not p.exists():
        return []
    raw = []
    with p.open() as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except Exception:
                pass
    if not raw:
        return []
    ts0 = min(r["admission_ts_ms"] for r in raw)
    out = []
    for r in raw:
        rel = (r["admission_ts_ms"] - ts0) / 1000.0
        if 30.0 <= rel < 120.0:  # warmup=30, duration=90 ⇒ window [30, 120]
            out.append((r["ttft_ms"], r["tpot_ms_mean"], r["output_tokens"]))
    return out


def progress() -> tuple[int, int, list[Path]]:
    done = sorted([d for d in ROOT.glob("*/") if (d / "qps_sweep_summary.json").exists()])
    return len(done), N_TOTAL, done


def current_run() -> str:
    if not LOG.exists():
        return "(no log)"
    # Look for the most recent "=====" header line
    try:
        tail = subprocess.check_output(["tail", "-200", str(LOG)], text=True)
    except Exception:
        return "(log read failed)"
    cur = None
    for line in tail.splitlines():
        m = re.match(r"^===== (\S*/results/phase_2_t6_burst_goodput/\S+)", line)
        if m:
            cur = Path(m.group(1)).name
    return cur or "(unknown)"


def wrapper_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_phase_2_t6_burst_goodput.sh"], text=True
        )
        for pid in out.split():
            return int(pid)
    except Exception:
        return None
    return None


def reclassify(reqs: list, slo_ttft: float, slo_tpot: float) -> tuple[float, float]:
    """Return (attainment%, goodput_tok_per_s) for given SLO."""
    if not reqs:
        return 0.0, 0.0
    n_meet = sum(1 for t, p, _ in reqs if t < slo_ttft and p < slo_tpot)
    tok_meet = sum(o for t, p, o in reqs if t < slo_ttft and p < slo_tpot)
    # window 90s (duration after warmup)
    return n_meet / len(reqs) * 100.0, tok_meet / 90.0


def paradigm_delta_table(verbose: bool = False) -> None:
    """For each (workload, SLO, seed, k) where m31 pid run exists AND matching nonpid exists,
    compute c1 / c3 / m31 attainment% at that SLO."""
    cells = []
    for workload in ("conv", "code"):
        ks = CONV_KS if workload == "conv" else CODE_KS
        slos = CONV_SLOS if workload == "conv" else CODE_SLOS
        for seed in SEEDS:
            nonpid_dir = lambda k: ROOT / f"{workload}_k{k}_seed{seed}_nonpid"
            for slo_label, (s_ttft, s_tpot) in slos.items():
                for k in ks:
                    pid_d = ROOT / f"{workload}_k{k}_{slo_label}_seed{seed}_pid"
                    nonpid_d = nonpid_dir(k)
                    pid_sum = _load_summary(pid_d)
                    nonpid_sum = _load_summary(nonpid_d)
                    if pid_sum is None or nonpid_sum is None:
                        continue
                    # m31 attainment at native SLO (already in summary)
                    m31_runs = pid_sum["configs"].get("c2_tdm_m31_2048", [])
                    if not m31_runs:
                        continue
                    w = m31_runs[0]["window"]
                    m31_attain = (w["n_meet_slo"] / w["n_matched"] * 100.0
                                  if w["n_matched"] else 0.0)
                    m31_qps = m31_runs[0]["qps_actual_arrival"]
                    m31_goodput = w["goodput_tok_s"]
                    # c1 / c3 reclassify from nonpid per-req
                    c1_reqs = _load_reqs(nonpid_d, "c1_baseline")
                    c3_reqs = _load_reqs(nonpid_d, "c3_cp")
                    c1_attain, c1_gp = reclassify(c1_reqs, s_ttft, s_tpot)
                    c3_attain, c3_gp = reclassify(c3_reqs, s_ttft, s_tpot)
                    cells.append({
                        "workload": workload, "slo_label": slo_label,
                        "slo": (s_ttft, s_tpot), "seed": seed, "k": k,
                        "m31_qps": m31_qps,
                        "m31_attain": m31_attain, "c1_attain": c1_attain, "c3_attain": c3_attain,
                        "m31_goodput": m31_goodput, "c1_goodput": c1_gp, "c3_goodput": c3_gp,
                    })
    if not cells:
        print("\n[paradigm Δ] 还没有 (m31_pid + matching nonpid) 同时落地的 cell。")
        print("            等到 T+6.5h~7h(seed 0 nonpid 全完 + seed 0 conv s1 m31 第一档完)")
        return

    # Pool by (workload, slo_label) — mean over seeds × k
    pooled = defaultdict(lambda: {"m31": [], "c1": [], "c3": [],
                                  "m31_gp": [], "c1_gp": [], "c3_gp": [], "n": 0})
    for c in cells:
        key = (c["workload"], c["slo_label"], c["slo"])
        pooled[key]["m31"].append(c["m31_attain"])
        pooled[key]["c1"].append(c["c1_attain"])
        pooled[key]["c3"].append(c["c3_attain"])
        pooled[key]["m31_gp"].append(c["m31_goodput"])
        pooled[key]["c1_gp"].append(c["c1_goodput"])
        pooled[key]["c3_gp"].append(c["c3_goodput"])
        pooled[key]["n"] += 1

    print("\n[paradigm Δ] 已落地的 cell, attainment% pooled across (seed, k):\n")
    print(f"{'workload':<8} {'SLO':<10} {'(ttft,tpot)':<14} "
          f"{'n_cell':>6} {'c1':>6} {'c3':>6} {'m31':>6}  "
          f"{'Δvs_c1':>7} {'Δvs_c3':>7} {'Δvs_max':>8}")
    print("-" * 92)
    for (wl, lbl, slo), v in sorted(pooled.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        m31 = mean(v["m31"]); c1 = mean(v["c1"]); c3 = mean(v["c3"])
        d_c1 = m31 - c1; d_c3 = m31 - c3; d_max = m31 - max(c1, c3)
        mark = "★" if d_max > 5 else ("·" if d_max > 2 else " ")
        print(f"{wl:<8} {lbl:<10} {str(slo):<14} {v['n']:>6}  "
              f"{c1:>5.1f}% {c3:>5.1f}% {m31:>5.1f}% "
              f"{d_c1:>+6.1f} {d_c3:>+6.1f} {d_max:>+7.1f}{mark}")

    # Goodput tok/s summary (用 mean across cells,简化版)
    print("\n[goodput tok/s] mean across (seed, k):\n")
    print(f"{'workload':<8} {'SLO':<10} {'c1':>9} {'c3':>9} {'m31':>9}  {'m31-max':>9}")
    print("-" * 60)
    for (wl, lbl, slo), v in sorted(pooled.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        m31 = mean(v["m31_gp"]); c1 = mean(v["c1_gp"]); c3 = mean(v["c3_gp"])
        d = m31 - max(c1, c3)
        mark = "★" if d > 30 else ("·" if d > 10 else " ")
        print(f"{wl:<8} {lbl:<10} {c1:>9.1f} {c3:>9.1f} {m31:>9.1f}  {d:>+8.1f}{mark}")

    if verbose:
        print("\n[verbose] per-cell raw (sorted by paradigm Δ desc):\n")
        cells.sort(key=lambda c: c["m31_attain"] - max(c["c1_attain"], c["c3_attain"]),
                   reverse=True)
        print(f"{'wl':<5} {'SLO':<5} {'seed':<5} {'k':>4} {'qps':>5} "
              f"{'c1%':>6} {'c3%':>6} {'m31%':>6} {'Δmax':>6}")
        for c in cells[:30]:
            dmax = c["m31_attain"] - max(c["c1_attain"], c["c3_attain"])
            print(f"{c['workload']:<5} {c['slo_label']:<5} {c['seed']:<5} {c['k']:>4} "
                  f"{c['m31_qps']:>5.2f} {c['c1_attain']:>5.1f}% {c['c3_attain']:>5.1f}% "
                  f"{c['m31_attain']:>5.1f}% {dmax:>+5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true", help="也打 per-cell 前 30")
    ap.add_argument("--kill", action="store_true", help="报告后 kill T6 wrapper")
    args = ap.parse_args()

    done, total, done_dirs = progress()
    cur = current_run()
    pid = wrapper_pid()

    # ETA: 用过去 10 个 sub-dir 的平均落地间隔 估算
    eta_str = "n/a"
    if done >= 2:
        mtimes = sorted(((d / "qps_sweep_summary.json").stat().st_mtime
                         for d in done_dirs[-min(10, done):]))
        if len(mtimes) >= 2:
            avg_gap = (mtimes[-1] - mtimes[0]) / (len(mtimes) - 1)
            remaining = total - done
            eta_s = remaining * avg_gap
            eta_str = f"~{eta_s/3600:.1f}h(基于最近 {len(mtimes)} 个 cell 平均落地间隔 {avg_gap:.0f}s)"

    print("=" * 70)
    print(f" T6 status (UTC {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())})")
    print("=" * 70)
    print(f" wrapper pid  : {pid if pid else 'NOT RUNNING'}")
    print(f" progress     : {done}/{total} sub-dirs done ({done/total*100:.1f}%)")
    print(f"                nonpid alone done: "
          f"{sum(1 for d in done_dirs if d.name.endswith('_nonpid'))}/{N_NONPID}")
    print(f"                pid alone done   : "
          f"{sum(1 for d in done_dirs if d.name.endswith('_pid'))}/{N_PID}")
    print(f" ETA          : {eta_str}")
    print(f" current run  : {cur}")

    paradigm_delta_table(verbose=args.verbose)

    if args.kill and pid:
        print(f"\n[kill] sending SIGTERM to wrapper pid={pid} ...")
        os.kill(pid, 15)
        print(f"      done. 已完成的 sub-dir 保留,re-run 同 sh 会 skip.")


if __name__ == "__main__":
    main()
