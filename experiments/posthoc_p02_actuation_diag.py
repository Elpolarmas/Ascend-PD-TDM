"""P0-2 posthoc: SLO 自适应控制器 PID 的 actuation 形态诊断.

Question: §3.4 conv 上 M3.1 比 M1+chunk 赢 +5.2pp,这个 +5.2pp 来自:
  (a) PID 真在持续微调 target_ratio (连续 actuation) ?
  (b) PID 只在少数事件点跳一下 (稀疏跳动) ?
  (c) PID 大部分时间被 latch/floor 钉死,几乎不动 (钉死) ?

Inputs:
  results/azure_main/conv_seed{0,1,2}/tdm_trace/qps_sweep_c2_tdm_m31_2048_ctrl.jsonl
  字段: iter_id ts_ms ratio_before ratio_after err_ttft err_tpot_effective
       ttft_viol_rate tpot_viol_rate tpot_saturated tpot_clamp_active
       kv_freeze_triggered cold_start deadband_dropped

量化指标 (done 标准 §6):
  1. ratio_after std / range / 离 floor(0.05) 距离 / 离初始(0.3) 距离
  2. latch duty cycle: cold_start / kv_freeze / tpot_saturated / tpot_clamp 各自占比
  3. deadband 被 drop 比例 (delta < 0.02)
  4. err 是否非零 (ReLU 后): 衡量"PID 真有信号 vs 无信号"
  5. ratio 轨迹 ASCII 速览 (3 seeds 并列)

分类 (3 类):
  - 连续 actuation: std > 0.05, 大部分时间不在 floor, 有围绕中值波动
  - 稀疏跳动: std 中等, 主要钉某值偶尔跳
  - 钉死: std < 0.01
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
CFG = "c2_tdm_m31_2048"
TRACE = "conv"
SEEDS = [0, 1, 2]
RATIO_FLOOR = 0.05
INITIAL_RATIO = 0.3
DEADBAND = 0.02


def load_ticks(seed):
    p = ROOT / f"{TRACE}_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    ticks = []
    for line in p.read_text().strip().split("\n"):
        ticks.append(json.loads(line))
    return ticks


def in_eval_window(ticks):
    """Drop the cold_start prefix and convert ts_ms → seconds since first tick."""
    if not ticks:
        return []
    t0 = ticks[0]["ts_ms"]
    out = []
    for t in ticks:
        rec = dict(t)
        rec["t_s"] = (t["ts_ms"] - t0) / 1000.0
        out.append(rec)
    return out


def quantize(ticks, floor=RATIO_FLOOR):
    """Compute summary stats for one seed."""
    n = len(ticks)
    if n == 0:
        return {}
    # post-cold-start view: drop cold_start=True rows for actuation stats
    warm = [t for t in ticks if not t.get("cold_start", False)]
    n_warm = len(warm)
    if n_warm == 0:
        return {"n": n, "warm": 0, "note": "all cold_start"}

    ratios = [t["ratio_after"] for t in warm]
    ratio_mean = statistics.mean(ratios)
    ratio_std = statistics.stdev(ratios) if n_warm > 1 else 0.0
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    pinned_floor = sum(1 for r in ratios if abs(r - floor) < 1e-3) / n_warm
    pinned_initial = sum(1 for r in ratios if abs(r - INITIAL_RATIO) < 1e-3) / n_warm

    # Latch duty cycles (over warm ticks)
    cold_pct = (n - n_warm) / n
    kvf = sum(1 for t in warm if t.get("kv_freeze_triggered")) / n_warm
    sat = sum(1 for t in warm if t.get("tpot_saturated")) / n_warm
    clamp = sum(1 for t in warm if t.get("tpot_clamp_active")) / n_warm
    deadband = sum(1 for t in warm if t.get("deadband_dropped")) / n_warm

    # Err signal magnitudes
    err_ttft_nz = sum(1 for t in warm if abs(t.get("err_ttft", 0)) > 1e-6) / n_warm
    err_tpot_nz = sum(1 for t in warm if abs(t.get("err_tpot_effective", 0)) > 1e-6) / n_warm

    # Step magnitude (|ratio_after - ratio_before|)
    steps = [abs(t["ratio_after"] - t["ratio_before"]) for t in warm]
    step_mean = statistics.mean(steps) if steps else 0.0
    step_max = max(steps) if steps else 0.0
    n_meaningful_steps = sum(1 for s in steps if s > 1e-4)

    return {
        "n": n, "warm": n_warm,
        "ratio_mean": ratio_mean, "ratio_std": ratio_std,
        "ratio_min": ratio_min, "ratio_max": ratio_max,
        "pinned_floor": pinned_floor, "pinned_initial": pinned_initial,
        "cold_pct": cold_pct, "kvf": kvf, "saturated": sat,
        "clamp": clamp, "deadband": deadband,
        "err_ttft_nz": err_ttft_nz, "err_tpot_nz": err_tpot_nz,
        "step_mean": step_mean, "step_max": step_max,
        "n_meaningful_steps": n_meaningful_steps,
    }


def classify(s):
    """Categorize PID behavior from stats."""
    if not s or s.get("n") == 0:
        return "NO_DATA"
    if s.get("warm", 0) == 0:
        return "ALL_COLD"
    if s["ratio_std"] < 0.01 and s["pinned_initial"] > 0.7:
        return "STUCK (主要钉初始 0.3)"
    if s["ratio_std"] < 0.01 and s["pinned_floor"] > 0.7:
        return "STUCK (主要钉 floor 0.05)"
    if s["ratio_std"] < 0.01:
        return "STUCK (其他值)"
    if s["ratio_std"] > 0.05:
        return "CONTINUOUS ACTUATION"
    return "SPARSE JUMPS"


def ascii_traj(ticks, width=80):
    """ASCII line of ratio_after over time (post-warm only)."""
    warm = [t for t in ticks if not t.get("cold_start", False)]
    if not warm:
        return "(no warm ticks)"
    # bucket by time index into `width` columns
    n = len(warm)
    cols = []
    for i in range(width):
        lo = i * n // width
        hi = max(lo + 1, (i + 1) * n // width)
        vals = [warm[j]["ratio_after"] for j in range(lo, hi)]
        cols.append(statistics.mean(vals) if vals else 0)
    # render: rows from top to bottom in 8 levels
    rows = 8
    lines = []
    for r in range(rows):
        thresh_hi = 1.0 - r / rows  # row 0 → ratio ≈ 1.0, row 7 → ≈ 0.125
        thresh_lo = 1.0 - (r + 1) / rows
        line = ""
        for c in cols:
            if thresh_lo <= c < thresh_hi or (r == rows - 1 and c < thresh_hi):
                line += "█"
            else:
                line += " "
        lines.append(f"  {thresh_hi:.2f} │{line}│ {thresh_lo:.2f}")
    return "\n".join(lines)


def main():
    print("=" * 100)
    print(f"P0-2: SLO 自适应控制器 actuation 诊断 (M3.1 on conv@1860s, 3 seeds)")
    print(f"  source: results/azure_main/conv_seed*/tdm_trace/qps_sweep_{CFG}_ctrl.jsonl")
    print("=" * 100)

    all_stats = {}
    for s in SEEDS:
        ticks = in_eval_window(load_ticks(s))
        all_stats[s] = (ticks, quantize(ticks))

    # ---- 1. Per-seed summary ----
    print("\n## 1. Per-seed actuation stats\n")
    header = f"{'metric':<28}" + "".join(f"  seed{s:>2}" for s in SEEDS) + f"   mean"
    print(header)
    print("-" * len(header))
    metrics = [
        ("n_ticks (total)",       "n",        "{:>7d}", False),
        ("n_warm (post cold)",    "warm",     "{:>7d}", False),
        ("ratio mean",            "ratio_mean", "{:>7.3f}", True),
        ("ratio std",             "ratio_std",  "{:>7.3f}", True),
        ("ratio min",             "ratio_min",  "{:>7.3f}", True),
        ("ratio max",             "ratio_max",  "{:>7.3f}", True),
        ("% pinned floor(0.05)",  "pinned_floor", "{:>6.1%}", True),
        ("% pinned initial(0.3)", "pinned_initial", "{:>6.1%}", True),
        ("% cold_start",          "cold_pct",   "{:>6.1%}", True),
        ("% kv_freeze",           "kvf",        "{:>6.1%}", True),
        ("% tpot_saturated",      "saturated",  "{:>6.1%}", True),
        ("% tpot_clamp_active",   "clamp",      "{:>6.1%}", True),
        ("% deadband_dropped",    "deadband",   "{:>6.1%}", True),
        ("% err_ttft != 0",       "err_ttft_nz","{:>6.1%}", True),
        ("% err_tpot_eff != 0",   "err_tpot_nz","{:>6.1%}", True),
        ("mean |step|",           "step_mean",  "{:>7.4f}", True),
        ("max |step|",            "step_max",   "{:>7.4f}", True),
        ("n meaningful steps",    "n_meaningful_steps", "{:>7d}", True),
    ]
    for label, key, fmt, is_num in metrics:
        row = f"{label:<28}"
        vals = []
        for s in SEEDS:
            v = all_stats[s][1].get(key, 0)
            row += "  " + fmt.format(v) if v is not None else "      —"
            if is_num:
                vals.append(v)
        if vals:
            m = statistics.mean(vals)
            try:
                if isinstance(m, float):
                    if "%" in fmt:
                        row += f"   {m:>6.1%}"
                    elif "f" in fmt:
                        row += f"   {m:>7.3f}"
                else:
                    row += f"   {m:>7d}"
            except Exception:
                row += "      —"
        print(row)

    # ---- 2. Classification per seed ----
    print("\n## 2. Behavior classification per seed\n")
    for s in SEEDS:
        c = classify(all_stats[s][1])
        print(f"  seed {s}: {c}")

    # ---- 3. ASCII trajectory per seed ----
    print("\n## 3. target_ratio trajectory (post-warm, time → right)\n")
    for s in SEEDS:
        ticks, _ = all_stats[s]
        print(f"\n  --- seed {s} ---")
        print(ascii_traj(ticks))

    # ---- 4. Verdict ----
    print("\n" + "=" * 100)
    print("Verdict template:")
    print("  连续 actuation (std > 0.05): PID 真在持续微调,+5.2pp 主因是闭环 actuation")
    print("  稀疏跳动:                    PID 在事件点起作用,+5.2pp 是事件响应")
    print("  钉死:                        PID 实际不工作,+5.2pp 来自 initial_ratio 选得好 / latch 启发式")
    print("=" * 100)


if __name__ == "__main__":
    main()
