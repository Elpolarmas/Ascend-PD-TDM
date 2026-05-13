"""P0-3 posthoc: conv@1860s ttft500/tpot150 tier 上 M3.1 输 C3 4.5pp 的根因诊断.

Question: thesis "M3.1 在区分带一致胜 C3" 在 conv strict tier 有反例
  ttft500/tpot150: M3.1=64.1±1.9 vs C3=68.6±1.3 (M3.1 -4.5pp)
  ttft500/tpot200: M3.1=88.6±0.6 vs C3=87.2±2.2 (M3.1 +1.4pp)

只有最严 tpot(150ms) 上 M3.1 输,tpot 放宽到 200ms 反超。
→ 嫌疑落在 **tpot 长尾**(150-200ms 这一段 M3.1 比 C3 多多少个 req)。

诊断维度 (done 标准 §6):
  1. 每 req 分四类 (ttft 超 / tpot 超 / 都超 / 都不超) 的 M3.1 vs C3 占比
  2. ttft / tpot p50 p90 p99 并列对比
  3. 按 prompt_tokens 分桶看违规率
  4. tpot 在 150-200ms 区间的累计分布 (新增,定位 §3.4 反例的关键)

归因判定 (4 类):
  a. ttft 长尾输       : ttft p99 M3.1 >> C3
  b. tpot 长尾输       : tpot p99 M3.1 >> C3 (或 tpot 150-200 区间 M3.1 多)
  c. 长 prompt 集中失分: prompt_tokens 大桶里 M3.1 违规率 >> C3
  d. 统计偶然          : 整体长尾 M3.1 更短,只是 strict tier 截止线碰巧不利
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACE = "conv"
CONFIGS = [("M3.1", "c2_tdm_m31_2048"), ("C3", "c3_cp")]
SEEDS = [0, 1, 2]
TIER_TTFT_MS = 500.0
TIER_TPOT_MS = 150.0
PROMPT_BUCKETS = [(0, 500), (500, 1500), (1500, 3000), (3000, 7000)]


def load(seed, cfg):
    p = ROOT / f"{TRACE}_seed{seed}" / f"{cfg}_qps0.0.json"
    return json.loads(p.read_text())


def in_window_reqs(d):
    """Return reqs that count toward meet_slo% (matched + in eval window)."""
    summ = d["summary"]
    w = summ["warmup_s"]
    we = w + summ["duration_s"]
    return [j for j in d["joined"]
            if w <= j["arrival_time_s"] < we
            and j["status"] == 200 and j.get("matched")
            and j.get("ttft_ms") is not None
            and j.get("tpot_ms_mean") is not None
            and j.get("output_tokens")]


def collect_all(cfg):
    reqs = []
    for s in SEEDS:
        reqs.extend(in_window_reqs(load(s, cfg)))
    return reqs


def pctl(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def classify(reqs, t_lim, p_lim):
    """4-class breakdown."""
    n = len(reqs)
    ttft_only = sum(1 for j in reqs if j["ttft_ms"] >= t_lim and j["tpot_ms_mean"] < p_lim)
    tpot_only = sum(1 for j in reqs if j["ttft_ms"] < t_lim and j["tpot_ms_mean"] >= p_lim)
    both = sum(1 for j in reqs if j["ttft_ms"] >= t_lim and j["tpot_ms_mean"] >= p_lim)
    neither = sum(1 for j in reqs if j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim)
    return {"ttft_only": ttft_only, "tpot_only": tpot_only,
            "both": both, "neither": neither, "n": n}


def by_prompt_bucket(reqs, t_lim, p_lim):
    rows = []
    for lo, hi in PROMPT_BUCKETS:
        bucket = [j for j in reqs if lo <= j["prompt_tokens"] < hi]
        n = len(bucket)
        meet = sum(1 for j in bucket if j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim)
        rows.append((f"{lo}-{hi}", n, meet, 100.0 * meet / max(1, n)))
    return rows


def tpot_band_count(reqs, lo_ms, hi_ms):
    return sum(1 for j in reqs if lo_ms <= j["tpot_ms_mean"] < hi_ms)


def main():
    print("=" * 100)
    print(f"P0-3: conv@1860s ttft<{TIER_TTFT_MS:.0f}/tpot<{TIER_TPOT_MS:.0f} 上 M3.1 输 C3 根因")
    print(f"  3 seeds × 60s eval window, pooled per config")
    print("=" * 100)

    data = {label: collect_all(cfg) for label, cfg in CONFIGS}

    # -------- 1. 4-class breakdown --------
    print(f"\n## 1. Per-req 4-class breakdown (tier={TIER_TTFT_MS:.0f}/{TIER_TPOT_MS:.0f})\n")
    print(f"{'':<10}{'n':>6}{'neither':>10}{'ttft_only':>11}{'tpot_only':>11}{'both':>8}"
          f"{'  ':>4}{'(% of n)':<8}")
    print("-" * 90)
    for label, _ in CONFIGS:
        c = classify(data[label], TIER_TTFT_MS, TIER_TPOT_MS)
        n = c["n"]
        print(f"{label:<10}{n:>6}"
              f"{c['neither']:>10}({100.0*c['neither']/n:>5.1f}%)"
              f"{c['ttft_only']:>5}({100.0*c['ttft_only']/n:>5.1f}%)"
              f"{c['tpot_only']:>5}({100.0*c['tpot_only']/n:>5.1f}%)"
              f"{c['both']:>5}({100.0*c['both']/n:>5.1f}%)")
    print("\n  → meet% = neither/n. M3.1 输 C3 来自哪一类的 Δ 占比最大?")

    # Per-class delta
    cm = classify(data["M3.1"], TIER_TTFT_MS, TIER_TPOT_MS)
    cc = classify(data["C3"], TIER_TTFT_MS, TIER_TPOT_MS)
    print(f"\n  Δ占比 (M3.1 - C3, pp):")
    for k in ("neither", "ttft_only", "tpot_only", "both"):
        d_pp = 100.0 * (cm[k]/cm["n"] - cc[k]/cc["n"])
        print(f"    {k:>10}: {d_pp:+.2f}pp")

    # -------- 2. ttft / tpot percentiles --------
    print("\n## 2. ttft / tpot percentiles (pooled across seeds)\n")
    print(f"{'metric':<22}{'M3.1':>12}{'C3':>12}{'Δ (M3.1-C3)':>16}")
    print("-" * 64)
    for metric, key in [("ttft_p50 (ms)", ("ttft_ms", 0.5)),
                        ("ttft_p90 (ms)", ("ttft_ms", 0.9)),
                        ("ttft_p99 (ms)", ("ttft_ms", 0.99)),
                        ("tpot_p50 (ms)", ("tpot_ms_mean", 0.5)),
                        ("tpot_p90 (ms)", ("tpot_ms_mean", 0.9)),
                        ("tpot_p99 (ms)", ("tpot_ms_mean", 0.99))]:
        field, q = key
        m = pctl([j[field] for j in data["M3.1"]], q)
        c = pctl([j[field] for j in data["C3"]], q)
        print(f"{metric:<22}{m:>12.1f}{c:>12.1f}{m-c:>+15.1f}")

    # -------- 3. by prompt_tokens bucket --------
    print(f"\n## 3. meet% by prompt_tokens bucket (tier={TIER_TTFT_MS:.0f}/{TIER_TPOT_MS:.0f})\n")
    print(f"{'bucket':<14}{'M3.1 n':>8}{'M3.1 meet%':>13}"
          f"{'C3 n':>8}{'C3 meet%':>13}{'Δ meet%':>10}")
    print("-" * 70)
    m_rows = by_prompt_bucket(data["M3.1"], TIER_TTFT_MS, TIER_TPOT_MS)
    c_rows = by_prompt_bucket(data["C3"], TIER_TTFT_MS, TIER_TPOT_MS)
    for (lbl, mn, mm, mp), (_, cn, cm_, cp) in zip(m_rows, c_rows):
        print(f"{lbl:<14}{mn:>8}{mp:>12.1f}%{cn:>8}{cp:>12.1f}%{mp-cp:>+9.1f}pp")

    # -------- 4. tpot 150-200 区间分布 (定位反例) --------
    print(f"\n## 4. tpot 分布在 150/200 截止线附近 (诊断 strict tier 反例)\n")
    print(f"{'tpot 区间 (ms)':<20}{'M3.1 n':>10}{'M3.1 %':>10}{'C3 n':>10}{'C3 %':>10}{'Δ%':>10}")
    print("-" * 70)
    bands = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 10**9)]
    for lo, hi in bands:
        mn = tpot_band_count(data["M3.1"], lo, hi)
        cn = tpot_band_count(data["C3"], lo, hi)
        m_tot = len(data["M3.1"])
        c_tot = len(data["C3"])
        m_pct = 100.0 * mn / m_tot
        c_pct = 100.0 * cn / c_tot
        hi_lbl = "inf" if hi > 1e6 else f"{hi}"
        print(f"{f'{lo}-{hi_lbl}':<20}{mn:>10}{m_pct:>9.1f}%{cn:>10}{c_pct:>9.1f}%{m_pct-c_pct:>+9.1f}pp")
    print("\n  → 若 150-200 区间 M3.1 比 C3 显著多 → 反例 = 'M3.1 tpot 长尾在 150-200 段超载'")

    # -------- 5. Verdict scaffold --------
    print("\n" + "=" * 100)
    print("Verdict (manual interpretation):")
    print("  a. ttft 长尾输       : ttft p99 M3.1 - C3 显著 > 0")
    print("  b. tpot 长尾输       : tpot p99 / 150-200 band M3.1 显著 > C3")
    print("  c. 长 prompt 集中失分: 大 prompt 桶 (1500+) 里 M3.1 - C3 显著负")
    print("  d. 统计偶然          : 上述都不显著,但 tpot 截止线刚好压在分布转折点")
    print("=" * 100)


if __name__ == "__main__":
    main()
