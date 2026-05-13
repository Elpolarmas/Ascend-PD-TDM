"""P1.0 posthoc: baseline C3 公平性 audit.

Question: c3_cp @ max_num_batched_tokens=8192(azure_main 现 baseline)实际几乎
不切 chunk。在 batched=2048/512 这两个真正切 chunk 的设置下,C3 表现是否显著
变好?

如果 C3@2048 比 C3@8192 显著好 (>2pp on strict tier):
  → 候选 A 成立 (max_num_batched_tokens 过大让 C3 几乎不切)
  → main result 不公平,需要重做完整 4 config × batched=2048 对比
否则:
  → 候选 A 不成立,8192 baseline 合理

Inputs:
  results/baseline_c3_audit/conv_b{2048,512}_seed{0,1,2}/c3_cp_qps0.0.json
  results/azure_main/conv_seed{0,1,2}/c3_cp_qps0.0.json          (8192 anchor)
  results/azure_main/conv_seed{0,1,2}/c2_tdm_m31_2048_qps0.0.json (M3.1 reference)
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

AUDIT_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/baseline_c3_audit")
AZURE_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACE = "conv"
SEEDS = [0, 1, 2]

# (label, max_batched, config, root)
POINTS = [
    ("C3@batched=512",  512,  "c3_cp",          AUDIT_ROOT),
    ("C3@batched=2048", 2048, "c3_cp",          AUDIT_ROOT),
    ("C3@batched=8192", 8192, "c3_cp",          AZURE_ROOT),  # current baseline
    ("M3.1 ref",        None, "c2_tdm_m31_2048", AZURE_ROOT),
]

TIERS = [
    ("strict          (ttft500/tpot50)",  500,  50),
    ("ttft500_tpot100",                   500, 100),
    ("ttft500_tpot150",                   500, 150),
    ("ttft500_tpot200",                   500, 200),
    ("ttft1000_tpot200",                 1000, 200),
    ("ttft1500_tpot300",                 1500, 300),
]


def load_audit(root, batched, seed, cfg):
    if batched is not None and root == AUDIT_ROOT:
        p = root / f"{TRACE}_b{batched}_seed{seed}" / f"{cfg}_qps0.0.json"
    else:
        p = root / f"{TRACE}_seed{seed}" / f"{cfg}_qps0.0.json"
    return json.loads(p.read_text())


def meet(joined, w, we, t_lim, p_lim):
    in_win = [j for j in joined if w <= j["arrival_time_s"] < we
              and j["status"] == 200 and j.get("matched")]
    n = len(in_win)
    nm = sum(1 for j in in_win
             if j.get("ttft_ms") is not None and j.get("tpot_ms_mean") is not None
             and j["ttft_ms"] < t_lim and j["tpot_ms_mean"] < p_lim
             and j.get("output_tokens"))
    return nm, n


def collect(root, batched, cfg, t_lim, p_lim):
    out = []
    for s in SEEDS:
        d = load_audit(root, batched, s, cfg)
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def window_stat(root, batched, cfg, key, sub=None):
    out = []
    for s in SEEDS:
        w = load_audit(root, batched, s, cfg)["summary"]["window"]
        v = w.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        out.append(v if v is not None else float("nan"))
    return out


def main():
    print("=" * 100)
    print("P1.0: baseline C3 公平性 audit — C3 在不同 max_num_batched_tokens 下的表现")
    print("  conv@1860s × 3 seeds × 60s")
    print("=" * 100)

    # ---- 1. meet_slo% table ----
    print("\n## 1. meet_slo% (3 seeds, mean±std)\n")
    header = f"{'tier':<36}" + "".join(f"{p[0]:>18}" for p in POINTS)
    print(header)
    print("-" * len(header))
    for tier_name, t_lim, p_lim in TIERS:
        row = f"{tier_name:<36}"
        for label, batched, cfg, root in POINTS:
            try:
                pcts = collect(root, batched, cfg, t_lim, p_lim)
                row += f"{fmt(pcts):>18}"
            except FileNotFoundError:
                row += f"{'(missing)':>18}"
        print(row)

    # ---- 2. Δ vs 8192 baseline ----
    print("\n## 2. C3 Δ vs azure_main C3@8192 baseline (paired by seed)\n")
    try:
        base = {tier_name: collect(AZURE_ROOT, None, "c3_cp", t_lim, p_lim)
                for tier_name, t_lim, p_lim in TIERS}
    except FileNotFoundError:
        print("  ERR: cannot find azure_main c3_cp anchor")
        return

    print(f"{'tier':<36}{'C3@2048 - C3@8192':>22}{'C3@512 - C3@8192':>22}")
    print("-" * 80)
    for tier_name, t_lim, p_lim in TIERS:
        row = f"{tier_name:<36}"
        for batched in (2048, 512):
            try:
                vals = collect(AUDIT_ROOT, batched, "c3_cp", t_lim, p_lim)
                diff = [v - b for v, b in zip(vals, base[tier_name])]
                m = statistics.mean(diff)
                sd = statistics.stdev(diff) if len(diff) > 1 else 0.0
                row += f"{m:>+15.1f}±{sd:<4.1f}pp"
            except FileNotFoundError:
                row += f"{'(missing)':>22}"
        print(row)

    # ---- 3. Window stats compare ----
    print("\n## 3. Window stats: ttft / tpot percentiles\n")
    metrics = [
        ("ttft_p50 (ms)",   "ttft_ms",      "p50"),
        ("ttft_p99 (ms)",   "ttft_ms",      "p99"),
        ("tpot_p50 (ms)",   "tpot_ms_mean", "p50"),
        ("tpot_p99 (ms)",   "tpot_ms_mean", "p99"),
        ("goodput (tok/s)", "goodput_tok_s", None),
    ]
    header = f"{'metric':<22}" + "".join(f"{p[0]:>18}" for p in POINTS)
    print(header)
    print("-" * len(header))
    for label, key, sub in metrics:
        row = f"{label:<22}"
        for plabel, batched, cfg, root in POINTS:
            try:
                vals = window_stat(root, batched, cfg, key, sub)
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"{m:>12.1f}±{sd:<4.1f}"
            except FileNotFoundError:
                row += f"{'(missing)':>18}"
        print(row)

    # ---- 4. Verdict ----
    print("\n" + "=" * 100)
    print("Verdict:")
    print("  - 若 C3@2048 - C3@8192 ≥ +2pp on strict tier:")
    print("       候选 A 成立,baseline 不公,main result 需重做 with batched=2048")
    print("  - 若变化不显著 (<1pp):")
    print("       候选 A 不成立,8192 baseline 合理,需找别的反常归因")
    print("=" * 100)


if __name__ == "__main__":
    main()
