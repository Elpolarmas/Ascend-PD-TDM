"""P1.0b posthoc: HCCL=AIV 下 C3 表现 vs FFTS+ 模式.

4 个对比点:
  (1) C3@8192 FFTS+ : azure_main  (原 baseline)
  (2) C3@2048 FFTS+ : baseline_c3_audit
  (3) C3@8192 AIV   : baseline_c3_audit_aiv (新)
  (4) C3@2048 AIV   : baseline_c3_audit_aiv (新)

假设验证:
  - 若 (4) > (2) 显著:AIV 模式让 mixed batch 命中 graph,chunked prefill 切多次的代价降低
  - 若 (4) > (1) 显著:在 AIV 下真正的 chunked prefill (切 chunk) 优于不切
  - 若 (4) > (3) 显著:小 batched 在 AIV 下反超大 batched,baseline 公平性问题成立
  - 若 (4) ≈ (2)    :AIV 没改变 C3 趋势,根因不在通信模式
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

AZURE_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
AUDIT_FFTS = Path("/vllm-workspace/Ascend-PD-TDM/results/baseline_c3_audit")
AUDIT_AIV = Path("/vllm-workspace/Ascend-PD-TDM/results/baseline_c3_audit_aiv")

TRACE = "conv"
SEEDS = [0, 1, 2]

# (label, source_root, source_subdir_template, cfg)
POINTS = [
    ("C3@2048 FFTS+", AUDIT_FFTS, "{trace}_b2048_seed{s}", "c3_cp"),
    ("C3@8192 FFTS+", AZURE_ROOT, "{trace}_seed{s}",       "c3_cp"),
    ("C3@2048 AIV",   AUDIT_AIV,  "{trace}_b2048_seed{s}", "c3_cp"),
    ("C3@8192 AIV",   AUDIT_AIV,  "{trace}_b8192_seed{s}", "c3_cp"),
    ("M3.1 ref (FFTS+)", AZURE_ROOT, "{trace}_seed{s}",    "c2_tdm_m31_2048"),
]

TIERS = [
    ("strict   (ttft500/tpot50)",    500,  50),
    ("ttft500_tpot100",              500, 100),
    ("ttft500_tpot150",              500, 150),
    ("ttft500_tpot200",              500, 200),
    ("ttft1000_tpot200",            1000, 200),
    ("ttft1500_tpot300",            1500, 300),
]


def load(root, subdir_template, cfg, seed):
    subdir = subdir_template.format(trace=TRACE, s=seed)
    p = root / subdir / f"{cfg}_qps0.0.json"
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


def collect(point, t_lim, p_lim):
    label, root, tmpl, cfg = point
    out = []
    for s in SEEDS:
        d = load(root, tmpl, cfg, s)
        summ = d["summary"]
        we = summ["warmup_s"] + summ["duration_s"]
        nm, nT = meet(d["joined"], summ["warmup_s"], we, t_lim, p_lim)
        out.append(100.0 * nm / max(1, nT))
    return out


def window_stat(point, key, sub=None):
    label, root, tmpl, cfg = point
    out = []
    for s in SEEDS:
        w = load(root, tmpl, cfg, s)["summary"]["window"]
        v = w.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        out.append(v if v is not None else float("nan"))
    return out


def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:5.1f}±{sd:4.1f}"


def main():
    print("=" * 110)
    print("P1.0b: HCCL=AIV 下 C3 表现 — 验证 capture sizes 限制是否是 P1.0 反常的根因")
    print("  conv@1860s × 3 seeds × 60s")
    print("  AIV 模式 capture sizes 23 (vs FFTS+ 模式 15),包含 mixed prefill-decode graph")
    print("=" * 110)

    print("\n## 1. meet_slo% (3 seeds, mean±std)\n")
    header = f"{'tier':<30}" + "".join(f"{p[0]:>18}" for p in POINTS)
    print(header)
    print("-" * len(header))
    for tier_name, t_lim, p_lim in TIERS:
        row = f"{tier_name:<30}"
        for point in POINTS:
            try:
                pcts = collect(point, t_lim, p_lim)
                row += f"{fmt(pcts):>18}"
            except FileNotFoundError:
                row += f"{'(missing)':>18}"
        print(row)

    # 关键 Δ 对比
    print("\n## 2. 关键配对 Δ (验证假设)\n")
    pairs = [
        ("AIV 效应 on C3@2048", ("C3@2048 AIV",), ("C3@2048 FFTS+",)),
        ("AIV 效应 on C3@8192", ("C3@8192 AIV",), ("C3@8192 FFTS+",)),
        ("AIV 下 small vs large batched", ("C3@2048 AIV",), ("C3@8192 AIV",)),
        ("AIV-C3@2048 vs FFTS+-C3@8192 (orig baseline)", ("C3@2048 AIV",), ("C3@8192 FFTS+",)),
    ]
    point_by_label = {p[0]: p for p in POINTS}

    print(f"{'tier':<30}" + "".join(f"{name:>40}" for name, _, _ in pairs))
    print("-" * (30 + 40 * len(pairs)))
    for tier_name, t_lim, p_lim in TIERS:
        row = f"{tier_name:<30}"
        for name, (a,), (b,) in pairs:
            try:
                A = collect(point_by_label[a], t_lim, p_lim)
                B = collect(point_by_label[b], t_lim, p_lim)
                d = [x - y for x, y in zip(A, B)]
                m = statistics.mean(d)
                sd = statistics.stdev(d) if len(d) > 1 else 0.0
                row += f"{m:>+33.1f}±{sd:<4.1f}pp"
            except FileNotFoundError:
                row += f"{'(missing)':>40}"
        print(row)

    # Window stats
    print("\n## 3. Window stats (mean across seeds)\n")
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
        for point in POINTS:
            try:
                vals = window_stat(point, key, sub)
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"{m:>12.1f}±{sd:<4.1f}"
            except FileNotFoundError:
                row += f"{'(missing)':>18}"
        print(row)

    print("\n" + "=" * 110)
    print("Verdict:")
    print("  - 若 AIV 效应明显 (C3@2048 AIV ≫ C3@2048 FFTS+): capture sizes 限制是机制")
    print("  - 若 AIV 下 C3@2048 反超 C3@8192: baseline 不公,main result 需要 in AIV 下重做")
    print("  - 若 AIV-C3@2048 仍输 azure_main M3.1: 即使公平 baseline 下我们仍胜")
    print("=" * 110)


if __name__ == "__main__":
    main()
