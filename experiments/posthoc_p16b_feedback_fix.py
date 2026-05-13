"""P1.6b posthoc — 反馈链修复前后对照

对比 azure_main 历史 M3.1(反馈链坏)vs azure_p16b 新 M3.1(F1+F2 修复)。
关键指标(M3.1 自身内部行为):
  1. n_ttft_window / n_tpot_window warm 时刻
  2. ratio 时序(钉 ratio_max 占比、mean、首次到 max 时刻)
  3. tpot_saturated / tpot_clamp_active 屏蔽占比
  4. err_tpot_effective 非零占比(双输入是否复活)
  5. |delta_slo| / |delta_q| 分布

PASS:warm <2s、ratio 钉 max 占比 <80%、tpot 屏蔽 <80% 之一显著改善 → 反馈链是根因
FAIL:三项都纹丝不动 → 控制律本身不适配,P1.6c 走 β

run: python3 experiments/posthoc_p16b_feedback_fix.py
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
TRACES = ["conv", "code"]
SEEDS = [0, 1, 2]
CFG = "c2_tdm_m31_2048"
RATIO_MAX = 0.8
MIN_SAMPLES = 16


def load(base: Path, trace_label: str, seed: int) -> list[dict]:
    """Load ctrl.jsonl. base is either azure_main (uses trace_seedN) or
    azure_p16b (uses trace_w1_seedN)."""
    if base.name == "azure_main":
        p = base / f"{trace_label}_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    else:
        p = base / f"{trace_label}_w1_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def summarise(ticks: list[dict]) -> dict:
    if not ticks:
        return {}
    t0 = ticks[0]["ts_ms"]
    warm = [t for t in ticks if not t["cold_start"]]

    # warm 时刻
    warm_ttft_s = None
    warm_tpot_s = None
    for t in ticks:
        if warm_ttft_s is None and t["n_ttft_window"] >= MIN_SAMPLES:
            warm_ttft_s = (t["ts_ms"] - t0) / 1000
        if warm_tpot_s is None and t["n_tpot_window"] >= MIN_SAMPLES:
            warm_tpot_s = (t["ts_ms"] - t0) / 1000
        if warm_ttft_s and warm_tpot_s:
            break

    # ratio 首次到 max
    first_max_s = None
    for t in ticks:
        if not t["cold_start"] and t["ratio_after"] >= RATIO_MAX - 1e-6:
            first_max_s = (t["ts_ms"] - t0) / 1000
            break

    if not warm:
        return {"n_warm": 0, "warm_ttft_s": warm_ttft_s,
                "warm_tpot_s": warm_tpot_s, "first_max_s": first_max_s}

    n = len(warm)
    n_clamp = sum(1 for t in warm if t["tpot_clamp_active"])
    n_sat = sum(1 for t in warm if t["tpot_saturated"])
    n_either = sum(1 for t in warm if t["tpot_clamp_active"] or t["tpot_saturated"])
    n_at_max = sum(1 for t in warm if t["ratio_after"] >= RATIO_MAX - 1e-6)
    n_eff_tpot_nonzero = sum(1 for t in warm
                              if abs(t["err_tpot_effective"]) > 1e-9)
    abs_delta_slo = [abs(t["delta_slo"]) for t in warm]
    abs_delta_q = [abs(t["delta_q"]) for t in warm]
    ratios = [t["ratio_after"] for t in warm]

    return {
        "n_warm": n,
        "warm_ttft_s": warm_ttft_s,
        "warm_tpot_s": warm_tpot_s,
        "first_max_s": first_max_s,
        "pct_clamp": 100 * n_clamp / n,
        "pct_sat": 100 * n_sat / n,
        "pct_either": 100 * n_either / n,
        "pct_at_max": 100 * n_at_max / n,
        "pct_eff_tpot_nonzero": 100 * n_eff_tpot_nonzero / n,
        "ratio_mean": mean(ratios),
        "abs_delta_slo_mean": mean(abs_delta_slo),
        "abs_delta_q_mean": mean(abs_delta_q),
    }


def aggregate(base: Path, trace: str) -> dict:
    rows = [summarise(load(base, trace, s)) for s in SEEDS]
    rows = [r for r in rows if r]
    if not rows:
        return {}
    keys = ["warm_ttft_s", "warm_tpot_s", "first_max_s", "pct_clamp",
            "pct_sat", "pct_either", "pct_at_max", "pct_eff_tpot_nonzero",
            "ratio_mean", "abs_delta_slo_mean", "abs_delta_q_mean"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        agg[k] = mean(vals) if vals else None
    return agg


def fmt(v, prec=2, suffix=""):
    return f"{v:.{prec}f}{suffix}" if v is not None else "  —  "


def main():
    print("=" * 98)
    print("P1.6b 反馈链修复前后对照 (M3.1 自身,跨 seed mean)")
    print("=" * 98)

    base_old = ROOT / "azure_main"
    base_new = ROOT / "azure_p16b"
    if not base_new.exists():
        print(f"\n⚠️  {base_new} 不存在 — P1.6b 实验未跑或未完成")
        return

    header = (f"{'指标':<32}{'azure_main(旧)':>20}"
              f"{'azure_p16b(新)':>20}{'变化':>20}")
    for trace in TRACES:
        print(f"\n── trace = {trace} ──")
        old = aggregate(base_old, trace)
        new = aggregate(base_new, trace)
        if not old or not new:
            print(f"  数据缺失:old={'有' if old else '无'} new={'有' if new else '无'}")
            continue

        print(header)
        print("-" * 92)

        rows = [
            ("warm_ttft (s,越小越好)", "warm_ttft_s", 1, "s"),
            ("warm_tpot (s,越小越好)", "warm_tpot_s", 1, "s"),
            ("ratio 首到 max (s)", "first_max_s", 1, "s"),
            ("ratio 钉 max 占比 (%)", "pct_at_max", 1, "%"),
            ("ratio mean", "ratio_mean", 3, ""),
            ("tpot_sat 屏蔽 (%)", "pct_sat", 1, "%"),
            ("tpot_clamp 屏蔽 (%)", "pct_clamp", 1, "%"),
            ("任一屏蔽 (%)", "pct_either", 1, "%"),
            ("err_tpot_eff 非零 (%)", "pct_eff_tpot_nonzero", 1, "%"),
            ("|delta_slo| mean", "abs_delta_slo_mean", 4, ""),
            ("|delta_q| backlog mean", "abs_delta_q_mean", 4, ""),
        ]
        for label, key, prec, suffix in rows:
            o = old.get(key)
            n = new.get(key)
            if o is not None and n is not None:
                diff = n - o
                diff_str = f"{diff:+.{prec}f}{suffix}"
            else:
                diff_str = "—"
            print(f"{label:<32}{fmt(o, prec, suffix):>20}"
                  f"{fmt(n, prec, suffix):>20}{diff_str:>20}")

        # PASS / FAIL 判定
        n_warm_ok = (new.get("warm_ttft_s") is not None
                     and new["warm_ttft_s"] < 2.0)
        n_max_ok = (new.get("pct_at_max") is not None
                    and new["pct_at_max"] < 80.0)
        n_sat_ok = (new.get("pct_either") is not None
                    and new["pct_either"] < 80.0)
        n_double_input = (new.get("pct_eff_tpot_nonzero") is not None
                          and new["pct_eff_tpot_nonzero"] > 20.0)
        criteria = [
            ("warm_ttft < 2s",           n_warm_ok),
            ("ratio 钉 max 占比 < 80%",   n_max_ok),
            ("tpot 屏蔽 < 80%",           n_sat_ok),
            ("双输入复活(err_tpot_eff 非零 > 20%)", n_double_input),
        ]
        n_pass = sum(1 for _, ok in criteria if ok)
        print(f"\n  判定标准({n_pass}/4 命中):")
        for label, ok in criteria:
            print(f"    {'✅' if ok else '❌'}  {label}")
        if n_pass >= 3:
            print(f"  ⇒ {trace}: PASS — 反馈链是根因,救回闭环反馈叙事")
        elif n_pass >= 1:
            print(f"  ⇒ {trace}: PARTIAL — 部分改善,机制混合")
        else:
            print(f"  ⇒ {trace}: FAIL — 控制律本身不适配,走 P1.6c β 走向")


if __name__ == "__main__":
    main()
