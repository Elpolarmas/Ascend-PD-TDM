"""P1.6a — PID tpot 屏蔽逻辑审计 posthoc

读 results/azure_main/{conv,code}_seed{0,1,2}/tdm_trace/*_m31_2048_ctrl.jsonl,
量化:
  1. 屏蔽触发占比(M2.1 clamp vs M2.4 saturation vs OR)
  2. 屏蔽时刻分布(cold_start 启动期 vs 稳态)
  3. 屏蔽前的 err_tpot_raw 分布(轻微超 SLO vs 远超)
  4. delta_slo vs delta_q 各自贡献(backlog 项还在不在做事)
  5. ratio 钉边界占比(ratio_after == ratio_max 的 tick 占比)
  6. 单 PID 等效 vs 双输入 PID 形态对照

run: python3 experiments/posthoc_p16a_pid_mask_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

RESULTS = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACES = ["conv", "code"]
SEEDS = [0, 1, 2]
CFG = "c2_tdm_m31_2048"  # M3.1 = 我们的 PID config
RATIO_MAX = 0.8  # 与 controller.py 默认对齐

def load(trace: str, seed: int) -> list[dict]:
    p = RESULTS / f"{trace}_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def summarise(ticks: list[dict]) -> dict:
    if not ticks:
        return {}
    warm = [t for t in ticks if not t["cold_start"]]
    if not warm:
        return {"n_warm": 0}

    n = len(warm)
    n_clamp = sum(1 for t in warm if t["tpot_clamp_active"])
    n_sat = sum(1 for t in warm if t["tpot_saturated"])
    n_either = sum(1 for t in warm if t["tpot_clamp_active"] or t["tpot_saturated"])
    n_both = sum(1 for t in warm if t["tpot_clamp_active"] and t["tpot_saturated"])

    # err_tpot_raw 分布(屏蔽时点)
    raw_when_masked = [
        t["err_tpot_raw"] for t in warm
        if (t["tpot_clamp_active"] or t["tpot_saturated"])
        and t["n_tpot_window"] >= 16
    ]
    raw_when_unmasked = [
        t["err_tpot_raw"] for t in warm
        if not (t["tpot_clamp_active"] or t["tpot_saturated"])
        and t["n_tpot_window"] >= 16
    ]
    tpot_viol_when_masked = [
        t["tpot_viol_rate"] for t in warm
        if (t["tpot_clamp_active"] or t["tpot_saturated"])
        and t["n_tpot_window"] >= 16
    ]

    # delta 分解
    abs_delta_slo = [abs(t["delta_slo"]) for t in warm]
    abs_delta_q = [abs(t["delta_q"]) for t in warm]
    n_nonzero_q = sum(1 for t in warm if abs(t["delta_q"]) > 1e-9)
    n_nonzero_slo = sum(1 for t in warm if abs(t["delta_slo"]) > 1e-9)

    # ratio 钉边界
    n_at_max = sum(1 for t in warm if t["ratio_after"] >= RATIO_MAX - 1e-6)
    n_at_min = sum(1 for t in warm if t["ratio_after"] <= 0.05 + 1e-6)

    # ratio 分布
    ratios = [t["ratio_after"] for t in warm]

    return {
        "n_warm": n,
        # 屏蔽路径
        "pct_clamp_M21": 100 * n_clamp / n,
        "pct_sat_M24": 100 * n_sat / n,
        "pct_either": 100 * n_either / n,
        "pct_both": 100 * n_both / n,
        # 屏蔽时的 err_tpot_raw(转回百分比:err_tpot_raw = viol_rate - 0.05)
        "err_tpot_raw_when_masked_mean": mean(raw_when_masked) if raw_when_masked else None,
        "err_tpot_raw_when_masked_p50": median(raw_when_masked) if raw_when_masked else None,
        "err_tpot_raw_when_masked_p99": (
            sorted(raw_when_masked)[int(0.99 * (len(raw_when_masked) - 1))]
            if raw_when_masked else None),
        "tpot_viol_when_masked_mean": mean(tpot_viol_when_masked) if tpot_viol_when_masked else None,
        "tpot_viol_when_masked_p50": median(tpot_viol_when_masked) if tpot_viol_when_masked else None,
        # delta 项贡献
        "abs_delta_slo_mean": mean(abs_delta_slo),
        "abs_delta_q_mean": mean(abs_delta_q),
        "pct_nonzero_delta_q": 100 * n_nonzero_q / n,
        "pct_nonzero_delta_slo": 100 * n_nonzero_slo / n,
        # ratio 钉边界
        "pct_at_ratio_max": 100 * n_at_max / n,
        "pct_at_ratio_min": 100 * n_at_min / n,
        "ratio_mean": mean(ratios),
        "ratio_p50": median(ratios),
    }


def main():
    print("=" * 84)
    print(f"P1.6a PID tpot 屏蔽逻辑审计 (M3.1 = {CFG})")
    print("=" * 84)

    for trace in TRACES:
        print(f"\n── trace = {trace} ──")
        per_seed = []
        for seed in SEEDS:
            ticks = load(trace, seed)
            s = summarise(ticks)
            if not s:
                print(f"  seed{seed}: no data")
                continue
            per_seed.append(s)
            print(f"\n  seed{seed} (n_warm_ticks = {s['n_warm']}):")
            print(f"    [屏蔽路径]    M2.1 clamp: {s['pct_clamp_M21']:5.1f}%  "
                  f"M2.4 sat: {s['pct_sat_M24']:5.1f}%  "
                  f"either: {s['pct_either']:5.1f}%  both: {s['pct_both']:5.1f}%")
            if s["err_tpot_raw_when_masked_mean"] is not None:
                print(f"    [屏蔽时 err_tpot_raw] mean={s['err_tpot_raw_when_masked_mean']:.3f}  "
                      f"p50={s['err_tpot_raw_when_masked_p50']:.3f}  "
                      f"p99={s['err_tpot_raw_when_masked_p99']:.3f}  "
                      f"(>0 = 违例率超 target 5%)")
                print(f"    [屏蔽时 tpot_viol_rate] mean={s['tpot_viol_when_masked_mean']*100:.1f}%  "
                      f"p50={s['tpot_viol_when_masked_p50']*100:.1f}%  "
                      f"(target=5%)")
            print(f"    [delta 项]   |delta_slo|={s['abs_delta_slo_mean']:.4f}  "
                  f"|delta_q|={s['abs_delta_q_mean']:.4f}  "
                  f"nonzero_q={s['pct_nonzero_delta_q']:.1f}%  "
                  f"nonzero_slo={s['pct_nonzero_delta_slo']:.1f}%")
            print(f"    [ratio 边界] at_max(0.8): {s['pct_at_ratio_max']:5.1f}%  "
                  f"at_min(0.05): {s['pct_at_ratio_min']:5.1f}%  "
                  f"mean={s['ratio_mean']:.3f}  p50={s['ratio_p50']:.3f}")

        if per_seed:
            keys = ["pct_clamp_M21", "pct_sat_M24", "pct_either",
                    "pct_at_ratio_max", "ratio_mean",
                    "abs_delta_slo_mean", "abs_delta_q_mean",
                    "pct_nonzero_delta_q"]
            print(f"\n  ⇒ {trace} 跨 seed mean:")
            for k in keys:
                vals = [s[k] for s in per_seed if k in s and s[k] is not None]
                if vals:
                    print(f"      {k:30s} = {mean(vals):.3f}")


if __name__ == "__main__":
    main()
