"""P1.6e posthoc — SLO target sweep 跨档对照

对比 PID 在不同 slo_tpot_ms 下的行为:
  - tpot50 = azure_p16b(F1+F2 已修反馈链)
  - tpot150 / tpot200 / tpot300 = azure_p16e/tpot{N}/

每档:M3.1 跨 2 windows × 3 seeds,ctrl.jsonl 提:
  - ratio 钉 max 占比、ratio_mean
  - tpot_sat 屏蔽占比
  - err_tpot_eff 非零率(双输入复活)
  - delta_slo / delta_q 分布

PASS:某档下 conv pct_at_max < 80% 且 err_tpot_eff 非零率 > 20% → 救回闭环反馈
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
CFG = "c2_tdm_m31_2048"
RATIO_MAX = 0.8
SLOS = [50, 150, 200, 300]


def base_and_layout(slo: int):
    """Returns (base_dir, path_pattern) for the given SLO档.
    50 -> azure_p16b/{trace}_w1_seed{seed}/...
    N  -> azure_p16e/tpot{N}/{trace}_w1_seed{seed}/...
    """
    if slo == 50:
        return ROOT / "azure_p16b", "{trace}_w1_seed{seed}"
    return ROOT / "azure_p16e" / f"tpot{slo}", "{trace}_w1_seed{seed}"


def load(slo: int, trace: str, seed: int) -> list[dict]:
    base, pat = base_and_layout(slo)
    sub = pat.format(trace=trace, seed=seed)
    p = base / sub / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def summarise(ticks: list[dict]) -> dict:
    if not ticks:
        return {}
    warm = [t for t in ticks if not t["cold_start"]]
    if not warm:
        return {"n": 0}
    n = len(warm)
    return {
        "n": n,
        "pct_at_max": 100 * sum(1 for t in warm if t["ratio_after"] >= RATIO_MAX - 1e-6) / n,
        "pct_sat": 100 * sum(1 for t in warm if t["tpot_saturated"]) / n,
        "pct_eff_tpot_nonzero": 100 * sum(1 for t in warm if abs(t["err_tpot_effective"]) > 1e-9) / n,
        "ratio_mean": mean(t["ratio_after"] for t in warm),
        "abs_delta_slo_mean": mean(abs(t["delta_slo"]) for t in warm),
        "tpot_viol_mean": mean(t["tpot_viol_rate"] for t in warm),
        "ttft_viol_mean": mean(t["ttft_viol_rate"] for t in warm),
    }


def aggregate(slo: int, trace: str) -> dict:
    rows = [summarise(load(slo, trace, s)) for s in (0, 1, 2)]
    rows = [r for r in rows if r.get("n", 0) > 0]
    if not rows:
        return {}
    keys = ["pct_at_max", "pct_sat", "pct_eff_tpot_nonzero", "ratio_mean",
            "abs_delta_slo_mean", "tpot_viol_mean", "ttft_viol_mean"]
    return {k: mean(r[k] for r in rows) for k in keys} | {"n_seeds": len(rows)}


def main():
    print("=" * 96)
    print("P1.6e SLO target sweep posthoc (M3.1 跨档对照,跨 seed mean)")
    print("=" * 96)

    for trace in ("conv", "code"):
        print(f"\n── trace = {trace} ──")
        header = f"{'指标':<30}" + "".join(f"{f'tpot={s}':>14}" for s in SLOS)
        print(header)
        print("-" * 96)

        rows = []
        for slo in SLOS:
            agg = aggregate(slo, trace)
            rows.append((slo, agg))

        labels = [
            ("ratio 钉 max (%)", "pct_at_max", 1, "%"),
            ("ratio mean", "ratio_mean", 3, ""),
            ("tpot_sat 屏蔽 (%)", "pct_sat", 1, "%"),
            ("err_tpot_eff 非零 (%)", "pct_eff_tpot_nonzero", 1, "%"),
            ("tpot_viol_rate mean", "tpot_viol_mean", 2, ""),
            ("ttft_viol_rate mean", "ttft_viol_mean", 2, ""),
            ("|delta_slo| mean", "abs_delta_slo_mean", 4, ""),
        ]
        for label, key, prec, suffix in labels:
            line = f"{label:<30}"
            for slo, agg in rows:
                if agg and key in agg:
                    v = f"{agg[key]:.{prec}f}{suffix}"
                else:
                    v = "—"
                line += f"{v:>14}"
            print(line)

        # 判定:哪档 PID 双输入复活?
        print()
        candidates = [(s, a) for s, a in rows if a]
        for slo, a in candidates:
            pid_alive = (a.get("pct_at_max", 100) < 80
                          and a.get("pct_eff_tpot_nonzero", 0) > 20)
            seeds = a.get("n_seeds", 0)
            mark = "✅ PID 双输入活" if pid_alive else "❌ 仍单输入"
            print(f"  tpot={slo}ms ({seeds} seeds): {mark}")


if __name__ == "__main__":
    main()
