"""P1.6f posthoc — target_violation_rate sweep
对比 PID 在不同 target_violation_rate 下的行为。
判定:err_tpot_eff 非零率 > 20% 且 ratio 钉 max < 80% → 双输入复活 → 走向 α
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p16f")
CFG = "c2_tdm_m31_2048"
RATIO_MAX = 0.8
TARGETS = ["005", "010", "020", "030", "050"]
TARGETS_DISPLAY = ["0.05", "0.10", "0.20", "0.30", "0.50"]


def load(target_tag: str, trace: str, seed: int):
    p = ROOT / f"tgt{target_tag}" / f"{trace}_w1_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def summarise(ticks):
    warm = [t for t in ticks if not t["cold_start"]]
    if not warm:
        return {}
    n = len(warm)
    return {
        "pct_at_max": 100 * sum(1 for t in warm if t["ratio_after"] >= RATIO_MAX - 1e-6) / n,
        "pct_sat": 100 * sum(1 for t in warm if t["tpot_saturated"]) / n,
        "pct_eff_tpot_nonzero": 100 * sum(1 for t in warm if abs(t["err_tpot_effective"]) > 1e-9) / n,
        "ratio_mean": mean(t["ratio_after"] for t in warm),
        "tpot_viol_mean": mean(t["tpot_viol_rate"] for t in warm),
        "ttft_viol_mean": mean(t["ttft_viol_rate"] for t in warm),
    }


def main():
    print(f"{'='*96}\nP1.6f target_violation_rate sweep (slo_tpot=200ms)\n{'='*96}")
    for trace in ("conv", "code"):
        print(f"\n── trace = {trace} ──")
        print(f"{'指标':<30}" + "".join(f"{f'tgt={d}':>14}" for d in TARGETS_DISPLAY))
        print("-" * 96)
        rows = []
        for tag in TARGETS:
            seed_rows = [summarise(load(tag, trace, s)) for s in (0, 1, 2)]
            seed_rows = [r for r in seed_rows if r]
            if seed_rows:
                rows.append({k: mean(r[k] for r in seed_rows) for k in seed_rows[0]})
            else:
                rows.append(None)
        for label, key, prec, suf in [
            ("ratio 钉 max (%)", "pct_at_max", 1, "%"),
            ("ratio mean", "ratio_mean", 3, ""),
            ("tpot_sat 屏蔽 (%)", "pct_sat", 1, "%"),
            ("err_tpot_eff 非零 (%)", "pct_eff_tpot_nonzero", 1, "%"),
            ("tpot_viol_rate mean", "tpot_viol_mean", 2, ""),
        ]:
            line = f"{label:<30}"
            for r in rows:
                v = f"{r[key]:.{prec}f}{suf}" if r else "—"
                line += f"{v:>14}"
            print(line)
        print()
        for d, r in zip(TARGETS_DISPLAY, rows):
            if not r:
                continue
            alive = r["pct_at_max"] < 80 and r["pct_eff_tpot_nonzero"] > 20
            print(f"  target={d}: {'✅ 双输入复活' if alive else '❌ 仍单输入'}  "
                  f"(at_max={r['pct_at_max']:.1f}%  eff_nonzero={r['pct_eff_tpot_nonzero']:.1f}%)")


if __name__ == "__main__":
    main()
