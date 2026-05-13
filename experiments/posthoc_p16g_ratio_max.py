"""P1.6g posthoc — ratio_max sweep
看 ratio 是否真在新边界停下来,以及不同 ratio_max 下 meet_slo% 怎么变。
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p16g")
CFG = "c2_tdm_m31_2048"
RATIO_MAXES = ["05", "07", "08", "09", "095", "10"]
RATIO_MAXES_VAL = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def load_ctrl(tag: str, trace: str, seed: int):
    p = ROOT / f"rmax{tag}" / f"{trace}_w1_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load_summary(tag: str, trace: str, seed: int):
    p = ROOT / f"rmax{tag}" / f"{trace}_w1_seed{seed}" / "qps_sweep_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def summarise_ctrl(ticks, rmax_val):
    warm = [t for t in ticks if not t["cold_start"]]
    if not warm:
        return {}
    n = len(warm)
    return {
        "pct_at_max": 100 * sum(1 for t in warm if t["ratio_after"] >= rmax_val - 1e-6) / n,
        "ratio_mean": mean(t["ratio_after"] for t in warm),
        "pct_eff_tpot_nonzero": 100 * sum(1 for t in warm if abs(t["err_tpot_effective"]) > 1e-9) / n,
        "tpot_viol_mean": mean(t["tpot_viol_rate"] for t in warm),
    }


def get_meet_slo(summary, tier_ttft, tier_tpot):
    """Approx: count req with ttft < tier_ttft AND tpot < tier_tpot from summary"""
    if not summary:
        return None
    for r in summary.get("c2_tdm_m31_2048", []):
        win = r.get("window") or {}
        n_ok = win.get("n_ok") or 0
        if n_ok == 0:
            return None
        # P1.6 multi-tier replay needs req.jsonl; here just report stored value
        n_meet = win.get("n_meet_slo") or 0
        return 100 * n_meet / n_ok
    return None


def main():
    print(f"{'='*96}\nP1.6g ratio_max sweep (slo_tpot=200ms)\n{'='*96}")
    for trace in ("conv", "code"):
        print(f"\n── trace = {trace} ──")
        print(f"{'指标':<30}" + "".join(f"{f'rmax={v}':>12}" for v in RATIO_MAXES_VAL))
        print("-" * 96)
        rows = []
        for tag, v in zip(RATIO_MAXES, RATIO_MAXES_VAL):
            srs = [summarise_ctrl(load_ctrl(tag, trace, s), v) for s in (0, 1, 2)]
            srs = [r for r in srs if r]
            if srs:
                rows.append({k: mean(r[k] for r in srs) for k in srs[0]})
            else:
                rows.append(None)
        for label, key, prec, suf in [
            ("ratio 钉 max (%)", "pct_at_max", 1, "%"),
            ("ratio mean", "ratio_mean", 3, ""),
            ("err_tpot_eff 非零 (%)", "pct_eff_tpot_nonzero", 1, "%"),
            ("tpot_viol_rate mean", "tpot_viol_mean", 2, ""),
        ]:
            line = f"{label:<30}"
            for r in rows:
                v = f"{r[key]:.{prec}f}{suf}" if r else "—"
                line += f"{v:>12}"
            print(line)
        print()
        for v, r in zip(RATIO_MAXES_VAL, rows):
            if not r:
                continue
            ceiling_real = r["pct_at_max"] > 50  # ratio 还是在堆上界
            print(f"  ratio_max={v}: at_max={r['pct_at_max']:.1f}%, mean={r['ratio_mean']:.3f}  "
                  f"{'(真天花板)' if ceiling_real else '(未顶到顶)'}")


if __name__ == "__main__":
    main()
