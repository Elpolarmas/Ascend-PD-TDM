"""P1.6h posthoc — calibrated SLO 下重测 baseline,跨 config meet_slo% 对照

复用 P1.5 / azure_main 思路:从 req.jsonl 重算 meet_slo% (多 tier),报
M3.1 vs C3 / M1+chunk / C1 在 calibrated SLO 下的区分度。
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p16h")
CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
CONFIG_LABEL = {"c1_baseline": "C1", "c2_tdm_m1_chunk2048": "M1+chunk",
                 "c2_tdm_m31_2048": "M3.1", "c3_cp": "C3"}
TIERS = [(500, 100), (500, 150), (500, 200), (1000, 200)]


def load_reqs(trace: str, seed: int, cfg: str):
    p = ROOT / f"{trace}_w1_seed{seed}" / "tdm_trace" / f"qps_sweep_{cfg}_req.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def meet_slo(reqs, ttft_ms, tpot_ms):
    valid = [r for r in reqs
             if r.get("ttft_ms") is not None
             and r.get("tpot_ms_mean") is not None]
    if not valid:
        return None, 0
    n_meet = sum(1 for r in valid
                  if r["ttft_ms"] < ttft_ms and r["tpot_ms_mean"] < tpot_ms)
    return 100 * n_meet / len(valid), len(valid)


def main():
    print(f"{'='*96}\nP1.6h calibrated baseline (slo_tpot=200ms 跑全套)\n{'='*96}")
    for trace in ("conv", "code"):
        print(f"\n── trace = {trace} ──")
        for ttft, tpot in TIERS:
            print(f"\n  tier ttft<{ttft}/tpot<{tpot}")
            print(f"    {'config':<14}{'meet_slo% (mean±std)':<24}{'n_req'}")
            print("    " + "-" * 50)
            means_by_cfg = {}
            for cfg in CONFIGS:
                per_seed = []
                ns = []
                for s in (0, 1, 2):
                    reqs = load_reqs(trace, s, cfg)
                    m, n = meet_slo(reqs, ttft, tpot)
                    if m is not None:
                        per_seed.append(m)
                        ns.append(n)
                if not per_seed:
                    continue
                m_mean = mean(per_seed)
                m_std = stdev(per_seed) if len(per_seed) > 1 else 0.0
                means_by_cfg[cfg] = m_mean
                print(f"    {CONFIG_LABEL[cfg]:<14}"
                      f"{m_mean:5.1f}±{m_std:.1f}".ljust(24) +
                      f"{int(mean(ns))}")
            # Print key delta vs C3
            if "c2_tdm_m31_2048" in means_by_cfg and "c3_cp" in means_by_cfg:
                d = means_by_cfg["c2_tdm_m31_2048"] - means_by_cfg["c3_cp"]
                print(f"    Δ(M3.1 - C3) = {d:+.1f}pp")
            if "c2_tdm_m31_2048" in means_by_cfg and "c2_tdm_m1_chunk2048" in means_by_cfg:
                d = means_by_cfg["c2_tdm_m31_2048"] - means_by_cfg["c2_tdm_m1_chunk2048"]
                print(f"    Δ(M3.1 - M1+chunk) = {d:+.1f}pp")


if __name__ == "__main__":
    main()
