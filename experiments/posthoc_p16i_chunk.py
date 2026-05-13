"""P1.6i posthoc — chunk_tokens sweep on Azure trace
看 chunk 在 Azure mild-burst 上是否仍 pareto 单调,或者有不同最优点。
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_p16i")
CONFIGS = ["c2_tdm_m31", "c2_tdm_m31_1024", "c2_tdm_m31_2048", "c2_tdm_m31_4096"]
CHUNK_SIZES = {"c2_tdm_m31": 512, "c2_tdm_m31_1024": 1024,
                "c2_tdm_m31_2048": 2048, "c2_tdm_m31_4096": 4096}
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
        return None
    return 100 * sum(1 for r in valid
                      if r["ttft_ms"] < ttft_ms and r["tpot_ms_mean"] < tpot_ms) / len(valid)


def main():
    print(f"{'='*96}\nP1.6i chunk_tokens sweep on Azure\n{'='*96}")
    for trace in ("conv", "code"):
        print(f"\n── trace = {trace} ──")
        for ttft, tpot in TIERS:
            print(f"\n  tier ttft<{ttft}/tpot<{tpot}  (mean±std across 3 seeds)")
            print(f"    {'chunk':<10}{'meet_slo%':<20}")
            print("    " + "-" * 32)
            for cfg in CONFIGS:
                chunk = CHUNK_SIZES[cfg]
                per_seed = []
                for s in (0, 1, 2):
                    m = meet_slo(load_reqs(trace, s, cfg), ttft, tpot)
                    if m is not None:
                        per_seed.append(m)
                if not per_seed:
                    continue
                m_mean = mean(per_seed)
                m_std = stdev(per_seed) if len(per_seed) > 1 else 0.0
                print(f"    {chunk:<10}{m_mean:5.1f}±{m_std:.1f}")


if __name__ == "__main__":
    main()
