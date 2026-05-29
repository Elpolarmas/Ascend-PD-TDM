"""Aggregate Phase A micro-ideal results → derive Sarathi-style SLO grid.

Reads `results/phase_a_micro_ideal/{conv,code}_seed{0,1,2}/` output JSONs from
`run_phase_a_micro_ideal.sh` (concurrent=1, N=50 per (workload, seed)),
computes per-workload ideal_ttft / ideal_tpot at p50/p99/mean across all
samples (pooled over 3 seeds, drop first 5 warmup reqs per seed),
then derives SLO 4 档 = {5×, 10×, 15×, 25×} × ideal_p99 (Sarathi-Serve
protocol).

Outputs:
- stdout summary table
- results/phase_a_micro_ideal/slo_grid.json (machine-readable)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM")
OUT = ROOT / "results/phase_a_micro_ideal"
WORKLOADS = ("conv", "code")
SEEDS = (0, 1, 2)
WARMUP_REQS = 5  # arrival_time_s = request index, warmup_s=5 drops first 5
SLO_MULTIPLIERS = (5, 10, 15, 25)


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    s = sorted(xs)
    k = q * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _load_joined(seed_dir: Path) -> list[dict]:
    """Read the single qps0.0 output for c1_baseline."""
    cands = list(seed_dir.glob("c1_baseline_qps*.json"))
    if not cands:
        raise FileNotFoundError(f"no c1_baseline output in {seed_dir}")
    if len(cands) > 1:
        raise RuntimeError(f"expected 1 c1_baseline file in {seed_dir}, "
                           f"found {len(cands)}: {cands}")
    payload = json.loads(cands[0].read_text())
    return payload.get("joined") or []


def _per_workload_ideal(workload: str) -> dict:
    """Pool TTFT/TPOT across 3 seeds, drop warmup, return stats + SLO grid."""
    pooled_ttft: list[float] = []
    pooled_tpot: list[float] = []
    pooled_e2e: list[float] = []
    pooled_prompt_tokens: list[int] = []
    pooled_output_tokens: list[int] = []
    n_total = 0
    n_kept = 0
    for seed in SEEDS:
        d = OUT / f"{workload}_seed{seed}"
        joined = _load_joined(d)
        # arrival_time_s = request index; drop first WARMUP_REQS
        for j in joined:
            n_total += 1
            if j.get("status") != 200:
                continue
            if j.get("arrival_time_s", 0.0) < WARMUP_REQS:
                continue
            ttft = j.get("ttft_ms")
            tpot = j.get("tpot_ms_mean")
            if ttft is None or tpot is None:
                continue
            pooled_ttft.append(float(ttft))
            pooled_tpot.append(float(tpot))
            e2e = j.get("e2e_ms")
            if e2e is not None:
                pooled_e2e.append(float(e2e))
            pt = j.get("prompt_tokens")
            ot = j.get("output_tokens")
            if pt is not None:
                pooled_prompt_tokens.append(int(pt))
            if ot is not None:
                pooled_output_tokens.append(int(ot))
            n_kept += 1

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(statistics.fmean(xs), 2),
            "p50": round(_percentile(xs, 0.50), 2),
            "p90": round(_percentile(xs, 0.90), 2),
            "p99": round(_percentile(xs, 0.99), 2),
            "min": round(min(xs), 2),
            "max": round(max(xs), 2),
        }

    ttft_stats = stats(pooled_ttft)
    tpot_stats = stats(pooled_tpot)
    ideal_ttft_p99 = ttft_stats.get("p99")
    ideal_tpot_p99 = tpot_stats.get("p99")

    slo_grid = {}
    if ideal_ttft_p99 is not None and ideal_tpot_p99 is not None:
        for m in SLO_MULTIPLIERS:
            slo_grid[f"{m}x"] = {
                "slo_ttft_ms": round(m * ideal_ttft_p99, 1),
                "slo_tpot_ms": round(m * ideal_tpot_p99, 1),
            }

    return {
        "workload": workload,
        "n_seeds": len(SEEDS),
        "n_total_records": n_total,
        "n_kept_after_warmup_and_status": n_kept,
        "warmup_reqs_dropped_per_seed": WARMUP_REQS,
        "prompt_tokens": stats([float(x) for x in pooled_prompt_tokens]),
        "output_tokens": stats([float(x) for x in pooled_output_tokens]),
        "ideal_ttft_ms": ttft_stats,
        "ideal_tpot_ms_mean": tpot_stats,
        "ideal_e2e_ms": stats(pooled_e2e),
        "slo_grid": slo_grid,
    }


def main() -> int:
    if not OUT.exists():
        print(f"ERROR: {OUT} not found — run run_phase_a_micro_ideal.sh first",
              file=sys.stderr)
        return 1
    result = {
        "model": "Qwen3-8B",
        "concurrent": 1,
        "method": "sequential micro-benchmark, Sarathi-Serve protocol",
        "slo_multipliers": list(SLO_MULTIPLIERS),
        "workloads": {w: _per_workload_ideal(w) for w in WORKLOADS},
    }
    out_path = OUT / "slo_grid.json"
    out_path.write_text(json.dumps(result, indent=2))

    # human summary
    print(f"# Phase A micro-ideal — Qwen3-8B, concurrent=1, "
          f"N=50 × 3 seeds per workload")
    print(f"# Output: {out_path}\n")
    for w in WORKLOADS:
        r = result["workloads"][w]
        print(f"## {w}")
        print(f"  records kept: {r['n_kept_after_warmup_and_status']} / "
              f"{r['n_total_records']}  "
              f"(dropped first {WARMUP_REQS} req per seed + non-200)")
        ttft = r["ideal_ttft_ms"]
        tpot = r["ideal_tpot_ms_mean"]
        pt = r["prompt_tokens"]
        ot = r["output_tokens"]
        if ttft.get("n", 0) > 0:
            print(f"  prompt_tokens: mean={pt.get('mean')} "
                  f"p50={pt.get('p50')} p99={pt.get('p99')}")
            print(f"  output_tokens: mean={ot.get('mean')} "
                  f"p50={ot.get('p50')} p99={ot.get('p99')}")
            print(f"  ideal_ttft_ms: mean={ttft['mean']} "
                  f"p50={ttft['p50']} p99={ttft['p99']}")
            print(f"  ideal_tpot_ms: mean={tpot['mean']} "
                  f"p50={tpot['p50']} p99={tpot['p99']}")
            print(f"  SLO grid (5×/10×/15×/25× × p99):")
            for m, g in r["slo_grid"].items():
                print(f"    {m:>4}: ttft<{g['slo_ttft_ms']}ms  "
                      f"tpot<{g['slo_tpot_ms']}ms")
        else:
            print("  NO DATA — check seed dirs")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
