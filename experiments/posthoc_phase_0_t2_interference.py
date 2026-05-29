"""T2 Post-hoc B (D-012, 2026-05-20): Interference shifting defining figure 数据版。

基于 thesis_framing.md L1 framework:不同 PD 调度策略 = 干扰转移操作。
本脚本从 azure_p15 已有 telemetry 算 mean / tail interference 坐标,
出 paper S5.3 defining figure 数据。

干扰算法(thesis_framing L106-107 / L171-172)
----------------------------------------------
- **mean interference 维度** = TTFT 相关(prefill queueing & 处理时间)
- **tail interference 维度** = TPOT_p99(最差 decode 延迟)

每个 (config × workload) 在 (TTFT_mean, TPOT_p99) 平面上是一个点:
- C1 (no scheduling):   预期高 TTFT + 高 TPOT_p99(双输,右上角)
- C3 (chunked prefill): 预期高 TTFT + 低 TPOT_p99(右下,trade mean for tail)
- M3.1 (phase-pure):    预期低 TTFT + 高 TPOT_p99(左上,trade tail for mean) ← D-012 thesis
- (c4_pd 不在 azure_p15;留 T8 / 概念上应在左下角双赢但成本高)

输出
----
- results/azure_p15/interference_coords.json
- stdout 表格
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/vllm-workspace/Ascend-PD-TDM")
P15 = ROOT / "results/azure_p15"
CONFIGS = ["c1_baseline", "c2_tdm_m1_chunk2048", "c2_tdm_m31_2048", "c3_cp"]
WORKLOADS = ["conv_w1", "code_w1"]  # peak window
SEEDS = [0, 1, 2]


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    s = sorted(xs)
    k = q * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 2),
        "p50": round(_percentile(xs, 0.50), 2),
        "p90": round(_percentile(xs, 0.90), 2),
        "p99": round(_percentile(xs, 0.99), 2),
    }


def _load_req_jsonl(cfg: str, workload: str, seed: int) -> list[dict]:
    f = P15 / f"{workload}_seed{seed}/tdm_trace/qps_sweep_{cfg}_req.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _coord_for(cfg: str, workload: str) -> dict:
    """Pool ttft (X) / tpot_p99 (Y) per req across 3 seeds for one (cfg, workload)."""
    ttfts: list[float] = []          # X 轴 mean interference 维度
    tpot_p99s: list[float] = []      # Y 轴 tail interference 维度
    tpot_means: list[float] = []     # 上下文(per-req decode iter 平均)
    for seed in SEEDS:
        reqs = _load_req_jsonl(cfg, workload, seed)
        for r in reqs:
            tt = r.get("ttft_ms")
            tp = r.get("tpot_ms_p99")
            tm = r.get("tpot_ms_mean")
            if tt is not None and tt > 0:
                ttfts.append(float(tt))
            if tp is not None and tp > 0:
                tpot_p99s.append(float(tp))
            if tm is not None and tm > 0:
                tpot_means.append(float(tm))
    return {
        "config": cfg,
        "workload": workload,
        "mean_interference_axis_ttft": _stats(ttfts),       # X 轴
        "tail_interference_axis_tpot_p99": _stats(tpot_p99s),  # Y 轴
        "tpot_mean_context": _stats(tpot_means),            # 上下文
    }


def main() -> int:
    if not P15.exists():
        print(f"ERROR: {P15} not found", file=sys.stderr)
        return 1
    coords = []
    for w in WORKLOADS:
        for cfg in CONFIGS:
            coords.append(_coord_for(cfg, w))

    # human-readable table
    print("# T2 defining figure — interference shifting coordinates")
    print(f"# Source: azure_p15/{{conv_w1, code_w1}} × 3 seed × 4 config")
    print(f"# X axis = TTFT_mean (mean interference)")
    print(f"# Y axis = TPOT_p99 mean (tail interference)")
    print()
    print(f"{'workload':<10} {'config':<22} "
          f"{'X_mean':>10} {'X_p99':>10} "
          f"{'Y_mean':>10} {'Y_p99':>10} "
          f"{'tpot_ctx':>10}")
    print("-" * 92)
    for r in coords:
        mX = r["mean_interference_axis_ttft"]
        tY = r["tail_interference_axis_tpot_p99"]
        tm = r["tpot_mean_context"]
        if mX["n"] == 0:
            print(f"{r['workload']:<10} {r['config']:<22} NO DATA")
            continue
        print(f"{r['workload']:<10} {r['config']:<22} "
              f"{mX['mean']:>10} {mX['p99']:>10} "
              f"{tY['mean']:>10} {tY['p99']:>10} "
              f"{tm['mean']:>10}")

    out = P15 / "interference_coords.json"
    out.write_text(json.dumps({
        "source": "azure_p15 req.jsonl × 4 cfg × 2 workload × 3 seed",
        "x_axis": "mean_interference = global mean(per-req ttft_ms)",
        "y_axis": "tail_interference = global mean(per-req tpot_ms_p99)",
        "framework": "thesis_framing L1: PD scheduling = interference shifting operation",
        "coords": coords,
    }, indent=2))
    print(f"\n[done] wrote {out}")

    # 定性观察:每个 quadrant 对应一个 paradigm 位置
    print("\n## Interference shifting quadrant placement:")
    for w in WORKLOADS:
        rs = [r for r in coords if r["workload"] == w]
        valid = [r for r in rs if r["mean_interference_axis_ttft"]["n"] > 0]
        if not valid:
            continue
        x_med = sorted(r["mean_interference_axis_ttft"]["mean"] for r in valid)[len(valid)//2]
        y_med = sorted(r["tail_interference_axis_tpot_p99"]["mean"] for r in valid)[len(valid)//2]
        print(f"\n### {w}  (median anchor: X={x_med:.1f}  Y={y_med:.1f})")
        for r in valid:
            x = r["mean_interference_axis_ttft"]["mean"]
            y = r["tail_interference_axis_tpot_p99"]["mean"]
            q_x = "low" if x < x_med else "high"
            q_y = "low" if y < y_med else "high"
            print(f"  {r['config']:<22}: X={x:>7.1f} ({q_x})  Y={y:>7.1f} ({q_y})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
