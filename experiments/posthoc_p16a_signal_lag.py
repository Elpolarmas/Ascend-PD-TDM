"""P1.6a 扩展 — 信号采集滞后量化

回答用户问题:ttft / tpot 信息是否被及时采集?
- ttft / tpot 滑窗 warm 起来要多久?
- warm 后窗口是否一直有数据流入,还是 saturated 下停滞?
- 启动 ramp 期间(0~10s)是不是因为信号还没到 PID 才推到上界?
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

RESULTS = Path("/vllm-workspace/Ascend-PD-TDM/results/azure_main")
TRACES = ["conv", "code"]
SEEDS = [0, 1, 2]
CFG = "c2_tdm_m31_2048"
MIN_SAMPLES = 16

def load(trace: str, seed: int) -> list[dict]:
    p = RESULTS / f"{trace}_seed{seed}" / "tdm_trace" / f"qps_sweep_{CFG}_ctrl.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def analyse(ticks: list[dict]) -> dict:
    if not ticks:
        return {}
    t0 = ticks[0]["ts_ms"]
    # warm 时刻(n_window >= min_samples 首次)
    warm_ttft_at_ms = None
    warm_tpot_at_ms = None
    for t in ticks:
        if warm_ttft_at_ms is None and t["n_ttft_window"] >= MIN_SAMPLES:
            warm_ttft_at_ms = t["ts_ms"] - t0
        if warm_tpot_at_ms is None and t["n_tpot_window"] >= MIN_SAMPLES:
            warm_tpot_at_ms = t["ts_ms"] - t0
        if warm_ttft_at_ms and warm_tpot_at_ms:
            break

    # 启动期 ratio 推升 vs 信号 warm 时刻
    first_ratio_at_max_ms = None
    for t in ticks:
        if not t["cold_start"] and t["ratio_after"] >= 0.8 - 1e-6:
            first_ratio_at_max_ms = t["ts_ms"] - t0
            break

    # 窗口 size 时间序列:取 5 个关键采样点(0%/25%/50%/75%/100%)
    n = len(ticks)
    sample_idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    samples = [(ticks[i]["ts_ms"] - t0, ticks[i]["n_ttft_window"],
                ticks[i]["n_tpot_window"], ticks[i]["ratio_after"])
               for i in sample_idxs]

    # 窗口最终大小
    last = ticks[-1]
    return {
        "wall_total_s": (ticks[-1]["ts_ms"] - t0) / 1000,
        "warm_ttft_s": warm_ttft_at_ms / 1000 if warm_ttft_at_ms else None,
        "warm_tpot_s": warm_tpot_at_ms / 1000 if warm_tpot_at_ms else None,
        "first_ratio_at_max_s": (first_ratio_at_max_ms / 1000
                                  if first_ratio_at_max_ms else None),
        "n_ttft_final": last["n_ttft_window"],
        "n_tpot_final": last["n_tpot_window"],
        "samples": samples,
    }


def main():
    print("=" * 88)
    print(f"P1.6a 扩展 — 信号采集滞后量化 (M3.1 = {CFG}, min_samples = {MIN_SAMPLES})")
    print("=" * 88)
    print("\n核心问题:ttft / tpot 信号几时到达 PID?ratio 推到上界发生在 warm 之前还是之后?\n")

    for trace in TRACES:
        print(f"\n── trace = {trace} ──")
        warms_ttft = []
        warms_tpot = []
        ramps = []
        for seed in SEEDS:
            r = analyse(load(trace, seed))
            if not r:
                continue
            print(f"\n  seed{seed}  (wall = {r['wall_total_s']:.1f}s)")
            print(f"    ttft window warm @ {r['warm_ttft_s']}s "
                  f"(n_final = {r['n_ttft_final']})")
            print(f"    tpot window warm @ {r['warm_tpot_s']}s "
                  f"(n_final = {r['n_tpot_final']})")
            print(f"    ratio 首次到 ratio_max=0.8 @ "
                  f"{r['first_ratio_at_max_s']}s")
            print(f"    窗口大小演进(t, n_ttft, n_tpot, ratio):")
            for t_s_ms, n_t, n_p, ratio in r["samples"]:
                print(f"        @ {t_s_ms/1000:5.1f}s  "
                      f"n_ttft={n_t:3d}  n_tpot={n_p:3d}  ratio={ratio:.3f}")
            if r["warm_ttft_s"]:
                warms_ttft.append(r["warm_ttft_s"])
            if r["warm_tpot_s"]:
                warms_tpot.append(r["warm_tpot_s"])
            if r["first_ratio_at_max_s"]:
                ramps.append(r["first_ratio_at_max_s"])
        if warms_ttft and ramps:
            print(f"\n  ⇒ {trace} 跨 seed mean:")
            print(f"      warm_ttft = {mean(warms_ttft):.1f}s "
                  f"| warm_tpot = {mean(warms_tpot):.1f}s "
                  f"| ratio→max = {mean(ramps):.1f}s")
            if mean(ramps) < min(mean(warms_ttft), mean(warms_tpot)):
                print(f"      🚨 ratio 推到上界发生在两个窗口都 warm **之前**——"
                      f"PID 在没有反馈信号的情况下就把 ratio 推满了")
            elif mean(ramps) < mean(warms_tpot):
                print(f"      ⚠️ ratio 推到上界发生在 tpot warm **之前**——"
                      f"PID 看不到 tpot 反馈就把 ratio 推满了")


if __name__ == "__main__":
    main()
