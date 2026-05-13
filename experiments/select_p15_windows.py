#!/usr/bin/env python3
"""P1.5 窗位选择脚本 —— 给跨窗实验确定窗位列表。

窗位选择标准(写入论文附录):
  对每条 trace,以 60s 窗、30s 步进枚举所有可能的回放窗位,以瞬时 QPS
  (窗内请求数 / 60) 排序,先剔除 QPS < p75 的"低压力窗"(M3.1 控制器
  在低压力下不发力,跑了也分不出方案),再以 QPS 降序贪心挑选 3 个互不
  重叠(start 间距 ≥ 120s)的窗位。

  这个标准:
  (a) 透明可复现 —— 无人工挑选,只有 p75 阈值 + ≥120s 间距两个旋钮
  (b) 与 thesis claim 对齐 —— M3.1 设计目标就是"压力 regime 下兜住 SLO"
  (c) 自动包含 azure_main 已用的 1860s (conv) / 570s (code) 锚点

用法:
  python3 select_p15_windows.py
"""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

DATA = Path("/vllm-workspace/Ascend-PD-TDM/data/azure_trace")
TRACES = [
    ("conv", DATA / "AzureLLMInferenceTrace_conv.csv"),
    ("code", DATA / "AzureLLMInferenceTrace_code.csv"),
]
WIN_S = 60.0
STEP_S = 30.0
MIN_GAP_S = 120.0
TOP_K = 3
QPS_PCTL_FLOOR = 75


def load_arrivals_s(path: Path) -> np.ndarray:
    """Return arrival times in seconds, normalized so first req at t=0."""
    ts = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            t = datetime.fromisoformat(row["TIMESTAMP"].split(".")[0])
            ts.append(t.timestamp())
    a = np.asarray(sorted(ts), dtype=np.float64)
    return a - a[0]


def window_qps(arrivals: np.ndarray, win: float, step: float) -> np.ndarray:
    """Return (start_s, qps) array for sliding windows."""
    span = arrivals[-1]
    starts = np.arange(0.0, span - win + 1, step)
    out = []
    for s in starts:
        n = ((arrivals >= s) & (arrivals < s + win)).sum()
        out.append((s, n / win))
    return np.asarray(out, dtype=np.float64)


def greedy_topk(table: np.ndarray, qps_floor: float, k: int, gap: float) -> list[tuple[float, float]]:
    """Greedy: rank by qps desc, drop windows < floor, enforce ≥gap between picks."""
    cand = table[table[:, 1] >= qps_floor]
    order = np.argsort(-cand[:, 1])  # desc
    picked: list[tuple[float, float]] = []
    for idx in order:
        s, q = cand[idx]
        if all(abs(s - ps) >= gap for ps, _ in picked):
            picked.append((float(s), float(q)))
            if len(picked) >= k:
                break
    # sort picks by start time for readability
    picked.sort(key=lambda x: x[0])
    return picked


def main() -> None:
    print(f"window={WIN_S:.0f}s step={STEP_S:.0f}s qps_floor=p{QPS_PCTL_FLOOR} gap≥{MIN_GAP_S:.0f}s top={TOP_K}")
    print()
    all_picks = {}
    for label, path in TRACES:
        arr = load_arrivals_s(path)
        tbl = window_qps(arr, WIN_S, STEP_S)
        floor = float(np.percentile(tbl[:, 1], QPS_PCTL_FLOOR))
        picks = greedy_topk(tbl, floor, TOP_K, MIN_GAP_S)
        anchor_old = {"conv": 1860.0, "code": 570.0}[label]
        anchor_q = float(tbl[np.isclose(tbl[:, 0], anchor_old), 1][0])
        print(f"=== {label} ===")
        print(f"  span={arr[-1]:.0f}s  N={len(arr)}  mean_qps={len(arr)/arr[-1]:.2f}  p{QPS_PCTL_FLOOR}_qps={floor:.2f}")
        print(f"  azure_main anchor: start={anchor_old:.0f}s qps={anchor_q:.2f}")
        print(f"  picked (start_s, qps):")
        for s, q in picks:
            tag = " <- matches azure_main anchor" if abs(s - anchor_old) < 1e-3 else ""
            print(f"    start={s:>5.0f}s  qps={q:.2f}{tag}")
        all_picks[label] = picks
        print()

    # emit shell-friendly TRACES array
    print("# shell snippet for runner:")
    for label, picks in all_picks.items():
        for i, (s, _q) in enumerate(picks):
            print(f'  "{label}_w{i}:$DATA/AzureLLMInferenceTrace_{label}.csv:{int(s)}"')


if __name__ == "__main__":
    main()
