"""Post-hoc SLO grid analysis on long-prompt sweep.

读已存盘的 12 个 per-config × per-qps JSON，按 lib/metrics.aggregate_window 同款
判定（ttft<SLO_ttft AND tpot_mean<SLO_tpot），在 SLO grid 上重算 meet_slo%。

动机：方案 A "采样模块离线校准固定 SLO 档位" —— 不重跑实验，只换 SLO 档位重算
现有数据，看 4 个 config 的 ranking 在不同 SLO 档下是否换序。

聚焦 qps=16（mid-intensity sweet spot），qps=8 太松无区分度，qps=32 已 saturated。
"""
from __future__ import annotations

import json
from pathlib import Path

SWEEP_DIR = Path("/vllm-workspace/Ascend-PD-TDM/results/long_prompt_sweep")

CONFIGS = ["c1_baseline", "c2_tdm", "c2_tdm_m27", "c3_cp"]
QPS_POINTS = [8.0, 16.0, 32.0]

# SLO grid：
#   ttft 档位：500 (interactive) / 1000 / 1500 / 2000 / 3000 / 5000 (doc-analysis)
#   tpot 档位：50 (interactive) / 100 / 150 / 200 / 300 / 500 (background)
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load_joined(config: str, qps: float) -> tuple[list[dict], float, float]:
    """返回 (joined records, warmup_s, window_end_s)。"""
    path = SWEEP_DIR / f"{config}_qps{qps}.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    warmup_s = s["warmup_s"]
    duration_s = s["duration_s"]
    window_end_s = warmup_s + duration_s
    return d["joined"], warmup_s, window_end_s


def meet_slo_count(joined: list[dict], warmup_s: float, window_end_s: float,
                   slo_ttft_ms: float, slo_tpot_ms: float) -> tuple[int, int]:
    """返回 (n_meet, n_matched)，与 lib/metrics.aggregate_window 判定一致。"""
    in_win = [j for j in joined
              if warmup_s <= j["arrival_time_s"] < window_end_s
              and j["status"] == 200 and j.get("matched")]
    n_matched = len(in_win)
    n_meet = sum(
        1 for j in in_win
        if j.get("ttft_ms") is not None
        and j.get("tpot_ms_mean") is not None
        and j["ttft_ms"] < slo_ttft_ms
        and j["tpot_ms_mean"] < slo_tpot_ms
        and j.get("output_tokens")
    )
    return n_meet, n_matched


def grid_for_qps(qps: float) -> dict[str, dict[tuple[int, int], float]]:
    """每个 config 在该 qps 下的 (ttft,tpot) → meet_slo% 网格。"""
    out: dict[str, dict[tuple[int, int], float]] = {}
    for cfg in CONFIGS:
        joined, w, we = load_joined(cfg, qps)
        cell: dict[tuple[int, int], float] = {}
        for tt in TTFT_GRID:
            for tp in TPOT_GRID:
                meet, total = meet_slo_count(joined, w, we, tt, tp)
                cell[(tt, tp)] = 100.0 * meet / total if total else 0.0
        out[cfg] = cell
    return out


def print_grid_for_qps(qps: float) -> None:
    print(f"\n{'='*78}\n=== qps={qps}  meet_slo% grid（行=tpot ms, 列=ttft ms）"
          f"\n{'='*78}")
    grids = grid_for_qps(qps)
    # 一个 config 一张 grid
    for cfg in CONFIGS:
        print(f"\n--- {cfg} ---")
        # header
        print(f"  tpot \\ ttft  | " + "  ".join(f"{tt:>6d}" for tt in TTFT_GRID))
        print(f"  {'-'*13} | " + "  ".join("-"*6 for _ in TTFT_GRID))
        for tp in TPOT_GRID:
            row = "  ".join(f"{grids[cfg][(tt, tp)]:>5.1f}%" for tt in TTFT_GRID)
            print(f"  tpot<{tp:<5d}     | {row}")


def print_ranking_table(qps: float) -> None:
    """对每个 (ttft, tpot) 格子，报 4-config 排名 + 第一名 SLO%。"""
    print(f"\n{'='*78}\n=== qps={qps}  ranking by SLO grid（█ = winner）"
          f"\n{'='*78}")
    grids = grid_for_qps(qps)
    print(f"  tpot \\ ttft  | " + "  ".join(f"{tt:>10d}" for tt in TTFT_GRID))
    print(f"  {'-'*13} | " + "  ".join("-"*10 for _ in TTFT_GRID))
    for tp in TPOT_GRID:
        cells = []
        for tt in TTFT_GRID:
            scores = {cfg: grids[cfg][(tt, tp)] for cfg in CONFIGS}
            winner = max(scores, key=scores.get)
            ws = scores[winner]
            tag = {"c1_baseline": "C1", "c2_tdm": "C2",
                   "c2_tdm_m27": "M27", "c3_cp": "C3"}[winner]
            cells.append(f"{tag:>3s}{ws:>6.1f}%")
        print(f"  tpot<{tp:<5d}     | " + "  ".join(cells))


def print_focus_table_qps16() -> None:
    """qps=16 mid-intensity 的精简表（每档对比 4 config）。"""
    print(f"\n{'='*78}\n=== qps=16 (mid-intensity sweet spot) — 4-config 各 SLO 档对照"
          f"\n{'='*78}")
    qps = 16.0
    grids = grid_for_qps(qps)
    # 选几个有代表性的档：strict / long-context / loose
    profiles = [
        ("Strict (interactive)",  500, 50),
        ("Mid-1",                1000, 100),
        ("Long-context",         1500, 200),
        ("Mid-2",                2000, 200),
        ("Loose (doc)",          5000, 500),
    ]
    print(f"{'profile':<26s} {'ttft':>6s} {'tpot':>6s} | "
          f"{'C1':>7s} {'C2':>7s} {'M2.7':>7s} {'C3':>7s}  winner")
    print("-" * 78)
    for name, tt, tp in profiles:
        c1 = grids["c1_baseline"][(tt, tp)]
        c2 = grids["c2_tdm"][(tt, tp)]
        m27 = grids["c2_tdm_m27"][(tt, tp)]
        c3 = grids["c3_cp"][(tt, tp)]
        scores = {"C1": c1, "C2": c2, "M2.7": m27, "C3": c3}
        winner = max(scores, key=scores.get)
        print(f"{name:<26s} {tt:>6d} {tp:>6d} | "
              f"{c1:>6.1f}% {c2:>6.1f}% {m27:>6.1f}% {c3:>6.1f}%  "
              f"<- {winner} ({scores[winner]:.1f}%)")


def main():
    print_focus_table_qps16()
    for qps in QPS_POINTS:
        print_ranking_table(qps)
    # 详细的每 config full grid 留作 verbose 输出
    print()
    print_grid_for_qps(16.0)


if __name__ == "__main__":
    main()
