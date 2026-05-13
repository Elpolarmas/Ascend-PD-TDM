"""Post-hoc SLO grid for the M3.1 chunk-size scan (mixed regime, qps=16).

Reads the joined per-request records from:
  results/m31_mixed_qps16/      (c2_tdm_m27, c2_tdm_m31, c3_cp)
  results/m31_chunksize_scan/   (c2_tdm_m31_1024, _2048, _4096)

Prints a meet_slo% grid (rows=tpot, cols=ttft) for each config so we can see
WHERE on the SLO surface chunk size buys vs costs. Strict-SLO single-point
deltas are <= 1pp = noise; the grid shows whether M3.1 actually moves the
discrimination band as task doc §1.4 claims.
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCES = {
    "c2_tdm_m27": Path("/vllm-workspace/Ascend-PD-TDM/results/m31_mixed_qps16"),
    "c2_tdm_m31_512": Path("/vllm-workspace/Ascend-PD-TDM/results/m31_mixed_qps16"),
    "c3_cp": Path("/vllm-workspace/Ascend-PD-TDM/results/m31_mixed_qps16"),
    "c2_tdm_m31_1024":
        Path("/vllm-workspace/Ascend-PD-TDM/results/m31_chunksize_scan"),
    "c2_tdm_m31_2048":
        Path("/vllm-workspace/Ascend-PD-TDM/results/m31_chunksize_scan"),
    "c2_tdm_m31_4096":
        Path("/vllm-workspace/Ascend-PD-TDM/results/m31_chunksize_scan"),
}
# Files use bare config name (m31_mixed_qps16 has c2_tdm_m31; need to map for c2_tdm_m31_512).
ALIAS = {"c2_tdm_m31_512": "c2_tdm_m31"}

QPS = 16.0
TTFT_GRID = [500, 1000, 1500, 2000, 3000, 5000]
TPOT_GRID = [50, 100, 150, 200, 300, 500]


def load_joined(config: str) -> tuple[list[dict], float, float]:
    src_dir = SOURCES[config]
    file_name = ALIAS.get(config, config)
    path = src_dir / f"{file_name}_qps{QPS}.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    warmup_s = s["warmup_s"]
    window_end_s = warmup_s + s["duration_s"]
    return d["joined"], warmup_s, window_end_s


def meet_slo_count(joined, warmup_s, window_end_s, slo_ttft, slo_tpot):
    in_win = [j for j in joined
              if warmup_s <= j["arrival_time_s"] < window_end_s
              and j["status"] == 200 and j.get("matched")]
    total = len(in_win)
    meet = sum(
        1 for j in in_win
        if j.get("ttft_ms") is not None
        and j.get("tpot_ms_mean") is not None
        and j["ttft_ms"] < slo_ttft
        and j["tpot_ms_mean"] < slo_tpot
        and j.get("output_tokens"))
    return meet, total


def main() -> None:
    grids = {}
    for cfg in SOURCES:
        joined, w, we = load_joined(cfg)
        cell = {}
        for tt in TTFT_GRID:
            for tp in TPOT_GRID:
                m, t = meet_slo_count(joined, w, we, tt, tp)
                cell[(tt, tp)] = 100.0 * m / t if t else 0.0
        grids[cfg] = cell

    # Print one grid per config.
    for cfg in SOURCES:
        print(f"\n--- {cfg} ---  (rows=tpot ms, cols=ttft ms)")
        print(f"  tpot \\ ttft  | "
              + "  ".join(f"{tt:>6d}" for tt in TTFT_GRID))
        print(f"  {'-'*13} | "
              + "  ".join("-" * 6 for _ in TTFT_GRID))
        for tp in TPOT_GRID:
            row = [grids[cfg][(tt, tp)] for tt in TTFT_GRID]
            print(f"  {tp:>5d}        | "
                  + "  ".join(f"{v:>6.1f}" for v in row))

    # Cross-config delta tables on key SLO points to see ranking changes.
    key_points = [
        ("strict", 500, 50),
        ("mid-1", 1000, 100),
        ("disc-band", 1500, 150),
        ("loose", 1500, 200),
        ("doc", 2000, 300),
    ]
    print("\n\n=== Cross-config ranking on key SLO points (meet_slo %) ===")
    print(f"  {'tier':<10s} {'tt':>5s} {'tp':>4s}  | "
          + "  ".join(f"{c:>16s}" for c in SOURCES))
    print(f"  {'-'*10} {'-'*5} {'-'*4}  | "
          + "  ".join("-" * 16 for _ in SOURCES))
    for label, tt, tp in key_points:
        row = [grids[c][(tt, tp)] for c in SOURCES]
        print(f"  {label:<10s} {tt:>5d} {tp:>4d}  | "
              + "  ".join(f"{v:>16.1f}" for v in row))


if __name__ == "__main__":
    main()
