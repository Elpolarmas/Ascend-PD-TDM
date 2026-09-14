#!/usr/bin/env python3
"""Recompute the paper-facing MaaS cohort metrics from raw replay JSON.

The replay driver is configured for 34,200 s, but the compressed arrival
stream ends at 4,559.84 s.  This script uses the active cohort window
[30, 4560) s, counts missing telemetry as non-attainment, and writes a small
auditable aggregate for the manuscript.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "maas_replay"
OUT = ROOT / "paper" / "data" / "maas_metrics.json"
START_S = 30.0
IDEAL_TTFT_MS = 936.0
IDEAL_TPOT_MS = 216.0


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("percentile of empty sequence")
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _active_end_s() -> float:
    max_arrival = max(
        max(row["arrival_time_s"] for row in json.loads(
            (RAW / f"maas_{name}_rs0.10.json").read_text())["joined"])
        for name in ("sarathi", "pdtdm")
    )
    return math.ceil(max_arrival)


ACTIVE_END_S = _active_end_s()


def summarize(name: str) -> dict:
    path = RAW / f"maas_{name}_rs0.10.json"
    data = json.loads(path.read_text())
    cohort = [
        row for row in data["joined"]
        if START_S <= row.get("arrival_time_s", -1.0) < ACTIVE_END_S
    ]
    matched = [
        row for row in cohort
        if row.get("matched") and row.get("status") == 200
        and row.get("ttft_ms") is not None
        and row.get("tpot_ms_mean") is not None
    ]
    out: dict = {
        "raw_path": str(path.relative_to(ROOT)),
        "cohort_window_s": [START_S, ACTIVE_END_S],
        "goodput_denominator_s": ACTIVE_END_S - START_S,
        "n_in_window": len(cohort),
        "n_matched": len(matched),
        "n_unmatched": len(cohort) - len(matched),
        "ttft_ms": {
            "mean": round(sum(r["ttft_ms"] for r in matched) / len(matched), 2),
            "p99": round(percentile([r["ttft_ms"] for r in matched], 0.99), 2),
        },
        "tpot_ms_mean": {
            "mean": round(sum(r["tpot_ms_mean"] for r in matched) / len(matched), 2),
            "p99": round(percentile([r["tpot_ms_mean"] for r in matched], 0.99), 2),
        },
        "slo": {},
    }
    for multiplier in (3.0, 1.2):
        ttft_slo = multiplier * IDEAL_TTFT_MS
        tpot_slo = multiplier * IDEAL_TPOT_MS
        good = [
            row for row in matched
            if row["ttft_ms"] < ttft_slo and row["tpot_ms_mean"] < tpot_slo
        ]
        output_tokens = sum(row.get("output_tokens", 0) or 0 for row in good)
        out["slo"][f"x{multiplier:g}"] = {
            "ttft_ms": ttft_slo,
            "tpot_ms": tpot_slo,
            "n_meet": len(good),
            "attainment": len(good) / len(cohort),
            "goodput_output_tok_s": output_tokens / (ACTIVE_END_S - START_S),
        }
    return out


def main() -> None:
    result = {
        "method": "arrival-cohort reaggregation",
        "active_window_rule": "[30, ceil(max arrival time)) s",
        "active_end_s": ACTIVE_END_S,
        "ideal_latency_ms": {
            "ttft": IDEAL_TTFT_MS,
            "tpot": IDEAL_TPOT_MS,
        },
        "methods": {
            "sarathi": summarize("sarathi"),
            "pdtdm": summarize("pdtdm"),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
