"""Build hybrid Azure traces for P1.9e workload-transition ablation.

Splices 30s of one workload + 30s of the other, with timestamps
re-normalized to a continuous 0-60s stream. Two directions:

  conv30s_code30s: light → heavy transition (PID should ramp ratio UP)
  code30s_conv30s: heavy → light transition (PID should ramp ratio DOWN)

Goal: expose the slow loop's transient role. Steady-state P1.9d already
showed slow loop pins to ratio_max — its contribution must live in the
transition window. M3.1 (PID-driven) vs M1@ratio_max (static) Δ in
transition phase is the only place slow loop can show a +Δ.
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path


def parse_ts(s: str) -> datetime:
    # "2023-11-16 18:15:46.6805900" — fractional has 7 digits, strptime
    # accepts up to 6, so trim.
    head, _, frac = s.partition(".")
    frac = (frac + "000000")[:6]
    return datetime.strptime(f"{head}.{frac}", "%Y-%m-%d %H:%M:%S.%f")


def load_window(path: Path, start_offset_s: float, duration_s: float):
    """Return rows in [start_offset, start_offset + duration] with the
    column tuple (ts_datetime, ContextTokens, GeneratedTokens). Timestamps
    are absolute from the source CSV (first row's ts is the trace t=0)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        first_ts = None
        for r in reader:
            ts = parse_ts(r["TIMESTAMP"])
            if first_ts is None:
                first_ts = ts
            offset = (ts - first_ts).total_seconds()
            if offset < start_offset_s:
                continue
            if offset > start_offset_s + duration_s:
                break
            rows.append((ts, int(r["ContextTokens"]), int(r["GeneratedTokens"])))
    return rows


def write_hybrid(out_path: Path, first_seg, second_seg, second_offset_s: float):
    """Concat two segments. Re-base both to start at a common synthetic
    epoch so the combined stream's elapsed times are continuous:
      - first_seg events keep their relative times within [0, T1)
      - second_seg events get shifted to [second_offset_s, second_offset_s + T2)
    `second_offset_s` is typically T1 (e.g. 30s) so there's no gap.
    """
    if not first_seg or not second_seg:
        raise ValueError("both segments must be non-empty")
    base1 = first_seg[0][0]
    base2 = second_seg[0][0]
    out = []
    for ts, ctx, gen in first_seg:
        rel = (ts - base1).total_seconds()
        out.append((rel, ctx, gen))
    for ts, ctx, gen in second_seg:
        rel = (ts - base2).total_seconds() + second_offset_s
        out.append((rel, ctx, gen))
    out.sort(key=lambda r: r[0])

    # Synthesize an absolute timestamp series anchored at an arbitrary epoch
    # so the driver's existing parser is happy. Use the original conv epoch
    # for stability across runs.
    anchor = datetime(2023, 11, 16, 18, 15, 0)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        for rel, ctx, gen in out:
            ts = anchor + timedelta(seconds=rel)
            # Format with 7-digit fractional to match source style
            frac = int((ts.microsecond) * 10)
            stamp = ts.strftime("%Y-%m-%d %H:%M:%S") + f".{frac:07d}"
            w.writerow([stamp, ctx, gen])
    print(f"wrote {out_path}  rows={len(out)}  first={out[0][0]:.3f}s  last={out[-1][0]:.3f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/vllm-workspace/Ascend-PD-TDM/data/azure_trace")
    ap.add_argument("--seg-duration", type=float, default=30.0,
                    help="each segment duration in seconds")
    ap.add_argument("--conv-offset", type=float, default=1860.0)
    ap.add_argument("--code-offset", type=float, default=570.0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    conv_path = data / "AzureLLMInferenceTrace_conv.csv"
    code_path = data / "AzureLLMInferenceTrace_code.csv"

    conv_seg = load_window(conv_path, args.conv_offset, args.seg_duration)
    code_seg = load_window(code_path, args.code_offset, args.seg_duration)
    print(f"conv window {args.conv_offset}-{args.conv_offset+args.seg_duration}s: {len(conv_seg)} rows")
    print(f"code window {args.code_offset}-{args.code_offset+args.seg_duration}s: {len(code_seg)} rows")

    write_hybrid(data / "AzureLLMInferenceTrace_hybrid_conv2code.csv",
                 conv_seg, code_seg, args.seg_duration)
    write_hybrid(data / "AzureLLMInferenceTrace_hybrid_code2conv.csv",
                 code_seg, conv_seg, args.seg_duration)


if __name__ == "__main__":
    main()
