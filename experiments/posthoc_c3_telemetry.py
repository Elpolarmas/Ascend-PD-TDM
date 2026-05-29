"""Post-hoc analyze c3-fair iter telemetry (2026-05-25)

Goal: quantify c3 mixed-iter time distribution + batch composition,
compare with m31's reported numbers (T6_FINDINGS):
  m31 prefill iter mean 150ms, batch 2048 token, 1-6 reqs/iter (mean 2.16)
  m31 decode iter  mean  39ms, 29 reqs/iter
  m31 cycle (1 prefill + 1 decode) = 189ms

Input: results/c3_telemetry/code_k{2.8,4.9}_seed0/tdm_trace/qps_sweep_c3_cp_iter.jsonl
Output: stdout report
"""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results/c3_telemetry")
WARMUP_S = 30.0


def percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[idx]


def analyze_cell(cell_dir: Path) -> None:
    print(f"\n{'='*80}\n=== {cell_dir.name}\n{'='*80}")
    iter_path = cell_dir / "tdm_trace" / "qps_sweep_c3_cp_iter.jsonl"
    if not iter_path.exists():
        print(f"  MISSING: {iter_path}")
        return
    iters = []
    with iter_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            iters.append(json.loads(line))
    if not iters:
        print("  empty iter jsonl")
        return
    print(f"  total iters in jsonl: {len(iters)}")
    print(f"  iters with iter_duration_ms: {sum(1 for x in iters if x.get('iter_duration_ms') is not None)}")
    # window: drop warmup
    t0 = iters[0]["ts_ms"]
    warm_cut_ms = t0 + WARMUP_S * 1000.0
    win = [x for x in iters if x["ts_ms"] >= warm_cut_ms
           and x.get("iter_duration_ms") is not None]
    print(f"  iters after warmup ({WARMUP_S}s): {len(win)}")
    if not win:
        return
    durs = [x["iter_duration_ms"] for x in win]
    reqs = [x["batch_num_reqs"] for x in win]
    toks = [x["batch_num_tokens"] for x in win]
    print(f"\n  ---- iter_duration_ms (mixed iter wall time) ----")
    print(f"    n      = {len(durs)}")
    print(f"    mean   = {mean(durs):.2f}")
    print(f"    median = {median(durs):.2f}")
    print(f"    p90    = {percentile(durs, 90):.2f}")
    print(f"    p99    = {percentile(durs, 99):.2f}")
    print(f"    min    = {min(durs):.2f}")
    print(f"    max    = {max(durs):.2f}")
    print(f"\n  ---- batch_num_reqs (reqs per iter) ----")
    print(f"    mean   = {mean(reqs):.2f}")
    print(f"    median = {median(reqs):.0f}")
    print(f"    p99    = {percentile(reqs, 99):.0f}")
    print(f"    min    = {min(reqs)}  max = {max(reqs)}")
    print(f"\n  ---- batch_num_tokens (tokens per iter) ----")
    print(f"    mean   = {mean(toks):.1f}")
    print(f"    median = {median(toks):.0f}")
    print(f"    p99    = {percentile(toks, 99):.0f}")
    print(f"    min    = {min(toks)}  max = {max(toks)}")
    # how many iters are at-or-near token cap (=2048)? proxy for "long prompt prefill 占满 budget"
    cap = 2048
    near_cap = sum(1 for t in toks if t >= cap * 0.95)
    print(f"\n  iters with tokens ≥ {int(cap*0.95)} (~chunk_budget cap): {near_cap}/{len(toks)} = {near_cap/len(toks)*100:.1f}%")
    # decode-dominated iters (token ≈ reqs, i.e., 1 token/req)
    decode_like_idx = [i for i, (r, t) in enumerate(zip(reqs, toks)) if t <= r * 1.5]
    print(f"  iters with tokens ≤ 1.5×reqs (decode-dominated): {len(decode_like_idx)}/{len(toks)} = {len(decode_like_idx)/len(toks)*100:.1f}%")

    # Sub-distribution: separate "mixed at chunk cap (prefill-dominant)"
    # from "decode-dominated" — direct apples-to-apples with m31 prefill iter
    # (mean 150ms, 2048 tok, 2.16 reqs) and m31 decode iter (mean 39ms, 29 reqs).
    cap_idx = [i for i, t in enumerate(toks) if t >= cap * 0.95]
    cap_durs = [durs[i] for i in cap_idx]
    cap_reqs = [reqs[i] for i in cap_idx]
    decode_durs = [durs[i] for i in decode_like_idx]
    decode_reqs = [reqs[i] for i in decode_like_idx]

    if cap_durs:
        print(f"\n  ---- SUB: chunk-cap mixed iters (n={len(cap_durs)}, ~prefill-dominant) ----")
        print(f"    iter_duration_ms  mean={mean(cap_durs):.1f}  median={median(cap_durs):.1f}  p99={percentile(cap_durs,99):.1f}")
        print(f"    reqs/iter         mean={mean(cap_reqs):.1f}  median={median(cap_reqs):.0f}  max={max(cap_reqs)}")
    if decode_durs:
        print(f"\n  ---- SUB: decode-dominated iters (n={len(decode_durs)}) ----")
        print(f"    iter_duration_ms  mean={mean(decode_durs):.1f}  median={median(decode_durs):.1f}  p99={percentile(decode_durs,99):.1f}")
        print(f"    reqs/iter         mean={mean(decode_reqs):.1f}  median={median(decode_reqs):.0f}  max={max(decode_reqs)}")

    print(f"\n  ---- vs m31 reference (T6_FINDINGS code k=2.8 s1) ----")
    print(f"    m31 prefill iter  : mean 150ms, batch 2048 tok, ~2.16 reqs")
    print(f"    m31 decode iter   : mean  39ms, 29 reqs/iter")
    print(f"    m31 cycle (P+D)   : 189ms")
    if cap_durs and decode_durs:
        approx_c3_cycle = mean(cap_durs) + mean(decode_durs)
        print(f"    c3 (chunk-cap + decode-dominated): {mean(cap_durs):.0f} + {mean(decode_durs):.0f} = {approx_c3_cycle:.0f}ms")
        print(f"    c3 median mixed iter: {median(durs):.0f}ms")
        print(f"    Δ vs m31 cycle: c3 median {median(durs)-189:+.0f}ms ({(median(durs)/189-1)*100:+.1f}%)")


def main():
    for name in ("code_k2.8_seed0", "code_k4.9_seed0"):
        d = ROOT / name
        if d.exists():
            analyze_cell(d)


if __name__ == "__main__":
    main()
