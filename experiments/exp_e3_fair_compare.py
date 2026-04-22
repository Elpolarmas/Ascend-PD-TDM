"""
Experiment E3: Fair 2-Card Comparison — Unified TP=2 vs Phased TP=2 vs Disagg 1P1D
====================================================================================
所有方案统一使用 2 张 910B3，公平对比：
  1. Unified TP=2:  2 卡 tensor parallel, continuous batching (chunked prefill ON/OFF)
  2. Phased TP=2:   2 卡 tensor parallel, enable_pd_transfer (chunked prefill ON/OFF)
  3. Disagg 1P1D:   1 卡 prefill + 1 卡 decode (已有数据直接加载)

用法：
  python3 exp_e3_fair_compare.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import gc
import json
import time


# ── Workloads (与 exp_e / exp_e2 一致) ─────────────────────────────
SHORT_PROMPT = "Hello, explain machine learning briefly."
LONG_PROMPT = (
    "Please write a comprehensive analysis of the global economic trends "
    "in the 21st century, covering topics such as globalization, technological "
    "disruption, income inequality, climate change impacts, geopolitical shifts, "
    "and the rise of emerging markets. Discuss how these factors interconnect "
    "and influence each other in complex ways."
)

WORKLOADS = [
    ("short_4req",  [SHORT_PROMPT] * 4,  200),
    ("short_16req", [SHORT_PROMPT] * 16, 200),
    ("short_32req", [SHORT_PROMPT] * 32, 200),
    ("long_4req",   [LONG_PROMPT] * 4,   200),
    ("long_16req",  [LONG_PROMPT] * 16,  200),
    ("mixed_16req", [SHORT_PROMPT] * 8 + [LONG_PROMPT] * 8, 200),
]


def run_workload(llm, prompts, max_tokens, num_rounds, label=""):
    """Run a workload multiple rounds and return averaged metrics (skip round 0)."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Warmup
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=5, temperature=0.0))

    times = []
    token_counts = []

    for _ in range(num_rounds):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        total_in = sum(len(o.prompt_token_ids) for o in outputs if o.prompt_token_ids)
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        token_counts.append((total_in, total_out))

    # Skip first round
    valid = slice(1, None)
    avg_time = sum(times[valid]) / len(times[valid])
    avg_in = sum(t[0] for t in token_counts[valid]) / len(token_counts[valid])
    avg_out = sum(t[1] for t in token_counts[valid]) / len(token_counts[valid])

    # TTFT estimate
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)
    ttft_times = []
    for _ in range(4):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp_one)
        elapsed = time.perf_counter() - t0
        ttft_times.append(elapsed * 1000 / len(prompts))
    avg_ttft = sum(ttft_times[1:]) / len(ttft_times[1:])

    avg_tokens_per_req = avg_out / len(prompts)
    tpot = (avg_time * 1000 / len(prompts) - avg_ttft) / max(avg_tokens_per_req - 1, 1)

    return {
        "label": label,
        "num_requests": len(prompts),
        "avg_time_s": round(avg_time, 3),
        "avg_input_tokens": round(avg_in),
        "avg_output_tokens": round(avg_out),
        "throughput_tps": round(avg_out / avg_time, 1),
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_tpot_ms": round(tpot, 2),
    }


def create_llm(model, gpu_mem, tp_size=2, pd_transfer=False,
               chunked_prefill=None):
    """Create LLM with specified config."""
    from vllm import LLM

    additional_config = {}
    if pd_transfer:
        additional_config["ascend_scheduler_config"] = {
            "enabled": True,
            "enable_pd_transfer": True,
        }
    if chunked_prefill is not None:
        additional_config.setdefault("ascend_scheduler_config", {})
        additional_config["ascend_scheduler_config"]["enable_chunked_prefill"] = chunked_prefill

    kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        tensor_parallel_size=tp_size,
    )
    if additional_config:
        kwargs["additional_config"] = additional_config

    return LLM(**kwargs)


def run_condition(name, model, gpu_mem, num_rounds, tp_size=2,
                  pd_transfer=False, chunked_prefill=None):
    """Run all workloads for one condition, return results list."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  TP={tp_size}, pd_transfer={pd_transfer}, chunked_prefill={chunked_prefill}")
    print(f"{'='*70}")

    llm = create_llm(model, gpu_mem, tp_size=tp_size,
                      pd_transfer=pd_transfer, chunked_prefill=chunked_prefill)

    results = []
    for wl_name, prompts, max_tok in WORKLOADS:
        print(f"  {wl_name}...", end=" ", flush=True)
        r = run_workload(llm, prompts, max_tok, num_rounds,
                         label=f"{name}_{wl_name}")
        results.append(r)
        print(f"TPS={r['throughput_tps']:>8.1f} | "
              f"TTFT={r['avg_ttft_ms']:>6.1f}ms | "
              f"TPOT={r['avg_tpot_ms']:>6.2f}ms | "
              f"Time={r['avg_time_s']:.3f}s")

    del llm
    gc.collect()
    return results


def print_comparison(all_results):
    """Print comparison table across all conditions."""
    conditions = list(all_results.keys())
    wl_names = [wl[0] for wl in WORKLOADS]

    # Build lookup: condition -> wl_name -> result
    lookup = {}
    for cond, results in all_results.items():
        lookup[cond] = {}
        for r in results:
            # Extract wl_name from label
            wl = r["label"].replace(f"{cond}_", "")
            lookup[cond][wl] = r

    # ── TPS comparison ──
    print(f"\n\n{'='*90}")
    print("THROUGHPUT (TPS) COMPARISON — All conditions use 2 × 910B3")
    print(f"{'='*90}")

    header = f"  {'Workload':<16s}"
    for c in conditions:
        header += f"  {c:>14s}"
    # Add relative columns vs first condition
    base = conditions[0]
    for c in conditions[1:]:
        header += f"  {'vs '+base:>12s}"
    print(header)
    print(f"  {'-'*16}" + f"  {'-'*14}" * len(conditions) + f"  {'-'*12}" * (len(conditions)-1))

    for wl in wl_names:
        line = f"  {wl:<16s}"
        tps_vals = []
        for c in conditions:
            tps = lookup.get(c, {}).get(wl, {}).get("throughput_tps", 0)
            tps_vals.append(tps)
            line += f"  {tps:>14.1f}"
        base_tps = tps_vals[0]
        for tps in tps_vals[1:]:
            if base_tps > 0:
                pct = (tps / base_tps - 1) * 100
                marker = "+" if pct >= 0 else ""
                line += f"  {marker}{pct:>10.1f}%"
            else:
                line += f"  {'N/A':>12s}"
        print(line)

    # ── TTFT comparison ──
    print(f"\n{'='*90}")
    print("TTFT (ms) COMPARISON")
    print(f"{'='*90}")

    header = f"  {'Workload':<16s}"
    for c in conditions:
        header += f"  {c:>14s}"
    print(header)
    print(f"  {'-'*16}" + f"  {'-'*14}" * len(conditions))

    for wl in wl_names:
        line = f"  {wl:<16s}"
        for c in conditions:
            ttft = lookup.get(c, {}).get(wl, {}).get("avg_ttft_ms", 0)
            line += f"  {ttft:>14.1f}"
        print(line)

    # ── TPOT comparison ──
    print(f"\n{'='*90}")
    print("TPOT (ms) COMPARISON")
    print(f"{'='*90}")

    header = f"  {'Workload':<16s}"
    for c in conditions:
        header += f"  {c:>14s}"
    print(header)
    print(f"  {'-'*16}" + f"  {'-'*14}" * len(conditions))

    for wl in wl_names:
        line = f"  {wl:<16s}"
        for c in conditions:
            tpot = lookup.get(c, {}).get(wl, {}).get("avg_tpot_ms", 0)
            line += f"  {tpot:>14.2f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Exp E3: Fair 2-card comparison")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-4B")
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_e3.json")
    args = parser.parse_args()

    all_results = {}

    # ── Condition 1: Unified TP=2, chunked_prefill OFF (Ascend default) ──
    all_results["unified_tp2"] = run_condition(
        "unified_tp2", args.model, args.gpu_mem, args.num_rounds,
        tp_size=2, pd_transfer=False, chunked_prefill=False)

    # ── Condition 2: Unified TP=2, chunked_prefill ON ──
    all_results["unified_tp2_cp"] = run_condition(
        "unified_tp2_cp", args.model, args.gpu_mem, args.num_rounds,
        tp_size=2, pd_transfer=False, chunked_prefill=True)

    # ── Condition 3: Phased TP=2 ──
    all_results["phased_tp2"] = run_condition(
        "phased_tp2", args.model, args.gpu_mem, args.num_rounds,
        tp_size=2, pd_transfer=True, chunked_prefill=False)

    # ── Condition 4: Disagg 1P1D (load existing data) ──
    disagg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results_exp_e2.json")
    if os.path.exists(disagg_path):
        with open(disagg_path) as f:
            disagg_data = json.load(f)
        disagg_results = []
        for r in disagg_data.get("experiments", []):
            # Rename label for consistency
            wl = r["label"].replace("disagg_", "")
            r_copy = dict(r)
            r_copy["label"] = f"disagg_1p1d_{wl}"
            # Use total time as avg_time_s for comparison
            r_copy["avg_time_s"] = r.get("avg_total_time_s", r.get("avg_time_s", 0))
            disagg_results.append(r_copy)
        all_results["disagg_1p1d"] = disagg_results
        print(f"\n  Loaded Disagg 1P1D data from {disagg_path}")
    else:
        print(f"\n  WARNING: {disagg_path} not found, skipping Disagg 1P1D")

    # ── Print comparison ──
    print_comparison(all_results)

    # ── Save ──
    output = {
        "model": os.path.basename(args.model),
        "hardware": "2x Ascend 910B3",
        "num_rounds": args.num_rounds,
        "conditions": {k: v for k, v in all_results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
