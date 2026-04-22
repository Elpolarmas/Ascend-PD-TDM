"""
Experiment D: Throughput vs QPS (Concurrent Load Scaling)
==========================================================
确定 TDM 的适用场景边界：什么负载下才有优势？

测试方式：
- 不同并发请求数模拟不同 QPS 压力
- 测量 TTFT / TPOT / 吞吐随并发数的变化
- 观察 decode 阶段在高并发下是否出现瓶颈（=TDM 优化空间）

用法：
  python exp_d_qps_scaling.py --model /vllm-workspace/models/models/Qwen3-0.6B
  python exp_d_qps_scaling.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import time


def measure_at_concurrency(llm, num_concurrent, max_tokens=100, num_rounds=5):
    """Measure performance at a given concurrency level."""
    from vllm import SamplingParams

    prompt = "Explain the concept of machine learning in simple terms."
    prompts = [prompt] * num_concurrent
    sp_full = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=5, temperature=0.0))

    # Measure TTFT (1-token generation)
    ttft_times = []
    for _ in range(min(num_rounds, 3)):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp_one)
        elapsed = time.perf_counter() - t0
        ttft_times.append(elapsed * 1000 / num_concurrent)  # per-request avg

    avg_ttft = sum(ttft_times[1:]) / len(ttft_times[1:]) if len(ttft_times) > 1 else ttft_times[0]

    # Measure full generation
    full_times = []
    output_tokens_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_full)
        elapsed = time.perf_counter() - t0
        full_times.append(elapsed)
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        output_tokens_list.append(total_out)

    # Skip first round for stable results
    avg_time = sum(full_times[1:]) / len(full_times[1:])
    avg_out = sum(output_tokens_list[1:]) / len(output_tokens_list[1:])
    tps = avg_out / avg_time if avg_time > 0 else 0

    # Per-request TPOT estimate
    avg_tokens_per_req = avg_out / num_concurrent
    if avg_tokens_per_req > 1:
        tpot = (avg_time * 1000 - avg_ttft) / (avg_tokens_per_req - 1)
    else:
        tpot = 0

    return {
        "num_concurrent": num_concurrent,
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_tpot_ms": round(tpot, 2),
        "avg_time_ms": round(avg_time * 1000, 2),
        "avg_output_tokens": round(avg_out, 1),
        "throughput_tps": round(tps, 1),
        "tokens_per_request": round(avg_tokens_per_req, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Exp D: QPS Scaling")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-0.6B")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_d.json")
    args = parser.parse_args()

    from vllm import LLM

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
    )
    print("Model loaded.\n")

    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 70)
    print("Experiment D: Throughput vs Concurrency")
    print("=" * 70)

    results = []
    for nc in concurrency_levels:
        print(f"\n  Testing concurrency={nc}...")
        try:
            r = measure_at_concurrency(llm, nc, max_tokens=args.max_tokens)
            results.append(r)
            print(f"    TTFT={r['avg_ttft_ms']:.1f}ms | TPOT={r['avg_tpot_ms']:.2f}ms | "
                  f"TPS={r['throughput_tps']:.1f} | out/req={r['tokens_per_request']:.0f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"num_concurrent": nc, "error": str(e)})

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Conc':>5s}  {'TTFT(ms)':>9s}  {'TPOT(ms)':>9s}  "
          f"{'TPS':>8s}  {'Time(ms)':>9s}  {'Tok/req':>8s}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}")

    for r in results:
        if "error" in r:
            print(f"  {r['num_concurrent']:>5d}  ERROR: {r['error']}")
            continue
        print(f"  {r['num_concurrent']:>5d}  {r['avg_ttft_ms']:>9.1f}  "
              f"{r['avg_tpot_ms']:>9.2f}  {r['throughput_tps']:>8.1f}  "
              f"{r['avg_time_ms']:>9.1f}  {r['tokens_per_request']:>8.0f}")

    # Analysis
    if len(results) >= 2 and "error" not in results[0]:
        base_tps = results[0]["throughput_tps"]
        peak_r = max((r for r in results if "error" not in r),
                     key=lambda x: x["throughput_tps"])
        print(f"\n  Peak throughput: {peak_r['throughput_tps']:.1f} tok/s "
              f"at concurrency={peak_r['num_concurrent']}")
        print(f"  Scaling from 1→{peak_r['num_concurrent']}: "
              f"{peak_r['throughput_tps']/base_tps:.1f}x")

        # Find where TPOT starts degrading significantly
        base_tpot = results[0]["avg_tpot_ms"] if results[0]["avg_tpot_ms"] > 0 else 1
        for r in results:
            if "error" not in r and r["avg_tpot_ms"] > base_tpot * 2:
                print(f"  ⚠ TPOT doubles at concurrency={r['num_concurrent']} "
                      f"({r['avg_tpot_ms']:.1f}ms vs base {base_tpot:.1f}ms)")
                print(f"    → This is where TDM scheduling becomes critical")
                break

    all_results = {"model": os.path.basename(args.model), "results": results}
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
