"""
Experiment F: Graph-Aware Batch Shaping vs Naive Batching
=========================================================
核心问题：调度器主动对齐已编译 Graph shape 能带来多少收益？

对比两种策略：
1. Naive: 有多少请求就组多大 batch（系统自动 pad，会有浪费）
2. Graph-Aware: 调度器主动选择最近的已编译 shape，或拆/合 batch 以减少浪费

具体测量：
- 不同 batch size 下的 padding 浪费率
- Graph 命中（pad 少）vs Graph 浪费大 vs Eager fallback 三种情况的吞吐对比
- 模拟 TDM 场景：纯 P batch 和纯 D batch 各自的 Graph 匹配效率

用法：
  python exp_f_graph_aware.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import time
import random


def get_compiled_shapes(llm):
    """获取已编译的 graph shapes。"""
    compile_config = llm.llm_engine.vllm_config.compilation_config
    return sorted(compile_config.cudagraph_capture_sizes)


def find_padded_size(actual_bs, compiled_sizes):
    """模拟 vLLM pad_for_cudagraph: 找到 >= actual_bs 的最小已编译 size。"""
    for s in compiled_sizes:
        if s >= actual_bs:
            return s
    return None  # 超过 max → eager


def test1_padding_waste_profile(llm):
    """全面分析 1~max_size 每个 batch size 的 padding 浪费率。
    量化 Graph-Aware Batch Shaping 的优化空间。
    """
    compiled_sizes = get_compiled_shapes(llm)
    max_size = max(compiled_sizes)

    print("\n" + "=" * 60)
    print("Test 1: Padding Waste Profile (全 batch size 扫描)")
    print("=" * 60)

    # 计算每个 bs 的浪费
    waste_data = []
    for bs in range(1, max_size + 1):
        padded = find_padded_size(bs, compiled_sizes)
        waste_pct = (padded - bs) / padded * 100 if padded else 100
        waste_data.append({
            "actual_bs": bs,
            "padded_bs": padded,
            "waste_pct": round(waste_pct, 1),
            "exact_match": bs in compiled_sizes,
        })

    # 按 TDM 场景分段统计
    segments = {
        "prefill_typical (1-8)": [d for d in waste_data if 1 <= d["actual_bs"] <= 8],
        "prefill_burst (1-32)": [d for d in waste_data if 1 <= d["actual_bs"] <= 32],
        "decode_light (1-16)": [d for d in waste_data if 1 <= d["actual_bs"] <= 16],
        "decode_medium (1-64)": [d for d in waste_data if 1 <= d["actual_bs"] <= 64],
        "decode_heavy (1-128)": [d for d in waste_data if 1 <= d["actual_bs"] <= 128],
        "all (1-max)": waste_data,
    }

    segment_stats = {}
    for seg_name, seg_data in segments.items():
        wastes = [d["waste_pct"] for d in seg_data]
        exact_matches = sum(1 for d in seg_data if d["exact_match"])
        avg_waste = sum(wastes) / len(wastes)
        max_waste = max(wastes)
        # Graph-Aware 策略：只选择精确匹配或浪费 <10% 的 size
        low_waste = [d for d in seg_data if d["waste_pct"] <= 10]
        coverage = len(low_waste) / len(seg_data) * 100

        stat = {
            "range": seg_name,
            "total_sizes": len(seg_data),
            "exact_matches": exact_matches,
            "avg_waste_pct": round(avg_waste, 1),
            "max_waste_pct": round(max_waste, 1),
            "low_waste_coverage_pct": round(coverage, 1),
        }
        segment_stats[seg_name] = stat
        print(f"  {seg_name:<25s}: avg_waste={avg_waste:>5.1f}%  "
              f"max={max_waste:>5.1f}%  exact_match={exact_matches}/{len(seg_data)}  "
              f"low_waste(<10%)={coverage:.0f}%")

    # 找出浪费最严重的 top-10
    worst = sorted(waste_data, key=lambda d: d["waste_pct"], reverse=True)[:10]
    print(f"\n  Worst padding cases:")
    for d in worst:
        print(f"    bs={d['actual_bs']:>3d} → pad={d['padded_bs']:>3d}  waste={d['waste_pct']:.1f}%")

    return {"segment_stats": segment_stats, "worst_cases": worst, "compiled_sizes": compiled_sizes}


def test2_graph_aware_vs_naive(llm, num_iters=8):
    """核心对比：Graph-Aware 选 batch size vs Naive 随机 batch size。

    模拟场景：
    - Naive: 请求队列给多少就用多少（随机 batch size）
    - Graph-Aware: 调度器主动选择已编译 shape（精确匹配或向下取整拆 batch）

    对每种 batch size 测量实际吞吐，然后对比两种策略的期望吞吐。
    """
    from vllm import SamplingParams

    compiled_sizes = get_compiled_shapes(llm)
    max_size = max(compiled_sizes)

    print("\n" + "=" * 60)
    print("Test 2: Graph-Aware vs Naive Batch Size Selection")
    print("=" * 60)

    prompt = "Hello"
    sp = SamplingParams(max_tokens=50, temperature=0.0)

    # 测试一组代表性 batch sizes (覆盖精确匹配、小浪费、大浪费)
    test_sizes = []
    # 精确匹配
    for s in compiled_sizes:
        if s <= 64:
            test_sizes.append(s)
    # 非匹配（会产生 padding 浪费）
    for s in [3, 5, 7, 9, 10, 13, 15, 17, 20, 25, 33, 50]:
        if s <= 64 and s not in test_sizes:
            test_sizes.append(s)
    test_sizes = sorted(set(test_sizes))

    results = []
    for bs in test_sizes:
        padded = find_padded_size(bs, compiled_sizes)
        waste_pct = (padded - bs) / padded * 100 if padded else 0
        exact = bs in compiled_sizes

        prompts = [prompt] * bs

        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = llm.generate(prompts, sp)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        # skip warmup
        avg_time = sum(times[2:]) / len(times[2:])
        total_out = sum(len(o.outputs[0].token_ids) for o in out)
        tps = total_out / avg_time
        # 有效吞吐 = 只算真实请求的 tokens（不含 padding 位置的浪费）
        effective_tps = tps  # vLLM 只对真实请求生成 tokens

        entry = {
            "batch_size": bs,
            "padded_to": padded,
            "waste_pct": round(waste_pct, 1),
            "exact_match": exact,
            "throughput_tps": round(tps, 1),
            "avg_time_ms": round(avg_time * 1000, 1),
            "total_output_tokens": total_out,
        }
        results.append(entry)
        marker = "EXACT" if exact else f"pad→{padded}"
        print(f"  bs={bs:>3d} [{marker:<10s}] waste={waste_pct:>5.1f}% | "
              f"{tps:>7.1f} tok/s | time={avg_time*1000:.1f}ms")

    # 对比分析：精确匹配 vs 非匹配的吞吐效率
    exact_results = [r for r in results if r["exact_match"]]
    padded_results = [r for r in results if not r["exact_match"]]

    if exact_results and padded_results:
        print(f"\n  === 效率对比 ===")
        # 计算 per-request 吞吐（归一化到每个请求）
        exact_per_req = [r["throughput_tps"] / r["batch_size"] for r in exact_results]
        padded_per_req = [r["throughput_tps"] / r["batch_size"] for r in padded_results]
        avg_exact = sum(exact_per_req) / len(exact_per_req)
        avg_padded = sum(padded_per_req) / len(padded_per_req)
        diff = (avg_exact - avg_padded) / avg_padded * 100
        print(f"  精确匹配 avg per-req TPS: {avg_exact:.1f}")
        print(f"  需 padding avg per-req TPS: {avg_padded:.1f}")
        print(f"  精确匹配优势: {diff:+.1f}%")

    return results


def test3_tdm_simulation(llm, num_iters=8):
    """模拟 TDM 场景下 Graph-Aware 的效果。

    模拟：一个请求流，用 Naive 和 Graph-Aware 两种策略组 batch，
    对比总吞吐差异。

    场景：
    - 假设一段时间内到达 N 个请求
    - Naive: 每次取所有可用请求组一个 batch（可能是任意 size）
    - Graph-Aware: 调度器选择最近的已编译 shape
      - 如果请求数 > 最近 shape：拆成精确匹配 shape + 剩余
      - 如果请求数 < 最近 shape：pad 到该 shape
    """
    from vllm import SamplingParams

    compiled_sizes = get_compiled_shapes(llm)

    print("\n" + "=" * 60)
    print("Test 3: TDM Simulation - Naive vs Graph-Aware Batch Strategy")
    print("=" * 60)

    prompt = "Hello, explain this concept briefly."
    sp = SamplingParams(max_tokens=100, temperature=0.0)

    # 模拟不同的请求到达场景
    scenarios = [
        # (name, description, batch_sizes_naive)
        ("prefill_sparse", "稀疏 prefill: 1-5 个请求/iter",
         [1, 3, 2, 5, 4, 1, 3, 2, 5, 4]),
        ("prefill_moderate", "中等 prefill: 5-20 个请求/iter",
         [7, 13, 9, 15, 11, 5, 17, 10, 7, 20]),
        ("decode_growing", "decode 队列增长: 4→40",
         [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
        ("decode_stable", "decode 稳定运行: ~32 个请求",
         [30, 33, 31, 35, 29, 34, 32, 30, 33, 31]),
    ]

    all_scenario_results = []

    for sc_name, sc_desc, naive_batches in scenarios:
        print(f"\n  --- {sc_name}: {sc_desc} ---")

        # Naive 策略：直接用原始 batch size（系统自动 pad）
        naive_total_time = 0
        naive_total_tokens = 0
        naive_total_waste = 0

        for bs in naive_batches:
            padded = find_padded_size(bs, compiled_sizes)
            waste = padded - bs if padded else 0
            naive_total_waste += waste

            prompts = [prompt] * bs
            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            avg_t = sum(times[2:]) / len(times[2:])
            total_out = sum(len(o.outputs[0].token_ids) for o in out)
            naive_total_time += avg_t
            naive_total_tokens += total_out

        naive_tps = naive_total_tokens / naive_total_time
        naive_avg_waste = naive_total_waste / len(naive_batches)

        # Graph-Aware 策略：选择最近的已编译 shape
        # 策略：向下取最近已编译 shape（减少 padding），剩余请求留到下一轮
        aware_total_time = 0
        aware_total_tokens = 0
        aware_total_waste = 0
        aware_batches = []

        for bs in naive_batches:
            # 找最近的不超过 bs 的已编译 shape
            best_fit = None
            for s in reversed(compiled_sizes):
                if s <= bs:
                    best_fit = s
                    break
            if best_fit is None:
                best_fit = compiled_sizes[0]  # 最小的

            # 如果精确匹配或接近，直接用 padded（向上）
            padded_up = find_padded_size(bs, compiled_sizes)
            waste_up = padded_up - bs if padded_up else 0
            waste_down = bs - best_fit

            # 选浪费更少的方案
            if waste_up <= waste_down:
                chosen_bs = bs  # 让系统 pad 到 padded_up
                waste = waste_up
            else:
                chosen_bs = best_fit  # 只处理 best_fit 个请求，剩余排队
                waste = 0  # 精确匹配，无 pad 浪费

            aware_batches.append(chosen_bs)
            aware_total_waste += waste

            prompts = [prompt] * chosen_bs
            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            avg_t = sum(times[2:]) / len(times[2:])
            total_out = sum(len(o.outputs[0].token_ids) for o in out)
            aware_total_time += avg_t
            aware_total_tokens += total_out

        aware_tps = aware_total_tokens / aware_total_time
        aware_avg_waste = aware_total_waste / len(naive_batches)

        speedup = (aware_tps / naive_tps - 1) * 100
        waste_reduction = naive_avg_waste - aware_avg_waste

        sc_result = {
            "scenario": sc_name,
            "description": sc_desc,
            "naive_batches": naive_batches,
            "aware_batches": aware_batches,
            "naive_tps": round(naive_tps, 1),
            "aware_tps": round(aware_tps, 1),
            "speedup_pct": round(speedup, 1),
            "naive_avg_waste": round(naive_avg_waste, 1),
            "aware_avg_waste": round(aware_avg_waste, 1),
            "waste_reduction": round(waste_reduction, 1),
        }
        all_scenario_results.append(sc_result)

        print(f"    Naive:       {naive_tps:>7.1f} tok/s  avg_pad_waste={naive_avg_waste:.1f}")
        print(f"    Graph-Aware: {aware_tps:>7.1f} tok/s  avg_pad_waste={aware_avg_waste:.1f}")
        print(f"    Speedup: {speedup:+.1f}%  Waste reduction: {waste_reduction:.1f} tokens/batch")
        print(f"    Naive batches:  {naive_batches}")
        print(f"    Aware batches:  {aware_batches}")

    return all_scenario_results


def test4_per_token_efficiency(llm, num_iters=8):
    """测量不同 padding 率下的 per-token 计算效率。

    核心问题：padding 的 token 虽然不产出有效 output，但占用了 NPU 算力。
    这意味着同样一个 graph（比如 shape=32），处理 17 个真实请求 vs 32 个真实请求，
    per-real-token 的效率差异有多大？
    """
    from vllm import SamplingParams

    compiled_sizes = get_compiled_shapes(llm)

    print("\n" + "=" * 60)
    print("Test 4: Per-Token Efficiency vs Padding Rate")
    print("=" * 60)

    prompt = "Hello"
    sp = SamplingParams(max_tokens=50, temperature=0.0)

    # 选一个典型的 compiled size（如 32），测试不同实际 bs
    target_shapes = [s for s in compiled_sizes if s in [8, 16, 32, 64]]

    results = []
    for shape in target_shapes:
        print(f"\n  Graph shape = {shape}:")
        # 测试实际 bs 从 上一个 shape+1 到 shape
        prev_shape = 0
        for s in compiled_sizes:
            if s < shape:
                prev_shape = s
        test_bs_list = list(range(max(prev_shape + 1, 1), shape + 1))
        # 只取几个代表性的点
        if len(test_bs_list) > 6:
            step = max(1, len(test_bs_list) // 5)
            test_bs_list = test_bs_list[::step]
            if shape not in test_bs_list:
                test_bs_list.append(shape)

        for actual_bs in test_bs_list:
            waste_pct = (shape - actual_bs) / shape * 100
            prompts = [prompt] * actual_bs

            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)

            avg_time = sum(times[2:]) / len(times[2:])
            total_out = sum(len(o.outputs[0].token_ids) for o in out)
            tps = total_out / avg_time
            per_req_tps = tps / actual_bs  # 每个真实请求的吞吐

            entry = {
                "graph_shape": shape,
                "actual_bs": actual_bs,
                "waste_pct": round(waste_pct, 1),
                "throughput_tps": round(tps, 1),
                "per_req_tps": round(per_req_tps, 1),
                "time_ms": round(avg_time * 1000, 1),
            }
            results.append(entry)
            print(f"    bs={actual_bs:>3d}/{shape}  waste={waste_pct:>5.1f}%  | "
                  f"{tps:>7.1f} tok/s  per_req={per_req_tps:.1f} tok/s  "
                  f"time={avg_time*1000:.1f}ms")

        # 分析：该 shape 下，满载 vs 最低载的效率差
        shape_results = [r for r in results if r["graph_shape"] == shape]
        if len(shape_results) >= 2:
            full = shape_results[-1]  # 满载
            min_load = shape_results[0]  # 最低载
            eff_loss = (full["per_req_tps"] - min_load["per_req_tps"]) / full["per_req_tps"] * 100
            print(f"    → 满载 per_req={full['per_req_tps']:.1f} vs "
                  f"最低载 per_req={min_load['per_req_tps']:.1f}  "
                  f"效率损失={eff_loss:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp F: Graph-Aware Batch Shaping")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-4B")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--tests", type=str, default="1,2,3,4",
                        help="Comma-separated test numbers to run (e.g., '1,2' or '1,2,3,4')")
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_f.json")
    args = parser.parse_args()

    tests_to_run = set(int(t) for t in args.tests.split(","))

    from vllm import LLM, SamplingParams

    all_results = {"model": os.path.basename(args.model)}

    print(f"Loading model: {args.model}")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    all_results["load_time_s"] = round(load_time, 1)

    # Warmup
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=10, temperature=0.0))

    if 1 in tests_to_run:
        all_results["test1_waste_profile"] = test1_padding_waste_profile(llm)
    if 2 in tests_to_run:
        all_results["test2_aware_vs_naive"] = test2_graph_aware_vs_naive(llm)
    if 3 in tests_to_run:
        all_results["test3_tdm_simulation"] = test3_tdm_simulation(llm)
    if 4 in tests_to_run:
        all_results["test4_per_token_efficiency"] = test4_per_token_efficiency(llm)

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT F SUMMARY")
    print("=" * 60)
    print("  核心问题：Graph-Aware Batch Shaping 能带来多少收益？")
    print("  Test 1: padding 浪费全景分析")
    print("  Test 2: 精确匹配 vs 需 padding 的吞吐对比")
    print("  Test 3: TDM 模拟场景下 Naive vs Graph-Aware 策略对比")
    print("  Test 4: 不同 padding 率下的 per-token 计算效率")
    print(f"\n  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
