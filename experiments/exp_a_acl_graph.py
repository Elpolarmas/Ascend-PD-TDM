"""
Experiment A (v2): ACL Graph Behavior Analysis for TDM
=======================================================
核心问题：ACL Graph 机制对 TDM 调度有什么影响？

背景理解（从源码分析得出）：
- vLLM 会自动 pad batch size 到最近的已编译 graph shape
- 仅当 total_tokens > max_compiled_size 时才 eager fallback
- 已编译 shape 是离散的（如 1,2,8,16,32,...,512）

测试内容：
1. Graph 配置分析：编译了哪些 shape？pad 映射关系？
2. Padding 效率分析：不同 batch size 的 padding 浪费比
3. Graph 模式 vs Eager 模式性能对比（>max_size 时才真正 eager）
4. Graph 编译耗时分析

用法：
  python3 exp_a_acl_graph.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import time
import math


def test1_graph_config(llm):
    """分析 graph 配置：编译了哪些 shape、pad 映射关系。"""
    compile_config = llm.llm_engine.vllm_config.compilation_config
    compiled_sizes = sorted(compile_config.cudagraph_capture_sizes)

    # 构建 pad 映射表（模拟 vLLM 内部逻辑）
    max_size = max(compiled_sizes)
    pad_map = {}  # actual_size -> padded_size
    for actual in range(1, max_size + 1):
        # 找到 >= actual 的最小已编译 size
        padded = None
        for s in compiled_sizes:
            if s >= actual:
                padded = s
                break
        pad_map[actual] = padded if padded else max_size

    # 分析 padding 浪费
    waste_examples = []
    for actual in [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 20, 25, 30, 32, 50, 64, 100]:
        if actual <= max_size:
            padded = pad_map[actual]
            waste_pct = (padded - actual) / padded * 100
            waste_examples.append({
                "actual": actual,
                "padded_to": padded,
                "waste_pct": round(waste_pct, 1),
            })

    # 统计：平均间距
    gaps = [compiled_sizes[i+1] - compiled_sizes[i] for i in range(len(compiled_sizes)-1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    print("\n" + "=" * 60)
    print("Test 1: ACL Graph Configuration & Padding Analysis")
    print("=" * 60)
    print(f"  Compiled shapes: {len(compiled_sizes)}")
    print(f"  Range: [{compiled_sizes[0]}, {compiled_sizes[-1]}]")
    print(f"  Avg gap between shapes: {avg_gap:.1f}")
    print(f"\n  Padding examples:")
    print(f"  {'Actual':>8s}  {'Padded':>8s}  {'Waste%':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for ex in waste_examples:
        print(f"  {ex['actual']:>8d}  {ex['padded_to']:>8d}  {ex['waste_pct']:>7.1f}%")

    # TDM 相关分析：P phase 和 D phase 典型 batch size 范围
    print(f"\n  TDM 场景分析：")
    print(f"  - Prefill phase: 通常 1-8 个新请求")
    print(f"  - Decode phase: 通常 1-64 个正在生成的请求")
    for phase, typical_range in [("Prefill", range(1, 9)), ("Decode", range(1, 65))]:
        wastes = [(pad_map[n] - n) / pad_map[n] * 100 for n in typical_range if n <= max_size]
        avg_waste = sum(wastes) / len(wastes)
        print(f"  - {phase} 平均 padding 浪费: {avg_waste:.1f}%")

    return {
        "compiled_sizes": compiled_sizes,
        "num_sizes": len(compiled_sizes),
        "max_size": max_size,
        "avg_gap": round(avg_gap, 1),
        "waste_examples": waste_examples,
    }


def test2_padding_impact(llm, num_iters=8):
    """测量 padding 对实际吞吐的影响。
    对比：刚好匹配 graph shape 的 bs vs 需要大量 padding 的 bs。
    """
    from vllm import SamplingParams

    compile_config = llm.llm_engine.vllm_config.compilation_config
    compiled_sizes = sorted(compile_config.cudagraph_capture_sizes)

    print("\n" + "=" * 60)
    print("Test 2: Padding Impact on Throughput")
    print("=" * 60)

    prompt = "Hello"
    sp = SamplingParams(max_tokens=50, temperature=0.0)

    # 测试：精确匹配 vs 刚好多 1（最大浪费场景）
    test_cases = []
    for s in compiled_sizes[:8]:  # 前 8 个编译 shape
        test_cases.append(("exact", s))
        if s > 1:
            test_cases.append(("pad+1", s + 1))  # 会 pad 到下一个 shape

    # 去重并排序
    seen = set()
    unique_cases = []
    for label, bs in test_cases:
        if bs not in seen and bs <= 64:  # 限制到 64 避免太慢
            seen.add(bs)
            # 找到实际 padded size
            padded = bs
            for s in compiled_sizes:
                if s >= bs:
                    padded = s
                    break
            unique_cases.append((label, bs, padded))
    unique_cases.sort(key=lambda x: x[1])

    results = []
    for label, bs, padded in unique_cases:
        prompts = [prompt] * bs
        waste_pct = (padded - bs) / padded * 100

        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = llm.generate(prompts, sp)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_time = sum(times[2:]) / len(times[2:])  # skip warmup
        total_out = sum(len(o.outputs[0].token_ids) for o in out)
        tps = total_out / avg_time
        per_tok = avg_time * 1000 / (total_out / bs) if total_out > 0 else 0

        entry = {
            "label": label,
            "actual_bs": bs,
            "padded_bs": padded,
            "waste_pct": round(waste_pct, 1),
            "throughput_tps": round(tps, 1),
            "per_tok_ms": round(per_tok, 2),
            "time_ms": round(avg_time * 1000, 1),
        }
        results.append(entry)
        print(f"  bs={bs:>3d} → pad={padded:>3d} (waste {waste_pct:>5.1f}%) | "
              f"{tps:>7.1f} tok/s | per_tok={per_tok:.2f}ms  [{label}]")

    return results


def test3_eager_boundary(llm, num_iters=5):
    """测试超过 max_compiled_size 时的 eager fallback 退化。
    这才是真正的 eager vs graph 对比。
    """
    from vllm import SamplingParams

    compile_config = llm.llm_engine.vllm_config.compilation_config
    compiled_sizes = sorted(compile_config.cudagraph_capture_sizes)
    max_size = max(compiled_sizes)

    print("\n" + "=" * 60)
    print(f"Test 3: Eager Boundary (max compiled size = {max_size})")
    print("=" * 60)

    prompt = "Hello"
    sp = SamplingParams(max_tokens=20, temperature=0.0)  # 短生成，聚焦 per-token

    # 测试：max_size 附近 + 超过 max_size
    test_sizes = []
    # max_size 以内（走 graph）
    for s in [max_size - 16, max_size]:
        if s > 0:
            test_sizes.append(s)
    # 超过 max_size（走 eager）
    for s in [max_size + 1, max_size + 16, max_size + 32]:
        test_sizes.append(s)

    results = []
    for bs in test_sizes:
        is_graph = bs <= max_size
        prompts = [prompt] * bs

        try:
            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)

            avg_time = sum(times[1:]) / len(times[1:])
            total_out = sum(len(o.outputs[0].token_ids) for o in out)
            tps = total_out / avg_time
            per_tok = avg_time * 1000 / (total_out / bs) if total_out > 0 else 0

            mode = "GRAPH" if is_graph else "EAGER"
            entry = {
                "batch_size": bs,
                "mode": mode,
                "throughput_tps": round(tps, 1),
                "per_tok_ms": round(per_tok, 2),
                "time_ms": round(avg_time * 1000, 1),
            }
            results.append(entry)
            print(f"  bs={bs:>4d} [{mode:5s}] | {tps:>7.1f} tok/s | per_tok={per_tok:.2f}ms")
        except Exception as e:
            print(f"  bs={bs:>4d} FAILED: {e}")
            results.append({"batch_size": bs, "error": str(e)})

    # 对比
    graph_results = [r for r in results if r.get("mode") == "GRAPH"]
    eager_results = [r for r in results if r.get("mode") == "EAGER"]
    if graph_results and eager_results:
        g_avg_pt = sum(r["per_tok_ms"] for r in graph_results) / len(graph_results)
        e_avg_pt = sum(r["per_tok_ms"] for r in eager_results) / len(eager_results)
        degradation = (e_avg_pt - g_avg_pt) / g_avg_pt * 100
        print(f"\n  Graph avg per_tok: {g_avg_pt:.2f}ms")
        print(f"  Eager avg per_tok: {e_avg_pt:.2f}ms")
        print(f"  Eager degradation: {degradation:+.1f}%")

    return results


def test4_compile_time(model_path, gpu_mem=0.9):
    """测量 graph 编译耗时：对比 enforce_eager（无编译）vs 默认模式（有编译）。
    差值 = graph 编译总耗时。
    """
    from vllm import LLM

    print("\n" + "=" * 60)
    print("Test 4: Graph Compilation Time")
    print("=" * 60)

    results = {}

    for mode_name, enforce_eager in [("eager_load", True), ("graph_load", False)]:
        print(f"  Loading model ({mode_name})...")
        t0 = time.time()
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_mem,
            max_model_len=4096,
            enforce_eager=enforce_eager,
        )
        load_time = time.time() - t0
        results[mode_name] = round(load_time, 1)
        print(f"    Time: {load_time:.1f}s")

        del llm
        import gc, torch
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(2)

    compile_time = results["graph_load"] - results["eager_load"]
    results["compile_overhead_s"] = round(compile_time, 1)
    print(f"\n  Graph compilation overhead: {compile_time:.1f}s")
    print(f"  (graph_load {results['graph_load']:.1f}s - eager_load {results['eager_load']:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp A v2: ACL Graph Analysis")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-4B")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--skip-compile-test", action="store_true",
                        help="Skip Test 4 (requires reloading model twice)")
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_a_v2.json")
    args = parser.parse_args()

    from vllm import LLM

    all_results = {"model": os.path.basename(args.model)}

    # Load model with graph mode
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
    from vllm import SamplingParams
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=10, temperature=0.0))

    # Tests 1-3
    all_results["test1_config"] = test1_graph_config(llm)
    all_results["test2_padding"] = test2_padding_impact(llm)
    all_results["test3_eager_boundary"] = test3_eager_boundary(llm)

    # Cleanup
    del llm
    import gc, torch
    gc.collect()
    torch.npu.empty_cache()

    # Test 4
    if not args.skip_compile_test:
        all_results["test4_compile"] = test4_compile_time(args.model, args.gpu_mem)

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {args.output}")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT A v2 SUMMARY")
    print("=" * 60)
    cfg = all_results["test1_config"]
    print(f"  Model: {all_results['model']}")
    print(f"  Compiled graph shapes: {cfg['num_sizes']} (range 1-{cfg['max_size']})")
    print(f"  Avg gap between shapes: {cfg['avg_gap']}")
    print(f"\n  对 TDM 的影响：")
    print(f"  1. batch_size ≤ {cfg['max_size']} 时自动 pad 到已编译 shape，走 graph（快）")
    print(f"  2. batch_size > {cfg['max_size']} 时 eager fallback（慢）")
    print(f"  3. Graph-Aware Batch Shaping 的目标：减少 padding 浪费，不是避免 eager")


if __name__ == "__main__":
    main()
