"""
Baseline Benchmark: Qwen3-0.6B / Qwen3-4B on single NPU
Collects: TTFT, TPOT, Throughput, E2E latency
Strategy: measure prefill via max_tokens=1, then full gen for TPOT
"""
import os
import time
import json
import argparse
from dataclasses import dataclass, asdict

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@dataclass
class BenchResult:
    model: str
    num_prompts: int
    avg_input_len: int
    max_output_tokens: int
    prompt_set: str
    # latency (per-request averages)
    avg_ttft_ms: float
    avg_tpot_ms: float
    avg_e2e_ms: float
    # throughput (from batch run)
    total_input_tokens: int
    total_output_tokens: int
    batch_time_s: float
    input_throughput_tps: float
    output_throughput_tps: float


PROMPTS_SHORT = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
    "Explain the concept of machine learning in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Java?",
    "Describe how a neural network works.",
    "List three benefits of cloud computing.",
]

PROMPTS_LONG = [
    "Please write a detailed explanation of how transformer neural networks work, "
    "including the attention mechanism, positional encoding, and the encoder-decoder architecture. "
    "Cover the key innovations that made transformers successful compared to previous architectures "
    "like RNNs and LSTMs. Discuss applications in NLP, computer vision, and other domains.",

    "Describe the history of artificial intelligence from its inception in the 1950s to the present day. "
    "Include key milestones such as the Turing test, expert systems, the AI winters, "
    "the rise of deep learning, and recent breakthroughs in large language models. "
    "Discuss the contributions of notable researchers and organizations.",

    "Explain the concept of distributed systems in computer science. "
    "Cover topics including consistency models, CAP theorem, consensus algorithms like Paxos and Raft, "
    "distributed databases, microservices architecture, and the challenges of building reliable "
    "distributed systems at scale.",

    "Write an in-depth analysis of the current state of renewable energy technologies. "
    "Compare solar, wind, hydroelectric, and geothermal power in terms of efficiency, cost, "
    "scalability, and environmental impact. Discuss the challenges of energy storage and grid integration.",

    "Explain the theory behind large language models including pre-training, fine-tuning, "
    "reinforcement learning from human feedback, and scaling laws. Discuss the emergent capabilities "
    "observed in larger models and the ongoing debate about whether these models truly understand language.",
]


def build_prompts(prompt_set: str, num_prompts: int) -> list[str]:
    pool = PROMPTS_SHORT if prompt_set == "short" else PROMPTS_LONG
    return [pool[i % len(pool)] for i in range(num_prompts)]


def run_benchmark(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    prompt_set: str,
    gpu_memory_utilization: float = 0.9,
) -> BenchResult:
    from vllm import LLM, SamplingParams

    sampling_full = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    sampling_one = SamplingParams(max_tokens=1, temperature=0.0)

    model_name = os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}, max_tokens: {max_tokens}, set: {prompt_set}")
    print(f"{'='*60}")

    print("Loading model...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Warmup
    print("Warmup...")
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=10, temperature=0.0))
    print("Warmup done.")

    # === 1. Measure TTFT: generate only 1 token per prompt ===
    print("Measuring TTFT (1-token generation)...")
    ttfts = []
    for i, prompt in enumerate(prompts):
        t_start = time.time()
        _ = llm.generate([prompt], sampling_one)
        ttft = (time.time() - t_start) * 1000
        ttfts.append(ttft)

    avg_ttft = sum(ttfts) / len(ttfts)
    print(f"  Avg TTFT: {avg_ttft:.1f} ms (per prompt, sequential)")

    # === 2. Measure E2E and TPOT: full generation per prompt ===
    print(f"Measuring E2E + TPOT ({max_tokens} tokens)...")
    e2es = []
    tpots = []
    actual_output_lens = []
    actual_input_lens = []
    for i, prompt in enumerate(prompts):
        t_start = time.time()
        out = llm.generate([prompt], sampling_full)
        e2e = (time.time() - t_start) * 1000
        e2es.append(e2e)

        n_out = len(out[0].outputs[0].token_ids)
        n_in = len(out[0].prompt_token_ids) if out[0].prompt_token_ids else 0
        actual_output_lens.append(n_out)
        actual_input_lens.append(n_in)

        if n_out > 1:
            # TPOT = (E2E - TTFT) / (n_out - 1)
            tpot = (e2e - ttfts[i]) / (n_out - 1)
            tpots.append(tpot)

    avg_e2e = sum(e2es) / len(e2es)
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0

    # === 3. Measure batch throughput ===
    print("Measuring batch throughput...")
    t_start = time.time()
    batch_outputs = llm.generate(prompts, sampling_full)
    batch_time = time.time() - t_start

    total_in = sum(len(o.prompt_token_ids) for o in batch_outputs if o.prompt_token_ids)
    total_out = sum(len(o.outputs[0].token_ids) for o in batch_outputs)

    result = BenchResult(
        model=model_name,
        num_prompts=len(prompts),
        avg_input_len=sum(actual_input_lens) // len(actual_input_lens),
        max_output_tokens=max_tokens,
        prompt_set=prompt_set,
        avg_ttft_ms=round(avg_ttft, 2),
        avg_tpot_ms=round(avg_tpot, 2),
        avg_e2e_ms=round(avg_e2e, 2),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        batch_time_s=round(batch_time, 3),
        input_throughput_tps=round(total_in / batch_time, 2),
        output_throughput_tps=round(total_out / batch_time, 2),
    )

    print(f"\n--- {model_name} Results ---")
    print(f"Avg TTFT:          {result.avg_ttft_ms:.1f} ms")
    print(f"Avg TPOT:          {result.avg_tpot_ms:.2f} ms")
    print(f"Avg E2E:           {result.avg_e2e_ms:.1f} ms")
    print(f"Batch input tput:  {result.input_throughput_tps:.1f} tokens/s")
    print(f"Batch output tput: {result.output_throughput_tps:.1f} tokens/s")
    print(f"Batch time:        {result.batch_time_s:.2f}s")
    print(f"Tokens (in/out):   {result.total_input_tokens}/{result.total_output_tokens}")

    # Cleanup
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.npu.empty_cache()
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for PD research")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--prompt-set", choices=["short", "long"], default="short")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run both models × both prompt sets")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    args = parser.parse_args()

    all_results = []

    if args.all_configs:
        models = [
            "/vllm-workspace/models/models/Qwen3-0.6B",
            "/vllm-workspace/models/models/Qwen3-4B",
        ]
        configs = [
            {"prompt_set": "short", "max_tokens": 200, "num_prompts": 8},
            {"prompt_set": "long", "max_tokens": 200, "num_prompts": 5},
        ]
        for model in models:
            for cfg in configs:
                prompts = build_prompts(cfg["prompt_set"], cfg["num_prompts"])
                result = run_benchmark(
                    model_path=model,
                    prompts=prompts,
                    max_tokens=cfg["max_tokens"],
                    prompt_set=cfg["prompt_set"],
                    gpu_memory_utilization=args.gpu_mem,
                )
                all_results.append(asdict(result))
    else:
        prompts = build_prompts(args.prompt_set, args.num_prompts)
        result = run_benchmark(
            model_path=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            prompt_set=args.prompt_set,
            gpu_memory_utilization=args.gpu_mem,
        )
        all_results.append(asdict(result))

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    header = (f"{'Model':<14} {'Set':<6} {'N':>3} {'InLen':>5} {'OutTok':>6} "
              f"{'TTFT(ms)':>9} {'TPOT(ms)':>9} {'E2E(ms)':>9} {'OutTPS':>8}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['model']:<14} {r['prompt_set']:<6} {r['num_prompts']:>3} "
              f"{r['avg_input_len']:>5} {r['max_output_tokens']:>6} "
              f"{r['avg_ttft_ms']:>9.1f} {r['avg_tpot_ms']:>9.2f} "
              f"{r['avg_e2e_ms']:>9.1f} {r['output_throughput_tps']:>8.1f}")

    # Save
    output_path = args.output or "/vllm-workspace/lzn-pro/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
