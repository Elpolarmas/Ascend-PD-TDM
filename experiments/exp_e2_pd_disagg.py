"""
Experiment E2: PD Disaggregated (1P1D) — 两卡分离 vs 单卡 Unified/Phased
=========================================================================
Device 0 = Prefill (kv_producer), Device 1 = Decode (kv_consumer)
使用 LLMDataDistCMgrConnector 通过 llm_datadist 传输 KV cache。

前置条件：
  - ranktable.json 已生成（见 ranktable.json）
  - 2 × Ascend 910B3 可用

用法：
  python exp_e2_pd_disagg.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import time
import multiprocessing as mp
from multiprocessing import Event, Process, Queue


RANKTABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ranktable.json")


def run_prefill_worker(model, gpu_mem, max_model_len, workloads, ranktable_path,
                       prefill_done_event, all_done_event, result_queue):
    """Prefill worker: runs on device 0, produces KV cache."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["DISAGGREGATED_PREFILL_RANK_TABLE_PATH"] = ranktable_path

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    ktc = KVTransferConfig(
        kv_connector="LLMDataDistCMgrConnector",
        kv_buffer_device="npu",
        kv_role="kv_producer",
        kv_parallel_size=1,
        kv_connector_module_path="vllm_ascend.distributed.llmdatadist_c_mgr_connector",
    )

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        kv_transfer_config=ktc,
    )

    # Prefill only generates 1 token (trigger KV transfer)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    for wl_name, prompts, max_tokens, num_rounds in workloads:
        for round_idx in range(num_rounds):
            # Signal decode worker that we're about to send
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            prefill_time = time.perf_counter() - t0

            total_input_tokens = sum(
                len(o.prompt_token_ids) for o in outputs if o.prompt_token_ids
            )
            result_queue.put({
                "type": "prefill",
                "workload": wl_name,
                "round": round_idx,
                "prefill_time_s": prefill_time,
                "input_tokens": total_input_tokens,
                "num_requests": len(prompts),
            })
            # Signal decode that prefill is done for this round
            prefill_done_event.set()

            # Wait for decode to finish this round before next
            while prefill_done_event.is_set():
                time.sleep(0.05)

    # Keep alive until decode finishes all work
    try:
        while not all_done_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        del llm
        _clean_up()


def run_decode_worker(model, gpu_mem, max_model_len, workloads, ranktable_path,
                      prefill_done_event, all_done_event, result_queue):
    """Decode worker: runs on device 1, consumes KV cache and generates."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["DISAGGREGATED_PREFILL_RANK_TABLE_PATH"] = ranktable_path
    os.environ["VLLM_ASCEND_LLMDD_RPC_PORT"] = "6634"

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    ktc = KVTransferConfig(
        kv_connector="LLMDataDistCMgrConnector",
        kv_buffer_device="npu",
        kv_role="kv_consumer",
        kv_parallel_size=1,
        kv_connector_module_path="vllm_ascend.distributed.llmdatadist_c_mgr_connector",
    )

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        kv_transfer_config=ktc,
    )

    for wl_name, prompts, max_tokens, num_rounds in workloads:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        for round_idx in range(num_rounds):
            # Wait for prefill to finish
            prefill_done_event.wait()

            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            decode_time = time.perf_counter() - t0

            total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
            result_queue.put({
                "type": "decode",
                "workload": wl_name,
                "round": round_idx,
                "decode_time_s": decode_time,
                "output_tokens": total_out,
                "num_requests": len(prompts),
            })
            # Clear event to let prefill proceed to next round
            prefill_done_event.clear()

    all_done_event.set()
    del llm
    _clean_up()


def _clean_up():
    import gc
    import torch
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel)
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


def aggregate_results(result_queue, workloads):
    """Collect results from queue and compute per-workload metrics."""
    prefill_data = {}  # wl_name -> [round_data]
    decode_data = {}

    while not result_queue.empty():
        item = result_queue.get_nowait()
        wl = item["workload"]
        if item["type"] == "prefill":
            prefill_data.setdefault(wl, []).append(item)
        else:
            decode_data.setdefault(wl, []).append(item)

    results = []
    for wl_name, _, max_tokens, num_rounds in workloads:
        p_rounds = sorted(prefill_data.get(wl_name, []), key=lambda x: x["round"])
        d_rounds = sorted(decode_data.get(wl_name, []), key=lambda x: x["round"])

        if len(p_rounds) < 2 or len(d_rounds) < 2:
            print(f"  WARNING: {wl_name} has insufficient rounds, skipping")
            continue

        # Skip first round (warmup)
        p_rounds = p_rounds[1:]
        d_rounds = d_rounds[1:]

        avg_prefill_time = sum(r["prefill_time_s"] for r in p_rounds) / len(p_rounds)
        avg_decode_time = sum(r["decode_time_s"] for r in d_rounds) / len(d_rounds)
        avg_total_time = avg_prefill_time + avg_decode_time
        avg_input_tokens = sum(r["input_tokens"] for r in p_rounds) / len(p_rounds)
        avg_output_tokens = sum(r["output_tokens"] for r in d_rounds) / len(d_rounds)
        num_requests = d_rounds[0]["num_requests"]

        # TTFT ≈ prefill_time / num_requests (all requests prefilled together)
        avg_ttft_ms = avg_prefill_time * 1000 / num_requests
        # TPOT ≈ (decode_time - 0) / (output_tokens_per_req)
        tokens_per_req = avg_output_tokens / num_requests
        avg_tpot_ms = avg_decode_time * 1000 / num_requests / max(tokens_per_req - 1, 1)
        # TPS = output_tokens / total_e2e_time
        tps = avg_output_tokens / avg_total_time

        results.append({
            "label": f"disagg_{wl_name}",
            "num_requests": num_requests,
            "avg_prefill_time_s": round(avg_prefill_time, 3),
            "avg_decode_time_s": round(avg_decode_time, 3),
            "avg_total_time_s": round(avg_total_time, 3),
            "avg_input_tokens": round(avg_input_tokens),
            "avg_output_tokens": round(avg_output_tokens),
            "throughput_tps": round(tps, 1),
            "avg_ttft_ms": round(avg_ttft_ms, 2),
            "avg_tpot_ms": round(avg_tpot_ms, 2),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp E2: PD Disaggregated 1P1D (2-card)")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-4B")
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_e2.json")
    args = parser.parse_args()

    short_prompt = "Hello, explain machine learning briefly."
    long_prompt = (
        "Please write a comprehensive analysis of the global economic trends "
        "in the 21st century, covering topics such as globalization, technological "
        "disruption, income inequality, climate change impacts, geopolitical shifts, "
        "and the rise of emerging markets. Discuss how these factors interconnect "
        "and influence each other in complex ways."
    )

    # (name, prompts, max_tokens, num_rounds)
    workloads = [
        ("short_4req", [short_prompt] * 4, 200, args.num_rounds),
        ("short_16req", [short_prompt] * 16, 200, args.num_rounds),
        ("short_32req", [short_prompt] * 32, 200, args.num_rounds),
        ("long_4req", [long_prompt] * 4, 200, args.num_rounds),
        ("long_16req", [long_prompt] * 16, 200, args.num_rounds),
        ("mixed_16req",
         [short_prompt] * 8 + [long_prompt] * 8, 200, args.num_rounds),
    ]

    print("=" * 70)
    print("Experiment E2: PD Disaggregated 1P1D")
    print(f"  Model: {args.model}")
    print(f"  Ranktable: {RANKTABLE_PATH}")
    print(f"  Device 0 = Prefill, Device 1 = Decode")
    print(f"  Rounds per workload: {args.num_rounds} (first discarded)")
    print("=" * 70)

    ctx = mp.get_context("spawn")
    prefill_done = ctx.Event()
    all_done = ctx.Event()
    result_queue = ctx.Queue()

    prefill_proc = ctx.Process(
        target=run_prefill_worker,
        args=(args.model, args.gpu_mem, args.max_model_len, workloads,
              RANKTABLE_PATH, prefill_done, all_done, result_queue),
    )
    decode_proc = ctx.Process(
        target=run_decode_worker,
        args=(args.model, args.gpu_mem, args.max_model_len, workloads,
              RANKTABLE_PATH, prefill_done, all_done, result_queue),
    )

    print("\nStarting prefill worker (device 0)...")
    prefill_proc.start()
    print("Starting decode worker (device 1)...")
    decode_proc.start()

    decode_proc.join()
    prefill_proc.join(timeout=10)
    if prefill_proc.is_alive():
        prefill_proc.terminate()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    disagg_results = aggregate_results(result_queue, workloads)

    print(f"\n  {'Workload':<16s}  {'TPS':>8s}  {'TTFT(ms)':>9s}  "
          f"{'TPOT(ms)':>9s}  {'P_time(s)':>10s}  {'D_time(s)':>10s}  {'Total(s)':>8s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}")

    for r in disagg_results:
        wl = r['label'].replace('disagg_', '')
        print(f"  {wl:<16s}  {r['throughput_tps']:>8.1f}  "
              f"{r['avg_ttft_ms']:>9.1f}  {r['avg_tpot_ms']:>9.2f}  "
              f"{r['avg_prefill_time_s']:>10.3f}  {r['avg_decode_time_s']:>10.3f}  "
              f"{r['avg_total_time_s']:>8.3f}")

    # Load exp_e results for comparison if available
    exp_e_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results_exp_e.json")
    if os.path.exists(exp_e_path):
        with open(exp_e_path) as f:
            exp_e = json.load(f)
        unified_map = {r['label'].replace('unified_', ''): r
                       for r in exp_e.get("experiments", {}).get("unified", [])}
        phased_map = {r['label'].replace('phased_', ''): r
                      for r in exp_e.get("experiments", {}).get("phased", [])}

        print(f"\n\n{'='*70}")
        print("COMPARISON: Unified vs Phased vs Disaggregated (1P1D)")
        print(f"{'='*70}")
        print(f"  {'Workload':<16s}  {'Unified':>10s}  {'Phased':>10s}  "
              f"{'Disagg':>10s}  {'vs Unified':>11s}  {'vs Phased':>11s}")
        print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*11}")

        for r in disagg_results:
            wl = r['label'].replace('disagg_', '')
            d_tps = r['throughput_tps']
            u_tps = unified_map.get(wl, {}).get('throughput_tps', 0)
            p_tps = phased_map.get(wl, {}).get('throughput_tps', 0)

            vs_u = f"{(d_tps/u_tps - 1)*100:+.1f}%" if u_tps > 0 else "N/A"
            vs_p = f"{(d_tps/p_tps - 1)*100:+.1f}%" if p_tps > 0 else "N/A"

            print(f"  {wl:<16s}  {u_tps:>10.1f}  {p_tps:>10.1f}  "
                  f"{d_tps:>10.1f}  {vs_u:>11s}  {vs_p:>11s}")

    output_data = {
        "model": os.path.basename(args.model),
        "mode": "disaggregated_1p1d",
        "device_prefill": 0,
        "device_decode": 1,
        "experiments": disagg_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
