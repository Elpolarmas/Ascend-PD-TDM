"""
Experiment C: DCMI Sampling Overhead on Inference
==================================================
验证：在推理过程中做 DCMI 采样，吞吐下降是否可忽略

对比：
1. 无采样 baseline
2. 每 iteration 采样 HBM BW（~1.1ms/call）
3. 同时采样 HBM BW + AICore（异步线程）

用法：
  python exp_c_dcmi_overhead.py --model /vllm-workspace/models/models/Qwen3-0.6B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import ctypes
import json
import threading
import time


class DCMIPoller:
    """Simulates DCMI polling at different frequencies during inference."""

    def __init__(self):
        self.dcmi = ctypes.CDLL("/usr/local/dcmi/libdcmi.so")
        self.dcmi.dcmi_init()
        self.dcmi.dcmi_get_device_utilization_rate.restype = ctypes.c_int
        self.dcmi.dcmi_get_device_utilization_rate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint)
        ]
        self._running = False
        self._thread = None
        self.call_count = 0

    def _poll_loop(self, metric_type, interval):
        rate = ctypes.c_uint(0)
        while self._running:
            self.dcmi.dcmi_get_device_utilization_rate(
                0, 0, metric_type, ctypes.byref(rate))
            self.call_count += 1
            time.sleep(interval)

    def start(self, metric_type=10, interval=0.002):
        self.call_count = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, args=(metric_type, interval), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


def run_bench(llm, label, poller_config=None, num_rounds=8):
    """Run inference benchmark with optional DCMI polling."""
    from vllm import SamplingParams

    prompts = ["Hello, explain machine learning briefly."] * 16
    sp = SamplingParams(max_tokens=200, temperature=0.0)

    # Warmup
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=10, temperature=0.0))

    poller = DCMIPoller() if poller_config else None

    times = []
    total_tokens_list = []

    for _ in range(num_rounds):
        if poller and poller_config:
            poller.start(**poller_config)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0

        if poller:
            poller.stop()

        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        times.append(elapsed)
        total_tokens_list.append(total_out)

    # Skip first round
    avg_time = sum(times[1:]) / len(times[1:])
    avg_tokens = sum(total_tokens_list[1:]) / len(total_tokens_list[1:])
    tps = avg_tokens / avg_time

    result = {
        "label": label,
        "avg_time_ms": round(avg_time * 1000, 2),
        "avg_output_tokens": round(avg_tokens, 1),
        "throughput_tps": round(tps, 1),
        "dcmi_calls": poller.call_count if poller else 0,
    }

    print(f"  {label:<30s} | time={avg_time*1000:.1f}ms | {tps:.1f} tok/s"
          f" | dcmi_calls={result['dcmi_calls']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Exp C: DCMI Overhead")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-0.6B")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_c.json")
    args = parser.parse_args()

    from vllm import LLM

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
    )

    print("\n" + "=" * 60)
    print("Experiment C: DCMI Sampling Overhead")
    print("=" * 60)

    results = []

    # 1. No DCMI sampling (baseline)
    results.append(run_bench(llm, "no_sampling"))

    # 2. HBM BW only, every 2ms
    results.append(run_bench(llm, "hbm_bw_2ms",
                             poller_config={"metric_type": 10, "interval": 0.002}))

    # 3. HBM BW only, every 5ms
    results.append(run_bench(llm, "hbm_bw_5ms",
                             poller_config={"metric_type": 10, "interval": 0.005}))

    # 4. HBM BW only, every 10ms
    results.append(run_bench(llm, "hbm_bw_10ms",
                             poller_config={"metric_type": 10, "interval": 0.01}))

    # 5. HBM BW only, every 20ms
    results.append(run_bench(llm, "hbm_bw_20ms",
                             poller_config={"metric_type": 10, "interval": 0.02}))

    # 6. AICore only, every 100ms
    results.append(run_bench(llm, "aicore_100ms",
                             poller_config={"metric_type": 2, "interval": 0.1}))

    # 7. Combined: HBM BW 10ms + AICore 100ms (both threads)
    # Run with HBM BW thread only for now; combined test would need two pollers

    # Summary
    baseline_tps = results[0]["throughput_tps"]

    all_results = {"model": os.path.basename(args.model), "results": results}

    print(f"\n  {'Config':<30s}  {'TPS':>8s}  {'Overhead':>9s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}")
    for r in results:
        overhead = (1 - r["throughput_tps"] / baseline_tps) * 100 if baseline_tps > 0 else 0
        print(f"  {r['label']:<30s}  {r['throughput_tps']:>7.1f}  {overhead:>8.2f}%")
        r["overhead_pct"] = round(overhead, 2)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
