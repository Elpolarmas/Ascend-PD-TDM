"""
Experiment B: NPU P/D Resource Utilization Profile
====================================================
支撑 motivation："Prefill 吃算力不吃带宽，Decode 吃带宽不吃算力"

测试内容：
1. 纯 Prefill 场景下的 AICore% 和 HBM BW%（大量 prompt，max_tokens=1）
2. 纯 Decode 场景下的 AICore% 和 HBM BW%（短 prompt，长生成）
3. 混合场景作为对照

采集方式：
- DCMI 后台线程持续采样 HBM BW（~1.1ms/call）和 AICore（~60ms/call）
- 前台跑推理，结束后汇总统计

用法：
  python exp_b_pd_profile.py --model /vllm-workspace/models/models/Qwen3-0.6B
  python exp_b_pd_profile.py --model /vllm-workspace/models/models/Qwen3-4B
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import ctypes
import json
import threading
import time


# ============================================================
# DCMI Sampler (background thread)
# ============================================================
class DCMISampler:
    """Background thread that samples DCMI metrics during inference."""

    # DCMI metric type IDs
    METRIC_HBM_BW = 10
    METRIC_AICORE = 2
    METRIC_HBM_USAGE = 6

    def __init__(self, device_id=0, hbm_bw_interval=0.005, aicore_interval=0.1):
        self.device_id = device_id
        self.hbm_bw_interval = hbm_bw_interval  # ~5ms between HBM BW samples
        self.aicore_interval = aicore_interval    # ~100ms between AICore samples

        self.dcmi = ctypes.CDLL("/usr/local/dcmi/libdcmi.so")
        self.dcmi.dcmi_init()
        self.dcmi.dcmi_get_device_utilization_rate.restype = ctypes.c_int
        self.dcmi.dcmi_get_device_utilization_rate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint)
        ]

        self.hbm_bw_samples = []
        self.aicore_samples = []
        self._running = False
        self._thread_hbm = None
        self._thread_aicore = None

    def _read_metric(self, metric_type):
        rate = ctypes.c_uint(0)
        ret = self.dcmi.dcmi_get_device_utilization_rate(
            0, self.device_id, metric_type, ctypes.byref(rate))
        if ret == 0:
            return rate.value
        return None

    def _sample_hbm_bw(self):
        while self._running:
            val = self._read_metric(self.METRIC_HBM_BW)
            if val is not None:
                self.hbm_bw_samples.append((time.perf_counter(), val))
            time.sleep(self.hbm_bw_interval)

    def _sample_aicore(self):
        while self._running:
            val = self._read_metric(self.METRIC_AICORE)
            if val is not None:
                self.aicore_samples.append((time.perf_counter(), val))
            time.sleep(self.aicore_interval)

    def start(self):
        self.hbm_bw_samples = []
        self.aicore_samples = []
        self._running = True
        self._thread_hbm = threading.Thread(target=self._sample_hbm_bw, daemon=True)
        self._thread_aicore = threading.Thread(target=self._sample_aicore, daemon=True)
        self._thread_hbm.start()
        self._thread_aicore.start()

    def stop(self):
        self._running = False
        if self._thread_hbm:
            self._thread_hbm.join(timeout=2)
        if self._thread_aicore:
            self._thread_aicore.join(timeout=2)

    def summary(self):
        def stats(samples):
            if not samples:
                return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p99": 0}
            vals = [v for _, v in samples]
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            return {
                "count": n,
                "avg": round(sum(vals) / n, 1),
                "min": vals_sorted[0],
                "max": vals_sorted[-1],
                "p50": vals_sorted[n // 2],
                "p99": vals_sorted[int(n * 0.99)],
            }
        return {
            "hbm_bw": stats(self.hbm_bw_samples),
            "aicore": stats(self.aicore_samples),
        }


# ============================================================
# Workload runners
# ============================================================
def run_prefill_heavy(llm, num_requests=200):
    """Prefill-heavy: continuously submit unique prefill batches to keep NPU busy.
    Each prompt is unique (appended index) to defeat prefix caching.
    Uses a loop of generate() calls so NPU stays active for 10+ seconds."""
    from vllm import SamplingParams
    import random

    base = (
        "Please write a comprehensive analysis of the global economic trends "
        "in the 21st century, covering topics such as globalization, technological "
        "disruption, income inequality, climate change impacts, geopolitical shifts, "
        "and the rise of emerging markets. Discuss how these factors interconnect "
        "and influence each other in complex ways. "
    )
    long_base = (base * 10).strip()
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    batch_size = 50
    num_batches = num_requests // batch_size
    all_outputs = []
    for batch_idx in range(num_batches):
        # Make each prompt unique to avoid prefix cache hits
        prompts_batch = [
            f"[Request {batch_idx * batch_size + i}] {long_base}"
            for i in range(batch_size)
        ]
        outputs = llm.generate(prompts_batch, sp)
        all_outputs.extend(outputs)
    return all_outputs


def run_decode_heavy(llm, num_requests=16):
    """Decode-heavy: short prompts, long generation.
    This keeps the NPU busy doing decode most of the time.
    Use 16 requests × 512 tokens to sustain decode for ~10s+."""
    from vllm import SamplingParams

    short_prompts = ["Hi"] * num_requests
    sp = SamplingParams(max_tokens=512, temperature=0.0)
    return llm.generate(short_prompts, sp)


def run_mixed(llm):
    """Mixed workload: combination of short and long prompts."""
    from vllm import SamplingParams

    prompts = [
        "Hello, my name is",
        "Write a detailed explanation of quantum computing principles.",
        "Hi",
        "Explain the theory behind large language models including pre-training, "
        "fine-tuning, and reinforcement learning from human feedback.",
        "What is 1+1?",
    ]
    sp = SamplingParams(max_tokens=150, temperature=0.0)
    return llm.generate(prompts, sp)


def run_scenario(llm, scenario_name, runner_fn, sampler):
    """Run a scenario with DCMI sampling."""
    print(f"\n--- Scenario: {scenario_name} ---")

    # Warmup
    from vllm import SamplingParams
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=5, temperature=0.0))
    time.sleep(0.5)

    sampler.start()
    t0 = time.perf_counter()
    outputs = runner_fn(llm)
    elapsed = time.perf_counter() - t0
    sampler.stop()

    total_in = sum(len(o.prompt_token_ids) for o in outputs if o.prompt_token_ids)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)

    stats = sampler.summary()

    result = {
        "scenario": scenario_name,
        "elapsed_s": round(elapsed, 3),
        "num_requests": len(outputs),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "input_tps": round(total_in / elapsed, 1),
        "output_tps": round(total_out / elapsed, 1),
        "dcmi": stats,
    }

    print(f"  Time: {elapsed:.2f}s | In: {total_in} tok | Out: {total_out} tok")
    print(f"  AICore:  avg={stats['aicore']['avg']}% "
          f"min={stats['aicore']['min']}% max={stats['aicore']['max']}% "
          f"({stats['aicore']['count']} samples)")
    print(f"  HBM BW:  avg={stats['hbm_bw']['avg']}% "
          f"min={stats['hbm_bw']['min']}% max={stats['hbm_bw']['max']}% "
          f"({stats['hbm_bw']['count']} samples)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Exp B: P/D Resource Profile")
    parser.add_argument("--model", type=str,
                        default="/vllm-workspace/models/models/Qwen3-0.6B")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--output", type=str,
                        default="/vllm-workspace/lzn-pro/results_exp_b.json")
    args = parser.parse_args()

    from vllm import LLM

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
    )
    print("Model loaded.")

    sampler = DCMISampler()
    all_results = {"model": os.path.basename(args.model), "scenarios": []}

    # Run 3 scenarios
    all_results["scenarios"].append(
        run_scenario(llm, "prefill_heavy", run_prefill_heavy, sampler))
    all_results["scenarios"].append(
        run_scenario(llm, "decode_heavy", run_decode_heavy, sampler))
    all_results["scenarios"].append(
        run_scenario(llm, "mixed", run_mixed, sampler))

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n" + "=" * 70)
    print("EXPERIMENT B SUMMARY: P/D Resource Utilization Profile")
    print("=" * 70)
    print(f"  {'Scenario':<16s}  {'AICore%':>8s}  {'HBM_BW%':>8s}  "
          f"{'In_tok':>7s}  {'Out_tok':>8s}  {'Time(s)':>8s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")
    for s in all_results["scenarios"]:
        print(f"  {s['scenario']:<16s}  "
              f"{s['dcmi']['aicore']['avg']:>7.1f}%  "
              f"{s['dcmi']['hbm_bw']['avg']:>7.1f}%  "
              f"{s['total_input_tokens']:>7d}  "
              f"{s['total_output_tokens']:>8d}  "
              f"{s['elapsed_s']:>8.2f}")

    print("\n预期结果：")
    print("  - prefill_heavy: AICore% 高, HBM_BW% 低 → compute-bound")
    print("  - decode_heavy:  AICore% 低, HBM_BW% 高 → memory-bound")
    print("  - 两者互补 → TDM 有空间")


if __name__ == "__main__":
    main()
