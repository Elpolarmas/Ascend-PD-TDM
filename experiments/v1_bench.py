"""TDM V1 vs AscendScheduler 基准对比.

两种模式都注入 TDMScheduler：
  - baseline: enable_tdm=False + passive_tracker=True，跳过决策流水线，
    仅用 tracker 收集 per-req TTFT/TPOT，apples-to-apples 对比
  - tdm:      enable_tdm=True，完整决策流水线 + tracker

per-mode 结果落 results/tdm_trace/v1_bench_{mode}_(req|iter).jsonl，
both 模式跑完两轮后聚合到 results/results_v1_bench.json。
两轮分别独立 LLM 子进程，避免 NPU/engine 状态串扰。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL_PATH = "/vllm-workspace/models/models/Qwen3-8B"
RESULTS_DIR = Path("/vllm-workspace/Ascend-PD-TDM/results")
TRACE_DIR = RESULTS_DIR / "tdm_trace"
COMBINED_PATH = RESULTS_DIR / "results_v1_bench.json"

# 同时混入短 / 中等 prompt，保证 prefill/decode 阶段都有得调度
PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Explain time-division multiplexing in one paragraph.",
    "Summarize the theory of relativity briefly.",
    "List three benefits of cloud computing.",
    "Describe how a neural network works in simple terms.",
    "What are the main differences between Python and Java?",
    "Write a short poem about the ocean.",
    "Explain the concept of distributed systems including consensus algorithms "
    "like Paxos and Raft, and the CAP theorem.",
    "Describe the history of artificial intelligence from the 1950s to today, "
    "including key milestones such as the Turing test and the rise of deep learning.",
    "Write a detailed explanation of how transformer neural networks work, "
    "including attention, positional encoding and the encoder-decoder architecture.",
    "Compare solar, wind, and hydroelectric power in terms of efficiency, cost, "
    "and environmental impact, and discuss energy storage challenges.",
    "What is reinforcement learning? Explain the key components: agent, "
    "environment, state, action, and reward, with a simple example.",
    "Explain how HTTP/2 differs from HTTP/1.1, covering multiplexing, "
    "header compression, and server push.",
    "Describe the Linux process lifecycle: fork, exec, exit, and zombie states, "
    "and explain how a parent process reaps children.",
    "Write a brief comparison of B-trees and LSM-trees, focusing on "
    "read/write trade-offs and typical use cases.",
]

MAX_TOKENS = 128


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[idx]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p99": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p99": round(percentile(values, 0.99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def build_additional_config(mode: str, run_id: str) -> dict:
    common_tdm = {
        "telemetry_enabled": True,
        "telemetry_dir": str(TRACE_DIR),
        "run_id": run_id,
    }
    if mode == "baseline":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": False,
                "passive_tracker": True,
                **common_tdm,
            },
        }
    if mode == "tdm":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                **common_tdm,
            },
        }
    raise ValueError(f"unknown mode: {mode}")


def run_single_mode(mode: str) -> dict:
    from vllm import LLM, SamplingParams

    run_id = f"v1_bench_{mode}"
    additional_config = build_additional_config(mode, run_id)

    # 清旧 jsonl，避免追加污染
    for suffix in ("iter", "req"):
        p = TRACE_DIR / f"{run_id}_{suffix}.jsonl"
        if p.exists():
            p.unlink()

    print(f"[{mode}] loading LLM...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enforce_eager=False,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        additional_config=additional_config,
    )
    load_s = time.time() - t0
    print(f"[{mode}] loaded in {load_s:.1f}s", flush=True)

    # 一次 warmup，丢弃指标，避免 ACL graph capture / 首次 alloc 抬高 ttft
    print(f"[{mode}] warmup...", flush=True)
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.0))

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    print(f"[{mode}] running batch ({len(PROMPTS)} prompts, max_tokens={MAX_TOKENS})...",
          flush=True)
    t_batch = time.time()
    outs = llm.generate(PROMPTS, sp)
    batch_wall_s = time.time() - t_batch
    print(f"[{mode}] batch finished in {batch_wall_s:.2f}s", flush=True)

    total_in = sum(len(o.prompt_token_ids) for o in outs if o.prompt_token_ids)
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)

    # 把 telemetry flush 到磁盘
    del llm
    import gc
    gc.collect()
    time.sleep(0.5)  # 等 telemetry writer 线程消化队列

    # 读 req.jsonl 算 per-req metric
    req_path = TRACE_DIR / f"{run_id}_req.jsonl"
    if not req_path.exists():
        raise RuntimeError(f"telemetry req file missing: {req_path}")

    # 过滤掉 warmup 那条（prompt 远短于真实 batch；也可按 admission 时间分段）
    # warmup 在 batch_start 之前 admit，可用 admission_ts 排序后取最后 N 条
    records = [json.loads(l) for l in req_path.read_text().splitlines() if l.strip()]
    records.sort(key=lambda r: r.get("admission_ts_ms", 0.0))
    bench_records = records[-len(PROMPTS):]  # 取批量最后 N 条
    if len(bench_records) != len(PROMPTS):
        print(f"[{mode}] WARN: expected {len(PROMPTS)} req records, got {len(bench_records)}",
              file=sys.stderr)

    ttfts = [r["ttft_ms"] for r in bench_records if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms_mean"] for r in bench_records if r.get("tpot_ms_mean") is not None]
    e2es = [r["finish_ts_ms"] - r["admission_ts_ms"] for r in bench_records
            if r.get("finish_ts_ms") is not None and r.get("admission_ts_ms") is not None]

    result = {
        "mode": mode,
        "model": MODEL_PATH,
        "tp": 2,
        "num_prompts": len(PROMPTS),
        "max_tokens": MAX_TOKENS,
        "load_time_s": round(load_s, 2),
        "batch_wall_s": round(batch_wall_s, 3),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "input_throughput_tps": round(total_in / batch_wall_s, 2),
        "output_throughput_tps": round(total_out / batch_wall_s, 2),
        "n_records_used": len(bench_records),
        "ttft_ms": summarize(ttfts),
        "tpot_ms": summarize(tpots),
        "e2e_ms": summarize(e2es),
    }

    out_path = TRACE_DIR / f"v1_bench_{mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{mode}] wrote {out_path}", flush=True)
    return result


def merge_results() -> dict:
    baseline = json.loads((TRACE_DIR / "v1_bench_baseline.json").read_text())
    tdm = json.loads((TRACE_DIR / "v1_bench_tdm.json").read_text())

    def pct(new, old):
        if old is None or new is None or old == 0:
            return None
        return round((new - old) / old * 100.0, 2)

    delta = {
        "ttft_mean_pct": pct(tdm["ttft_ms"]["mean"], baseline["ttft_ms"]["mean"]),
        "ttft_p99_pct": pct(tdm["ttft_ms"]["p99"], baseline["ttft_ms"]["p99"]),
        "tpot_mean_pct": pct(tdm["tpot_ms"]["mean"], baseline["tpot_ms"]["mean"]),
        "tpot_p99_pct": pct(tdm["tpot_ms"]["p99"], baseline["tpot_ms"]["p99"]),
        "e2e_mean_pct": pct(tdm["e2e_ms"]["mean"], baseline["e2e_ms"]["mean"]),
        "output_tput_pct": pct(tdm["output_throughput_tps"],
                               baseline["output_throughput_tps"]),
    }

    combined = {
        "workload": {
            "model": MODEL_PATH,
            "num_prompts": len(PROMPTS),
            "max_tokens": MAX_TOKENS,
        },
        "baseline": baseline,
        "tdm": tdm,
        "delta_tdm_vs_baseline": delta,
    }
    COMBINED_PATH.write_text(json.dumps(combined, indent=2))
    print(f"\nCombined → {COMBINED_PATH}")
    print(f"  TTFT mean:   baseline {baseline['ttft_ms']['mean']} ms → "
          f"tdm {tdm['ttft_ms']['mean']} ms  (Δ {delta['ttft_mean_pct']}%)")
    print(f"  TTFT p99:    baseline {baseline['ttft_ms']['p99']} ms → "
          f"tdm {tdm['ttft_ms']['p99']} ms  (Δ {delta['ttft_p99_pct']}%)")
    print(f"  TPOT mean:   baseline {baseline['tpot_ms']['mean']} ms → "
          f"tdm {tdm['tpot_ms']['mean']} ms  (Δ {delta['tpot_mean_pct']}%)")
    print(f"  E2E mean:    baseline {baseline['e2e_ms']['mean']} ms → "
          f"tdm {tdm['e2e_ms']['mean']} ms  (Δ {delta['e2e_mean_pct']}%)")
    print(f"  Output tput: baseline {baseline['output_throughput_tps']} tok/s → "
          f"tdm {tdm['output_throughput_tps']} tok/s  (Δ {delta['output_tput_pct']}%)")
    return combined


def run_both() -> int:
    py = sys.executable
    script = os.path.abspath(__file__)
    for mode in ("baseline", "tdm"):
        print(f"\n{'='*60}\n=== Subprocess: {mode}\n{'='*60}", flush=True)
        rc = subprocess.call([py, script, "--mode", mode])
        if rc != 0:
            print(f"[orchestrator] {mode} subprocess failed with rc={rc}",
                  file=sys.stderr)
            return rc
    merge_results()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "tdm", "both"],
                        default="both")
    args = parser.parse_args()

    if args.mode == "both":
        return run_both()
    run_single_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
