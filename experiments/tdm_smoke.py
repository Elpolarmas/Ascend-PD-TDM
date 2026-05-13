"""TDM V1 端到端 smoke test.

通过 ascend_scheduler_config.scheduler_cls 注入 TDMScheduler，
拉起 Qwen3-8B (TP=2) 跑几条 prompt，落 telemetry 到 results/tdm_trace/。

判据：
  1. 进程不崩，输出 4 条生成
  2. results/tdm_trace/smoke_v1_iter.jsonl 有内容
  3. results/tdm_trace/smoke_v1_req.jsonl 有 4 行含 ttft_ms
  4. iter 日志中 actual_phase 同时出现 'prefill' 和 'decode'
"""
import os

# Worker 进程用 spawn，确保配置/import 时序与主进程一致
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

MODEL_PATH = "/vllm-workspace/models/models/Qwen3-8B"
TELEMETRY_DIR = "/vllm-workspace/Ascend-PD-TDM/results/tdm_trace"
RUN_ID = "smoke_v1"


def main() -> int:
    additional_config = {
        # 启用 AscendScheduler 路径，并把 scheduler_cls 替换为 TDMScheduler。
        # platform.py L288 仅在 enabled=True 时调用 initialize_from_config，
        # schedule_config.py L57-59 会用此对象的 scheduler_cls 覆盖默认值。
        "ascend_scheduler_config": {
            "enabled": True,
            "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
        },
        # TDMScheduler 自己读 additional_config["tdm"]
        "tdm": {
            "enable_tdm": True,
            "static_ratio": 0.3,
            "min_slice_iters": 2,
            "max_slice_iters": 8,
            "kv_free_watermark": 0.05,
            "telemetry_enabled": True,
            "telemetry_dir": TELEMETRY_DIR,
            "run_id": RUN_ID,
            "initial_phase": "prefill",
        },
    }

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enforce_eager=False,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        additional_config=additional_config,
    )

    prompts = [
        "Explain time-division multiplexing in one paragraph.",
        "Write a Python function that reverses a string.",
        "What is the capital of France?",
        "Summarize the theory of relativity briefly.",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(prompts, sp)

    print("=" * 60)
    print(f"Generated {len(outs)} outputs:")
    for i, o in enumerate(outs):
        text = o.outputs[0].text.replace("\n", " ")
        print(f"  [{i}] {text[:80]}")
    print("=" * 60)
    print(f"Telemetry written to: {TELEMETRY_DIR}/{RUN_ID}_(iter|req).jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
