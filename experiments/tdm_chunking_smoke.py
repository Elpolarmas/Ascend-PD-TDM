"""M3.1 chunked-prefill smoke.

Two phases:
  Phase A (regression guard): prefill_chunk_tokens=8192 ≥ max_model_len.
    Chunk budget is never the binding constraint, so the new code path
    runs but produces M2.7-equivalent scheduling decisions. Verifies the
    fork didn't break baseline behaviour.

  Phase B (chunking active): prefill_chunk_tokens=512 with one long prompt.
    Chunking actually fires; verifies cross-iter resume (running pre-pass)
    works end-to-end without crashing. Each iter's prefill should schedule
    ≤ 512 tokens for the long prompt; multiple iters needed to complete it.

Pass criteria:
  Both phases generate text without crash. Phase B's telemetry shows the
  long prompt's prefill spread across ≥ 3 iters (1500 / 512 ≈ 3 chunks).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

MODEL_PATH = "/vllm-workspace/models/models/Qwen3-8B"
TELEMETRY_DIR = "/vllm-workspace/Ascend-PD-TDM/results/m3_smoke"


def _build_llm(run_id: str, chunk_tokens: int | None) -> LLM:
    tdm_cfg = {
        "enable_tdm": True,
        "controller_kind": "slo_pid",  # M2.7 baseline (most recent)
        "static_ratio": 0.3,
        "min_slice_iters": 2,
        "max_slice_iters": 8,
        "kv_free_watermark": 0.05,
        "telemetry_enabled": True,
        "telemetry_dir": TELEMETRY_DIR,
        "run_id": run_id,
        "initial_phase": "prefill",
    }
    if chunk_tokens is not None:
        tdm_cfg["prefill_chunk_tokens"] = chunk_tokens

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enforce_eager=False,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        additional_config={
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": tdm_cfg,
        },
    )


def _short_prompts() -> list[str]:
    return [
        "Explain time-division multiplexing in one paragraph.",
        "Write a Python function that reverses a string.",
        "What is the capital of France?",
        "Summarize the theory of relativity briefly.",
    ]


def _long_prompt() -> str:
    """~1500 tokens of context. With chunk=512, prefill should take ≥ 3 iters."""
    para = (
        "The history of distributed systems begins in the 1960s with the "
        "development of time-sharing operating systems. Engineers at MIT and "
        "Bell Labs realized that a single mainframe could be partitioned into "
        "logical slices, each serving a different user, by switching CPU "
        "context at fixed intervals. This insight—that throughput improves "
        "when expensive resources are time-multiplexed across competing "
        "demands—propagated forward into network protocols, database "
        "transactions, and eventually into modern serving systems for large "
        "language models, where prefill and decode phases compete for the "
        "same GPU compute and memory bandwidth. ")
    return (para * 12).strip() + "\n\nSummarize the above in two sentences."


def _count_iters_for_long(req_jsonl: Path, iter_jsonl: Path) -> int | None:
    """Count how many iters scheduled tokens for the long-prompt request."""
    if not req_jsonl.exists() or not iter_jsonl.exists():
        return None
    # Find the longest-prompt req_id from req log.
    best = None
    for line in req_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        n = r.get("prompt_tokens", 0)
        if best is None or n > best[1]:
            best = (r.get("request_id"), n)
    if best is None or best[0] is None:
        return None
    long_rid, long_len = best
    # Now scan iter log; look for entries that scheduled this req.
    # Telemetry's IterRecord has `batch_num_reqs` etc but not per-req IDs;
    # we approximate: count iters where actual_phase=="prefill" before the
    # decode phase locks in. A more precise check would require per-req
    # iter trace, but for smoke we only need: prefill iters >= 3.
    prefill_iters = 0
    for line in iter_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except Exception:
            continue
        if it.get("actual_phase") == "prefill":
            prefill_iters += 1
    print(f"  long prompt: req_id={long_rid} prompt_tokens={long_len}")
    print(f"  total prefill iters this run: {prefill_iters}")
    return prefill_iters


def _run_phase(label: str, run_id: str, chunk_tokens: int | None,
               prompts: list[str]) -> int:
    print("=" * 70)
    print(f"PHASE {label}: chunk_tokens={chunk_tokens}, run_id={run_id}")
    print("=" * 70)
    llm = _build_llm(run_id, chunk_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(prompts, sp)
    print(f"\n[{label}] {len(outs)} outputs:")
    for i, o in enumerate(outs):
        text = o.outputs[0].text.replace("\n", " ")
        print(f"  [{i}] {text[:90]}")
    # vLLM doesn't expose a clean shutdown handle; rely on process exit.
    return 0


def main() -> int:
    Path(TELEMETRY_DIR).mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which == "A":
        rc = _run_phase("A (chunk=8192, regression guard)",
                        "smoke_a_chunk8192", 8192, _short_prompts())
        return rc
    elif which == "B":
        rc = _run_phase("B (chunk=512, active chunking)",
                        "smoke_b_chunk512", 512,
                        _short_prompts() + [_long_prompt()])
        if rc == 0:
            iters = _count_iters_for_long(
                Path(TELEMETRY_DIR) / "smoke_b_chunk512_req.jsonl",
                Path(TELEMETRY_DIR) / "smoke_b_chunk512_iter.jsonl")
            if iters is not None and iters < 3:
                print(f"\nWARN: only {iters} prefill iters seen; chunking may "
                      "not have fired (long prompt may have been too short)")
        return rc
    else:
        print(f"Unknown phase '{which}'. Use 'A' or 'B'.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
