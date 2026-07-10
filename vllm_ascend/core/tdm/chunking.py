"""M3.1 prefill chunking — pure-logic planner.

Decides how many new prefill tokens each request gets in one P iter, given an
iter-wide chunk budget. Two-pass design aligned with vanilla vLLM v1
(vllm/v1/core/sched/scheduler.py:208-220):

  1. **Resume pass** over self.running. Picks up partial-prefilled requests
     (num_computed_tokens < num_prompt_tokens) and gives them the next chunk.
     Decode-state requests (num_computed_tokens >= num_prompt_tokens) are
     filtered out so the resulting iter stays phase-pure prefill — Ascend's
     V0 dedicated kernel path (`_forward_prefill_no_cache` /
     `_forward_decode_only`) cannot mix P and D tokens in one batch.

  2. **Fresh pass** over self.waiting. New prefill admissions, truncated to
     the remaining chunk budget instead of being skipped (the original
     AscendScheduler behaviour at scheduler.py:204).

The planner is deliberately KV-alloc / LoRA / encoder / watermark agnostic —
those checks remain in TDMScheduler.schedule()'s prefill section. This module
only produces token-count decisions; the caller wires them into Ascend's
existing per-request bookkeeping.

Cross-iter resume contract: a request truncated by this planner stays in
self.running with num_computed_tokens advanced. Next iter, the resume pass
picks it up via the running queue. This matches v1 conventions; only the
phase-pure filter on running differs from upstream.
"""
from dataclasses import dataclass
from typing import Iterable, Literal, NamedTuple


class ReqView(NamedTuple):
    """Snapshot of the fields the planner reads from a Request.

    Detached from vLLM's Request class so tests can drive the planner without
    importing the engine. TDMScheduler builds these from real Request objects
    at call time.
    """
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int


@dataclass(frozen=True)
class ChunkDecision:
    request_id: str
    num_new_tokens: int
    source: Literal["resume", "fresh"]


def plan_chunks(
    *,
    chunk_budget: int,
    token_budget: int,
    running: Iterable[ReqView],
    waiting: Iterable[ReqView],
) -> tuple[list[ChunkDecision], int, int]:
    """Plan one prefill iter.

    Args:
        chunk_budget: iter-wide cap on total prefill tokens scheduled this
            iter. Pass a very large int (e.g. 10**9) to effectively disable
            chunking — the planner then only truncates by token_budget,
            matching pre-M3.1 behaviour for fresh reqs.
        token_budget: Ascend's existing iter-wide token budget (typically
            scheduler_config.max_num_batched_tokens minus already-scheduled).
        running: self.running snapshot, in scheduling order.
        waiting: self.waiting snapshot, in scheduling order.

    Returns:
        (decisions, residual_chunk_budget, residual_token_budget). Decisions
        are emitted in the order: resume(running) first, then fresh(waiting).
    """
    decisions: list[ChunkDecision] = []
    cb, tb = chunk_budget, token_budget

    # Pass 1: resume partial-prefilled running reqs.
    for req in running:
        remaining = req.num_prompt_tokens - req.num_computed_tokens
        if remaining <= 0:
            # Decode-state — leave it for the decode loop, preserve phase-pure.
            continue
        if cb <= 0 or tb <= 0:
            return decisions, cb, tb
        n = min(remaining, cb, tb)
        decisions.append(ChunkDecision(req.request_id, n, "resume"))
        cb -= n
        tb -= n

    # Pass 2: fresh waiting reqs.
    for req in waiting:
        remaining = req.num_prompt_tokens - req.num_computed_tokens
        if remaining <= 0:
            # Shouldn't happen for waiting reqs in normal flow, but guard
            # against P/D resume edges that pre-populate num_computed_tokens.
            continue
        if cb <= 0 or tb <= 0:
            return decisions, cb, tb
        n = min(remaining, cb, tb)
        decisions.append(ChunkDecision(req.request_id, n, "fresh"))
        cb -= n
        tb -= n

    return decisions, cb, tb
