"""M3.1 prefill chunking — pure-logic planner tests.

Run via:
    /usr/local/python3.11.13/bin/python3 -m vllm_ascend.core.tdm.tests._runner

These tests pin down the ChunkPlanner contract before TDMScheduler is wired
up. Each test exercises one invariant from the M3.1 design (see
chunking.py module docstring).
"""
from vllm_ascend.core.tdm.chunking import (ChunkDecision, ReqView,
                                            plan_chunks)


def _r(rid: str, prompt: int, computed: int = 0) -> ReqView:
    return ReqView(request_id=rid, num_prompt_tokens=prompt,
                   num_computed_tokens=computed)


# ---------------------------------------------------------------------------
# Test 1: chunk-disabled equivalent (chunk_budget = inf-ish) does not truncate
# beyond token_budget. This is the back-compat smoke at the planner level —
# with budget >> any prompt, every fresh req gets its full prompt.
# ---------------------------------------------------------------------------
def test_chunk_off_inf_budget_does_not_truncate():
    waiting = [_r("a", 200), _r("b", 300), _r("c", 500)]
    decisions, cb, tb = plan_chunks(
        chunk_budget=10**9, token_budget=10**9,
        running=[], waiting=waiting,
    )
    assert decisions == [
        ChunkDecision("a", 200, "fresh"),
        ChunkDecision("b", 300, "fresh"),
        ChunkDecision("c", 500, "fresh"),
    ], decisions
    # Both budgets only consumed by the actual scheduled tokens.
    assert cb == 10**9 - 1000
    assert tb == 10**9 - 1000


# ---------------------------------------------------------------------------
# Test 2: chunk_budget=512 truncates a single long fresh req. This is the
# core M3.1 motion that scheduler.py:204's skip_cur_request will be replaced
# with.
# ---------------------------------------------------------------------------
def test_chunk_on_single_fresh_truncated_to_chunk():
    decisions, cb, tb = plan_chunks(
        chunk_budget=512, token_budget=8192,
        running=[], waiting=[_r("a", 2000)],
    )
    assert decisions == [ChunkDecision("a", 512, "fresh")], decisions
    assert cb == 0
    assert tb == 8192 - 512


# ---------------------------------------------------------------------------
# Test 3: partial-prefilled req in running is resumed first via the pre-pass,
# with the same truncate semantics. This is the v1-aligned cross-iter resume
# contract — without this, partial-prefilled reqs would stall forever because
# Ascend's prefill loop only iterates self.waiting.
# ---------------------------------------------------------------------------
def test_chunk_on_partial_resume_via_running_pre_pass():
    # Req "a": prompt=2000, already chunked once last iter (512 done).
    running = [_r("a", 2000, computed=512)]
    decisions, cb, tb = plan_chunks(
        chunk_budget=512, token_budget=8192,
        running=running, waiting=[],
    )
    assert decisions == [ChunkDecision("a", 512, "resume")], decisions
    assert cb == 0
    assert tb == 8192 - 512


# ---------------------------------------------------------------------------
# Test 4: phase-pure invariant — a decode-state req in self.running (num_
# computed >= num_prompt, i.e. prefill done) MUST NOT be picked up by the
# pre-pass. If it were, the resulting iter would mix P and D tokens and
# break Ascend's V0 dedicated kernel dispatch.
# ---------------------------------------------------------------------------
def test_phase_pure_decode_req_not_picked_up():
    running = [
        _r("decode_req", 200, computed=200),  # prefill done, in decode
        _r("partial_req", 2000, computed=512),  # still in prefill
    ]
    decisions, _, _ = plan_chunks(
        chunk_budget=512, token_budget=8192,
        running=running, waiting=[_r("fresh_req", 100)],
    )
    # decode_req must be absent. partial_req picked up first (resume), then
    # fresh_req gets the remainder of the chunk budget (0, because partial
    # took it all) so it's also absent.
    rids = [d.request_id for d in decisions]
    assert "decode_req" not in rids, decisions
    assert decisions == [ChunkDecision("partial_req", 512, "resume")], decisions


# ---------------------------------------------------------------------------
# Test 5: iter-wide chunk_budget caps total prefill tokens across multiple
# reqs (the §4 design point: "切的对象 = iter 内总 prefill token cap").
# ---------------------------------------------------------------------------
def test_iter_wide_chunk_budget_cap_across_multiple_reqs():
    waiting = [_r("a", 600), _r("b", 600), _r("c", 600)]
    decisions, cb, tb = plan_chunks(
        chunk_budget=1024, token_budget=8192,
        running=[], waiting=waiting,
    )
    assert decisions == [
        ChunkDecision("a", 600, "fresh"),
        ChunkDecision("b", 424, "fresh"),  # truncated by remaining cb
        # "c" not scheduled: cb exhausted
    ], decisions
    assert cb == 0
    assert tb == 8192 - 1024


# ---------------------------------------------------------------------------
# Bonus test: resume-before-fresh ordering. The two-pass design must always
# emit resume decisions before fresh ones — partial-prefilled reqs already
# hold KV blocks, so giving them priority avoids unnecessary preemptions
# under tight chunk_budget.
# ---------------------------------------------------------------------------
def test_resume_decisions_emitted_before_fresh():
    running = [_r("partial", 2000, computed=1024)]
    waiting = [_r("new", 100)]
    decisions, _, _ = plan_chunks(
        chunk_budget=4096, token_budget=8192,
        running=running, waiting=waiting,
    )
    assert [d.source for d in decisions] == ["resume", "fresh"], decisions
    # And both fit comfortably under the budget.
    assert decisions[0].num_new_tokens == 2000 - 1024  # remaining prompt
    assert decisions[1].num_new_tokens == 100
