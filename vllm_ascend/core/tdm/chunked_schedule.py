"""M3.1 chunked-prefill fork of AscendScheduler.schedule().

This is a forked copy of ``vllm_ascend/core/scheduler.py::AscendScheduler.schedule``
(v0.11.0rc1, lines 64-505) with five modifications. Diffs #1–#3 modify the
prefill section. Diff #4 is shared (num_computed advance). Diff #5 protects
the decode loop from partial-prefill requests. Everything else — phase-decode
bookkeeping, KV connector, output assembly — is verbatim copy.

  Diff #1 (new): a running pre-pass that picks up partial-prefilled requests
    via :func:`vllm_ascend.core.tdm.chunking.plan_chunks`. Mirrors vanilla
    vLLM v1's running-loop convention (vllm/v1/core/sched/scheduler.py:208).
    Phase-pure filter (decode-state reqs skipped) applied inside the planner;
    the pass itself is gated on ``self.phase == "prefill"`` so decode iters
    never advance prefill work.

  Diff #2: waiting loop's iter-wide chunk_budget. ``token_budget`` is the
    Ascend-original cap (max_num_batched_tokens), ``chunk_budget`` is the
    M3.1-added cap (TDMConfig.prefill_chunk_tokens). Each scheduled req
    decrements both; the loop terminates when either depletes.

  Diff #3: parent's line 204 ``skip_cur_request()`` → truncate via
    ``num_new_tokens = min(num_new_tokens, chunk_budget, token_budget)``.
    Skip semantics for non-budget reasons (LoRA, watermark, long-prefill
    threshold, prompt-too-long) preserved verbatim.

  Diff #4: ``num_computed_tokens`` advance at the very end of the method
    is shared between the resume pass and the waiting pass.

  Diff #5: partial-prefill filter inside the decode loop. A request whose
    prefill is mid-chunking (``num_computed < num_prompt``) lives in
    ``self.running`` across iters; parent's decode loop asserts
    ``num_tokens - num_computed == 1`` for every running req and crashes on
    partials. We skip them; they wait for the next prefill iter's resume
    pass. This is the bug Phase-B smoke caught in 2026-05-08.

The fork is exposed as a free function (``schedule_chunked(scheduler)``)
rather than a method to keep the routing on TDMScheduler explicit and the
diff against parent.schedule() reviewable side-by-side.

Pinned to vllm-ascend v0.11.0rc1; resync if upstream AscendScheduler.schedule
changes shape (see notes.md §0 for the version-lock rationale).
"""
import time
from collections import deque

from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import logger
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import RequestStatus

from .chunking import ReqView, plan_chunks
from .types import ChunkPlanRecord


def schedule_chunked(self) -> SchedulerOutput:
    """Forked AscendScheduler.schedule() with M3.1 prefill chunking.

    ``self`` is expected to be a ``TDMScheduler`` (i.e. an ``AscendScheduler``
    subclass) with a valid ``self.tdm_cfg.prefill_chunk_tokens``.
    """
    chunk_budget = self.tdm_cfg.prefill_chunk_tokens
    assert chunk_budget is not None and chunk_budget >= 1, chunk_budget

    # P1 telemetry: capture initial budgets + per-source counters so we can
    # emit a ChunkPlanRecord at end-of-prefill-section. Counters tally
    # resume/fresh contributions, plus how many requests hit the
    # chunk_budget/token_budget ceiling vs. naturally fitting.
    initial_chunk_budget = chunk_budget
    ck_resume_count = 0
    ck_resume_tokens = 0
    ck_resume_truncated = 0
    ck_fresh_count = 0
    ck_fresh_tokens = 0
    ck_fresh_truncated = 0

    # ---- Verbatim from parent: local accumulators ---------------------
    scheduled_new_reqs = []
    scheduled_resumed_reqs = []
    scheduled_running_reqs = []
    preempted_reqs = []

    req_to_new_blocks = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    initial_token_budget = token_budget  # P1 telemetry snapshot

    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_budget = self.max_num_encoder_input_tokens

    scheduled_spec_decode_tokens: dict[str, list[int]] = {}

    scheduled_timestamp = time.monotonic()
    scheduled_loras: set[int] = set()

    skipped_waiting_requests: deque = deque()

    # ---- Verbatim from parent (lines 93-112): phase=prefill bookkeeping
    if self.phase == "prefill":
        remaining_running_reqs = []
        for request in self.running:
            if request.num_tokens > request.num_prompt_tokens:
                self.finished_prefill_reqs.append(request)
            else:
                remaining_running_reqs.append(request)
        self.running = remaining_running_reqs
        if not self.waiting and not self.running:
            self.phase = "decode"

    if self.vllm_config.scheduler_config.long_prefill_token_threshold == 0:
        long_prefill_budget = float('inf')
        long_prefill_token_threshold = float('inf')
    else:
        long_prefill_budget = (
            self.vllm_config.scheduler_config.max_long_partial_prefills)
        long_prefill_token_threshold = (
            self.vllm_config.scheduler_config.long_prefill_token_threshold)

    # ===================================================================
    # Diff #1: running pre-pass for partial-prefill resume.
    # Gated on phase=="prefill" — phase-pure means decode iters do NOT
    # advance prefill work, even for partial-prefilled reqs. Those reqs
    # simply wait for the next prefill iter (and are filtered out of the
    # decode loop below — see Diff #5).
    # ===================================================================
    running_views = [
        ReqView(request_id=r.request_id,
                num_prompt_tokens=r.num_prompt_tokens,
                num_computed_tokens=r.num_computed_tokens)
        for r in self.running
    ] if self.phase == "prefill" else []
    resume_decisions, _, _ = plan_chunks(
        chunk_budget=chunk_budget,
        token_budget=token_budget,
        running=running_views,
        waiting=[],  # waiting handled in pass 2 with the original loop
    )

    if resume_decisions:
        # Map req_id → Request for fast lookup (self.running is small).
        running_by_id = {r.request_id: r for r in self.running}
        for d in resume_decisions:
            request = running_by_id.get(d.request_id)
            if request is None:
                continue  # finished/aborted between view snapshot and now
            num_new_tokens = d.num_new_tokens

            # KV alloc for the resume chunk. num_computed_tokens is already
            # the post-last-iter value, so allocate_slots reads it correctly.
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # KV pressure — abort the resume pass; do NOT preempt running
                # reqs from under our own feet (parent does this in its decode
                # loop with self.running.pop(); we keep resume conservative).
                break

            # Bookkeeping mirrors parent's decode-loop tail (lines 396-401),
            # NOT its waiting-loop tail. Resume reqs are already RUNNING and
            # go through scheduled_running_reqs → _make_cached_request_data
            # which expects the *delta* blocks (just-allocated), same as the
            # decode loop does at parent line 398. The waiting loop's
            # get_blocks(req_id) returns the full block list and is only
            # correct for scheduled_new_reqs which uses NewRequestData.
            scheduled_running_reqs.append(request)
            self.scheduled_req_ids.add(request.request_id)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            chunk_budget -= num_new_tokens
            if num_new_tokens > long_prefill_token_threshold:
                long_prefill_budget -= 1
            # P1 telemetry: count this resume schedule. Truncated iff the
            # planner gave fewer tokens than the request still needs — i.e.
            # chunk_budget or token_budget was the binding constraint.
            ck_resume_count += 1
            ck_resume_tokens += num_new_tokens
            remaining_orig = (request.num_prompt_tokens
                              - request.num_computed_tokens)
            if num_new_tokens < remaining_orig:
                ck_resume_truncated += 1
            # Note: request.num_computed_tokens is advanced at the end of
            # this method (line marked "Diff #4" below), shared with parent.

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)

            if chunk_budget <= 0 or token_budget <= 0:
                break

    # ===================================================================
    # Waiting loop. Verbatim from parent lines 115-301 EXCEPT:
    #   Diff #2 / Diff #3: chunk_budget tracked + line 204 skip → truncate.
    #   Diff #6 (2026-05-24): gate on phase=="prefill". Without this,
    #     decode iters admit new reqs from the waiting queue (giving them a
    #     prefill chunk) and then skip the decode loop entirely via L396's
    #     `len(scheduled_req_ids) == 0` guard — starving already-running
    #     decode reqs while only serving 1 new prefill per "decode" iter.
    #     Strict phase-pure (matching the design intent stated at L13 and
    #     L120-125) requires gating fresh admission to prefill iters too.
    # ===================================================================
    while (self.phase == "prefill" and self.waiting
           and token_budget > 0 and chunk_budget > 0):
        if len(self.running) == (self.decode_max_num_running_reqs
                                 if self.phase == "decode" else
                                 self.max_num_running_reqs):
            break

        request = self.waiting[0]

        def skip_cur_request():
            self.waiting.popleft()
            skipped_waiting_requests.appendleft(request)

        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            is_ready = self._update_waiting_for_remote_kv(request)
            if is_ready:
                request.status = RequestStatus.WAITING
            else:
                skip_cur_request()
                continue

        if (self.lora_config and request.lora_request and
            (len(scheduled_loras) == self.lora_config.max_loras
             and request.lora_request.lora_int_id not in scheduled_loras)):
            skip_cur_request()
            continue

        num_external_computed_tokens = 0
        load_kv_async = False

        if request.num_computed_tokens == 0:
            new_computed_blocks, num_new_local_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(request)
            if self.connector is not None:
                num_external_computed_tokens, load_kv_async = (
                    self.connector.get_num_new_matched_tokens(
                        request, num_new_local_computed_tokens))
            num_computed_tokens = (num_new_local_computed_tokens +
                                   num_external_computed_tokens)
        else:
            new_computed_blocks = (
                self.kv_cache_manager.create_empty_block_list())
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        encoder_inputs_to_schedule = None
        new_encoder_budget = encoder_budget

        if load_kv_async:
            assert num_external_computed_tokens > 0
            num_new_tokens = 0
            blocks = None
        else:
            prompt_limit = self._get_prompt_limit(request)
            num_new_tokens = request.num_tokens - num_computed_tokens
            max_tokens_in_kvcache = (self.kv_cache_config.num_blocks *
                                     self.block_size)
            prompt_limit = min(prompt_limit, max_tokens_in_kvcache)

            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens, prompt_limit)
                request.status = RequestStatus.FINISHED_IGNORED
                self.finished_req_ids.add(request.request_id)
                self.waiting.popleft()
                continue

            # ===== Diff #3: skip → truncate =====
            # Parent did:
            #   if num_new_tokens > token_budget:
            #       skip_cur_request(); continue
            # M3.1: truncate by min(chunk_budget, token_budget). Both budgets
            # are positive here (loop guard) so the result is >= 1.
            #
            # P1 telemetry: snapshot pre-truncation count to detect truncation.
            num_new_tokens_pre_min = num_new_tokens
            num_new_tokens = min(num_new_tokens, chunk_budget, token_budget)
            assert num_new_tokens > 0
            blocks = new_computed_blocks.blocks[0]

            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0 or len(
                        encoder_inputs_to_schedule) == 0:
                    break

        watermark = getattr(self.scheduler_config, "watermark", 0.01)
        if not self._check_watermark_for_prefill(request, num_new_tokens,
                                                 blocks, watermark):
            skip_cur_request()
            continue

        if (num_new_tokens > long_prefill_token_threshold
                and long_prefill_budget <= 0):
            skip_cur_request()
            continue

        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens + num_external_computed_tokens,
            num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=self.num_lookahead_tokens,
            delay_cache_blocks=load_kv_async)
        if new_blocks is None:
            break

        if self.connector is not None:
            self.connector.update_state_after_alloc(
                request,
                new_computed_blocks + new_blocks,
                num_external_computed_tokens,
            )

        self.waiting.popleft()
        if load_kv_async:
            skipped_waiting_requests.appendleft(request)
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            continue

        self.running.append(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.SCHEDULED,
                                 scheduled_timestamp)
        self.scheduled_req_ids.add(request.request_id)
        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(request)
        elif request.status == RequestStatus.PREEMPTED:
            scheduled_resumed_reqs.append(request)
        else:
            raise RuntimeError(f"Invalid request status: {request.status}")

        if self.lora_config and request.lora_request:
            scheduled_loras.add(request.lora_request.lora_int_id)

        req_to_new_blocks[request.request_id] = (
            self.kv_cache_manager.get_blocks(request.request_id))
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        chunk_budget -= num_new_tokens  # Diff #2: chunk_budget tracked
        if num_new_tokens > long_prefill_token_threshold:
            long_prefill_budget -= 1
        # P1 telemetry: count fresh schedule and truncation.
        ck_fresh_count += 1
        ck_fresh_tokens += num_new_tokens
        if num_new_tokens < num_new_tokens_pre_min:
            ck_fresh_truncated += 1
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        if request.num_cached_tokens < 0:
            request.num_cached_tokens = num_computed_tokens

        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_budget = new_encoder_budget

    if skipped_waiting_requests:
        self.waiting.extendleft(skipped_waiting_requests)

    # P1 telemetry: emit one ChunkPlanRecord per P-iter (skip decode-only
    # iters where neither resume nor fresh prefill work happened — the chunk
    # budget is unused in that case and a record would just be noise). We
    # use "any prefill activity" as the gate so passive+chunk ablation runs
    # also produce records.
    if ck_resume_count > 0 or ck_fresh_count > 0:
        emit_chunk = getattr(getattr(self, "_tdm_telemetry", None),
                             "record_chunk_plan", None)
        if emit_chunk is not None:
            emit_chunk(
                ChunkPlanRecord(
                    iter_id=getattr(self, "_tdm_iter_id", -1),
                    ts_ms=time.monotonic() * 1000.0,
                    chunk_budget_in=initial_chunk_budget,
                    chunk_budget_used=initial_chunk_budget - chunk_budget,
                    token_budget_in=initial_token_budget,
                    token_budget_used=initial_token_budget - token_budget,
                    num_resume_reqs=ck_resume_count,
                    num_fresh_reqs=ck_fresh_count,
                    resume_tokens=ck_resume_tokens,
                    fresh_tokens=ck_fresh_tokens,
                    truncated_count=(ck_resume_truncated +
                                     ck_fresh_truncated),
                ))

    # ---- Verbatim from parent (lines 306-425): phase=decode + decode loop
    if self.phase == "decode":
        while (len(self.running) < self.decode_max_num_running_reqs
               and self.finished_prefill_reqs):
            request = self.finished_prefill_reqs.popleft()
            self.running.append(request)

    if len(self.scheduled_req_ids) == 0:
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in self.scheduled_req_ids:
                req_index += 1
                continue

            # ===== Diff #5: partial-prefill filter =====
            # A request whose prefill is mid-flight (num_computed <
            # num_prompt) lives in self.running across chunks. Parent's
            # decode loop asserts (num_tokens - num_computed) == 1, which
            # blows up for partials. Skip them — they will be picked up by
            # the resume pre-pass on the next prefill iter. This preserves
            # phase-pure: decode iters never advance prefill work.
            if request.num_computed_tokens < request.num_prompt_tokens:
                req_index += 1
                continue

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            assert (request.num_tokens - request.num_computed_tokens) == 1
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if self.lora_config and request.lora_request and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id
                    not in scheduled_loras):
                num_new_tokens = 0

            if num_new_tokens == 0:
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED,
                            scheduled_timestamp)
                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            scheduled_running_reqs.append(request)
            self.scheduled_req_ids.add(request.request_id)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)

    # ---- Verbatim from parent (lines 427-505): assertions + output ----
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
    assert token_budget >= 0
    assert (len(self.running) <= self.decode_max_num_running_reqs
            if self.phase == "decode" else self.max_num_running_reqs)
    assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
            len(scheduled_running_reqs) <= len(self.running))

    num_common_prefix_blocks = [0] * len(
        self.kv_cache_config.kv_cache_groups)
    if self.running:
        any_request = self.running[0]
        num_common_prefix_blocks = (
            self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request, len(self.running)))

    new_reqs_data = [
        NewRequestData.from_request(
            req, req_to_new_blocks[req.request_id].get_block_ids())
        for req in scheduled_new_reqs
    ]

    cached_reqs_data = self._make_cached_request_data(
        scheduled_running_reqs, scheduled_resumed_reqs,
        num_scheduled_tokens, scheduled_spec_decode_tokens,
        req_to_new_blocks)
    scheduled_cached_reqs = cached_reqs_data

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs_data,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        finished_req_ids=self.finished_req_ids,
        free_encoder_mm_hashes=self.encoder_cache_manager.
        get_freed_mm_hashes(),
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    if self.connector is not None:
        meta = self.connector.build_connector_meta(scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    events = self.kv_cache_manager.take_events()
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    # Diff #4: shared with parent — advance num_computed_tokens for
    # both the resume pass AND the waiting pass in one place.
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        self.requests[req_id].num_computed_tokens += num_scheduled_token

    self.finished_req_ids = set()
    return scheduler_output
