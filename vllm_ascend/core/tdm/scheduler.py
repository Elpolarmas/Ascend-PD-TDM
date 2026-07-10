from typing import TYPE_CHECKING

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.core.scheduler import AscendScheduler

from .boundary import BoundaryGuard
from .chunked_schedule import schedule_chunked
from .config import TDMConfig
from .constraints import HardConstraints
from .controller import make_controller
from .engine import PhaseEngine
from .monitor import QueueMonitor
from .selector import TokenBucketSelector
from .telemetry import Telemetry
from .timing import TimeAccountant
from .tracker import RequestTracker
from .types import IterRecord, Phase, PhaseDecision

if TYPE_CHECKING:
    pass


class TDMScheduler(AscendScheduler):
    """Time-Division Multiplexing scheduler.

    Wraps AscendScheduler. Per iter: peek a phase via the controller +
    selector + constraints + boundary pipeline, set ``self.phase``, hand
    off to the parent's prefill-first / decode loop, then read back the
    actually executed phase to keep our bucket / counters honest (parent
    may auto-flip prefill→decode when both queues empty — Option W).
    """

    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self.tdm_cfg = TDMConfig.from_vllm_config(vllm_config)

        self._tdm_active = self.tdm_cfg.enable_tdm
        self._tdm_passive = (not self._tdm_active
                             and self.tdm_cfg.passive_tracker)

        if not (self._tdm_active or self._tdm_passive):
            return

        cfg = self.tdm_cfg
        # Lightweight components shared by active + passive: tracker,
        # telemetry, time, admission state. Passive mode uses these to
        # collect baseline TTFT/TPOT without altering scheduling.
        self._tdm_telemetry = Telemetry(
            window_size=cfg.window_size,
            cold_dir=cfg.telemetry_dir if cfg.telemetry_enabled else None,
            run_id=cfg.run_id,
            cold_enabled=cfg.telemetry_enabled,
        )
        self._tdm_time = TimeAccountant(window=cfg.window_size)
        self._tdm_tracker = RequestTracker()
        self._tdm_iter_id: int = 0
        self._tdm_active_req_ids: set[str] = set()
        # Monitor is shared active+passive: in passive (c3_cp) we still want
        # snapshot fields (waiting_depth, kv_free_ratio) on each IterRecord
        # so post-hoc analysis can correlate c3 mixed-iter duration with
        # batch/queue state. Cost is O(running_depth) per iter.
        self._tdm_monitor = QueueMonitor(tracker=self._tdm_tracker)

        if not self._tdm_active:
            return

        # Force a real initial phase. AscendScheduler defaults to "" when
        # enable_pd_transfer is False, but L93 only enters the prefill path
        # when phase == "prefill", so a blank phase would make us decode-only.
        self.phase = self.tdm_cfg.initial_phase
        self._tdm_engine = PhaseEngine(initial_phase=cfg.initial_phase)
        self._tdm_selector = TokenBucketSelector(cap=cfg.bucket_cap)
        self._tdm_controller = make_controller(
            cfg, self._tdm_telemetry, tracker=self._tdm_tracker)
        self._tdm_constraints = HardConstraints(
            min_slice_iters=cfg.min_slice_iters,
            max_slice_iters=cfg.max_slice_iters,
        )
        self._tdm_boundary = BoundaryGuard(
            kv_free_watermark=cfg.kv_free_watermark)

        # c2_tdm_m31_fia ablation: 强制 attention 走 FIA(monkey-patch
        # AscendAttentionBackendImpl.forward)。详见 attn_patch.py。
        if cfg.force_fia_attention:
            from .attn_patch import enable_force_fia
            enable_force_fia()

    # ---------------- Entry 1 ----------------

    def schedule(self) -> SchedulerOutput:
        if not (self._tdm_active or self._tdm_passive):
            return super().schedule()

        now_ms = self._tdm_time.now_ms()

        if self._tdm_active:
            snap = self._tdm_monitor.snapshot(self, now_ms)
            target_ratio = self._tdm_controller.get_target_ratio(
                snap, self._tdm_iter_id)
            cfg = self.tdm_cfg
            urgency_ms = (cfg.urgency_ttft_threshold * cfg.slo_ttft_ms
                          if cfg.urgency_ttft_enabled else 0.0)
            starvation_ms = (cfg.starvation_tpot_threshold * cfg.slo_tpot_ms
                             if cfg.starvation_tpot_enabled else 0.0)
            planned: Phase = self._tdm_selector.peek(
                target_ratio, snap,
                urgency_threshold_ms=urgency_ms,
                charge_bucket_on_urgency=cfg.urgency_charge_bucket,
                starvation_threshold_ms=starvation_ms,
                starvation_decouple_bucket=cfg.starvation_decouple_bucket,
            )
            if self._tdm_selector.last_override_urgent:
                src_sel = "urgency_ttft"
            elif self._tdm_selector.last_override_starvation:
                src_sel = "urgency_tpot"
            else:
                src_sel = "controller"
            after_c, src_c = self._tdm_constraints.enforce(
                planned, self._tdm_engine.phase, self._tdm_engine.phase_iters)
            # Preserve urgency_ttft tag through constraints when constraints
            # didn't actually flip the decision (constraint enforcement
            # returns "controller" as a no-op sentinel).
            src_c = src_sel if src_c == "controller" else src_c
            candidate, src_b = self._tdm_boundary.override(after_c, snap)
            source = src_b if src_b != "controller" else src_c

            self._tdm_engine.apply(self, candidate)
            self._tdm_time.mark_schedule(now_ms)

        # Chunked-prefill route. Always route through chunked_schedule when
        # chunking is configured, regardless of:
        #   - phase (decode iters need the partial-prefill filter to avoid
        #     parent's `num_tokens - num_computed == 1` assertion)
        #   - TDM active/passive (passive+chunk = ablation cell that tests
        #     whether the controller adds value on top of the chunking
        #     foundation; no controller pipeline runs in passive but the
        #     chunked schedule still applies the truncate semantics)
        if self.tdm_cfg.prefill_chunk_tokens is not None:
            out = schedule_chunked(self)
        else:
            out = super().schedule()

        # Admit new requests at schedule time, not at update_from_output time.
        # Otherwise admission_ts and first_token_ts both come from the same
        # update_from_output now_ms tick and TTFT collapses to 0.
        for nr in out.scheduled_new_reqs:
            req_id = nr.req_id
            if req_id in self._tdm_active_req_ids:
                continue
            req = self.requests.get(req_id)
            prompt_tokens = int(getattr(req, "num_prompt_tokens", 0) or 0)
            self._tdm_tracker.on_admit(req_id, prompt_tokens, now_ms)
            self._tdm_active_req_ids.add(req_id)

        if self._tdm_active:
            actual: Phase = self.phase if self.phase in ("prefill",
                                                         "decode") else candidate
            flipped = self._tdm_engine.reconcile(actual, candidate)
            self._tdm_selector.commit(actual)

            decision = PhaseDecision(
                phase=candidate,
                source="parent_auto_flip" if flipped else source,
                target_ratio=target_ratio,
            )
            rec = IterRecord(
                iter_id=self._tdm_iter_id,
                ts_ms=now_ms,
                decision=decision,
                actual_phase=actual,
                phase_iters=self._tdm_engine.phase_iters,
                snapshot=snap,
                batch_num_reqs=len(out.num_scheduled_tokens),
                batch_num_tokens=int(out.total_num_scheduled_tokens),
            )
            self._tdm_telemetry.record_iter(rec)
            self._tdm_iter_id += 1
        elif self._tdm_passive:
            # c3_cp telemetry: synthesize IterRecord with no controller
            # decision (phase concept doesn't apply — c3 mixed iter does
            # prefill+decode in same batch). Source="fallback" + actual_phase
            # ="decode" are placeholders; batch_num_reqs / batch_num_tokens
            # + iter_duration_ms are the real signal for mechanism analysis.
            snap = self._tdm_monitor.snapshot(self, now_ms)
            rec = IterRecord(
                iter_id=self._tdm_iter_id,
                ts_ms=now_ms,
                decision=PhaseDecision(
                    phase="decode", source="fallback", target_ratio=0.0),
                actual_phase="decode",
                phase_iters=0,
                snapshot=snap,
                batch_num_reqs=len(out.num_scheduled_tokens),
                batch_num_tokens=int(out.total_num_scheduled_tokens),
            )
            self._tdm_telemetry.record_iter(rec)
            self._tdm_iter_id += 1
            self._tdm_time.mark_schedule(now_ms)
        return out

    # ---------------- Entry 2 ----------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        if not (self._tdm_active or self._tdm_passive):
            return super().update_from_output(scheduler_output,
                                              model_runner_output)

        now_ms = self._tdm_time.now_ms()

        if self._tdm_active:
            dur = self._tdm_time.take_iter_duration(now_ms)
            if dur is not None:
                self._tdm_time.record_iter(self._tdm_engine.phase, dur)
                self._tdm_telemetry.backfill_iter_duration(
                    self._tdm_iter_id - 1, dur)
        elif self._tdm_passive:
            dur = self._tdm_time.take_iter_duration(now_ms)
            if dur is not None:
                self._tdm_telemetry.backfill_iter_duration(
                    self._tdm_iter_id - 1, dur)

        # Token production: per-request, by index into req_ids.
        # P1.6b F1: tracker.on_token_produced returns the record on the
        # first-token transition so we can push ttft to telemetry immediately
        # — controller no longer has to wait for on_finish to see ttft.
        if model_runner_output.sampled_token_ids:
            for req_id, idx in model_runner_output.req_id_to_index.items():
                toks = model_runner_output.sampled_token_ids[idx]
                if toks:
                    rec_ft = self._tdm_tracker.on_token_produced(
                        req_id, len(toks), now_ms)
                    if rec_ft is not None:
                        self._tdm_telemetry.record_request(rec_ft)

        result = super().update_from_output(scheduler_output,
                                            model_runner_output)

        # Anything we admitted but parent has dropped from .requests is done.
        finished = [r for r in self._tdm_active_req_ids
                    if r not in self.requests]
        for req_id in finished:
            self._tdm_active_req_ids.discard(req_id)
            rec = self._tdm_tracker.on_finish(req_id, now_ms)
            if rec is not None:
                self._tdm_telemetry.record_request(rec)

        return result

    def shutdown(self) -> None:
        if getattr(self, "_tdm_active", False) or getattr(
                self, "_tdm_passive", False):
            try:
                self._tdm_telemetry.close()
            except Exception:
                pass
        if hasattr(super(), "shutdown"):
            super().shutdown()
