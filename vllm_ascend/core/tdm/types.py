from dataclasses import dataclass, field
from typing import Literal, Optional

Phase = Literal["prefill", "decode"]
DecisionSource = Literal[
    "controller",
    "urgency_ttft",
    "urgency_tpot",
    "constraint_max_slice",
    "constraint_min_slice",
    "boundary_kv_pressure",
    "parent_auto_flip",
    "fallback",
]


@dataclass(frozen=True)
class QueueSnapshot:
    waiting_depth: int
    waiting_oldest_age_ms: float
    finished_prefill_depth: int
    running_depth: int
    kv_free_blocks: int
    kv_total_blocks: int
    # P1.7b TPOT starvation signal: max(now - last_token_ts) over running
    # requests that have already produced their first token. 0.0 means
    # either no running decodes or all running reqs are still in prefill.
    decode_oldest_silence_ms: float = 0.0

    @property
    def kv_free_ratio(self) -> float:
        if self.kv_total_blocks <= 0:
            return 1.0
        return self.kv_free_blocks / self.kv_total_blocks


@dataclass(frozen=True)
class PhaseDecision:
    phase: Phase
    source: DecisionSource
    target_ratio: float


@dataclass
class IterRecord:
    iter_id: int
    ts_ms: float
    decision: PhaseDecision
    actual_phase: Phase
    phase_iters: int
    snapshot: QueueSnapshot
    batch_num_reqs: int
    batch_num_tokens: int
    iter_duration_ms: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "iter_id": self.iter_id,
            "ts_ms": self.ts_ms,
            "phase": self.decision.phase,
            "actual_phase": self.actual_phase,
            "source": self.decision.source,
            "target_ratio": self.decision.target_ratio,
            "phase_iters": self.phase_iters,
            "waiting_depth": self.snapshot.waiting_depth,
            "waiting_oldest_age_ms": self.snapshot.waiting_oldest_age_ms,
            "finished_prefill_depth": self.snapshot.finished_prefill_depth,
            "running_depth": self.snapshot.running_depth,
            "decode_oldest_silence_ms": self.snapshot.decode_oldest_silence_ms,
            "kv_free_ratio": self.snapshot.kv_free_ratio,
            "batch_num_reqs": self.batch_num_reqs,
            "batch_num_tokens": self.batch_num_tokens,
            "iter_duration_ms": self.iter_duration_ms,
        }


@dataclass
class ControllerTickRecord:
    """One PID update tick. Emitted only when SLOReactiveController actually
    runs the gradient step (i.e. cache-miss branch); cached returns do NOT
    emit. Captures both raw and effective error signals so post-hoc analysis
    can tell apart "PID didn't see the error" vs "PID saw it but the
    saturation/clamp state machine zeroed it out".
    """

    iter_id: int
    ts_ms: float
    # Decision movement
    ratio_before: float
    ratio_after: float
    delta_slo: float
    delta_q: float
    deadband_dropped: bool
    # Error signals
    err_ttft_raw: float
    err_tpot_raw: float
    err_ttft: float
    err_tpot_effective: float
    ttft_viol_rate: float
    tpot_viol_rate: float
    n_ttft_window: int
    n_tpot_window: int
    # State-machine flags
    tpot_saturated: bool
    tpot_clamp_active: bool
    tpot_violate_streak: int
    tpot_recovery_streak: int
    kv_freeze_triggered: bool
    cold_start: bool

    def to_json(self) -> dict:
        return {
            "iter_id": self.iter_id,
            "ts_ms": self.ts_ms,
            "ratio_before": self.ratio_before,
            "ratio_after": self.ratio_after,
            "delta_slo": self.delta_slo,
            "delta_q": self.delta_q,
            "deadband_dropped": self.deadband_dropped,
            "err_ttft_raw": self.err_ttft_raw,
            "err_tpot_raw": self.err_tpot_raw,
            "err_ttft": self.err_ttft,
            "err_tpot_effective": self.err_tpot_effective,
            "ttft_viol_rate": self.ttft_viol_rate,
            "tpot_viol_rate": self.tpot_viol_rate,
            "n_ttft_window": self.n_ttft_window,
            "n_tpot_window": self.n_tpot_window,
            "tpot_saturated": self.tpot_saturated,
            "tpot_clamp_active": self.tpot_clamp_active,
            "tpot_violate_streak": self.tpot_violate_streak,
            "tpot_recovery_streak": self.tpot_recovery_streak,
            "kv_freeze_triggered": self.kv_freeze_triggered,
            "cold_start": self.cold_start,
        }


@dataclass
class ChunkPlanRecord:
    """One P-iter's chunk plan summary. Emitted after the resume + fresh
    passes complete in chunked_schedule. Diagnoses whether
    `prefill_chunk_tokens` is actively limiting work or just nominally
    present (truncated_count > 0 → real limit; budget_used/budget_in < 0.5
    → headroom to shrink chunk size with near-zero overhead cost).
    """

    iter_id: int
    ts_ms: float
    chunk_budget_in: int
    chunk_budget_used: int
    token_budget_in: int
    token_budget_used: int
    num_resume_reqs: int
    num_fresh_reqs: int
    resume_tokens: int
    fresh_tokens: int
    truncated_count: int

    def to_json(self) -> dict:
        return {
            "iter_id": self.iter_id,
            "ts_ms": self.ts_ms,
            "chunk_budget_in": self.chunk_budget_in,
            "chunk_budget_used": self.chunk_budget_used,
            "token_budget_in": self.token_budget_in,
            "token_budget_used": self.token_budget_used,
            "num_resume_reqs": self.num_resume_reqs,
            "num_fresh_reqs": self.num_fresh_reqs,
            "resume_tokens": self.resume_tokens,
            "fresh_tokens": self.fresh_tokens,
            "truncated_count": self.truncated_count,
        }


@dataclass
class RequestRecord:
    request_id: str
    prompt_tokens: int
    admission_ts_ms: float
    first_token_ts_ms: Optional[float] = None
    finish_ts_ms: Optional[float] = None
    output_tokens: int = 0
    decode_intervals_ms: list = field(default_factory=list)
    _last_token_ts_ms: Optional[float] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_ts_ms is None:
            return None
        return self.first_token_ts_ms - self.admission_ts_ms

    @property
    def tpot_ms_mean(self) -> Optional[float]:
        if not self.decode_intervals_ms:
            return None
        return sum(self.decode_intervals_ms) / len(self.decode_intervals_ms)

    @property
    def tpot_ms_p99(self) -> Optional[float]:
        if not self.decode_intervals_ms:
            return None
        s = sorted(self.decode_intervals_ms)
        idx = max(0, int(round(0.99 * (len(s) - 1))))
        return s[idx]

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "admission_ts_ms": self.admission_ts_ms,
            "first_token_ts_ms": self.first_token_ts_ms,
            "finish_ts_ms": self.finish_ts_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms_mean": self.tpot_ms_mean,
            "tpot_ms_p99": self.tpot_ms_p99,
        }
