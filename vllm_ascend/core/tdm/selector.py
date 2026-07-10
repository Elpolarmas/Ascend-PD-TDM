from .types import Phase, QueueSnapshot


class TokenBucketSelector:
    """Two-phase peek/commit selector driven by a target prefill ratio.

    Bucket holds up to `cap` tokens; each iter we credit `target_ratio`
    tokens. peek() chooses PREFILL if the bucket has >= 1 token AND there
    are pending prefills, else DECODE. commit(actual) consumes one token
    iff the actually executed phase was PREFILL.

    Decoupling peek from consume is deliberate: the parent AscendScheduler
    can auto-flip our intended PREFILL → DECODE (Option W). We must charge
    the bucket against the phase that ran, not the phase we asked for.

    P1.7 leading indicator (urgency_ttft): force PREFILL when head-of-line
    waiting request has aged past urgency_threshold_ms.

    P1.7b bidirectional fast loop (starvation_tpot): force DECODE when
    the oldest running decode has been silent past starvation_threshold_ms.
    The two overrides are mutually exclusive by construction: urgency only
    triggers when the bucket wants decode (tokens < 1), starvation only
    when the bucket wants prefill (tokens >= 1), so a single iter can hit
    at most one. Overrides are exposed via `last_override_urgent` /
    `last_override_starvation` so callers can stamp the DecisionSource.
    """

    def __init__(self, cap: int = 4, initial: float = 0.0):
        self.cap = max(1, cap)
        self._tokens: float = max(0.0, min(float(cap), initial))
        self._last_override_urgent: bool = False
        self._last_override_starvation: bool = False

    @property
    def tokens(self) -> float:
        return self._tokens

    @property
    def last_override_urgent(self) -> bool:
        """Whether the most recent peek() returned PREFILL via the TTFT
        urgency override (rather than the normal bucket path). Reset on
        every peek()."""
        return self._last_override_urgent

    @property
    def last_override_starvation(self) -> bool:
        """Whether the most recent peek() returned DECODE via the TPOT
        starvation override. Reset on every peek()."""
        return self._last_override_starvation

    def peek(
        self,
        target_ratio: float,
        snap: QueueSnapshot,
        urgency_threshold_ms: float = 0.0,
        charge_bucket_on_urgency: bool = True,
        starvation_threshold_ms: float = 0.0,
        starvation_decouple_bucket: bool = False,
    ) -> Phase:
        # Credit this iter's share of the prefill budget.
        r = max(0.0, min(1.0, target_ratio))
        self._tokens = min(self.cap, self._tokens + r)
        self._last_override_urgent = False
        self._last_override_starvation = False

        has_prefill_work = snap.waiting_depth > 0
        has_decode_work = (snap.running_depth + snap.finished_prefill_depth) > 0

        if not has_prefill_work and not has_decode_work:
            return "decode"
        if not has_prefill_work:
            return "decode"
        if not has_decode_work:
            return "prefill"

        # P1.7b TPOT starvation override: rescue stalled decode head.
        # Default (decouple=False, M3.3 back-compat): fires only when bucket
        # wants prefill (tokens >= 1) AND silence > threshold. Empirically
        # this AND-gate pins rescue to 0.5% of iters on saturated code even
        # though silence > threshold on 20% — the bidirectional-warning
        # thesis is not actually exercised by this gate.
        # decouple=True: fires on silence alone. Bucket is still NOT debited
        # (preserve slow-loop P/D anchor) so rescue can override either
        # bucket-decode (no-op direction-wise) or bucket-prefill iters.
        # Priority above urgency_ttft (TPOT stall is a flowing-output break,
        # more user-visible than waiting a bit longer for TTFT).
        if (starvation_threshold_ms > 0.0
                and snap.decode_oldest_silence_ms >= starvation_threshold_ms
                and (starvation_decouple_bucket or self._tokens >= 1.0)):
            self._last_override_starvation = True
            return "decode"

        # P1.7 urgency override: head-of-line waiting request about to
        # blow TTFT — force PREFILL even when bucket says decode.
        # Applied only when there is real contention (both queues nonempty);
        # the unconditional has_prefill_work-only branch above already
        # handles the no-decode-pressure case without needing urgency.
        if (urgency_threshold_ms > 0.0
                and snap.waiting_oldest_age_ms >= urgency_threshold_ms
                and self._tokens < 1.0):
            self._last_override_urgent = True
            if charge_bucket_on_urgency:
                # Zero the bucket so the slow loop "sees" this prefill as
                # accounted for (next iter's ratio credit resumes from 0).
                # NOT going negative: we don't want urgency to forever
                # suppress the next natural prefill.
                self._tokens = 0.0
            return "prefill"

        if self._tokens >= 1.0:
            return "prefill"
        return "decode"

    def commit(self, actual: Phase) -> None:
        if actual == "prefill" and self._tokens >= 1.0:
            self._tokens -= 1.0
