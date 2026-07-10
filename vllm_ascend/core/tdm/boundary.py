from .types import DecisionSource, Phase, QueueSnapshot


class BoundaryGuard:
    """Physical safety: forbid switching to PREFILL when KV cache is below
    watermark — prefill allocates new blocks and would risk OOM-style preempt.
    Empty-queue handling is delegated to the parent scheduler (Option W
    auto-flip on L103-104), so we don't replicate it here.
    """

    def __init__(self, kv_free_watermark: float = 0.05):
        self.kv_free_watermark = kv_free_watermark

    def override(self, candidate: Phase,
                 snap: QueueSnapshot) -> tuple[Phase, DecisionSource]:
        if (candidate == "prefill"
                and snap.kv_free_ratio < self.kv_free_watermark):
            return "decode", "boundary_kv_pressure"
        return candidate, "controller"
