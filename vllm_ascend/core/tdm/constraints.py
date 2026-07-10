from .types import DecisionSource, Phase


class HardConstraints:
    """min_slice_iters: ignore switch requests until the current slice has
    been held long enough (debounce).
    max_slice_iters: force a switch once a slice has been held too long
    (anti-starvation upper bound).
    """

    def __init__(self, min_slice_iters: int = 2, max_slice_iters: int = 8):
        self.min_slice_iters = max(1, min_slice_iters)
        self.max_slice_iters = max(self.min_slice_iters, max_slice_iters)

    def enforce(self, planned: Phase, current: Phase,
                phase_iters: int) -> tuple[Phase, DecisionSource]:
        if phase_iters >= self.max_slice_iters and planned == current:
            other: Phase = "decode" if current == "prefill" else "prefill"
            return other, "constraint_max_slice"
        if planned != current and phase_iters < self.min_slice_iters:
            return current, "constraint_min_slice"
        return planned, "controller"
