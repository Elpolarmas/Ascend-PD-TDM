from typing import Any

from .types import Phase


class PhaseEngine:
    """Mechanical phase application + phase_iters bookkeeping.

    Reconciles the *intended* phase with what the parent scheduler actually
    ran, since AscendScheduler.schedule() may auto-flip prefill→decode when
    its waiting/running queues both empty (Option W).
    """

    def __init__(self, initial_phase: Phase = "decode"):
        self._phase: Phase = initial_phase
        self._phase_iters: int = 0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def phase_iters(self) -> int:
        return self._phase_iters

    def apply(self, scheduler: Any, candidate: Phase) -> None:
        scheduler.phase = candidate
        self._phase = candidate

    def reconcile(self, actual: Phase, candidate: Phase) -> bool:
        """Update phase_iters by the actually executed phase.
        Returns True if parent auto-flipped (actual != candidate).
        """
        flipped = actual != candidate
        if actual == self._phase:
            self._phase_iters += 1
        else:
            self._phase = actual
            self._phase_iters = 1
        return flipped
