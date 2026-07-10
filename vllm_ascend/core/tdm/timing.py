import time
from collections import defaultdict, deque
from typing import Optional

from .types import Phase


class TimeAccountant:
    """Wall-clock + per-phase iter duration sliding-window mean."""

    def __init__(self, window: int = 32):
        self._window = window
        self._per_phase: dict[Phase, deque] = defaultdict(
            lambda: deque(maxlen=window))
        self._last_schedule_ts_ms: Optional[float] = None

    @staticmethod
    def now_ms() -> float:
        return time.monotonic() * 1000.0

    def mark_schedule(self, ts_ms: float) -> None:
        self._last_schedule_ts_ms = ts_ms

    def take_iter_duration(self, now_ms: float) -> Optional[float]:
        if self._last_schedule_ts_ms is None:
            return None
        return now_ms - self._last_schedule_ts_ms

    def record_iter(self, phase: Phase, duration_ms: float) -> None:
        self._per_phase[phase].append(duration_ms)

    def mean_duration(self, phase: Phase) -> Optional[float]:
        d = self._per_phase.get(phase)
        if not d:
            return None
        return sum(d) / len(d)
