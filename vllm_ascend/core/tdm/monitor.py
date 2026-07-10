from typing import Any, Optional

from .tracker import RequestTracker
from .types import QueueSnapshot


class QueueMonitor:
    """Snapshot scheduler queue state once per iter (cheap, O(1) reads).

    When a RequestTracker is wired in (P1.7b), snapshot() also computes
    `decode_oldest_silence_ms` — max age over running requests that have
    already produced their first token. Used by the fast loop to detect
    TPOT starvation (decode head silent for too long → force decode).
    Cost: O(running_depth) per iter, negligible.
    """

    def __init__(self, tracker: Optional[RequestTracker] = None):
        self._tracker = tracker

    def snapshot(self, scheduler: Any, now_ms: float) -> QueueSnapshot:
        waiting = getattr(scheduler, "waiting", None) or []
        running = getattr(scheduler, "running", None) or []
        finished_prefill = getattr(scheduler, "finished_prefill_reqs",
                                   None) or []

        # Oldest waiting age — best-effort; falls back to 0 if Request lacks
        # an admission timestamp. Real per-request admission ts is tracked by
        # RequestTracker; this field is only a coarse hint for the controller.
        oldest_age_ms = 0.0
        if waiting:
            head = waiting[0]
            arrival = getattr(head, "arrival_time", None)
            if arrival is not None:
                # arrival_time is wall-clock seconds; convert to monotonic ms
                # by treating it as a relative reference.
                import time as _t
                oldest_age_ms = max(0.0, _t.time() * 1000.0 - arrival * 1000.0)

        kv_total = 0
        kv_free = 0
        kv_cfg = getattr(scheduler, "kv_cache_config", None)
        if kv_cfg is not None:
            kv_total = int(getattr(kv_cfg, "num_blocks", 0) or 0)
        kv_mgr = getattr(scheduler, "kv_cache_manager", None)
        if kv_mgr is not None:
            pool = getattr(kv_mgr, "block_pool", None)
            if pool is not None and hasattr(pool, "get_num_free_blocks"):
                try:
                    kv_free = int(pool.get_num_free_blocks())
                except Exception:
                    kv_free = 0

        # P1.7b TPOT starvation signal: scan running reqs that already
        # produced first token, take max(now - _last_token_ts_ms). Reqs
        # still in prefill (no first token yet) are skipped — their TTFT
        # risk is covered by waiting_oldest_age_ms / urgency_ttft path.
        decode_silence_ms = 0.0
        if self._tracker is not None and running:
            records = self._tracker._records
            for req in running:
                req_id = getattr(req, "request_id", None)
                if req_id is None:
                    continue
                rec = records.get(req_id)
                if rec is None or rec._last_token_ts_ms is None:
                    continue
                age = now_ms - rec._last_token_ts_ms
                if age > decode_silence_ms:
                    decode_silence_ms = age

        return QueueSnapshot(
            waiting_depth=len(waiting),
            waiting_oldest_age_ms=oldest_age_ms,
            finished_prefill_depth=len(finished_prefill),
            running_depth=len(running),
            kv_free_blocks=kv_free,
            kv_total_blocks=kv_total,
            decode_oldest_silence_ms=decode_silence_ms,
        )
