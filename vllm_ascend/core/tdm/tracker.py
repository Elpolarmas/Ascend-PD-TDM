from typing import Iterator, Optional

from .types import RequestRecord


class RequestTracker:
    """Per-request lifecycle timestamps, stored off to the side of vLLM's
    Request objects so we never mutate them.

    Signal availability:
      - on_admit:  request enters waiting → admission_ts_ms set
      - on_token_produced (first):  first_token_ts_ms set → ttft now computable
      - on_token_produced (subsequent):  decode_intervals_ms grows → in-flight
        tpot_ms_mean reflects current decode rate
      - on_finish: terminal record popped from _records

    P1.6b: on_token_produced returns the record at first-token so the caller
    (scheduler) can push it to telemetry immediately — no longer waiting for
    on_finish to surface ttft to the controller.

    P1.6b: iter_in_flight_records exposes still-running records so the
    controller can poll fresh in-flight tpot samples each tick instead of
    only seeing finished-request means (which lag the entire request).
    """

    def __init__(self):
        self._records: dict[str, RequestRecord] = {}

    def on_admit(self, req_id: str, prompt_tokens: int, ts_ms: float) -> None:
        if req_id in self._records:
            return
        self._records[req_id] = RequestRecord(
            request_id=req_id,
            prompt_tokens=prompt_tokens,
            admission_ts_ms=ts_ms,
        )

    def on_token_produced(
            self, req_id: str, num_new_tokens: int,
            ts_ms: float) -> Optional[RequestRecord]:
        """Update record on token production. Returns the record only on the
        first-token transition (so caller can push ttft to telemetry without
        waiting for on_finish). Subsequent calls return None."""
        rec = self._records.get(req_id)
        if rec is None or num_new_tokens <= 0:
            return None
        if rec.first_token_ts_ms is None:
            rec.first_token_ts_ms = ts_ms
            rec._last_token_ts_ms = ts_ms
            rec.output_tokens += num_new_tokens
            return rec
        if rec._last_token_ts_ms is not None:
            interval = (ts_ms - rec._last_token_ts_ms) / max(1, num_new_tokens)
            for _ in range(num_new_tokens):
                rec.decode_intervals_ms.append(interval)
        rec._last_token_ts_ms = ts_ms
        rec.output_tokens += num_new_tokens
        return None

    def on_finish(self, req_id: str,
                  ts_ms: float) -> Optional[RequestRecord]:
        rec = self._records.pop(req_id, None)
        if rec is None:
            return None
        rec.finish_ts_ms = ts_ms
        return rec

    def iter_in_flight_records(self) -> Iterator[RequestRecord]:
        """Yield still-running records (admitted but not yet finished).
        Used by the controller for in-flight tpot polling and by the selector
        (P1.7b) for decode-starvation detection."""
        yield from self._records.values()
