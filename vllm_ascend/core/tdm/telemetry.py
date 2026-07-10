import json
import os
import queue
import threading
from collections import deque
from typing import Optional

from .types import (ChunkPlanRecord, ControllerTickRecord, IterRecord,
                    RequestRecord)

_SENTINEL = object()


class Telemetry:
    """Two channels:
      - Hot ring buffer (in-memory deque) that the controller reads each
        update tick to compute SLO attainment / latency stats.
      - Cold JSONL log (background thread, append-only) for offline analysis.
    """

    def __init__(
        self,
        window_size: int = 32,
        cold_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        request_buffer: int = 256,
        cold_enabled: bool = True,
    ):
        self._iters: deque[IterRecord] = deque(maxlen=window_size)
        self._reqs: deque[RequestRecord] = deque(maxlen=request_buffer)
        # iter_id → IterRecord, for late duration backfill before flush.
        self._pending_iters: dict[int, IterRecord] = {}

        self._cold_enabled = cold_enabled and cold_dir is not None
        self._iter_path: Optional[str] = None
        self._req_path: Optional[str] = None
        self._ctrl_path: Optional[str] = None
        self._chunk_path: Optional[str] = None
        self._q: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        if self._cold_enabled:
            os.makedirs(cold_dir, exist_ok=True)
            rid = run_id or "run"
            self._iter_path = os.path.join(cold_dir, f"{rid}_iter.jsonl")
            self._req_path = os.path.join(cold_dir, f"{rid}_req.jsonl")
            self._ctrl_path = os.path.join(cold_dir, f"{rid}_ctrl.jsonl")
            self._chunk_path = os.path.join(cold_dir, f"{rid}_chunk.jsonl")
            self._q = queue.Queue(maxsize=4096)
            self._thread = threading.Thread(
                target=self._writer_loop, name="tdm-telemetry", daemon=True)
            self._thread.start()

    # ---------- hot ring ----------

    def record_iter(self, rec: IterRecord) -> None:
        self._iters.append(rec)
        self._pending_iters[rec.iter_id] = rec

    def backfill_iter_duration(self, iter_id: int,
                               duration_ms: float) -> None:
        rec = self._pending_iters.pop(iter_id, None)
        if rec is None:
            return
        rec.iter_duration_ms = duration_ms
        self._enqueue(("iter", rec.to_json()))

    def record_request(self, rec: RequestRecord) -> None:
        self._reqs.append(rec)
        self._enqueue(("req", rec.to_json()))

    def record_controller_tick(self, rec: ControllerTickRecord) -> None:
        # Hot ring intentionally skipped — PID internals are diagnostic-only
        # and only consumed by post-hoc analysis from the cold log.
        self._enqueue(("ctrl", rec.to_json()))

    def record_chunk_plan(self, rec: ChunkPlanRecord) -> None:
        # Hot ring intentionally skipped — same reason as record_controller_tick.
        self._enqueue(("chunk", rec.to_json()))

    def recent_iters(self, n: int) -> list[IterRecord]:
        if n >= len(self._iters):
            return list(self._iters)
        return list(self._iters)[-n:]

    def recent_requests(self, n: int) -> list[RequestRecord]:
        if n >= len(self._reqs):
            return list(self._reqs)
        return list(self._reqs)[-n:]

    # ---------- cold log ----------

    def _enqueue(self, item) -> None:
        if not self._cold_enabled or self._q is None:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # Drop on overflow rather than block schedule().
            pass

    def _writer_loop(self) -> None:
        assert (self._iter_path and self._req_path and self._ctrl_path
                and self._chunk_path)
        iter_f = open(self._iter_path, "a", buffering=1)
        req_f = open(self._req_path, "a", buffering=1)
        ctrl_f = open(self._ctrl_path, "a", buffering=1)
        chunk_f = open(self._chunk_path, "a", buffering=1)
        kind_to_file = {
            "iter": iter_f,
            "req": req_f,
            "ctrl": ctrl_f,
            "chunk": chunk_f,
        }
        try:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    break
                kind, payload = item
                line = json.dumps(payload, separators=(",", ":")) + "\n"
                f = kind_to_file.get(kind, req_f)  # unknown → req for safety
                f.write(line)
        finally:
            for f in kind_to_file.values():
                f.close()

    def close(self) -> None:
        if self._cold_enabled and self._q is not None:
            try:
                self._q.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass
            if self._thread is not None:
                self._thread.join(timeout=2.0)
