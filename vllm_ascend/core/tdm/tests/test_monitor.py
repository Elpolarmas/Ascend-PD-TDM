"""QueueMonitor tests, including P1.7b decode_oldest_silence_ms signal."""
from vllm_ascend.core.tdm.monitor import QueueMonitor
from vllm_ascend.core.tdm.tracker import RequestTracker


class FakeReq:
    def __init__(self, req_id, arrival_time=None):
        self.request_id = req_id
        if arrival_time is not None:
            self.arrival_time = arrival_time


class FakeScheduler:
    def __init__(self, waiting=None, running=None, finished_prefill=None):
        self.waiting = waiting or []
        self.running = running or []
        self.finished_prefill_reqs = finished_prefill or []
        self.kv_cache_config = None
        self.kv_cache_manager = None


def test_snapshot_no_tracker_silence_zero():
    """Backward compat: monitor without tracker still works; silence=0."""
    mon = QueueMonitor()
    sched = FakeScheduler(running=[FakeReq("r1"), FakeReq("r2")])
    snap = mon.snapshot(sched, now_ms=1000.0)
    assert snap.running_depth == 2
    assert snap.decode_oldest_silence_ms == 0.0


def test_snapshot_empty_running_silence_zero():
    tr = RequestTracker()
    mon = QueueMonitor(tracker=tr)
    sched = FakeScheduler(running=[])
    snap = mon.snapshot(sched, now_ms=1000.0)
    assert snap.decode_oldest_silence_ms == 0.0


def test_snapshot_running_in_prefill_silence_zero():
    """Running req that has not yet produced first token must NOT contribute
    to decode silence — its TTFT risk is the urgency_ttft path's job."""
    tr = RequestTracker()
    tr.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    # No token produced yet → _last_token_ts_ms is None
    mon = QueueMonitor(tracker=tr)
    sched = FakeScheduler(running=[FakeReq("r1")])
    snap = mon.snapshot(sched, now_ms=500.0)
    assert snap.decode_oldest_silence_ms == 0.0


def test_snapshot_decode_silence_single_req():
    tr = RequestTracker()
    tr.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    tr.on_token_produced("r1", 1, ts_ms=100.0)  # first token at t=100
    mon = QueueMonitor(tracker=tr)
    sched = FakeScheduler(running=[FakeReq("r1")])
    snap = mon.snapshot(sched, now_ms=350.0)
    assert snap.decode_oldest_silence_ms == 250.0


def test_snapshot_decode_silence_max_over_running():
    tr = RequestTracker()
    tr.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    tr.on_token_produced("r1", 1, ts_ms=100.0)  # last_token=100
    tr.on_admit("r2", prompt_tokens=10, ts_ms=50.0)
    tr.on_token_produced("r2", 1, ts_ms=200.0)  # last_token=200
    tr.on_token_produced("r2", 1, ts_ms=300.0)  # last_token=300 (fresher)
    mon = QueueMonitor(tracker=tr)
    sched = FakeScheduler(running=[FakeReq("r1"), FakeReq("r2")])
    snap = mon.snapshot(sched, now_ms=400.0)
    # r1 silent 300ms, r2 silent 100ms → max = 300
    assert snap.decode_oldest_silence_ms == 300.0


def test_snapshot_decode_silence_skips_req_not_in_tracker():
    """Defensive: running req with no tracker record (edge case during
    admit / shutdown races) is silently skipped, not crashed."""
    tr = RequestTracker()
    tr.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    tr.on_token_produced("r1", 1, ts_ms=100.0)
    mon = QueueMonitor(tracker=tr)
    # r_phantom is in running but never admitted to tracker
    sched = FakeScheduler(running=[FakeReq("r1"), FakeReq("r_phantom")])
    snap = mon.snapshot(sched, now_ms=350.0)
    assert snap.decode_oldest_silence_ms == 250.0


def test_snapshot_decode_silence_mixed_prefill_decode():
    """Mix: one running req still in prefill (skipped), one already
    decoding. Silence is computed only over the decoding one."""
    tr = RequestTracker()
    tr.on_admit("r_decoding", prompt_tokens=10, ts_ms=0.0)
    tr.on_token_produced("r_decoding", 1, ts_ms=50.0)
    tr.on_admit("r_prefill", prompt_tokens=10, ts_ms=10.0)
    # r_prefill: no token produced yet
    mon = QueueMonitor(tracker=tr)
    sched = FakeScheduler(
        running=[FakeReq("r_decoding"), FakeReq("r_prefill")])
    snap = mon.snapshot(sched, now_ms=500.0)
    # only r_decoding counts: 500 - 50 = 450
    assert snap.decode_oldest_silence_ms == 450.0
