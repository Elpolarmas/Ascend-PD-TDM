"""End-to-end pipeline integration without instantiating AscendScheduler.

Mirrors TDMScheduler.schedule()'s exact call order so any future drift in the
real scheduler that breaks this also breaks here.
"""
from vllm_ascend.core.tdm.boundary import BoundaryGuard
from vllm_ascend.core.tdm.constraints import HardConstraints
from vllm_ascend.core.tdm.controller import StaticRatioController
from vllm_ascend.core.tdm.engine import PhaseEngine
from vllm_ascend.core.tdm.selector import TokenBucketSelector
from vllm_ascend.core.tdm.types import QueueSnapshot


class FakeParent:
    """Mimics AscendScheduler's L93/L103 phase-handling: if asked to run
    PREFILL but its waiting queue is empty, it auto-flips to DECODE."""

    def __init__(self):
        self.phase = ""
        self.waiting = []
        self.running = []

    def schedule(self):
        if self.phase == "prefill" and not self.waiting and not self.running:
            self.phase = "decode"
        return None


def _snap(waiting, running, kv_free=100, kv_total=100):
    return QueueSnapshot(
        waiting_depth=waiting,
        waiting_oldest_age_ms=0.0,
        finished_prefill_depth=0,
        running_depth=running,
        kv_free_blocks=kv_free,
        kv_total_blocks=kv_total,
    )


def _step(parent, snap, controller, selector, constraints, boundary,
          engine, iter_id):
    target = controller.get_target_ratio(snap, iter_id)
    planned = selector.peek(target, snap)
    after_c, _ = constraints.enforce(planned, engine.phase, engine.phase_iters)
    candidate, _ = boundary.override(after_c, snap)
    engine.apply(parent, candidate)
    parent.schedule()
    actual = parent.phase if parent.phase in ("prefill", "decode") else candidate
    flipped = engine.reconcile(actual, candidate)
    selector.commit(actual)
    return candidate, actual, flipped


def test_option_w_auto_flip_keeps_bucket_honest():
    parent = FakeParent()  # waiting & running both empty
    controller = StaticRatioController(static_ratio=1.0, update_interval=1)
    selector = TokenBucketSelector(cap=2)
    constraints = HardConstraints(min_slice_iters=1, max_slice_iters=999)
    boundary = BoundaryGuard(kv_free_watermark=0.0)
    engine = PhaseEngine(initial_phase="decode")

    flipped_count = 0
    for i in range(5):
        snap = _snap(waiting=0, running=0)  # selector will pick decode anyway
        cand, actual, flipped = _step(parent, snap, controller, selector,
                                      constraints, boundary, engine, i)
        if flipped:
            flipped_count += 1
        assert actual == "decode"

    # Bucket accumulates but never gets consumed (no actual prefill ran).
    # Critical correctness: tokens stayed in bucket → Option W safe.
    assert selector.tokens > 0


def test_kv_pressure_blocks_prefill_mid_stream():
    parent = FakeParent()
    parent.waiting = ["fake_req"]
    controller = StaticRatioController(static_ratio=1.0, update_interval=1)
    selector = TokenBucketSelector(cap=4)
    constraints = HardConstraints(min_slice_iters=1, max_slice_iters=999)
    boundary = BoundaryGuard(kv_free_watermark=0.10)
    engine = PhaseEngine(initial_phase="decode")

    snap = _snap(waiting=2, running=4, kv_free=5, kv_total=100)  # 5% < 10%
    cand, actual, _ = _step(parent, snap, controller, selector, constraints,
                            boundary, engine, 0)
    assert cand == "decode"  # boundary overrode prefill plan


def test_max_slice_breaks_long_decode_slice():
    parent = FakeParent()
    parent.waiting = ["w1", "w2"]
    parent.running = ["r1"]
    controller = StaticRatioController(static_ratio=0.0, update_interval=1)
    selector = TokenBucketSelector(cap=4)
    constraints = HardConstraints(min_slice_iters=1, max_slice_iters=3)
    boundary = BoundaryGuard(kv_free_watermark=0.0)
    engine = PhaseEngine(initial_phase="decode")

    cands = []
    for i in range(5):
        snap = _snap(waiting=2, running=1)
        cand, _, _ = _step(parent, snap, controller, selector, constraints,
                           boundary, engine, i)
        cands.append(cand)
    # static_ratio=0 → controller never wants prefill; max_slice=3 must
    # force one once decode has run for 3 iters.
    assert "prefill" in cands
