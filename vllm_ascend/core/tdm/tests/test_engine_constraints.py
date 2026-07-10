from vllm_ascend.core.tdm.boundary import BoundaryGuard
from vllm_ascend.core.tdm.constraints import HardConstraints
from vllm_ascend.core.tdm.engine import PhaseEngine
from vllm_ascend.core.tdm.types import QueueSnapshot


def test_min_slice_blocks_premature_switch():
    c = HardConstraints(min_slice_iters=3, max_slice_iters=10)
    out, src = c.enforce("decode", "prefill", phase_iters=1)
    assert out == "prefill" and src == "constraint_min_slice"


def test_max_slice_forces_switch():
    c = HardConstraints(min_slice_iters=2, max_slice_iters=5)
    out, src = c.enforce("decode", "decode", phase_iters=5)
    assert out == "prefill" and src == "constraint_max_slice"


def test_normal_pass_through():
    c = HardConstraints(min_slice_iters=2, max_slice_iters=8)
    out, src = c.enforce("prefill", "decode", phase_iters=4)
    assert out == "prefill" and src == "controller"


def test_boundary_blocks_prefill_under_kv_pressure():
    g = BoundaryGuard(kv_free_watermark=0.10)
    snap = QueueSnapshot(0, 0.0, 0, 4, kv_free_blocks=3, kv_total_blocks=100)
    out, src = g.override("prefill", snap)
    assert out == "decode" and src == "boundary_kv_pressure"


def test_boundary_passes_when_kv_free():
    g = BoundaryGuard(kv_free_watermark=0.10)
    snap = QueueSnapshot(0, 0.0, 0, 4, kv_free_blocks=80, kv_total_blocks=100)
    out, src = g.override("prefill", snap)
    assert out == "prefill" and src == "controller"


def test_engine_reconcile_matches_actual():
    e = PhaseEngine(initial_phase="decode")

    class FakeSched:
        phase = ""
    fs = FakeSched()
    e.apply(fs, "prefill")
    assert fs.phase == "prefill"
    assert e.phase == "prefill"

    # Parent auto-flipped to decode (Option W) — engine must reset counter.
    flipped = e.reconcile(actual="decode", candidate="prefill")
    assert flipped is True
    assert e.phase == "decode"
    assert e.phase_iters == 1

    e.apply(fs, "decode")
    e.reconcile(actual="decode", candidate="decode")
    assert e.phase_iters == 2
