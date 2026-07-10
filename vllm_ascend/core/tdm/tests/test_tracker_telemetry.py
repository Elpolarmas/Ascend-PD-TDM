import json
import os
import tempfile

from vllm_ascend.core.tdm.telemetry import Telemetry
from vllm_ascend.core.tdm.tracker import RequestTracker
from vllm_ascend.core.tdm.types import (ChunkPlanRecord,
                                        ControllerTickRecord, IterRecord,
                                        PhaseDecision, QueueSnapshot)


def test_tracker_ttft_and_tpot():
    t = RequestTracker()
    t.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    t.on_token_produced("r1", 1, ts_ms=50.0)    # first token: TTFT=50
    t.on_token_produced("r1", 1, ts_ms=70.0)    # interval 20
    t.on_token_produced("r1", 1, ts_ms=110.0)   # interval 40
    rec = t.on_finish("r1", ts_ms=120.0)
    assert rec is not None
    assert rec.ttft_ms == 50.0
    assert rec.tpot_ms_mean == 30.0
    assert rec.output_tokens == 3


def test_telemetry_hot_ring_and_cold_log():
    with tempfile.TemporaryDirectory() as d:
        tm = Telemetry(window_size=4, cold_dir=d, run_id="t",
                       cold_enabled=True)
        snap = QueueSnapshot(0, 0.0, 0, 0, 0, 0)
        for i in range(6):
            r = IterRecord(
                iter_id=i, ts_ms=float(i),
                decision=PhaseDecision("decode", "controller", 0.5),
                actual_phase="decode",
                phase_iters=1, snapshot=snap,
                batch_num_reqs=0, batch_num_tokens=0,
            )
            tm.record_iter(r)
            tm.backfill_iter_duration(i, 10.0)
        recent = tm.recent_iters(10)
        # Window=4 → only the last 4 retained in the hot ring.
        assert len(recent) == 4
        assert [r.iter_id for r in recent] == [2, 3, 4, 5]
        tm.close()
        with open(os.path.join(d, "t_iter.jsonl")) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 6
        assert all(l["iter_duration_ms"] == 10.0 for l in lines)


def _mk_tick(iter_id: int) -> ControllerTickRecord:
    return ControllerTickRecord(
        iter_id=iter_id,
        ts_ms=float(iter_id),
        ratio_before=0.30,
        ratio_after=0.32,
        delta_slo=0.04,
        delta_q=-0.02,
        deadband_dropped=False,
        err_ttft_raw=0.10,
        err_tpot_raw=-0.02,
        err_ttft=0.10,
        err_tpot_effective=0.0,
        ttft_viol_rate=0.15,
        tpot_viol_rate=0.03,
        n_ttft_window=64,
        n_tpot_window=64,
        tpot_saturated=False,
        tpot_clamp_active=False,
        tpot_violate_streak=0,
        tpot_recovery_streak=2,
        kv_freeze_triggered=False,
        cold_start=False,
    )


def _mk_chunk(iter_id: int) -> ChunkPlanRecord:
    return ChunkPlanRecord(
        iter_id=iter_id,
        ts_ms=float(iter_id),
        chunk_budget_in=2048,
        chunk_budget_used=2048,
        token_budget_in=8192,
        token_budget_used=2048,
        num_resume_reqs=1,
        num_fresh_reqs=0,
        resume_tokens=2048,
        fresh_tokens=0,
        truncated_count=1,
    )


def test_record_serialization_roundtrip():
    """to_json output is JSON-serializable and round-trips through json."""
    tick = _mk_tick(7)
    chunk = _mk_chunk(7)
    j_tick = json.loads(json.dumps(tick.to_json()))
    j_chunk = json.loads(json.dumps(chunk.to_json()))
    # Spot-check a few discriminating fields per record type.
    assert j_tick["ratio_after"] == 0.32
    assert j_tick["err_tpot_effective"] == 0.0
    assert j_tick["tpot_saturated"] is False
    assert j_chunk["chunk_budget_in"] == 2048
    assert j_chunk["truncated_count"] == 1


def test_telemetry_four_kind_dispatch_no_crosstalk():
    """ctrl/chunk/iter/req each land in their own JSONL — verify by sentinel
    field unique to each record type. Catches a writer-loop dispatch bug
    where a kind would silently fall through to the wrong file."""
    with tempfile.TemporaryDirectory() as d:
        tm = Telemetry(window_size=4, cold_dir=d, run_id="t",
                       cold_enabled=True)

        snap = QueueSnapshot(0, 0.0, 0, 0, 0, 0)
        for i in range(3):
            r = IterRecord(
                iter_id=i, ts_ms=float(i),
                decision=PhaseDecision("decode", "controller", 0.5),
                actual_phase="decode",
                phase_iters=1, snapshot=snap,
                batch_num_reqs=0, batch_num_tokens=0,
            )
            tm.record_iter(r)
            tm.backfill_iter_duration(i, 10.0)

        for i in range(2):
            tm.record_controller_tick(_mk_tick(i))
        tm.record_chunk_plan(_mk_chunk(0))
        tm.close()

        def _read(name: str) -> list[dict]:
            with open(os.path.join(d, name)) as f:
                return [json.loads(l) for l in f if l.strip()]

        iters = _read("t_iter.jsonl")
        ctrl = _read("t_ctrl.jsonl")
        chunk = _read("t_chunk.jsonl")
        # Counts.
        assert len(iters) == 3
        assert len(ctrl) == 2
        assert len(chunk) == 1
        # Crosstalk: each file's lines have its discriminating field, and
        # not the others'. (iter has phase, ctrl has ratio_before, chunk
        # has chunk_budget_in.)
        assert all("phase" in r and "ratio_before" not in r for r in iters)
        assert all("ratio_before" in r and "chunk_budget_in" not in r
                   for r in ctrl)
        assert all("chunk_budget_in" in r and "phase" not in r
                   for r in chunk)


# ---------- P1.6b F1+F2 ----------


def test_tracker_returns_rec_on_first_token_only():
    """P1.6b F1: on_token_produced returns the record at first-token so the
    scheduler can push ttft to telemetry without waiting for on_finish.
    Subsequent token calls return None."""
    t = RequestTracker()
    t.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    rec_first = t.on_token_produced("r1", 1, ts_ms=50.0)
    assert rec_first is not None
    assert rec_first.ttft_ms == 50.0
    assert rec_first.tpot_ms_mean is None  # no intervals yet
    rec_second = t.on_token_produced("r1", 1, ts_ms=70.0)
    assert rec_second is None
    rec_third = t.on_token_produced("r1", 1, ts_ms=110.0)
    assert rec_third is None
    # First-token rec is the same record stored in tracker — mutations show.
    assert rec_first.tpot_ms_mean == 30.0


def test_tracker_iter_in_flight_records_yields_live_only():
    """P1.6b F2: iter_in_flight_records yields admitted-but-not-finished
    records so controller can poll fresh in-flight tpot samples each tick."""
    t = RequestTracker()
    t.on_admit("r1", prompt_tokens=10, ts_ms=0.0)
    t.on_admit("r2", prompt_tokens=10, ts_ms=10.0)
    t.on_token_produced("r1", 1, ts_ms=50.0)
    t.on_token_produced("r1", 1, ts_ms=80.0)  # r1 has tpot now
    t.on_token_produced("r2", 1, ts_ms=60.0)
    in_flight = list(t.iter_in_flight_records())
    assert {r.request_id for r in in_flight} == {"r1", "r2"}
    # Finish r1 — only r2 should remain in-flight.
    t.on_finish("r1", ts_ms=100.0)
    in_flight = list(t.iter_in_flight_records())
    assert {r.request_id for r in in_flight} == {"r2"}
