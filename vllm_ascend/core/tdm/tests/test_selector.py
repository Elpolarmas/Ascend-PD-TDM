from vllm_ascend.core.tdm.selector import TokenBucketSelector
from vllm_ascend.core.tdm.types import QueueSnapshot


def _snap(waiting: int, running: int, finished_p: int = 0,
          kv_total: int = 100, kv_free: int = 100,
          oldest_age_ms: float = 0.0,
          decode_silence_ms: float = 0.0) -> QueueSnapshot:
    return QueueSnapshot(
        waiting_depth=waiting,
        waiting_oldest_age_ms=oldest_age_ms,
        finished_prefill_depth=finished_p,
        running_depth=running,
        kv_free_blocks=kv_free,
        kv_total_blocks=kv_total,
        decode_oldest_silence_ms=decode_silence_ms,
    )


def test_peek_decode_when_no_prefill_work():
    s = TokenBucketSelector(cap=4)
    assert s.peek(1.0, _snap(waiting=0, running=4)) == "decode"


def test_peek_prefill_when_no_decode_work():
    s = TokenBucketSelector(cap=4)
    # No decode work → must prefill regardless of bucket.
    assert s.peek(0.0, _snap(waiting=2, running=0)) == "prefill"


def test_peek_respects_target_ratio_over_window():
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=8, running=8)
    # ratio=0.25 → expect ~1 prefill per 4 iters.
    prefills = 0
    for _ in range(40):
        ph = s.peek(0.25, snap)
        s.commit(ph)
        if ph == "prefill":
            prefills += 1
    # Allow ±2 from theoretical 10.
    assert 8 <= prefills <= 12, prefills


def test_commit_only_consumes_on_actual_prefill():
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=8, running=8)
    s.peek(1.0, snap)            # bucket: 0 + 1 = 1
    assert s.tokens >= 1.0
    s.commit("decode")           # parent flipped — must NOT consume
    assert s.tokens >= 1.0
    s.commit("prefill")
    assert s.tokens < 1.0


def test_bucket_capped():
    s = TokenBucketSelector(cap=2)
    snap = _snap(waiting=0, running=4)   # decode-only → never consume
    for _ in range(20):
        s.peek(1.0, snap)
    assert s.tokens <= 2.0


# ----- P1.7 urgency override -----

def test_urgency_disabled_when_threshold_zero():
    s = TokenBucketSelector(cap=4)
    # Bucket empty, contention, oldest age huge — but threshold=0 disables.
    snap = _snap(waiting=4, running=4, oldest_age_ms=10_000.0)
    assert s.peek(0.0, snap, urgency_threshold_ms=0.0) == "decode"
    assert not s.last_override_urgent


def test_urgency_below_threshold_no_override():
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=4, oldest_age_ms=300.0)
    # 300ms < 400ms threshold → normal bucket path (empty → decode)
    assert s.peek(0.0, snap, urgency_threshold_ms=400.0) == "decode"
    assert not s.last_override_urgent


def test_urgency_above_threshold_forces_prefill():
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=4, oldest_age_ms=450.0)
    # Bucket empty (ratio=0) and would decode, but urgency rescues.
    assert s.peek(0.0, snap, urgency_threshold_ms=400.0) == "prefill"
    assert s.last_override_urgent


def test_urgency_charges_bucket_to_zero_default():
    s = TokenBucketSelector(cap=4, initial=0.5)
    snap = _snap(waiting=4, running=4, oldest_age_ms=500.0)
    # tokens=0.5+0=0.5 (ratio=0) < 1, urgency fires, default charges → 0.0
    s.peek(0.0, snap, urgency_threshold_ms=400.0)
    assert s.tokens == 0.0
    assert s.last_override_urgent


def test_urgency_no_charge_preserves_tokens():
    s = TokenBucketSelector(cap=4, initial=0.5)
    snap = _snap(waiting=4, running=4, oldest_age_ms=500.0)
    s.peek(0.0, snap, urgency_threshold_ms=400.0,
           charge_bucket_on_urgency=False)
    assert s.tokens == 0.5
    assert s.last_override_urgent


def test_urgency_skipped_when_bucket_already_full():
    # If bucket >= 1 we go prefill anyway via the normal path; urgency
    # tag should NOT fire (no rescue needed).
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, oldest_age_ms=500.0)
    assert s.peek(0.0, snap, urgency_threshold_ms=400.0) == "prefill"
    assert not s.last_override_urgent


def test_urgency_skipped_when_no_decode_pressure():
    # waiting>0, running=0, finished_p=0 → forced prefill regardless.
    # urgency tag must not fire (we'd be prefilling anyway).
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=0, oldest_age_ms=500.0)
    assert s.peek(0.0, snap, urgency_threshold_ms=400.0) == "prefill"
    assert not s.last_override_urgent


def test_urgency_flag_resets_on_next_peek():
    s = TokenBucketSelector(cap=4)
    snap_urgent = _snap(waiting=4, running=4, oldest_age_ms=500.0)
    snap_quiet = _snap(waiting=4, running=4, oldest_age_ms=100.0)
    s.peek(0.0, snap_urgent, urgency_threshold_ms=400.0)
    assert s.last_override_urgent
    s.peek(0.0, snap_quiet, urgency_threshold_ms=400.0)
    assert not s.last_override_urgent


# ----- P1.7b TPOT starvation override -----

def test_starvation_disabled_when_threshold_zero():
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, decode_silence_ms=1000.0)
    # threshold=0 → disabled; bucket has tokens → prefill
    assert s.peek(0.0, snap, starvation_threshold_ms=0.0) == "prefill"
    assert not s.last_override_starvation


def test_starvation_below_threshold_no_override():
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, decode_silence_ms=100.0)
    # silence < threshold → no override, bucket goes prefill
    assert s.peek(0.0, snap, starvation_threshold_ms=150.0) == "prefill"
    assert not s.last_override_starvation


def test_starvation_above_threshold_forces_decode():
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    # Bucket full (would prefill), but decode head silent >= threshold → rescue.
    assert s.peek(0.0, snap, starvation_threshold_ms=150.0) == "decode"
    assert s.last_override_starvation


def test_starvation_preserves_bucket_tokens():
    """Critical: starvation must NOT debit bucket. Slow loop's prefill
    credit is preserved across the rescue (anchor invariance)."""
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    s.peek(0.0, snap, starvation_threshold_ms=150.0)
    assert s.tokens == 2.0  # unchanged
    assert s.last_override_starvation


def test_starvation_skipped_when_bucket_wants_decode():
    """When bucket already wants decode (tokens < 1), starvation should
    not stamp the flag — the rescue is unneeded."""
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    assert s.peek(0.0, snap, starvation_threshold_ms=150.0) == "decode"
    assert not s.last_override_starvation


def test_starvation_skipped_when_no_prefill_pressure():
    # waiting=0 → forced decode anyway; starvation flag must not fire.
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=0, running=4, decode_silence_ms=200.0)
    assert s.peek(1.0, snap, starvation_threshold_ms=150.0) == "decode"
    assert not s.last_override_starvation


def test_starvation_and_urgency_mutually_exclusive_by_tokens():
    """Bucket state forces at-most-one override per iter:
    - tokens < 1 → only urgency_ttft is eligible
    - tokens >= 1 → only starvation_tpot is eligible
    Even if both threshold conditions are met simultaneously, the bucket
    state picks exactly one path."""
    s_low = TokenBucketSelector(cap=4)
    snap_both = _snap(waiting=4, running=4,
                      oldest_age_ms=500.0, decode_silence_ms=200.0)
    # tokens=0 < 1 → urgency wins
    assert s_low.peek(0.0, snap_both,
                      urgency_threshold_ms=400.0,
                      starvation_threshold_ms=150.0) == "prefill"
    assert s_low.last_override_urgent
    assert not s_low.last_override_starvation

    s_hi = TokenBucketSelector(cap=4, initial=2.0)
    # tokens=2 >= 1 → starvation wins (also priority since starvation is
    # checked first in peek, but the bucket-state gate is what enforces
    # exclusivity)
    assert s_hi.peek(0.0, snap_both,
                     urgency_threshold_ms=400.0,
                     starvation_threshold_ms=150.0) == "decode"
    assert s_hi.last_override_starvation
    assert not s_hi.last_override_urgent


def test_starvation_flag_resets_on_next_peek():
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap_hi = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    snap_lo = _snap(waiting=4, running=4, decode_silence_ms=50.0)
    s.peek(0.0, snap_hi, starvation_threshold_ms=150.0)
    assert s.last_override_starvation
    s.peek(0.0, snap_lo, starvation_threshold_ms=150.0)
    assert not s.last_override_starvation


# ----- P1.7b decouple flag (2026-05-17 diagnostic) -----

def test_starvation_decouple_fires_when_bucket_wants_decode():
    """With decouple=True, starvation fires on silence alone — even when
    tokens < 1 (bucket already wants decode). In the default AND-gate path
    this same call would silently fall through to bucket-decoded without
    stamping the flag (see test_starvation_skipped_when_bucket_wants_decode)."""
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    assert s.peek(0.0, snap, starvation_threshold_ms=150.0,
                  starvation_decouple_bucket=True) == "decode"
    assert s.last_override_starvation


def test_starvation_decouple_still_preserves_bucket():
    """Decouple drops the AND-gate but must still leave bucket untouched —
    slow loop's prefill credit invariance is the whole point of the rescue
    not charging the bucket."""
    s = TokenBucketSelector(cap=4, initial=0.4)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    s.peek(0.3, snap, starvation_threshold_ms=150.0,
           starvation_decouple_bucket=True)
    # Credit added (0.4 + 0.3 = 0.7); rescue does not consume.
    assert abs(s.tokens - 0.7) < 1e-9
    assert s.last_override_starvation


def test_starvation_decouple_still_skipped_below_threshold():
    """Decouple is a gate widener, not a threshold bypass. Silence below
    threshold must still not fire."""
    s = TokenBucketSelector(cap=4, initial=2.0)
    snap = _snap(waiting=4, running=4, decode_silence_ms=100.0)
    s.peek(0.0, snap, starvation_threshold_ms=150.0,
           starvation_decouple_bucket=True)
    assert not s.last_override_starvation


def test_starvation_decouple_default_false_matches_legacy():
    """Default arg keeps M3.3 back-compat — tokens<1 case still skipped."""
    s = TokenBucketSelector(cap=4)
    snap = _snap(waiting=4, running=4, decode_silence_ms=200.0)
    s.peek(0.0, snap, starvation_threshold_ms=150.0)  # no kwarg
    assert not s.last_override_starvation
