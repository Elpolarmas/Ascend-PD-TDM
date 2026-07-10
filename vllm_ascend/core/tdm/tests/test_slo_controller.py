"""Unit tests for SLOReactiveController (M2).

Uses a minimal FakeTelemetry to avoid the JSONL writer thread / Telemetry
construction cost. Does not require pytest — runs under tests/_runner.py.
"""
from typing import List, Optional

from vllm_ascend.core.tdm.config import TDMConfig
from vllm_ascend.core.tdm.controller import (SLOReactiveController,
                                              StaticRatioController,
                                              make_controller)
from vllm_ascend.core.tdm.types import QueueSnapshot, RequestRecord


class FakeTelemetry:
    def __init__(self):
        self._reqs: List[RequestRecord] = []
        # P1: capture controller ticks if asked. Off by default so existing
        # tests behave as before (no method missing → emit no-op via getattr).
        self.captured_ticks: list = []
        self._capture_ticks: bool = False

    def recent_requests(self, n: int) -> List[RequestRecord]:
        if n >= len(self._reqs):
            return list(self._reqs)
        return self._reqs[-n:]

    def enable_tick_capture(self) -> None:
        self._capture_ticks = True

    def record_controller_tick(self, rec) -> None:
        if self._capture_ticks:
            self.captured_ticks.append(rec)

    def add(
        self,
        rid: str,
        ttft_ms: Optional[float],
        tpot_ms_mean: Optional[float],
    ) -> None:
        rec = RequestRecord(
            request_id=rid,
            prompt_tokens=100,
            admission_ts_ms=0.0,
        )
        if ttft_ms is not None:
            rec.first_token_ts_ms = ttft_ms  # admission_ts=0 → ttft = this
        if tpot_ms_mean is not None:
            # mean of intervals == tpot_ms_mean if all equal
            rec.decode_intervals_ms = [tpot_ms_mean, tpot_ms_mean]
        self._reqs.append(rec)


def _snap(kv_free_ratio: float = 1.0,
          waiting_oldest_age_ms: float = 0.0) -> QueueSnapshot:
    free = int(round(kv_free_ratio * 100))
    return QueueSnapshot(
        waiting_depth=0,
        waiting_oldest_age_ms=waiting_oldest_age_ms,
        finished_prefill_depth=0,
        running_depth=0,
        kv_free_blocks=free,
        kv_total_blocks=100,
    )


def _mk(telem: FakeTelemetry, **overrides) -> SLOReactiveController:
    # NB: kp_q, tpot_saturation, and relu_err all default OFF here so the
    # fixture isolates pure-PID behaviour; later-feature tests opt in. The
    # constructor defaults (production settings) are kp_q=0.5, saturation
    # enabled, relu_err=True (M2.3 / M2.4 / M2.5).
    kw = dict(
        telemetry=telem,
        slo_ttft_ms=500.0,
        slo_tpot_ms=50.0,
        target_violation_rate=0.05,
        kp=0.5,
        ratio_min=0.05,
        ratio_max=0.8,
        update_interval=8,
        initial_ratio=0.3,
        ema_alpha=0.3,
        deadband=0.02,
        min_samples=16,
        window_size=128,
        kv_freeze_watermark=0.05,
        kv_freeze_margin=0.05,
        kp_q=0.0,
        tpot_saturation_enabled=False,
        relu_err=False,
    )
    kw.update(overrides)
    return SLOReactiveController(**kw)


def test_cold_start_returns_initial_ratio():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.42)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.42) < 1e-9, r


def test_throttling_returns_cached_between_updates():
    telem = FakeTelemetry()
    c = _mk(telem, update_interval=8, initial_ratio=0.30)
    # First call sets the cache.
    r0 = c.get_target_ratio(_snap(), iter_id=0)
    # Now flood telemetry with severe ttft violations — but iter_id < interval,
    # so the cache should NOT update.
    for i in range(40):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r1 = c.get_target_ratio(_snap(), iter_id=4)
    r2 = c.get_target_ratio(_snap(), iter_id=7)
    assert r0 == r1 == r2, (r0, r1, r2)


def test_ttft_violation_pushes_ratio_up():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0)
    # 20 finished requests, all blow ttft.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    # First call: cold start path (last_iter_id==-1 enters update branch).
    r = c.get_target_ratio(_snap(), iter_id=0)
    # err_ttft ≈ 1.0 - 0.05 = 0.95, err_tpot ≈ 0 - 0.05 = -0.05
    # delta = 0.5 * (0.95 - (-0.05)) = 0.5
    # ema_alpha=1.0 ⇒ smoothed = 0.30 + 0.5 = 0.80 (clamped to 0.8)
    assert r > 0.30, r
    assert r <= 0.8 + 1e-9, r


def test_tpot_violation_pushes_ratio_down():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.50, kp=0.5, ema_alpha=1.0, deadband=0.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert r < 0.50, r
    assert r >= 0.05 - 1e-9, r


def test_balanced_violations_inside_deadband():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, deadband=0.05)
    # All meet SLO comfortably → both violation rates ~ 0; err_ttft = err_tpot
    # = -target = -0.05, so delta = 0 → deadband applies trivially.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.30) < 1e-9, r


def test_deadband_drops_small_delta():
    telem = FakeTelemetry()
    # Tune so the raw delta is tiny: 1/20 ttft viol = 0.05 == target → err=0;
    # 1/20 tpot viol == target too. Then delta = 0.
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.10)
    for i in range(19):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    telem.add("rX", ttft_ms=2000.0, tpot_ms_mean=2000.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.30) < 1e-9, r


def test_ema_smoothing_partial_step():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=0.3, deadband=0.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    # candidate = 0.30 + 0.5 = 0.80; smoothed = 0.7*0.30 + 0.3*0.80 = 0.45.
    assert abs(r - 0.45) < 1e-6, r


def test_kv_pressure_freezes_update():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kv_freeze_watermark=0.05, kv_freeze_margin=0.05)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    # kv_free_ratio = 0.08 < watermark+margin (0.10) → freeze.
    r = c.get_target_ratio(_snap(kv_free_ratio=0.08), iter_id=0)
    assert abs(r - 0.30) < 1e-9, r
    # When pressure releases, the controller resumes updating.
    r2 = c.get_target_ratio(_snap(kv_free_ratio=0.5), iter_id=8)
    assert r2 > 0.30, r2


def test_ratio_clamped_to_min_max():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=10.0, ema_alpha=1.0, deadband=0.0,
            ratio_min=0.10, ratio_max=0.6)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.6) < 1e-9, r


def test_dedup_seen_ids_not_double_counted():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            min_samples=8)
    for i in range(8):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r1 = c.get_target_ratio(_snap(), iter_id=0)
    # Second tick, no new records — dedup keeps the window stable, but the
    # controller still RE-EVALUATES on the same window. The point: violation
    # rate hasn't changed → ratio should re-step the same way (not double).
    # Easier check: window size stays 8, not 16.
    r2 = c.get_target_ratio(_snap(), iter_id=8)
    # Pull internal state.
    assert len(c._ttft_window) == 8, len(c._ttft_window)
    # Both calls saw violation_rate=1.0 → both compute the same +0.5 delta.
    # First step: 0.30 → 0.80. Second step: 0.80 + 0.5 = 1.3, clamped to 0.8.
    assert abs(r2 - 0.80) < 1e-9, r2
    assert r1 < r2 + 1e-9, (r1, r2)


def test_min_samples_holds_when_cold():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, min_samples=16)
    # Only 5 records → below min_samples on both sides → controller holds.
    for i in range(5):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=2000.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.30) < 1e-9, r


def test_make_controller_static_default():
    telem = FakeTelemetry()
    cfg = TDMConfig()  # controller_kind defaults to "static"
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, StaticRatioController), type(ctrl)


def test_make_controller_slo_pid():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid")
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController), type(ctrl)


def test_make_controller_disable_overrides_kind():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid", disable_controller=True)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, StaticRatioController), type(ctrl)


def test_make_controller_unknown_kind_raises():
    telem = FakeTelemetry()
    cfg = TDMConfig()
    cfg.controller_kind = "bang_bang"  # not validated until enable_tdm path
    try:
        make_controller(cfg, telem)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown controller_kind")


def test_initial_ratio_is_clamped():
    telem = FakeTelemetry()
    # initial_ratio above ratio_max should clamp.
    c = _mk(telem, initial_ratio=0.95, ratio_min=0.1, ratio_max=0.7)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert abs(r - 0.7) < 1e-9, r


# ---- M2.1 starvation guard ----

def test_starvation_guard_activates_after_n_ticks():
    telem = FakeTelemetry()
    # Pin to floor + both SLOs violated. Without the guard, err_ttft and
    # err_tpot cancel out (delta = 0) and ratio sits at floor forever; with
    # the guard, after starvation_min_ticks consecutive at-floor ticks we
    # zero err_tpot and let err_ttft lift ratio off the floor.
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=3)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    r0 = c.get_target_ratio(_snap(), iter_id=0)
    r1 = c.get_target_ratio(_snap(), iter_id=8)
    # First two ticks: streak counts, clamp inactive, delta cancels.
    assert abs(r0 - 0.05) < 1e-9, r0
    assert abs(r1 - 0.05) < 1e-9, r1
    assert c._tpot_clamp_active is False
    # Third tick: streak hits threshold, clamp activates THIS tick, delta
    # becomes kp * err_ttft = 0.5 * 0.95 = 0.475 → ratio = 0.525.
    r2 = c.get_target_ratio(_snap(), iter_id=16)
    assert c._tpot_clamp_active is True
    assert abs(r2 - 0.525) < 1e-6, r2


def test_starvation_guard_releases_once_ratio_escapes_floor():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=3)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    c.get_target_ratio(_snap(), iter_id=0)
    c.get_target_ratio(_snap(), iter_id=8)
    c.get_target_ratio(_snap(), iter_id=16)
    assert c._tpot_clamp_active is True
    # Next tick: last_ratio is 0.525 → at_floor=False → guard releases.
    c.get_target_ratio(_snap(), iter_id=24)
    assert c._tpot_clamp_active is False
    assert c._tpot_clamp_streak == 0


def test_starvation_guard_does_not_activate_when_not_at_floor():
    telem = FakeTelemetry()
    # Start above floor. Even with both SLOs violated, the *first* tick has
    # last_ratio=0.50 → at_floor=False → streak stays 0.
    c = _mk(telem, initial_ratio=0.50, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=3)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_clamp_streak == 0, c._tpot_clamp_streak
    assert c._tpot_clamp_active is False


def test_starvation_guard_does_not_activate_when_tpot_met():
    telem = FakeTelemetry()
    # ttft violated (would drive up), tpot met → guard should never engage.
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=3)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    for tick, it in enumerate([0, 8, 16, 24]):
        c.get_target_ratio(_snap(), iter_id=it)
    assert c._tpot_clamp_active is False
    assert c._tpot_clamp_streak == 0


def test_starvation_guard_streak_resets_when_tpot_recovers():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=3,
            window_size=20)
    # First 2 ticks at floor with tpot violating → streak = 2.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    c.get_target_ratio(_snap(), iter_id=0)
    c.get_target_ratio(_snap(), iter_id=8)
    assert c._tpot_clamp_streak == 2
    # Now flush window with tpot-met records (window_size=20, dedup keeps the
    # original 20 from re-counting). The deque retains its maxlen so older
    # entries roll out as new ones arrive.
    for i in range(20, 40):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    c.get_target_ratio(_snap(), iter_id=16)
    # tpot now meets SLO → tpot_violating=False → streak resets to 0.
    assert c._tpot_clamp_streak == 0
    assert c._tpot_clamp_active is False


def test_starvation_guard_min_ticks_one_activates_immediately():
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.5,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=1)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    # min_ticks=1 means clamp activates on the very first at-floor tick.
    assert c._tpot_clamp_active is True
    # Same scenario as the multi-tick activation test, just one tick early.
    assert abs(r - 0.525) < 1e-6, r


def test_make_controller_passes_starvation_min_ticks():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid", slo_starvation_min_ticks=7)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController)
    assert ctrl._starvation_min_ticks == 7


# ---- M2.2 hysteresis release ----

def test_starvation_guard_hysteresis_holds_within_margin():
    """Guard stays active while ratio rises but is still within
    ratio_min + release_margin; releases once it exceeds."""
    telem = FakeTelemetry()
    # Small kp so ratio rises in small steps; min_ticks=1 so guard arms on
    # the first at-floor tick.
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.1,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=1,
            starvation_release_margin=0.10)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    # Tick 1: at floor, both violating, min_ticks=1 → activate. With guard
    # active, delta = kp * err_ttft = 0.1 * 0.95 = 0.095 → ratio = 0.145.
    r0 = c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_clamp_active is True
    assert abs(r0 - 0.145) < 1e-6, r0
    # Tick 2: last_ratio=0.145 < 0.05+0.10=0.15 → released=False → guard
    # stays. delta=0.095 again → ratio=0.240.
    r1 = c.get_target_ratio(_snap(), iter_id=8)
    assert c._tpot_clamp_active is True, \
        "guard should hold while ratio is still within hysteresis margin"
    assert abs(r1 - 0.240) < 1e-6, r1
    # Tick 3: last_ratio=0.240 > 0.15 → released → guard clears.
    c.get_target_ratio(_snap(), iter_id=16)
    assert c._tpot_clamp_active is False, \
        "guard should release once ratio exceeds hysteresis margin"


def test_starvation_guard_margin_zero_recovers_m21_behaviour():
    """release_margin=0.0 reverts to M2.1: guard releases immediately
    once ratio rises above ratio_min."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.05, ratio_min=0.05, kp=0.1,
            ema_alpha=1.0, deadband=0.0, starvation_min_ticks=1,
            starvation_release_margin=0.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_clamp_active is True
    # Margin=0 → next tick releases as soon as last_ratio > ratio_min.
    c.get_target_ratio(_snap(), iter_id=8)
    assert c._tpot_clamp_active is False


def test_make_controller_passes_release_margin():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid",
                    slo_starvation_release_margin=0.25)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController)
    assert ctrl._starvation_release_margin == 0.25


# ---- M2.3 backlog-aware additive term ----

def test_backlog_pushes_ratio_up_when_queue_aged():
    """Queue head has burned 100% of its TTFT budget → backlog term
    pushes ratio up even when SLO violation rates are 0."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kp_q=0.5, backlog_target=0.5, slo_ttft_ms=500.0)
    # All requests meet SLO → err_ttft=err_tpot=-0.05 → delta_slo=0.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # Queue head waited 500ms = 100% of slo_ttft_ms.
    # backlog_norm = 1.0, err_backlog = 1.0 - 0.5 = 0.5, delta_q = 0.25.
    # ema_alpha=1.0 ⇒ ratio = 0.30 + 0.25 = 0.55.
    r = c.get_target_ratio(_snap(waiting_oldest_age_ms=500.0), iter_id=0)
    assert abs(r - 0.55) < 1e-6, r


def test_backlog_neutral_at_target():
    """waiting_oldest_age = backlog_target * slo_ttft_ms → delta_q=0."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kp_q=0.5, backlog_target=0.5, slo_ttft_ms=500.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # 250ms = 0.5 * slo_ttft_ms → backlog_norm=0.5 → err_backlog=0 → delta_q=0.
    r = c.get_target_ratio(_snap(waiting_oldest_age_ms=250.0), iter_id=0)
    assert abs(r - 0.30) < 1e-6, r


def test_backlog_below_target_pushes_ratio_down():
    """Empty waiting queue → backlog term pushes ratio toward decode."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.50, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kp_q=0.5, backlog_target=0.5, slo_ttft_ms=500.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # age=0 → backlog_norm=0 → err_backlog=-0.5 → delta_q=-0.25.
    # ratio = 0.50 - 0.25 = 0.25.
    r = c.get_target_ratio(_snap(waiting_oldest_age_ms=0.0), iter_id=0)
    assert abs(r - 0.25) < 1e-6, r


def test_backlog_partially_offsets_tpot_pressure():
    """When tpot is heavily violated PID alone would bottom-out ratio;
    backlog term should partially counter that and prevent starvation."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.50, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kp_q=0.5, backlog_target=0.5, slo_ttft_ms=500.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    # delta_slo = 0.5 * ((0 - 0.05) - (1.0 - 0.05)) = 0.5 * -1.0 = -0.5
    # backlog at full SLO budget: delta_q = 0.5 * (1.0 - 0.5) = 0.25
    # delta = -0.5 + 0.25 = -0.25, ratio = 0.50 - 0.25 = 0.25.
    r_with_backlog = c.get_target_ratio(_snap(waiting_oldest_age_ms=500.0),
                                        iter_id=0)
    assert abs(r_with_backlog - 0.25) < 1e-6, r_with_backlog
    # Without backlog (kp_q=0) PID alone would drive ratio to floor.
    c2 = _mk(telem, initial_ratio=0.50, kp=0.5, ema_alpha=1.0, deadband=0.0,
             kp_q=0.0)
    r_pid_only = c2.get_target_ratio(_snap(waiting_oldest_age_ms=500.0),
                                     iter_id=0)
    # delta_slo=-0.5 → ratio=0 → clamped to floor 0.05.
    assert abs(r_pid_only - 0.05) < 1e-6, r_pid_only
    assert r_with_backlog > r_pid_only


def test_kp_q_zero_recovers_pure_pid():
    """kp_q=0 ignores backlog signal regardless of queue state."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            kp_q=0.0, slo_ttft_ms=500.0)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # Even with massive backlog, kp_q=0 → delta_q=0 → PID alone.
    # delta_slo = 0.5 * (-0.05 - -0.05) = 0 → ratio stays 0.30.
    r = c.get_target_ratio(_snap(waiting_oldest_age_ms=10000.0), iter_id=0)
    assert abs(r - 0.30) < 1e-6, r


def test_make_controller_passes_kp_q_and_backlog_target():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid", slo_pid_kp_q=0.7,
                    slo_pid_backlog_target=0.4)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController)
    assert ctrl._kp_q == 0.7
    assert ctrl._backlog_target == 0.4


# ---- M2.4 tpot saturation detector ----

def test_tpot_saturation_activates_after_n_ticks():
    """Persistent tpot violation → saturation flips on after min_ticks."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            tpot_saturation_enabled=True, tpot_saturation_min_ticks=3,
            tpot_saturation_release_ticks=3)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    # Each call sees the same window (dedup), tpot_viol=1.0 every time.
    c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_violate_streak == 1
    assert c._tpot_saturated is False
    c.get_target_ratio(_snap(), iter_id=8)
    c.get_target_ratio(_snap(), iter_id=16)
    assert c._tpot_violate_streak == 3
    assert c._tpot_saturated is True


def test_tpot_saturation_releases_after_recovery_hysteresis():
    """Once saturated, takes release_ticks of recovery to clear."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            tpot_saturation_enabled=True, tpot_saturation_min_ticks=2,
            tpot_saturation_release_ticks=2, window_size=20)
    # Phase 1: violations push state into saturated.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    c.get_target_ratio(_snap(), iter_id=0)
    c.get_target_ratio(_snap(), iter_id=8)
    assert c._tpot_saturated is True
    # Phase 2: feed tpot-met records; window rolls so old violations clear.
    for i in range(20, 40):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # First recovery tick: streak=1, still saturated.
    c.get_target_ratio(_snap(), iter_id=16)
    assert c._tpot_recovery_streak == 1
    assert c._tpot_saturated is True
    # Second recovery tick: streak=2, hits release threshold.
    c.get_target_ratio(_snap(), iter_id=24)
    assert c._tpot_saturated is False
    assert c._tpot_violate_streak == 0


def test_tpot_saturation_disabled_bypasses_state_machine():
    """tpot_saturation_enabled=False → state never flips."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            tpot_saturation_enabled=False, tpot_saturation_min_ticks=1)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    for it in (0, 8, 16, 24):
        c.get_target_ratio(_snap(), iter_id=it)
    assert c._tpot_saturated is False
    assert c._tpot_violate_streak == 0


def test_tpot_saturation_lets_ttft_drive_ratio():
    """Saturated state zeroes err_tpot so ttft alone can lift ratio."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            tpot_saturation_enabled=True, tpot_saturation_min_ticks=1)
    # Both violated. Without saturation: delta = 0.5*(0.95-0.95) = 0.
    # With saturation (active on first tick): delta = 0.5 * 0.95 = 0.475.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=200.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_saturated is True
    # 0.30 + 0.475 = 0.775 (clamped under ratio_max=0.8).
    assert abs(r - 0.775) < 1e-6, r


def test_tpot_saturation_streak_holds_when_window_cold():
    """Cold tpot window (n_tpot < min_samples) shouldn't move the
    saturation streak — otherwise short bursts at start would flip state."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, min_samples=16,
            tpot_saturation_enabled=True, tpot_saturation_min_ticks=2)
    # Only 5 records — below min_samples → cold window → no streak change.
    for i in range(5):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    for it in (0, 8, 16):
        c.get_target_ratio(_snap(), iter_id=it)
    assert c._tpot_violate_streak == 0
    assert c._tpot_saturated is False


def test_make_controller_passes_saturation_fields():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid",
                    slo_tpot_saturation_enabled=False,
                    slo_tpot_saturation_min_ticks=8,
                    slo_tpot_saturation_release_ticks=4)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController)
    assert ctrl._tpot_saturation_enabled is False
    assert ctrl._tpot_saturation_min_ticks == 8
    assert ctrl._tpot_saturation_release_ticks == 4


# ---- M2.5 ReLU clipping on PID error ----

def test_relu_clips_negative_err_ttft():
    """With ReLU on, a comfortably-met ttft (viol < target) shouldn't push
    ratio toward decode — err_ttft is clamped to 0."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            relu_err=True)
    # ttft far below SLO (viol=0), tpot violated (viol=1.0).
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    # err_ttft raw=-0.05 → ReLU=0; err_tpot=0.95.
    # delta = 0.5 * (0 - 0.95) = -0.475. ratio = 0.30 - 0.475 = clamped 0.05.
    assert abs(r - 0.05) < 1e-9, r


def test_relu_clips_negative_err_tpot():
    """With ReLU on, a comfortably-met tpot shouldn't drag ratio up into
    prefill bias — err_tpot is clamped to 0."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            relu_err=True)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    # err_ttft = 0.95; err_tpot raw=-0.05 → ReLU=0.
    # delta = 0.5 * (0.95 - 0) = 0.475. ratio = 0.30 + 0.475 = 0.775.
    assert abs(r - 0.775) < 1e-6, r


def test_relu_disabled_recovers_signed_err():
    """relu_err=False keeps the original signed err arithmetic."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            relu_err=False)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    r = c.get_target_ratio(_snap(), iter_id=0)
    # delta = 0.5 * (0.95 - (-0.05)) = 0.5. ratio = 0.30 + 0.5 = 0.80.
    assert abs(r - 0.80) < 1e-9, r


def test_relu_both_met_holds_initial_ratio():
    """Both SLOs met → both clipped to 0 → delta=0 → ratio doesn't drift.
    This is the key property M2.5 fixes; M2..M2.4 would silently decay
    ratio toward the floor in this state."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            relu_err=True)
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=10.0)
    # Several update ticks — ratio should never drift.
    for it in (0, 8, 16, 24, 32):
        r = c.get_target_ratio(_snap(), iter_id=it)
        assert abs(r - 0.30) < 1e-9, (it, r)


def test_relu_with_saturation_holds_initial_ratio_when_tpot_unreachable():
    """End-to-end: saturation zeroes err_tpot once tpot is hardware-bound;
    ReLU zeroes err_ttft when ttft has headroom; together → controller
    holds its initial ratio (graceful degradation to static)."""
    telem = FakeTelemetry()
    c = _mk(telem, initial_ratio=0.30, kp=0.5, ema_alpha=1.0, deadband=0.0,
            tpot_saturation_enabled=True, tpot_saturation_min_ticks=2,
            relu_err=True)
    # ttft has plenty of headroom, tpot persistently violates (hardware bound).
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=10.0, tpot_ms_mean=200.0)
    # Tick 1: not yet saturated. err_ttft=0 (ReLU clipped), err_tpot=0.95.
    # delta = -0.475 → ratio descends.
    c.get_target_ratio(_snap(), iter_id=0)
    # Tick 2: saturation flips on (min_ticks=2). err_tpot zeroed.
    # err_ttft=0 (still met). delta=0. ratio holds at whatever it landed on.
    r1 = c.get_target_ratio(_snap(), iter_id=8)
    # Tick 3+ : saturation stays on, both err = 0, ratio stays put.
    for it in (16, 24, 32):
        r_n = c.get_target_ratio(_snap(), iter_id=it)
        assert abs(r_n - r1) < 1e-9, (it, r1, r_n)
    assert c._tpot_saturated is True


def test_make_controller_passes_relu_err():
    telem = FakeTelemetry()
    cfg = TDMConfig(controller_kind="slo_pid", slo_pid_relu_err=False)
    ctrl = make_controller(cfg, telem)
    assert isinstance(ctrl, SLOReactiveController)
    assert ctrl._relu_err is False


# ---------- P1 telemetry: ControllerTickRecord emission ----------


def test_emits_tick_only_on_update_branch():
    """Cache-hit calls (between update_intervals) MUST NOT emit a tick — the
    JSONL would otherwise drown in N×update_interval duplicate rows. Only
    the path that actually recomputes ratio gets a record."""
    telem = FakeTelemetry()
    telem.enable_tick_capture()
    c = _mk(telem, update_interval=8, initial_ratio=0.30)

    c.get_target_ratio(_snap(), iter_id=0)        # update branch (cold start)
    c.get_target_ratio(_snap(), iter_id=2)        # cached
    c.get_target_ratio(_snap(), iter_id=7)        # cached
    c.get_target_ratio(_snap(), iter_id=8)        # update branch
    c.get_target_ratio(_snap(), iter_id=15)       # cached
    c.get_target_ratio(_snap(), iter_id=16)       # update branch

    iter_ids = [t.iter_id for t in telem.captured_ticks]
    assert iter_ids == [0, 8, 16], iter_ids


def test_tick_captures_branch_flags():
    """The four early-exit branches each set the right flag in the tick:
    cold_start (no samples yet), kv_freeze_triggered (KV near watermark),
    and the gradient-step branch (neither flag, real err values)."""
    telem = FakeTelemetry()
    telem.enable_tick_capture()
    c = _mk(telem, update_interval=8, initial_ratio=0.30, kp=0.5,
            ema_alpha=1.0, deadband=0.0)

    # Tick 0: cold start (no requests yet).
    c.get_target_ratio(_snap(), iter_id=0)
    cold = telem.captured_ticks[-1]
    assert cold.cold_start is True
    assert cold.kv_freeze_triggered is False
    assert cold.err_ttft == 0.0 and cold.err_tpot_effective == 0.0

    # Tick 8: feed lots of ttft violations, real gradient step.
    for i in range(20):
        telem.add(f"r{i}", ttft_ms=2000.0, tpot_ms_mean=10.0)
    c.get_target_ratio(_snap(), iter_id=8)
    grad = telem.captured_ticks[-1]
    assert grad.cold_start is False
    assert grad.kv_freeze_triggered is False
    assert grad.ttft_viol_rate > 0.5
    assert grad.ratio_after > grad.ratio_before  # ttft pressure pushed up

    # Tick 16: KV near watermark — freeze branch, no learning.
    snap_kv = _snap(kv_free_ratio=0.07)  # below 0.05+0.05=0.10 threshold
    c.get_target_ratio(snap_kv, iter_id=16)
    frozen = telem.captured_ticks[-1]
    assert frozen.kv_freeze_triggered is True
    assert frozen.cold_start is False
    assert frozen.ratio_after == frozen.ratio_before


# ---------- P1.6b F1+F2 controller-side ----------


class FakeTracker:
    """Minimal RequestTracker double for in-flight tpot polling tests."""

    def __init__(self):
        self._records: list = []

    def add_in_flight(self, rid: str, tpot_ms_mean: float) -> None:
        rec = RequestRecord(
            request_id=rid, prompt_tokens=100, admission_ts_ms=0.0)
        rec.first_token_ts_ms = 10.0
        if tpot_ms_mean is not None:
            rec.decode_intervals_ms = [tpot_ms_mean, tpot_ms_mean]
        # finish_ts_ms stays None — req is still in-flight.
        self._records.append(rec)

    def iter_in_flight_records(self):
        yield from self._records


def test_p16b_f1_ttft_seen_dedup_no_double_count():
    """F1: scheduler pushes the same RequestRecord twice (first-token then
    finish). _ttft_seen_ids dedups so ttft contributes once per request."""
    telem = FakeTelemetry()
    c = _mk(telem, min_samples=1)
    # Push the same record twice (F1 push at first-token + finish push)
    telem.add("r1", ttft_ms=800.0, tpot_ms_mean=None)
    telem.add("r1", ttft_ms=800.0, tpot_ms_mean=60.0)
    c.get_target_ratio(_snap(), iter_id=0)  # warm up the controller
    # ttft window should contain exactly ONE sample for r1.
    assert len(c._ttft_window) == 1
    assert c._ttft_window[0] == 800.0


def test_p16b_f2_in_flight_tpot_visible_without_finish():
    """F2: with a tracker attached, in-flight tpot samples reach the
    controller without requiring on_finish. _tpot_window is populated
    even though no record has finish_ts_ms set."""
    telem = FakeTelemetry()  # empty telemetry — no finished requests
    tracker = FakeTracker()
    tracker.add_in_flight("r1", tpot_ms_mean=120.0)
    tracker.add_in_flight("r2", tpot_ms_mean=180.0)
    c = _mk(telem, tracker=tracker, min_samples=1)
    c.get_target_ratio(_snap(), iter_id=0)
    # _tpot_window rebuilt from in-flight — both samples present.
    assert sorted(c._tpot_window) == [120.0, 180.0]


def test_p16b_f2_tpot_window_rebuilds_each_tick():
    """F2: _tpot_window is rebuilt each tick — stale in-flight samples
    drop out the moment requests finish."""
    telem = FakeTelemetry()
    tracker = FakeTracker()
    tracker.add_in_flight("r1", tpot_ms_mean=100.0)
    c = _mk(telem, tracker=tracker, update_interval=1, min_samples=1)
    c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_window == [100.0]
    # Simulate r1 finishing — drops out of in-flight, doesn't enter telemetry.
    tracker._records.clear()
    c.get_target_ratio(_snap(), iter_id=1)
    assert c._tpot_window == []


def test_p16b_f2_no_tracker_falls_back_to_telemetry():
    """tracker=None preserves legacy behaviour: tpot comes from completed
    requests in telemetry, not in-flight polling."""
    telem = FakeTelemetry()
    telem.add("r1", ttft_ms=200.0, tpot_ms_mean=80.0)
    telem._reqs[-1].finish_ts_ms = 1000.0  # mark as finished
    c = _mk(telem, tracker=None, min_samples=1)
    c.get_target_ratio(_snap(), iter_id=0)
    assert c._tpot_window == [80.0]


def test_p16b_f2_avoids_double_count_in_flight_and_telemetry():
    """If a request appears both in tracker (in-flight) and telemetry
    (pushed at first-token without finish_ts_ms), the controller must not
    count it twice. Rule: when tracker available, telemetry contributes
    only finished requests (finish_ts_ms set)."""
    telem = FakeTelemetry()
    # Same req r1 pushed at first-token (no finish_ts_ms set).
    telem.add("r1", ttft_ms=200.0, tpot_ms_mean=90.0)
    tracker = FakeTracker()
    tracker.add_in_flight("r1", tpot_ms_mean=90.0)  # same value
    c = _mk(telem, tracker=tracker, min_samples=1)
    c.get_target_ratio(_snap(), iter_id=0)
    # Should see r1 exactly ONCE via tracker, NOT twice.
    assert c._tpot_window == [90.0]
