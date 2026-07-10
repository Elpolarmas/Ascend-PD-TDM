import time
from collections import deque
from typing import Optional

from .telemetry import Telemetry
from .types import ControllerTickRecord, QueueSnapshot


class StaticRatioController:
    """V1 starter controller: returns a fixed target_ratio every tick.

    Recomputes only every `update_interval` iters to mirror the V2 closed-loop
    cadence (so the selector and downstream code see the same call pattern).
    SLO-reactive logic (and later AIMD) plug in by replacing this class.
    """

    def __init__(
        self,
        static_ratio: float = 0.5,
        update_interval: int = 8,
        telemetry: Optional[Telemetry] = None,
    ):
        self._ratio = max(0.0, min(1.0, static_ratio))
        self._update_interval = max(1, update_interval)
        self._telemetry = telemetry
        self._cached: float = self._ratio
        self._last_iter_id: int = -1

    def get_target_ratio(self, snap: QueueSnapshot, iter_id: int) -> float:
        if (self._last_iter_id < 0
                or iter_id - self._last_iter_id >= self._update_interval):
            self._cached = self._ratio
            self._last_iter_id = iter_id
        return self._cached


class SLOReactiveController:
    """M2: PID-lite controller on `target_ratio` driven by SLO violation rates.

    Reads completed-request stats from Telemetry once per `update_interval`
    iters, computes ttft / tpot violation rates against `slo_*_ms`, and steers
    `target_ratio` toward the side that is more violated:

        delta = kp * (err_ttft - err_tpot)
        smoothed = (1 - ema_alpha) * last + ema_alpha * (last + delta)

    Behaviour notes:
    - Cold start (window samples below `min_samples`) returns `initial_ratio`.
    - KV pressure freeze: when `kv_free_ratio` is within `kv_freeze_margin`
      of the BoundaryGuard watermark, the controller does NOT update — both
      directions of the gradient are biased while BoundaryGuard is overriding
      decisions.
    - Deadband: deltas below `deadband` are dropped so we don't chatter
      around the equilibrium.
    - Starvation guard (M2.1): when ratio is pinned to ratio_min for
      `starvation_min_ticks` consecutive update ticks while tpot still
      violates SLO, treat tpot as unreachable and zero out `err_tpot` so
      only the ttft side can drive ratio.
    - Hysteresis release (M2.2): guard releases only when ratio rises by
      at least `starvation_release_margin` above ratio_min — Schmitt-trigger
      style separation between activate and release thresholds prevents the
      fast oscillation seen with margin=0 (PID immediately pulls ratio back
      to floor each time guard releases).
    - Backlog-aware additive term (M2.3): adds
      `kp_q * (waiting_oldest_age_ms / slo_ttft_ms - backlog_target)` to
      delta. Acts as a leading indicator on prefill pressure — pushes ratio
      up when the head of the waiting queue has burned through more than
      `backlog_target` of its TTFT budget, even before SLO violations land
      in the telemetry window. Set kp_q=0 to disable (recovers M2 behaviour).
    - tpot saturation detector (M2.4): when tpot_violation_rate stays
      above target_violation_rate for `tpot_saturation_min_ticks`
      consecutive update ticks, the controller declares tpot SLO
      physically unreachable and zeroes out err_tpot — the signal would
      otherwise drag ratio uselessly toward the floor since no amount of
      decode bias can hit a hardware-bound tpot ceiling. Released after
      `tpot_saturation_release_ticks` consecutive ticks of tpot back
      under target (hysteresis prevents flapping near the boundary).
    - ReLU err clipping (M2.5): err_ttft / err_tpot are clipped to
      max(0, .) so a comfortably-met SLO never pushes ratio in the wrong
      direction. With raw signed err, an SLO with plenty of headroom
      generates err = -target (negative) and steadily decays ratio toward
      the floor even when no help is needed. ReLU makes the controller a
      one-sided pressure response: only actual violations move ratio.
    """

    def __init__(
        self,
        *,
        telemetry: Telemetry,
        tracker=None,  # P1.6b: RequestTracker for in-flight tpot polling
        slo_ttft_ms: float,
        slo_tpot_ms: float,
        target_violation_rate: float = 0.05,
        kp: float = 0.5,
        ratio_min: float = 0.05,
        ratio_max: float = 0.8,
        update_interval: int = 8,
        initial_ratio: float = 0.3,
        ema_alpha: float = 0.3,
        deadband: float = 0.02,
        min_samples: int = 16,
        window_size: int = 128,
        seen_ids_cap: int = 512,
        kv_freeze_watermark: float = 0.05,
        kv_freeze_margin: float = 0.05,
        starvation_min_ticks: int = 3,
        starvation_release_margin: float = 0.10,
        kp_q: float = 0.5,
        backlog_target: float = 0.5,
        tpot_saturation_enabled: bool = True,
        tpot_saturation_min_ticks: int = 5,
        tpot_saturation_release_ticks: int = 5,
        relu_err: bool = True,
    ):
        if not 0.0 <= ratio_min <= ratio_max <= 1.0:
            raise ValueError(
                f"need 0 <= ratio_min <= ratio_max <= 1, "
                f"got {ratio_min}/{ratio_max}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._telemetry = telemetry
        self._tracker = tracker  # P1.6b: None falls back to telemetry-only path
        self._slo_ttft_ms = slo_ttft_ms
        self._slo_tpot_ms = slo_tpot_ms
        self._target_violation_rate = target_violation_rate
        self._kp = kp
        self._ratio_min = ratio_min
        self._ratio_max = ratio_max
        self._update_interval = max(1, update_interval)
        self._ema_alpha = ema_alpha
        self._deadband = deadband
        self._min_samples = max(1, min_samples)
        self._window_size = max(1, window_size)
        self._seen_ids_cap = max(self._window_size, seen_ids_cap)
        self._kv_freeze_watermark = kv_freeze_watermark
        self._kv_freeze_margin = kv_freeze_margin
        self._starvation_min_ticks = max(1, starvation_min_ticks)
        self._starvation_release_margin = max(0.0, starvation_release_margin)
        self._tpot_clamp_streak: int = 0
        self._tpot_clamp_active: bool = False
        self._kp_q = max(0.0, kp_q)
        self._backlog_target = backlog_target
        self._tpot_saturation_enabled = tpot_saturation_enabled
        self._tpot_saturation_min_ticks = max(1, tpot_saturation_min_ticks)
        self._tpot_saturation_release_ticks = max(
            1, tpot_saturation_release_ticks)
        self._tpot_saturated: bool = False
        self._tpot_violate_streak: int = 0
        self._tpot_recovery_streak: int = 0
        self._relu_err = relu_err

        clamped = max(ratio_min, min(ratio_max, initial_ratio))
        self._last_ratio: float = clamped
        self._cached: float = clamped
        self._last_iter_id: int = -1

        self._ttft_window: deque[float] = deque(maxlen=self._window_size)
        # P1.6b F2: _tpot_window is rebuilt every tick from tracker
        # (in-flight) + telemetry (recently finished) so PID sees fresh tpot
        # instead of a deque of stale finished-only means.
        self._tpot_window: list[float] = []
        # P1.6b F1: separate seen-id sets — ttft is one-shot per req
        # (push at first-token), tpot is rebuilt so doesn't need dedup.
        self._ttft_seen_ids: set[str] = set()

    # ---- public ----

    def get_target_ratio(self, snap: QueueSnapshot, iter_id: int) -> float:
        if (self._last_iter_id >= 0
                and iter_id - self._last_iter_id < self._update_interval):
            return self._cached

        self._last_iter_id = iter_id
        ratio_before = self._last_ratio

        # KV-pressure freeze: BoundaryGuard is or will be overriding decisions,
        # so the SLO signal is biased — hold the last ratio rather than learn
        # the wrong gradient.
        kv_threshold = self._kv_freeze_watermark + self._kv_freeze_margin
        if snap.kv_free_ratio < kv_threshold:
            self._cached = self._last_ratio
            self._emit_tick(
                iter_id=iter_id,
                ratio_before=ratio_before,
                ratio_after=self._last_ratio,
                delta_slo=0.0,
                delta_q=0.0,
                deadband_dropped=False,
                err_ttft_raw=0.0,
                err_tpot_raw=0.0,
                err_ttft=0.0,
                err_tpot_effective=0.0,
                ttft_viol_rate=0.0,
                tpot_viol_rate=0.0,
                n_ttft_window=len(self._ttft_window),
                n_tpot_window=len(self._tpot_window),
                kv_freeze_triggered=True,
                cold_start=False,
            )
            return self._cached

        self._pull_recent_violations()

        n_ttft = len(self._ttft_window)
        n_tpot = len(self._tpot_window)
        # Need at least one side fully warm to make a real decision; if both
        # are cold we just hold last_ratio.
        if n_ttft < self._min_samples and n_tpot < self._min_samples:
            self._cached = self._last_ratio
            self._emit_tick(
                iter_id=iter_id,
                ratio_before=ratio_before,
                ratio_after=self._last_ratio,
                delta_slo=0.0,
                delta_q=0.0,
                deadband_dropped=False,
                err_ttft_raw=0.0,
                err_tpot_raw=0.0,
                err_ttft=0.0,
                err_tpot_effective=0.0,
                ttft_viol_rate=0.0,
                tpot_viol_rate=0.0,
                n_ttft_window=n_ttft,
                n_tpot_window=n_tpot,
                kv_freeze_triggered=False,
                cold_start=True,
            )
            return self._cached

        ttft_viol = self._violation_rate(self._ttft_window, self._slo_ttft_ms)
        tpot_viol = self._violation_rate(self._tpot_window, self._slo_tpot_ms)

        # Drop the side that is still cold so it doesn't push a noisy 0.
        err_ttft_raw = (ttft_viol - self._target_violation_rate
                        if n_ttft >= self._min_samples else 0.0)
        err_tpot_raw = (tpot_viol - self._target_violation_rate
                        if n_tpot >= self._min_samples else 0.0)
        err_ttft = err_ttft_raw
        err_tpot = err_tpot_raw
        # ReLU: a met SLO (viol < target) shouldn't drive ratio in the
        # opposite direction — that's how M2..M2.4 ended up parking ratio
        # at the floor when ttft had plenty of headroom.
        if self._relu_err:
            err_ttft = max(0.0, err_ttft)
            err_tpot = max(0.0, err_tpot)

        # tpot saturation detector: identify when tpot SLO is physically
        # unreachable so we stop letting err_tpot drag ratio toward the
        # floor in vain. Streaks are only updated when the tpot side is
        # warm (n_tpot >= min_samples) so cold-window noise can't flip the
        # state machine.
        if self._tpot_saturation_enabled and n_tpot >= self._min_samples:
            if tpot_viol > self._target_violation_rate:
                self._tpot_violate_streak += 1
                self._tpot_recovery_streak = 0
                if (not self._tpot_saturated and self._tpot_violate_streak
                        >= self._tpot_saturation_min_ticks):
                    self._tpot_saturated = True
            else:
                self._tpot_recovery_streak += 1
                self._tpot_violate_streak = 0
                if (self._tpot_saturated and self._tpot_recovery_streak
                        >= self._tpot_saturation_release_ticks):
                    self._tpot_saturated = False

        # Starvation guard: if ratio sits at ratio_min while tpot still
        # violates, the tpot SLO is hardware-bound — stop fighting the floor
        # so err_ttft can lift ratio back up. Hysteresis on release: ratio
        # must rise by at least starvation_release_margin above ratio_min
        # before the guard lets go, otherwise PID's tpot pull immediately
        # snaps ratio back to floor and we oscillate.
        at_floor = self._last_ratio <= self._ratio_min + 1e-6
        released = (self._last_ratio
                    > self._ratio_min + self._starvation_release_margin)
        tpot_violating = (n_tpot >= self._min_samples
                          and tpot_viol > self._target_violation_rate)
        if self._tpot_clamp_active:
            if released:
                self._tpot_clamp_active = False
                self._tpot_clamp_streak = 0
        elif at_floor and tpot_violating:
            self._tpot_clamp_streak += 1
            if self._tpot_clamp_streak >= self._starvation_min_ticks:
                self._tpot_clamp_active = True
        else:
            self._tpot_clamp_streak = 0

        # Two independent reasons to ignore err_tpot: starvation guard (M2.1)
        # triggers on ratio-floor pinning, saturation detector (M2.4) triggers
        # on physical unreachability. Either one is enough.
        if self._tpot_clamp_active or self._tpot_saturated:
            effective_err_tpot = 0.0
        else:
            effective_err_tpot = err_tpot
        delta_slo = self._kp * (err_ttft - effective_err_tpot)

        # Backlog term: leading indicator on prefill pressure. Uses the
        # oldest waiting request's age, normalised by slo_ttft_ms — a value
        # of 1.0 means "queue head has used up its entire TTFT budget".
        # backlog_target offsets the neutral point so a half-burned budget
        # does nothing (above it pushes ratio up, below pushes ratio down).
        backlog_norm = snap.waiting_oldest_age_ms / max(self._slo_ttft_ms,
                                                        1e-6)
        err_backlog = backlog_norm - self._backlog_target
        delta_q = self._kp_q * err_backlog

        delta = delta_slo + delta_q
        deadband_dropped = abs(delta) < self._deadband
        if deadband_dropped:
            delta = 0.0

        candidate = self._last_ratio + delta
        smoothed = ((1.0 - self._ema_alpha) * self._last_ratio
                    + self._ema_alpha * candidate)
        smoothed = max(self._ratio_min, min(self._ratio_max, smoothed))

        self._last_ratio = smoothed
        self._cached = smoothed

        self._emit_tick(
            iter_id=iter_id,
            ratio_before=ratio_before,
            ratio_after=smoothed,
            delta_slo=delta_slo,
            delta_q=delta_q,
            deadband_dropped=deadband_dropped,
            err_ttft_raw=err_ttft_raw,
            err_tpot_raw=err_tpot_raw,
            err_ttft=err_ttft,
            err_tpot_effective=effective_err_tpot,
            ttft_viol_rate=ttft_viol,
            tpot_viol_rate=tpot_viol,
            n_ttft_window=n_ttft,
            n_tpot_window=n_tpot,
            kv_freeze_triggered=False,
            cold_start=False,
        )
        return self._cached

    # ---- diagnostic emit ----

    def _emit_tick(self, *, iter_id, ratio_before, ratio_after, delta_slo,
                   delta_q, deadband_dropped, err_ttft_raw, err_tpot_raw,
                   err_ttft, err_tpot_effective, ttft_viol_rate, tpot_viol_rate,
                   n_ttft_window, n_tpot_window, kv_freeze_triggered,
                   cold_start) -> None:
        """Emit one ControllerTickRecord. No-op if telemetry has no
        record_controller_tick (e.g. test stubs)."""
        emit = getattr(self._telemetry, "record_controller_tick", None)
        if emit is None:
            return
        rec = ControllerTickRecord(
            iter_id=iter_id,
            # Use monotonic (matches IterRecord.ts_ms / ChunkPlanRecord.ts_ms)
            # so post-hoc cross-correlation lines up on a single clock axis.
            ts_ms=time.monotonic() * 1000.0,
            ratio_before=ratio_before,
            ratio_after=ratio_after,
            delta_slo=delta_slo,
            delta_q=delta_q,
            deadband_dropped=deadband_dropped,
            err_ttft_raw=err_ttft_raw,
            err_tpot_raw=err_tpot_raw,
            err_ttft=err_ttft,
            err_tpot_effective=err_tpot_effective,
            ttft_viol_rate=ttft_viol_rate,
            tpot_viol_rate=tpot_viol_rate,
            n_ttft_window=n_ttft_window,
            n_tpot_window=n_tpot_window,
            tpot_saturated=self._tpot_saturated,
            tpot_clamp_active=self._tpot_clamp_active,
            tpot_violate_streak=self._tpot_violate_streak,
            tpot_recovery_streak=self._tpot_recovery_streak,
            kv_freeze_triggered=kv_freeze_triggered,
            cold_start=cold_start,
        )
        emit(rec)

    # ---- internals ----

    def _pull_recent_violations(self) -> None:
        """P1.6b reworked feedback path.

        ttft (F1): scheduler pushes each RequestRecord to telemetry at
        first-token, so recent_requests yields ttft_ms ≈ on the order of
        request latency rather than full lifetime. _ttft_seen_ids dedups
        so each request contributes one ttft sample.

        tpot (F2): _tpot_window is rebuilt fresh each tick from
          (a) in-flight requests (tracker.iter_in_flight_records) — their
              tpot_ms_mean reflects current decode rate, no lifetime lag
          (b) recently finished requests (telemetry.recent_requests, last
              window_size) — keeps the post-completion tail visible
        No dedup needed: rebuild discards previous samples.
        """
        records = self._telemetry.recent_requests(self._window_size)

        # F1: ttft from per-req push (deduped, one-shot per req).
        for r in records:
            if r.request_id in self._ttft_seen_ids:
                continue
            if r.ttft_ms is not None:
                self._ttft_window.append(r.ttft_ms)
                self._ttft_seen_ids.add(r.request_id)
        # Bound the seen-ids set: keep only ids still in the telemetry
        # buffer (older ones evicted, can't reappear).
        if len(self._ttft_seen_ids) > self._seen_ids_cap:
            self._ttft_seen_ids = {r.request_id for r in records}

        # F2: rebuild tpot window from in-flight + recently-finished.
        new_tpot: list[float] = []
        if self._tracker is not None:
            for r in self._tracker.iter_in_flight_records():
                mean_tpot = r.tpot_ms_mean
                if mean_tpot is not None:
                    new_tpot.append(mean_tpot)
        # Fallback: any records that finished after our in-flight scan
        # still contribute via telemetry.recent_requests.
        for r in records:
            if r.finish_ts_ms is None:
                # Already counted as in-flight via tracker (or no tracker
                # available); avoid double-counting via tracker path.
                if self._tracker is not None:
                    continue
            mean_tpot = r.tpot_ms_mean
            if mean_tpot is not None:
                new_tpot.append(mean_tpot)
        # Bound the rebuilt window — keep most recent samples if oversize.
        if len(new_tpot) > self._window_size:
            new_tpot = new_tpot[-self._window_size:]
        self._tpot_window = new_tpot

    @staticmethod
    def _violation_rate(window: deque, threshold: float) -> float:
        if not window:
            return 0.0
        n_viol = sum(1 for v in window if v > threshold)
        return n_viol / len(window)


def make_controller(cfg, telemetry: Telemetry, tracker=None):
    """Dispatch on `cfg.controller_kind` to build the right controller.

    Falls back to StaticRatioController for ``static`` (default) or when
    ``disable_controller`` is set, so existing configs keep working untouched.

    P1.6b: ``tracker`` (RequestTracker) is optional; SLOReactiveController
    uses it for in-flight tpot polling. None preserves legacy behaviour.
    """
    if cfg.disable_controller or cfg.controller_kind == "static":
        return StaticRatioController(
            static_ratio=cfg.static_ratio,
            update_interval=cfg.controller_update_interval,
            telemetry=telemetry,
        )
    if cfg.controller_kind == "slo_pid":
        return SLOReactiveController(
            telemetry=telemetry,
            tracker=tracker,
            slo_ttft_ms=cfg.slo_ttft_ms,
            slo_tpot_ms=cfg.slo_tpot_ms,
            target_violation_rate=cfg.slo_target_violation_rate,
            kp=cfg.slo_pid_kp,
            ratio_min=cfg.slo_ratio_min,
            ratio_max=cfg.slo_ratio_max,
            update_interval=cfg.controller_update_interval,
            initial_ratio=cfg.static_ratio,
            ema_alpha=cfg.slo_pid_ema_alpha,
            deadband=cfg.slo_pid_deadband,
            min_samples=cfg.slo_pid_min_samples,
            kv_freeze_watermark=cfg.kv_free_watermark,
            starvation_min_ticks=cfg.slo_starvation_min_ticks,
            starvation_release_margin=cfg.slo_starvation_release_margin,
            kp_q=cfg.slo_pid_kp_q,
            backlog_target=cfg.slo_pid_backlog_target,
            tpot_saturation_enabled=cfg.slo_tpot_saturation_enabled,
            tpot_saturation_min_ticks=cfg.slo_tpot_saturation_min_ticks,
            tpot_saturation_release_ticks=cfg.slo_tpot_saturation_release_ticks,
            relu_err=cfg.slo_pid_relu_err,
        )
    raise ValueError(f"unknown controller_kind: {cfg.controller_kind!r}")
