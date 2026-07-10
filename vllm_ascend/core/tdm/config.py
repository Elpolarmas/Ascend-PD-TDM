from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TDMConfig:
    """Pure-data TDM knobs, loaded from
    ``vllm_config.additional_config["tdm"]``.

    We deliberately do NOT subclass AscendSchedulerConfig — that would
    require modifying schedule_config.py to register the subclass. Instead
    TDMScheduler reads this dataclass directly, leaving every existing
    file untouched.
    """

    enable_tdm: bool = False
    # Passive observation: skip the decision pipeline but keep tracker +
    # telemetry running. Used by baseline benchmarks that want apples-to-apples
    # per-request TTFT/TPOT against the active TDM run. Ignored when
    # enable_tdm is True.
    passive_tracker: bool = False
    # Controller
    window_size: int = 32
    controller_update_interval: int = 8
    disable_controller: bool = False
    static_ratio: float = 0.5
    # "static" | "slo_pid"; static keeps M1 behaviour, slo_pid is M2.
    controller_kind: str = "static"
    # SLO targets (ms) — consumed by SLOReactiveController; static still
    # accepts them and logs unchanged.
    slo_ttft_ms: float = 500.0
    slo_tpot_ms: float = 50.0
    # SLOReactiveController knobs (only read when controller_kind == "slo_pid")
    slo_target_violation_rate: float = 0.05
    slo_pid_kp: float = 0.5
    slo_pid_ema_alpha: float = 0.3
    slo_pid_deadband: float = 0.02
    slo_pid_min_samples: int = 16
    slo_ratio_min: float = 0.05
    slo_ratio_max: float = 0.8
    # M2.3 backlog-aware additive term: delta_q = kp_q * (oldest_age_ms /
    # slo_ttft_ms - backlog_target). Adds a leading-indicator pressure on
    # ratio that survives steady state (where SLO violation rates decay to
    # 0). Set kp_q=0 to recover pure PID (M2) behaviour.
    slo_pid_kp_q: float = 0.5
    slo_pid_backlog_target: float = 0.5
    # M2.4 tpot saturation detector: when tpot_violation_rate stays above
    # target_violation_rate for `min_ticks` consecutive update ticks, the
    # tpot SLO is treated as physically unreachable and err_tpot is zeroed
    # out so it stops dragging ratio toward the floor in vain. Hysteresis
    # release: tpot_violation_rate must fall below target for
    # `release_ticks` consecutive ticks to clear saturation. Disable to
    # restore M2.3-and-earlier semantics.
    slo_tpot_saturation_enabled: bool = True
    slo_tpot_saturation_min_ticks: int = 5
    slo_tpot_saturation_release_ticks: int = 5
    # M2.5 ReLU clipping on PID error: clamp err_ttft / err_tpot to max(0, .)
    # so an SLO that is already satisfied (violation_rate < target) cannot
    # push ratio in the wrong direction. With raw signed err, a comfortably-
    # met ttft generates err_ttft = -target which steadily decays ratio
    # toward the floor — even when no help is needed. ReLU makes the
    # controller a one-sided pressure response: only violations move ratio.
    slo_pid_relu_err: bool = True
    # M2.1 starvation guard: when ratio is pinned to slo_ratio_min and tpot
    # still violates SLO for this many consecutive update ticks, the controller
    # zeroes out err_tpot so only ttft can drive ratio off the floor.
    slo_starvation_min_ticks: int = 3
    # M2.2 hysteresis: guard releases only when ratio rises by at least this
    # margin above slo_ratio_min. Prevents the activate ↔ release oscillation
    # observed when margin=0 (PID immediately pulls ratio back to floor every
    # time guard releases). Set to 0.0 to recover M2.1 behaviour.
    slo_starvation_release_margin: float = 0.10
    # Selector
    selector_type: str = "token_bucket"
    bucket_cap: int = 4
    initial_phase: str = "decode"
    # P1.7 leading indicator (TTFT urgency fast loop). When enabled, peek()
    # overrides the token-bucket decision with PREFILL whenever the oldest
    # waiting request's age exceeds urgency_ttft_threshold * slo_ttft_ms.
    # Motivation: P1.5 showed M3.1 vs M1+chunk delta collapsed to ~0 on
    # saturated code traces under strict TPOT — the slow PID loop cannot
    # react inside a single SLO budget. The fast loop short-circuits when
    # head-of-line TTFT is about to violate. urgency_charge_bucket=True
    # debits a bucket token on rescue so the slow loop "sees" the prefill.
    urgency_ttft_enabled: bool = False
    urgency_ttft_threshold: float = 0.8
    urgency_charge_bucket: bool = True
    # P1.7b TPOT starvation fast loop (companion to urgency_ttft). When the
    # oldest running decode has been silent > starvation_tpot_threshold *
    # slo_tpot_ms, peek() overrides a bucket-PREFILL decision with DECODE.
    # Threshold is in units of slo_tpot_ms: e.g. 3.0 = decode head silent
    # for more than 3 TPOT budgets in a row. Bucket is NOT charged on this
    # override (slow loop's prefill credit is preserved across the rescue).
    # P1.7b motivation: P1.7 single-sided urgency_ttft never triggered in
    # azure traces (0% rate) AND only handled the TTFT direction; thesis
    # claim "iter-grain bidirectional warning" needs both sides.
    starvation_tpot_enabled: bool = False
    starvation_tpot_threshold: float = 3.0
    # P1.7b post-hoc (2026-05-17): the default `tokens >= 1.0` AND-gate in
    # selector.peek() pins starvation rescue to the narrow window where the
    # slow loop also wants prefill. Empirically on code workload, silence >=
    # threshold fires on 20% of iters but the combined AND fires only 0.5% —
    # the bidirectional-warning thesis claim is not actually exercised by
    # the default implementation. starvation_decouple_bucket=True drops the
    # AND-gate: starvation_tpot fires on silence alone, regardless of bucket
    # state. Bucket is still not charged on rescue (preserves slow-loop P/D
    # anchor), but rescue can override either bucket-decode or bucket-prefill
    # iters once silence is high. Default False keeps M3.3 back-compat.
    starvation_decouple_bucket: bool = False
    # HardConstraints
    min_slice_iters: int = 2
    max_slice_iters: int = 8
    # BoundaryGuard
    kv_free_watermark: float = 0.05
    # M3.1 prefill chunking: iter-wide cap on total prefill tokens scheduled
    # in a single P iter. None = chunking disabled (M2.7-equivalent back-compat
    # path, same code edits but truncate budget = inf). When set, partial-
    # prefill requests are kept in self.running across iters and resumed via a
    # pre-pass at the head of the prefill section, mirroring vanilla vLLM v1's
    # running-loop convention (vllm/v1/core/sched/scheduler.py:208-220) while
    # preserving Ascend's phase-pure invariant (decode reqs filtered out of
    # the pre-pass, left to the existing decode loop).
    prefill_chunk_tokens: Optional[int] = None
    # c2_tdm_m31_fia ablation: 强制 attention 走 FIA (_forward_v1_style),
    # 绕开 vllm-ascend v0.11.0rc1 的 dedicated kernels。用于拆分 phase-pure
    # 调度策略贡献 vs dedicated kernel 速度贡献。等价于 vllm-ascend v0.13+
    # 把所有 attention 路径统一到 FIA 的行为。
    # 实装通过 monkey-patch AscendAttentionBackendImpl.forward,见 attn_patch.py。
    force_fia_attention: bool = False
    # Telemetry
    telemetry_dir: str = "./results/tdm_trace"
    telemetry_enabled: bool = True
    run_id: Optional[str] = None

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "TDMConfig":
        raw = {}
        ac = getattr(vllm_config, "additional_config", None) or {}
        if isinstance(ac, dict):
            raw = ac.get("tdm", {}) or {}
        # Only keep known fields; drop unknowns so users get a typo warning
        # via the absence of effect rather than a crash.
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**kwargs)
        cfg._validate(vllm_config)
        return cfg

    def _validate(self, vllm_config: Any) -> None:
        if not self.enable_tdm:
            return
        sched = getattr(vllm_config, "scheduler_config", None)
        if sched is not None:
            if getattr(sched, "chunked_prefill_enabled", False):
                raise ValueError(
                    "enable_tdm requires chunked_prefill disabled "
                    "(AscendScheduler short-circuits to vanilla scheduler "
                    "when CP is on, bypassing TDM entirely).")
            if getattr(sched, "enable_pd_transfer", False):
                raise ValueError(
                    "enable_tdm is mutually exclusive with enable_pd_transfer.")
        if self.initial_phase not in ("prefill", "decode"):
            raise ValueError(
                f"initial_phase must be 'prefill' or 'decode', "
                f"got {self.initial_phase!r}")
        if not 0.0 <= self.static_ratio <= 1.0:
            raise ValueError(
                f"static_ratio must be in [0, 1], got {self.static_ratio}")
        if self.min_slice_iters < 1 or self.max_slice_iters < self.min_slice_iters:
            raise ValueError(
                "Require 1 <= min_slice_iters <= max_slice_iters; got "
                f"{self.min_slice_iters}/{self.max_slice_iters}")
        if self.controller_kind not in ("static", "slo_pid"):
            raise ValueError(
                "controller_kind must be 'static' or 'slo_pid', "
                f"got {self.controller_kind!r}")
        if not 0.0 <= self.slo_ratio_min <= self.slo_ratio_max <= 1.0:
            raise ValueError(
                "Require 0 <= slo_ratio_min <= slo_ratio_max <= 1; got "
                f"{self.slo_ratio_min}/{self.slo_ratio_max}")
        if not 0.0 < self.slo_pid_ema_alpha <= 1.0:
            raise ValueError(
                "slo_pid_ema_alpha must be in (0, 1], got "
                f"{self.slo_pid_ema_alpha}")
        if self.slo_starvation_min_ticks < 1:
            raise ValueError(
                "slo_starvation_min_ticks must be >= 1, got "
                f"{self.slo_starvation_min_ticks}")
        if self.slo_starvation_release_margin < 0.0:
            raise ValueError(
                "slo_starvation_release_margin must be >= 0, got "
                f"{self.slo_starvation_release_margin}")
        if self.slo_pid_kp_q < 0.0:
            raise ValueError(
                f"slo_pid_kp_q must be >= 0, got {self.slo_pid_kp_q}")
        if self.slo_tpot_saturation_min_ticks < 1:
            raise ValueError(
                "slo_tpot_saturation_min_ticks must be >= 1, got "
                f"{self.slo_tpot_saturation_min_ticks}")
        if self.slo_tpot_saturation_release_ticks < 1:
            raise ValueError(
                "slo_tpot_saturation_release_ticks must be >= 1, got "
                f"{self.slo_tpot_saturation_release_ticks}")
        if self.prefill_chunk_tokens is not None:
            if self.prefill_chunk_tokens < 1:
                raise ValueError(
                    "prefill_chunk_tokens must be >= 1 when set, got "
                    f"{self.prefill_chunk_tokens}")
        if self.urgency_ttft_threshold <= 0.0:
            raise ValueError(
                "urgency_ttft_threshold must be > 0, got "
                f"{self.urgency_ttft_threshold}")
        if self.starvation_tpot_threshold <= 0.0:
            raise ValueError(
                "starvation_tpot_threshold must be > 0, got "
                f"{self.starvation_tpot_threshold}")
