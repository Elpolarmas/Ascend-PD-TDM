"""可插拔 workload 抽象。

Phase 1 实现 SyntheticPoisson（Poisson 到达 + lognormal 长度分布，近似 ShareGPT），
Phase 3 在同一 WorkloadSource 接口下实现 AzureTraceReplay，driver 不变。

Prompt profiles
---------------
`PROMPT_PROFILES` 注册表是 prompt 形状的**单一真源**。两个 driver
脚本（`qps_sweep.py` / `run_qps_sweep_all.py`）都从这里取默认值，避免
四个数字在多处 drift。

现有 profile：
  - short  : v1_bench / ShareGPT 近似，p50≈90 / p99≈1029。C1 hybrid
             在此 regime 下天然占优（单 iter prefill 几十毫秒，decode
             顺路搭车不阻塞）。8K canonical sweep 用的就是它。
  - long   : TDM 主场 regime，p50≈1097 / p99≈4000(clamp)。C1 hybrid
             单 iter 必须 prefill 整条长 prompt → ttft 违例 + 所有
             running decode 卡多 iter → tpot 爆 → SLO% 双违例叠加。
             phase-pure (TDM) + chunking (M3) 预期双胜。
             详见 current_task.md §1.3 / §3.1.1。
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Request:
    req_id: str
    arrival_time_s: float  # 相对 driver 启动时刻
    prompt: str
    max_tokens: int
    target_prompt_tokens: int  # 采样目标，便于事后核对


class WorkloadSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Request]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# 简单 filler 词表，用于按目标长度生成 prompt。
# 不追求 token 严格相等，driver 会从 server 响应里读 actual prompt_tokens。
_FILLER_WORDS = (
    "the quick brown fox jumps over a lazy dog while reading a long book about "
    "distributed systems consensus algorithms latency throughput batching prefill "
    "decode attention transformer tensor parallel scheduling memory pressure "
    "watermark admission queue arrival inter token output completion request "
).split()


def _make_prompt(target_tokens: int, rng: random.Random) -> str:
    # 经验：英文 1 word ≈ 1 token，按 word 数近似目标
    n_words = max(4, target_tokens)
    words = [rng.choice(_FILLER_WORDS) for _ in range(n_words)]
    return "Summarize: " + " ".join(words)


# ---------------------------------------------------------------------------
# Prompt profile registry — 单一真源，driver 脚本从这里取默认
# ---------------------------------------------------------------------------

PROMPT_PROFILES: dict[str, dict[str, float]] = {
    # 短 prompt：8K canonical sweep 用的，等价 v1_bench / ShareGPT 近似
    "short": {
        "prompt_mu": 4.5,
        "prompt_sigma": 1.0,
        "prompt_min": 10,
        "prompt_max": 1024,
    },
    # 长 prompt：TDM 主场，C1 hybrid 在此 regime 下应暴露 prefill 阻塞弱点
    "long": {
        "prompt_mu": 7.0,
        "prompt_sigma": 0.8,
        "prompt_min": 256,
        "prompt_max": 4000,
    },
    # 混合 (bimodal)：70% short + 30% long，近似 ShareGPT/Azure 真实长尾。
    # 短 prompt 撑高 batch 利用率（让 hybrid 看似有优势），
    # 长 prompt 触发 prefill 阻塞（让 phase-pure 显出价值）—— 真正的 TDM 主场。
    # `prompt_mixture` 字段当 SyntheticPoisson 检测到时走加权采样路径；
    # 同级别保留 flat prompt_* 字段是给读 workload dict 的下游工具兜底。
    "mixed": {
        "prompt_mixture": [
            {"weight": 0.7, "prompt_mu": 4.5, "prompt_sigma": 1.0,
             "prompt_min": 10,  "prompt_max": 1024},
            {"weight": 0.3, "prompt_mu": 7.0, "prompt_sigma": 0.8,
             "prompt_min": 256, "prompt_max": 4000},
        ],
        # 兜底：weighted-mean / 包络 min-max（informational only）
        "prompt_mu": 5.25,
        "prompt_sigma": 1.0,
        "prompt_min": 10,
        "prompt_max": 4000,
    },
}

DEFAULT_PROMPT_PROFILE = "short"


def resolve_prompt_profile(name: str) -> dict[str, float]:
    """根据 profile 名取出 prompt 形状参数；返回新 dict（调用方可放心修改）。"""
    if name not in PROMPT_PROFILES:
        raise ValueError(
            f"unknown prompt profile {name!r}; "
            f"available: {sorted(PROMPT_PROFILES)}"
        )
    return dict(PROMPT_PROFILES[name])


class SyntheticPoisson(WorkloadSource):
    """Poisson 到达 + lognormal 长度，近似 ShareGPT 工况。

    参数默认值参考 v1_bench 的实测范围，prompt 中位数 ~90 token、output ~150 token。
    """

    def __init__(
        self,
        qps: float,
        duration_s: float,
        seed: int = 0,
        prompt_mu: float = 4.5,
        prompt_sigma: float = 1.0,
        prompt_min: int = 10,
        prompt_max: int = 1024,
        output_mu: float = 5.0,
        output_sigma: float = 0.8,
        output_min: int = 16,
        output_max: int = 512,
        prompt_mixture: list[dict] | None = None,
    ) -> None:
        self.qps = qps
        self.duration_s = duration_s
        self.seed = seed
        self.prompt_mu = prompt_mu
        self.prompt_sigma = prompt_sigma
        self.prompt_min = prompt_min
        self.prompt_max = prompt_max
        self.output_mu = output_mu
        self.output_sigma = output_sigma
        self.output_min = output_min
        self.output_max = output_max
        # mixture 模式：list of {weight, prompt_mu, prompt_sigma, prompt_min, prompt_max}
        # 每个 req 按 weight 抽一个 sub-profile 再从其 lognormal 采。
        self.prompt_mixture = prompt_mixture
        if prompt_mixture:
            ws = [c["weight"] for c in prompt_mixture]
            tot = sum(ws)
            if tot <= 0:
                raise ValueError("prompt_mixture weights must sum > 0")
            self._mix_weights = [w / tot for w in ws]

    @property
    def name(self) -> str:
        return f"poisson_qps{self.qps}_dur{int(self.duration_s)}s"

    def _sample_prompt_tokens(self, rng: random.Random) -> int:
        if self.prompt_mixture:
            # weighted-pick sub-profile 再采样
            sub = rng.choices(self.prompt_mixture, weights=self._mix_weights, k=1)[0]
            mu = sub["prompt_mu"]
            sigma = sub["prompt_sigma"]
            lo = int(sub["prompt_min"])
            hi = int(sub["prompt_max"])
        else:
            mu, sigma = self.prompt_mu, self.prompt_sigma
            lo, hi = self.prompt_min, self.prompt_max
        n = int(round(rng.lognormvariate(mu, sigma)))
        return max(lo, min(hi, n))

    def __iter__(self) -> Iterator[Request]:
        rng = random.Random(self.seed)
        t = 0.0
        idx = 0
        while True:
            # 指数分布 inter-arrival
            t += rng.expovariate(self.qps)
            if t >= self.duration_s:
                return
            target_p = self._sample_prompt_tokens(rng)
            target_o = int(round(rng.lognormvariate(self.output_mu, self.output_sigma)))
            target_o = max(self.output_min, min(self.output_max, target_o))
            yield Request(
                req_id=f"syn-{self.seed}-{idx:06d}",
                arrival_time_s=t,
                prompt=_make_prompt(target_p, rng),
                max_tokens=target_o,
                target_prompt_tokens=target_p,
            )
            idx += 1


class SyntheticBurst(SyntheticPoisson):
    """周期性方波突发到达 — 推理请求抖动测试场景。

    在 ``burst_period_s`` 周期内：
      - 前 ``burst_high_frac × burst_period_s`` 秒按 ``burst_high_qps`` 到达
      - 其余时间按 ``burst_low_qps`` 到达
    平均到达率 = burst_high_qps × burst_high_frac
                + burst_low_qps × (1 - burst_high_frac)

    用 thinning 算法生成非稳态 Poisson 过程：以 max(high, low) 为采样率，
    按当前时刻所处段的 qps/max_qps 概率接受样本。

    设计目的：验证闭环控制器（M2.7 PID）在动态环境下相对开环（M1 static
    ratio）的优势——burst 段 SLO 违反率上升时，PID 能调整 ratio，static
    不能。
    """

    def __init__(
        self,
        burst_period_s: float,
        burst_high_qps: float,
        burst_low_qps: float,
        burst_high_frac: float,
        duration_s: float,
        **kwargs,
    ) -> None:
        if not (0.0 < burst_high_frac < 1.0):
            raise ValueError(f"burst_high_frac must be in (0, 1), "
                             f"got {burst_high_frac}")
        if burst_high_qps <= burst_low_qps:
            raise ValueError(f"burst_high_qps ({burst_high_qps}) must "
                             f"exceed burst_low_qps ({burst_low_qps})")
        avg_qps = (burst_high_qps * burst_high_frac
                   + burst_low_qps * (1.0 - burst_high_frac))
        super().__init__(qps=avg_qps, duration_s=duration_s, **kwargs)
        self.burst_period_s = burst_period_s
        self.burst_high_qps = burst_high_qps
        self.burst_low_qps = burst_low_qps
        self.burst_high_frac = burst_high_frac
        self.t_high = burst_period_s * burst_high_frac

    @property
    def name(self) -> str:
        return (f"burst_avg{self.qps:.1f}_period{self.burst_period_s:.0f}s"
                f"_high{self.burst_high_qps:.0f}_low{self.burst_low_qps:.0f}"
                f"_dur{int(self.duration_s)}s")

    def _qps_at(self, t: float) -> float:
        phase_t = t % self.burst_period_s
        return (self.burst_high_qps if phase_t < self.t_high
                else self.burst_low_qps)

    def __iter__(self) -> Iterator[Request]:
        rng = random.Random(self.seed)
        max_qps = self.burst_high_qps
        t = 0.0
        idx = 0
        while True:
            t += rng.expovariate(max_qps)
            if t >= self.duration_s:
                return
            # Thinning: accept with prob qps_t / max_qps
            current_qps = self._qps_at(t)
            if rng.random() >= current_qps / max_qps:
                continue
            target_p = self._sample_prompt_tokens(rng)
            target_o = int(round(
                rng.lognormvariate(self.output_mu, self.output_sigma)))
            target_o = max(self.output_min, min(self.output_max, target_o))
            yield Request(
                req_id=f"burst-{self.seed}-{idx:06d}",
                arrival_time_s=t,
                prompt=_make_prompt(target_p, rng),
                max_tokens=target_o,
                target_prompt_tokens=target_p,
            )
            idx += 1


# ---------------------------------------------------------------------------
# Trace replay — Azure LLM Inference Trace 等真实生产 trace 回放
# ---------------------------------------------------------------------------


class AzureTraceReplay(WorkloadSource):
    """按真实 trace timestamps 回放请求。

    输入 CSV 至少要有 3 个数值字段(自动识别列名,大小写不敏感):
      - **timestamp**:到达时刻;支持 epoch 秒、ISO8601 字符串、或相对秒数。
        会自动归一化到"相对第一条"为 0 秒。
      - **prompt tokens** (`ContextTokens` / `prompt_tokens` / `input_tokens`)
      - **output tokens** (`GeneratedTokens` / `output_tokens` / `completion_tokens`)

    Azure LLM Inference Trace(Splitwise/DistServe/Sarathi 用过的那个)直接
    满足:`TIMESTAMP, ContextTokens, GeneratedTokens`。

    回放策略
    ---------
    - 用合成 token (filler words) 拼到目标 prompt 长度。模型计算量 / 调度
      决策 / SLO 测量只依赖长度,不依赖语义内容,无信息损失(参 Splitwise 等
      论文 standard practice)。
    - `max_tokens = output_tokens` 配合服务端 `ignore_eos=True` 强制生成
      够数 token,使 decode 时长严格匹配 trace。
    - `time_scale` 缩放时间轴(>1 加速回放,<1 放慢):60 分钟真实 trace
      用 ``time_scale=60`` 压成 1 分钟,burst pattern 几何保留。
    - `start_offset_s` / `duration_s` 截取 trace 片段;默认 0 / 整段。
    """

    def __init__(
        self,
        trace_csv: str,
        duration_s: float | None = None,
        start_offset_s: float = 0.0,
        time_scale: float = 1.0,
        seed: int = 0,
        max_rows: int | None = None,
        max_prompt_tokens: int | None = None,
        max_output_tokens: int | None = None,
        skip_oversize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        max_prompt_tokens / max_output_tokens : int | None
            If set, rows whose ContextTokens / GeneratedTokens exceed the
            limit are either skipped (``skip_oversize=True``, default,
            preserves arrival rate truthfulness) or clamped to the limit
            (``skip_oversize=False``).  Use to keep within server
            ``max_model_len`` / time-budget bounds.
        skip_oversize : bool
            See above.
        """
        if time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {time_scale}")
        self.trace_csv = trace_csv
        self.duration_s = duration_s
        self.start_offset_s = start_offset_s
        self.time_scale = time_scale
        self.seed = seed
        self.max_rows = max_rows
        self.max_prompt_tokens = max_prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.skip_oversize = skip_oversize
        self._records: list[tuple[float, int, int]] | None = None  # (t_s_rel, pt, ot)

    @property
    def name(self) -> str:
        base = self.trace_csv.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        scale = "" if self.time_scale == 1.0 else f"_x{self.time_scale:g}"
        dur = "" if self.duration_s is None else f"_dur{int(self.duration_s)}s"
        off = "" if self.start_offset_s == 0 else f"_off{int(self.start_offset_s)}s"
        return f"trace_{base}{scale}{off}{dur}"

    def _load(self) -> list[tuple[float, int, int]]:
        """读 CSV → list[(t_rel_s, prompt_tokens, output_tokens)] 已按 t 排序。"""
        if self._records is not None:
            return self._records
        import csv
        from datetime import datetime

        # 候选列名(全小写比较)
        ts_keys = {"timestamp", "ts", "arrival", "arrival_time", "time"}
        pt_keys = {"contexttokens", "prompt_tokens", "input_tokens",
                   "promptlength", "context_tokens"}
        ot_keys = {"generatedtokens", "output_tokens", "completion_tokens",
                   "generationlength", "generated_tokens"}

        rows: list[tuple[float, int, int]] = []
        with open(self.trace_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"trace CSV {self.trace_csv} has no header")
            lc = {k.lower().replace(" ", "").replace("_", ""): k
                  for k in reader.fieldnames}

            def pick(keys: set[str]) -> str:
                for k in keys:
                    norm = k.replace("_", "")
                    if norm in lc:
                        return lc[norm]
                raise KeyError(f"none of {keys} found in {reader.fieldnames}")

            ts_col = pick(ts_keys)
            pt_col = pick(pt_keys)
            ot_col = pick(ot_keys)

            for i, row in enumerate(reader):
                if self.max_rows is not None and i >= self.max_rows:
                    break
                raw_ts = row[ts_col].strip()
                # try float seconds first; else ISO datetime
                try:
                    t = float(raw_ts)
                except ValueError:
                    t = datetime.fromisoformat(raw_ts).timestamp()
                pt = int(float(row[pt_col]))
                ot = int(float(row[ot_col]))
                if pt <= 0 or ot <= 0:
                    continue
                # Length-limit handling
                if self.max_prompt_tokens is not None and pt > self.max_prompt_tokens:
                    if self.skip_oversize:
                        continue
                    pt = self.max_prompt_tokens
                if self.max_output_tokens is not None and ot > self.max_output_tokens:
                    if self.skip_oversize:
                        continue
                    ot = self.max_output_tokens
                rows.append((t, pt, ot))
        if not rows:
            raise ValueError(f"no usable rows in {self.trace_csv}")
        rows.sort(key=lambda r: r[0])
        t0 = rows[0][0]
        rows = [(t - t0, pt, ot) for (t, pt, ot) in rows]
        self._records = rows
        return rows

    def __iter__(self) -> Iterator[Request]:
        records = self._load()
        rng = random.Random(self.seed)
        emitted = 0
        win_end = (self.start_offset_s + self.duration_s
                   if self.duration_s is not None else float("inf"))
        for orig_t, pt, ot in records:
            if orig_t < self.start_offset_s:
                continue
            if orig_t >= win_end:
                break
            # 归一化:t=0 对应 driver 启动;time_scale 加速/放慢
            t_rel = (orig_t - self.start_offset_s) / self.time_scale
            yield Request(
                req_id=f"trace-{self.seed}-{emitted:06d}",
                arrival_time_s=t_rel,
                prompt=_make_prompt(pt, rng),
                max_tokens=ot,
                target_prompt_tokens=pt,
            )
            emitted += 1
