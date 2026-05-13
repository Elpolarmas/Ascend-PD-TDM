"""Async QPS-sweep driver — 走 vllm OpenAI-compat HTTP server。

设计要点（详见 memory/current_task.md）：
- workload 通过 WorkloadSource 抽象解耦，Phase 1 用 SyntheticPoisson
- 异步注入：每个 Request 的 arrival_time 决定何时 POST，背压由 server 自身排队承担
- 客户端只记录 e2e / 状态码 / token 数，per-req TTFT/TPOT 走 server 端 TDM passive_tracker 落盘
- 稳态窗口：arrival_time ∈ [warmup_s, duration_s] 的请求计入指标，前后两段丢弃
- 每 QPS 点一个独立运行（外部脚本编排不同 QPS / 不同配置 / 不同 server）

冒烟用法（server 已在 :8000 跑、passive_tracker run_id=server_smoke）：
  python qps_sweep.py --qps 4 --duration 30 --warmup 0 \\
      --base-url http://127.0.0.1:8000 --model /vllm-workspace/models/models/Qwen3-8B \\
      --config-name smoke --out results/qps_sweep_smoke.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from lib.metrics import aggregate_window, join, load_tracker_records
from lib.workload import (  # noqa: F401
    AzureTraceReplay,
    SyntheticBurst,
    DEFAULT_PROMPT_PROFILE,
    PROMPT_PROFILES,
    Request,
    SyntheticPoisson,
    WorkloadSource,
    resolve_prompt_profile,
)


@dataclass
class ClientRecord:
    req_id: str
    arrival_time_s: float
    submit_ts: float
    finish_ts: float
    status: int
    prompt_tokens: int | None
    completion_tokens: int | None
    server_req_id: str | None
    error: str | None = None
    # streaming 模式下客户端自测的 TTFT/TPOT。stream=False 时为 None。
    # PD 分离没 server tracker，必须靠这两条对齐 c1/c2/c3。
    client_ttft_ms: float | None = None
    client_tpot_ms_mean: float | None = None
    client_n_chunks: int | None = None

    @property
    def e2e_s(self) -> float:
        return self.finish_ts - self.submit_ts


@dataclass
class RunSummary:
    config_name: str
    workload_name: str
    qps_target: float
    duration_s: float
    warmup_s: float
    base_url: str
    model: str
    tracker_jsonl: str | None
    n_submitted: int = 0
    n_ok: int = 0
    n_err: int = 0
    wallclock_s: float = 0.0
    qps_actual_arrival: float = 0.0  # n_submitted / duration_s（到达率）
    window: dict = field(default_factory=dict)  # 见 lib.metrics.aggregate_window


async def _send_one(
    client: httpx.AsyncClient,
    model: str,
    req: Request,
    t0: float,
    records: list[ClientRecord],
    stream: bool,
    ignore_eos: bool = False,
) -> None:
    # arrival_time 是相对 t0 的目标提交时刻；超时不补偿，落到记录里
    delay = req.arrival_time_s - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    submit_ts = time.perf_counter()
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if ignore_eos:
        # Trace-replay mode: force decode to exactly max_tokens so output
        # length faithfully matches the trace; synthetic prompts may
        # otherwise trigger early EOS at random points.
        payload["ignore_eos"] = True
    if stream:
        payload["stream_options"] = {"include_usage": True}
    rec = ClientRecord(
        req_id=req.req_id,
        arrival_time_s=req.arrival_time_s,
        submit_ts=submit_ts,
        finish_ts=submit_ts,
        status=0,
        prompt_tokens=None,
        completion_tokens=None,
        server_req_id=None,
    )
    try:
        if not stream:
            r = await client.post("/v1/completions", json=payload)
            rec.finish_ts = time.perf_counter()
            rec.status = r.status_code
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usage") or {}
                rec.prompt_tokens = usage.get("prompt_tokens")
                rec.completion_tokens = usage.get("completion_tokens")
                rec.server_req_id = j.get("id")
            else:
                rec.error = r.text[:200]
        else:
            await _send_one_stream(client, payload, rec, submit_ts)
    except Exception as e:  # noqa: BLE001
        rec.finish_ts = time.perf_counter()
        rec.status = -1
        rec.error = repr(e)[:200]
    records.append(rec)


async def _send_one_stream(
    client: httpx.AsyncClient,
    payload: dict,
    rec: ClientRecord,
    submit_ts: float,
) -> None:
    """streaming 客户端：解析 SSE chunks 自测 TTFT 和 inter-token latency。

    TTFT = 首个非空 chunk 到达时刻 - submit_ts
    TPOT = (last_chunk_ts - first_chunk_ts) / max(1, n_chunks - 1)
    用于 c4_pd 等无 server tracker 的场景。
    """
    chunk_timestamps: list[float] = []
    server_id: str | None = None
    usage: dict | None = None
    async with client.stream("POST", "/v1/completions", json=payload) as r:
        rec.status = r.status_code
        if r.status_code != 200:
            rec.error = (await r.aread())[:200].decode("utf-8", "replace")
            rec.finish_ts = time.perf_counter()
            return
        async for raw in r.aiter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            now = time.perf_counter()
            if server_id is None:
                server_id = obj.get("id")
            choices = obj.get("choices") or []
            has_text = bool(choices) and bool(
                choices[0].get("text") or choices[0].get("delta", {}).get("content")
            )
            if has_text:
                chunk_timestamps.append(now)
            u = obj.get("usage")
            if u:
                usage = u
    rec.finish_ts = time.perf_counter()
    rec.server_req_id = server_id
    if usage:
        rec.prompt_tokens = usage.get("prompt_tokens")
        rec.completion_tokens = usage.get("completion_tokens")
    if chunk_timestamps:
        rec.client_ttft_ms = round((chunk_timestamps[0] - submit_ts) * 1000.0, 3)
        rec.client_n_chunks = len(chunk_timestamps)
        if len(chunk_timestamps) >= 2:
            span = chunk_timestamps[-1] - chunk_timestamps[0]
            rec.client_tpot_ms_mean = round(
                span * 1000.0 / max(1, len(chunk_timestamps) - 1), 3)


async def run_driver(
    workload: WorkloadSource,
    base_url: str,
    model: str,
    duration_s: float,
    max_concurrency: int = 256,
    stream: bool = False,
    ignore_eos: bool = False,
) -> tuple[list[ClientRecord], float]:
    # 连接池要够大：高 QPS × 长 read（max_tokens 大）会让 in-flight 请求数远超 QPS。
    # pool timeout 拉到和 read 一样长，避免「拿不到连接」被误判成 server 崩溃。
    # 见前一轮坑：max_connections=256 在 qps=24 时 PoolTimeout 滚雪球，被误读为 server 过载。
    limits = httpx.Limits(max_connections=max_concurrency,
                          max_keepalive_connections=max_concurrency)
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=300.0)
    records: list[ClientRecord] = []
    tasks: list[asyncio.Task] = []
    # noproxy: server 是本地 loopback，避免被 http_proxy 拦截
    transport = httpx.AsyncHTTPTransport(retries=0)
    async with httpx.AsyncClient(
        base_url=base_url, limits=limits, timeout=timeout,
        transport=transport, trust_env=False,
    ) as client:
        t0 = time.perf_counter()
        for req in workload:
            tasks.append(asyncio.create_task(
                _send_one(client, model, req, t0, records, stream,
                          ignore_eos=ignore_eos)))
            # 让出控制权，使 sleep/调度真正并发
            if len(tasks) % 32 == 0:
                await asyncio.sleep(0)
        # 等所有任务结束（受 timeout 兜底）
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        wallclock = time.perf_counter() - t0
    return records, wallclock


def aggregate(
    records: list[ClientRecord],
    cfg: RunSummary,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> tuple[RunSummary, list[dict]]:
    cfg.n_submitted = len(records)
    cfg.n_ok = sum(1 for r in records if r.status == 200)
    cfg.n_err = cfg.n_submitted - cfg.n_ok
    cfg.qps_actual_arrival = round(cfg.n_submitted / max(1e-9, cfg.duration_s), 3)

    client_dicts = [asdict(r) for r in records]
    tracker_idx = (load_tracker_records(cfg.tracker_jsonl)
                   if cfg.tracker_jsonl else {})
    joined = join(client_dicts, tracker_idx)
    cfg.window = aggregate_window(
        joined,
        warmup_s=cfg.warmup_s,
        window_end_s=cfg.duration_s,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
    )
    return cfg, [asdict(j) for j in joined]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--duration", type=float, default=90.0,
                        help="总时长 s（含 warmup）")
    parser.add_argument("--warmup", type=float, default=30.0,
                        help="前 N 秒作为 warmup，不计入指标")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-name", required=True,
                        help="逻辑配置名（如 c1_baseline / c2_tdm），仅用于落盘")
    parser.add_argument("--out", required=True, help="结果 json 路径")
    parser.add_argument("--max-concurrency", type=int, default=2048,
                        help="httpx 连接池上限。qps×latency 决定 in-flight 峰值，"
                             "默认 2048 足够 qps=64 + 长 read。")
    parser.add_argument("--tracker-jsonl", default=None,
                        help="server 端 TDM passive_tracker 的 _req.jsonl 路径，"
                             "用于 join per-req TTFT/TPOT")
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    # workload 形状：lognormal(mu, sigma) clamped 到 [min, max]
    # `--prompt-profile` 提供命名预设（见 lib/workload.PROMPT_PROFILES），
    # 个别 `--prompt-*` flag 仍可单点覆盖（None=未传 → 取 profile 值）。
    parser.add_argument("--prompt-profile",
                        choices=sorted(PROMPT_PROFILES),
                        default=DEFAULT_PROMPT_PROFILE,
                        help="prompt 形状预设：short=v1_bench/8K canonical "
                             "(p50≈90, p99≈1029); long=TDM 主场 regime "
                             "(p50≈1100, p99 clamp@4000)")
    parser.add_argument("--prompt-mu", type=float, default=None)
    parser.add_argument("--prompt-sigma", type=float, default=None)
    parser.add_argument("--prompt-min", type=int, default=None)
    parser.add_argument("--prompt-max", type=int, default=None)
    parser.add_argument("--output-mu", type=float, default=5.0)
    parser.add_argument("--output-sigma", type=float, default=0.8)
    parser.add_argument("--output-min", type=int, default=16)
    parser.add_argument("--output-max", type=int, default=512)
    parser.add_argument("--stream", action="store_true",
                        help="启用 streaming：客户端自测 TTFT/TPOT，"
                             "用于无 server tracker 的 PD-disagg 等场景")
    # Arrival pattern selection. "poisson" (default) = stationary process;
    # "burst" = periodic square-wave;
    # "trace" = real workload replay (e.g. Azure LLM Inference Trace).
    parser.add_argument("--arrival-mode",
                        choices=["poisson", "burst", "trace"],
                        default="poisson")
    parser.add_argument("--burst-period", type=float, default=10.0,
                        help="Burst period in seconds (only for burst mode)")
    parser.add_argument("--burst-high-qps", type=float, default=64.0,
                        help="Peak qps during burst high-segment")
    parser.add_argument("--burst-low-qps", type=float, default=4.0,
                        help="Quiet qps during burst low-segment")
    parser.add_argument("--burst-high-frac", type=float, default=0.2,
                        help="Fraction of period that is high-segment "
                             "(rest is low-segment)")
    # Trace replay mode (arrival-mode=trace). CSV must contain timestamp +
    # prompt-tokens + output-tokens columns; see AzureTraceReplay docstring.
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Path to trace CSV (required when "
                             "--arrival-mode=trace)")
    parser.add_argument("--trace-time-scale", type=float, default=1.0,
                        help="Time-axis scale factor for trace replay. "
                             ">1 compresses (faster replay), <1 stretches.")
    parser.add_argument("--trace-start-offset", type=float, default=0.0,
                        help="Skip first N seconds of trace before replay")
    parser.add_argument("--trace-max-rows", type=int, default=None,
                        help="Limit number of trace rows loaded (debug)")
    parser.add_argument("--trace-max-prompt-tokens", type=int, default=None,
                        help="Skip trace rows with prompt > this (keep arrival "
                             "rate honest); typically set just below "
                             "--max-model-len")
    parser.add_argument("--trace-max-output-tokens", type=int, default=None,
                        help="Skip trace rows with output > this (avoid "
                             "marathon decodes blowing trace window)")
    args = parser.parse_args()

    # profile 给默认；个别 flag 显式传值时单点覆盖
    prof = resolve_prompt_profile(args.prompt_profile)
    prompt_mu = args.prompt_mu if args.prompt_mu is not None else prof["prompt_mu"]
    prompt_sigma = (args.prompt_sigma if args.prompt_sigma is not None
                    else prof["prompt_sigma"])
    prompt_min = (args.prompt_min if args.prompt_min is not None
                  else int(prof["prompt_min"]))
    prompt_max = (args.prompt_max if args.prompt_max is not None
                  else int(prof["prompt_max"]))
    # mixture 模式（如 "mixed" profile）从 profile 拿；CLI 不暴露（profile-only 特性）
    prompt_mixture = prof.get("prompt_mixture")
    if args.arrival_mode == "trace":
        if not args.trace_file:
            raise SystemExit("--arrival-mode=trace requires --trace-file PATH")
        # Trace mode ignores --qps / --prompt-* / --output-* (lengths come
        # from trace rows). --duration still bounds the replay window
        # (post-scale seconds); use None to replay entire trace.
        workload = AzureTraceReplay(
            trace_csv=args.trace_file,
            duration_s=args.duration if args.duration > 0 else None,
            start_offset_s=args.trace_start_offset,
            time_scale=args.trace_time_scale,
            seed=args.seed,
            max_rows=args.trace_max_rows,
            max_prompt_tokens=args.trace_max_prompt_tokens,
            max_output_tokens=args.trace_max_output_tokens,
        )
    elif args.arrival_mode == "burst":
        # Burst mode: --qps is treated as ignored (avg derived from high/low/frac).
        workload = SyntheticBurst(
            burst_period_s=args.burst_period,
            burst_high_qps=args.burst_high_qps,
            burst_low_qps=args.burst_low_qps,
            burst_high_frac=args.burst_high_frac,
            duration_s=args.duration, seed=args.seed,
            prompt_mu=prompt_mu, prompt_sigma=prompt_sigma,
            prompt_min=prompt_min, prompt_max=prompt_max,
            output_mu=args.output_mu, output_sigma=args.output_sigma,
            output_min=args.output_min, output_max=args.output_max,
            prompt_mixture=prompt_mixture,
        )
    else:
        workload = SyntheticPoisson(
            qps=args.qps, duration_s=args.duration, seed=args.seed,
            prompt_mu=prompt_mu, prompt_sigma=prompt_sigma,
            prompt_min=prompt_min, prompt_max=prompt_max,
            output_mu=args.output_mu, output_sigma=args.output_sigma,
            output_min=args.output_min, output_max=args.output_max,
            prompt_mixture=prompt_mixture,
        )
    cfg = RunSummary(
        config_name=args.config_name,
        workload_name=workload.name,
        qps_target=args.qps,
        duration_s=args.duration,
        warmup_s=args.warmup,
        base_url=args.base_url,
        model=args.model,
        tracker_jsonl=args.tracker_jsonl,
    )

    print(f"[driver] {workload.name}, target qps={args.qps}, "
          f"duration={args.duration}s, warmup={args.warmup}s",
          flush=True)
    t0 = time.time()
    records, wallclock = asyncio.run(run_driver(
        workload, args.base_url, args.model, args.duration,
        max_concurrency=args.max_concurrency,
        stream=args.stream,
        # trace replay needs ignore_eos to make decode length deterministic
        ignore_eos=(args.arrival_mode == "trace"),
    ))
    cfg.wallclock_s = round(wallclock, 3)
    cfg, joined_dicts = aggregate(records, cfg, args.slo_ttft_ms, args.slo_tpot_ms)
    w = cfg.window
    print(f"[driver] done in {time.time()-t0:.1f}s wall, "
          f"submitted={cfg.n_submitted} ok={cfg.n_ok} err={cfg.n_err}, "
          f"arrival_qps={cfg.qps_actual_arrival}",
          flush=True)
    if w:
        print(f"[driver] window={w['window_s']} "
              f"matched={w['n_matched']}/{w['n_ok']} "
              f"unmatched={w['n_unmatched']} meet_slo={w['n_meet_slo']}",
              flush=True)
        if w["ttft_ms"]["mean"] is not None:
            print(f"[driver] ttft ms mean={w['ttft_ms']['mean']} "
                  f"p99={w['ttft_ms']['p99']} | "
                  f"tpot ms mean={w['tpot_ms_mean']['mean']} "
                  f"p99={w['tpot_ms_mean']['p99']} | "
                  f"e2e ms mean={w['e2e_ms']['mean']} p99={w['e2e_ms']['p99']}",
                  flush=True)
        print(f"[driver] output_tput={w['output_throughput_tok_s']} tok/s, "
              f"goodput={w['goodput_tok_s']} tok/s "
              f"(SLO ttft<{args.slo_ttft_ms}ms ∧ tpot<{args.slo_tpot_ms}ms)",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(cfg),
        "joined": joined_dicts,
        "client_records": [asdict(r) for r in records],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[driver] wrote {out_path}", flush=True)
    return 0 if cfg.n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
