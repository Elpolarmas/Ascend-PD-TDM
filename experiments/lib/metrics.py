"""Driver 后处理：把客户端记录与 server 端 TDM tracker 落的 jsonl 做 join，
按稳态窗口聚合 TTFT/TPOT/E2E/Throughput/Goodput。

关键：通过 server_req_id 精确匹配，不依赖时间窗 — 同一 server 跨多 QPS 点
连跑也不会混（tracker.request_id 形如 'cmpl-xxx-0'，去掉末尾 -<n> 即客户端
拿到的 server_req_id）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[i]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p99": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p99": round(percentile(values, 0.99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _strip_sample_suffix(req_id: str) -> str:
    # tracker 写的是 "<server_id>-<n>"，n 是 sampling 索引；vllm 单 n=1 时为 -0
    head, _, tail = req_id.rpartition("-")
    return head if tail.isdigit() else req_id


def load_tracker_records(path: str | Path) -> dict[str, dict]:
    """读 tracker jsonl，按 server_req_id 建索引（去掉 -n 后缀）。"""
    p = Path(path)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = rec.get("request_id")
        if rid is None:
            continue
        out[_strip_sample_suffix(rid)] = rec
    return out


@dataclass
class JoinedRecord:
    req_id: str
    arrival_time_s: float
    server_req_id: str | None
    status: int
    e2e_ms: float | None
    ttft_ms: float | None
    tpot_ms_mean: float | None
    tpot_ms_p99: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    matched: bool


def join(client_records: list[dict], tracker_index: dict[str, dict]) -> list[JoinedRecord]:
    """优先用 server tracker（精确，覆盖 step-level），fallback 到 client streaming
    自测字段（仅在 stream=True 时存在）。c4_pd 没 tracker，全靠 client。"""
    out: list[JoinedRecord] = []
    for r in client_records:
        srv = r.get("server_req_id")
        t = tracker_index.get(srv) if srv else None
        e2e_s = r.get("finish_ts", 0.0) - r.get("submit_ts", 0.0)
        # tracker 优先；缺则用 client streaming 字段
        ttft_ms = t.get("ttft_ms") if t else r.get("client_ttft_ms")
        tpot_ms_mean = t.get("tpot_ms_mean") if t else r.get("client_tpot_ms_mean")
        tpot_ms_p99 = t.get("tpot_ms_p99") if t else None  # client 单值无 p99
        # matched 现在的语义是"有 ttft/tpot 数据可用"，不再仅指 tracker 命中
        has_metrics = ttft_ms is not None and tpot_ms_mean is not None
        out.append(JoinedRecord(
            req_id=r["req_id"],
            arrival_time_s=r["arrival_time_s"],
            server_req_id=srv,
            status=r["status"],
            e2e_ms=round(e2e_s * 1000.0, 3) if r["status"] == 200 else None,
            ttft_ms=ttft_ms,
            tpot_ms_mean=tpot_ms_mean,
            tpot_ms_p99=tpot_ms_p99,
            prompt_tokens=(r.get("prompt_tokens") if r.get("prompt_tokens") is not None
                           else (t.get("prompt_tokens") if t else None)),
            output_tokens=(r.get("completion_tokens") if r.get("completion_tokens") is not None
                           else (t.get("output_tokens") if t else None)),
            matched=has_metrics,
        ))
    return out


def aggregate_window(
    joined: list[JoinedRecord],
    warmup_s: float,
    window_end_s: float,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> dict:
    """对 arrival_time ∈ [warmup_s, window_end_s) 的 OK 请求聚合。

    Goodput 定义：稳态窗口内、status==200、ttft<SLO_ttft、tpot_mean<SLO_tpot 的
    请求 output token 总数 / 稳态窗口时长（s）。
    """
    win = [j for j in joined
           if warmup_s <= j.arrival_time_s < window_end_s]
    ok = [j for j in win if j.status == 200]
    matched = [j for j in ok if j.matched]
    n_unmatched = len(ok) - len(matched)

    ttfts = [j.ttft_ms for j in matched if j.ttft_ms is not None]
    tpots = [j.tpot_ms_mean for j in matched if j.tpot_ms_mean is not None]
    e2es = [j.e2e_ms for j in ok if j.e2e_ms is not None]
    out_tok = [j.output_tokens for j in ok if j.output_tokens]

    window_dur = max(1e-9, window_end_s - warmup_s)
    output_tput = sum(out_tok) / window_dur if out_tok else 0.0

    good = [
        j for j in matched
        if j.ttft_ms is not None and j.tpot_ms_mean is not None
        and j.ttft_ms < slo_ttft_ms and j.tpot_ms_mean < slo_tpot_ms
        and j.output_tokens
    ]
    goodput_tok_s = sum(j.output_tokens for j in good) / window_dur

    return {
        "window_s": [round(warmup_s, 3), round(window_end_s, 3)],
        "n_in_window": len(win),
        "n_ok": len(ok),
        "n_matched": len(matched),
        "n_unmatched": n_unmatched,
        "n_meet_slo": len(good),
        "ttft_ms": summarize(ttfts),
        "tpot_ms_mean": summarize(tpots),
        "e2e_ms": summarize(e2es),
        "output_throughput_tok_s": round(output_tput, 2),
        "goodput_tok_s": round(goodput_tok_s, 2),
        "slo_ttft_ms": slo_ttft_ms,
        "slo_tpot_ms": slo_tpot_ms,
    }
