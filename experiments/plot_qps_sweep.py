"""Phase 1 出图 + 期望验证：

读取 `run_qps_sweep_all.py` 写出的 `qps_sweep_summary.json`，画四子图
（TTFT-p99 / TPOT-p99 / Throughput / Goodput vs QPS），并对核心假设做
量化检查：

  H1（低 QPS 重合）：最低 QPS 点 c2_tdm 与 c1_baseline 的 goodput 相对差 < 15%。
    含义：TDM 不引入显著常态开销。
  H2（饱和退化更慢）：最高 QPS 点 c2_tdm 的 goodput 不低于 c1_baseline，
    且 TTFT-p99 不更差（允许等式以容忍噪声）。
    含义：在临近饱和工况下，TDM 切片比 hybrid 退化更平滑。

用法：
  python plot_qps_sweep.py
  python plot_qps_sweep.py --summary <path>.json --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SUMMARY = Path("/vllm-workspace/Ascend-PD-TDM/results/qps_sweep/qps_sweep_summary.json")

CONFIG_STYLE = {
    "c1_baseline": {"label": "C1 hybrid (baseline)", "color": "#2196F3", "marker": "o"},
    "c2_tdm":      {"label": "C2 TDM",                "color": "#FF9800", "marker": "s"},
}

# (key path into window, label, ylabel)
PANELS = [
    (("ttft_ms", "p99"),       "TTFT p99",        "ms"),
    (("tpot_ms_mean", "p99"),  "TPOT p99",        "ms"),
    (("output_throughput_tok_s",), "Output throughput", "tok/s"),
    (("goodput_tok_s",),       "Goodput",         "tok/s"),
]


def _get(d: dict, path: tuple[str, ...]):
    cur = d
    for k in path:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _is_overloaded(s: dict) -> bool:
    """判定该 QPS 点是否 server 端过载——表征：tracker 完全没匹配上，
    或客户端错误率 > 50%。这种点应从曲线剔除（它反映的是测量崩溃而非系统能力）。"""
    w = s.get("window") or {}
    n_ok = s.get("n_ok") or 0
    n_sub = s.get("n_submitted") or 0
    n_matched = w.get("n_matched") or 0
    if n_sub > 0 and n_ok / n_sub < 0.5:
        return True
    if n_ok > 0 and n_matched == 0:
        return True
    return False


def collect_series(runs: list[dict], path: tuple[str, ...],
                   skip_overload: bool = True) -> tuple[list[float], list[float]]:
    """从一组 per-QPS summary 抽 (qps, value) 序列，按 qps 升序。
    skip_overload=True 时跳过 server 过载的点（goodput=0 / tracker 全丢）。"""
    pts = []
    for s in runs:
        if skip_overload and _is_overloaded(s):
            continue
        w = s.get("window") or {}
        v = _get(w, path)
        if v is None:
            continue
        pts.append((float(s["qps_target"]), float(v)))
    pts.sort(key=lambda x: x[0])
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def plot(summary: dict, out_path: Path) -> None:
    configs: dict[str, list[dict]] = summary.get("configs") or {}
    slo_ttft = summary.get("slo_ttft_ms")
    slo_tpot = summary.get("slo_tpot_ms")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()

    for ax, (path, title, ylabel) in zip(axes, PANELS):
        for cfg_name, runs in configs.items():
            style = CONFIG_STYLE.get(cfg_name, {"label": cfg_name})
            xs, ys = collect_series(runs, path)
            if not xs:
                continue
            ax.plot(xs, ys, label=style["label"],
                    color=style.get("color"), marker=style.get("marker"),
                    linewidth=1.8, markersize=6)
        ax.set_title(title)
        ax.set_xlabel("Target QPS")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # SLO 参考线
        if path == ("ttft_ms", "p99") and slo_ttft is not None:
            ax.axhline(slo_ttft, linestyle="--", color="red", alpha=0.5,
                       label=f"SLO {slo_ttft:.0f}ms")
        if path == ("tpot_ms_mean", "p99") and slo_tpot is not None:
            ax.axhline(slo_tpot, linestyle="--", color="red", alpha=0.5,
                       label=f"SLO {slo_tpot:.0f}ms")
        ax.legend(fontsize=9)

    fig.suptitle("Phase 1: C1 hybrid vs C2 TDM — QPS sweep "
                 f"(Qwen3-8B, TP=2, SLO {slo_ttft:.0f}ms/{slo_tpot:.0f}ms)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def verify_expectations(summary: dict, low_overlap_pct: float = 15.0) -> None:
    configs = summary.get("configs") or {}
    if "c1_baseline" not in configs or "c2_tdm" not in configs:
        print("[verify] 跳过：缺 c1_baseline 或 c2_tdm")
        return
    c1 = sorted(configs["c1_baseline"], key=lambda s: s["qps_target"])
    c2 = sorted(configs["c2_tdm"],      key=lambda s: s["qps_target"])
    if not c1 or not c2:
        print("[verify] 跳过：某 config 没有数据点")
        return

    print("\n=== 期望验证 ===")
    # 对照表（按 QPS 对齐）
    qps_set = sorted(set(s["qps_target"] for s in c1) & set(s["qps_target"] for s in c2))
    if not qps_set:
        print("[verify] 两 config 的 QPS 点无交集，跳过")
        return

    print(f"{'qps':<6} {'c1_goodput':<12} {'c2_goodput':<12} {'Δ%':<8} "
          f"{'c1_ttft_p99':<12} {'c2_ttft_p99':<12}")
    by_qps = {"c1": {s["qps_target"]: s for s in c1},
              "c2": {s["qps_target"]: s for s in c2}}
    for q in qps_set:
        s1 = by_qps["c1"][q]; s2 = by_qps["c2"][q]
        g1 = (s1.get("window") or {}).get("goodput_tok_s") or 0.0
        g2 = (s2.get("window") or {}).get("goodput_tok_s") or 0.0
        t1 = ((s1.get("window") or {}).get("ttft_ms") or {}).get("p99")
        t2 = ((s2.get("window") or {}).get("ttft_ms") or {}).get("p99")
        d = (g2 - g1) / g1 * 100.0 if g1 > 0 else float("nan")
        print(f"{q:<6} {g1:<12.2f} {g2:<12.2f} {d:<+8.2f} "
              f"{(t1 if t1 is not None else float('nan')):<12.2f} "
              f"{(t2 if t2 is not None else float('nan')):<12.2f}")

    # H1：最低 QPS 点 goodput 相对差 < 阈值
    q_lo = qps_set[0]
    g1_lo = (by_qps["c1"][q_lo].get("window") or {}).get("goodput_tok_s") or 0.0
    g2_lo = (by_qps["c2"][q_lo].get("window") or {}).get("goodput_tok_s") or 0.0
    rel_lo = abs(g2_lo - g1_lo) / g1_lo * 100.0 if g1_lo > 0 else float("inf")
    h1_ok = rel_lo < low_overlap_pct
    print(f"\nH1 低 QPS 重合（qps={q_lo}）: |Δgoodput|={rel_lo:.2f}% "
          f"{'✓ <' if h1_ok else '✗ ≥'} {low_overlap_pct:.0f}%  "
          f"(c1={g1_lo:.1f}, c2={g2_lo:.1f} tok/s)")

    # H2：取「两 config 都未过载」的最高 QPS 点比较——
    # 只有 0/0 或 server 崩溃的点不该作为 saturation 证据
    valid_q = [q for q in qps_set
               if not _is_overloaded(by_qps["c1"][q])
               and not _is_overloaded(by_qps["c2"][q])]
    if not valid_q:
        print("H2 饱和退化更慢: 跳过——所有 QPS 点两 config 都过载")
        return
    q_hi = valid_q[-1]
    g1_hi = (by_qps["c1"][q_hi].get("window") or {}).get("goodput_tok_s") or 0.0
    g2_hi = (by_qps["c2"][q_hi].get("window") or {}).get("goodput_tok_s") or 0.0
    t1_hi = ((by_qps["c1"][q_hi].get("window") or {}).get("ttft_ms") or {}).get("p99")
    t2_hi = ((by_qps["c2"][q_hi].get("window") or {}).get("ttft_ms") or {}).get("p99")
    g_ok = g2_hi >= g1_hi
    t_ok = (t1_hi is None or t2_hi is None) or (t2_hi <= t1_hi)
    h2_ok = g_ok and t_ok
    print(f"H2 饱和退化更慢（最高有效 qps={q_hi}）: "
          f"goodput c2{'≥' if g_ok else '<'}c1 ({g2_hi:.1f} vs {g1_hi:.1f}); "
          f"ttft_p99 c2{'≤' if t_ok else '>'}c1 "
          f"({(t2_hi if t2_hi is not None else float('nan')):.1f} vs "
          f"{(t1_hi if t1_hi is not None else float('nan')):.1f})  "
          f"=> {'✓' if h2_ok else '✗'}")
    # 提示用户：如有过载点说明 sweep 范围超过饱和——下次应在 [valid_q[-1], 第一个过载 q] 之间加密
    overloaded = [q for q in qps_set if q not in valid_q]
    if overloaded:
        print(f"  注意：qps={overloaded} 处两 config 至少一边过载（server 崩溃 / "
              f"tracker 全丢），这些点不是真实 goodput=0，仅说明 sweep 范围超过了"
              f"系统饱和点。建议下轮在 ({q_hi}, {overloaded[0]}) 之间加密 QPS。")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    p.add_argument("--outdir", default=None,
                   help="图保存目录，默认 summary 同目录")
    p.add_argument("--low-overlap-pct", type=float, default=15.0,
                   help="H1 低 QPS 重合阈值（goodput 相对差），默认 15%%")
    args = p.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[error] summary 不存在：{summary_path}\n"
              f"  先跑：python run_qps_sweep_all.py")
        return 2
    summary = json.loads(summary_path.read_text())
    out_dir = Path(args.outdir) if args.outdir else summary_path.parent
    out_path = out_dir / "qps_sweep.png"

    plot(summary, out_path)
    verify_expectations(summary, low_overlap_pct=args.low_overlap_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
