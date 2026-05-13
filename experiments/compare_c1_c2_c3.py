"""三方对比图 + 表：C1 hybrid / C2 TDM / C3 chunked-prefill。

读两个 sweep summary：
  - results/qps_sweep_8k/qps_sweep_summary.json     （c1_baseline + c2_tdm）
  - results/qps_sweep_8k_c3/qps_sweep_summary.json  （c3_cp）

按 QPS 点对齐画四子图（TTFT-p99 / TPOT-p99 / Throughput / Goodput）+
打印对照表（每 QPS 一行，C1/C2/C3 三列）。

用法：
  python3 compare_c1_c2_c3.py
  python3 compare_c1_c2_c3.py --c12 path1.json --c3 path2.json --outdir <dir>

注意：python3（system）有 matplotlib，vllm venv 没有。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_C12 = Path("/vllm-workspace/Ascend-PD-TDM/results/qps_sweep_8k/qps_sweep_summary.json")
DEFAULT_C3 = Path("/vllm-workspace/Ascend-PD-TDM/results/qps_sweep_8k_c3/qps_sweep_summary.json")

CONFIG_STYLE = {
    "c1_baseline": {"label": "C1 hybrid",        "color": "#2196F3", "marker": "o"},
    "c2_tdm":      {"label": "C2 TDM (basic/M1)", "color": "#FF9800", "marker": "s"},
    "c2_tdm_m2":   {"label": "M2 SLO-adaptive",   "color": "#9C27B0", "marker": "*"},
    "c2_tdm_m21":  {"label": "M2.1 + starv-guard", "color": "#C2185B", "marker": "P"},
    "c2_tdm_m22":  {"label": "M2.2 + hysteresis", "color": "#5D4037", "marker": "X"},
    "c2_tdm_m23":  {"label": "M2.3 + backlog",    "color": "#00838F", "marker": "v"},
    "c2_tdm_m24":  {"label": "M2.4 + tpot-sat",   "color": "#1A237E", "marker": "h"},
    "c2_tdm_m25":  {"label": "M2.5 + ReLU",        "color": "#33691E", "marker": "<"},
    "c2_tdm_m26":  {"label": "M2.6 sat-tick=1",    "color": "#BF360C", "marker": ">"},
    "c2_tdm_m27":  {"label": "M2.7 sat-tick=2",    "color": "#827717", "marker": "p"},
    "c3_cp":       {"label": "C3 chunked prefill", "color": "#4CAF50", "marker": "^"},
    "c4_pd":       {"label": "C4 PD-disagg 1P1D", "color": "#E91E63", "marker": "D"},
}
ORDERED_CONFIGS = ("c1_baseline", "c2_tdm", "c2_tdm_m2", "c2_tdm_m21", "c2_tdm_m22", "c2_tdm_m23", "c2_tdm_m24", "c2_tdm_m25", "c2_tdm_m26", "c2_tdm_m27", "c3_cp", "c4_pd")

PANELS = [
    (("ttft_ms", "p99"),           "TTFT p99",          "ms"),
    (("tpot_ms_mean", "p99"),      "TPOT p99",          "ms"),
    (("output_throughput_tok_s",), "Output throughput", "tok/s"),
    (("goodput_tok_s",),           "Goodput",           "tok/s"),
]


def _get(d, path):
    cur = d
    for k in path:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _is_overloaded(s):
    w = s.get("window") or {}
    n_ok = s.get("n_ok") or 0
    n_sub = s.get("n_submitted") or 0
    n_matched = w.get("n_matched") or 0
    if n_sub > 0 and n_ok / n_sub < 0.5:
        return True
    if n_ok > 0 and n_matched == 0:
        return True
    return False


def collect_series(runs, path, skip_overload=True):
    pts = []
    for s in runs:
        if skip_overload and _is_overloaded(s):
            continue
        v = _get(s.get("window") or {}, path)
        if v is None:
            continue
        pts.append((float(s["qps_target"]), float(v)))
    pts.sort(key=lambda x: x[0])
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def merge_configs(*summaries):
    """把任意数量的 summary 的 configs 合并成单一 dict。
    冲突时先到先得（保留先出现的版本，新数据放后面以便覆盖时显式选择）。"""
    out = {}
    for src in summaries:
        cfgs = (src or {}).get("configs") or {}
        for name, runs in cfgs.items():
            if name not in out:
                out[name] = runs
    return out


def plot(merged_configs, slo_ttft, slo_tpot, out_path, title_prefix=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()
    for ax, (path, title, ylabel) in zip(axes, PANELS):
        all_ys = []
        for cfg_name in ORDERED_CONFIGS:
            runs = merged_configs.get(cfg_name)
            if not runs:
                continue
            style = CONFIG_STYLE.get(cfg_name, {"label": cfg_name})
            xs, ys = collect_series(runs, path)
            if not xs:
                continue
            ax.plot(xs, ys, label=style["label"],
                    color=style.get("color"), marker=style.get("marker"),
                    linewidth=1.8, markersize=6)
            all_ys.extend(ys)
        ax.set_title(title)
        ax.set_xlabel("Target QPS")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        # 数值范围跨 1 数量级以上自动 log scale（c4_pd ttft 比 c1 大 100-500x）
        if all_ys and min(y for y in all_ys if y > 0) > 0:
            y_min = min(y for y in all_ys if y > 0)
            y_max = max(all_ys)
            if y_max / y_min >= 30:
                ax.set_yscale("log")
        if path == ("ttft_ms", "p99") and slo_ttft is not None:
            ax.axhline(slo_ttft, linestyle="--", color="red", alpha=0.5,
                       label=f"SLO {slo_ttft:.0f}ms")
        if path == ("tpot_ms_mean", "p99") and slo_tpot is not None:
            ax.axhline(slo_tpot, linestyle="--", color="red", alpha=0.5,
                       label=f"SLO {slo_tpot:.0f}ms")
        ax.legend(fontsize=8)
    prefix = title_prefix or "8K canonical regime"
    fig.suptitle(f"{prefix}: C1 hybrid vs C2 TDM vs C3 chunked-prefill "
                 f"(Qwen3-8B, TP=2, SLO {slo_ttft:.0f}ms/{slo_tpot:.0f}ms)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _index_by_qps(runs):
    return {s["qps_target"]: s for s in (runs or [])}


def print_table(merged_configs):
    """对照表：每 QPS 一行，所有可用配置的 goodput / ttft_p99 / tpot_p99 / SLO meet%。"""
    by = {n: _index_by_qps(merged_configs.get(n, [])) for n in ORDERED_CONFIGS}
    qps_union = sorted(set().union(*[set(b.keys()) for b in by.values() if b]))
    if not qps_union:
        print("[table] 无数据")
        return

    def cell(s, path):
        if s is None or _is_overloaded(s):
            return None
        return _get(s.get("window") or {}, path)

    def slo_pct(s):
        if s is None or _is_overloaded(s):
            return None
        w = s.get("window") or {}
        n_ok = w.get("n_ok") or 0
        n_meet = w.get("n_meet_slo") or 0
        return 100.0 * n_meet / n_ok if n_ok > 0 else None

    def fmt(v, w=6, prec=1):
        if v is None:
            return f"{'-':<{w}}"
        return f"{v:<{w}.{prec}f}"

    short = {"c1_baseline": "c1", "c2_tdm": "c2", "c2_tdm_m2": "m2",
             "c2_tdm_m21": "m21", "c2_tdm_m22": "m22", "c2_tdm_m23": "m23",
             "c2_tdm_m24": "m24", "c2_tdm_m25": "m25",
             "c2_tdm_m26": "m26", "c2_tdm_m27": "m27",
             "c3_cp": "c3", "c4_pd": "c4"}
    avail = [c for c in ORDERED_CONFIGS if any(by[c])]
    if not avail:
        return
    print("\n=== 多方对照（- 表示 overloaded / 无数据）===")
    parts_hdr = [f"{'qps':<5}"]
    parts_hdr.append("|")
    parts_hdr.extend(f"{short[c]+' goodput':<11}" for c in avail)
    parts_hdr.append("|")
    parts_hdr.extend(f"{short[c]+' ttft_p99':<13}" for c in avail)
    parts_hdr.append("|")
    parts_hdr.extend(f"{short[c]+' tpot_p99':<12}" for c in avail)
    parts_hdr.append("|")
    parts_hdr.extend(f"{short[c]+' SLO%':<8}" for c in avail)
    hdr = " ".join(parts_hdr)
    print(hdr)
    print("-" * len(hdr))
    for q in qps_union:
        ss = {c: by[c].get(q) for c in avail}
        row = [f"{q:<5}", "|"]
        row.extend(fmt(cell(ss[c], ("goodput_tok_s",)), 11, 1) for c in avail)
        row.append("|")
        row.extend(fmt(cell(ss[c], ("ttft_ms", "p99")), 13, 1) for c in avail)
        row.append("|")
        row.extend(fmt(cell(ss[c], ("tpot_ms_mean", "p99")), 12, 1) for c in avail)
        row.append("|")
        row.extend(fmt(slo_pct(ss[c]), 8, 1) for c in avail)
        print(" ".join(row))


def verify_c3_story(merged_configs):
    """三方对比的核心叙事：在 8K canonical regime
       - C3 是论文的真正 SOTA 主对比
       - 全系统（最终的 M4）必须明显赢 C3，故事才立
       - 当前 C2（裸 TDM/M1）期望未必赢 C3——仅作 ablation 起点
    打印一段 narrative 说明每个 QPS 点的 C3 vs C1 / C3 vs C2 状态。"""
    by_q = {n: _index_by_qps(merged_configs.get(n, [])) for n in
            ("c1_baseline", "c2_tdm", "c3_cp")}
    qs = sorted(set().union(*[set(b.keys()) for b in by_q.values() if b]))
    print("\n=== C3 叙事检查 ===")
    for q in qs:
        s1, s2, s3 = by_q["c1_baseline"].get(q), by_q["c2_tdm"].get(q), by_q["c3_cp"].get(q)
        if s3 is None or _is_overloaded(s3):
            print(f"  qps={q}: C3 数据缺/过载")
            continue
        g3 = (s3.get("window") or {}).get("goodput_tok_s") or 0.0
        t3 = ((s3.get("window") or {}).get("ttft_ms") or {}).get("p99") or 0.0
        msgs = [f"qps={q}: C3 goodput={g3:.0f}, ttft_p99={t3:.0f}ms"]
        if s1 and not _is_overloaded(s1):
            g1 = (s1.get("window") or {}).get("goodput_tok_s") or 0.0
            t1 = ((s1.get("window") or {}).get("ttft_ms") or {}).get("p99") or 0.0
            d_g = (g3 - g1) / g1 * 100.0 if g1 > 0 else 0.0
            d_t = (t3 - t1) / t1 * 100.0 if t1 > 0 else 0.0
            msgs.append(f"vs C1: Δgoodput={d_g:+.1f}% Δttft_p99={d_t:+.1f}%")
        if s2 and not _is_overloaded(s2):
            g2 = (s2.get("window") or {}).get("goodput_tok_s") or 0.0
            t2 = ((s2.get("window") or {}).get("ttft_ms") or {}).get("p99") or 0.0
            d_g = (g3 - g2) / g2 * 100.0 if g2 > 0 else 0.0
            d_t = (t3 - t2) / t2 * 100.0 if t2 > 0 else 0.0
            msgs.append(f"vs C2: Δgoodput={d_g:+.1f}% Δttft_p99={d_t:+.1f}%")
        print("  " + " | ".join(msgs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c12", default=str(DEFAULT_C12),
                   help="C1+C2 sweep summary path")
    p.add_argument("--c3", default=str(DEFAULT_C3),
                   help="C3 sweep summary path")
    p.add_argument("--c4", default=None,
                   help="C4 PD-disagg sweep summary path（可选）")
    p.add_argument("--m2", default=None,
                   help="M2 SLO-adaptive (c2_tdm_m2) sweep summary path（可选）")
    p.add_argument("--m21", default=None,
                   help="M2.1 + starvation guard (c2_tdm_m21) sweep summary "
                   "path（可选）")
    p.add_argument("--m22", default=None,
                   help="M2.2 + hysteresis release (c2_tdm_m22) sweep summary "
                   "path（可选）")
    p.add_argument("--m23", default=None,
                   help="M2.3 + backlog-aware (c2_tdm_m23) sweep summary "
                   "path（可选）")
    p.add_argument("--m24", default=None,
                   help="M2.4 + tpot saturation detector (c2_tdm_m24) "
                   "sweep summary path（可选）")
    p.add_argument("--m25", default=None,
                   help="M2.5 + ReLU err clipping (c2_tdm_m25) sweep summary "
                   "path（可选）")
    p.add_argument("--m26", default=None,
                   help="M2.6 saturation_min_ticks=1 (c2_tdm_m26) sweep "
                   "summary path（可选）")
    p.add_argument("--m27", default=None,
                   help="M2.7 saturation_min_ticks=2 (c2_tdm_m27) sweep "
                   "summary path（可选）")
    p.add_argument("--outdir", default=None,
                   help="图保存目录，默认 c3 summary 同目录")
    p.add_argument("--title-prefix", default=None,
                   help="图标题前缀，默认 '8K canonical regime'")
    p.add_argument("--out-name", default=None,
                   help="输出 PNG 文件名（不含目录），默认 compare_c1_c2_c3.png")
    args = p.parse_args()

    c12_path = Path(args.c12)
    c3_path = Path(args.c3)
    if not c12_path.exists():
        print(f"[error] missing {c12_path}")
        return 2
    if not c3_path.exists():
        print(f"[error] missing {c3_path}\n"
              f"  先跑：python run_qps_sweep_all.py --configs c3_cp "
              f"--qps 8,16,24,32 --duration 60 --warmup 20 "
              f"--outdir /vllm-workspace/Ascend-PD-TDM/results/qps_sweep_8k_c3")
        return 2

    c12 = json.loads(c12_path.read_text())
    c3 = json.loads(c3_path.read_text())
    c4 = None
    if args.c4:
        c4_path = Path(args.c4)
        if c4_path.exists():
            c4 = json.loads(c4_path.read_text())
        else:
            print(f"[warn] --c4 {c4_path} not found, skipping C4")
    m2 = None
    if args.m2:
        m2_path = Path(args.m2)
        if m2_path.exists():
            m2 = json.loads(m2_path.read_text())
        else:
            print(f"[warn] --m2 {m2_path} not found, skipping M2")
    m21 = None
    if args.m21:
        m21_path = Path(args.m21)
        if m21_path.exists():
            m21 = json.loads(m21_path.read_text())
        else:
            print(f"[warn] --m21 {m21_path} not found, skipping M2.1")
    m22 = None
    if args.m22:
        m22_path = Path(args.m22)
        if m22_path.exists():
            m22 = json.loads(m22_path.read_text())
        else:
            print(f"[warn] --m22 {m22_path} not found, skipping M2.2")
    m23 = None
    if args.m23:
        m23_path = Path(args.m23)
        if m23_path.exists():
            m23 = json.loads(m23_path.read_text())
        else:
            print(f"[warn] --m23 {m23_path} not found, skipping M2.3")
    m24 = None
    if args.m24:
        m24_path = Path(args.m24)
        if m24_path.exists():
            m24 = json.loads(m24_path.read_text())
        else:
            print(f"[warn] --m24 {m24_path} not found, skipping M2.4")
    m25 = None
    if args.m25:
        m25_path = Path(args.m25)
        if m25_path.exists():
            m25 = json.loads(m25_path.read_text())
        else:
            print(f"[warn] --m25 {m25_path} not found, skipping M2.5")
    m26 = None
    if args.m26:
        m26_path = Path(args.m26)
        if m26_path.exists():
            m26 = json.loads(m26_path.read_text())
        else:
            print(f"[warn] --m26 {m26_path} not found, skipping M2.6")
    m27 = None
    if args.m27:
        m27_path = Path(args.m27)
        if m27_path.exists():
            m27 = json.loads(m27_path.read_text())
        else:
            print(f"[warn] --m27 {m27_path} not found, skipping M2.7")
    merged = merge_configs(c12, c3, c4, m2, m21, m22, m23, m24, m25, m26, m27)

    slo_ttft = c12.get("slo_ttft_ms") or c3.get("slo_ttft_ms") or 500.0
    slo_tpot = c12.get("slo_tpot_ms") or c3.get("slo_tpot_ms") or 50.0

    out_dir = Path(args.outdir) if args.outdir else c3_path.parent
    out_name = args.out_name or "compare_c1_c2_c3.png"
    out_path = out_dir / out_name
    plot(merged, slo_ttft, slo_tpot, out_path, title_prefix=args.title_prefix)
    print_table(merged)
    verify_c3_story(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
