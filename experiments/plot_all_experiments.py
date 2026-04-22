"""
Generate all experiment visualizations for advisor meeting.
Outputs PNG charts to /vllm-workspace/lzn-pro/figures/
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIGDIR = "/vllm-workspace/lzn-pro/figures"
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']


def load(name):
    for prefix in ['/vllm-workspace/lzn-pro/', '/vllm-workspace/lzn-pro/result/']:
        p = prefix + name
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(name)


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Exp B — P/D Resource Utilization Profile (Bar chart)
# ═══════════════════════════════════════════════════════════════════
def plot_exp_b():
    # Use data from notes (dcmi v2 had 0 readings due to sampling issue, use notes data)
    scenarios = ['Prefill-heavy', 'Decode-heavy', 'Mixed']
    aicore = [69.8, 42.6, 33.9]
    hbm_bw = [15.8, 25.1, 26.1]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scenarios))
    w = 0.32
    bars1 = ax.bar(x - w/2, aicore, w, label='AICore Utilization %', color=COLORS[0])
    bars2 = ax.bar(x + w/2, hbm_bw, w, label='HBM Bandwidth Utilization %', color=COLORS[1])

    ax.set_ylabel('Utilization (%)')
    ax.set_title('Exp B: NPU Resource Utilization — Prefill vs Decode')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0, 85)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    # Add annotation
    ax.annotate('Compute-bound\n(1.64× AICore)', xy=(0, 69.8), xytext=(0.6, 78),
                fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Memory-bound\n(1.59× HBM BW)', xy=(1.16, 25.1), xytext=(1.8, 45),
                fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

    fig.savefig(f'{FIGDIR}/fig1_exp_b_resource_profile.png')
    plt.close(fig)
    print("  fig1_exp_b_resource_profile.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Exp D — Throughput & TPOT vs Concurrency
# ═══════════════════════════════════════════════════════════════════
def plot_exp_d():
    d = load('results_exp_d_4b.json')
    results = d['results']
    conc = [r['num_concurrent'] for r in results]
    tps = [r['throughput_tps'] for r in results]
    tpot = [r['avg_tpot_ms'] for r in results]
    ttft = [r['avg_ttft_ms'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: TPS vs concurrency
    ax1.plot(conc, tps, 'o-', color=COLORS[0], linewidth=2, markersize=6)
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('Throughput Scales Near-Linearly')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(conc)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Add ideal linear reference
    ideal = [tps[0] * c for c in conc]
    ax1.plot(conc, ideal, '--', color='gray', alpha=0.5, label='Ideal linear')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: TPOT stays flat
    ax2.plot(conc, tpot, 's-', color=COLORS[1], linewidth=2, markersize=6, label='TPOT')
    ax2.plot(conc, ttft, '^-', color=COLORS[2], linewidth=2, markersize=6, label='TTFT')
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('TPOT Stays Flat — NPU Underutilized at Low Concurrency')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(conc)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 35)

    # Shade "TDM opportunity zone"
    ax2.axvspan(0.8, 8, alpha=0.08, color='red', label='TDM opportunity')
    ax2.text(2, 32, 'TDM Opportunity\n(NPU underutilized)', fontsize=8,
             ha='center', color='red', alpha=0.7)

    fig.suptitle('Exp D: Concurrency Scaling — Qwen3-4B, Single 910B3', fontsize=13, y=1.02)
    fig.savefig(f'{FIGDIR}/fig2_exp_d_concurrency.png')
    plt.close(fig)
    print("  fig2_exp_d_concurrency.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Exp A — ACL Graph Compiled Sizes & Eager Boundary
# ═══════════════════════════════════════════════════════════════════
def plot_exp_a():
    d = load('results_exp_a_v2.json')
    sizes = d['test1_config']['compiled_sizes']
    eager = d['test3_eager_boundary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: compiled sizes distribution
    gaps = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
    ax1.bar(range(len(gaps)), gaps, color=COLORS[0], alpha=0.8)
    ax1.set_xlabel('Shape Index')
    ax1.set_ylabel('Gap to Next Shape')
    ax1.set_title(f'{len(sizes)} Compiled Shapes — Gaps Vary Widely')
    ax1.axhline(y=np.mean(gaps), color='red', linestyle='--', label=f'Mean gap={np.mean(gaps):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: eager boundary performance
    bs_list = [r['batch_size'] for r in eager]
    per_tok = [r.get('per_tok_ms', r.get('per_token_ms', 0)) for r in eager]
    modes = [r['mode'] for r in eager]
    colors_pts = [COLORS[0] if m == 'GRAPH' else COLORS[3] for m in modes]

    ax2.bar(range(len(bs_list)), per_tok, color=colors_pts)
    ax2.set_xticks(range(len(bs_list)))
    ax2.set_xticklabels(bs_list, rotation=45)
    ax2.set_xlabel('Batch Size (total tokens)')
    ax2.set_ylabel('Per-Token Latency (ms)')
    ax2.set_title('Graph→Eager Fallback: +26% Latency at bs>512')

    # Legend
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=COLORS[0], label='GRAPH mode'),
                        Patch(color=COLORS[3], label='EAGER mode')])
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Exp A: ACL Graph Behavior — Qwen3-4B, 910B3', fontsize=13, y=1.02)
    fig.savefig(f'{FIGDIR}/fig3_exp_a_acl_graph.png')
    plt.close(fig)
    print("  fig3_exp_a_acl_graph.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Exp F — Graph-Aware Batch Shaping
# ═══════════════════════════════════════════════════════════════════
def plot_exp_f():
    d = load('results_exp_f.json')

    # Test 1: Waste profile
    waste = d['test1_waste_profile']['segment_stats']
    segments = list(waste.keys())
    avg_waste = [waste[s]['avg_waste_pct'] for s in segments]
    max_waste = [waste[s]['max_waste_pct'] for s in segments]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(segments))
    w = 0.35
    ax1.bar(x - w/2, avg_waste, w, label='Avg Waste %', color=COLORS[0])
    ax1.bar(x + w/2, max_waste, w, label='Max Waste %', color=COLORS[3], alpha=0.7)
    ax1.set_xticks(x)
    short_labels = [s.split('(')[0].strip() for s in segments]
    ax1.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Padding Waste (%)')
    ax1.set_title('Padding Waste by Batch Size Range')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test 3: TDM simulation naive vs graph-aware
    tdm = d['test3_tdm_simulation']
    scenarios = [r['scenario'] for r in tdm]
    naive_tps = [r['naive_tps'] for r in tdm]
    aware_tps = [r['aware_tps'] for r in tdm]

    x2 = np.arange(len(scenarios))
    ax2.bar(x2 - w/2, naive_tps, w, label='Naive Batching', color=COLORS[0])
    ax2.bar(x2 + w/2, aware_tps, w, label='Graph-Aware', color=COLORS[2])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8)
    ax2.set_ylabel('Throughput (tok/s)')
    ax2.set_title('Naive vs Graph-Aware Scheduling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add speedup labels
    for i, (n, a) in enumerate(zip(naive_tps, aware_tps)):
        pct = (a/n - 1) * 100
        color = 'green' if pct > 0 else 'red'
        ax2.text(i + w/2, a + 15, f'{pct:+.1f}%', ha='center', fontsize=9, color=color)

    fig.suptitle('Exp F: Graph-Aware Batch Shaping Analysis', fontsize=13, y=1.02)
    fig.savefig(f'{FIGDIR}/fig4_exp_f_graph_aware.png')
    plt.close(fig)
    print("  fig4_exp_f_graph_aware.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Exp C — DCMI Sampling Overhead
# ═══════════════════════════════════════════════════════════════════
def plot_exp_c():
    d = load('results_exp_c_4b_v2.json')
    results = d['results']

    labels = [r['label'] for r in results]
    tps = [r['throughput_tps'] for r in results]
    overhead = [r['overhead_pct'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = [COLORS[2] if o < 1 else COLORS[1] for o in overhead]
    bars = ax.bar(range(len(labels)), tps, color=colors_bar)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
    ax.set_ylabel('Throughput (tok/s)')
    ax.set_title('Exp C: DCMI Online Sampling Overhead')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, o) in enumerate(zip(bars, overhead)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{o:.1f}%' if o > 0 else 'baseline',
                ha='center', fontsize=9, color='red' if o > 1 else 'green')

    ax.set_ylim(0, max(tps) * 1.12)
    fig.savefig(f'{FIGDIR}/fig5_exp_c_dcmi_overhead.png')
    plt.close(fig)
    print("  fig5_exp_c_dcmi_overhead.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 6: Exp E3 — Fair 2-Card TPS Comparison (grouped bar)
# ═══════════════════════════════════════════════════════════════════
def plot_exp_e3():
    d = load('results_exp_e3.json')
    conds = d['conditions']

    wl_names = ['short_4req', 'short_16req', 'short_32req',
                'long_4req', 'long_16req', 'mixed_16req']

    # Build TPS matrix
    cond_names = ['unified_tp2', 'unified_tp2_cp', 'phased_tp2', 'disagg_1p1d']
    cond_labels = ['Unified TP=2', 'Unified TP=2\n+ChunkedPrefill', 'Phased TP=2', 'Disagg 1P1D']

    tps_matrix = []
    for cond in cond_names:
        results = conds.get(cond, [])
        tps_row = []
        for wl in wl_names:
            val = 0
            for r in results:
                if wl in r['label']:
                    val = r['throughput_tps']
                    break
            tps_row.append(val)
        tps_matrix.append(tps_row)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(wl_names))
    n = len(cond_names)
    w = 0.19

    for i, (tps_row, label) in enumerate(zip(tps_matrix, cond_labels)):
        bars = ax.bar(x + (i - n/2 + 0.5) * w, tps_row, w,
                      label=label, color=COLORS[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([wl.replace('_', '\n') for wl in wl_names])
    ax.set_ylabel('Throughput (tok/s)')
    ax.set_title('Exp E3: Fair 2-Card Comparison — All Methods Use 2×910B3')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(f'{FIGDIR}/fig6_exp_e3_fair_compare.png')
    plt.close(fig)
    print("  fig6_exp_e3_fair_compare.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 7: Exp E3 — TTFT Comparison
# ═══════════════════════════════════════════════════════════════════
def plot_exp_e3_ttft():
    d = load('results_exp_e3.json')
    conds = d['conditions']

    wl_names = ['short_4req', 'short_16req', 'short_32req',
                'long_4req', 'long_16req', 'mixed_16req']
    cond_names = ['unified_tp2', 'unified_tp2_cp', 'phased_tp2', 'disagg_1p1d']
    cond_labels = ['Unified TP=2', 'Unified+CP', 'Phased TP=2', 'Disagg 1P1D']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # TTFT
    for i, cond in enumerate(cond_names):
        results = conds.get(cond, [])
        ttft_row = []
        for wl in wl_names:
            for r in results:
                if wl in r['label']:
                    ttft_row.append(r['avg_ttft_ms'])
                    break
        ax1.plot(range(len(wl_names)), ttft_row, 'o-', color=COLORS[i],
                 label=cond_labels[i], linewidth=2, markersize=5)

    ax1.set_xticks(range(len(wl_names)))
    ax1.set_xticklabels([wl.replace('_', '\n') for wl in wl_names], fontsize=8)
    ax1.set_ylabel('TTFT (ms)')
    ax1.set_title('Time to First Token')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # TPOT
    for i, cond in enumerate(cond_names):
        results = conds.get(cond, [])
        tpot_row = []
        for wl in wl_names:
            for r in results:
                if wl in r['label']:
                    tpot_row.append(r['avg_tpot_ms'])
                    break
        ax2.plot(range(len(wl_names)), tpot_row, 's-', color=COLORS[i],
                 label=cond_labels[i], linewidth=2, markersize=5)

    ax2.set_xticks(range(len(wl_names)))
    ax2.set_xticklabels([wl.replace('_', '\n') for wl in wl_names], fontsize=8)
    ax2.set_ylabel('TPOT (ms)')
    ax2.set_title('Time Per Output Token')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Exp E3: Latency Comparison — 2×910B3', fontsize=13, y=1.02)
    fig.savefig(f'{FIGDIR}/fig7_exp_e3_latency.png')
    plt.close(fig)
    print("  fig7_exp_e3_latency.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 8: Summary — TDM Architecture Overview (concept diagram)
# ═══════════════════════════════════════════════════════════════════
def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('TDM System Architecture — Three-Layer Design', fontsize=14, fontweight='bold', pad=15)

    # Layer boxes
    layers = [
        (1, 4.8, 3.8, 1.6, '#E3F2FD', 'Layer 1\nSLO-Aware\nPhase Switch',
         '• Monitor TTFT/TPOT SLO\n• Dynamic P↔D ratio\n• Priority scheduling'),
        (5.1, 4.8, 3.8, 1.6, '#FFF3E0', 'Layer 2\nGraph-Aware\nBatch Shaping',
         '• Align to ACL compiled shapes\n• "Round-up" not "round-down"\n• Reduce padding waste 23%→<5%'),
        (9.2, 4.8, 3.8, 1.6, '#E8F5E9', 'Layer 3\nAdaptive\nIntrospection',
         '• DCMI AICore sampling (~0%)\n• Offline profiling table\n• Phase transition triggers'),
    ]

    for x, y, w, h, color, title, desc in layers:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333',
                              linewidth=1.5, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='top',
                fontsize=10, fontweight='bold', zorder=3)
        ax.text(x + 0.15, y + 0.1, desc, ha='left', va='bottom',
                fontsize=7.5, family='monospace', zorder=3)

    # Arrows between layers
    for x1, x2 in [(4.8, 5.1), (8.9, 9.2)]:
        ax.annotate('', xy=(x2, 5.6), xytext=(x1, 5.6),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=1.5))

    # Bottom: Data flow
    ax.text(7, 4.3, 'Data-Driven Evidence (Experiments A–F)', ha='center',
            fontsize=11, fontweight='bold', color='#333')

    evidence = [
        (1.5, 2.5, 'Exp B: P/D\nResource\nComplementarity\nAICore 1.64×\nHBM BW 1.59×'),
        (4.0, 2.5, 'Exp D: TPOT\nFlat to 64×\nConcurrency\n→ NPU idle at\nlow concurrency'),
        (6.5, 2.5, 'Exp A/F: ACL\nGraph 48 shapes\n23% waste at\nsmall batch\n→ shape-aware'),
        (9.0, 2.5, 'Exp C: DCMI\nAICore ~0%\nHBM BW ~8%\n→ offline+online\nhybrid strategy'),
        (11.5, 2.5, 'Exp E3: Fair\n2-card compare\nCP +5~11%\nDisagg -2~12%\n→ TDM target'),
    ]

    for x, y, text in evidence:
        rect = plt.Rectangle((x - 0.9, y - 0.9), 2.2, 2.0, facecolor='#FAFAFA',
                              edgecolor='#999', linewidth=1, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.2, y, text, ha='center', va='center', fontsize=7, zorder=3)

    # Arrows from evidence to layers
    for ex, lx in [(1.5, 2.9), (4.0, 2.9), (6.5, 7.0), (9.0, 11.1), (11.5, 11.1)]:
        ax.annotate('', xy=(lx, 4.8), xytext=(ex + 0.2, 3.6),
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=1, ls='--'))

    fig.savefig(f'{FIGDIR}/fig8_architecture_overview.png')
    plt.close(fig)
    print("  fig8_architecture_overview.png")


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures...")
    plot_exp_b()
    plot_exp_d()
    plot_exp_a()
    plot_exp_f()
    plot_exp_c()
    plot_exp_e3()
    plot_exp_e3_ttft()
    plot_architecture()
    print(f"\nAll figures saved to {FIGDIR}/")
