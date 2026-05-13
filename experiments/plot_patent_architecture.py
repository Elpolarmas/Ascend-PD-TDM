"""
Render the patent architecture diagram (three-layer SLO-Adaptive TDM).
Layer 1 = core decision; Layer 2 = advisory; Layer 3 = hardware introspection.
Outputs PNG files to /vllm-workspace/lzn-pro/figures/.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGDIR = "/vllm-workspace/Ascend-PD-TDM/figures"
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 160,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

C_L1 = '#E8F1FF'; E_L1 = '#1F4FB0'
C_L2 = '#EFF8EA'; E_L2 = '#2E7D32'
C_L3 = '#FFF4E0'; E_L3 = '#C77700'
C_EP = '#F5F5F5'; E_EP = '#555555'


def round_box(ax, x, y, w, h, face, edge, lw=1.6, pad=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2,
    )
    ax.add_patch(box)


def arrow(ax, xy_from, xy_to, color='#444', lw=1.7, style='->', ls='-', curve=0.0):
    a = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, color=color, lw=lw, linestyle=ls,
        mutation_scale=16, connectionstyle=f"arc3,rad={curve}", zorder=3,
    )
    ax.add_patch(a)


def plot_architecture_main():
    """Main architecture diagram — Layer 1 internally split into
    iteration-scale (solid) and window-scale (dashed) sub-bands so the
    two-timescale closed loop is explicit at the §3.1 framework level."""
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.5, 7.6); ax.axis('off')
    ax.set_title(
        'SLO-Adaptive P/D Time-Division Multiplexing — Three-Layer Architecture',
        fontsize=13, fontweight='bold', pad=14,
    )

    DASHED = (0, (5, 3))

    # Source / sink boxes (top row)
    round_box(ax, 0.6, 6.20, 2.8, 0.65, C_EP, E_EP)
    ax.text(2.0, 6.525, 'Incoming Requests',
            ha='center', va='center', fontsize=10, fontweight='bold')

    round_box(ax, 10.3, 6.20, 2.5, 0.65, C_EP, E_EP)
    ax.text(11.55, 6.525, 'Execution Engine',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Layer 1 — taller container so the two-timescale split is visible
    L1_X, L1_Y, L1_W, L1_H = 3.85, 3.20, 4.95, 3.50
    round_box(ax, L1_X, L1_Y, L1_W, L1_H, C_L1, E_L1, lw=2.1)
    ax.text(L1_X + L1_W / 2, L1_Y + L1_H - 0.30,
            'Layer 1  —  SLO-Adaptive Control',
            ha='center', va='center', fontsize=12.3, fontweight='bold', color=E_L1)

    # ─── Iter-scale sub-band (upper) ──────────────────────────────
    iter_y0 = L1_Y + 1.85
    iter_h  = 1.20
    round_box(ax, L1_X + 0.18, iter_y0, L1_W - 0.36, iter_h,
              '#FFFFFF', '#6A8FCF', lw=1.0, pad=0.025)
    ax.text(L1_X + 0.30, iter_y0 + iter_h - 0.18,
            'Iteration scale  ·  every iteration',
            ha='left', va='top', fontsize=9.6, fontweight='bold',
            color=E_L1, style='italic')
    # mini illustration: state → decision → mode+batch (solid)
    ax.text(L1_X + L1_W / 2, iter_y0 + iter_h / 2 - 0.05,
            'state  +  advice  +  ratio   →   { mode , batch }',
            ha='center', va='center', fontsize=10.3, color='#1d1d1d')
    arrow(ax,
          (L1_X + 0.55, iter_y0 + 0.20),
          (L1_X + L1_W - 0.55, iter_y0 + 0.20),
          color=E_L1, lw=1.4, style='->')

    # ─── Window-scale sub-band (lower) ────────────────────────────
    win_y0 = L1_Y + 0.30
    win_h  = 1.30
    round_box(ax, L1_X + 0.18, win_y0, L1_W - 0.36, win_h,
              '#FFFFFF', '#6A8FCF', lw=1.0, pad=0.025)
    ax.text(L1_X + 0.30, win_y0 + win_h - 0.18,
            'Window scale  ·  every window (W iters)',
            ha='left', va='top', fontsize=9.6, fontweight='bold',
            color=E_L1, style='italic')
    ax.text(L1_X + L1_W / 2, win_y0 + win_h / 2 - 0.05,
            'SLO attainment   →   P/D ratio param update\n(smoothed, hard-bound clipped)',
            ha='center', va='center', fontsize=10.0, color='#1d1d1d')
    arrow(ax,
          (L1_X + 0.55, win_y0 + 0.22),
          (L1_X + L1_W - 0.55, win_y0 + 0.22),
          color=E_L1, lw=1.4, style='->', ls=DASHED)

    # Internal vertical: window → iter (P/D ratio param feeds decision)
    arrow(ax, (L1_X + L1_W - 0.55, win_y0 + win_h - 0.05),
              (L1_X + L1_W - 0.55, iter_y0 + 0.05),
          color='#444', lw=1.5, style='->')
    ax.text(L1_X + L1_W - 0.40, (win_y0 + win_h + iter_y0) / 2 - 0.05,
            'P/D ratio\nparam',
            ha='left', va='center', fontsize=8.6, color='#444', style='italic')

    # ─── Layer 2 / Layer 3 (bottom row) ───────────────────────────
    round_box(ax, 3.3, 0.55, 3.5, 2.10, C_L2, E_L2, lw=2.1)
    ax.text(5.05, 2.05, 'Layer 2',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color=E_L2)
    ax.text(5.05, 1.60, 'Graph-Recognition Module',
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=E_L2)
    ax.text(5.05, 1.05, 'advisor  ·  shape-alignment\nsuggestions to Layer 1',
            ha='center', va='center', fontsize=9, style='italic', color='#222')

    round_box(ax, 7.8, 0.55, 3.5, 2.10, C_L3, E_L3, lw=2.1)
    ax.text(9.55, 2.05, 'Layer 3',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color=E_L3)
    ax.text(9.55, 1.60, 'State-Sampling Module',
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=E_L3)
    ax.text(9.55, 1.05, 'introspection  ·  offline profile\n+ online low-freq sampling',
            ha='center', va='center', fontsize=9, style='italic', color='#222')

    # ─── Outer flows ──────────────────────────────────────────────
    # Requests → Layer 1 (iter band, solid)
    arrow(ax, (3.4, 6.525), (L1_X, iter_y0 + iter_h * 0.55),
          color='#444', lw=1.7, style='->', curve=-0.05)

    # Layer 1 → Engine (mode + batch, solid, iter scale)
    arrow(ax, (L1_X + L1_W, iter_y0 + iter_h * 0.55), (10.3, 6.525),
          color=E_L1, lw=2.1, style='->', curve=0.05)
    ax.text((L1_X + L1_W + 10.3) / 2, iter_y0 + iter_h + 0.45,
            'mode + batch',
            ha='center', fontsize=9.2, color=E_L1, fontweight='bold')

    # Engine → Layer 1 (window-scale feedback, dashed)
    arrow(ax, (11.55, 6.20), (11.55, win_y0 + win_h * 0.50),
          color='#666', lw=1.5, style='->', ls=DASHED)
    arrow(ax, (11.55, win_y0 + win_h * 0.50),
              (L1_X + L1_W, win_y0 + win_h * 0.50),
          color='#666', lw=1.5, style='->', ls=DASHED)
    ax.text(11.75, (6.20 + win_y0 + win_h * 0.50) / 2,
            'SLO attainment\n(per-iter outcomes\naggregated\nover window)',
            ha='left', va='center', fontsize=8.3, color='#555', style='italic')

    # L2 → Layer 1 iter band (advice, solid — queried per iter).
    # Route around L1's left side so the arrow does not cross the window band.
    arrow(ax, (3.30, 2.65), (L1_X, iter_y0 + iter_h * 0.40),
          color=E_L2, lw=1.7, style='->', curve=-0.32)
    ax.text(2.30, (2.65 + iter_y0) / 2 + 0.40, 'batch-shape\nadvice',
            ha='center', fontsize=9, color=E_L2, fontweight='bold')

    # L3 → Layer 1 iter band (hardware state, solid).
    # Route around L1's right side, also avoids the dashed Engine→L1 feedback
    # which enters the window band lower down.
    arrow(ax, (11.30, 2.65), (L1_X + L1_W, iter_y0 + iter_h * 0.40),
          color=E_L3, lw=1.7, style='->', curve=0.32)
    ax.text(12.30, (2.65 + iter_y0) / 2 + 0.40, 'hardware\nstate',
            ha='center', fontsize=9, color=E_L3, fontweight='bold')

    # L3 → L2 shape-cost profile
    arrow(ax, (7.8, 1.60), (6.8, 1.60), color=E_L3, lw=1.7, style='->')
    ax.text(7.30, 1.85, 'shape-cost profile',
            ha='center', fontsize=8.7, color=E_L3, fontweight='bold')

    # ─── Line-style legend (bottom) ───────────────────────────────
    leg_y = -0.20
    ax.plot([0.6, 1.55], [leg_y, leg_y], '-', color='#444', lw=1.7)
    ax.text(1.65, leg_y, 'solid  =  iteration-scale path  (fires per iteration)',
            ha='left', va='center', fontsize=9.2, color='#222')
    ax.plot([6.4, 7.35], [leg_y, leg_y], color='#444', lw=1.7, ls=DASHED)
    ax.text(7.45, leg_y,
            'dashed  =  window-scale path  (fires per sliding window)',
            ha='left', va='center', fontsize=9.2, color='#222')

    fig.savefig(f'{FIGDIR}/patent_architecture_main.png')
    plt.close(fig)
    print("  patent_architecture_main.png")


def plot_layer1_twoscale():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis('off')
    ax.set_title(
        'Layer 1 Internal View — Two-Timescale Closed-Loop Control',
        fontsize=13, fontweight='bold', pad=14,
    )

    # Iter scale (left)
    round_box(ax, 0.4, 0.7, 6.2, 5.0, '#E8F1FF', '#1F4FB0', lw=2.0)
    ax.text(3.5, 5.42, 'Iteration Scale  —  fires every iteration',
            ha='center', fontsize=11.5, fontweight='bold', color='#1F4FB0')

    iter_steps = [
        (0.9, 4.75, 'Read state\n(SLO slack, queue depth, hw util)'),
        (0.9, 3.75, 'Query Layer 2 advice'),
        (0.9, 2.75, 'Check anti-starvation guard'),
        (0.9, 1.75, 'Select mode from P/D ratio parameter\n+ apply batch-shape advice'),
        (0.9, 0.95, 'Emit { execution mode, batch composition } to engine'),
    ]
    for x, y, t in iter_steps:
        round_box(ax, x, y - 0.32, 5.2, 0.68, '#FFFFFF', '#1F4FB0', lw=1.2, pad=0.03)
        ax.text(x + 2.6, y, t, ha='center', va='center', fontsize=9)
    for y1, y2 in [(4.35, 4.10), (3.35, 3.10), (2.35, 2.10), (1.35, 1.30)]:
        arrow(ax, (3.5, y1), (3.5, y2), color='#1F4FB0', lw=1.4)

    # Window scale (right)
    round_box(ax, 6.85, 0.7, 5.85, 5.0, '#F3E8FF', '#6A1B9A', lw=2.0)
    ax.text(9.78, 5.42, 'Window Scale  —  fires at window boundary',
            ha='center', fontsize=11.5, fontweight='bold', color='#6A1B9A')

    win_steps = [
        (7.3, 4.75, 'Aggregate SLO attainment within window'),
        (7.3, 3.75, 'SLO violation?'),
        (7.3, 2.60, 'YES: shift toward lower Prefill frequency\n(protect decode latency)'),
        (7.3, 1.35, 'NO + queue non-empty: shift toward higher\nPrefill frequency (admit more work)'),
    ]
    for i, (x, y, t) in enumerate(win_steps):
        face = '#FFFFFF'
        round_box(ax, x, y - 0.35, 4.95, 0.76, face, '#6A1B9A', lw=1.2, pad=0.03)
        ax.text(x + 2.48, y, t, ha='center', va='center', fontsize=9)
    arrow(ax, (9.78, 4.40), (9.78, 4.10), color='#6A1B9A', lw=1.4)
    arrow(ax, (9.78, 3.40), (9.78, 2.95), color='#6A1B9A', lw=1.4)
    arrow(ax, (9.78, 2.25), (9.78, 1.70), color='#6A1B9A', lw=1.4)

    ax.text(9.78, 0.45,
            'Output: smoothly-updated P/D ratio parameter',
            ha='center', fontsize=9.5, style='italic', color='#6A1B9A', fontweight='bold')

    # Cross-scale arrow
    arrow(ax, (6.85, 3.2), (6.6, 3.2), color='#888', lw=1.7, ls=(0, (5, 3)))
    ax.text(6.72, 3.5, 'ratio baseline\n(continuously evolving)',
            ha='center', va='center', fontsize=8.8, color='#555')

    fig.savefig(f'{FIGDIR}/patent_architecture_layer1.png')
    plt.close(fig)
    print("  patent_architecture_layer1.png")


def plot_timeline():
    """Timeline showing P/D alternation across iterations at different load regimes."""
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(-0.5, 26); ax.set_ylim(-1.8, 3.6); ax.axis('off')
    ax.set_title(
        'P/D Time-Division Multiplexing — Iteration Timeline (one cell = one precompiled-graph replay)',
        fontsize=12.5, fontweight='bold', pad=12,
    )

    # Iteration sequence: D = decode, P = prefill
    seq = list('DDDPDDDDPDDPPDDDDDDPDDDPD')
    col_D = '#D9E8FF'; edge_D = '#1F4FB0'
    col_P = '#FFE0B2'; edge_P = '#C77700'

    cell_w = 0.9
    y0 = 1.0
    h = 1.4
    for i, ch in enumerate(seq):
        x = i * (cell_w + 0.08)
        if ch == 'D':
            face, edge, txt_color = col_D, edge_D, edge_D
        else:
            face, edge, txt_color = col_P, edge_P, edge_P
        round_box(ax, x, y0, cell_w, h, face, edge, lw=1.3, pad=0.02)
        ax.text(x + cell_w / 2, y0 + h / 2, ch, ha='center', va='center',
                fontsize=12, fontweight='bold', color=txt_color)
        ax.text(x + cell_w / 2, y0 - 0.28, f'i{i+1}',
                ha='center', va='center', fontsize=7.5, color='#888')

    # Time axis arrow
    total_w = len(seq) * (cell_w + 0.08) - 0.08
    arrow(ax, (-0.2, 0.3), (total_w + 0.3, 0.3), color='#555', lw=1.5)
    ax.text(total_w / 2, -0.05, 'time (iteration index)',
            ha='center', fontsize=9.5, color='#555', style='italic')

    # Regime annotations (coarse grouping to avoid label crowding)
    regimes = [
        (0.0,                    2.7, 8  * (cell_w + 0.08), 'decode-dominant (sparse prefill insertion)', edge_D),
        (8  * (cell_w + 0.08),   2.7, 5  * (cell_w + 0.08), 'bursty prefill',            edge_P),
        (13 * (cell_w + 0.08),   2.7, 6  * (cell_w + 0.08), 'sustained decode',          edge_D),
        (19 * (cell_w + 0.08),   2.7, 6  * (cell_w + 0.08), 'mixed tail',                '#555'),
    ]
    for x, y, w, label, color in regimes:
        ax.plot([x + 0.05, x + w - 0.05], [y, y], color=color, lw=2.0, zorder=2)
        ax.plot([x + 0.05, x + 0.05], [y - 0.12, y + 0.12], color=color, lw=2.0, zorder=2)
        ax.plot([x + w - 0.05, x + w - 0.05], [y - 0.12, y + 0.12], color=color, lw=2.0, zorder=2)
        ax.text(x + w / 2, y + 0.32, label, ha='center', va='bottom',
                fontsize=8.5, color=color, fontweight='bold')

    # Legend
    lx = 0.0; ly = -1.25
    round_box(ax, lx, ly, 0.8, 0.55, col_D, edge_D, lw=1.3, pad=0.02)
    ax.text(lx + 0.4, ly + 0.28, 'D', ha='center', va='center',
            fontsize=11, fontweight='bold', color=edge_D)
    ax.text(lx + 1.0, ly + 0.28, 'decode-only iteration',
            ha='left', va='center', fontsize=9, color='#333')

    lx = 5.8
    round_box(ax, lx, ly, 0.8, 0.55, col_P, edge_P, lw=1.3, pad=0.02)
    ax.text(lx + 0.4, ly + 0.28, 'P', ha='center', va='center',
            fontsize=11, fontweight='bold', color=edge_P)
    ax.text(lx + 1.0, ly + 0.28, 'prefill-only iteration',
            ha='left', va='center', fontsize=9, color='#333')

    ax.text(12.0, ly + 0.28,
            'P/D ratio is adjusted every window via SLO-attainment feedback;\n'
            'each iteration emits one mode after anti-starvation guard.',
            ha='left', va='center', fontsize=8.8, color='#555', style='italic')

    fig.savefig(f'{FIGDIR}/patent_architecture_timeline.png')
    plt.close(fig)
    print("  patent_architecture_timeline.png")


def plot_combined():
    """Publication-style composite: three-layer architecture (top) and the
    realized P/D iteration tape produced by the Inference Engine (bottom),
    linked by a single vertical arrow. Aims at conference-paper aesthetics:
    muted palette, consistent typography, no decorative clutter."""

    # ── Palette (desaturated, paper-friendly) ──────────────────────────────
    L1_FILL = '#EAF0F9'; L1_EDGE = '#335C9E'
    L2_FILL = '#EAF2EA'; L2_EDGE = '#2F7A3E'
    L3_FILL = '#F6EED8'; L3_EDGE = '#A86A00'
    NB_FILL = '#F3F3F3'; NB_EDGE = '#5A5A5A'
    TXT     = '#1F1F1F'
    MUTED   = '#666666'

    D_FILL  = '#DCE6F5'; D_EDGE = '#335C9E'
    P_FILL  = '#F4E3C3'; P_EDGE = '#A86A00'

    # ── Canvas ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 9.8); ax.axis('off')
    # Typography defaults
    F_TITLE   = 13.0
    F_HEADER  = 11.0
    F_SUB     = 9.5
    F_BODY    = 9.0
    F_NOTE    = 8.3
    F_CAP     = 8.2
    LW_MAIN   = 1.6
    LW_AUX    = 1.3
    LW_EDGE   = 1.5

    ax.set_title(
        'SLO-Adaptive P/D Time-Division Multiplexing — Architecture and Realized Iteration Tape',
        fontsize=F_TITLE, fontweight='bold', pad=12, color=TXT,
    )

    # ═════════════════════════════════════════════════════════════════════
    # UPPER — three-layer architecture  (y ∈ [4.5, 9.4])
    # ═════════════════════════════════════════════════════════════════════
    # Top rail y = 8.85; bottom rail y = 5.1–6.9
    BOX_PAD = 0.025  # tighter corner radius for paper look

    # --- Requests (boundary source, rectangular framing) ---
    round_box(ax, 0.55, 8.60, 2.55, 0.60, NB_FILL, NB_EDGE, lw=LW_EDGE, pad=BOX_PAD)
    ax.text(1.825, 8.90, 'Incoming Requests',
            ha='center', va='center', fontsize=F_SUB, fontweight='bold', color=TXT)

    # --- Layer 1 (the controller — visually the heaviest) ---
    round_box(ax, 3.85, 7.95, 5.40, 1.55, L1_FILL, L1_EDGE, lw=2.0, pad=BOX_PAD)
    ax.text(6.55, 9.12, 'Layer 1 · SLO-Adaptive Control',
            ha='center', va='center', fontsize=F_HEADER, fontweight='bold', color=L1_EDGE)
    ax.text(6.55, 8.63,
            'iteration-level   mode + batch decision',
            ha='center', va='center', fontsize=F_BODY, color=TXT)
    ax.text(6.55, 8.23,
            'window-level   P/D ratio feedback',
            ha='center', va='center', fontsize=F_BODY, color=TXT)

    # --- Inference Engine (boundary sink, annotated subtitle) ---
    round_box(ax, 10.10, 8.55, 2.85, 0.70, NB_FILL, NB_EDGE, lw=LW_EDGE, pad=BOX_PAD)
    ax.text(11.525, 8.99, 'Inference Engine',
            ha='center', va='center', fontsize=F_SUB, fontweight='bold', color=TXT)
    ax.text(11.525, 8.66, 'precompiled-graph replay',
            ha='center', va='center', fontsize=F_CAP, style='italic', color=MUTED)

    # --- Layer 2 (advisor) ---
    round_box(ax, 3.25, 5.45, 3.70, 1.75, L2_FILL, L2_EDGE, lw=2.0, pad=BOX_PAD)
    ax.text(5.10, 6.78, 'Layer 2 · Graph-Recognition',
            ha='center', va='center', fontsize=F_HEADER, fontweight='bold', color=L2_EDGE)
    ax.text(5.10, 6.25, 'advisor',
            ha='center', va='center', fontsize=F_SUB, style='italic', color=TXT)
    ax.text(5.10, 5.80,
            'shape-alignment suggestions\nfrom pre-compiled shape set',
            ha='center', va='center', fontsize=F_BODY, color=TXT)

    # --- Layer 3 (introspection) ---
    round_box(ax, 8.10, 5.45, 3.70, 1.75, L3_FILL, L3_EDGE, lw=2.0, pad=BOX_PAD)
    ax.text(9.95, 6.78, 'Layer 3 · State Sampling',
            ha='center', va='center', fontsize=F_HEADER, fontweight='bold', color=L3_EDGE)
    ax.text(9.95, 6.25, 'introspection',
            ha='center', va='center', fontsize=F_SUB, style='italic', color=TXT)
    ax.text(9.95, 5.80,
            'offline profile  +  online\nlow-frequency sampling',
            ha='center', va='center', fontsize=F_BODY, color=TXT)

    # --- Top-rail flow (Requests → L1 → Engine) ---
    arrow(ax, (3.10, 8.90), (3.85, 8.725), color=NB_EDGE, lw=LW_MAIN, style='->')
    arrow(ax, (9.25, 8.725), (10.10, 8.90), color=L1_EDGE, lw=LW_MAIN, style='->')
    ax.text(9.675, 9.17, '{mode, batch}',
            ha='center', va='bottom', fontsize=F_CAP, color=L1_EDGE, style='italic')

    # --- Support-layer arrows (solid thin, consistent style) ---
    # L2 → L1 (advice)
    arrow(ax, (5.10, 7.20), (5.45, 7.95), color=L2_EDGE, lw=LW_AUX,
          style='->', ls=(0, (4, 2.5)))
    ax.text(4.60, 7.56, 'batch-shape advice',
            ha='right', va='center', fontsize=F_CAP, color=L2_EDGE)

    # L3 → L1 (state)
    arrow(ax, (9.95, 7.20), (8.55, 7.95), color=L3_EDGE, lw=LW_AUX,
          style='->', ls=(0, (4, 2.5)))
    ax.text(10.45, 7.56, 'hardware state',
            ha='left', va='center', fontsize=F_CAP, color=L3_EDGE)

    # L3 → L2 (shape-cost profile)
    arrow(ax, (8.10, 6.32), (6.95, 6.32), color=L3_EDGE, lw=LW_AUX,
          style='->', ls=(0, (4, 2.5)))
    ax.text(7.525, 6.42, 'shape-cost profile',
            ha='center', va='bottom', fontsize=F_CAP, color=L3_EDGE)

    # ═════════════════════════════════════════════════════════════════════
    # BRIDGE — vertical stream from Inference Engine to iteration tape
    # ═════════════════════════════════════════════════════════════════════
    bridge_x = 11.525
    arrow(ax, (bridge_x, 8.55), (bridge_x, 4.10),
          color=L1_EDGE, lw=2.0, style='->')
    ax.text(bridge_x + 0.23, 6.35,
            'emitted iteration tape\n(1 cell = 1 graph replay)',
            rotation=90, ha='left', va='center',
            fontsize=F_CAP, color=L1_EDGE, style='italic')

    # ═════════════════════════════════════════════════════════════════════
    # LOWER — realized P/D iteration tape  (y ∈ [0.3, 4.1])
    # ═════════════════════════════════════════════════════════════════════
    seq = list('DDDPDDDDPDDPPDDDDDDPDDDPD')

    # Subpanel framing line (thin rule at top of the tape region)
    ax.plot([0.4, 13.1], [4.05, 4.05], color='#D0D0D0', lw=0.8, zorder=1)
    ax.text(0.4, 3.82, 'Realized iteration tape (one cell = one replay of the inference engine)',
            ha='left', va='center', fontsize=F_SUB, fontweight='bold', color=TXT)

    cell_w = 0.44
    x0     = 0.55
    y_cell = 1.90
    h_cell = 1.00
    for i, ch in enumerate(seq):
        x = x0 + i * (cell_w + 0.05)
        if ch == 'D':
            face, edge, txt_color = D_FILL, D_EDGE, D_EDGE
        else:
            face, edge, txt_color = P_FILL, P_EDGE, P_EDGE
        round_box(ax, x, y_cell, cell_w, h_cell, face, edge, lw=1.1, pad=0.015)
        ax.text(x + cell_w / 2, y_cell + h_cell / 2, ch,
                ha='center', va='center', fontsize=9.3, fontweight='bold', color=txt_color)
        ax.text(x + cell_w / 2, y_cell - 0.22, f'i{i+1}',
                ha='center', va='center', fontsize=6.3, color=MUTED)

    total_w = len(seq) * (cell_w + 0.05) - 0.05

    # Time axis
    arrow(ax, (x0 - 0.12, 1.35), (x0 + total_w + 0.12, 1.35),
          color='#444', lw=1.2, style='->')
    ax.text(x0 + total_w / 2, 1.10, 'time (iteration index)',
            ha='center', va='center', fontsize=F_CAP, color=MUTED, style='italic')

    # Regime brackets above
    regimes = [
        (0,  8, 'decode-dominant (sparse prefill insertion)', D_EDGE),
        (8,  5, 'bursty prefill',                             P_EDGE),
        (13, 6, 'sustained decode',                           D_EDGE),
        (19, 6, 'mixed tail',                                 MUTED),
    ]
    y_reg = y_cell + h_cell + 0.22
    for start, span, label, color in regimes:
        xa = x0 + start * (cell_w + 0.05) + 0.05
        xb = x0 + (start + span) * (cell_w + 0.05) - 0.05
        ax.plot([xa, xb], [y_reg, y_reg], color=color, lw=1.4, zorder=2)
        ax.plot([xa, xa], [y_reg - 0.07, y_reg + 0.07], color=color, lw=1.4, zorder=2)
        ax.plot([xb, xb], [y_reg - 0.07, y_reg + 0.07], color=color, lw=1.4, zorder=2)
        ax.text((xa + xb) / 2, y_reg + 0.12, label,
                ha='center', va='bottom', fontsize=F_CAP, fontweight='bold', color=color)

    # Legend + note (bottom row)
    ly = 0.42
    lx = 0.55
    round_box(ax, lx, ly, 0.40, 0.34, D_FILL, D_EDGE, lw=1.0, pad=0.015)
    ax.text(lx + 0.20, ly + 0.17, 'D', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=D_EDGE)
    ax.text(lx + 0.52, ly + 0.17, 'decode-only iteration',
            ha='left', va='center', fontsize=F_CAP, color=TXT)

    lx = 3.55
    round_box(ax, lx, ly, 0.40, 0.34, P_FILL, P_EDGE, lw=1.0, pad=0.015)
    ax.text(lx + 0.20, ly + 0.17, 'P', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=P_EDGE)
    ax.text(lx + 0.52, ly + 0.17, 'prefill-only iteration',
            ha='left', va='center', fontsize=F_CAP, color=TXT)

    ax.text(6.60, ly + 0.17,
            'P/D ratio adjusted every window from SLO-attainment; '
            'per-iteration mode is gated by the anti-starvation guard.',
            ha='left', va='center', fontsize=F_NOTE, color=MUTED, style='italic')

    # Footer caption clarifying the Inference Engine abstraction
    ax.text(0.4, 0.08,
            'Fig.: The Inference Engine denotes the accelerator-resident worker that replays a '
            'precompiled computation graph once per iteration (e.g., ACL-Graph / CUDA-Graph replay); '
            'it is treated as a black-box executor invoked by Layer 1.',
            ha='left', va='bottom', fontsize=F_NOTE, color=MUTED, style='italic', wrap=True)

    fig.savefig(f'{FIGDIR}/patent_architecture_combined.png')
    plt.close(fig)
    print("  patent_architecture_combined.png")


if __name__ == '__main__':
    plot_architecture_main()
    plot_layer1_twoscale()
    plot_timeline()
    plot_combined()
    print('Done. Figures at', FIGDIR)
