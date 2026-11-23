#!/usr/bin/env python3
"""Figures for the report and the talk, drawn from the saved measurement files (PNG, light theme, 200 dpi).
Usage: make_figures.py [OUT_DIR]   (default /home/vlsi/fused-moe/figures)
"""
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

P = Path('/media/vlsi/NVMESSD/fused-moe/profiling')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/vlsi/fused-moe/figures')
PEAK = 180.0

# Reference palette, light mode: one accent, one contrast hue, greys for context.
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29d'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})

SHAPE_RE = re.compile(r'NW=(\d) R=(\d+)\s+cols=(\d+)\s+(\d+) blk \|\s+(\d+) B/block\s+(\d+)/(\d+)\s+thr \|\s+([\d.]+) us\s+([\d.]+) GB/s')


def load_shapes(files):
    """gemv_shape section D runs -> {(warps, rows, cols): (bytes per block, busy threads, threads, median GB/s)}."""
    runs = []
    for f in files:
        d = {}
        for line in open(f):
            m = SHAPE_RE.search(line)
            if m:
                nw, r, c, _, b, u, t, _, gb = m.groups()
                d[(int(nw), int(r), int(c))] = (int(b), int(u), int(t), float(gb))
        runs.append(d)
    return {k: runs[0][k][:3] + (st.median(r[k][3] for r in runs),) for k in runs[0]}


def title(ax, head, sub):
    ax.set_title(head, loc='left', fontsize=12, fontweight='bold', color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va='bottom')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / name)


def fig_row_length():
    """Stock block shape vs the best shape at each row length, same 7.08 MB of Q4_K weights."""
    d = load_shapes([P / f'rowlen/run{i}.txt' for i in (1, 2, 3)])
    cols = [512, 768, 1024, 2048, 4096]
    stock = [d[(4, 1, c)][3] for c in cols]
    best = [max(v[3] for k, v in d.items() if k[2] == c) for c in cols]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = range(len(cols))
    for i in y:
        ax.plot([stock[i], best[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(stock, y, s=60, color=GREY, zorder=2, label='llama.cpp shape (4 warps, 1 row per block)')
    ax.scatter(best, y, s=60, color=BLUE, zorder=3, label='best shape found')
    for i in y:
        ax.text(stock[i] - 4, i, f'{stock[i]:.0f}', ha='right', va='center', fontsize=9, color=INK2)
        ax.text(best[i] + 4, i, f'{best[i]:.0f}', ha='left', va='center', fontsize=9, color=INK)
    ax.axvline(PEAK, color=INK2, lw=1, ls='--')
    ax.text(PEAK - 2, -0.55, 'measured peak 180', ha='right', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), [f'{c} ({c * 9 // 16} B)' for c in cols])
    ax.set_ylabel('row length in weights (bytes per Q4_K row)')
    ax.set_xlabel('achieved bandwidth (GB/s)')
    ax.set_xlim(0, 200)
    ax.set_ylim(len(cols) - 0.5, -0.8)  # top row first, room above it for the peak label
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, frameon=False, fontsize=8.5)
    title(ax, 'Short rows are slow only in the stock block shape',
          'Same 7.08 MB of Q4_K expert weights cut into rows of 512-4096 weights, Jetson AGX Orin, median of 3 runs')
    save(fig, 'fig_row_length.png')


def fig_bytes_per_block():
    """GB/s against bytes per block for shapes with every thread busy: row lengths fall on one curve."""
    d = load_shapes([P / f'rowlen/run{i}.txt' for i in (1, 2, 3)])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for c, color in ((512, BLUE), (1024, ORANGE), (2048, AQUA)):  # three series: the all-pairs-safe palette slots
        pts = {}
        for (nw, r, cc), (b, used, thr, gb) in d.items():
            if cc == c and used == thr:
                pts[b] = max(pts.get(b, 0), gb)
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], marker='o', ms=5, lw=2, color=color, label=f'rows of {c} weights')
    ax.axhline(PEAK, color=INK2, lw=1, ls='--')
    ax.text(300, PEAK - 9, 'measured peak 180 GB/s', fontsize=8.5, color=INK2)
    ax.axvline(2304, color=GRID, lw=1)
    ax.text(2304 * 1.05, 100, 'saturates near 2.3 KB', fontsize=8.5, color=INK2)
    ax.set_xscale('log', base=2)
    ax.set_xticks([288, 576, 1152, 2304, 4608, 9216, 18432], ['288', '576', '1.1K', '2.3K', '4.6K', '9.2K', '18K'])
    ax.set_xlim(240, 60000)
    ax.set_ylim(0, 200)
    ax.set_xlabel('weight bytes per thread block')
    ax.set_ylabel('achieved bandwidth (GB/s)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    title(ax, 'Speed follows bytes per block, whatever the row length',
          'Best shape at each block size among shapes with no idle threads, Q4_K, 16 SMs')
    save(fig, 'fig_bytes_per_block.png')


def fig_sm_scaling():
    """Per-SM bandwidth when a green context limits the kernel to fewer SMs (clocks and DRAM unchanged)."""
    sms = [4, 6, 8, 12, 16]
    data = {n: load_shapes([P / f'rowlen/sms{n}_run{i}.txt' for i in (1, 2, 3)]) for n in sms}

    def per_sm(nbytes):
        out = []
        for n in sms:
            # every thread busy and the row covered in one pass: the same shapes as the table in RESULTS.md
            vals = [v[3] for k, v in data[n].items()
                    if v[0] == nbytes and v[1] == v[2] and -(-(k[2] // 256) // (2 * k[0])) == 1]
            out.append(st.median(vals) / n)
        return out

    small, big = per_sm(288), per_sm(4608)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(sms, small, marker='o', ms=6, lw=2, color=BLUE, label='288 B per block')
    ax.plot(sms, big, marker='o', ms=6, lw=2, color=GREY, label='4.6 KB per block')
    ax.text(16.4, small[-1], '288 B per block: flat,\nthe SMs are the limit', va='center', fontsize=9, color=INK)
    ax.text(16.4, big[-1], '4.6 KB per block: falls as SMs\nare added, DRAM is the limit', va='center', fontsize=9, color=INK2)
    for x, v in zip(sms, small):
        ax.text(x, v + 0.6, f'{v:.1f}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(sms)
    ax.set_xlim(3, 16.2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('SMs available to the kernel (green context)')
    ax.set_ylabel('GB/s per SM')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, frameon=False, fontsize=8.5)
    title(ax, 'Small blocks deliver the same 3.1 GB/s per SM at any SM count',
          'Q4_K GEMV, median over shapes with no idle threads that cover a row in one pass, 3 runs each')
    save(fig, 'fig_sm_scaling.png')


def tok_s(csv_path, depth=0):
    """avg_ts at one context depth from a llama-bench CSV."""
    import csv
    for row in csv.DictReader(open(csv_path)):
        if row['n_gen'] != '0' and int(row['n_depth']) == depth:
            return float(row['avg_ts'])
    raise ValueError(f'no depth {depth} in {csv_path}')


def fig_end_to_end():
    """Decode speed relative to the latest llama.cpp master: where we started, master, and ours on master."""
    models = [('Qwen3-30B-A3B', 'Qwen3-30B-A3B'), ('Qwen3.5-35B-A3B', 'Qwen3.5-35B-A3B'),
              ('OLMoE-1B-7B-0125-Instruct', 'OLMoE-1B-7B')]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (m, name) in enumerate(models):
        old = tok_s(P / f'models_bytes/{m}/base.csv')
        master = tok_s(P / f'master/{m}/base.csv')
        ours = tok_s(P / f'master/{m}/bytes_all.csv')
        ax.plot([old / master, ours / master], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([old / master], [i], s=45, color=GREY, zorder=2, label='llama.cpp 13ec8f6 (Nov 4)' if i == 0 else None)
        ax.scatter([ours / master], [i], s=60, color=BLUE, zorder=3, label='ours on master' if i == 0 else None)
        ax.text(ours / master + 0.012, i, f'+{(ours / master - 1) * 100:.0f}%  ({master:.1f} \u2192 {ours:.1f} tok/s)',
                va='center', fontsize=9, color=INK)
    ax.axvline(1.0, color=INK2, lw=1, ls='--')
    ax.text(1.006, -0.75, 'latest master e300919 = 1.0', ha='left', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(models)), [n for _, n in models])
    ax.set_ylim(len(models) - 0.5, -1.0)
    ax.set_xlim(0.8, 1.62)
    ax.set_xlabel('decode speed relative to the latest llama.cpp master (context 0)')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.22), ncol=2, frameon=False, fontsize=8.5)
    title(ax, 'Ours is 17-28% faster than the latest master on every MoE model',
          'llama-bench, 64 tokens, 5 runs, Jetson AGX Orin; master already has an Orin GEMV table (Nov 20)')
    save(fig, 'fig_end_to_end.png')


if __name__ == '__main__':
    fig_row_length()
    fig_bytes_per_block()
    fig_sm_scaling()
    fig_end_to_end()
