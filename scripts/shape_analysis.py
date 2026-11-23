#!/usr/bin/env python3
"""Analyze the forced-shape sweep (scripts/shape_sweep.sh): best GEMV block shape per format, row length and variant,
and how close candidate shape rules come to it. Registers per kernel come from regs.txt (cuobjdump of the bb build).
Usage: shape_analysis.py [SWEEP_DIR]
"""
import math
import re
import statistics as st
import sys

O = sys.argv[1] if len(sys.argv) > 1 else '/media/vlsi/NVMESSD/fused-moe/profiling/shapes'
TYPE_ID = {'q4_0': 2, 'q4_1': 3, 'q5_0': 6, 'q5_1': 7, 'q8_0': 8, 'q2_K': 10, 'q3_K': 11, 'q4_K': 12, 'q5_K': 13,
           'q6_K': 14, 'iq4_nl': 20, 'iq4_xs': 23, 'mxfp4': 39}
LANES_PER_COL = {'q8_0': 1 / 8, 'q2_K': 1 / 32, 'q3_K': 1 / 32, 'iq4_xs': 1 / 32}  # others: 1/16
SHAPES = [(nw, r) for nw in (1, 2, 4) for r in (1, 2, 4, 8)]
SM_REGS, SM_WARPS, SM_BLOCKS = 65536, 48, 16  # sm_87


def load(path):
    out = {}
    for line in open(path):
        m = re.match(r'(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%', line)
        if m and m.group(1) in TYPE_ID:
            out[(m.group(1), int(m.group(2)))] = (int(m.group(3)), float(m.group(6)))
    return out


regs = {}
for line in open(f'{O}/regs.txt'):
    t, fus, sk, hi, nw, r, rg, sh = map(int, line.split())
    regs[(t, fus, nw, r)] = rg


def resident(t, fused, nw, r):
    rg = regs[(TYPE_ID[t], int(fused), nw, r)]
    per_warp = math.ceil(rg * 32 / 256) * 256
    return min(SM_BLOCKS, SM_WARPS // nw, SM_REGS // per_warp // nw)


def lanes(t, cols):
    return cols * LANES_PER_COL.get(t, 1 / 16)


def old_rule(t, cols, br, fused, target=4096):
    ln = lanes(t, cols)
    nw = 1 if ln <= 32 else 2 if ln <= 64 else 4
    r = 1
    while r < 8 and r * br < target:
        r *= 2
    return nw, r


def inflight_rule(t, cols, br, fused, cap=None):
    """Most weight bytes per SM in one pass: resident blocks x rows x matrices x bytes covered per pass."""
    n_mats = 2 if fused else 1
    best = None
    for nw, r in SHAPES:
        passes = math.ceil(lanes(t, cols) / (nw * 32))
        b = resident(t, fused, nw, r) * r * n_mats * br / passes
        if cap:
            b = min(b, cap)
        key = (b, -r, -nw)  # ties: fewer rows, then fewer warps
        if best is None or key > best[0]:
            best = (key, (nw, r))
    return best[1]


for variant in ('plain', 'fused'):
    fused = variant == 'fused'
    data = {s: load(f'{O}/{variant}_{s[0]}x{s[1]}.txt') for s in SHAPES}
    bb = load(f'{O}/{variant}_bb_v1.txt')
    ded = load(f'{O}/{variant}_dedicated.txt')
    print(f'\n######## {variant}: efficiency % (wall clock incl. ~2 us input quantization)')
    print(f"{'type':<7}{'cols':>5} {'best':>10} {'bb_v1':>6} {'old4K':>10} {'inflight':>10} {'cap40K':>10}")
    ratios = {k: [] for k in ('bb_v1', 'old4K', 'inflight', 'cap40K')}
    for key in sorted(bb, key=lambda k: (list(TYPE_ID).index(k[0]), k[1])):
        t, cols = key
        br = bb[key][0]
        effs = {s: data[s][key][1] for s in SHAPES}
        best = max(effs, key=effs.get)
        picks = {'old4K': old_rule(t, cols, br, fused), 'inflight': inflight_rule(t, cols, br, fused),
                 'cap40K': inflight_rule(t, cols, br, fused, cap=40960)}
        ratios['bb_v1'].append(bb[key][1] / effs[best])
        cells = []
        for k, s in picks.items():
            ratios[k].append(effs[s] / effs[best])
            cells.append(f'{s[0]}x{s[1]} {effs[s]:4.0f}')
        print(f'{t:<7}{cols:>5} {best[0]}x{best[1]} {effs[best]:4.0f}  {bb[key][1]:5.0f} ' + ' '.join(f'{c:>10}' for c in cells)
              + (f'   dedicated {ded[key][1]:.0f}' if abs(ded[key][1] - bb[key][1]) > 1 else ''))
    for k, v in ratios.items():
        print(f'# {variant} {k:<9}: median {st.median(v) * 100:5.1f}% of best, min {min(v) * 100:5.1f}%, '
              f'within 5% of best: {sum(x >= 0.95 for x in v)}/{len(v)}')
