#!/usr/bin/env python3
"""Judge the separation experiment (RESULTS.md 7.3) from one GPU's results.

H1 passes:    same rows per block, 2 vs 4 warps, on large matrices where both fill the GPU many times over.
H2 fill:      efficiency against waves = blocks / (SMs x resident blocks per SM), where resident blocks come from
              the per-kernel register table; and pairs at equal rows that differ only in blocks (r=1 vs r=4).
H3 registers: the same comparison as H1 on the fused (gate+up) kernel.
Models:       median decode speed over interleaved rounds, relative to master, with the spread across rounds.
Usage: ablation_analysis.py DIR SMS   (DIR holds dense/dense.csv, regs.txt, models/csv/)
"""
import csv
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TYPE_ID = {'q4_K': 12, 'q6_K': 14, 'q8_0': 8}
SM_BLOCKS, SM_WARPS, SM_REGS = 16, 48, 65536  # sm_86 and sm_87


def load_regs(path):
    regs = {}
    for line in open(path):
        t, fus, sk, hi, nw, r, rg, sh = map(int, line.split())
        if sk == 0 and hi == 0:
            regs[(t, fus, nw, r)] = rg
    return regs


def resident(regs, t, fused, nw, r):
    rg = regs[(TYPE_ID[t], int(fused), nw, r)]
    per_warp = math.ceil(rg * 32 / 256) * 256
    return min(SM_BLOCKS, SM_WARPS // nw, SM_REGS // per_warp // nw)


def main():
    d, sms = Path(sys.argv[1]), int(sys.argv[2])
    regs = load_regs(d / 'regs.txt')
    rows = list(csv.DictReader(open(d / 'dense' / 'dense.csv')))
    us = {(r['label'], r['variant'], r['type'], int(r['cols']), int(r['rows'])): float(r['us']) for r in rows}
    eff = {(r['label'], r['variant'], r['type'], int(r['cols']), int(r['rows'])): float(r['pct_peak']) for r in rows}
    shape = {(r['label'], r['variant'], r['type'], int(r['cols']), int(r['rows'])): (int(r['warps']), int(r['rows_per_block']), int(r['blocks'])) for r in rows}
    sizes = sorted({k[2:] for k in us})

    print(f'#### {d} ({sms} SMs): {len(rows)} measurements')
    # H1 and H3: 2 vs 4 warps at equal rows per block, large matrices only (>= 8192 rows: many waves for both)
    for variant, name in (('plain', 'H1 passes (plain)'), ('fused', 'H3 registers (fused)')):
        print(f'\n{name}: time with 2 warps / time with 4 warps, same rows per block, rows >= 8192')
        for r in (1, 2, 4, 8):
            ratios = [us[(f's2x{r}', variant, *s)] / us[(f's4x{r}', variant, *s)] for s in sizes
                      if s[2] >= 8192 and (f's2x{r}', variant, *s) in us and (f's4x{r}', variant, *s) in us]
            if ratios:
                print(f'  {r} rows/block: median {st.median(ratios):.3f}  range {min(ratios):.3f}-{max(ratios):.3f}  (n={len(ratios)})')

    # H2: relative efficiency (to the same shape's best over sizes of the same type/cols) against waves
    print('\nH2 fill: efficiency relative to the same shape at its largest size, by waves (median over shapes and sizes)')
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
    for variant in ('plain', 'fused'):
        acc = defaultdict(list)
        for (label, v, t, cols, nrows), e in eff.items():
            if v != variant or not label.startswith('s'):
                continue
            nw, r, blocks = shape[(label, v, t, cols, nrows)]
            top = eff.get((label, v, t, cols, 16384))
            if not top:
                continue
            waves = blocks / (sms * resident(regs, t, v == 'fused', nw, r))
            for lo, hi in bins:
                if lo <= waves < hi:
                    acc[(lo, hi)].append(e / top)
        print(f'  {variant}: ' + '  '.join(f'[{lo:g},{hi if hi < 1e9 else "inf"}) {st.median(acc[(lo, hi)]) * 100:5.1f}% n={len(acc[(lo, hi)])}'
                                          for lo, hi in bins if acc[(lo, hi)]))
    print('\nH2 pairs: time with 4 rows per block / time with 1 row per block (4x fewer blocks, same passes)')
    for variant in ('plain', 'fused'):
        for nw in (2, 4):
            line = []
            for nrows in (256, 512, 1024, 2048, 4096, 16384):
                ratios = [us[(f's{nw}x4', variant, t, cols, nrows)] / us[(f's{nw}x1', variant, t, cols, nrows)]
                          for t in TYPE_ID for cols in (2048, 4096, 12288) if (f's{nw}x4', variant, t, cols, nrows) in us]
                if ratios:
                    line.append(f'{nrows}:{st.median(ratios):.2f}')
            print(f'  {variant} {nw} warps: ' + '  '.join(line))

    # rule configurations against master and the best forced shape
    print('\nRule configurations: efficiency % (median over types and cols), by rows')
    labels = ['master', 'r8', 'r8_nw4', 'r8_mb4', 'r8_nw4_mb4', 'best']
    for variant in ('plain', 'fused'):
        print(f'  {variant}: ' + ' '.join(f'{l:>11}' for l in labels))
        for nrows in (256, 512, 1024, 2048, 4096, 8192, 16384):
            cells = []
            for l in labels:
                vals = []
                for t in TYPE_ID:
                    for cols in (2048, 4096, 12288):
                        if l == 'best':
                            cand = [eff[k] for k in eff if k[1] == variant and k[0].startswith('s') and k[2:] == (t, cols, nrows)]
                            if cand:
                                vals.append(max(cand))
                        elif (l, variant, t, cols, nrows) in eff:
                            vals.append(eff[(l, variant, t, cols, nrows)])
                cells.append(f'{st.median(vals):11.1f}' if vals else ' ' * 11)
            print(f'  {nrows:>6} ' + ''.join(cells))

    # real models
    mdir = d / 'models' / 'csv'
    if mdir.exists():
        print('\nModels: decode tok/s, median over rounds (spread = max-min over rounds), change vs master median')
        data = defaultdict(list)
        for f in mdir.glob('*.csv'):
            m, c, _ = f.stem.split('__')
            for row in csv.DictReader(open(f)):
                if row['n_gen'] != '0':
                    data[(m, c)].append(float(row['avg_ts']))
        for m in sorted({k[0] for k in data}):
            base = st.median(data[(m, 'base')])
            parts = []
            for c in ('base', 'r8', 'r8_nw4', 'r8_mb4', 'r8_nw4_mb1', 'r8_nw4_mb2', 'r8_nw4_mb4', 'r8_nw4_mb8', 'r8_nw4_mb16'):
                v = data.get((m, c))
                if v:
                    parts.append(f'{c} {st.median(v) / base * 100 - 100:+5.1f}% (±{(max(v) - min(v)) / base * 50:.1f})')
            print(f'  {m:<26} base {base:7.2f} | ' + ' | '.join(parts[1:]))


if __name__ == '__main__':
    main()
