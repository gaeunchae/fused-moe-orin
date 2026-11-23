#!/usr/bin/env python3
"""Score rule v4 (RESULTS.md 7.5) against rule r8 and master on the forced-shape sweeps already measured on Orin
and the RTX 3090, without new measurements. For every configuration the rule's shape is looked up among the 12
forced shapes and compared with the best of them.

v4 = r8, then while waves < W: halve rows per block, else (rows = 1) double warps up to 4 if the row has lanes left.
waves = blocks / (SMs x resident blocks per SM), resident from the kernel's registers and the SM limits.
Usage: rule_v4_score.py
"""
import csv
import math
import re
import statistics as st
from pathlib import Path

P = Path('/media/vlsi/NVMESSD/fused-moe/profiling')
GPUS = {
    'Orin': dict(sms=16, regs=P / 'ablation_orin/regs.txt', dense=P / 'ablation_orin/dense/dense.csv', moe=P / 'shapes'),
    '3090': dict(sms=82, regs=P / 'rtx3090/ablation/regs.txt', dense=P / 'rtx3090/ablation/dense/dense.csv', moe=P / 'rtx3090/run1/shapes'),
}
TYPE_ID = {'q4_0': 2, 'q4_1': 3, 'q5_0': 6, 'q5_1': 7, 'q8_0': 8, 'q2_K': 10, 'q3_K': 11, 'q4_K': 12, 'q5_K': 13,
           'q6_K': 14, 'iq4_nl': 20, 'iq4_xs': 23, 'mxfp4': 39}
LANES_PER_COL = {'q8_0': 1 / 8, 'q2_K': 1 / 32, 'q3_K': 1 / 32, 'iq4_xs': 1 / 32}  # the rest 1/16
SHAPES = [(nw, r) for nw in (1, 2, 4) for r in (1, 2, 4, 8)]
SM_BLOCKS, SM_WARPS, SM_REGS = 16, 48, 65536
MOE_ROWS, MOE_EXPERTS = 2048, 8


def load_regs(path):
    out = {}
    for line in open(path):
        t, fus, sk, hi, nw, r, rg, sh = map(int, line.split())
        if sk == 0 and hi == 0:
            out[(t, fus, nw, r)] = rg
    return out


def resident(regs, t, fused, nw, r):
    rg = regs[(TYPE_ID[t], int(fused), nw, r)]
    return min(SM_BLOCKS, SM_WARPS // nw, SM_REGS // (math.ceil(rg * 32 / 256) * 256) // nw)


def r8(t, cols, row_bytes, fused):
    lanes = cols * LANES_PER_COL.get(t, 1 / 16)
    nw = 1 if lanes <= 32 else 2
    n_mats, sums = (2, 2) if fused else (1, 1)
    rows = 1
    while rows * n_mats * row_bytes < 8192 and 2 * rows * sums <= 4:
        rows *= 2
    return nw, rows


def v4(t, cols, row_bytes, fused, out_rows, grid_y, sms, regs, w_min):
    nw, rows = r8(t, cols, row_bytes, fused)
    lanes = cols * LANES_PER_COL.get(t, 1 / 16)
    while True:
        waves = math.ceil(out_rows / rows) * grid_y / (sms * resident(regs, t, fused, nw, rows))
        if waves >= w_min:
            break
        if rows > 1:
            rows //= 2
        elif nw < 4 and lanes > 32 * nw:
            nw *= 2
        else:
            break
    return nw, rows


def dense_configs(path):
    """{(variant, type, cols, rows): {label: (us, row_bytes)}}"""
    out = {}
    for r in csv.DictReader(open(path)):
        k = (r['variant'], r['type'], int(r['cols']), int(r['rows']))
        out.setdefault(k, {})[r['label']] = (float(r['us']), int(r['row_bytes']))
    return out


def moe_configs(d):
    out = {}
    for v in ('plain', 'fused'):
        for label, fname in [(f's{nw}x{r}', f'{v}_{nw}x{r}.txt') for nw, r in SHAPES] + [('master', f'{v}_bb_v1.txt')]:
            for line in open(d / fname):
                m = re.match(r'(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+%', line)
                if m and m.group(1) in TYPE_ID:
                    k = (v, m.group(1), int(m.group(2)), MOE_ROWS)
                    out.setdefault(k, {})[label] = (float(m.group(4)), int(m.group(3)))
    return out


def score(gpu, cfg, data, grid_y, dataset):
    regs, sms = load_regs(cfg['regs']), cfg['sms']
    print(f'\n## {gpu} {dataset}: % of the best forced shape (time best/time), median | min | within 5% | n')
    res = {}
    for v in ('plain', 'fused'):
        rows_out = {k: d for k, d in data.items() if k[0] == v and all(f's{nw}x{r}' in d for nw, r in SHAPES)}
        picks = {
            'master': lambda k, d: 'master',
            'r8': lambda k, d: 's%dx%d' % r8(k[1], k[2], d['s1x1'][1], v == 'fused'),
            'v4 W=1': lambda k, d: 's%dx%d' % v4(k[1], k[2], d['s1x1'][1], v == 'fused', k[3], grid_y, sms, regs, 1),
            'v4 W=2': lambda k, d: 's%dx%d' % v4(k[1], k[2], d['s1x1'][1], v == 'fused', k[3], grid_y, sms, regs, 2),
            'v4 W=4': lambda k, d: 's%dx%d' % v4(k[1], k[2], d['s1x1'][1], v == 'fused', k[3], grid_y, sms, regs, 4),
        }
        for name, pick in picks.items():
            vals = []
            for k, d in rows_out.items():
                lab = pick(k, d)
                if lab not in d:
                    continue
                best = min(d[f's{nw}x{r}'][0] for nw, r in SHAPES)
                vals.append((best / d[lab][0], k))
            res[(v, name)] = dict((k, x) for x, k in vals)
            xs = [x for x, _ in vals]
            print(f'  {v:<5} {name:<7} {st.median(xs) * 100:5.1f}% | {min(xs) * 100:5.1f}% | {sum(x >= 0.95 for x in xs):3d}/{len(xs)}')
    return res


def main():
    summary = {}
    for gpu, cfg in GPUS.items():
        dense = score(gpu, cfg, dense_configs(cfg['dense']), 1, 'dense')
        moe = score(gpu, cfg, moe_configs(cfg['moe']), MOE_EXPERTS, 'MoE (2048 rows x 8 experts)')
        summary[gpu] = (dense, moe)

    print('\n## Success criteria (W=2)')
    s1 = []
    for gpu, (dense, moe) in summary.items():
        for name, res in (('dense', dense), ('MoE', moe)):
            for v in ('plain', 'fused'):
                med = st.median(res[(v, 'v4 W=2')].values())
                s1.append(med >= 0.97)
                print(f'  S1 {gpu} {name} {v}: median {med * 100:.1f}% of best ({"ok" if med >= 0.97 else "FAIL"})')
    dense, moe = summary['Orin']
    worse = []
    for res in (dense, moe):
        for v in ('plain', 'fused'):
            for k, x in res[(v, 'v4 W=2')].items():
                r = res[(v, 'r8')].get(k)
                if r and x < r * 0.97:
                    worse.append((k, x / r))
    print(f'  S2 Orin: configurations where v4 is more than 3% slower than r8: {len(worse)} '
          + ', '.join(f'{k[0]} {k[1]} {k[2]}x{k[3]} x{x:.2f}' for k, x in worse[:6]))
    dense3090 = summary['3090'][0]
    for v in ('plain', 'fused'):
        small = [k for k in dense3090[(v, 'v4 W=2')] if k[3] <= 1024]
        a = st.median(dense3090[(v, 'v4 W=2')][k] for k in small)
        m = st.median(dense3090[(v, 'master')][k] for k in small if k in dense3090[(v, 'master')])
        print(f'  S3 3090 dense {v} rows <= 1024: v4 {a * 100:.1f}% vs master {m * 100:.1f}% of best ({"ok" if a >= m else "FAIL"})')


if __name__ == '__main__':
    main()
