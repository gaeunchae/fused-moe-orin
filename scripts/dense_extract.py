#!/usr/bin/env python3
"""Kernel times per configuration from an nsys trace of bench/dense_shapes (or mmid_formats with --mmid).

The benchmark prints one line per configuration with its node count. Each configuration launches (WARMUP + RUNS)
x nodes GEMV kernels in that order, so the trace is split by those counts. The first WARMUP x nodes launches are
dropped and the median of the rest is the kernel time. The grid and block of the launches give the shape that
actually ran: warps = blockY, rows per block = ceil(rows / gridX).
Usage: dense_extract.py STDOUT.txt TRACE.sqlite LABEL VARIANT PEAK_GBS OUT.csv   (appends rows to OUT.csv)
"""
import csv
import math
import re
import sqlite3
import statistics as st
import sys
from pathlib import Path

WARMUP, RUNS = 3, 15


def main():
    stdout, sqlite, label, variant, peak, out = sys.argv[1:7]
    peak = float(peak)
    n_mats = 2 if variant == 'fused' else 1
    cfgs = []
    for line in open(stdout):
        m = re.match(r'(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+%', line)
        if m and m.group(1) != 'type':
            t, cols, rows, row_bytes, nodes = m.group(1), *map(int, m.groups()[1:])
            cfgs.append((t, cols, rows, row_bytes, nodes))
    db = sqlite3.connect(sqlite)
    ks = db.execute("""SELECT s.value, k.gridX, k.blockY, k.end - k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
                       JOIN StringIds s ON s.id = k.shortName WHERE s.value LIKE 'mul_mat_vec%' ORDER BY k.start""").fetchall()
    need = sum((WARMUP + RUNS) * c[4] for c in cfgs)
    if len(ks) != need:
        sys.exit(f'{sqlite}: {len(ks)} GEMV launches, expected {need}; the trace does not match the printed configurations')
    new = not Path(out).exists()
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['label', 'variant', 'type', 'cols', 'rows', 'row_bytes', 'nodes', 'kernel', 'warps', 'rows_per_block',
                        'blocks', 'us', 'gbps', 'pct_peak'])
        i = 0
        for t, cols, rows, row_bytes, nodes in cfgs:
            chunk = ks[i:i + (WARMUP + RUNS) * nodes]
            i += len(chunk)
            timed = chunk[WARMUP * nodes:]
            names, grids, warps = {c[0] for c in timed}, {c[1] for c in timed}, {c[2] for c in timed}
            if len(names) != 1 or len(grids) != 1 or len(warps) != 1:
                sys.exit(f'{label} {t} {cols}x{rows}: mixed kernels or shapes {names} {grids} {warps}')
            us = st.median(c[3] for c in timed) / 1e3
            gx, nw = grids.pop(), warps.pop()
            gbps = n_mats * row_bytes * rows / us / 1e3
            w.writerow([label, variant, t, cols, rows, row_bytes, nodes, names.pop(), nw, math.ceil(rows / gx), gx,
                        f'{us:.3f}', f'{gbps:.1f}', f'{gbps / peak * 100:.1f}'])


if __name__ == '__main__':
    main()
