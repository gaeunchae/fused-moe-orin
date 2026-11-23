#!/usr/bin/env python3
"""Split per-token GPU time into the weight-transfer floor (bytes / peak) and time above it. Usage: floor_breakdown.py TRACE.sqlite"""
import re
import sqlite3
import statistics as st
import sys

PEAK = 180e9
# Bytes read per launch, keyed by (ggml type, rows covered by one launch, experts per launch).
Q4K, Q6K = 144, 210


def classify(name, gx, gy, block_y):
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)1, \(bool\)(\d), \(bool\)(\d)", name)
    if m:
        t, fused, small_k = int(m.group(1)), m.group(2) == "1", m.group(3) == "1"
        bs = Q4K if t == 12 else Q6K
        tname = "Q4_K" if t == 12 else "Q6_K"
        if gy == 8:
            if fused:
                rows = 768
                return "expert gate+up", 2 * 8 * rows * 8 * bs
            return f"expert down {tname}", 8 * 2048 * 3 * bs
        if gx in (151936, 75968):
            return "output head", 151936 * 8 * bs
        if gx in (4096, 2048) and not fused and gx * (1 if block_y == 4 else 2) == 4096:
            return "attn q", 4096 * 8 * bs
        if gx in (2048, 1024):
            return "attn output", 2048 * 16 * bs
        if gx in (512, 256):
            return f"attn k/v", 512 * 8 * bs
        return f"other GEMV {tname}", None
    if "mul_mat_vec_f" in name:
        return "router", 128 * 2048 * 4
    if "flash_attn" in name:
        return "attention (flash)", None
    return "non-GEMV ops", None


def main(path):
    c = sqlite3.connect(path)
    strs = dict(c.execute("select id, value from StringIds"))
    rows = c.execute("select start,end,demangledName,gridX,gridY,blockY from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    toks, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[1] > 300_000:
            toks.append(cur)
            cur = []
        cur.append(b)
    toks.append(cur)
    mode = st.mode(len(t) for t in toks)
    dec = [t for t in toks if abs(len(t) - mode) <= 4][2:]
    n = len(dec)
    agg = {}
    for t in dec:
        for s, e, nm, gx, gy, by in t:
            cat, b = classify(strs[nm], gx, gy, by)
            a = agg.setdefault(cat, [0.0, 0.0, 0])
            a[0] += (e - s) / 1e6
            a[1] += (b or 0)
            a[2] += 1
    span = st.mean(t[-1][1] - t[0][0] for t in dec) / 1e6
    busy = sum(v[0] for v in agg.values()) / n
    period = st.median(dec[i + 1][0][0] - dec[i][0][0] for i in range(n - 1)) / 1e6
    print(f"{path.split('/')[-1]}: {n} tokens, period {period:.2f} ms, GPU span {span:.2f} ms")
    print(f"{'category':<20}{'ms/tok':>8}{'floor ms':>10}{'above floor':>13}{'% of peak':>11}")
    tot_floor = 0
    for cat, (ms, b, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        ms /= n
        fl = b / n / PEAK * 1e3
        tot_floor += fl
        pk = f"{fl / ms * 100:.0f}%" if b else "-"
        print(f"{cat:<20}{ms:>8.2f}{fl:>10.2f}{ms - fl:>13.2f}{pk:>11}")
    print(f"{'gaps between kernels':<20}{span - busy:>8.2f}{0:>10.2f}{span - busy:>13.2f}")
    print(f"{'host between tokens':<20}{period - span:>8.2f}{0:>10.2f}{period - span:>13.2f}")
    print(f"{'total':<20}{period:>8.2f}{tot_floor:>10.2f}{period - tot_floor:>13.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
