#!/usr/bin/env python3
"""Split per-token decode time from a node-level nsys trace into a data-transfer floor and the time above it.

The floor of a kernel is the bytes it must read divided by the measured peak read bandwidth (180 GB/s).
Weight bytes come from Qwen3-30B-A3B Q4_K_M shapes; attention reads the KV cache, 98,304 B per cached token.

Usage: python3 -I profile_breakdown.py TRACE.sqlite [context_depth] [--csv OUT]
"""
import re
import sqlite3
import statistics as st
import sys

PEAK = 180e9
Q4K, Q6K = 144, 210
KV_PER_TOKEN = 48 * 2 * 4 * 128 * 2  # layers x (K, V) x KV heads x head dim x f16

NON_GEMV = [
    ("rms_norm_mul_rope", "rope (fused with norm)"),
    ("rms_norm", "rms norm"),
    ("quantize_q8_1", "activation quantize"),
    ("topk_moe", "router top-k"),
    ("moe_weighted_reduction", "expert weighted sum"),
    ("k_bin_bcast", "residual add"),
    ("k_set_rows", "KV cache write"),
    ("k_get_rows", "embedding lookup"),
]


def classify(name, gx, gy, block_y, depth):
    if "mul_mat_vec_q4_K_moe_swiglu" in name:
        return "GEMV", "expert gate+up", 2 * 8 * 768 * 8 * Q4K
    m = re.search(r"mul_mat_vec_q_moe_short_rows<\(ggml_type\)(\d+)", name)
    if m:
        t = int(m.group(1))
        return "GEMV", "expert down " + ("Q4_K" if t == 12 else "Q6_K"), 8 * 2048 * 3 * (Q4K if t == 12 else Q6K)
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)1, \(bool\)(\d)", name)
    if m:
        t, fused = int(m.group(1)), m.group(2) == "1"
        bs = Q4K if t == 12 else Q6K
        if gy == 8:
            if fused:
                return "GEMV", "expert gate+up", 2 * 8 * 768 * 8 * bs
            return "GEMV", "expert down " + ("Q4_K" if t == 12 else "Q6_K"), 8 * 2048 * 3 * bs
        if gx in (151936, 75968):
            return "GEMV", "output head", 151936 * 8 * bs
        rows_per_block = 1 if block_y == 4 else 2
        rows = gx * rows_per_block
        if rows == 4096 and not fused:
            return "GEMV", "attn q", 4096 * 8 * bs
        if rows == 2048:
            return "GEMV", "attn output", 2048 * 16 * bs
        if rows in (512, 1024):
            return "GEMV", "attn k/v", 512 * 8 * bs
        return "GEMV", "other GEMV", None
    if "mul_mat_vec_f" in name:
        return "GEMV", "router GEMV", 128 * 2048 * 4
    if "flash_attn" in name:
        return "attention", "flash attention", KV_PER_TOKEN * (depth + 16) / 48 if "fixup" not in name else 0
    for key, label in NON_GEMV:
        if key in name:
            return "small ops", label, None
    return "small ops", "other", None


def main(path, depth=0, csv=None):
    c = sqlite3.connect(path)
    strs = dict(c.execute("select id, value from StringIds"))
    rows = c.execute("select start,end,demangledName,gridX,gridY,blockY from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    # A decode call ends with a host gap well above any gap between kernels inside it.
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
            grp, cat, b = classify(strs[nm], gx, gy, by, depth)
            a = agg.setdefault((grp, cat), [0.0, 0.0, 0])
            a[0] += (e - s) / 1e6
            a[1] += b or 0
            a[2] += 1
    span = st.mean(t[-1][1] - t[0][0] for t in dec) / 1e6
    busy = sum(v[0] for v in agg.values()) / n
    period = st.median(dec[i + 1][0][0] - dec[i][0][0] for i in range(n - 1)) / 1e6
    memcpy = c.execute(
        f"select count(*), coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_MEMCPY where start>={dec[0][0][0]} and end<={dec[-1][-1][1]}"
    ).fetchone()

    print(f"{path.split('/')[-1]}  depth={depth}  tokens={n}  kernels/token={mode}  period={period:.2f} ms ({1e3 / period:.1f} tok/s traced)")
    print(f"{'group':<10}{'item':<24}{'#/tok':>7}{'ms/tok':>8}{'floor':>8}{'above':>8}{'% peak':>8}")
    out, floor_total, groups = [], 0.0, {}
    for (grp, cat), (ms, b, cnt) in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
        ms /= n
        fl = b / n / PEAK * 1e3
        floor_total += fl
        g = groups.setdefault(grp, [0.0, 0.0])
        g[0] += ms
        g[1] += ms - fl
        pk = f"{fl / ms * 100:.0f}%" if b else "-"
        print(f"{grp:<10}{cat:<24}{cnt / n:>7.0f}{ms:>8.2f}{fl:>8.2f}{ms - fl:>8.2f}{pk:>8}")
        out.append((grp, cat, cnt / n, ms, fl))
    host = period - span
    gaps = span - busy
    print(f"{'host':<10}{'between tokens':<24}{'':>7}{host:>8.2f}{0:>8.2f}{host:>8.2f}")
    print(f"{'host':<10}{'gaps inside a token':<24}{'':>7}{gaps:>8.2f}{0:>8.2f}{gaps:>8.2f}")
    print(f"{'copy':<10}{'host-device memcpy':<24}{memcpy[0] / n:>7.0f}{memcpy[1] / n / 1e6:>8.3f}")
    print(f"\ntransfer floor {floor_total:.2f} ms, above floor {period - floor_total:.2f} ms of {period:.2f} ms:")
    above = period - floor_total
    for grp, (ms, ab) in groups.items():
        print(f"  {grp:<10} {ab:6.2f} ms ({ab / above * 100:4.1f}%)")
    print(f"  {'host':<10} {host + gaps:6.2f} ms ({(host + gaps) / above * 100:4.1f}%)")
    if csv:
        with open(csv, "w") as f:
            f.write("depth,group,item,count_per_token,ms_per_token,floor_ms\n")
            for grp, cat, cnt, ms, fl in out:
                f.write(f"{depth},{grp},{cat},{cnt:.1f},{ms:.4f},{fl:.4f}\n")
            f.write(f"{depth},host,between tokens,,{host:.4f},0\n{depth},host,gaps inside a token,,{gaps:.4f},0\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    csv = None
    if "--csv" in args:
        i = args.index("--csv")
        csv = args[i + 1]
        del args[i : i + 2]
    main(args[0], int(args[1]) if len(args) > 1 else 0, csv)
