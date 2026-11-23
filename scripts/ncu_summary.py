#!/usr/bin/env python3
"""Summarize an ncu raw CSV export of decode kernels by kernel kind.

For each kind: time, DRAM bytes (L2 fills from system memory) against the bytes it must read,
occupancy, issue rate, and the top warp stall reasons.

Export first: ncu --import REPORT.ncu-rep --page raw --csv > raw.csv
Usage: python3 -I ncu_summary.py raw.csv
"""
import csv
import re
import sys
from collections import defaultdict

PEAK = 180e9
Q4K, Q6K = 144, 210


def kind(name, gx, gy, by):
    m = re.search(r"mul_mat_vec_q<(\d+), (\d+), (\d+), (\d+)", name)
    if m:
        t, fused = int(m.group(1)), m.group(3) == "1"
        bs, tn = (Q4K, "Q4_K") if t == 12 else (Q6K, "Q6_K")
        if gy == 8:
            return ("expert gate+up", 2 * 8 * 768 * 8 * bs) if fused else (f"expert down {tn}", 8 * 2048 * 3 * bs)
        rows = gx * (1 if by == 4 else 2)
        if rows == 4096:
            return "attn q", 4096 * 8 * bs
        if rows == 2048:
            return "attn output", 2048 * 16 * bs
        if rows == 512 or rows == 1024:
            return f"attn k/v {tn}", 512 * 8 * bs
        return f"GEMV other {gx}x{gy}", None
    if "mul_mat_vec_f" in name:
        return "router GEMV", 128 * 2048 * 4
    if "flash_attn" in name:
        return ("flash attn fixup" if "fixup" in name else "flash attention"), None
    for key, label in (("rms_norm_mul_rope", "rope+norm"), ("rms_norm", "rms norm"), ("quantize_q8_1", "quantize"),
                       ("topk_moe", "router top-k"), ("moe_weighted_reduction", "expert weighted sum"),
                       ("k_bin_bcast", "residual add"), ("k_set_rows", "KV write"), ("k_get_rows", "get rows")):
        if key in name:
            if key == "quantize_q8_1":
                return f"quantize grid {gx}x{gy}", None
            return label, None
    return name[:30], None


def f(row, col):
    try:
        return float(row[col].replace(",", ""))
    except (KeyError, ValueError):
        return float("nan")


def main(path):
    rows = list(csv.DictReader(open(path)))
    units = rows[0]
    rows = [r for r in rows[1:] if r.get("ID", "").strip()]
    tunit = units["gpu__time_duration.sum"]
    tscale = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "nsecond": 1e-3, "ms": 1e3, "msecond": 1e3}[tunit]
    stall_cols = [c for c in rows[0] if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]

    groups = defaultdict(list)
    for r in rows:
        k, need = kind(r["Kernel Name"], int(f(r, "launch__grid_dim_x")), int(f(r, "launch__grid_dim_y")),
                       int(f(r, "launch__block_dim_y")))
        groups[k].append((r, need))

    print(f"{len(rows)} kernels profiled (kernel replay, caches flushed, CUDA graphs off)\n")
    print(f"{'kind':<22}{'n':>4}{'us':>8}{'DRAM MB':>9}{'need MB':>9}{'ratio':>7}{'GB/s':>7}{'grid':>11}{'thr':>5}"
          f"{'regs':>5}{'occ%':>6}{'issue%':>7}{'elig':>6}  top stalls (share of stalled warps)")
    order = sorted(groups, key=lambda k: -sum(f(r, "gpu__time_duration.sum") for r, _ in groups[k]))
    for k in order:
        g = groups[k]
        n = len(g)
        avg = lambda c: sum(f(r, c) for r, _ in g) / n
        us = avg("gpu__time_duration.sum") * tscale
        dram = avg("lts__d_sectors_fill_sysmem.sum") * 32 / 1e6
        need = g[0][1]
        ratio = f"{dram * 1e6 / need:.2f}" if need else "-"
        needs = f"{need / 1e6:.2f}" if need else "-"
        gbs = dram * 1e6 / (us * 1e-6) / 1e9 if us else 0
        grid = f"{int(avg('launch__grid_dim_x'))}x{int(avg('launch__grid_dim_y'))}"
        stalls = {c[len('smsp__average_warps_issue_stalled_'):-len('_per_issue_active.ratio')]: avg(c) for c in stall_cols}
        tot = sum(v for v in stalls.values() if v == v)
        top = sorted(stalls.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))[:4]
        tops = ", ".join(f"{s} {v / tot * 100:.0f}%" for s, v in top if tot)
        print(f"{k:<22}{n:>4}{us:>8.1f}{dram:>9.2f}{needs:>9}{ratio:>7}{gbs:>7.0f}{grid:>11}{int(avg('launch__block_size')):>5}"
              f"{int(avg('launch__registers_per_thread')):>5}{avg('sm__warps_active.avg.pct_of_peak_sustained_active'):>6.0f}"
              f"{avg('smsp__issue_active.avg.pct_of_peak_sustained_active'):>7.1f}{avg('smsp__warps_eligible.avg.per_cycle_active'):>6.2f}  {tops}")

    print("\noccupancy limits for the expert GEMVs (blocks per SM allowed by each resource):")
    for k in order:
        if k.startswith("expert"):
            r = groups[k][0][0]
            lim = {c.replace("launch__occupancy_limit_", ""): r[c] for c in r if c.startswith("launch__occupancy_limit_")}
            print(f"  {k:<20} {lim}  waves/SM {r['launch__waves_per_multiprocessor']}")


if __name__ == "__main__":
    main(sys.argv[1])
