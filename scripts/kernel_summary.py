#!/usr/bin/env python3
"""Summarize GPU kernels in an nsys sqlite export for any model: launches, mean time and share of GPU time.

Unlike profile_breakdown.py this needs no model shapes, so it works for any GGUF. Kernels are grouped by
demangled name and grid size, which separates expert GEMVs (gridY = experts used) from dense ones.
Usage: kernel_summary.py TRACE.sqlite [--top N] [--match SUBSTRING]
"""
import argparse
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--match", default="", help="only kernels whose name contains this")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    rows = db.execute(
        """SELECT s.value, k.gridX, k.gridY, k.blockX, k.blockY, COUNT(*), SUM(k.end - k.start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
           GROUP BY s.value, k.gridX, k.gridY, k.blockX, k.blockY"""
    ).fetchall()
    total_ns = sum(r[6] for r in rows)
    rows = [r for r in rows if args.match in r[0]]
    rows.sort(key=lambda r: -r[6])

    print(f"total GPU kernel time {total_ns / 1e6:.1f} ms")
    print(f"{'share':>6} {'launches':>8} {'mean us':>8}  {'grid':>11} {'block':>7}  kernel")
    for name, gx, gy, bx, by, n, ns in rows[: args.top]:
        short = name.split("(")[0] if "(" in name and "<" not in name.split("(")[0] else name
        short = short.replace("(ggml_type)", "t").replace("void ", "")[:110]
        print(f"{ns / total_ns * 100:5.1f}% {n:8d} {ns / n / 1e3:8.1f}  {gx:>5}x{gy:<5} {bx:>3}x{by:<3}  {short}")


if __name__ == "__main__":
    main()
