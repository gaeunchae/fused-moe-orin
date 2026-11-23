#!/usr/bin/env python3
"""What the host does while the GPU idles between decode tokens, from an nsys trace with CPU sampling.

The window for each token pair runs from the end of token t's last kernel to the start of token t+1's first kernel.
Inside it we report CUDA runtime calls on the main thread and the CPU samples on that thread, grouped by function.

Usage: python3 -I host_gap.py TRACE.sqlite
"""
import re
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict


def short(sym):
    sym = re.sub(r"\(.*", "", sym or "?")
    sym = re.sub(r"<.*>", "<>", sym)
    return sym.split("::")[-1] if len(sym) > 60 else sym


def main(path):
    c = sqlite3.connect(path)
    strs = dict(c.execute("select id, value from StringIds"))
    ks = c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    toks, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b[0] - a[1] > 300_000:
            toks.append(cur)
            cur = []
        cur.append(b)
    toks.append(cur)
    mode = st.mode(len(t) for t in toks)
    idx = [i for i, t in enumerate(toks) if abs(len(t) - mode) <= 4][2:]
    wins = [(toks[i][-1][1], toks[i + 1][0][0]) for i in idx if i + 1 < len(toks) and abs(len(toks[i + 1]) - mode) <= 4]
    n = len(wins)
    gap_ms = [(b - a) / 1e6 for a, b in wins]
    print(f"{n} inter-token windows, GPU idle median {st.median(gap_ms):.3f} ms, mean {st.mean(gap_ms):.3f} ms")

    main_tid = c.execute(
        "select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by count(*) desc limit 1"
    ).fetchone()[0]

    def in_wins(s, e):
        for a, b in wins:
            lo, hi = max(s, a), min(e, b)
            if hi > lo:
                yield hi - lo

    print("\n-- CUDA runtime calls on the main thread, time inside the windows (ms per window) --")
    api = defaultdict(lambda: [0, 0.0])
    for s, e, nid in c.execute(f"select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid={main_tid}"):
        for ov in in_wins(s, e):
            a = api[strs[nid]]
            a[0] += 1
            a[1] += ov / 1e6
    for name, (cnt, ms) in sorted(api.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {name[:46]:<48}{cnt / n:6.1f}/win {ms / n:8.3f}")

    print("\n-- CPU samples on the main thread inside the windows --")
    samples = c.execute(f"select id, start from COMPOSITE_EVENTS where globalTid={main_tid} order by start").fetchall()
    chains = defaultdict(list)
    for cid, sym, mod, depth in c.execute("select id, symbol, module, stackDepth from SAMPLING_CALLCHAINS"):
        chains[cid].append((depth, strs.get(sym, "?"), strs.get(mod, "?")))
    total_in = 0
    leaf, incl, mods = Counter(), Counter(), Counter()
    j = 0
    for a, b in wins:
        while j < len(samples) and samples[j][1] < a:
            j += 1
        k = j
        while k < len(samples) and samples[k][1] < b:
            ch = sorted(chains.get(samples[k][0], []))
            if ch:
                total_in += 1
                leaf[(short(ch[0][1]), ch[0][2].split("/")[-1])] += 1
                mods[ch[0][2].split("/")[-1]] += 1
                for f in {short(s) for _, s, m in ch if "libllama" in m or "libggml" in m or "llama-bench" in m}:
                    incl[f] += 1
            k += 1
    per_win = total_in / n
    period_us = 137600 / 2.2e3
    print(f"  {total_in} samples, {per_win:.1f} per window (about {period_us:.0f} us apart at 2.2 GHz)")
    print("  by module (leaf):")
    for m, v in mods.most_common(8):
        print(f"    {m:<40}{v / total_in * 100:5.1f}%")
    print("  leaf functions:")
    for (f, m), v in leaf.most_common(14):
        print(f"    {f[:52]:<54}{m[:18]:<20}{v / total_in * 100:5.1f}%")
    print("  inclusive, llama.cpp functions on the stack:")
    for f, v in incl.most_common(16):
        print(f"    {f[:70]:<72}{v / total_in * 100:5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
