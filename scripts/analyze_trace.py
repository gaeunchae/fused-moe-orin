#!/usr/bin/env python3
"""Per-token breakdown of a node-level nsys CUDA trace. Run with: python3 -I analyze_trace.py FILE.sqlite"""
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict


def short(name):
    name = re.sub(r"^void ", "", name or "?")
    m = re.match(r"([A-Za-z_0-9:]+)(<.*>)?", name)
    base = m.group(1) if m else name[:40]
    tmpl = m.group(2) if m and m.group(2) else ""
    if tmpl:
        tmpl = re.sub(r"\(ggml_type\)", "t", tmpl)
        tmpl = re.sub(r"\s+", "", tmpl)[:40]
    return base.split("::")[-1] + tmpl


def main(path):
    c = sqlite3.connect(path)
    strs = dict(c.execute("select id, value from StringIds"))
    rows = c.execute(
        "select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    ks = [dict(s=r[0], e=r[1], n=short(strs.get(r[2])), g=(r[3], r[4], r[5]), b=(r[6], r[7], r[8])) for r in rows]

    gaps = [ks[i + 1]["s"] - ks[i]["e"] for i in range(len(ks) - 1)]
    print(f"kernels={len(ks)}  span={(ks[-1]['e'] - ks[0]['s']) / 1e9:.3f}s")

    # A long host-side gap separates decode calls.
    THRESH = 300_000
    tokens, cur = [], [ks[0]]
    for i, g in enumerate(gaps):
        if g > THRESH:
            tokens.append(cur)
            cur = []
        cur.append(ks[i + 1])
    tokens.append(cur)
    print("kernels per cluster (first 8):", [len(t) for t in tokens[:8]], "clusters:", len(tokens))
    mode = st.mode(len(t) for t in tokens)
    dec = [t for t in tokens if abs(len(t) - mode) <= max(4, mode // 50)]
    dec = dec[2:] if len(dec) > 4 else dec  # skip warm start
    print(f"modal kernels/token={mode}  decode tokens used={len(dec)}")

    busy, wall, gap_sum, per = [], [], [], defaultdict(lambda: [0, 0])
    for i, t in enumerate(dec):
        w = t[-1]["e"] - t[0]["s"]
        b, end = 0, t[0]["s"]
        for k in t:
            s = max(k["s"], end)
            if k["e"] > s:
                b += k["e"] - s
            end = max(end, k["e"])
            key = (k["n"], k["g"], k["b"])
            per[key][0] += 1
            per[key][1] += k["e"] - k["s"]
        busy.append(b)
        wall.append(w)
        gap_sum.append(w - b)
    n = len(dec)
    print(f"\ntoken GPU span  mean {st.mean(wall) / 1e6:.2f} ms | kernel busy {st.mean(busy) / 1e6:.2f} ms | gaps between kernels {st.mean(gap_sum) / 1e6:.2f} ms")
    # Token-to-token period, includes host time between decode calls.
    periods = [dec[i + 1][0]["s"] - dec[i][0]["s"] for i in range(n - 1)]
    if periods:
        print(f"token period    median {st.median(periods) / 1e6:.2f} ms ({1e9 / st.median(periods):.1f} tok/s under tracing)")

    print(f"\n{'kernel':<44}{'grid':<18}{'block':<14}{'#/tok':>7}{'us each':>9}{'ms/tok':>8}{'%busy':>7}")
    total_busy = st.mean(busy)
    for key, (cnt, dur) in sorted(per.items(), key=lambda kv: -kv[1][1])[:24]:
        print(f"{key[0][:43]:<44}{str(key[1]):<18}{str(key[2]):<14}{cnt / n:>7.1f}{dur / cnt / 1e3:>9.1f}{dur / n / 1e6:>8.3f}{dur / n / total_busy * 100:>7.1f}")

    def where(pred):
        cnt = sum(v[0] for k, v in per.items() if pred(k)) / n
        ms = sum(v[1] for k, v in per.items() if pred(k)) / n / 1e6
        return cnt, ms

    print("\n-- by routed-expert grid (gridY == 8 means k experts in one launch) --")
    cnt, ms = where(lambda k: k[0].startswith("mul_mat_vec_q") and k[1][1] == 8)
    print(f"expert GEMV launches/token {cnt:.1f}, {ms:.2f} ms/token")
    for key, (cnt, dur) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        if key[1][1] == 8 and "mul_mat" in key[0]:
            print(f"  {key[0][:48]:<50}{str(key[1]):<16}{cnt / n:>6.1f}/tok {dur / cnt / 1e3:>8.1f} us {dur / n / 1e6:>7.3f} ms/tok")

    print("\n-- other activity per token --")
    for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "sync")):
        try:
            cnt, dur = c.execute(f"select count(*), coalesce(sum(end-start),0) from {table} where start>={dec[0][0]['s']} and end<={dec[-1][-1]['e']}").fetchone()
            print(f"{label:<8} {cnt / n:>7.1f}/tok {dur / n / 1e6:>8.3f} ms/tok")
        except sqlite3.Error as e:
            print(label, "n/a", e)
    q = (
        "select s.value, count(*), sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
        f"where r.start>={dec[0][0]['s']} and r.end<={dec[-1][-1]['e']} group by s.value order by 3 desc limit 8"
    )
    for name, cnt, dur in c.execute(q):
        print(f"  {name[:40]:<42}{cnt / n:>8.1f}/tok {dur / n / 1e6:>8.3f} ms/tok")

    print("\n-- gap distribution inside a token (us) --")
    inner = [dec[0][i + 1]["s"] - dec[0][i]["e"] for i in range(len(dec[0]) - 1)]
    inner = sorted(inner)
    print("median", inner[len(inner) // 2] / 1e3, "p90", inner[int(len(inner) * 0.9)] / 1e3, "max", inner[-1] / 1e3)


if __name__ == "__main__":
    main(sys.argv[1])
