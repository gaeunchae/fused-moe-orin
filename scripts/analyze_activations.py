#!/usr/bin/env python3
"""Channel sparsity statistics from moe_probe's activations.bin. Usage: analyze_activations.py RUN_DIR"""
import sys

import numpy as np

K, N = 8, 768
DT = np.dtype([("step", "i4"), ("il", "i4"), ("ids", "8i4"), ("w", "8f4"), ("gate", f"{K*N}f4"), ("up", f"{K*N}f4")])
SPARS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def silu(x):
    return x / (1.0 + np.exp(-x))


def main(run):
    a = np.memmap(f"{run}/activations.bin", dtype=DT, mode="r")
    steps = int(a["step"].max()) + 1
    print(f"records={len(a)} tokens={steps} layers/token={len(a) / steps:.1f}")

    doms = []
    with open(f"{run}/ref_tokens.txt") as f:
        for line in f:
            t = line.split()
            doms += [t[1]] * (len(t) - 3)
    doms = np.array(doms)
    print("tokens per domain:", {d: int((doms == d).sum()) for d in np.unique(doms)})

    edges = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    cnt = {k: np.zeros(len(edges)) for k in ("gate", "up", "h")}
    layer_gate = np.zeros((48, len(edges)))
    n_vals, n_layer = 0, np.zeros(48)
    energy = {k: np.zeros(len(SPARS)) for k in ("gate", "up", "h", "random")}
    energy_total = 0.0
    dom_energy = {d: np.zeros((len(SPARS), 2)) for d in np.unique(doms)}
    runs = {s: [] for s in SPARS}
    sigma = {s: [[] for _ in range(48)] for s in (0.5, 0.85)}
    jac = {s: [] for s in SPARS}
    rng = np.random.default_rng(0)
    ord_events = {}

    for i in range(len(a)):
        r = a[i]
        il = int(r["il"])
        g = silu(np.asarray(r["gate"]).reshape(K, N))
        u = np.asarray(r["up"]).reshape(K, N)
        h = g * u
        ag, au, ah = np.abs(g), np.abs(u), np.abs(h)
        w2 = np.asarray(r["w"])[:, None] ** 2
        for name, v in (("gate", ag), ("up", au), ("h", ah)):
            cnt[name] += [(v < e).sum() for e in edges]
        layer_gate[il] += [(ag < e).sum() for e in edges]
        n_layer[il] += K * N
        n_vals += K * N
        e2 = w2 * h * h
        tot = e2.sum()
        energy_total += tot
        d = doms[int(r["step"])]
        for si, s in enumerate(SPARS):
            keep = N - int(N * s + 0.5)
            ig = np.argpartition(-ag, keep - 1, axis=1)[:, :keep]
            iu = np.argpartition(-au, keep - 1, axis=1)[:, :keep]
            ih = np.argpartition(-ah, keep - 1, axis=1)[:, :keep]
            eg = np.take_along_axis(e2, ig, 1).sum()
            energy["gate"][si] += eg
            energy["up"][si] += np.take_along_axis(e2, iu, 1).sum()
            energy["h"][si] += np.take_along_axis(e2, ih, 1).sum()
            energy["random"][si] += tot * keep / N
            dom_energy[d][si] += [eg, tot]
            if i % 8 == 0:
                mg = np.zeros((K, N), bool)
                np.put_along_axis(mg, ig, True, 1)
                mu = np.zeros((K, N), bool)
                np.put_along_axis(mu, iu, True, 1)
                jac[s].append((mg & mu).sum() / max(1, (mg | mu).sum()))
                edge = np.diff(np.concatenate([np.zeros((K, 1), bool), mg], 1).astype(np.int8), axis=1)
                runs[s].append(keep * K / max(1, (edge == 1).sum()))
            if s in sigma and i % 8 == 0:
                sigma[s][il].append(float(np.median(np.partition(ag, N - keep, axis=1)[:, N - keep])))
        if i % 16 == 0:
            keep = N - int(N * 0.85 + 0.5)
            ig = np.argpartition(-ag, keep - 1, axis=1)[:, :keep]
            for e in range(K):
                ord_events.setdefault((il, int(r["ids"][e])), []).append(np.sort(ig[e]).astype(np.int16))

    print("\n== Fraction of values below threshold ==")
    print(f"{'|x| <':<10}" + "".join(f"{e:>9}" for e in edges))
    for k, label in (("gate", "|SiLU(g)|"), ("up", "|u|"), ("h", "|h|")):
        print(f"{label:<10}" + "".join(f"{v / n_vals * 100:>8.1f}%" for v in cnt[k]))

    print("\n== |SiLU(g)| < 0.003 / 0.03 by layer group ==")
    for lo, hi in ((0, 12), (12, 24), (24, 36), (36, 48)):
        t = layer_gate[lo:hi].sum(0) / n_layer[lo:hi].sum()
        print(f"layers {lo:>2}-{hi - 1:<2}: {t[1] * 100:5.1f}% / {t[3] * 100:5.1f}%")

    print("\n== Output energy of h retained when keeping the top channels per event (router-weighted) ==")
    print(f"{'skipped':<10}" + "".join(f"{k:>10}" for k in ("gate", "up", "oracle h", "random")) + f"{'jaccard g/u':>13}{'mean run':>10}")
    for si, s in enumerate(SPARS):
        vals = [energy[k][si] / energy_total * 100 for k in ("gate", "up", "h", "random")]
        print(f"{s:<10.2f}" + "".join(f"{v:>9.1f}%" for v in vals) + f"{np.mean(jac[s]):>13.3f}{np.mean(runs[s]):>10.2f}")

    print("\n== Gate-mask energy retained by domain ==")
    for d, v in dom_energy.items():
        print(f"{d:<8}" + "".join(f"  {int(s * 100)}%:{v[si, 0] / v[si, 1] * 100:5.1f}" for si, s in enumerate(SPARS)))

    print("\n== Per-layer |SiLU(g)| cutoff (median over events) ==")
    for s in sigma:
        m = np.array([np.median(x) for x in sigma[s] if x])
        print(f"skip {int(s * 100)}%: min {m.min():.4f} median {np.median(m):.4f} max {m.max():.4f}")

    # Channel reordering by frequency, learned on the first half of events and scored on the second half.
    before, after, n_exp, ideal = [], [], 0, 115 / 1
    for key, ev in ord_events.items():
        if len(ev) < 8:
            continue
        n_exp += 1
        cut = len(ev) // 2
        freq = np.zeros(N)
        for m in ev[:cut]:
            freq[m] += 1
        rank = np.empty(N, np.int16)
        rank[np.argsort(-freq, kind="stable")] = np.arange(N, dtype=np.int16)
        for m in ev[cut:]:
            d0 = np.diff(m)
            before.append(len(m) / (1 + (d0 != 1).sum()))
            m2 = np.sort(rank[m])
            d1 = np.diff(m2)
            after.append(len(m2) / (1 + (d1 != 1).sum()))
    if before:
        print(f"\n== Frequency reordering at 85% skipped, gate mask ({n_exp} experts, held-out events) ==")
        print(f"mean run length: checkpoint {np.mean(before):.2f}, reordered {np.mean(after):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
