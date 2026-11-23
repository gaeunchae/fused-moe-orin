#!/usr/bin/env python3
"""Pairwise agreement between saved logit files. Usage: compare_logits.py NAME=PATH ..."""
import itertools
import sys

import numpy as np

V = 151936


def load(path):
    return np.memmap(path, dtype=np.float32, mode="r").reshape(-1, V)


def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max())
    p /= p.sum()
    q = np.exp(q_logits - q_logits.max())
    q /= q.sum()
    m = p > 1e-30
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum())


def main(args):
    files = dict(a.split("=", 1) for a in args)
    data = {k: load(v) for k, v in files.items()}
    print(f"{'pair':<26}{'top1':>9}{'identical':>11}{'max |dlogit|':>14}{'KL mean':>10}{'KL p95':>9}{'flip margin med/max':>22}")
    for x, y in itertools.combinations(data, 2):
        a, b = data[x], data[y]
        n = min(len(a), len(b))
        same = flips = exact = 0
        kls, margins, dmax = [], [], 0.0
        for i in range(n):
            ra, rb = np.asarray(a[i]), np.asarray(b[i])
            exact += bool(np.array_equal(ra, rb))
            dmax = max(dmax, float(np.abs(ra - rb).max()))
            ia, ib = int(ra.argmax()), int(rb.argmax())
            same += ia == ib
            if ia != ib:
                flips += 1
                margins.append(float(ra[ia] - ra[ib]))
            kls.append(kl(ra, rb))
        kls = np.sort(kls)
        fm = f"{np.median(margins):.3f}/{np.max(margins):.3f}" if margins else "-"
        print(f"{x + ' vs ' + y:<26}{same / n * 100:>8.2f}%{exact / n * 100:>10.1f}%{dmax:>14.3f}{kls.mean():>10.5f}{kls[int(n * 0.95)]:>9.5f}{fm:>22}")


if __name__ == "__main__":
    main(sys.argv[1:])
