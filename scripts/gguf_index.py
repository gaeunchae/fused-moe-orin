#!/usr/bin/env python3
"""Write name, absolute file offset and byte size of expert tensors. Run with: python3 -I gguf_index.py FILE OUT"""
import re
import struct
import sys

BLOCK = {12: (144, 256), 14: (210, 256), 0: (4, 1), 1: (2, 1), 30: (2, 1)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_str(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def read_val(f, t):
    if t in SCALARS:
        return read(f, SCALARS[t])
    if t == 8:
        return read_str(f)
    et, n = read(f, "I"), read(f, "Q")
    if et in SCALARS:
        f.seek(struct.calcsize("<" + SCALARS[et]) * n, 1)
    else:
        for _ in range(n):
            read_val(f, et)
    return None


with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    read(f, "I")
    n_tensors, n_kv = read(f, "Q"), read(f, "Q")
    meta = {}
    for _ in range(n_kv):
        k = read_str(f)
        meta[k] = read_val(f, read(f, "I"))
    infos = []
    for _ in range(n_tensors):
        name = read_str(f)
        dims = [read(f, "Q") for _ in range(read(f, "I"))]
        ty = read(f, "I")
        infos.append((name, dims, ty, read(f, "Q")))
    align = meta.get("general.alignment", 32)
    data = (f.tell() + align - 1) // align * align

with open(sys.argv[2], "w") as out:
    for name, dims, ty, off in infos:
        if not re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight$", name) or ty not in BLOCK:
            continue
        n = 1
        for d in dims:
            n *= d
        bs, es = BLOCK[ty]
        out.write(f"{name}\t{data + off}\t{n * bs // es}\t{ty}\t{'x'.join(map(str, dims))}\n")
print("indexed", sum(1 for _ in open(sys.argv[2])))
