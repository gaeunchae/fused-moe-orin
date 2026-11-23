#!/usr/bin/env python3
"""Weight bytes read per decoded token, from a GGUF header. Run with: python3 -I gguf_bytes.py FILE"""
import json
import re
import struct
import sys
from collections import Counter, defaultdict

# Bytes per block and elements per block for the ggml types a Q4_K_M file can hold.
TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 7: ("Q8_0", 34, 32),
         12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 30: ("BF16", 2, 1)}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_str(f):
    n = read(f, "Q")
    return f.read(n).decode("utf-8", "replace")


SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_val(f, t):
    if t in SCALARS:
        return read(f, SCALARS[t])
    if t == 8:
        return read_str(f)
    if t == 9:
        et, n = read(f, "I"), read(f, "Q")
        if et in SCALARS:
            f.seek(struct.calcsize("<" + SCALARS[et]) * n, 1)
        else:
            for _ in range(n):
                read_val(f, et)
        return None
    raise ValueError(f"unknown gguf value type {t}")


def main(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        read(f, "I")
        n_tensors, n_kv = read(f, "Q"), read(f, "Q")
        meta = {}
        for _ in range(n_kv):
            k = read_str(f)
            meta[k] = read_val(f, read(f, "I"))
        tensors = {}
        for _ in range(n_tensors):
            name = read_str(f)
            nd = read(f, "I")
            dims = [read(f, "Q") for _ in range(nd)]
            ty = read(f, "I")
            read(f, "Q")
            label, bs, es = TYPES[ty]
            n = 1
            for d in dims:
                n *= d
            tensors[name] = (dims, label, n * bs // es)

    arch = meta["general.architecture"]
    k, n_exp = meta[f"{arch}.expert_used_count"], meta[f"{arch}.expert_count"]
    per = defaultdict(int)
    types = defaultdict(Counter)
    for name, (dims, label, nbytes) in tensors.items():
        m = re.match(r"blk\.\d+\.(.+)\.weight$", name)
        short = m.group(1) if m else name
        read_bytes = nbytes * k // n_exp if short.endswith("_exps") else nbytes
        if name == "token_embd.weight":
            read_bytes = 0  # one row per token
        per[short] += read_bytes
        types[short][label] += 1

    groups = {
        "moe_experts": ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
        "router": ["ffn_gate_inp"],
        "attention": ["attn_q", "attn_k", "attn_v", "attn_output"],
        "lm_head": ["output.weight"],
    }
    total = sum(per.values())
    print(json.dumps({"arch": arch, "file_type": meta.get("general.file_type"), "experts": n_exp, "top_k": k,
                      "layers": meta[f"{arch}.block_count"]}))
    print(f"{'tensor':<22}{'types':<30}{'MB read/token':>14}")
    for short in sorted(per, key=lambda s: -per[s]):
        if per[short]:
            print(f"{short:<22}{str(dict(types[short])):<30}{per[short] / 1e6:>14.1f}")
    print(f"\ntotal weight bytes read per token: {total / 1e9:.3f} GB")
    for g, names in groups.items():
        b = sum(per[n] for n in names)
        print(f"  {g:<12} {b / 1e9:6.3f} GB  {b / total * 100:5.1f}%")
    exps = tensors[next(n for n in tensors if n.endswith("ffn_gate_exps.weight"))]
    print("one gate_exps tensor:", exps[0], exps[1], f"{exps[2] / n_exp / 1e6:.2f} MB/expert")


if __name__ == "__main__":
    main(sys.argv[1])
