#!/usr/bin/env python3
"""Weight bytes read per decoded token for a safetensors MoE checkpoint (Qwen3-MoE layout), from the file header only.

Counts every tensor once, except: routed-expert tensors count top_k / n_experts of their bytes (8 of 128 experts per
token), the token embedding counts one row, and GPTQ g_idx tensors are skipped (not read with desc_act=False).
Usage: bytes_per_token_st.py MODEL_DIR [TOP_K] [N_EXPERTS]
"""
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

DTYPE_BYTES = {'F16': 2, 'BF16': 2, 'F32': 4, 'I32': 4, 'I64': 8, 'I8': 1, 'U8': 1, 'F64': 8, 'I16': 2}


def main():
    d = Path(sys.argv[1])
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_exp = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    files = sorted(d.glob('*.safetensors'))
    total, read = 0, 0
    groups = defaultdict(lambda: [0, 0])
    for f in files:
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, t in hdr.items():
            if name == '__metadata__':
                continue
            nbytes = t['data_offsets'][1] - t['data_offsets'][0]
            total += nbytes
            if name.endswith('g_idx'):
                share, g = 0.0, 'g_idx (skipped)'
            elif '.mlp.experts.' in name:
                share, g = top_k / n_exp, 'routed experts (8/128)'
            elif 'embed_tokens' in name:
                vocab = t['shape'][0]
                share, g = 1 / vocab, 'embedding (one row)'
            elif name.startswith('lm_head'):
                share, g = 1.0, f'lm_head ({t["dtype"]})'
            elif '.self_attn.' in name:
                share, g = 1.0, 'attention'
            elif '.mlp.gate.' in name:
                share, g = 1.0, f'router ({t["dtype"]})'
            else:
                share, g = 1.0, 'norms and other'
            groups[g][0] += nbytes
            groups[g][1] += nbytes * share
            read += nbytes * share
    print(f'{d.name}: file tensors {total / 1e9:.3f} GB, read per token {read / 1e9:.3f} GB')
    for g, (tot, rd) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f'  {g:<28} stored {tot / 1e9:7.3f} GB   read/token {rd / 1e9:6.3f} GB')
    print(f'  ceiling at 180 GB/s: {180e9 / read:.1f} tokens/s')


if __name__ == '__main__':
    main()
