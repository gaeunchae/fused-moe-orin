#!/usr/bin/env bash
# Trace batch-1 decode at several context depths and split each token's time against the transfer floor.
# Usage: scripts/nsys_depth_sweep.sh LLAMA_BENCH LABEL [depths...]
# Before running: clocks pinned (sudo jetson_clocks), no builds or other GPU jobs running.
set -euo pipefail
BENCH=$1
LABEL=$2
shift 2
DEPTHS=${*:-0 1024 4096 16384}
MODEL=/media/vlsi/NVMESSD/gguf/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/media/vlsi/NVMESSD/fused-moe/profiling/$LABEL
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for d in $DEPTHS; do
  # --cuda-graph-trace=node: llama.cpp replays CUDA graphs; without it a whole graph shows as one block.
  # --sample=none --cpuctxsw=none: keep tracing overhead low; the host-side sampling run is separate.
  nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$OUT/d$d" "$BENCH" -m "$MODEL" -ngl 99 -p 0 -n 32 -d "$d" -r 1 -o md > "$OUT/d$d.bench.md" 2> "$OUT/d$d.bench.err"
  nsys export --type=sqlite --force-overwrite=true -o "$OUT/d$d.sqlite" "$OUT/d$d.nsys-rep" > /dev/null 2>&1
  python3 -I "$SCRIPTS/profile_breakdown.py" "$OUT/d$d.sqlite" "$d" --csv "$OUT/d$d.csv" | tee "$OUT/d$d.txt"
  echo
done
