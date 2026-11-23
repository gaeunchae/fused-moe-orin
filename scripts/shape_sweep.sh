#!/usr/bin/env bash
# Measure every forced GEMV block shape (warps x rows) for plain (down-like) and fused (gate+up) single-token
# MUL_MAT_ID in all 13 formats and 5 row lengths, plus the current patch and the dedicated kernels as references.
# Run inside tmux. Output: profiling/shapes/<variant>_<config>.txt
set -uo pipefail
BIN=/home/vlsi/fused-moe/bench/mmid_formats_bb
O=/media/vlsi/NVMESSD/fused-moe/profiling/shapes
mkdir -p "$O"
for v in plain fused; do
  GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 "$BIN" all $v > "$O/${v}_bb_v1.txt" 2>&1
  GGML_CUDA_MOE_GLU=1 GGML_CUDA_MOE_DOWN=1 "$BIN" all $v > "$O/${v}_dedicated.txt" 2>&1
  for nw in 1 2 4; do
    for r in 1 2 4 8; do
      echo "$(date '+%T') $v ${nw}x${r}"
      GGML_CUDA_MMVQ_SHAPE=$nw,$r GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 "$BIN" all $v > "$O/${v}_${nw}x${r}.txt" 2>&1
    done
  done
done
echo "done $(date '+%F %T')"
