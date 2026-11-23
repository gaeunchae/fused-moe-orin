#!/usr/bin/env bash
# Separate "too few warps" (rows covered in several passes) from "too few blocks" (GPU not filled) on dense
# single-token GEMV. Runs bench/dense_shapes under nsys for master, our rule with and without the SM-aware knobs,
# and all 12 forced shapes, plain and fused; kernel times per configuration go to OUT/dense.csv.
# Usage: BASE_BIN=<dense_shapes linked to master> BB_BIN=<dense_shapes linked to the patched build> \
#        OUT=<dir> PEAK_GBS=<measured peak> scripts/dense_sweep.sh
set -uo pipefail
S=$(cd "$(dirname "$0")" && pwd)
: "${BASE_BIN:?}" "${BB_BIN:?}" "${OUT:?}" "${PEAK_GBS:?}"
mkdir -p "$OUT/raw"
run() {  # label variant binary [env...]
  local label=$1 v=$2 bin=$3
  shift 3
  local base=$OUT/raw/${label}_$v
  env "$@" nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$base" "$bin" all $v > "$base.txt" 2>&1
  nsys export --type=sqlite --force-overwrite=true -o "$base.sqlite" "$base.nsys-rep" > /dev/null 2>&1
  python3 "$S/dense_extract.py" "$base.txt" "$base.sqlite" "$label" "$v" "$PEAK_GBS" "$OUT/dense.csv" || echo "  extract failed: $label $v"
  rm -f "$base.nsys-rep" "$base.sqlite"
  echo "$(date '+%T') $label $v"
}
for v in plain fused; do
  run master     $v "$BASE_BIN"
  run r8         $v "$BB_BIN" GGML_CUDA_MMVQ_BYTES=8192
  run r8_nw4     $v "$BB_BIN" GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4
  run r8_mb4     $v "$BB_BIN" GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MINBLK=4
  run r8_nw4_mb4 $v "$BB_BIN" GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=4
  for nw in 1 2 4; do
    for r in 1 2 4 8; do
      run s${nw}x$r $v "$BB_BIN" GGML_CUDA_MMVQ_SHAPE=$nw,$r
    done
  done
done
