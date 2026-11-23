#!/usr/bin/env bash
# A/B decode speed: base build, then the patched build with each MoE kernel switched off and on.
# Usage: [MODEL=file.gguf] [OUT=dir] [CONFIGS="base bb_v1 ..."] [BYTES=4096] scripts/ab_bench.sh [depths]
#   defaults: Qwen3-30B-A3B, profiling/ab, configs "base bb_v1 bb_glu bb_down bb_all", depths "0,1024,4096".
#   bytes and bytes_all switch on the bytes-based GEMV block shape (GGML_CUDA_MMVQ_BYTES=$BYTES).
#   BASE_BUILD / BB_BUILD pick other build folders (default build/base and build/bb).
set -euo pipefail
B=/media/vlsi/NVMESSD/fused-moe/build
BASE=${BASE_BUILD:-$B/base}
BB=${BB_BUILD:-$B/bb}
MODEL=${MODEL:-/media/vlsi/NVMESSD/gguf/Qwen3-30B-A3B-Q4_K_M.gguf}
OUT=${OUT:-/media/vlsi/NVMESSD/fused-moe/profiling/ab}
DEPTHS=${1:-0,1024,4096}
CONFIGS=${CONFIGS:-base bb_v1 bb_glu bb_down bb_all}
BYTES=${BYTES:-4096}
mkdir -p "$OUT"
run() {
  local label=$1 bin=$2
  shift 2
  echo "== $label"
  env "$@" "$bin" -m "$MODEL" -ngl 99 -p 0 -n 64 -d "$DEPTHS" -r 5 -o csv 2> /dev/null | tee "$OUT/$label.csv" \
    | awk -F, 'NR > 1 { gsub(/"/, ""); printf "   depth %-6s %7.2f +- %.2f tok/s\n", $(NF-5), $(NF-1), $NF }'
}
for c in $CONFIGS; do
  case $c in
    base)      run base      $BASE/bin/llama-bench GGML_CUDA_MOE_GLU=0 ;;
    bb_v1)     run bb_v1     $BB/bin/llama-bench   GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 ;;
    bb_glu)    run bb_glu    $BB/bin/llama-bench   GGML_CUDA_MOE_GLU=1 GGML_CUDA_MOE_DOWN=0 ;;
    bb_down)   run bb_down   $BB/bin/llama-bench   GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=1 ;;
    bb_all)    run bb_all    $BB/bin/llama-bench   GGML_CUDA_MOE_GLU=1 GGML_CUDA_MOE_DOWN=1 ;;
    bytes)     run bytes     $BB/bin/llama-bench   GGML_CUDA_MMVQ_BYTES=$BYTES GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 ;;
    bytes_all) run bytes_all $BB/bin/llama-bench   GGML_CUDA_MMVQ_BYTES=$BYTES GGML_CUDA_MOE_GLU=1 GGML_CUDA_MOE_DOWN=1 ;;
    *)         echo "unknown config: $c" >&2; exit 1 ;;
  esac
done
