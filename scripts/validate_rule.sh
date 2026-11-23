#!/usr/bin/env bash
# Validate the bytes-based GEMV shape at one target: CPU-checked tests, the format benchmark (plain and fused),
# then decode speed of four models. Run inside tmux. Usage: scripts/validate_rule.sh [BYTES, default 8192]
set -uo pipefail
BYTES=${1:-8192}
B=/media/vlsi/NVMESSD/fused-moe/build/bb/bin
O=/media/vlsi/NVMESSD/fused-moe/profiling/rule_$BYTES
G=/media/vlsi/NVMESSD/gguf
mkdir -p "$O"
echo "$(date '+%T') tests"
for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
  GGML_CUDA_MMVQ_BYTES=$BYTES GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 "$B/test-backend-ops" test -b CUDA0 -o $op 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' > "$O/test_$op.txt"
  echo "  $op: $(grep -E 'tests passed' "$O/test_$op.txt" | tail -1), FAIL lines $(grep -c FAIL "$O/test_$op.txt")"
done
echo "$(date '+%T') formats"
for v in plain fused; do
  GGML_CUDA_MMVQ_BYTES=$BYTES GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 /home/vlsi/fused-moe/bench/mmid_formats_bb all $v > "$O/formats_$v.txt" 2>&1
done
echo "$(date '+%T') models"
for m in Qwen3-30B-A3B Qwen3.5-35B-A3B OLMoE-1B-7B-0125-Instruct Qwen3-8B; do
  echo "######## $m"
  MODEL=$G/$m-Q4_K_M.gguf OUT=$O/$m BYTES=$BYTES CONFIGS="bb_v1 bb_all bytes bytes_all" \
    bash "$(dirname "$0")/ab_bench.sh" 0,1024,4096
done
echo "done $(date '+%F %T')"
