#!/usr/bin/env bash
# Compare our GEMV changes with the latest llama.cpp master (which has its own Orin table since 2026-11-20).
# 1) CPU-checked tests, rule only and with the dedicated kernels. 2) format benchmark, master vs rule.
# 3) decode speed of four models: master, master + dedicated kernels, + rule, + both. Run inside tmux.
set -uo pipefail
W=/home/vlsi/fused-moe
B=/media/vlsi/NVMESSD/fused-moe/build
O=/media/vlsi/NVMESSD/fused-moe/profiling/master
G=/media/vlsi/NVMESSD/gguf
mkdir -p "$O"
echo "master $(git -C $W/llama.cpp-master rev-parse --short HEAD)"
echo "$(date '+%T') tests"
for cfg in rule all; do
  if [ $cfg = rule ]; then E="GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0"; else E=""; fi
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_FUSION; do
    env GGML_CUDA_MMVQ_BYTES=8192 $E "$B/master_bb/bin/test-backend-ops" test -b CUDA0 -o $op 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g' > "$O/test_${cfg}_$op.txt"
    echo "  $cfg $op: $(grep -E 'tests passed' "$O/test_${cfg}_$op.txt" | tail -1), FAIL lines $(grep -c FAIL "$O/test_${cfg}_$op.txt")"
  done
done
echo "$(date '+%T') formats"
for v in plain fused; do
  "$W/bench/mmid_formats_master_base" all $v > "$O/formats_master_$v.txt" 2>&1
  GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 "$W/bench/mmid_formats_master_bb" all $v > "$O/formats_rule_$v.txt" 2>&1
done
echo "$(date '+%T') models"
for m in Qwen3-30B-A3B Qwen3.5-35B-A3B OLMoE-1B-7B-0125-Instruct Qwen3-8B; do
  echo "######## $m"
  MODEL=$G/$m-Q4_K_M.gguf OUT=$O/$m BASE_BUILD=$B/master_base BB_BUILD=$B/master_bb BYTES=8192 \
    CONFIGS="base bb_all bytes bytes_all" bash "$W/scripts/ab_bench.sh" 0,1024,4096
done
echo "done $(date '+%F %T')"
