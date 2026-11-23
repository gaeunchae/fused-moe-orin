#!/usr/bin/env bash
# Real-model ablation of the SM-aware knobs: decode speed for each knob alone and together, and the MINBLK value
# from 1 to 16. Configurations are interleaved over ROUNDS rounds so a slow drift of the GPU cannot favour one of
# them. Then nsys kernel summaries per configuration for Qwen3-8B and Qwen3-30B-A3B. Output: OUT/
# Usage: BASE_BENCH=<master llama-bench> BB_BENCH=<patched llama-bench> MODELS=<gguf dir> OUT=<dir> scripts/model_ablation.sh
set -uo pipefail
S=$(cd "$(dirname "$0")" && pwd)
: "${BASE_BENCH:?}" "${BB_BENCH:?}" "${MODELS:?}" "${OUT:?}"
ROUNDS=${ROUNDS:-3}
MODEL_LIST=${MODEL_LIST:-Qwen3-8B Qwen3-30B-A3B Qwen3.5-35B-A3B OLMoE-1B-7B-0125-Instruct}
mkdir -p "$OUT/csv" "$OUT/nsys"
OFF="GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0"   # the rule covers every GEMV; dedicated MoE kernels off
declare -A CFG=(
  [base]=""
  [r8]="GGML_CUDA_MMVQ_BYTES=8192 $OFF"
  [r8_nw4]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 $OFF"
  [r8_mb4]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MINBLK=4 $OFF"
  [r8_nw4_mb1]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=1 $OFF"
  [r8_nw4_mb2]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=2 $OFF"
  [r8_nw4_mb4]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=4 $OFF"
  [r8_nw4_mb8]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=8 $OFF"
  [r8_nw4_mb16]="GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=16 $OFF"
)
ORDER="base r8 r8_nw4 r8_mb4 r8_nw4_mb1 r8_nw4_mb2 r8_nw4_mb4 r8_nw4_mb8 r8_nw4_mb16"
bin_of() { if [ "$1" = base ]; then echo "$BASE_BENCH"; else echo "$BB_BENCH"; fi; }

for round in $(seq 1 "$ROUNDS"); do
  for m in $MODEL_LIST; do
    f=$MODELS/$m-Q4_K_M.gguf
    [ -f "$f" ] || { echo "skip $m"; continue; }
    for c in $ORDER; do
      env ${CFG[$c]} "$(bin_of $c)" -m "$f" -ngl 99 -p 0 -n 64 -d 0 -r 5 -o csv 2> /dev/null > "$OUT/csv/${m}__${c}__r$round.csv"
      echo "$(date '+%T') round $round $m $c $(awk -F, 'NR > 1 { gsub(/"/, ""); printf "%.2f", $(NF-1) }' "$OUT/csv/${m}__${c}__r$round.csv")" | tee -a "$OUT/decode.log"
    done
  done
done

for m in Qwen3-8B Qwen3-30B-A3B; do
  f=$MODELS/$m-Q4_K_M.gguf
  [ -f "$f" ] || continue
  for c in base r8 r8_nw4 r8_mb4 r8_nw4_mb4; do
    env ${CFG[$c]} nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
      -o "$OUT/nsys/${m}__$c" "$(bin_of $c)" -m "$f" -ngl 99 -p 0 -n 32 -d 0 -r 1 -o md > /dev/null 2>&1
    nsys export --type=sqlite --force-overwrite=true -o "$OUT/nsys/${m}__$c.sqlite" "$OUT/nsys/${m}__$c.nsys-rep" > /dev/null 2>&1
    python3 "$S/kernel_summary.py" "$OUT/nsys/${m}__$c.sqlite" --top 40 > "$OUT/nsys/${m}__${c}_kernels.txt" 2>&1
    rm -f "$OUT/nsys/${m}__$c.nsys-rep" "$OUT/nsys/${m}__$c.sqlite"
    echo "$(date '+%T') nsys $m $c"
  done
done
echo "done $(date '+%F %T')"
