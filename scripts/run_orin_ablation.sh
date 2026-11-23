#!/usr/bin/env bash
# The separation experiment on Orin (same as run_ablation_3090.sh): tests with the SM-aware knobs, the dense GEMV
# sweep under nsys, and the real-model ablation. Needs build/master_bb rebuilt with the knobs. Run inside tmux.
set -uo pipefail
W=/home/vlsi/fused-moe
B=/media/vlsi/NVMESSD/fused-moe/build
O=/media/vlsi/NVMESSD/fused-moe/profiling/ablation_orin
mkdir -p "$O/tests"
export PEAK_GBS=180

g++ -O2 -std=c++17 -I$W/llama.cpp-bb-master/ggml/include $W/bench/dense_shapes.cpp -L$B/master_bb/bin -lggml -lggml-base -lggml-cuda \
    -Wl,-rpath,$B/master_bb/bin -o $W/bench/dense_shapes_master_bb || exit 1

echo "$(date '+%T') tests"
for op in MUL_MAT MUL_MAT_ID MUL_MAT_FUSION; do
  GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MMVQ_MAXNW=4 GGML_CUDA_MMVQ_MINBLK=4 GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 \
    "$B/master_bb/bin/test-backend-ops" test -b CUDA0 -o $op 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$O/tests/$op.txt"
  echo "  nw4 mb4 $op: $(grep -E 'tests passed' "$O/tests/$op.txt" | tail -1), FAIL lines $(grep -c FAIL "$O/tests/$op.txt")"
done

echo "$(date '+%T') dense sweep"
BASE_BIN=$W/bench/dense_shapes_master_base BB_BIN=$W/bench/dense_shapes_master_bb OUT=$O/dense bash $W/scripts/dense_sweep.sh

echo "$(date '+%T') model ablation"
BASE_BENCH=$B/master_base/bin/llama-bench BB_BENCH=$B/master_bb/bin/llama-bench MODELS=/media/vlsi/NVMESSD/gguf OUT=$O/models \
  bash $W/scripts/model_ablation.sh
echo "done $(date '+%F %T')"
