#!/usr/bin/env bash
# Build llama.cpp at the latest master (llama.cpp-master) and the same master with our patch (llama.cpp-bb-master).
# Same flags as build/base and build/bb. Run inside tmux; a full CUDA build takes a while on Orin.
set -euo pipefail
W=/home/vlsi/fused-moe
B=/media/vlsi/NVMESSD/fused-moe/build
FLAGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release"
cmake -S $W/llama.cpp-master    -B $B/master_base $FLAGS > /dev/null
cmake -S $W/llama.cpp-bb-master -B $B/master_bb   $FLAGS -DLLAMA_BUILD_TESTS=ON > /dev/null
echo "$(date '+%T') base";  cmake --build $B/master_base --target llama-bench llama-perplexity -j 8 2>&1 | tail -1
echo "$(date '+%T') bb";    cmake --build $B/master_bb   --target llama-bench llama-perplexity test-backend-ops -j 8 2>&1 | tail -1
echo "done $(date '+%F %T')"
