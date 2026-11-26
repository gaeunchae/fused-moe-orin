#!/usr/bin/env bash
# Same-day re-measurement of llama.cpp for the engine comparison: master e300919 and our patch (rule + dedicated
# kernels), Qwen3-30B-A3B Q4_K_M, decode only (-p 0 -n 64), contexts 0 and 4096, two interleaved rounds.
# Run inside tmux, never while another GPU job runs.
set -u
B=/media/vlsi/NVMESSD/fused-moe/baselines
BUILD=/media/vlsi/NVMESSD/fused-moe/build
M=/media/vlsi/NVMESSD/gguf/Qwen3-30B-A3B-Q4_K_M.gguf
O=$B/results/llamacpp
mkdir -p $O
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_start.txt
for r in 1 2; do
  $BUILD/master_base/bin/llama-bench -m $M -ngl 99 -p 0 -n 64 -d 0,4096 -r 5 -o csv > $O/master_r$r.csv 2> $O/master_r$r.err
  GGML_CUDA_MMVQ_BYTES=8192 $BUILD/master_bb/bin/llama-bench -m $M -ngl 99 -p 0 -n 64 -d 0,4096 -r 5 -o csv \
      > $O/ours_r$r.csv 2> $O/ours_r$r.err
done
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_end.txt
echo done
