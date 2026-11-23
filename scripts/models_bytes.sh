#!/usr/bin/env bash
# Decode speed of four models: base, current patch and the bytes-based GEMV shape. Run it inside tmux so it outlives the editor.
# Usage: scripts/models_bytes.sh   (log: profiling/models_bytes/run.log, CSVs per model and config)
set -uo pipefail
G=/media/vlsi/NVMESSD/gguf
P=/media/vlsi/NVMESSD/fused-moe/profiling/models_bytes
mkdir -p "$P"
for m in Qwen3-30B-A3B Qwen3.5-35B-A3B OLMoE-1B-7B-0125-Instruct Qwen3-8B; do
  echo "######## $m"
  MODEL=$G/$m-Q4_K_M.gguf OUT=$P/$m CONFIGS="base bb_v1 bb_all bytes bytes_all" bash "$(dirname "$0")/ab_bench.sh" 0,1024,4096
done
echo "######## done $(date '+%F %T')"
