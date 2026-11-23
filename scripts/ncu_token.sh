#!/usr/bin/env bash
# Profile every kernel of one steady decode token with Nsight Compute.
# Usage: sudo scripts/ncu_token.sh LLAMA_BENCH LABEL
# Run as root (this Orin's ncu needs it). Pin clocks first (sudo jetson_clocks) and keep the board otherwise idle:
# CPU work shares DRAM with the GPU on Jetson, so a build running alongside slows memory-bound kernels.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
BENCH=$1
LABEL=$2
NCU=/opt/nvidia/nsight-compute/2024.3.1/ncu
MODEL=/media/vlsi/NVMESSD/gguf/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/media/vlsi/NVMESSD/fused-moe/profiling/ncu
OWNER=${SUDO_USER:-vlsi}
mkdir -p "$OUT"
trap 'chown -R "$OWNER": "$OUT"' EXIT

# Kernels before the third steady decode token, counted with nsys for "-p 0 -n 4 -r 1" with CUDA graphs off.
# A token is 48 layers of the same 23 kernels plus 6, and kernel replay saves and restores device memory
# (about 6.6 s per kernel here, since the model holds 17 GB), so profile two layers by default.
# COUNT=1110 covers the whole token in about two hours; SKIP=4411 COUNT=29 covers the last layer and output head.
SKIP=${SKIP:-3330}
COUNT=${COUNT:-46}

"$NCU" --list-sections > "$OUT/sections.txt" 2>&1 || true
"$NCU" --query-metrics > "$OUT/metrics.txt" 2>&1 || true
have_section() { grep -qE "^\s*$1\b" "$OUT/sections.txt"; }
have_metric() { grep -qE "^$1\s" "$OUT/metrics.txt"; }

SECTIONS=()
for s in SpeedOfLight LaunchStats Occupancy WarpStateStats SchedulerStats ComputeWorkloadAnalysis InstructionStats MemoryWorkloadAnalysis; do
  if have_section "$s"; then SECTIONS+=(--section "$s"); else echo "section $s not available, skipped"; fi
done
# No dram__ metrics on this iGPU: L2 fills from system memory stand in for DRAM reads.
METRICS=gpu__time_duration.sum
for m in lts__d_sectors_fill_sysmem lts__t_sectors_aperture_sysmem_op_read sm__ctas_launched smsp__warps_launched; do
  if have_metric "$m"; then METRICS+=",$m.sum"; else echo "metric $m not available, skipped"; fi
done

COMMON=(--target-processes all
  --clock-control none  # keep the jetson_clocks setting; the default locks the GPU to its base clock
  --cache-control all   # flush caches between replays, as in real decode where weights come from DRAM
  "${SECTIONS[@]}" --metrics "$METRICS")
RUN=(env GGML_CUDA_DISABLE_GRAPHS=1 "$BENCH" -m "$MODEL" -ngl 99 -p 0 -n 4 -r 1 -o md)

echo "== smoke test: one kernel =="
"$NCU" "${COMMON[@]}" --launch-skip "$SKIP" --launch-count 1 -f -o "$OUT/${LABEL}_smoke" "${RUN[@]}" > "$OUT/${LABEL}_smoke.log" 2>&1
grep -E "==ERROR==|==WARNING==" "$OUT/${LABEL}_smoke.log" | head -5 || true

echo "== $COUNT kernels from launch $SKIP, about $(( COUNT * 7 / 60 + 1 )) minutes =="
start=$(date +%s)
"$NCU" "${COMMON[@]}" --launch-skip "$SKIP" --launch-count "$COUNT" -f -o "$OUT/${LABEL}_token" "${RUN[@]}" > "$OUT/${LABEL}_token.log" 2>&1
echo "done in $(( $(date +%s) - start )) s: $OUT/${LABEL}_token.ncu-rep"
grep -cE "==PROF==" "$OUT/${LABEL}_token.log" || true
grep -E "==ERROR==" "$OUT/${LABEL}_token.log" | head -5 || true
