#!/usr/bin/env bash
# Accuracy of the final GEMV settings against the base build, on the single-token path the patch changes.
# 1) logits: teacher-forced 28 prompts (1,792 tokens) through bench/moe_probe_bb, compared with the saved base runs.
# 2) perplexity: wikitext-2 test, 20 chunks of 512, -ub 1 so every token takes the single-token GEMV path.
# Run inside tmux (about 40 min). Output: profiling/accuracy/
set -uo pipefail
W=/home/vlsi/fused-moe
B=/media/vlsi/NVMESSD/fused-moe/build
G=/media/vlsi/NVMESSD/gguf
O=/media/vlsi/NVMESSD/fused-moe/profiling/accuracy
R=/media/vlsi/NVMESSD/fused_moe_baseline/sparsity_probe
C=$R/run1/control
WIKI=/media/vlsi/NVMESSD/fused-moe/data/wikitext-2-raw/wiki.test.raw
mkdir -p "$O"

echo "$(date '+%T') logits"
cd "$C"
GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MOE_GLU=0 GGML_CUDA_MOE_DOWN=0 PROBE_NO_SPLIT=1 PROBE_SAVE=$O/bytes.bin \
  "$W/bench/moe_probe_bb" "$G/Qwen3-30B-A3B-Q4_K_M.gguf" "$R/prompts.tsv" . mask 64 none:0 > /dev/null 2>&1
GGML_CUDA_MMVQ_BYTES=8192 PROBE_NO_SPLIT=1 PROBE_SAVE=$O/bytes_all.bin \
  "$W/bench/moe_probe_bb" "$G/Qwen3-30B-A3B-Q4_K_M.gguf" "$R/prompts.tsv" . mask 64 none:0 > /dev/null 2>&1
python3 -I "$W/scripts/compare_logits.py" base_a=$C/base_a.bin base_b=$C/base_b.bin bb_all=$C/bb_all.bin \
  bytes=$O/bytes.bin bytes_all=$O/bytes_all.bin 2>&1 | tee "$O/compare_logits.txt"

echo "$(date '+%T') perplexity"
ppl() {  # label build model env...
  local label=$1 build=$2 model=$3
  shift 3
  env "$@" "$B/$build/bin/llama-perplexity" -m "$G/$model-Q4_K_M.gguf" -f "$WIKI" -ngl 99 -c 512 --chunks 20 -b 512 -ub 1 \
    > "$O/ppl_${model}_$label.txt" 2>&1
  echo "  $model $label: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$O/ppl_${model}_$label.txt")"
}
for m in Qwen3-30B-A3B Qwen3.5-35B-A3B OLMoE-1B-7B-0125-Instruct; do
  ppl base      base $m
  ppl bytes_all bb   $m GGML_CUDA_MMVQ_BYTES=8192
done
ppl bb_all bb Qwen3-30B-A3B
echo "done $(date '+%F %T')"
