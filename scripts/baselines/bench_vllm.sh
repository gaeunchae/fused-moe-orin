#!/usr/bin/env bash
# Batch-1 decode speed of vLLM on Jetson AGX Orin for Qwen3-30B-A3B GPTQ INT4, comparable to llama-bench tg:
# one request at a time (--max-concurrency 1), fixed output length with EOS ignored, and decode speed = 1000 / TPOT
# (TPOT excludes the first token, so prefill is not counted). Two contexts: a 64-token prompt (like context 0)
# and a 4096-token prompt (like llama-bench -d 4096). Run inside tmux; nothing else may use the GPU meanwhile.
set -u
B=/media/vlsi/NVMESSD/fused-moe/baselines
V=/media/vlsi/NVMESSD/fused-moe/venv-vllm
M=/media/vlsi/NVMESSD/gguf/gptq/Qwen3-30B-A3B-GPTQ-Int4
O=$B/results/vllm
mkdir -p $O
export LD_LIBRARY_PATH=$(ls -d $B/libcudss-*/lib):${LD_LIBRARY_PATH:-}
# Triton on JetPack 6: the Jetson wheel ships no CUDA headers or sm_87 ptxas, so point it at the system CUDA 12.6
export CUDA_HOME=/usr/local/cuda CPATH=/usr/local/cuda/include TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas PATH=/usr/local/cuda/bin:$PATH
export TMPDIR=$B/tmp HF_HOME=/media/vlsi/NVMESSD/hf_cache VLLM_CACHE_ROOT=$B/vllm-cache
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_start.txt

$V/bin/vllm serve $M --served-model-name qwen3moe --max-model-len 8192 --gpu-memory-utilization 0.6 \
    --max-num-seqs 4 --port 8011 > $O/server.log 2>&1 &
SERVER=$!
for i in $(seq 1 360); do
  curl -s -m 2 localhost:8011/health > /dev/null && break
  kill -0 $SERVER 2> /dev/null || { echo "server died"; exit 1; }
  sleep 5
done
echo "server up after $((i * 5)) s"

run() {  # name input_len output_len prompts
  $V/bin/vllm bench serve --backend vllm --base-url http://localhost:8011 --model qwen3moe --tokenizer $M \
      --dataset-name random --random-input-len $2 --random-output-len $3 --ignore-eos \
      --max-concurrency 1 --num-prompts $4 --seed 0 \
      --save-result --result-dir $O --result-filename $1.json > $O/$1.txt 2>&1
  grep -E "TPOT|ITL|TTFT|Output token throughput" $O/$1.txt
}
run warmup 64 64 3
for r in 1 2 3; do
  run short_r$r 64 128 5
  run long_r$r 4096 128 3
done
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_end.txt
kill $SERVER; wait $SERVER 2> /dev/null
echo done
