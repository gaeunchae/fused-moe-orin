#!/usr/bin/env bash
# Batch-1 decode speed of TensorRT Edge-LLM v0.6.0 on the Orin, Qwen3-30B-A3B-GPTQ-Int4, measured the way NVIDIA's
# performance tables do (llm_inference with warmup runs and --dumpProfile). Inputs are raw text without a chat
# template: 61 tokens (as in NVIDIA's Orin row) and 4096 tokens, greedy, 128 generated tokens.
# Three separate processes per input. Run inside tmux; nothing else may use the GPU meanwhile.
set -u
B=/media/vlsi/NVMESSD/fused-moe/baselines
E=$B/TensorRT-Edge-LLM/build/examples/llm
W=$B/edgellm-ws/Qwen3-30B-A3B-GPTQ-Int4
O=$B/results/edgellm
mkdir -p $O
export TMPDIR=$B/tmp EDGELLM_PLUGIN_PATH=$B/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_start.txt
for r in 1 2 3; do
  for name in short long; do
    warm=10; [ $name = long ] && warm=5
    $E/llm_inference --engineDir $W/engines --inputFile $B/edgellm-ws/inputs/$name.json \
        --outputFile $O/${name}_r$r.out.json --warmup $warm --dumpProfile --profileOutputFile $O/${name}_r$r.profile.json \
        > $O/${name}_r$r.log 2>&1
    echo "$name r$r exit $?"
  done
done
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq > $O/gpu_freq_end.txt
echo done
