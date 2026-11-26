#!/usr/bin/env bash
# Build the TensorRT engine for Qwen3-30B-A3B-GPTQ-Int4 on the Orin (Edge-LLM v0.6.0). Max batch 1 first; v0.6.0 notes
# that batch 1 can fail to build for some models, so fall back to max batch 2 (inference still runs at batch 1).
set -u
E=/media/vlsi/NVMESSD/fused-moe/baselines/TensorRT-Edge-LLM/build/examples/llm
W=/media/vlsi/NVMESSD/fused-moe/baselines/edgellm-ws/Qwen3-30B-A3B-GPTQ-Int4
export TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp
export EDGELLM_PLUGIN_PATH=/media/vlsi/NVMESSD/fused-moe/baselines/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
date
$E/llm_build --onnxDir $W/onnx --engineDir $W/engines --maxBatchSize 1 --maxInputLen 4160 --maxKVCacheCapacity 4352 \
  || { echo "batch-1 build failed, retrying with max batch 2"; rm -rf $W/engines;
       $E/llm_build --onnxDir $W/onnx --engineDir $W/engines --maxBatchSize 2 --maxInputLen 4160 --maxKVCacheCapacity 4352; }
date
ls -la $W/engines
