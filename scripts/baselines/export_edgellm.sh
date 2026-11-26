#!/usr/bin/env bash
# TensorRT Edge-LLM v0.6.0: export Qwen3-30B-A3B-GPTQ-Int4 to ONNX on the Orin's CPU (the documented MoE path is CPU-only).
set -e
export TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp HF_HOME=/media/vlsi/NVMESSD/hf_cache
V=/media/vlsi/NVMESSD/fused-moe/venv-edgellm
W=/media/vlsi/NVMESSD/fused-moe/baselines/edgellm-ws/Qwen3-30B-A3B-GPTQ-Int4
mkdir -p $W
cd /media/vlsi/NVMESSD/fused-moe/baselines/edgellm-ws
date; $V/bin/tensorrt-edgellm-export-llm --model_dir /media/vlsi/NVMESSD/gguf/gptq/Qwen3-30B-A3B-GPTQ-Int4 \
    --output_dir $W/onnx --device cpu
ls -la $W/onnx
date
