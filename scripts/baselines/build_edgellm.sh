#!/usr/bin/env bash
# Build the TensorRT Edge-LLM v0.6.0 C++ runtime and plugins on the Orin (JetPack 6.2: CUDA 12.6, TensorRT 10.3, sm_87).
set -e
cd /media/vlsi/NVMESSD/fused-moe/baselines/TensorRT-Edge-LLM
export TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
    -DTRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu -DONNX_PARSER_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DNV_ONNX_PARSER_LIB=/usr/lib/aarch64-linux-gnu/libnvonnxparser.so \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6
cmake --build build -j 8
ls -la build/examples/llm/
