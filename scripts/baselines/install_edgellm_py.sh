#!/usr/bin/env bash
# Python export environment for TensorRT Edge-LLM v0.6.0 on the Orin's CPU (MoE export runs on CPU only).
# CPU-only torch 2.10 for aarch64 from the PyTorch index, then the package and the documented MoE extras.
set -e
export PIP_CACHE_DIR=/media/vlsi/NVMESSD/pip-cache TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp
V=/media/vlsi/NVMESSD/fused-moe/venv-edgellm
[ -x $V/bin/python ] || python3 -m venv $V
$V/bin/pip install -U pip wheel setuptools
$V/bin/pip install --retries 20 --timeout 60 --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
cd /media/vlsi/NVMESSD/fused-moe/baselines/TensorRT-Edge-LLM
$V/bin/pip install --retries 20 --timeout 60 .
BUILD_CUDA_EXT=0 $V/bin/pip install -v gptqmodel==4.2.5 --no-build-isolation 2>&1 | tail -20
$V/bin/pip install optimum==2.0.0
$V/bin/python -c "import torch, transformers, modelopt, gptqmodel; print('torch', torch.__version__, 'transformers', transformers.__version__)"
$V/bin/tensorrt-edgellm-export-llm --help | head -20
