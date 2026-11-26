#!/usr/bin/env bash
# vLLM for JetPack 6 (CUDA 12.6, Python 3.10) in a venv on the NVMe drive; caches and temp files stay off the full root disk.
# The vLLM wheel is fetched with resumable curl first, because pip stalled on the Jetson index.
set -e
export PIP_CACHE_DIR=/media/vlsi/NVMESSD/pip-cache TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp
B=/media/vlsi/NVMESSD/fused-moe/baselines
V=/media/vlsi/NVMESSD/fused-moe/venv-vllm
IDX=https://pypi.jetson-ai-lab.io/jp6/cu126
W=$B/wheels/vllm-0.20.0+cu126-cp310-cp310-linux_aarch64.whl
[ -x $V/bin/python ] || python3 -m venv $V
for i in $(seq 1 30); do
  curl -sS -L -C - --retry 5 -o $W "$IDX/+f/1fb/8a57facd277fc/vllm-0.20.0+cu126-cp310-cp310-linux_aarch64.whl" && break
  sleep 5
done
echo "1fb8a57facd277fc4b25bd3863d0bd496271450eeb9d2b994879441d8a51f4f8  $W" | sha256sum -c
$V/bin/pip install --retries 20 --timeout 60 --index-url $IDX "$W"
$V/bin/python -c "import torch, vllm; print('torch', torch.__version__, torch.cuda.is_available(), 'vllm', vllm.__version__)"
