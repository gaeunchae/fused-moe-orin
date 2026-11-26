#!/usr/bin/env bash
# Download the official Qwen3-30B-A3B GPTQ INT4 checkpoint (same format as TensorRT Edge-LLM's "INT4 GPTQ" rows).
export HF_HOME=/media/vlsi/NVMESSD/hf_cache TMPDIR=/media/vlsi/NVMESSD/fused-moe/baselines/tmp
/media/vlsi/NVMESSD/fused-moe/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-30B-A3B-GPTQ-Int4', local_dir='/media/vlsi/NVMESSD/gguf/gptq/Qwen3-30B-A3B-GPTQ-Int4', max_workers=4)
print('done', p)
PY
