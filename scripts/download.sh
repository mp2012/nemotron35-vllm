#!/usr/bin/env bash
# 下载 Nemotron-3.5-Lightning NVFP4 权重到本地（用容器内 hf CLI + hf-mirror）
# 用法：bash scripts/download.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/azureuser/models/nemotron-3.5-lightning-nvfp4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
IMAGE="vllm/vllm-openai:v0.27.1"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

mkdir -p "$(dirname "$MODEL_DIR")"

echo ">>> 下载 ${MODEL_ID} -> ${MODEL_DIR} (via ${HF_ENDPOINT})"
docker run --rm --name nemotron-dl --network=host --entrypoint bash \
  -v "${MODEL_DIR}:/models/nemotron-out" \
  -e "HF_ENDPOINT=${HF_ENDPOINT}" \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  "${IMAGE}" -c \
  'hf download '"${MODEL_ID}"' --local-dir /models/nemotron-out'

echo ">>> 完成。大小："
du -sh "${MODEL_DIR}"
