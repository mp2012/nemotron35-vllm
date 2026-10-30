#!/usr/bin/env bash
# 启动 Nemotron-3.5-Lightning vLLM 推理服务（单卡 A100 适配）
# 用法：bash scripts/serve.sh   （可改 MODEL_DIR 指向其他权重目录）
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/azureuser/models/nemotron-3.5-lightning-nvfp4}"
IMAGE="vllm/vllm-openai:v0.27.1"

docker rm -f nemotron35 2>/dev/null || true

docker run -d --name nemotron35 \
  --gpus all --ipc=host --shm-size=16g \
  -p 8000:8000 \
  -v "${MODEL_DIR}:/models/nemotron:ro" \
  "${IMAGE}" \
  /models/nemotron \
  --served-model-name nemotron-3.5-lightning \
  --trust-remote-code \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8000

echo ">>> 已启动，等待就绪："
echo "    docker logs -f nemotron35"
echo "    curl http://localhost:8000/v1/models"
