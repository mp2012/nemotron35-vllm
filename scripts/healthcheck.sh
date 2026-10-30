#!/usr/bin/env bash
# vLLM 服务存活探测
#
# 用法：
#   bash scripts/healthcheck.sh                          # 单次探测 http://localhost:8000
#   VLLM_URL=http://nemotron35:8000 bash scripts/healthcheck.sh   # 从其他容器网络探测
#   INTERVAL=30 bash scripts/healthcheck.sh              # 每 30s 循环探测（Ctrl-C 退出）
#   CHECK_INFER=1 bash scripts/healthcheck.sh           # 额外做一次轻量推理探测
#
# 退出码：0 = 健康；1 = 异常（可接 cron / 监控告警）
set -uo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
INTERVAL="${INTERVAL:-0}"          # 0 = 单次；>0 = 循环间隔（秒）
CHECK_MODELS="${CHECK_MODELS:-1}"  # 1 = 额外校验 /v1/models 能返回模型
CHECK_INFER="${CHECK_INFER:-0}"    # 1 = 额外做一次轻量推理探测
MODEL_NAME="${MODEL_NAME:-nemotron-3.5-lightning}"

probe() {
  local url="$1"
  local code
  # /health 在引擎就绪时返回 200
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/health" 2>/dev/null)
  if [ "$code" != "200" ]; then
    echo "[$(date '+%F %T')] UNHEALTHY: /health -> HTTP $code"
    return 1
  fi

  if [ "${CHECK_MODELS}" = "1" ]; then
    if ! curl -s -m 5 "$url/v1/models" 2>/dev/null | grep -q "$MODEL_NAME"; then
      echo "[$(date '+%F %T')] DEGRADED: /health ok but model '$MODEL_NAME' not found in /v1/models"
      return 1
    fi
  fi

  if [ "${CHECK_INFER}" = "1" ]; then
    local out
    out=$(curl -s -m 60 "$url/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":16}" \
      2>/dev/null)
    if ! echo "$out" | grep -q '"content"'; then
      echo "[$(date '+%F %T')] DEGRADED: inference probe failed (no content)"
      return 1
    fi
  fi

  echo "[$(date '+%F %T')] OK: $url healthy (model=$MODEL_NAME)"
  return 0
}

if [ "${INTERVAL:-0}" -gt 0 ]; then
  echo ">>> 循环探测 $VLLM_URL 每 ${INTERVAL}s（Ctrl-C 退出）"
  while true; do
    probe "$VLLM_URL" || true
    sleep "$INTERVAL"
  done
else
  probe "$VLLM_URL"
  exit $?
fi
