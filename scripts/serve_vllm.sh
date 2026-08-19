#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the selected checkpoint directory}"

vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048 \
  --enable-lora \
  --lora-modules "im-reply-lora=${ADAPTER_PATH}"

