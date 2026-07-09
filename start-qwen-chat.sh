#!/usr/bin/env bash
set -euo pipefail
set -x

MODEL=${1:-Qwen/Qwen3.6-27B}

./launch-cluster.sh --solo --hf-online exec \
  vllm serve \
    ${MODEL} \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
