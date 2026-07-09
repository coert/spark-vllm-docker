#!/usr/bin/env bash
set -euo pipefail
set -x

#MODEL="Qwen/Qwen3-Coder-Next"
MODEL="Qwen/Qwen3.6-27B"
#MODEL="Qwen/Qwen3.6-35B-A3B"
#MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

GPU_MEMORY_UTILIZATION=0.9

./launch-cluster.sh --solo exec \
  vllm serve \
    ${MODEL} \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len auto \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --tool-call-parser qwen3_coder
