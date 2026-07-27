#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${GEMMA4_MODEL:-google/gemma-4-E4B-it}}"

if [[ $# -gt 0 ]]; then
  shift
fi

VLLM_IMAGE="${VLLM_IMAGE:-vllm-node}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm_gemma4_node}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.5}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMMA_CHAT_TEMPLATE_HOST="${SCRIPT_DIR}/tool_chat_template_gemma4.jinja"
GEMMA_CHAT_TEMPLATE_CONTAINER="/tmp/tool_chat_template_gemma4.jinja"

if [[ ! -f "${GEMMA_CHAT_TEMPLATE_HOST}" ]]; then
  echo "Missing chat template: ${GEMMA_CHAT_TEMPLATE_HOST}" >&2
  exit 1
fi

if [[ "${MODEL}" != *-it* ]]; then
  echo "Warning: ${MODEL} does not look instruction-tuned; chat output may be incoherent." >&2
fi

VLLM_SPARK_EXTRA_DOCKER_ARGS="${VLLM_SPARK_EXTRA_DOCKER_ARGS:-} -v ${GEMMA_CHAT_TEMPLATE_HOST}:${GEMMA_CHAT_TEMPLATE_CONTAINER}:ro"
export VLLM_SPARK_EXTRA_DOCKER_ARGS

exec "${SCRIPT_DIR}/launch-cluster.sh" -t "${VLLM_IMAGE}" --name "${VLLM_CONTAINER_NAME}" --solo exec \
  vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --host "${VLLM_HOST}" \
    --chat-template "${GEMMA_CHAT_TEMPLATE_CONTAINER}" \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    -tp "${VLLM_TENSOR_PARALLEL}" \
    "$@"
