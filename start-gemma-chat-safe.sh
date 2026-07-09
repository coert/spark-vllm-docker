#!/usr/bin/env bash
set -euo pipefail
set -x

MODEL=${1:-bahadirakdemir/gemma-4-31B-it-text-fp8}
VLLM_IMAGE=${VLLM_IMAGE:-vllm-node}
VLLM_MEM_LIMIT_GB=${VLLM_MEM_LIMIT_GB:-110}
VLLM_MEM_SWAP_LIMIT_GB=${VLLM_MEM_SWAP_LIMIT_GB:-110}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-49152}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-49152}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1}
VLLM_SPARK_EXTRA_DOCKER_ARGS="${VLLM_SPARK_EXTRA_DOCKER_ARGS:-} --oom-kill-disable=false"
export VLLM_SPARK_EXTRA_DOCKER_ARGS

# Optional generation defaults consumed by my-launch-cluster.sh:
#   VLLM_GENERATION_CONFIG=vllm
#   VLLM_TEMPERATURE=1.0 VLLM_TOP_K=64 VLLM_TOP_P=0.95
#
# Sampling knobs:
#   temperature: randomness scale. Lower is more deterministic; higher is more varied.
#   top_k: keep only the K most likely next tokens before sampling. Lower narrows choices.
#   top_p: keep the smallest token set whose cumulative probability reaches P. Lower narrows choices.
#
# Or provide the full JSON directly with VLLM_OVERRIDE_GENERATION_CONFIG.

export VLLM_TEMPERATURE=${VLLM_TEMPERATURE:-0.6}

./my-launch-cluster.sh -t "${VLLM_IMAGE}" --solo --hf-online \
  --non-privileged \
  --mem-limit-gb "${VLLM_MEM_LIMIT_GB}" \
  --mem-swap-limit-gb "${VLLM_MEM_SWAP_LIMIT_GB}" \
  exec \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --tool-call-parser gemma4 \
    --enable-auto-tool-choice