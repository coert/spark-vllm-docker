#!/bin/env bash

export VLLM_SPARK_EXTRA_DOCKER_ARGS="\
-v $PWD/tool_chat_template_gemma4.jinja:/tmp/tool_chat_template_gemma4.jinja:ro"

./launch-cluster.sh \
  -n 10.200.0.1,10.200.0.2 \
  -t vllm-node \
  --name vllm_gemma4_cluster \
  --eth-if enP2p1s0f1np1 \
  --ib-if roceP2p1s0f1 \
  --ray \
  --nccl-debug INFO \
  exec \
  vllm serve google/gemma-4-E4B-it \
    --port 8000 \
    --host 0.0.0.0 \
    --chat-template /tmp/tool_chat_template_gemma4.jinja \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --tensor-parallel-size 2
