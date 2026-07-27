#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch-cluster.sh"

HEAD_IP="10.200.0.1"
WORKER_IP="10.200.0.2"
NODES="${HEAD_IP},${WORKER_IP}"

ETH_IF="enP2p1s0f1np1"
IB_IF="roceP2p1s0f1"

IMAGE="${VLLM_IMAGE:-vllm-node}"
CONTAINER="${VLLM_CONTAINER_NAME:-vllm_gemma4_cluster}"
MODEL="${1:-${GEMMA4_MODEL:-google/gemma-4-E4B-it}}"

PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.5}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

CHAT_TEMPLATE_HOST="${SCRIPT_DIR}/tool_chat_template_gemma4.jinja"
CHAT_TEMPLATE_CONTAINER="/tmp/tool_chat_template_gemma4.jinja"
VLLM_LOG="/tmp/vllm-gemma4.log"

usage() {
    cat <<EOF
Usage:
  $0 start [model]
  $0 stop
  $0 restart [model]
  $0 status
  $0 logs
  $0 ray-status

Environment overrides:
  VLLM_IMAGE
  VLLM_CONTAINER_NAME
  GEMMA4_MODEL
  VLLM_PORT
  VLLM_GPU_MEMORY_UTIL
  VLLM_MAX_MODEL_LEN
  VLLM_MAX_NUM_BATCHED_TOKENS
  VLLM_MAX_NUM_SEQS
EOF
}

container_exists() {
    docker inspect "${CONTAINER}" >/dev/null 2>&1
}

wait_for_ray() {
    echo "Waiting for Ray to report two GPUs..."

    local attempts=60
    local resources

    for ((i = 1; i <= attempts; i++)); do
        resources="$(
            docker exec "${CONTAINER}" python3 -c '
import ray
ray.init(address="auto", logging_level="ERROR")
r = ray.cluster_resources()
print(int(r.get("GPU", 0)))
' 2>/dev/null || true
        )"

        if [[ "${resources}" == "2" ]]; then
            echo "Ray cluster is ready with 2 GPUs."
            return 0
        fi

        sleep 2
    done

    echo "Ray did not become ready with two GPUs." >&2
    docker exec "${CONTAINER}" ray status || true
    return 1
}

start_cluster() {
    if [[ ! -x "${LAUNCHER}" ]]; then
        echo "Launcher is missing or not executable: ${LAUNCHER}" >&2
        exit 1
    fi

    if [[ ! -f "${CHAT_TEMPLATE_HOST}" ]]; then
        echo "Missing chat template: ${CHAT_TEMPLATE_HOST}" >&2
        exit 1
    fi

    if container_exists; then
        echo "Container ${CONTAINER} already exists."
        echo "Use '$0 status', '$0 stop', or '$0 restart'."
        exit 1
    fi

    echo "Starting containers and Ray cluster..."

    export VLLM_SPARK_EXTRA_DOCKER_ARGS="\
${VLLM_SPARK_EXTRA_DOCKER_ARGS:-} \
-v ${CHAT_TEMPLATE_HOST}:${CHAT_TEMPLATE_CONTAINER}:ro"

    "${LAUNCHER}" \
        -n "${NODES}" \
        -t "${IMAGE}" \
        --name "${CONTAINER}" \
        --eth-if "${ETH_IF}" \
        --ib-if "${IB_IF}" \
        --ray \
        --nccl-debug INFO \
        -d \
        start

    wait_for_ray

    echo "Starting distributed vLLM server..."
    echo "Model: ${MODEL}"
    echo "Log:   ${VLLM_LOG}"

    docker exec -d "${CONTAINER}" \
        bash -lc "
            exec vllm serve '${MODEL}' \
                --port '${PORT}' \
                --host 0.0.0.0 \
                --chat-template '${CHAT_TEMPLATE_CONTAINER}' \
                --reasoning-parser gemma4 \
                --tool-call-parser gemma4 \
                --enable-auto-tool-choice \
                --gpu-memory-utilization '${GPU_MEMORY_UTIL}' \
                --max-model-len '${MAX_MODEL_LEN}' \
                --max-num-batched-tokens '${MAX_BATCHED_TOKENS}' \
                --max-num-seqs '${MAX_NUM_SEQS}' \
                --tensor-parallel-size 2 \
                --distributed-executor-backend ray \
                >'${VLLM_LOG}' 2>&1
        "

    echo
    echo "vLLM is starting in the background."
    echo "Follow startup with:"
    echo "  $0 logs"
}

stop_cluster() {
    echo "Stopping vLLM and the complete cluster..."

    if container_exists; then
        docker exec "${CONTAINER}" \
            bash -lc '
                pkill -TERM -f "[v]llm serve" 2>/dev/null || true
                pkill -TERM -f "EngineCore" 2>/dev/null || true
            ' || true

        sleep 2
    fi

    "${LAUNCHER}" \
        -n "${NODES}" \
        -t "${IMAGE}" \
        --name "${CONTAINER}" \
        --eth-if "${ETH_IF}" \
        --ib-if "${IB_IF}" \
        --ray \
        stop || true

    # Defensive cleanup in case the launcher stopped only one side.
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

    ssh -o BatchMode=yes "${WORKER_IP}" \
        "docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true" || true

    echo "Cluster stopped."
}

show_status() {
    echo "=== Head container ==="
    docker ps -a \
        --filter "name=^/${CONTAINER}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

    echo
    echo "=== Worker container ==="
    ssh "${WORKER_IP}" \
        "docker ps -a \
            --filter 'name=^/${CONTAINER}$' \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"

    if container_exists; then
        echo
        echo "=== Ray ==="
        docker exec "${CONTAINER}" ray status || true

        echo
        echo "=== vLLM processes ==="
        docker exec "${CONTAINER}" \
            bash -lc 'pgrep -af "vllm serve|EngineCore|APIServer" || true'

        echo
        echo "=== API ==="
        curl --silent --show-error --max-time 2 \
            "http://${HEAD_IP}:${PORT}/v1/models" || true
        echo
    fi
}

show_logs() {
    if ! container_exists; then
        echo "Container ${CONTAINER} is not running." >&2
        exit 1
    fi

    docker exec -it "${CONTAINER}" \
        bash -lc "touch '${VLLM_LOG}' && tail -n 100 -F '${VLLM_LOG}'"
}

show_ray_status() {
    if ! container_exists; then
        echo "Container ${CONTAINER} is not running." >&2
        exit 1
    fi

    docker exec "${CONTAINER}" ray status
}

ACTION="${1:-}"
shift || true

case "${ACTION}" in
    start)
        MODEL="${1:-${GEMMA4_MODEL:-google/gemma-4-E4B-it}}"
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    restart)
        MODEL="${1:-${GEMMA4_MODEL:-google/gemma-4-E4B-it}}"
        stop_cluster
        start_cluster
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    ray-status)
        show_ray_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
