#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  start_llama.sh — Launch llama-server (docker OR local backend)
#  Auto-detects available GPUs and configures the VRAM split
#
#  Backend chosen by LLAMA_BACKEND (in .env or the environment):
#    docker = pinned image ghcr.io/ggml-org/llama.cpp (no compilation)
#    local  = binary compiled in llama.cpp/build (legacy fallback)
#
#  Usage:
#    ./start_llama.sh                          # Auto-detection
#    ./start_llama.sh --mode dataset           # Dataset generation
#    ./start_llama.sh --mode rag               # RAG inference
#    ./start_llama.sh --model ./models/foo.gguf
#    ./start_llama.sh --force-1gpu             # Force a single GPU
#    ./start_llama.sh --port 12080
#    ./start_llama.sh --stop                   # Stop server/container
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Default values ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${MODEL_FILE:-./models/Qwen3.8-27B-UD-IQ3_S.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-./llama.cpp/build/bin/llama-server}"
PORT="${LLAMACPP_PORT:-12080}"
MODE="auto"       # auto | dataset | rag
FORCE_1GPU=false
THINKING=true     # true for RAG, false for dataset
STOP_ONLY=false

# ── Argument parsing ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL_FILE="$2";  shift 2 ;;
    --port)       PORT="$2";        shift 2 ;;
    --mode)       MODE="$2";        shift 2 ;;
    --force-1gpu) FORCE_1GPU=true;  shift   ;;
    --no-thinking) THINKING=false;  shift   ;;
    --stop)       STOP_ONLY=true;   shift   ;;
    *) echo "Unknown option: $1"; exit 1  ;;
  esac
done

# ── Configuration .env ────────────────────────────────────────
# We load .env for LLAMA_BACKEND / images / container name, WITHOUT overwriting
# a variable already explicitly provided in the caller's environment.
__BK="${LLAMA_BACKEND-}"; __IMG="${LLAMA_CPP_IMAGE-}"; __TOOLS="${LLAMA_CPP_TOOLS_IMAGE-}"
__NAME="${LLAMA_CONTAINER_NAME-}"; __MODEL="${MODEL_FILE-}"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; . "$SCRIPT_DIR/.env"; set +a
fi
[ -n "$__BK" ]    && LLAMA_BACKEND="$__BK"
[ -n "$__IMG" ]   && LLAMA_CPP_IMAGE="$__IMG"
[ -n "$__TOOLS" ] && LLAMA_CPP_TOOLS_IMAGE="$__TOOLS"
[ -n "$__NAME" ]  && LLAMA_CONTAINER_NAME="$__NAME"
[ -n "$__MODEL" ] && MODEL_FILE="$__MODEL"

LLAMA_BACKEND="${LLAMA_BACKEND:-local}"
LLAMA_CPP_IMAGE="${LLAMA_CPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:d154fba060de774527c436ffecf74a64444874bfae2ad57b4be8faa7c52fb40c}"
LLAMA_CPP_TOOLS_IMAGE="${LLAMA_CPP_TOOLS_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1a4983fc58a092dc0a63e13d07213f58ffbcf454143f0a433b25bcfe4d109f06}"
LLAMA_CONTAINER_NAME="${LLAMA_CONTAINER_NAME:-code-civil-llama}"

# ── Absolute paths ────────────────────────────────────────────
[[ "$MODEL_FILE"   != /* ]] && MODEL_FILE="$SCRIPT_DIR/${MODEL_FILE#./}"
[[ "$LLAMA_SERVER" != /* ]] && LLAMA_SERVER="$SCRIPT_DIR/${LLAMA_SERVER#./}"
LLAMA_BIN_DIR="$(dirname "$LLAMA_SERVER")"

# ── llama.cpp shared libraries (local backend only) ──
# We force the binary's folder into LD_LIBRARY_PATH (RUNPATH frozen at
# compile time, obsolete if the project is moved). No effect in docker.
export LD_LIBRARY_PATH="$LLAMA_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Colors ───────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}ℹ  $*\033[0m"; }
ok()    { echo -e "${GREEN}✅ $*\033[0m"; }
warn()  { echo -e "${YELLOW}⚠️  $*\033[0m"; }
err()   { echo -e "${RED}❌ $*\033[0m"; }

# ══════════════════════════════════════════════════════════════
#  STOP (--stop) — used by make llama-stop / clean-services
# ══════════════════════════════════════════════════════════════

stop_container() {
    if command -v docker >/dev/null 2>&1 && \
       docker container inspect "$LLAMA_CONTAINER_NAME" >/dev/null 2>&1; then
        info "Stopping container '$LLAMA_CONTAINER_NAME'..."
        docker stop "$LLAMA_CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm -f "$LLAMA_CONTAINER_NAME" >/dev/null 2>&1 || true
        ok "Container stopped"
    fi
    rm -f /tmp/llamacpp.container
}

stop_local() {
    if [ -f /tmp/llamacpp.pid ]; then
        local pid
        pid="$(cat /tmp/llamacpp.pid 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            ok "llama-server stopped (PID: $pid)"
        fi
        rm -f /tmp/llamacpp.pid
    fi
    pkill -x llama-server 2>/dev/null || true
}

stop_backend() {
    stop_container
    stop_local
    rm -f /tmp/llamacpp.backend
}

if [ "$STOP_ONLY" = true ]; then
    info "Stopping llama-server (backend: $LLAMA_BACKEND, port $PORT)..."
    stop_backend
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        warn "A server still responds on port $PORT — check manually"
    else
        ok "No active server on port $PORT"
    fi
    exit 0
fi

# ══════════════════════════════════════════════════════════════
#  GPU DETECTION
# ══════════════════════════════════════════════════════════════

detect_gpus() {
    N_GPUS=0
    NAME_LIST=(); VRAM_LIST=(); FREE_LIST=()
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    fi

    # Docker backend and nvidia-smi unavailable on the host (sandboxed session,
    # container without NVIDIA utilities...): we query the GPU INSIDE the
    # container. The nvidia runtime injects the devices there if --gpus all works.
    if [ "$N_GPUS" -eq 0 ] && [ "$LLAMA_BACKEND" = docker ] && \
       command -v docker >/dev/null 2>&1 && \
       docker image inspect "$LLAMA_CPP_IMAGE" >/dev/null 2>&1; then
        local out
        out=$(docker run --rm --gpus all --entrypoint nvidia-smi "$LLAMA_CPP_IMAGE" \
                --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
        if [ -n "$out" ]; then
            while IFS= read -r line; do
                [ -z "$line" ] && continue
                NAME_LIST+=("$(echo "$line" | cut -d, -f1 | xargs)")
                VRAM_LIST+=("$(echo "$line" | cut -d, -f2 | xargs)")
                FREE_LIST+=("$(echo "$line" | cut -d, -f3 | xargs)")
            done <<< "$out"
            N_GPUS=${#NAME_LIST[@]}
            info "GPU detected via the Docker image ($N_GPUS)"
        fi
    fi

    if [ "$N_GPUS" -eq 0 ]; then
        warn "No GPU detected — CPU mode only"
        GPU_CONFIG="cpu"
        NGL_ARG=""; SPLIT_ARG=""; MAIN_GPU_ARG=""
        return
    fi

    # Do not overwrite a detection already done via the container.
    if [ "${#NAME_LIST[@]}" -eq 0 ]; then
        mapfile -t VRAM_LIST < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        mapfile -t FREE_LIST < <(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits)
        mapfile -t NAME_LIST < <(nvidia-smi --query-gpu=name          --format=csv,noheader)
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Detected GPUs:"
    for i in "${!NAME_LIST[@]}"; do
        FREE_GB=$(echo "scale=1; ${FREE_LIST[$i]} / 1024" | bc)
        TOTAL_GB=$(echo "scale=1; ${VRAM_LIST[$i]} / 1024" | bc)
        echo "  GPU $i: ${NAME_LIST[$i]} — $FREE_GB GB free / $TOTAL_GB GB total"
    done
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

compute_ngl_split() {
    local MODEL_SIZE_MB MODEL_SIZE_GB
    MODEL_SIZE_MB=$(du -m "$MODEL_FILE" | cut -f1)
    MODEL_SIZE_GB=$(echo "scale=1; $MODEL_SIZE_MB / 1024" | bc)

    if [ "$FORCE_1GPU" = true ] || [ "$N_GPUS" -eq 1 ]; then
        GPU_CONFIG="1gpu"
        NGL_ARG="-ngl 99"
        SPLIT_ARG=""
        MAIN_GPU_ARG=""

        local FREE_GB FREE_MIB NGL_LAYERS
        FREE_GB=$(echo "scale=1; ${FREE_LIST[0]} / 1024" | bc)
        info "1 GPU mode: ${NAME_LIST[0]} ($FREE_GB GB free)"

        FREE_MIB=${FREE_LIST[0]}
        if [ "$MODEL_SIZE_MB" -gt "$FREE_MIB" ]; then
            warn "Model ($MODEL_SIZE_GB GB) > free VRAM ($FREE_GB GB)"
            warn "Partial CPU offload enabled automatically"
            local TOTAL_LAYERS=32 LAYER_SIZE_MB
            LAYER_SIZE_MB=$((MODEL_SIZE_MB / TOTAL_LAYERS))
            NGL_LAYERS=$((FREE_MIB * 85 / 100 / LAYER_SIZE_MB))
            NGL_LAYERS=$((NGL_LAYERS > TOTAL_LAYERS ? TOTAL_LAYERS : NGL_LAYERS))
            NGL_ARG="-ngl $NGL_LAYERS"
            warn "GPU layers: $NGL_LAYERS / $TOTAL_LAYERS (rest on CPU)"
        fi

    elif [ "$N_GPUS" -ge 2 ]; then
        GPU_CONFIG="2gpu"

        local VRAM_0 VRAM_1 TOTAL_VRAM TOTAL_GB RATIO_0 RATIO_1
        VRAM_0=${VRAM_LIST[0]}
        VRAM_1=${VRAM_LIST[1]}
        TOTAL_VRAM=$((VRAM_0 + VRAM_1))
        TOTAL_GB=$(echo "scale=1; $TOTAL_VRAM / 1024" | bc)

        RATIO_0=$(echo "scale=4; $VRAM_0 / $TOTAL_VRAM" | bc)
        RATIO_1=$(echo "scale=4; $VRAM_1 / $TOTAL_VRAM" | bc)

        info "2 GPU mode: ${NAME_LIST[0]} ($(echo "scale=1; $VRAM_0/1024" | bc) GB) + ${NAME_LIST[1]} ($(echo "scale=1; $VRAM_1/1024" | bc) GB)"
        info "Total VRAM: $TOTAL_GB GB — Split: $RATIO_0 / $RATIO_1"

        NGL_ARG="-ngl 99"
        SPLIT_ARG="--tensor-split $RATIO_0,$RATIO_1"
        MAIN_GPU_ARG="--main-gpu 0"

        if [ "$MODEL_SIZE_MB" -gt "$TOTAL_VRAM" ]; then
            warn "Model ($MODEL_SIZE_GB GB) > total VRAM ($TOTAL_GB GB) — partial CPU offload"
        else
            ok "Model ($MODEL_SIZE_GB GB) fits in the combined $TOTAL_GB GB VRAM"
        fi
    fi
}

# ══════════════════════════════════════════════════════════════
#  ARGUMENT BUILDING (common to both backends)
# ══════════════════════════════════════════════════════════════

build_server_args() {
    local model_path="$1" host="$2"
    local CTX_SIZE PARALLEL BATCH_SIZE EXTRA_FLAGS

    case "$MODE" in
        dataset)
            CTX_SIZE=8192
            PARALLEL=4
            BATCH_SIZE=512
            EXTRA_FLAGS="--no-cache-prompt"
            THINKING=false
            ;;
        rag)
            CTX_SIZE=8192
            PARALLEL=2
            BATCH_SIZE=512
            EXTRA_FLAGS=""
            THINKING=true
            ;;
        auto|*)
            CTX_SIZE=8192
            PARALLEL=4
            BATCH_SIZE=512
            EXTRA_FLAGS=""
            ;;
    esac

    # Overrides: LLAMACPP_PARALLEL, LLAMACPP_CTX, LLAMACPP_EXTRA_ARGS
    # (EXTRA_ARGS is split on spaces).
    # Warning: on Qwen3.8-27B IQ3_S, --spec-type draft-mtp fails at
    # context creation (tested): do not enable it for now.
    PARALLEL="${LLAMACPP_PARALLEL:-$PARALLEL}"
    CTX_SIZE="${LLAMACPP_CTX:-$CTX_SIZE}"

    SERVER_ARGS=(-m "$model_path")

    if [ -n "${NGL_ARG:-}" ]; then
        read -r -a _ngl <<< "$NGL_ARG"; SERVER_ARGS+=("${_ngl[@]}")
    fi
    if [ -n "${SPLIT_ARG:-}" ]; then
        read -r -a _split <<< "$SPLIT_ARG"; SERVER_ARGS+=("${_split[@]}")
    fi
    if [ -n "${MAIN_GPU_ARG:-}" ]; then
        read -r -a _mgpu <<< "$MAIN_GPU_ARG"; SERVER_ARGS+=("${_mgpu[@]}")
    fi

    SERVER_ARGS+=(--ctx-size "$CTX_SIZE" --port "$PORT" --parallel "$PARALLEL" \
                  --cont-batching --batch-size "$BATCH_SIZE" -t 8 --host "$host")

    if [ "$THINKING" = false ]; then
        SERVER_ARGS+=(--chat-template-kwargs '{"enable_thinking":false}')
    fi
    if [ -n "${EXTRA_FLAGS:-}" ]; then
        read -r -a _extra <<< "$EXTRA_FLAGS"; SERVER_ARGS+=("${_extra[@]}")
    fi
}

print_cmd() {
    printf '  '
    printf '%q ' "$@"
    echo ""
}

# ══════════════════════════════════════════════════════════════
#  VALIDATIONS
# ══════════════════════════════════════════════════════════════

if [ ! -f "$MODEL_FILE" ]; then
    err "Model not found: $MODEL_FILE"
    echo "   Specify with: --model ./models/my_model.gguf"
    exit 1
fi

case "$LLAMA_BACKEND" in
    docker)
        if ! command -v docker >/dev/null 2>&1; then
            err "Docker backend selected but the docker command is missing."
            exit 1
        fi
        if ! docker info >/dev/null 2>&1; then
            err "Docker inaccessible (daemon stopped or insufficient permissions)."
            echo "   → sudo systemctl start docker ; docker group: newgrp docker"
            exit 1
        fi
        if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
            err "NVIDIA runtime missing from Docker (the GPU would not be visible)."
            echo "   → sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            exit 1
        fi
        if ! docker image inspect "$LLAMA_CPP_IMAGE" >/dev/null 2>&1; then
            err "llama.cpp image missing: $LLAMA_CPP_IMAGE"
            echo "   → make llama-image-pull   (downloads the pinned image, no compilation)"
            exit 1
        fi
        ;;
    local)
        if [ ! -f "$LLAMA_SERVER" ]; then
            err "llama-server not found: $LLAMA_SERVER"
            echo "   → make llama-build            (local compilation)"
            echo "   → or LLAMA_BACKEND=docker     (prebuilt image)"
            exit 1
        fi
        ;;
    *)
        err "Invalid LLAMA_BACKEND: '$LLAMA_BACKEND' (expected: docker or local)"
        exit 1
        ;;
esac

if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    warn "A server is already running on port $PORT — stopping..."
    stop_backend
    sleep 2
fi

detect_gpus
compute_ngl_split

# ══════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend: $LLAMA_BACKEND"
if [ "$LLAMA_BACKEND" = docker ]; then
    echo "  llama-server command (container):"
    MODEL_DIR="$(cd "$(dirname "$MODEL_FILE")" && pwd)"
    MODEL_IN_CONTAINER="/models/$(basename "$MODEL_FILE")"
    build_server_args "$MODEL_IN_CONTAINER" "0.0.0.0"
    print_cmd docker run --gpus all -p "127.0.0.1:$PORT:$PORT" -v "$MODEL_DIR:/models:ro" "$LLAMA_CPP_IMAGE" "${SERVER_ARGS[@]}"
else
    echo "  llama-server command (local):"
    build_server_args "$MODEL_FILE" "127.0.0.1"
    print_cmd "$LLAMA_SERVER" "${SERVER_ARGS[@]}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$LLAMA_BACKEND" = docker ]; then
    info "Starting the container (mode: $MODE, GPU: $GPU_CONFIG)..."
    docker rm -f "$LLAMA_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker run -d \
        --name "$LLAMA_CONTAINER_NAME" \
        --no-healthcheck \
        --gpus all \
        -p "127.0.0.1:$PORT:$PORT" \
        -v "$MODEL_DIR:/models:ro" \
        --restart unless-stopped \
        "$LLAMA_CPP_IMAGE" "${SERVER_ARGS[@]}" >/dev/null
    echo docker > /tmp/llamacpp.backend
    echo "$LLAMA_CONTAINER_NAME" > /tmp/llamacpp.container
else
    info "Starting llama-server (mode: $MODE, GPU: $GPU_CONFIG)..."
    ( "$LLAMA_SERVER" "${SERVER_ARGS[@]}" ) > "${LLAMACPP_LOG:-/tmp/llamacpp.log}" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > /tmp/llamacpp.pid
    echo local > /tmp/llamacpp.backend
fi

info "Waiting for startup..."
READY=false
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 2
done

if [ "$READY" = true ]; then
    if [ "$LLAMA_BACKEND" = docker ]; then
        ok "llama-server ready at http://127.0.0.1:$PORT (container: $LLAMA_CONTAINER_NAME)"
    else
        ok "llama-server ready at http://127.0.0.1:$PORT (PID: $SERVER_PID)"
    fi
    echo ""
    if [ "${N_GPUS:-0}" -gt 0 ] 2>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null | \
            awk -F', ' '{printf "  GPU %s: %s — %s / %s\n", $1, $2, $3, $4}' || true
        echo ""
    fi
    exit 0
fi

err "Timeout — llama-server did not start. Logs:"
if [ "$LLAMA_BACKEND" = docker ]; then
    docker logs --tail 60 "$LLAMA_CONTAINER_NAME" 2>&1 || true
else
    cat "${LLAMACPP_LOG:-/tmp/llamacpp.log}" 2>/dev/null || true
fi
exit 1
