#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  setup_and_run.sh  —  Complete Code Civil RAG pipeline
#  Qwen3.5-9B + Unsloth + Qdrant + RTX 3080 Ti
#
#  Usage:
#    chmod +x setup_and_run.sh
#    ./setup_and_run.sh                    # Full pipeline
#    ./setup_and_run.sh --skip-install     # Skip installation
#    ./setup_and_run.sh --skip-download    # Skip downloads
#    ./setup_and_run.sh --test             # Quick test mode
#    ./setup_and_run.sh --step dataset     # A single step
#
#  LLM backend: local llama.cpp by default, or external API if
#  LLM_API_BASE_URL / LLM_API_KEY / LLM_MODEL are set in .env.
# ═══════════════════════════════════════════════════════════════

set -e  # Stop if a command fails

# ── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="rag-legifrance"
PYTHON_VERSION="3.11"
LLAMA_CPP_DIR="$PROJECT_DIR/llama.cpp"
MODELS_DIR="$PROJECT_DIR/models"
MODEL_FILE="Qwen3.5-9B-Q4_K_M.gguf"   # GGUF for llama.cpp inference
MODEL_REPO="unsloth/Qwen3.5-9B-GGUF"

# HuggingFace model for fine-tuning (safetensors)
# 4B recommended for RTX 3080 Ti 16 GB (~2h fine-tuning, ~8 GB VRAM)
FINETUNE_MODEL="unsloth/Qwen3.5-4B"
# FINETUNE_MODEL="unsloth/Qwen3.5-9B"  # better quality but ~19 GB RAM when loading
LEGI_DUMP_URL="https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20250713-140000.tar.gz"
LEGI_DUMP_FILE="legi.tar.gz"
LLAMACPP_PORT=12080
LLAMACPP_CTX=8192          # total ctx: 8192 / parallel = 2048 tokens/slot
DATASET_WORKERS=4          # Parallel dataset generation workers
DISABLE_THINKING=true      # Disable thinking mode for dataset generation
DATASET_SIZE=10000

# ── Flags ─────────────────────────────────────────────────────
SKIP_INSTALL=false
SKIP_DOWNLOAD=false
TEST_MODE=false
SINGLE_STEP=""

for arg in "$@"; do
  case $arg in
    --skip-install)   SKIP_INSTALL=true ;;
    --skip-download)  SKIP_DOWNLOAD=true ;;
    --test)           TEST_MODE=true ;;
    --step)           shift; SINGLE_STEP="$1" ;;
    --step=*)         SINGLE_STEP="${arg#*=}" ;;
  esac
done

# ── Utility functions ─────────────────────────────────────────

banner() {
  echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${BLUE}  🏛️  $1${NC}"
  echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }
info() { echo -e "${CYAN}ℹ️  $1${NC}"; }

check_env_vars() {
  if [ ! -f "$PROJECT_DIR/.env" ]; then
    err ".env not found. Copy env.sample (cp env.sample .env) and fill in the Légifrance keys."
  fi
  source "$PROJECT_DIR/.env"
  if [ -z "$LEGIFRANCE_CLIENT_ID" ] || [[ "$LEGIFRANCE_CLIENT_ID" == *"your_"* ]]; then
    err "LEGIFRANCE_CLIENT_ID not set in .env\n→ Create keys at https://piste.gouv.fr"
  fi
  ok "Environment variables OK"
}

# Loads .env without requiring the Légifrance keys (dataset/rag steps).
source_env() {
  if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
  fi
}

# True if LLM calls must use the external API (same decision as llm_backend.py).
llm_external_configured() {
  local backend="${LLM_BACKEND:-auto}"
  local base="${LLM_API_BASE_URL:-}"
  local key="${LLM_API_KEY:-}"
  local model="${LLM_MODEL:-}"

  case "$base" in *your_*) base="" ;; esac
  case "$key" in *your_*) key="" ;; esac
  case "$model" in *your_*) model="" ;; esac

  case "$backend" in
    local) return 1 ;;
    external)
      [ -n "$base" ] && [ -n "$key" ] && [ -n "$model" ] && return 0
      return 1 ;;
    auto)
      [ -n "$base" ] && [ -n "$key" ] && [ -n "$model" ] && return 0
      return 1 ;;
    *) return 1 ;;
  esac
}

activate_conda() {
  # Initialize conda if needed
  if [ -f "$HOME/.miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/.miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV" 2>/dev/null || true
}

# ══════════════════════════════════════════════════════════════
#  STEP 0 — Create the project structure
# ══════════════════════════════════════════════════════════════

step_structure() {
  banner "STEP 0 — Project structure"
  cd "$PROJECT_DIR"

  # Create the folders
  mkdir -p ingestion dataset finetuning/output rag interface models qdrant_storage

  # Create the __init__.py files
  for pkg in ingestion dataset finetuning rag interface; do
    touch "$pkg/__init__.py"
  done

  # Create .env from env.sample if missing (fallback: env.example)
  if [ ! -f ".env" ] && [ -f "env.sample" ]; then
    cp env.sample .env
    warn ".env created from env.sample — fill in your Légifrance keys!"
  elif [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    warn ".env created from .env.example — fill in your Légifrance keys!"
  fi

  # Create docker-compose.yml if missing
  if [ ! -f "docker-compose.yml" ]; then
    cat > docker-compose.yml << 'EOF'
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: code-civil-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
EOF
  fi

  ok "Project structure created"
}

# ══════════════════════════════════════════════════════════════
#  STEP 1 — System packages (apt)
# ══════════════════════════════════════════════════════════════

step_apt() {
  banner "STEP 1 — System packages"

  sudo apt update -qq

  # Base packages
  sudo apt install -y \
    git curl wget \
    build-essential cmake \
    docker.io docker-compose \
    nvidia-cuda-toolkit \
    python3-pip

  # Add the user to the docker group
  if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    warn "Added to the docker group — a re-login may be required"
    warn "In the meantime, docker commands will be run with 'sudo'"
  fi

  ok "System packages installed"
}

# ══════════════════════════════════════════════════════════════
#  STEP 2 — Python environment (conda)
# ══════════════════════════════════════════════════════════════

step_python_env() {
  banner "STEP 2 — Python environment"

  activate_conda

  # Create the conda environment if missing
  if ! conda env list | grep -q "$CONDA_ENV"; then
    info "Creating conda environment '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
  else
    info "Environment '$CONDA_ENV' already exists"
  fi

  conda activate "$CONDA_ENV"

  # PyTorch: embedded CUDA 13.2 runtime (driver >= 580 required; system nvcc
  # may stay at 12.4, it is only used to compile llama.cpp)
  info "Installing PyTorch (CUDA 13.2)..."
  pip install torch==2.14.1+cu132 torchvision==0.29.1+cu132 \
    --index-url https://download.pytorch.org/whl/cu132 -q

  # Check CUDA
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
  ok "PyTorch + CUDA OK"

  # Unsloth (special installation)
  info "Installing Unsloth..."
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
  pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes -q
  ok "Unsloth installed"

  # Remaining dependencies
  info "Installing Python dependencies..."
  pip install -r requirements.txt -q
  ok "Python dependencies installed"
}

# ══════════════════════════════════════════════════════════════
#  STEP 3 — Compile llama.cpp
# ══════════════════════════════════════════════════════════════

step_llamacpp() {
  banner "STEP 3 — llama.cpp compilation"
  cd "$PROJECT_DIR"

  if [ -f "$LLAMA_CPP_DIR/build/bin/llama-server" ]; then
    ok "llama.cpp already compiled"
    return
  fi

  if [ ! -d "$LLAMA_CPP_DIR" ]; then
    info "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
  fi

  cd "$LLAMA_CPP_DIR"

  info "Compiling with CUDA (RTX 3080 Ti)..."
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j"$(nproc)"

  [ -f "build/bin/llama-server" ] || err "llama.cpp compilation failed"
  ok "llama.cpp compiled successfully"
  cd "$PROJECT_DIR"
}

# ══════════════════════════════════════════════════════════════
#  STEP 4 — Download the GGUF model
# ══════════════════════════════════════════════════════════════

step_download_model() {
  banner "STEP 4 — Qwen3.5-9B Q8_0 model download"
  cd "$PROJECT_DIR"
  mkdir -p "$MODELS_DIR"

  if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    ok "Model already present: $MODEL_FILE"
    return
  fi

  info "Downloading $MODEL_FILE (~9.5 GB)..."
  wget -c \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/$MODEL_FILE" \
    -O "$MODELS_DIR/$MODEL_FILE" \
    --progress=bar:force

  [ -f "$MODELS_DIR/$MODEL_FILE" ] || err "Model download failed"
  ok "Model downloaded: $MODELS_DIR/$MODEL_FILE"
}

# ══════════════════════════════════════════════════════════════
#  STEP 5 — Download the Légifrance data
# ══════════════════════════════════════════════════════════════

step_download_legi() {
  banner "STEP 5 — Légifrance data download"
  cd "$PROJECT_DIR"

  # Check whether already extracted
  if [ -d "legi" ] && [ "$(find legi -name 'LEGIARTI*.xml' | head -1)" ]; then
    ok "Légifrance data already extracted"
    return
  fi

  # Download if missing
  if [ ! -f "$LEGI_DUMP_FILE" ]; then
    info "Downloading the LEGI dump (~1.1 GB)..."
    info "URL: $LEGI_DUMP_URL"
    wget -c "$LEGI_DUMP_URL" -O "$LEGI_DUMP_FILE" --progress=bar:force || {
      warn "Primary URL failed, trying the latest available version..."
      LATEST=$(curl -s https://echanges.dila.gouv.fr/OPENDATA/LEGI/ \
        | grep -o 'Freemium_legi_global_[^"]*\.tar\.gz' \
        | sort | tail -1)
      if [ -n "$LATEST" ]; then
        wget -c "https://echanges.dila.gouv.fr/OPENDATA/LEGI/$LATEST" \
          -O "$LEGI_DUMP_FILE" --progress=bar:force
      else
        err "Unable to download the Légifrance data"
      fi
    }
  fi

  info "Extracting the archive (~5 min)..."
  tar xzf "$LEGI_DUMP_FILE"
  ok "Légifrance data extracted"
}

# ══════════════════════════════════════════════════════════════
#  STEP 6 — Start Qdrant
# ══════════════════════════════════════════════════════════════

step_qdrant() {
  banner "STEP 6 — Starting Qdrant"
  cd "$PROJECT_DIR"

  # Check whether Qdrant already responds
  if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    ok "Qdrant already running"
    return
  fi

  # Start with docker compose (v2)
  if command -v docker &> /dev/null; then
    info "Starting Qdrant via Docker..."
    sudo docker compose up -d qdrant 2>/dev/null || \
    docker compose up -d qdrant 2>/dev/null || \
    sudo docker-compose up -d qdrant

    # Wait for Qdrant to be ready
    info "Waiting for Qdrant to start..."
    for i in $(seq 1 30); do
      if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
        ok "Qdrant ready at http://localhost:6333"
        return
      fi
      sleep 2
    done
    err "Qdrant did not start in time"
  else
    err "Docker not available — install Docker with: sudo apt install docker.io"
  fi
}

# ══════════════════════════════════════════════════════════════
#  STEP 7 — Légifrance ingestion → Qdrant
# ══════════════════════════════════════════════════════════════

step_ingest() {
  banner "STEP 7 — Légifrance ingestion → Qdrant"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  # Check the cache
  if [ -f "ingestion/cache_articles.json" ]; then
    COUNT=$(python -c "import json; a=json.load(open('ingestion/cache_articles.json')); print(len(a))" 2>/dev/null || echo "0")
    if [ "$COUNT" -gt 100 ]; then
      ok "Existing article cache: $COUNT articles"
      info "Using the cache (--from-cache)"
      python run.py --step ingest --from-cache
      return
    fi
  fi

  if $TEST_MODE; then
    python run.py --step ingest --test
  else
    python run.py --step ingest
  fi
  ok "Ingestion complete"
}

# ══════════════════════════════════════════════════════════════
#  STEP 8 — Q/A dataset generation
# ══════════════════════════════════════════════════════════════

step_dataset() {
  banner "STEP 8 — Q/A dataset generation"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  source_env
  # Start llama.cpp in the background only with the local backend.
  if llm_external_configured; then
    ok "External LLM backend configured — local llama.cpp not started"
  elif ! curl -s http://localhost:$LLAMACPP_PORT/v1/models > /dev/null 2>&1; then
    info "Starting llama.cpp on port $LLAMACPP_PORT..."
    # Disable thinking mode (Qwen3.5 otherwise puts everything in reasoning_content)
    THINKING_FLAG=""
    if $DISABLE_THINKING; then
      THINKING_FLAG="--chat-template-kwargs {\"enable_thinking\":false}"
    fi

    "$LLAMA_CPP_DIR/build/bin/llama-server" \
      -m "$MODELS_DIR/$MODEL_FILE" \
      -ngl 99 \
      --ctx-size $LLAMACPP_CTX \
      --port $LLAMACPP_PORT \
      --parallel $DATASET_WORKERS \
      --cont-batching \
      --batch-size 512 \
      --no-cache-prompt \
      -t 8 \
      --host 127.0.0.1 \
      $THINKING_FLAG \
      > /tmp/llamacpp.log 2>&1 &
    LLAMACPP_PID=$!
    echo $LLAMACPP_PID > /tmp/llamacpp.pid

    # Wait for the server to be ready
    info "Waiting for llama.cpp to start..."
    for i in $(seq 1 60); do
      if curl -s http://localhost:$LLAMACPP_PORT/v1/models > /dev/null 2>&1; then
        ok "llama.cpp ready at http://localhost:$LLAMACPP_PORT"
        break
      fi
      sleep 3
    done
  else
    ok "llama.cpp already running"
  fi

  # Check that the server responds (local backend only)
  if ! llm_external_configured; then
    curl -s http://localhost:$LLAMACPP_PORT/v1/models > /dev/null 2>&1 || \
      err "llama.cpp does not respond on port $LLAMACPP_PORT"
  fi

  # Generate the dataset
  if $TEST_MODE; then
    python run.py --step dataset --max 100 --workers $DATASET_WORKERS
  else
    # Resume if interrupted
    RESUME_FLAG=""
    if [ -f "dataset/code_civil_qa.jsonl" ]; then
      CURRENT=$(wc -l < dataset/code_civil_qa.jsonl)
      if [ "$CURRENT" -gt 0 ] && [ "$CURRENT" -lt "$DATASET_SIZE" ]; then
        warn "$CURRENT examples already generated → resuming"
        RESUME_FLAG="--resume"
      fi
    fi
    python run.py --step dataset --max $DATASET_SIZE --workers $DATASET_WORKERS $RESUME_FLAG
  fi
  ok "Dataset generated"
}

# ══════════════════════════════════════════════════════════════
#  STEP 9 — Fine-tuning Qwen3.5-9B
# ══════════════════════════════════════════════════════════════

step_train() {
  banner "STEP 9 — Fine-tuning Qwen3.5-9B (Unsloth + QLoRA)"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  # Stop llama.cpp to free VRAM
  if [ -f /tmp/llamacpp.pid ]; then
    PID=$(cat /tmp/llamacpp.pid)
    if kill -0 "$PID" 2>/dev/null; then
      info "Stopping llama.cpp to free VRAM..."
      kill "$PID"
      rm /tmp/llamacpp.pid
      sleep 3
    fi
  fi

  info "Checking available VRAM..."
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
    awk '{if($1 < 12000) print "⚠️  Less than 12 GB VRAM free — OOM risk"}'

  if $TEST_MODE; then
    python finetuning/train_qwen35.py --test
  else
    python finetuning/train_qwen35.py
  fi
  ok "Fine-tuning complete"
}

# ══════════════════════════════════════════════════════════════
#  STEP 10 — LoRA merge + GGUF export
# ══════════════════════════════════════════════════════════════

step_merge() {
  banner "STEP 10 — LoRA merge + GGUF export"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  python run.py --step merge
  ok "Merge and export complete"
}

# ══════════════════════════════════════════════════════════════
#  STEP 11 — RAG test
# ══════════════════════════════════════════════════════════════

step_rag() {
  banner "STEP 11 — RAG pipeline test"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  source_env
  # Start the fine-tuned model only with the local backend.
  SERVE_SCRIPT="finetuning/output/qwen35-9b-code-civil/serve.sh"
  if llm_external_configured; then
    ok "External LLM backend — local model not started"
  elif [ -f "$SERVE_SCRIPT" ]; then
    info "Starting the fine-tuned model..."
    bash "$SERVE_SCRIPT" > /tmp/llamacpp_ft.log 2>&1 &
    echo $! > /tmp/llamacpp_ft.pid
    sleep 10
  else
    warn "serve.sh script not found — using the base model"
    # For RAG: keep thinking enabled (better legal quality)
    "$LLAMA_CPP_DIR/build/bin/llama-server" \
      -m "$MODELS_DIR/$MODEL_FILE" \
      -ngl 99 \
      --ctx-size $LLAMACPP_CTX \
      --port $LLAMACPP_PORT \
      --parallel 1 \
      --cont-batching \
      > /tmp/llamacpp.log 2>&1 &
    echo $! > /tmp/llamacpp.pid
    sleep 10
  fi

  python run.py --step rag --parallel "${LLM_PARALLEL:-4}"
  ok "RAG tests complete"
}

# ══════════════════════════════════════════════════════════════
#  STEP 12 — Streamlit interface
# ══════════════════════════════════════════════════════════════

step_app() {
  banner "STEP 12 — Streamlit interface"
  cd "$PROJECT_DIR"
  activate_conda && conda activate "$CONDA_ENV"

  info "Interface available at http://localhost:8501"
  python run.py --step app
}

# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

print_summary() {
  echo -e "\n${BOLD}${GREEN}"
  echo "╔═══════════════════════════════════════════════════════╗"
  echo "║  🎉 Full pipeline complete!                           ║"
  echo "╠═══════════════════════════════════════════════════════╣"
  echo "║                                                       ║"
  echo "║  To launch the interface:                             ║"
  echo "║    ./setup_and_run.sh --step app                      ║"
  echo "║                                                       ║"
  echo "║  To rerun the RAG alone:                              ║"
  echo "║    ./setup_and_run.sh --step rag                      ║"
  echo "║                                                       ║"
  echo "║  Qdrant dashboard:                                    ║"
  echo "║    http://localhost:6334/dashboard                    ║"
  echo "╚═══════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# ══════════════════════════════════════════════════════════════
#  MAIN — Orchestration
# ══════════════════════════════════════════════════════════════

main() {
  echo -e "${BOLD}${CYAN}"
  echo "╔═══════════════════════════════════════════════════════╗"
  echo "║   Code Civil RAG — Qwen3.5-9B × RTX 3080 Ti 16 GB     ║"
  echo "╚═══════════════════════════════════════════════════════╝"
  echo -e "${NC}"

  cd "$PROJECT_DIR"

  # Single-step mode
  if [ -n "$SINGLE_STEP" ]; then
    case "$SINGLE_STEP" in
      structure) step_structure ;;
      apt)       step_apt ;;
      env)       step_python_env ;;
      llamacpp)  step_llamacpp ;;
      model)     step_download_model ;;
      legi)      step_download_legi ;;
      qdrant)    step_qdrant ;;
      ingest)    check_env_vars && step_ingest ;;
      dataset)   step_dataset ;;
      train)     step_train ;;
      merge)     step_merge ;;
      rag)       step_rag ;;
      app)       step_app ;;
      *) err "Unknown step: $SINGLE_STEP" ;;
    esac
    exit 0
  fi

  # Full pipeline
  step_structure

  if ! $SKIP_INSTALL; then
    step_apt
    step_python_env
    step_llamacpp
  fi

  if ! $SKIP_DOWNLOAD; then
    step_download_model
    step_download_legi
  fi

  check_env_vars
  step_qdrant
  step_ingest
  step_dataset
  step_train
  step_merge
  step_rag

  print_summary
}

main "$@"
