# ══════════════════════════════════════════════════════════════
#  Makefile — Code Civil RAG pipeline + fine-tuning
#  Qwen3.8 + Unsloth + Qdrant + llama.cpp
#
#  Usage:
#    make help          → show all commands
#    make all           → full pipeline (first time)
#    make train         → run only the fine-tuning
#    make status        → status of all services
# ══════════════════════════════════════════════════════════════

# ── Shell: propagated errors ────────────────────────────────
# Without pipefail, `command | tee log` returns tee's exit code (0):
# make thinks the step succeeded and chains the next ones although it
# failed (e.g. missing Légifrance credentials). We therefore force bash +
# pipefail for ALL recipes: the first error stops the pipeline.
SHELL       := /bin/bash
# -O xpg_echo: preserves the behavior of the old /bin/sh (dash) where
# `echo "\033[0;36m…"` interpreted the sequences → colors unchanged.
.SHELLFLAGS := -o pipefail -O xpg_echo -c

# ── Configuration ─────────────────────────────────────────────
CONDA_ENV      := rag-legifrance
# ⚠️  Do NOT use `conda run -n $(CONDA_ENV) python`: when the conda base
# is auto-activated (auto_activate_base=True) and
# $HOME/.unsloth/studio/unsloth_studio/bin is at the head of PATH, `conda run`
# resolves to the Unsloth Studio venv (Python 3.13) instead of the conda env
# (3.11) — pip installs then go to the wrong interpreter,
# silently. We therefore derive the prefix explicitly, without relying
# on PATH. conda is located only once, at parse time (no side effect).
CONDA_BASE     := $(shell \
	for c in "$${CONDA_EXE:-}" "$$(command -v conda 2>/dev/null)" \
	         "$${HOME:-/root}/miniforge3/bin/conda" \
	         "$${HOME:-/root}/mambaforge/bin/conda" \
	         "$${HOME:-/root}/miniconda3/bin/conda" \
	         "$${HOME:-/root}/anaconda3/bin/conda" \
	         "/opt/conda/bin/conda" "/opt/miniforge3/bin/conda"; do \
	  [ -n "$$c" ] && [ -x "$$c" ] || continue; \
	  b="$$("$$c" info --base 2>/dev/null)"; \
	  [ -n "$$b" ] && { echo "$$b"; exit 0; }; \
	done; exit 0)
CONDA_ENV_DIR  := $(CONDA_BASE)/envs/$(CONDA_ENV)
CONDA_PY       := $(CONDA_ENV_DIR)/bin/python
PYTHON         := $(CONDA_PY)
PIP            := $(CONDA_PY) -m pip

# ── AI stack versions (upgraded 2026-11-10) ──────────────────
# Caps imposed by unsloth 2026.11.3: torch < 2.15.0,
# transformers <= 5.17.0, trl <= 1.15.0, datasets < 5.0.0.
# NVIDIA driver 595.99.02 → CUDA 13.2 wheels (independent of the toolkit
# nvcc 12.4 that compiles llama.cpp). torchvision 0.29.1 requires torch>=2.14.0.
PYTORCH_INDEX       := https://download.pytorch.org/whl/cu132
TORCH_PKG           := torch==2.14.1+cu132
TORCHVISION_PKG     := torchvision==0.29.1+cu132
TRANSFORMERS_PKG    := transformers==5.17.0
PEFT_PKG            := peft==0.21.2
TRL_PKG             := trl==1.15.0
BITSANDBYTES_PKG    := bitsandbytes==0.50.2
ACCELERATE_PKG      := accelerate==1.15.0
DATASETS_PKG        := datasets==4.8.5

PROJECT_DIR    := $(shell pwd)
LLAMA_DIR      := $(PROJECT_DIR)/llama.cpp
LLAMA_SERVER   := $(LLAMA_DIR)/build/bin/llama-server
MODELS_DIR     := $(PROJECT_DIR)/models
MODEL_GGUF     := $(MODELS_DIR)/Qwen3.8-27B-UD-IQ3_S.gguf

# ── Automatic provisioning (make all downloads what is missing) ──
MODEL_GGUF_REPO   ?= unsloth/Qwen3.8-27B-GGUF
MODEL_GGUF_FILE   ?= $(notdir $(MODEL_GGUF))
BASE_MODEL_DIR    := $(or $(shell sed -n 's/^BASE_MODEL=//p' .env 2>/dev/null | head -1),./models/Qwen3.8-9B-Distill)
BASE_MODEL_REPO   ?= empero-ai/Qwen3.8-9B-Distill
CACHE_SOURCE      ?= auto
LEGI_OPENDATA_URL ?= https://echanges.dila.gouv.fr/OPENDATA/LEGI/
HF_CLI            := $(firstword $(wildcard $(CONDA_ENV_DIR)/bin/hf) hf)

# ── llama.cpp engine: docker (pinned image) or local (compilation) ──
# LLAMA_BACKEND is read by start_llama.sh from .env; the values below
# are defaults (and can be overridden on the command line).
# Defaults = validated digests (the rolling server-cuda/full-cuda tag is pinned
# here; b47xx builds do not have Qwen3.8's qwen3_5 support).
LLAMA_CPP_IMAGE        ?= $(or $(shell sed -n 's/^LLAMA_CPP_IMAGE=//p' .env 2>/dev/null | head -1),ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:d154fba060de774527c436ffecf74a64444874bfae2ad57b4be8faa7c52fb40c)
LLAMA_CPP_TOOLS_IMAGE  ?= $(or $(shell sed -n 's/^LLAMA_CPP_TOOLS_IMAGE=//p' .env 2>/dev/null | head -1),ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1a4983fc58a092dc0a63e13d07213f58ffbcf454143f0a433b25bcfe4d109f06)
LLAMA_CONTAINER_NAME   ?= $(or $(shell sed -n 's/^LLAMA_CONTAINER_NAME=//p' .env 2>/dev/null | head -1),code-civil-llama)

# ── LLM backend: external API or local llama.cpp ──────────────
# Values read from .env; "your_…" placeholders are ignored.
# Priority: environment variable / command line, then .env.
LLM_BACKEND_RAW       := $(or $(LLM_BACKEND),$(shell sed -n 's/^LLM_BACKEND=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))
LLM_API_BASE_URL_RAW  := $(or $(LLM_API_BASE_URL),$(shell sed -n 's/^LLM_API_BASE_URL=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))
LLM_API_KEY_RAW       := $(or $(LLM_API_KEY),$(shell sed -n 's/^LLM_API_KEY=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))
LLM_MODEL_RAW         := $(or $(LLM_MODEL),$(shell sed -n 's/^LLM_MODEL=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))
LLM_PARALLEL_RAW      := $(or $(LLM_PARALLEL),$(shell sed -n 's/^LLM_PARALLEL=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))
DATASET_WORKERS_RAW   := $(or $(DATASET_WORKERS),$(shell sed -n 's/^DATASET_WORKERS=//p' .env 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'))

LLM_VALID = $(if $(strip $(1)),$(if $(findstring your_,$(1)),,$(strip $(1))),)
LLM_BACKEND_CFG       := $(call LLM_VALID,$(LLM_BACKEND_RAW))
LLM_API_BASE_URL_CFG  := $(call LLM_VALID,$(LLM_API_BASE_URL_RAW))
LLM_API_KEY_CFG       := $(call LLM_VALID,$(LLM_API_KEY_RAW))
LLM_MODEL_CFG         := $(call LLM_VALID,$(LLM_MODEL_RAW))
LLM_PARALLEL_CFG      := $(or $(call LLM_VALID,$(LLM_PARALLEL_RAW)),4)
DATASET_WORKERS_CFG   := $(or $(call LLM_VALID,$(DATASET_WORKERS_RAW)),$(LLM_PARALLEL_CFG))
LLM_BACKEND_OK        := $(if $(strip $(LLM_BACKEND_CFG)),$(LLM_BACKEND_CFG),auto)
LLM_EXTERNAL          := $(if $(and $(strip $(LLM_API_BASE_URL_CFG)),$(strip $(LLM_API_KEY_CFG)),$(strip $(LLM_MODEL_CFG)),$(filter-out local,$(LLM_BACKEND_OK))),yes,)

# Opt-in only: default behavior unchanged if CUDA_ARCH is empty.
# Example: make llama-build CUDA_ARCH=89
CUDA_ARCH      ?=
LLAMA_CUDA_ARCH_ARG := $(if $(strip $(CUDA_ARCH)),-DCMAKE_CUDA_ARCHITECTURES=$(strip $(CUDA_ARCH)),)
LLAMACPP_PORT  := 12080
QDRANT_PORT    := 6333
LOG_DIR        := $(PROJECT_DIR)/logs

# ── Colors ────────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
YELLOW:= \033[1;33m
RED   := \033[0;31m
BOLD  := \033[1m
NC    := \033[0m

# ── Phony targets (no files) ──────────────────────────────────
.PHONY: help all all-skip-install install apt conda pip-torch pip-unsloth pip-deps \
        llama-build llama-build-local llama-start llama-stop llama-status \
        llama-image-pull llama-image-pull-all llama-tools-pull llama-image-build clean-llama-image \
        qdrant-start qdrant-stop qdrant-status qdrant-check \
        ingest dataset dataset-full dataset-resume dataset-test train merge app \
        status logs gpu-check cuda-info preflight \
        model-gguf legi-dump articles-cache base-model \
        clean clean-services clean-build clean-data clean-models clean-downloads \
        clean-all purge clean-check env-init env-check env-check-legifrance \
        env-check-llm llm-ready llm-ready-rag llm-status test

# ══════════════════════════════════════════════════════════════
#  HELP
# ══════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "$(BOLD)$(CYAN)══════════════════════════════════════════════════════$(NC)"
	@echo "$(BOLD)  🏛️  Code Civil RAG — Pipeline Makefile$(NC)"
	@echo "$(BOLD)$(CYAN)══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(BOLD)FULL PIPELINE$(NC)"
	@echo "  $(GREEN)make all$(NC)            Full pipeline (downloads missing model/data)"
	@echo "  $(GREEN)make all-skip-install$(NC) Pipeline without reinstalling the packages"
	@echo ""
	@echo "$(BOLD)AUTOMATIC PROVISIONING (make all downloads what is missing)$(NC)"
	@echo "  $(GREEN)make model-gguf$(NC)      Download the inference GGUF (~12 GB)"
	@echo "  $(GREEN)make legi-dump$(NC)       Download + extract the LEGI dump (~1.2 GB / ~27 GB)"
	@echo "  $(GREEN)make articles-cache$(NC)   Build ingestion/cache_articles.json (API or dump)"
	@echo "  $(GREEN)make base-model$(NC)      Download the fine-tuning base model (~19 GB)"
	@echo ""
	@echo "$(BOLD)ENVIRONMENT$(NC)"
	@echo "  $(GREEN)make env-init$(NC)        Create .env from env.sample (never overwrites)"
	@echo "  $(GREEN)make env-check$(NC)       Check that .env exists and is configured"
	@echo "  $(GREEN)make env-check-llm$(NC)   Check the LLM config (external API / local)"
	@echo "  $(GREEN)make llm-status$(NC)      Active LLM backend (external API or llama.cpp)"
	@echo ""
	@echo "$(BOLD)INSTALLATION$(NC)"
	@echo "  $(GREEN)make install$(NC)        Install everything (apt + conda + pip + llama)"
	@echo "  $(GREEN)make apt$(NC)            System packages (cmake, git, docker...)"
	@echo "  $(GREEN)make conda$(NC)          Create the Python 3.11 conda environment"
	@echo "  $(GREEN)make pip-torch$(NC)      Install PyTorch CUDA 13.2 (cu132 wheels)"
	@echo "  $(GREEN)make pip-unsloth$(NC)    Install Unsloth + zoo"
	@echo "  $(GREEN)make pip-deps$(NC)       Install requirements.txt"
	@echo "  $(GREEN)make llama-build$(NC)    llama.cpp: docker (pull image) or local (compile)"
	@echo "  $(GREEN)make llama-image-pull$(NC) Download the pinned server image (docker)"
	@echo "  $(GREEN)make llama-tools-pull$(NC) Download the tools image (convert/quantize)"
	@echo "  $(GREEN)make llama-image-build$(NC) Build the image from sources (contingency)"
	@echo "  $(GREEN)make clean-llama-image$(NC) Remove the llama.cpp images"
	@echo ""
	@echo "$(BOLD)SERVICES$(NC)"
	@echo "  $(GREEN)make qdrant-start$(NC)   Start Qdrant (Docker)"
	@echo "  $(GREEN)make qdrant-stop$(NC)    Stop Qdrant"
	@echo "  $(GREEN)make qdrant-check$(NC)   Check the code_civil collection"
	@echo "  $(GREEN)make llama-start$(NC)    Start local llama-server (port $(LLAMACPP_PORT))"
	@echo "  $(GREEN)make llama-stop$(NC)     Stop llama-server (container or process)"
	@echo "  $(GREEN)make logs-llama$(NC)     Live logs (docker logs or local file)"
	@echo "  $(YELLOW)LLM_PARALLEL=4$(NC) in .env: concurrent calls (dataset / RAG batch)"
	@echo ""
	@echo "$(BOLD)PIPELINE DATA & TRAINING$(NC)"
	@echo "  $(GREEN)make ingest$(NC)         Légifrance ingestion → Qdrant"
	@echo "  $(GREEN)make dataset$(NC)        Generate the Q/A dataset (DATASET_SIZE cap from .env)"
	@echo "  $(GREEN)make dataset-full$(NC)   COMPLETE dataset (~29,000 Q/A, ~55 h)"
	@echo "  $(GREEN)make dataset-resume$(NC) Resume an interrupted generation"
	@echo "  $(GREEN)make train$(NC)          QLoRA fine-tuning (RTX 3080 Ti)"
	@echo "  $(GREEN)make train-h200$(NC)     Fine-tuning on RunPod H200 (~60 min)"
	@echo "  $(GREEN)make merge$(NC)          LoRA merge + GGUF export"
	@echo "  $(GREEN)make app$(NC)            Launch the Streamlit interface"
	@echo ""
	@echo "$(BOLD)MONITORING$(NC)"
	@echo "  $(GREEN)make status$(NC)         Status of all services"
	@echo "  $(GREEN)make gpu-check$(NC)      Available VRAM and GPUs"
	@echo "  $(GREEN)make cuda-info$(NC)      CUDA versions (driver, nvcc, PyTorch)"
	@echo "  $(GREEN)make logs$(NC)           View the logs of the current training"
	@echo "  $(GREEN)make logs-llama$(NC)     llama.cpp server logs"
	@echo "  $(GREEN)make test$(NC)           Unit tests (LLM backend / parallelism)"
	@echo ""
	@echo "$(BOLD)CLEANUP$(NC)"
	@echo "  $(GREEN)make clean$(NC)              Logs, Python caches, temporary files"
	@echo "  $(GREEN)make clean-services$(NC)     Stop llama-server and Qdrant"
	@echo "  $(GREEN)make clean-build$(NC)        llama.cpp build, Unsloth/notebook caches"
	@echo "  $(GREEN)make clean-data$(NC)         Dataset, article cache, Qdrant, training outputs"
	@echo "  $(GREEN)make clean-models$(NC)       Downloaded models (GGUF + HF bases + caches)"
	@echo "  $(GREEN)make clean-downloads$(NC)    Extracted Légifrance dump + .tar.gz archives"
	@echo "  $(GREEN)make clean-all$(NC)          EVERYTHING: data, articles, builds, models, downloads"
	@echo "  $(GREEN)make purge$(NC)              clean-all + llama.cpp clone + backups/ (fresh clone)"
	@echo "  $(GREEN)make clean-check$(NC)        Self-test: DRY=1 must delete nothing (safety guard)"
	@echo ""
	@echo "  $(YELLOW)Options:$(NC) DRY=1 (preview, deletes nothing) | FORCE=1 (no confirmation)"
	@echo "              CLEAN_HF=1 (HuggingFace cache) | KEEP_BACKUPS=0 | REMOVE_LLAMA=1"
	@echo ""

# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

all: env-init install qdrant-start llm-ready ingest dataset train merge
	@echo ""
	@echo "$(BOLD)$(GREEN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(GREEN)║  ✅ Full pipeline complete!                           ║$(NC)"
	@echo "$(BOLD)$(GREEN)║  Launch the interface: make app                      ║$(NC)"
	@echo "$(BOLD)$(GREEN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""

all-skip-install: env-init qdrant-start llm-ready ingest dataset train merge

# ══════════════════════════════════════════════════════════════
#  INSTALLATION
# ══════════════════════════════════════════════════════════════

install: env-init apt conda pip-torch pip-unsloth pip-deps llama-build
	@echo "$(GREEN)✅ Installation complete$(NC)"

# ── Environment (.env) ───────────────────────────────────────
# The configuration template is env.sample. .env is NEVER overwritten.
env-init:
	@if [ -f "$(PROJECT_DIR)/.env" ]; then \
		echo "$(GREEN)✅ .env already present$(NC)"; \
	else \
		cp "$(PROJECT_DIR)/env.sample" "$(PROJECT_DIR)/.env"; \
		echo "$(YELLOW)⚠️  .env created from env.sample$(NC)"; \
		echo "   → fill in LEGIFRANCE_CLIENT_ID / LEGIFRANCE_CLIENT_SECRET (https://piste.gouv.fr)"; \
	fi

# Checks that .env exists (otherwise a hard stop with the procedure to follow).
env-check:
	@if [ ! -f "$(PROJECT_DIR)/.env" ]; then \
		echo "$(RED)❌ .env missing — the command stops.$(NC)"; \
		echo "   → make env-init        (creates .env from env.sample)"; \
		echo "   → then fill in LEGIFRANCE_CLIENT_ID / LEGIFRANCE_CLIENT_SECRET"; \
		exit 1; \
	fi

# PISTE credentials required for ingestion (otherwise: immediate stop).
env-check-legifrance: env-check
	@get() { grep -E "^$$1=" "$(PROJECT_DIR)/.env" 2>/dev/null | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//' | tr -d '\"'; }; \
	ok() { [ -n "$$1" ] || return 1; printf '%s' "$$1" | grep -qi 'your_' && return 1; return 0; }; \
	v_id="$$(get LEGIFRANCE_CLIENT_ID)";       ok "$$v_id"     || v_id="$${LEGIFRANCE_CLIENT_ID:-}"; \
	v_secret="$$(get LEGIFRANCE_CLIENT_SECRET)"; ok "$$v_secret" || v_secret="$${LEGIFRANCE_CLIENT_SECRET:-}"; \
	missing=""; \
	ok "$$v_id"     || missing="$$missing LEGIFRANCE_CLIENT_ID"; \
	ok "$$v_secret" || missing="$$missing LEGIFRANCE_CLIENT_SECRET"; \
	if [ -n "$$missing" ]; then \
		echo "$(RED)❌ Missing or placeholder Légifrance credentials:$$missing$(NC)"; \
		echo "   → template: env.sample   (cp env.sample .env)"; \
		echo "   → keys to create: https://piste.gouv.fr → Légifrance application"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✅ Légifrance credentials configured$(NC)"

# ── LLM backend: configuration validation ────────────────────
env-check-llm: env-check
	@backend="$(LLM_BACKEND_OK)"; \
	case "$$backend" in \
	  auto|external|local) ;; \
	  *) echo "$(RED)❌ Invalid LLM_BACKEND: '$$backend' (expected: auto, external or local)$(NC)"; exit 1 ;; \
	esac; \
	base="$(LLM_API_BASE_URL_CFG)"; key="$(LLM_API_KEY_CFG)"; model="$(LLM_MODEL_CFG)"; \
	if [ "$$backend" = "external" ] || { [ "$$backend" = "auto" ] && [ -n "$$base" ] && [ -n "$$key" ] && [ -n "$$model" ]; }; then \
	  missing=""; \
	  [ -n "$$base" ] || missing="$$missing LLM_API_BASE_URL"; \
	  [ -n "$$key" ] || missing="$$missing LLM_API_KEY"; \
	  [ -n "$$model" ] || missing="$$missing LLM_MODEL"; \
	  if [ -n "$$missing" ]; then \
	    echo "$(RED)❌ Incomplete external LLM backend:$$missing$(NC)"; \
	    echo "   → fill in these variables in .env, or set LLM_BACKEND=local"; \
	    exit 1; \
	  fi; \
	  echo "$(GREEN)✅ External LLM backend configured ($(LLM_API_BASE_URL_CFG), model $(LLM_MODEL_CFG))$(NC)"; \
	elif [ "$$backend" = "auto" ] && { [ -n "$$base" ] || [ -n "$$key" ] || [ -n "$$model" ]; }; then \
	  echo "$(YELLOW)⚠️  Incomplete external LLM configuration (base URL, API key and model are all required) — using local llama.cpp$(NC)"; \
	else \
	  echo "$(GREEN)✅ Local LLM backend (llama.cpp)$(NC)"; \
	fi; \
	parallel="$(LLM_PARALLEL_CFG)"; \
	case "$$parallel" in \
	  ''|*[!0-9]*) echo "$(RED)❌ Invalid LLM_PARALLEL: '$$parallel' (integer >= 1 expected)$(NC)"; exit 1 ;; \
	esac; \
	[ "$$parallel" -ge 1 ] || { echo "$(RED)❌ LLM_PARALLEL must be >= 1 (received: $$parallel)$(NC)"; exit 1; }; \
	workers="$(DATASET_WORKERS_CFG)"; \
	case "$$workers" in \
	  ''|*[!0-9]*) echo "$(RED)❌ Invalid DATASET_WORKERS: '$$workers' (integer >= 1 expected)$(NC)"; exit 1 ;; \
	esac; \
	[ "$$workers" -ge 1 ] || { echo "$(RED)❌ DATASET_WORKERS must be >= 1 (received: $$workers)$(NC)"; exit 1; }

# ── System packages ──────────────────────────────────────────
# Required: any error here makes `make apt` fail (fatal).
APT_REQUIRED   := cmake ninja-build git curl wget build-essential \
                  libssl-dev python3-dev pkg-config docker.io
# Convenience: best-effort, never blocking (nvtop comes from multiverse).
APT_OPTIONAL   := nvtop htop

apt:
	@echo "$(CYAN)ℹ  Installing system packages...$(NC)"
	sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq || exit 1
	@# Ubuntu 26.04: docker-compose (v1) is gone, replaced by
	@# docker-compose-v2 (Provides: docker-compose). We install whichever exists.
	@DC=""; \
	if apt-cache policy docker-compose-v2 2>/dev/null | grep -q 'Candidate: [0-9]'; then \
		DC="docker-compose-v2"; \
	elif apt-cache policy docker-compose 2>/dev/null | grep -q 'Candidate: [0-9]'; then \
		DC="docker-compose"; \
	fi; \
	if [ -n "$$DC" ]; then \
		echo "  $(CYAN)→ Docker Compose package: $$DC$(NC)"; \
	else \
		echo "  $(YELLOW)⚠️  neither docker-compose-v2 nor docker-compose available$(NC)"; \
	fi; \
	sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
		$(APT_REQUIRED) $$DC || exit 1
	@# Optional: a failure must never interrupt the installation
	@for p in $(APT_OPTIONAL); do \
		sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$$p" \
			|| echo "  $(YELLOW)⚠️  optional package '$$p' ignored$(NC)"; \
	done
	@# Add the user to the docker group
	@if ! groups "$(USER)" | grep -q docker; then \
		sudo usermod -aG docker "$(USER)"; \
		echo "$(YELLOW)⚠️  Added to the docker group — a re-login may be required$(NC)"; \
	fi
	@echo "$(GREEN)✅ System packages installed$(NC)"

# ── Preflight ─────────────────────────────────────────────────
# Side-effect-free validation: conda must be locatable.
preflight:
	@if [ -z "$(CONDA_BASE)" ]; then \
		echo "$(RED)❌ No conda installation detected.$(NC)"; \
		echo "   Install Miniforge: https://github.com/conda-forge/miniforge"; \
		echo "   then rerun: make conda"; \
		exit 1; \
	fi
	@if [ ! -x "$(CONDA_BASE)/bin/conda" ]; then \
		echo "$(RED)❌ conda not found in '$(CONDA_BASE)'$(NC)"; \
		echo "   Check the installation then rerun: make conda"; \
		exit 1; \
	fi

# ── Conda environment ─────────────────────────────────────────
# Idempotent: NEVER deletes and recreates an existing environment.
# Fails loudly if the env exists with a Python != 3.11 (pinned wheels).
conda: preflight
	@echo "$(CYAN)ℹ  Checking conda environment '$(CONDA_ENV)'...$(NC)"
	@if [ ! -d "$(CONDA_ENV_DIR)" ]; then \
		echo "$(CYAN)ℹ  Creating conda environment '$(CONDA_ENV)' (python 3.11)...$(NC)"; \
		"$(CONDA_BASE)/bin/conda" create -n $(CONDA_ENV) python=3.11 -y || exit 1; \
		echo "$(GREEN)✅ Environment '$(CONDA_ENV)' created$(NC)"; \
	else \
		echo "$(GREEN)✅ Environment '$(CONDA_ENV)' already exists: $(CONDA_ENV_DIR)$(NC)"; \
	fi
	@# Interpreter derived from the prefix: never `conda run` (see header).
	@PYV="$$("$(CONDA_PY)" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"; \
	if [ -z "$$PYV" ]; then \
		echo "$(RED)❌ Interpreter missing: $(CONDA_PY)$(NC)"; \
		echo "   The environment '$(CONDA_ENV)' is incomplete (Python not installed)."; \
		echo "   Manual fix (nothing is deleted automatically):"; \
		echo "     $(CONDA_BASE)/bin/conda env remove -n $(CONDA_ENV) && make conda"; \
		exit 1; \
	fi; \
	if [ "$$PYV" != "3.11" ]; then \
		echo "$(RED)❌ The environment '$(CONDA_ENV)' uses Python $$PYV (3.11 expected).$(NC)"; \
		echo "   The pinned wheels (torch 2.14.1+cu132, bitsandbytes, unsloth)"; \
		echo "   do not install correctly outside Python 3.11."; \
		echo "   Manual fix (nothing is deleted automatically):"; \
		echo "     $(CONDA_BASE)/bin/conda env remove -n $(CONDA_ENV) && make conda"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✅ $$("$(CONDA_PY)" -V 2>&1) — interpreter: $(CONDA_PY)$(NC)"

# ── PyTorch CUDA 13.2 (must be installed FIRST) ──────────────
# No -q: pip output remains visible (multi-minute installs).
# triton is no longer pinned: torch automatically pulls the
# matching version (triton~=3.8.0 for torch 2.14).
pip-torch: conda
	@echo "$(CYAN)ℹ  Installing PyTorch CUDA 13.2...$(NC)"
	$(PIP) install $(TORCH_PKG) $(TORCHVISION_PKG) \
		--index-url $(PYTORCH_INDEX)
	@$(PYTHON) -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
	@echo "$(GREEN)✅ PyTorch installed$(NC)"

# ── Unsloth (must be installed AFTER torch) ───────────────────
pip-unsloth: pip-torch
	@echo "$(CYAN)ℹ  Installing Unsloth...$(NC)"
	$(PIP) install "unsloth @ git+https://github.com/unslothai/unsloth.git"
	$(PIP) install unsloth-zoo --upgrade
	@echo "$(GREEN)✅ Unsloth installed$(NC)"

# ── Dependencies (after torch + unsloth) ──────────────────────
pip-deps: pip-unsloth
	@echo "$(CYAN)ℹ  Installing requirements.txt...$(NC)"
	$(PIP) install \
		$(TRANSFORMERS_PKG) \
		$(PEFT_PKG) \
		$(TRL_PKG) \
		$(BITSANDBYTES_PKG) \
		$(ACCELERATE_PKG) \
		$(DATASETS_PKG)
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"
	@echo ""
	@echo "$(CYAN)  Installed versions:$(NC)"
	@$(PYTHON) -c "\
import torch, transformers, peft, trl, bitsandbytes, unsloth; \
print(f'  torch:          {torch.__version__}'); \
print(f'  transformers:   {transformers.__version__}'); \
print(f'  peft:           {peft.__version__}'); \
print(f'  trl:            {trl.__version__}'); \
print(f'  bitsandbytes:   {bitsandbytes.__version__}'); \
print(f'  unsloth:        OK'); \
"

# ── llama.cpp compilation ─────────────────────────────────────
# -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON: llama.cpp is compiled into shared
# libraries (libmtmd/libllama/libggml) and the default RUNPATH is the
# ABSOLUTE path of the build folder: as soon as the project is moved, llama-server
# no longer finds its .so files. With $ORIGIN the RUNPATH becomes relative to the binary.
# Dispatch: docker (pinned image, no compilation) or local (compilation).
# LLAMA_BACKEND is re-read here from .env to handle the case where env-init has just
# created the file during `make install`.
llama-build:
	@backend="$${LLAMA_BACKEND:-$$(sed -n 's/^LLAMA_BACKEND=//p' $(PROJECT_DIR)/.env 2>/dev/null | head -1)}"; backend="$${backend:-local}"; \
	if [ "$$backend" = "docker" ]; then \
		$(MAKE) --no-print-directory llama-image-pull; \
	else \
		$(MAKE) --no-print-directory llama-build-local; \
	fi

# ── Docker backend: pinned image (recommended) ────────────────
llama-image-pull:
	@echo "$(CYAN)ℹ  Downloading the llama.cpp image: $(LLAMA_CPP_IMAGE)$(NC)"
	@docker info >/dev/null 2>&1 || { echo "$(RED)❌ Docker inaccessible (daemon stopped / permissions)$(NC)"; exit 1; }
	docker pull $(LLAMA_CPP_IMAGE)
	@echo "$(CYAN)  Engine version:$(NC)"; docker run --rm --gpus all --entrypoint /app/llama-server $(LLAMA_CPP_IMAGE) --version 2>&1 | head -2 || true
	@echo "$(CYAN)  Digest (to pin in .env if desired):$(NC)"; docker image inspect --format '{{index .RepoDigests 0}}' $(LLAMA_CPP_IMAGE) 2>/dev/null || true
	@echo "$(GREEN)✅ llama.cpp image ready (no compilation)$(NC)"

llama-tools-pull:
	@echo "$(CYAN)ℹ  Downloading the tools image (conversion + quantization): $(LLAMA_CPP_TOOLS_IMAGE)$(NC)"
	@docker info >/dev/null 2>&1 || { echo "$(RED)❌ Docker inaccessible$(NC)"; exit 1; }
	docker pull $(LLAMA_CPP_TOOLS_IMAGE)
	@echo "$(GREEN)✅ Tools image ready$(NC)"

llama-image-pull-all: llama-image-pull llama-tools-pull

# Contingency: build the image from local sources (compiles INSIDE
# Docker, not on the host). Use only if the published image does not fit.
llama-image-build:
	@echo "$(CYAN)ℹ  Local build of the llama.cpp image (CUDA)...$(NC)"
	@[ -f "$(LLAMA_DIR)/.devops/cuda.Dockerfile" ] || { echo "$(RED)❌ llama.cpp source missing: git clone then rerun$(NC)"; exit 1; }
	cd $(LLAMA_DIR) && docker buildx build --target server -f .devops/cuda.Dockerfile -t local/llama.cpp:server-cuda-local .
	@echo "$(GREEN)✅ Image locale construite — LLAMA_CPP_IMAGE=local/llama.cpp:server-cuda-local$(NC)"

clean-llama-image:
	@echo "$(CYAN)ℹ  Removing the llama.cpp images...$(NC)"
	@if [ "$(DRY)" = "1" ]; then \
		echo "  (preview) docker rmi $(LLAMA_CPP_IMAGE) $(LLAMA_CPP_TOOLS_IMAGE)"; \
	else \
		docker rmi $(LLAMA_CPP_IMAGE) 2>/dev/null || true; \
		docker rmi $(LLAMA_CPP_TOOLS_IMAGE) 2>/dev/null || true; \
		echo "$(GREEN)✅ Images removed$(NC)"; \
	fi

# ── Local backend: compilation (legacy fallback) ──────────────
llama-build-local:
	@echo "$(CYAN)ℹ  Compiling llama.cpp (local backend)...$(NC)"
	@if [ -f "$(LLAMA_SERVER)" ]; then \
		echo "$(GREEN)✅ llama-server already compiled$(NC)"; \
	else \
		if [ ! -d "$(LLAMA_DIR)" ]; then \
			git clone https://github.com/ggerganov/llama.cpp $(LLAMA_DIR); \
		fi; \
		if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then \
			echo "$(RED)❌ cmake and/or ninja missing — run: make apt$(NC)"; \
			exit 1; \
		fi; \
		cd $(LLAMA_DIR) && \
		cmake -B build \
			-DGGML_CUDA=ON \
			-DCMAKE_BUILD_TYPE=Release \
			-DLLAMA_CURL=OFF $(LLAMA_CUDA_ARCH_ARG) \
			-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
			-G Ninja && \
		cmake --build build --config Release -j$(shell nproc) && \
		echo "$(GREEN)✅ llama.cpp compiled$(NC)"; \
	fi

# ══════════════════════════════════════════════════════════════
#  AUTOMATIC PROVISIONING (used by make all)
# ══════════════════════════════════════════════════════════════

# Downloads the inference GGUF if missing (dependency of llama-start).
model-gguf:
	@if [ -f "$(MODEL_GGUF)" ] && [ "$$(stat -c %s "$(MODEL_GGUF)" 2>/dev/null || echo 0)" -gt 1000000000 ]; then \
		echo "$(GREEN)✅ GGUF model present: $(MODEL_GGUF)$(NC)"; \
	else \
		echo "$(CYAN)ℹ  Downloading the model $(MODEL_GGUF_REPO)/$(MODEL_GGUF_FILE) (~12 GB)...$(NC)"; \
		tok=$$(sed -n 's/^HF_TOKEN=//p' "$(PROJECT_DIR)/.env" 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'); \
		case "$$tok" in ''|your_*) unset HF_TOKEN;; *) export HF_TOKEN="$$tok";; esac; \
		mkdir -p "$(MODELS_DIR)"; \
		"$(HF_CLI)" download "$(MODEL_GGUF_REPO)" "$(MODEL_GGUF_FILE)" --local-dir "$(MODELS_DIR)" || \
			{ echo "$(RED)❌ Model download failed (network/HF_TOKEN?)$(NC)"; exit 1; }; \
		[ -f "$(MODEL_GGUF)" ] || { echo "$(RED)❌ GGUF not found after download$(NC)"; exit 1; }; \
		echo "$(GREEN)✅ Model downloaded: $(MODEL_GGUF)$(NC)"; \
	fi

# Downloads and extracts the Légifrance dump if missing (offline source).
legi-dump:
	@bash "$(PROJECT_DIR)/ingestion/fetch_legi_dump.sh" "$(LEGI_OPENDATA_URL)" "$(PROJECT_DIR)"

# Ensures ingestion/cache_articles.json (API source if PISTE keys, otherwise LEGI dump).
articles-cache:
	@if [ -s "$(PROJECT_DIR)/ingestion/cache_articles.json" ]; then \
		echo "$(GREEN)✅ Article cache present$(NC)"; exit 0; \
	fi; \
	mode="$(CACHE_SOURCE)"; \
	if [ "$$mode" = "auto" ]; then \
		id="$${LEGIFRANCE_CLIENT_ID:-$$(sed -n 's/^LEGIFRANCE_CLIENT_ID=//p' "$(PROJECT_DIR)/.env" 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"')}"; \
		case "$$id" in ''|*your_*) mode=dump;; *) mode=api;; esac; \
	fi; \
	echo "$(CYAN)ℹ  Building the article cache (source: $$mode)...$(NC)"; \
	if [ "$$mode" = "api" ]; then \
		$(PYTHON) -c "from ingestion.legifrance_client import LegifranceClient; n=len(LegifranceClient().fetch_all(use_cache=False)); print(n, 'articles')" || exit 1; \
	elif [ "$$mode" = "dump" ]; then \
		bash "$(PROJECT_DIR)/ingestion/fetch_legi_dump.sh" "$(LEGI_OPENDATA_URL)" "$(PROJECT_DIR)" || exit 1; \
		$(PYTHON) ingestion/build_cache_from_legi.py || exit 1; \
	else \
		echo "$(RED)❌ Invalid CACHE_SOURCE: '$$mode' (auto|api|dump)$(NC)"; exit 1; \
	fi; \
	[ -s "$(PROJECT_DIR)/ingestion/cache_articles.json" ] || \
		{ echo "$(RED)❌ Cache not created$(NC)"; exit 1; }; \
	echo "$(GREEN)✅ Article cache ready$(NC)"

# Downloads the fine-tuning base model if missing (local path).
base-model:
	@path="$(BASE_MODEL_DIR)"; \
	case "$$path" in \
		/*|./*) if [ -d "$$path" ] && [ -n "$$(ls -A "$$path" 2>/dev/null)" ]; then \
			echo "$(GREEN)✅ Base model present: $$path$(NC)"; \
		else \
			echo "$(CYAN)ℹ  Downloading the base model $(BASE_MODEL_REPO) -> $$path (~19 GB)...$(NC)"; \
			tok=$$(sed -n 's/^HF_TOKEN=//p' "$(PROJECT_DIR)/.env" 2>/dev/null | head -1 | sed 's/[[:space:]]*#.*//' | tr -d '"'); \
			case "$$tok" in ''|your_*) unset HF_TOKEN;; *) export HF_TOKEN="$$tok";; esac; \
			mkdir -p "$$path"; \
			"$(HF_CLI)" download "$(BASE_MODEL_REPO)" --local-dir "$$path" || \
				{ echo "$(RED)❌ Base model download failed$(NC)"; exit 1; }; \
			echo "$(GREEN)✅ Base model downloaded$(NC)"; \
		fi ;; \
		*) echo "$(GREEN)✅ HF model '$$path': download handled by transformers$(NC)" ;; \
	esac

# ══════════════════════════════════════════════════════════════
#  SERVICES
# ══════════════════════════════════════════════════════════════

# ── Qdrant ────────────────────────────────────────────────────
qdrant-start:
	@echo "$(CYAN)ℹ  Starting Qdrant...$(NC)"
	@if curl -s http://localhost:$(QDRANT_PORT)/healthz > /dev/null 2>&1; then \
		echo "$(GREEN)✅ Qdrant already running$(NC)"; \
	else \
		docker compose up -d qdrant 2>/dev/null || \
		sudo docker compose up -d qdrant 2>/dev/null || \
		sudo docker-compose up -d qdrant; \
		sleep 3; \
		curl -s http://localhost:$(QDRANT_PORT)/healthz > /dev/null 2>&1 && \
			echo "$(GREEN)✅ Qdrant started on port $(QDRANT_PORT)$(NC)" || \
			echo "$(RED)❌ Qdrant did not start$(NC)"; \
	fi

qdrant-stop:
	@echo "$(CYAN)ℹ  Stopping Qdrant...$(NC)"
	docker compose stop qdrant 2>/dev/null || \
	sudo docker compose stop qdrant 2>/dev/null || true
	@echo "$(GREEN)✅ Qdrant stopped$(NC)"

qdrant-status:
	@echo "$(CYAN)ℹ  Qdrant status:$(NC)"
	@curl -s http://localhost:$(QDRANT_PORT)/healthz > /dev/null 2>&1 && \
		echo "  🟢 Running (port $(QDRANT_PORT))" || \
		echo "  🔴 Stopped"

qdrant-check:
	@echo "$(CYAN)ℹ  Checking code_civil collection:$(NC)"
	@curl -s http://localhost:$(QDRANT_PORT)/collections/code_civil | \
		$(PYTHON) -m json.tool 2>/dev/null || \
		echo "$(RED)❌ Collection not found or Qdrant stopped$(NC)"

# ── llama-server ──────────────────────────────────────────────
llama-start: model-gguf
	@echo "$(CYAN)ℹ  Starting llama-server (port $(LLAMACPP_PORT))...$(NC)"
	@if curl -sf http://localhost:$(LLAMACPP_PORT)/v1/models > /dev/null 2>&1; then \
		echo "$(GREEN)✅ llama-server already running$(NC)"; \
	else \
		bash $(PROJECT_DIR)/start_llama.sh --mode dataset; \
	fi

llama-start-rag: model-gguf
	@echo "$(CYAN)ℹ  Starting llama-server in RAG mode...$(NC)"
	bash $(PROJECT_DIR)/start_llama.sh --mode rag

# ── LLM readiness: external API or local llama.cpp ────────────
llm-ready: env-check env-check-llm
	@if [ "$(LLM_EXTERNAL)" = "yes" ]; then \
		echo "$(GREEN)✅ External LLM backend — local llama-server not started$(NC)"; \
	else \
		$(MAKE) --no-print-directory llama-start; \
	fi

llm-ready-rag: env-check env-check-llm
	@if [ "$(LLM_EXTERNAL)" = "yes" ]; then \
		echo "$(GREEN)✅ External LLM backend — local llama-server not started$(NC)"; \
	else \
		$(MAKE) --no-print-directory llama-start-rag; \
	fi

llama-stop:
	@echo "$(CYAN)ℹ  Stopping llama-server...$(NC)"
	@bash $(PROJECT_DIR)/start_llama.sh --stop

llama-status:
	@echo "$(CYAN)ℹ  llama-server status:$(NC)"
	@backend=$$(sed -n 's/^LLAMA_BACKEND=//p' $(PROJECT_DIR)/.env 2>/dev/null | head -1); backend="$${backend:-local}"; \
	echo "  Configured backend: $$backend"; \
	if [ "$$backend" = docker ]; then \
		docker ps --filter "name=$(LLAMA_CONTAINER_NAME)" --format '  Container: {{.Names}} ({{.Status}})' 2>/dev/null || true; \
	fi
	@curl -sf http://localhost:$(LLAMACPP_PORT)/v1/models > /dev/null 2>&1 && \
		echo "  🟢 Running (port $(LLAMACPP_PORT))" || \
		echo "  🔴 Stopped"

llm-status: env-check-llm
	@echo "$(CYAN)ℹ  LLM backend:$(NC)"
	@if [ "$(LLM_EXTERNAL)" = "yes" ]; then \
		echo "  External API: $(LLM_API_BASE_URL_CFG) — model $(LLM_MODEL_CFG)"; \
	else \
		echo "  local llama.cpp: http://localhost:$(LLAMACPP_PORT)/v1"; \
	fi
	@echo "  Parallelism: $(LLM_PARALLEL_CFG) concurrent calls (LLM_PARALLEL)"

# ══════════════════════════════════════════════════════════════
#  PIPELINE DATA & TRAINING
# ══════════════════════════════════════════════════════════════

# ── Ingestion ─────────────────────────────────────────────────
ingest: env-check qdrant-start articles-cache
	@echo "$(CYAN)ℹ  Légifrance ingestion → Qdrant (from the article cache)...$(NC)"
	mkdir -p $(LOG_DIR)
	LD_LIBRARY_PATH=$(LLAMA_DIR)/build/bin:$$LD_LIBRARY_PATH \
		$(PYTHON) run.py --step ingest --from-cache 2>&1 | tee $(LOG_DIR)/ingest.log
	@echo "$(GREEN)✅ Ingestion complete$(NC)"

# ── Dataset generation ────────────────────────────────────────
dataset: env-check qdrant-start llm-ready articles-cache
	@echo "$(CYAN)ℹ  Generating the Q/A dataset...$(NC)"
	@echo "$(YELLOW)  LLM backend: $(if $(LLM_EXTERNAL),external API,local llama.cpp)$(NC)"
	@if [ -f "$(PROJECT_DIR)/dataset/code_civil_qa.jsonl" ]; then \
		echo "$(YELLOW)⚠️  dataset/code_civil_qa.jsonl already exists: this command OVERWRITES it.$(NC)"; \
		echo "   To complete/resume without losing the existing data: make dataset-resume"; \
	fi
	@# The number of examples comes from DATASET_SIZE (.env): run.py --step dataset
	@# does not pass --max, so generate_dataset applies the configured cap.
	@echo "$(YELLOW)  ⏱  Cap: DATASET_SIZE from .env (0 = all, ~29,000 Q/A) — $(DATASET_WORKERS_CFG) workers$(NC)"
	@echo "$(YELLOW)     Measured rate: ~0.15 Q/A/s → ~55 h for the complete dataset$(NC)"
	@echo "$(YELLOW)     Interruption: resume with make dataset-resume$(NC)"
	mkdir -p $(LOG_DIR)
	DATASET_WORKERS=$(DATASET_WORKERS_CFG) LLM_PARALLEL=$(LLM_PARALLEL_CFG) \
	LD_LIBRARY_PATH=$(LLAMA_DIR)/build/bin:$$LD_LIBRARY_PATH \
		$(PYTHON) run.py --step dataset 2>&1 | tee $(LOG_DIR)/dataset.log
	@echo "$(GREEN)✅ Dataset generated$(NC)"

# Complete dataset: forces --max 0 (all tasks) regardless of .env
dataset-full: env-check qdrant-start llm-ready articles-cache
	@echo "$(CYAN)ℹ  Generating the COMPLETE dataset (~29,000 Q/A, ~55 h)...$(NC)"
	@echo "$(YELLOW)  LLM backend: $(if $(LLM_EXTERNAL),external API,local llama.cpp)$(NC)"
	@if [ -f "$(PROJECT_DIR)/dataset/code_civil_qa.jsonl" ]; then \
		echo "$(YELLOW)⚠️  dataset/code_civil_qa.jsonl already exists: this command OVERWRITES it.$(NC)"; \
		echo "   To complete/resume without losing the existing data: make dataset-resume"; \
	fi
	mkdir -p $(LOG_DIR)
	DATASET_WORKERS=$(DATASET_WORKERS_CFG) LLM_PARALLEL=$(LLM_PARALLEL_CFG) \
	LD_LIBRARY_PATH=$(LLAMA_DIR)/build/bin:$$LD_LIBRARY_PATH \
		$(PYTHON) run.py --step dataset --max 0 2>&1 | tee $(LOG_DIR)/dataset.log
	@echo "$(GREEN)✅ Complete dataset generated$(NC)"

# Resume after interruption (the file is completed, not overwritten)
dataset-resume: env-check qdrant-start llm-ready articles-cache
	@echo "$(CYAN)ℹ  Resuming dataset generation...$(NC)"
	@echo "$(YELLOW)  LLM backend: $(if $(LLM_EXTERNAL),external API,local llama.cpp)$(NC)"
	mkdir -p $(LOG_DIR)
	DATASET_WORKERS=$(DATASET_WORKERS_CFG) LLM_PARALLEL=$(LLM_PARALLEL_CFG) \
	LD_LIBRARY_PATH=$(LLAMA_DIR)/build/bin:$$LD_LIBRARY_PATH \
		$(PYTHON) run.py --step dataset --resume 2>&1 | tee -a $(LOG_DIR)/dataset.log
	@echo "$(GREEN)✅ Dataset completed$(NC)"

dataset-test: env-check qdrant-start llm-ready articles-cache
	@echo "$(CYAN)ℹ  Dataset test (100 examples)...$(NC)"
	@echo "$(YELLOW)  LLM backend: $(if $(LLM_EXTERNAL),external API,local llama.cpp)$(NC)"
	mkdir -p $(LOG_DIR)
	DATASET_WORKERS=$(DATASET_WORKERS_CFG) LLM_PARALLEL=$(LLM_PARALLEL_CFG) \
	LD_LIBRARY_PATH=$(LLAMA_DIR)/build/bin:$$LD_LIBRARY_PATH \
		$(PYTHON) run.py --step dataset --max 100 --workers 2 2>&1 | tee $(LOG_DIR)/dataset-test.log

# ── Local fine-tuning (RTX 3080 Ti) ───────────────────────────
train: env-check llama-stop gpu-check base-model
	@echo "$(CYAN)ℹ  Fine-tuning Qwen3.8-9B-Distill (RTX 3080 Ti)...$(NC)"
	@echo "$(YELLOW)  ⏱  Estimated duration: ~7h$(NC)"
	@if [ ! -s "$(PROJECT_DIR)/dataset/code_civil_qa.jsonl" ]; then \
		echo "$(RED)❌ Training dataset not found: dataset/code_civil_qa.jsonl$(NC)"; \
		echo "   → make dataset        (or make dataset-resume)"; \
		exit 1; \
	fi
	mkdir -p $(LOG_DIR)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		$(PYTHON) run.py --step train 2>&1 | tee $(LOG_DIR)/train.log
	@echo "$(GREEN)✅ Fine-tuning complete$(NC)"

train-resume:
	@echo "$(CYAN)ℹ  Resuming fine-tuning from the last checkpoint...$(NC)"
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		$(PYTHON) finetuning/train_qwen38.py --resume 2>&1 | tee -a $(LOG_DIR)/train.log

# ── Fine-tuning RunPod H200 ───────────────────────────────────
train-h200:
	@echo "$(CYAN)ℹ  Preparing the RunPod H200 script...$(NC)"
	@echo ""
	@echo "$(BOLD)  Steps:$(NC)"
	@echo "  On H200 (141 GB) we train the REAL Qwen3.8-27B in LoRA BF16"
	@echo "  (impossible on the 16 GB 3080 Ti, which uses the 9B-Distill in QLoRA)."
	@echo ""
	@echo "  1. Create a RunPod pod: 1× H200 SXM, 400 GB disk, PyTorch template"
	@echo "     (driver >= 580 for cu132, otherwise automatic cu126 fallback)"
	@echo "  2. Copy the complete repository to /workspace/code-civil (train_qwen38.py,"
	@echo "     finetuning/, llama.cpp for the export) then upload the dataset:"
	@echo "     $(CYAN)scp -P PORT dataset/code_civil_qa.jsonl root@POD_IP:/workspace/code-civil/dataset/$(NC)"
	@echo "  3. In the RunPod terminal:"
	@echo "     $(CYAN)cd /workspace/code-civil && bash runpod_h200.sh$(NC)"
	@echo "  4. Retrieve the GGUF files (~1.5-2 h later):"
	@echo "     $(CYAN)scp -P PORT root@POD_IP:/workspace/code-civil/output/qwen38-27b-code-civil/gguf/*.gguf ./models/$(NC)"
	@echo ""
	@echo "$(YELLOW)  ⏱  Estimated duration: ~1.5-2 h training + ~20-30 min export | Cost: ~\$$7-10$(NC)"

# ── LoRA merge + GGUF export ──────────────────────────────────
merge: env-check
	@echo "$(CYAN)ℹ  LoRA merge + GGUF export...$(NC)"
	mkdir -p $(LOG_DIR)
	$(PYTHON) run.py --step merge 2>&1 | tee $(LOG_DIR)/merge.log
	@echo "$(GREEN)✅ Merge complete — GGUF files available in finetuning/output/$(NC)"

# ── Streamlit interface ───────────────────────────────────────
app: env-check qdrant-start llm-ready-rag
	@echo "$(CYAN)ℹ  Launching the Streamlit interface...$(NC)"
	@echo "$(YELLOW)  LLM backend: $(if $(LLM_EXTERNAL),external API,local llama.cpp)$(NC)"
	$(PYTHON) run.py --step app

# ══════════════════════════════════════════════════════════════
#  MONITORING
# ══════════════════════════════════════════════════════════════

status: gpu-check
	@echo ""
	@echo "$(BOLD)$(CYAN)── Services ──────────────────────────────────────────$(NC)"
	@$(MAKE) -s qdrant-status
	@$(MAKE) -s llama-status
	@$(MAKE) -s llm-status
	@echo ""
	@echo "$(BOLD)$(CYAN)── Dataset ───────────────────────────────────────────$(NC)"
	@if [ -f dataset/code_civil_qa.jsonl ]; then \
		LINES=$$(wc -l < dataset/code_civil_qa.jsonl); \
		echo "  📊 dataset/code_civil_qa.jsonl: $$LINES examples"; \
	else \
		echo "  ⚪ dataset/code_civil_qa.jsonl: not generated"; \
	fi
	@echo ""
	@echo "$(BOLD)$(CYAN)── Models ───────────────────────────────────────────$(NC)"
	@if [ -f "$(MODEL_GGUF)" ]; then \
		SIZE=$$(du -h "$(MODEL_GGUF)" | cut -f1); \
		echo "  📦 $(MODEL_GGUF): $$SIZE"; \
	else \
		echo "  ⚪ Base GGUF model: not downloaded"; \
	fi
	@if [ -d finetuning/output ]; then \
		for gguf in finetuning/output/**/*.gguf; do \
			[ -f "$$gguf" ] && SIZE=$$(du -h "$$gguf" | cut -f1) && \
			echo "  🏋️  $$gguf: $$SIZE"; \
		done; \
	fi
	@echo ""
	@echo "$(BOLD)$(CYAN)── Training checkpoints ──────────────────────────────$(NC)"
	@if [ -d finetuning/output ]; then \
		ls -1 finetuning/output/*/checkpoint-* 2>/dev/null | head -5 || \
		echo "  ⚪ No checkpoint"; \
	fi
	@echo ""

gpu-check:
	@echo "$(BOLD)$(CYAN)── GPU ───────────────────────────────────────────────$(NC)"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu \
		--format=csv,noheader 2>/dev/null | \
		awk -F', ' '{printf "  GPU %s: %s | VRAM %s / %s (free: %s) | %s | %s°C\n", $$1, $$2, $$3, $$5, $$4, $$6, $$7}' || \
		echo "  $(RED)❌ nvidia-smi not available$(NC)"
	@echo ""

# ── CUDA / GPU versions (both stacks) ────────────────────────
cuda-info:
	@echo "$(BOLD)$(CYAN)── CUDA / GPU ─────────────────────────────────────────$(NC)"
	@d=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); case "$$d" in [0-9]*) ;; *) d="n/a (driver inaccessible from this session)";; esac; echo "  NVIDIA driver: $$d"
	@echo "  Toolkit nvcc  : $$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | cut -d' ' -f2 || echo 'n/a')   (used to compile llama.cpp)"
	@echo "  PyTorch       : $$($(PYTHON) -c 'import torch; print(torch.__version__, "| runtime CUDA", torch.version.cuda)' 2>/dev/null || echo 'not installed')"
	@g=$$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1); echo "$$g" | grep -q ',' || g="n/a (driver inaccessible from this session)"; echo "  GPU           : $$g"
	@backend=$$(sed -n 's/^LLAMA_BACKEND=//p' $(PROJECT_DIR)/.env 2>/dev/null | head -1); backend="$${backend:-local}"; \
	echo "  llama backend: $$backend"; \
	if [ "$$backend" = docker ]; then \
		echo "  llama image   : $(LLAMA_CPP_IMAGE)"; \
		docker image inspect --format '  Digest        : {{index .RepoDigests 0}}' $(LLAMA_CPP_IMAGE) 2>/dev/null || \
			echo "  Digest        : (image not downloaded — make llama-image-pull)"; \
	else \
		echo "  Local binary : $(LLAMA_SERVER)"; \
	fi
	@echo ""
	@echo "  $(YELLOW)PyTorch embeds its own CUDA runtime (cu132 wheels); nvcc 12.4"
	@echo "  is only used to compile llama.cpp (local backend). In docker backend, the"
	@echo "  CUDA runtime is included in the image: no host compilation.$(NC)"
	@echo ""

logs:
	@echo "$(CYAN)ℹ  Current training logs:$(NC)"
	@if [ -f $(LOG_DIR)/train.log ]; then \
		tail -f $(LOG_DIR)/train.log; \
	else \
		echo "$(YELLOW)⚠️  No training log found$(NC)"; \
	fi

logs-llama:
	@echo "$(CYAN)ℹ  llama-server logs:$(NC)"
	@backend=$$(sed -n 's/^LLAMA_BACKEND=//p' $(PROJECT_DIR)/.env 2>/dev/null | head -1); backend="$${backend:-local}"; \
	if [ "$$backend" = docker ]; then \
		docker logs -f --tail 100 $(LLAMA_CONTAINER_NAME) 2>/dev/null || \
			echo "$(YELLOW)⚠️  Container '$(LLAMA_CONTAINER_NAME)' not found$(NC)"; \
	else \
		if [ -f /tmp/llamacpp.log ]; then tail -f /tmp/llamacpp.log; \
		else echo "$(YELLOW)⚠️  No llama-server log found$(NC)"; fi; \
	fi

logs-dataset:
	@if [ -f $(LOG_DIR)/dataset.log ]; then \
		tail -f $(LOG_DIR)/dataset.log; \
	else \
		echo "$(YELLOW)⚠️  No dataset log found$(NC)"; \
	fi

# ── Tests ─────────────────────────────────────────────────────
test:
	@echo "$(CYAN)ℹ  Unit tests...$(NC)"
	$(PYTHON) -m unittest discover -s tests -v

# ══════════════════════════════════════════════════════════════
#  CLEANUP (in increasing severity)
# ══════════════════════════════════════════════════════════════
# All destructive targets ask for confirmation, except FORCE=1.
# DRY=1 shows what would be deleted without touching anything.
#
# Options:
#   DRY=1            preview only (deletes nothing)
#   FORCE=1          delete without asking (scripts / CI)
#   KEEP_BACKUPS=0   also deletes backups/
#   REMOVE_LLAMA=1   also deletes the complete llama.cpp clone (source + build)
#                    (useful only with the local backend)
#   REMOVE_LLAMA_IMAGE=1  also deletes the llama.cpp Docker images (docker backend)
#   CLEAN_HF=1       also deletes the shared HuggingFace cache (~/.cache/huggingface)
#
# NEVER deleted by default: source code, .env, env.sample, README,
# CGU_Legifrance_*.pdf, backups/ and the llama.cpp clone.

# Interactive confirmation: a single shell line (usable in a recipe).
CLEAN_CONFIRM = if [ "$(DRY)" != "1" ] && [ "$(FORCE)" != "1" ]; then printf "$(YELLOW)Confirm deletion? [yes/N] $(NC)"; read -r _a; case "$$_a" in yes|YES|y|Y) ;; *) echo "Cancelled."; exit 1 ;; esac; fi

# What each target deletes ($$(wildcard) = the item only exists at this moment).
CLEAN_BUILD_PATHS := llama.cpp/build unsloth_compiled_cache .ipynb_checkpoints
CLEAN_DATA_PATHS  := $(wildcard dataset/*.jsonl) ingestion/cache_articles.json finetuning/output qdrant_storage
CLEAN_MODEL_PATHS := $(wildcard models/*.gguf) $(wildcard models/Qwen3.8-*) models/.cache models/.hf-cache
CLEAN_DL_PATHS    := legi $(wildcard legi.tar.gz) $(wildcard Freemium_legi_global_*.tar.gz)
CLEAN_ALL_PATHS   := $(CLEAN_BUILD_PATHS) $(CLEAN_DATA_PATHS) $(CLEAN_MODEL_PATHS) $(CLEAN_DL_PATHS)

# ── Light cleanup (no confirmation) ──────────────────────────
clean:
	@echo "$(CYAN)ℹ  Cleaning temporary files...$(NC)"
	@if [ "$(DRY)" = "1" ]; then \
		echo "  (preview) logs/*.log, /tmp/llamacpp.*, __pycache__, *.pyc"; \
	else \
		rm -rf $(LOG_DIR)/*.log; \
		rm -f /tmp/llamacpp.pid /tmp/llamacpp.log /tmp/llamacpp_ft.pid /tmp/llamacpp.backend /tmp/llamacpp.container; \
		find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true; \
		find . -name "*.pyc" -delete 2>/dev/null || true; \
	fi
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

# ── Stopping services ────────────────────────────────────────
clean-services:
	@echo "$(CYAN)ℹ  Stopping llama-server and Qdrant...$(NC)"
	@if [ "$(DRY)" = "1" ]; then \
		echo "  (preview) stop llama-server (docker/container or process) + docker compose down"; \
	else \
		bash $(PROJECT_DIR)/start_llama.sh --stop >/dev/null 2>&1 || true; \
		docker compose down 2>/dev/null || sudo docker compose down 2>/dev/null || \
			sudo docker-compose down 2>/dev/null || true; \
		echo "$(GREEN)✅ Services stopped$(NC)"; \
	fi

# ── Builds and caches ────────────────────────────────────────
clean-build:
	@echo "$(CYAN)ℹ  llama.cpp builds and compilation caches...$(NC)"
	@$(CLEAN_CONFIRM)
	@if [ "$(DRY)" = "1" ]; then \
		for p in $(CLEAN_BUILD_PATHS); do [ -e "$$p" ] && echo "  (preview) $$p"; done; \
		echo "$(YELLOW)  (preview — nothing was deleted)$(NC)"; \
	else \
		for p in $(CLEAN_BUILD_PATHS); do \
			[ -e "$$p" ] || continue; \
			echo "  → deleting $$p"; rm -rf -- "$$p" 2>/dev/null || sudo rm -rf -- "$$p"; \
		done; \
		echo "$(GREEN)✅ Builds deleted (make llama-build to recompile)$(NC)"; \
	fi

# ── Generated data + articles + training outputs ─────────────
clean-data:
	@echo "$(CYAN)ℹ  Generated data (dataset, articles, Qdrant, training)...$(NC)"
	@$(CLEAN_CONFIRM)
	@if [ "$(DRY)" = "1" ]; then \
		for p in $(CLEAN_DATA_PATHS); do [ -e "$$p" ] && echo "  (preview) $$p"; done; \
		echo "$(YELLOW)  (preview — nothing was deleted)$(NC)"; \
	else \
		echo "  → stopping Qdrant (docker compose down -v)"; \
		docker compose down -v 2>/dev/null || sudo docker compose down -v 2>/dev/null || \
			sudo docker-compose down -v 2>/dev/null || true; \
		for p in $(CLEAN_DATA_PATHS); do \
			[ -e "$$p" ] || continue; \
			echo "  → deleting $$p"; rm -rf -- "$$p" 2>/dev/null || sudo rm -rf -- "$$p"; \
		done; \
		echo "$(GREEN)✅ Data deleted$(NC)"; \
	fi

# ── Downloaded models ────────────────────────────────────────
clean-models:
	@echo "$(CYAN)ℹ  Downloaded models (GGUF, HF bases, HF caches)...$(NC)"
	@$(CLEAN_CONFIRM)
	@if [ "$(DRY)" = "1" ]; then \
		for p in $(CLEAN_MODEL_PATHS); do [ -e "$$p" ] && echo "  (preview) $$p"; done; \
		echo "$(YELLOW)  (preview — nothing was deleted)$(NC)"; \
	else \
		for p in $(CLEAN_MODEL_PATHS); do \
			[ -e "$$p" ] || continue; \
			echo "  → deleting $$p"; rm -rf -- "$$p" 2>/dev/null || sudo rm -rf -- "$$p"; \
		done; \
		echo "$(GREEN)✅ Models deleted$(NC)"; \
	fi

# ── Légifrance downloads ─────────────────────────────────────
clean-downloads:
	@echo "$(CYAN)ℹ  Légifrance dump (extracted) and downloaded archives...$(NC)"
	@$(CLEAN_CONFIRM)
	@if [ "$(DRY)" = "1" ]; then \
		for p in $(CLEAN_DL_PATHS); do [ -e "$$p" ] && echo "  (preview) $$p"; done; \
		echo "$(YELLOW)  (preview — nothing was deleted)$(NC)"; \
	else \
		for p in $(CLEAN_DL_PATHS); do \
			[ -e "$$p" ] || continue; \
			echo "  → deleting $$p"; rm -rf -- "$$p" 2>/dev/null || sudo rm -rf -- "$$p"; \
		done; \
		echo "$(GREEN)✅ Downloads deleted$(NC)"; \
	fi

# ── COMPLETE cleanup ─────────────────────────────────────────
clean-all:
	@echo ""
	@echo "$(BOLD)$(YELLOW)⚠️  COMPLETE CLEANUP$(NC)"
	@echo "  Deletes: generated dataset, cached articles, training outputs, Qdrant,"
	@echo "             llama.cpp build, GGUF models, extracted Légifrance dump and archives."
	@echo "  Keeps: source code, .env, env.sample, README, CGU, backups/, llama.cpp."
	@echo "             (except KEEP_BACKUPS=0 / REMOVE_LLAMA=1)"
	@echo ""
	@echo "$(CYAN)  Reclaimable space:$(NC)"
	@for p in $(CLEAN_ALL_PATHS); do \
		[ -e "$$p" ] || continue; \
		s=$$(timeout 10 du -sh --apparent-size "$$p" 2>/dev/null | cut -f1); \
		[ -n "$$s" ] || s="?"; \
		printf '    %8s  %s\n' "$$s" "$$p"; \
	done
	@if [ "$(CLEAN_HF)" = "1" ] && [ -d "$$HOME/.cache/huggingface" ]; then \
		echo "            + ~/.cache/huggingface (CLEAN_HF=1)"; \
	fi
	@echo ""
	@$(CLEAN_CONFIRM)
	@# ⚠️  The entire destructive part is INSIDE a conditional block: with DRY=1
	@# no rm can be executed (bug fixed on 2026-11-10: an `exit 0`
	@# only stopped the current line's shell, not the rest of the recipe).
	@if [ "$(DRY)" = "1" ]; then \
		echo "$(YELLOW)  (preview — nothing was deleted)$(NC)"; \
	else \
		$(MAKE) --no-print-directory clean-services || true; \
		$(MAKE) --no-print-directory FORCE=1 clean || true; \
		for p in $(CLEAN_ALL_PATHS); do \
			[ -e "$$p" ] || continue; \
			echo "  → deleting $$p"; rm -rf -- "$$p" 2>/dev/null || sudo rm -rf -- "$$p"; \
		done; \
		if [ "$(CLEAN_HF)" = "1" ]; then \
			echo "  → deleting ~/.cache/huggingface (shared cache)"; rm -rf -- "$$HOME/.cache/huggingface"; \
		fi; \
		if [ "$(REMOVE_LLAMA)" = "1" ]; then \
			echo "  → deleting the llama.cpp clone"; rm -rf -- llama.cpp; \
		fi; \
		if [ "$(KEEP_BACKUPS)" = "0" ]; then \
			echo "  → deleting backups/"; rm -rf -- backups; \
		fi; \
		if [ "$(REMOVE_LLAMA_IMAGE)" = "1" ]; then \
			echo "  → deleting the llama.cpp images (docker rmi)"; \
			docker rmi $(LLAMA_CPP_IMAGE) $(LLAMA_CPP_TOOLS_IMAGE) 2>/dev/null || true; \
		fi; \
		echo ""; \
		echo "$(GREEN)✅ Complete cleanup finished.$(NC)"; \
		echo "   Start from scratch: make install then make all"; \
	fi

# ── Cleanup self-test (safety guard) ─────────────────────────
# Verifies that DRY=1 deletes NOTHING and that FORCE=1 really deletes,
# on a disposable folder: protects against a regression of the kind
# "an exit 0 in a recipe does not stop the rest of the recipe".
clean-check:
	@echo "$(CYAN)ℹ  Cleanup self-test...$(NC)"
	@rm -rf .clean-selftest; mkdir -p .clean-selftest; echo ok > .clean-selftest/keep
	@$(MAKE) --no-print-directory clean-build DRY=1 CLEAN_BUILD_PATHS=.clean-selftest >/dev/null 2>&1; \
	if [ -f .clean-selftest/keep ]; then \
		echo "$(GREEN)  ✅ DRY=1: no file deleted$(NC)"; \
	else echo "$(RED)  ❌ DRY=1 deleted files!$(NC)"; rm -rf .clean-selftest; exit 1; fi
	@$(MAKE) --no-print-directory clean-build FORCE=1 CLEAN_BUILD_PATHS=.clean-selftest >/dev/null 2>&1; \
	if [ -e .clean-selftest ]; then \
		echo "$(RED)  ❌ FORCE=1 deleted nothing!$(NC)"; rm -rf .clean-selftest; exit 1; \
	else echo "$(GREEN)  ✅ FORCE=1: deletion effective$(NC)"; fi
	@echo "$(GREEN)✅ Cleanup self-test OK$(NC)"

# ── Full reset (equivalent to a fresh clone) ─────────────────
purge:
	@$(MAKE) --no-print-directory clean-all REMOVE_LLAMA=1 KEEP_BACKUPS=0
