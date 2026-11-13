# ⚖️ Civil Code RAG — Qwen3.8 × RTX 3080 Ti 16 GB

Complete RAG system (Option C: SFT + RAG) optimized for **RTX 3080 Ti 16 GB**.

## Architecture

```
Question
   │
   ▼
[Embedder multilingual-e5-large]
   │
   ▼
[Qdrant — Top-5 articles]
   │
   ▼
[LLM backend: local llama.cpp or external API]  ← thinking mode enabled
   │                       for legal reasoning
   ▼
Legal answer citing the source articles
```

## Technical stack

| Component        | Tool                             |
|------------------|----------------------------------|
| Data source      | Légifrance API (PISTE)           |
| Embeddings       | `multilingual-e5-large`          |
| Vector database  | Qdrant (Docker)                  |
| Q/A generation   | local llama.cpp or external API  |
| Fine-tuning      | Unsloth + QLoRA 4-bit            |
| RAG inference    | local llama.cpp or external API  |
| Interface        | Streamlit                        |

---

## 🚀 Installation

> `make all` is **self-sufficient**: it downloads whatever is missing (27B GGUF ~12 GB,
> Légifrance dump ~1.2 GB + extraction ~27 GB if the PISTE keys are absent,
> fine-tuning base model ~19 GB), builds the article cache, starts
> Qdrant/llama-server (Docker), ingests, generates the dataset, trains and exports.
> Force the cache source: `make articles-cache CACHE_SOURCE=dump` (offline)
> or `CACHE_SOURCE=api` (PISTE API).
>
> If an external LLM API is configured in .env (see below), the 27B GGUF
> and llama-server are not needed: `make all` / `make dataset`
> use the external API directly.

### 1. Python environment
```bash
python -m venv venv && source venv/bin/activate

# Unsloth (special installation)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Remaining dependencies
pip install -r requirements.txt
```

### 2. llama.cpp — Docker (recommended) or local build

By default the project uses a **pinned Docker image**: no compilation,
the CUDA runtime is included in the image (only the NVIDIA driver is required).

```bash
# Option A — Docker (default, LLAMA_BACKEND=docker in .env)
make llama-image-pull        # server image (digest pinned in .env)
make llama-tools-pull        # tools image (GGUF conversion + quantisation), on demand
make llama-start             # docker run --gpus all, port 12080
make llama-stop              # stops the container
```

```bash
# Option B — local build (fallback, LLAMA_BACKEND=local)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
cmake --build build --config Release -j$(nproc)
# then: LLAMA_BACKEND=local make llama-build llama-start
```

> The official images include their own CUDA runtime (12.8 for the `-cuda` tags;
> `-cuda13` variants exist). Verify the setup with
> `make cuda-info`. With the docker backend, a system `nvcc` is no longer needed
> for llama.cpp.
>
> ⚠️ **Do not go back to the versioned `...-b47xx` tags**: those builds predate
> support for the `qwen3_5` architecture of Qwen3.8 (the converter replies
> `Model Qwen3_5ForConditionalGeneration is not supported`). The `.env` pins
> images from the rolling tag validated here by digest; `make llama-tools-pull`
> prints the current digest if you want to refresh it.

> **Two CUDA versions coexist** (this is expected):
> - **PyTorch** ships its own CUDA 13.2 runtime (`cu132` wheels) — no
>   system toolkit is required for Python training/inference.
> - **nvcc 12.4** (system package) is only used to build llama.cpp (`-DGGML_CUDA=ON`).
> - Verify the setup with `make cuda-info` (driver, nvcc, PyTorch runtime).
> - Prerequisite: NVIDIA driver ≥ 580 for the cu132 wheels (here 595.99.02).

### 3. Models (downloaded automatically by `make all`)
```bash
# Nothing to do: make all calls `make model-gguf` and `make base-model`.
# Manually (if needed):
make model-gguf     # Qwen3.8-27B GGUF (~12 GB)
make base-model     # Qwen3.8-9B-Distill for fine-tuning (~19 GB)
```
```bash
# Qwen3.8-27B (dynamic Q3 ~12 GB): dataset generation + RAG inference
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-IQ3_S.gguf --local-dir ./models

# Qwen3.8-9B-Distill (~19 GB BF16): base fine-tunable in QLoRA 4-bit on 16 GB
hf download empero-ai/Qwen3.8-9B-Distill --local-dir ./models/Qwen3.8-9B-Distill

# Even larger variant (outside the VRAM budget for fine-tuning):
# ~14 GB VRAM — fits in 16 GB with a reduced context
# hf download Qwen/Qwen3.8-Flash-Next-GGUF --local-dir ./models
```

### 4. Qdrant
```bash
docker-compose up -d qdrant
# Dashboard: http://localhost:6334/dashboard
```

### 5. Configuration
```bash
cp env.sample .env
# → Fill in LEGIFRANCE_CLIENT_ID, LEGIFRANCE_CLIENT_SECRET
#   (optional: without keys, make all builds the article cache from the
#    automatically downloaded LEGI dump — slower on the first run, ~15 min)
#
# LLM backend (optional):
#   LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL to use an external API
#   LLM_PARALLEL=4: number of simultaneous calls (dataset / RAG batch)
```


## 🔌 LLM backend: external API (optional)

By default, the pipeline uses **local llama.cpp**. If an OpenAI-compatible API
is configured in .env, the pipeline uses that API instead of the local model:

- **LLM_BACKEND=auto** (default): external API if **LLM_API_BASE_URL**,
  **LLM_API_KEY** and **LLM_MODEL** are set; otherwise local llama.cpp.
- **LLM_BACKEND=external**: forces the external API; **LLM_MODEL** is then
  mandatory, otherwise the process stops with an explicit message.
- **LLM_BACKEND=local**: forces local llama.cpp, even if an API is configured.

Variables available in .env:

    LLM_BACKEND=auto
    LLM_API_BASE_URL=https://api.openai.com/v1
    LLM_API_KEY=sk-...
    LLM_MODEL=gpt-4o-mini
    LLM_TIMEOUT=600
    LLM_MAX_RETRIES=2
    LLM_PARALLEL=4
    LLM_THINKING_KWARG=auto
    # LLM_MAX_TOKENS_PARAM=max_tokens

OpenAI-compatible example:

    LLM_API_BASE_URL=https://api.openai.com/v1
    LLM_API_KEY=sk-...
    LLM_MODEL=gpt-4o-mini

The URL can be given with or without /v1: the code appends /v1 if no path
is provided, and also accepts a full endpoint .../v1/chat/completions.

With an external backend, `make dataset` and `make app` do not start
llama-server: `make llm-ready` / `make llm-ready-rag` detect the
configuration and skip the local startup. The explicit targets
`make llama-start` and `make llama-start-rag` remain available to
start the local server (useful with LLM_BACKEND=local).

**Security**: .env contains the API key and must never be committed.
Logs only show the endpoint and the model, never the key.

### Parallelism

**LLM_PARALLEL=4** in .env is the default value for simultaneous calls
(dataset and RAG batch). It can be overridden by:

    python run.py --step dataset --workers 8
    python run.py --step dataset --parallel 8
    python run.py --step dataset --sequential   # only 1 call at a time
    python run.py --step rag --parallel 8

In local mode, the number of simultaneous calls is limited by llama-server's
`--parallel`; in external mode, watch out for the provider's rate limits.

---

## 📋 Step-by-step pipeline

### Step 1 — Ingestion (~1h, or ~20 min from the cache)
```bash
# `make ingest` first guarantees ingestion/cache_articles.json:
#   - via the PISTE API if LEGIFRANCE_CLIENT_ID is set,
#   - otherwise via the LEGI dump downloaded/extracted automatically.
make ingest
```

### Step 2 — Dataset generation (~55 h for the full dataset)
```bash
# Local backend: make llama-start (or let make dataset start it)
# External backend: set LLM_API_BASE_URL / LLM_API_KEY / LLM_MODEL
make dataset          # DATASET_SIZE cap from .env (0 = all)
# interruption: make dataset-resume
```

### Step 3 — Fine-tuning (~5h on RTX 3080 Ti)
```bash
python run.py --step train

# Monitor GPU usage
watch -n 5 nvidia-smi
```

### Step 4 — Merge + GGUF export (~20min)
```bash
python run.py --step merge
# → Generates finetuning/output/qwen38-9b-code-civil/gguf/
```

### Step 5 — Start the fine-tuned server (or use the external API)
```bash
# Local backend:
bash finetuning/output/qwen38-9b-code-civil/serve.sh

# External backend: no local server, configure LLM_* in .env
```

### Step 6 — Interface
```bash
python run.py --step app
# → http://localhost:8501
```

---

## 🧪 Quick test (30 min)
```bash
python run.py --step all --test
```

---

## 📊 RTX 3080 Ti 16 GB VRAM usage

| Phase              | VRAM used     |
|--------------------|---------------|
| Fine-tuning (train)| ~14.5 GB ✅   |
| 9B GGUF inference  | ~6 GB ✅      |
| 35B GGUF inference | ~14 GB ✅     |

> With an external backend, LLM generation does not consume local VRAM:
> only embedding and Qdrant remain on the machine.

---

## 💡 Tip: thinking mode

Explicit reasoning mode is useful for complex legal questions. Locally,
it uses llama.cpp's `chat_template_kwargs`; with an external backend,
it is only sent if **LLM_THINKING_KWARG=on**.

```python
# Enabled (default) — reasons before answering
result = rag.ask("Complex question...", thinking=True)

# Disabled — direct and fast answer
result = rag.ask("Simple definition...", thinking=False)

# In the interactive terminal
python rag/rag_pipeline.py
> think   # toggle ON/OFF
```

---

## ⚠️ Legal disclaimer

This system is a tool to help understand French civil law.
It does not replace the advice of a lawyer or a qualified legal professional.
Texts provided by Légifrance (Etalab 2.0 open licence).
Qwen3.8 model under the Apache 2.0 licence.
