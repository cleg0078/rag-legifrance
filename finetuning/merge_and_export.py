"""
merge_and_export.py
───────────────────
Merges the LoRA adapters into Qwen3.5-9B,
then exports to GGUF for llama.cpp (inference on RTX 3080 Ti).

llama.cpp backend: LLAMA_BACKEND=docker|local (read from .env);
serve.sh is generated for local binaries or for a Docker container.

Usage:
  python finetuning/merge_and_export.py
  python finetuning/merge_and_export.py --push-hub myuser/qwen35-code-civil
"""

import os, sys, argparse, subprocess
from pathlib import Path
from dotenv import load_dotenv

# Allow `import finetuning.qwen38_convert` even when this script is launched
# directly (sys.path[0] is then the finetuning/ folder).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

BASE_MODEL    = os.getenv("BASE_MODEL", "Qwen/Qwen3.5-9B")
OUTPUT_DIR    = Path(os.getenv("FINETUNED_MODEL_PATH", "./finetuning/output/qwen35-9b-code-civil"))
LORA_PATH     = OUTPUT_DIR / "lora_adapters"
MERGED_PATH   = OUTPUT_DIR / "merged"
GGUF_PATH     = OUTPUT_DIR / "gguf"
MODEL_NAME    = "qwen38-code-civil"

# llama.cpp backend of the generated server (see .env / Makefile).
DEFAULT_SERVER_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:d154fba060de774527c436ffecf74a64444874bfae2ad57b4be8faa7c52fb40c"


def _llama_backend() -> str:
    """llama.cpp backend: 'docker' if LLAMA_BACKEND=docker, otherwise 'local'."""
    return (os.getenv("LLAMA_BACKEND") or "local").strip().lower()


def _server_image() -> str:
    """Docker image of the llama.cpp server (docker backend)."""
    return os.getenv("LLAMA_CPP_IMAGE", DEFAULT_SERVER_IMAGE).strip() or DEFAULT_SERVER_IMAGE


def _container_name() -> str:
    """Container name of the llama.cpp server (docker backend)."""
    return os.getenv("LLAMA_CONTAINER_NAME", "code-civil-llama").strip() or "code-civil-llama"


def _model_type() -> str:
    """Reads model_type from config.json (base model or adapter)."""
    import json
    for p in (Path(BASE_MODEL), LORA_PATH):
        cfg = Path(p) / "config.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "") or ""
            except Exception:
                return ""
    return ""


def _is_qwen38() -> bool:
    mt = _model_type()
    return mt.startswith("qwen3_5") or mt.startswith("qwen4")


def merge_lora():
    """Merges the LoRA adapters into the base model."""
    if _is_qwen38():
        # Qwen3.8 is multimodal (weights under model.language_model.*):
        # Unsloth FastLanguageModel is not suitable, we go through transformers+peft.
        from finetuning.qwen38_convert import merge as merge_qwen38
        merge_qwen38(BASE_MODEL, LORA_PATH, MERGED_PATH)
        return
    print("🔀 Merging the LoRA adapters...")

    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(LORA_PATH),
        max_seq_length = 2048,
        load_in_4bit   = True,
        dtype          = torch.bfloat16,
    )

    print(f"   Saving the merged model → {MERGED_PATH}")
    MERGED_PATH.mkdir(parents=True, exist_ok=True)

    # Unsloth merges and saves in a single operation
    model.save_pretrained_merged(
        str(MERGED_PATH),
        tokenizer,
        save_method = "merged_16bit",   # Save in float16
    )
    print("✅ Merge complete")


def export_gguf():
    """Exports the merged model to GGUF (Q4_K_M + Q8_0) for llama.cpp."""
    if _is_qwen38():
        # llama.cpp supports the qwen35 architecture: conversion + quantization
        # through its own tools (more reliable than Unsloth for a VLM).
        from finetuning.qwen38_convert import export_gguf as export_qwen38
        export_qwen38(MERGED_PATH, GGUF_PATH, MODEL_NAME)
        return
    print("\n📦 Exporting to GGUF Q4_K_M...")

    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(MERGED_PATH),
        max_seq_length = 2048,
        load_in_4bit   = False,  # Float16 for the export
        dtype          = torch.float16,
    )

    GGUF_PATH.mkdir(parents=True, exist_ok=True)

    # Unsloth >= 2026.11 writes the .gguf files into a SIBLING folder of the source folder:
    #   <source_folder>_gguf   (and no longer inside the source folder itself)
    GGUF_SRC = Path(f"{MERGED_PATH}_gguf")

    # save_pretrained_gguf expects:
    #   arg1 = SOURCE folder containing config.json (= MERGED_PATH)
    #   arg2 = tokenizer
    #   quantization_method = quant
    # We then move the produced .gguf to GGUF_PATH
    for quant in ["q4_k_m", "q8_0"]:
        print(f"   Quantizing {quant.upper()}...")
        model.save_pretrained_gguf(
            str(MERGED_PATH),
            tokenizer,
            quantization_method = quant,
        )
        # Move the .gguf of THIS quant to GGUF_PATH
        # (the intermediate .F16 stays in GGUF_SRC)
        import shutil, glob
        pattern = f"*.{quant.upper()}.gguf"
        produced = glob.glob(str(GGUF_SRC / pattern))
        if not produced:
            # Older Unsloth: the .gguf was written into the source folder
            produced = glob.glob(str(MERGED_PATH / pattern))
        if not produced:
            print(f"   ⚠️  No .gguf found for {quant} (looked in {GGUF_SRC})")
        for f in produced:
            dst = GGUF_PATH / f"{MODEL_NAME}-{quant}.gguf"
            shutil.move(f, dst)
            print(f"   → {dst} ({dst.stat().st_size / 1e9:.1f} GB)")

    print(f"✅ GGUF exported to {GGUF_PATH}")
    print(f"   Q4_K_M (~5 GB): for daily use")
    print(f"   Q8_0   (~9 GB): better quality")


def create_modelfile():
    """Creates the llama.cpp server configuration file."""
    serve_local = f"""#!/bin/bash
# Launching the llama.cpp server with the fine-tuned model
# OpenAI API compatible → used by the RAG pipeline

# libmtmd.so.0 is in the llama.cpp build folder
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export LD_LIBRARY_PATH="$PROJECT_DIR/llama.cpp/build/bin:$LD_LIBRARY_PATH"

llama-server \\
  -m {GGUF_PATH.resolve()}/{MODEL_NAME}-q4_k_m.gguf \\
  -ngl 99 \\
  --ctx-size 4096 \\
  --port 12080 \\
  --host 0.0.0.0 \\
  -np 1
# The chat template and thinking mode come from the Qwen3.8 GGUF metadata.
# Do NOT pass --chat-template: the new llama.cpp expects a Jinja template,
# not a model name. To disable thinking per request:
#   "chat_template_kwargs": {{"enable_thinking": false}}

# Test:
# curl http://localhost:12080/v1/chat/completions \\
#   -H "Content-Type: application/json" \\
#   -d '{{"model":"qwen38-code-civil","messages":[{{"role":"user","content":"What is Article 1240?"}}]}}'
"""

    serve_docker = f"""#!/bin/bash
# Launching the llama.cpp server in a Docker container with the fine-tuned model
# OpenAI API compatible → used by the RAG pipeline
# Backend LLAMA_BACKEND=docker (see .env), image {_server_image()}

# Replace any existing container with the same name
docker rm -f {_container_name()} >/dev/null 2>&1 || true

docker run -d --name {_container_name()} --no-healthcheck --gpus all \\
  -p 127.0.0.1:12080:12080 \\
  -v "{GGUF_PATH.resolve()}:/models:ro" \\
  {_server_image()} \\
  -m "/models/{MODEL_NAME}-q4_k_m.gguf" \\
  -ngl 99 \\
  --ctx-size 4096 \\
  --host 0.0.0.0 \\
  --port 12080 \\
  -np 1
# The chat template and thinking mode come from the Qwen3.8 GGUF metadata.
# Do NOT pass --chat-template: the new llama.cpp expects a Jinja template,
# not a model name. To disable thinking per request:
#   "chat_template_kwargs": {{"enable_thinking": false}}

# Test:
# curl http://localhost:12080/v1/chat/completions \\
#   -H "Content-Type: application/json" \\
#   -d '{{"model":"qwen38-code-civil","messages":[{{"role":"user","content":"What is Article 1240?"}}]}}'

# Follow the logs / stop the server:
#   docker logs -f {_container_name()}
#   docker stop {_container_name()}
"""

    config = serve_docker if _llama_backend() == "docker" else serve_local
    script_path = OUTPUT_DIR / "serve.sh"
    script_path.write_text(config)
    script_path.chmod(0o755)
    print(f"✅ Launch script: {script_path}")


def push_to_hub(repo_id: str):
    """Publishes to the HuggingFace Hub under the Apache 2.0 license."""
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\n🚀 Publishing to HuggingFace: {repo_id}")
    api.upload_folder(
        folder_path = str(MERGED_PATH),
        repo_id     = repo_id,
        repo_type   = "model",
    )
    print(f"✅ Published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf",  action="store_true")
    parser.add_argument("--push-hub",   type=str, default=None)
    args = parser.parse_args()

    if not args.skip_merge:
        merge_lora()

    if not args.skip_gguf:
        export_gguf()

    create_modelfile()

    if args.push_hub:
        push_to_hub(args.push_hub)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅ Model ready!                                      ║
║                                                      ║
║  Start the server:                                   ║
║    bash {OUTPUT_DIR}/serve.sh
║                                                      ║
║  Then the RAG pipeline:                              ║
║    python run.py --step rag                          ║
╚══════════════════════════════════════════════════════╝
""")
