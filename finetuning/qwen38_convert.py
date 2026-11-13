"""
qwen38_convert.py - LoRA merge and GGUF export for Qwen3.8 (arch qwen3_5).

Uses transformers + peft for the merge (multimodal model, weights under
model.language_model.*) then the llama.cpp tools that support qwen35:
  - convert_hf_to_gguf.py (f16)
  - llama-quantize (Q4_K_M, Q8_0)

Conversion/quantization backend (LLAMA_BACKEND variable, read from .env
or the environment, default = local):
  - local  : binaries built in llama.cpp/ (historical behavior)
  - docker : llama.cpp tools image (LLAMA_CPP_TOOLS_IMAGE), nothing to build

Do not use Unsloth FastLanguageModel here: it is meant for plain text.
"""

import os, sys, subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Loads .env for LLAMA_BACKEND / LLAMA_CPP_TOOLS_IMAGE when launched directly.
load_dotenv(ROOT / ".env")

LLAMA_DIR = ROOT / "llama.cpp"
CONVERTER = LLAMA_DIR / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA_DIR / "build" / "bin" / "llama-quantize"

# Docker backend: "full" image (convert_hf_to_gguf.py + llama-quantize).
DEFAULT_TOOLS_IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1a4983fc58a092dc0a63e13d07213f58ffbcf454143f0a433b25bcfe4d109f06"
TOOLS_SH = "/app/tools.sh"
QUANTS = (("q4_k_m", "Q4_K_M"), ("q8_0", "Q8_0"))

DOCKER_HINT = (
    "check the tools image: make llama-tools-pull ; "
    "if the image is private: docker login ghcr.io"
)
LOCAL_HINT = "rebuild llama.cpp: cd llama.cpp && cmake --build build -j"


def llama_backend() -> str:
    """llama.cpp backend: 'docker' if LLAMA_BACKEND=docker, otherwise 'local'."""
    return (os.getenv("LLAMA_BACKEND") or "local").strip().lower()


def _tools_image() -> str:
    """Docker image of the llama.cpp tools (conversion + quantization)."""
    return os.getenv("LLAMA_CPP_TOOLS_IMAGE", DEFAULT_TOOLS_IMAGE).strip() or DEFAULT_TOOLS_IMAGE


def _run(cmd, action: str, hint: str = DOCKER_HINT) -> None:
    """Runs a command (check=True) with an actionable error message."""
    print("  $", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{action}: command not found ({cmd[0]}).") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{action} failed (code {exc.returncode}); {hint}"
        ) from exc


def _ensure_tools_image(image: str) -> None:
    """Checks that the tools image is present; otherwise docker pull."""
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except FileNotFoundError as exc:
        raise RuntimeError(
            "docker not found in PATH: install Docker or pass "
            "LLAMA_BACKEND=local."
        ) from exc
    except subprocess.CalledProcessError:
        pass  # image missing -> download

    print("Tools image missing, downloading", image, "...")
    _run(["docker", "pull", image], "docker pull " + image)


def _docker_convert(image: str, merged_path: Path, gguf_path: Path, f16: Path) -> None:
    """HF -> GGUF f16 conversion inside the tools image (bind mounts)."""
    _run(
        [
            "docker", "run", "--rm",
            "-v", f"{merged_path.resolve()}:/in:ro",
            "-v", f"{gguf_path.resolve()}:/out",
            "--entrypoint", TOOLS_SH,
            image,
            "--convert", "/in",
            "--outtype", "f16",
            "--outfile", f"/out/{f16.name}",
        ],
        "HF -> GGUF conversion (f16, docker)",
    )


def _docker_quantize(image: str, gguf_path: Path, f16: Path, dst: Path, label: str) -> None:
    """Quantization of an f16 GGUF inside the tools image."""
    _run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            # llama-quantize is linked against libcuda in -cuda images:
            # without --gpus all it fails at load time (libcuda.so.1 missing).
            "-v", f"{gguf_path.resolve()}:/out",
            "--entrypoint", TOOLS_SH,
            image,
            "--quantize", f"/out/{f16.name}", f"/out/{dst.name}", label,
        ],
        "Quantization " + label + " (docker)",
    )


def _docker_fix_owner(image: str, gguf_path: Path) -> None:
    """Gives the current user ownership of the files written by the root container."""
    uid, gid = os.getuid(), os.getgid()
    _run(
        [
            "docker", "run", "--rm",
            "-v", f"{gguf_path.resolve()}:/out",
            "--entrypoint", "/bin/sh",
            image,
            "-c", f"chown -R {uid}:{gid} /out",
        ],
        "Fixing permissions of the GGUF folder",
    )


def merge(base_model: str, lora_path: Path, merged_path: Path) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    import os
    device = os.getenv("MERGE_DEVICE", "cpu")   # H200: MERGE_DEVICE=auto
    print("Merging Qwen3.8 LoRA via transformers (device:", device, ", bf16)...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_model), dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    AutoProcessor.from_pretrained(str(base_model)).save_pretrained(str(merged_path))
    print("Merge complete ->", merged_path)


def export_gguf(merged_path: Path, gguf_path: Path, model_name: str) -> None:
    merged_path = Path(merged_path)
    gguf_path = Path(gguf_path)
    backend = llama_backend()

    if backend == "docker":
        image = _tools_image()
        if not merged_path.is_dir():
            raise FileNotFoundError(
                f"Merged model not found: {merged_path}; run the merge first"
            )
        _ensure_tools_image(image)
    else:
        if not CONVERTER.exists() or not QUANTIZE.exists():
            raise FileNotFoundError(
                "convert_hf_to_gguf.py or llama-quantize not found in llama.cpp/; "
                "rebuild: cd llama.cpp && cmake --build build -j"
            )

    gguf_path.mkdir(parents=True, exist_ok=True)
    f16 = gguf_path / (model_name + "-f16.gguf")

    print("Converting HF -> GGUF (f16) via llama.cpp...")
    if backend == "docker":
        _docker_convert(image, merged_path, gguf_path, f16)
    else:
        _run(
            [sys.executable, str(CONVERTER), str(merged_path),
             "--outtype", "f16", "--outfile", str(f16)],
            "HF -> GGUF conversion (f16)",
            hint=LOCAL_HINT,
        )

    for quant, label in QUANTS:
        dst = gguf_path / (model_name + "-" + quant + ".gguf")
        print("Quantizing", label, "...")
        if backend == "docker":
            _docker_quantize(image, gguf_path, f16, dst, label)
        else:
            _run([str(QUANTIZE), str(f16), str(dst), label],
                 "Quantization " + label, hint=LOCAL_HINT)
        print("  ->", dst, round(dst.stat().st_size / 1e9, 1), "GB")

    # The container writes as root: give the files back to the current user.
    if backend == "docker":
        _docker_fix_owner(image, gguf_path)

    f16.unlink(missing_ok=True)
    print("GGUF exported to", gguf_path)
