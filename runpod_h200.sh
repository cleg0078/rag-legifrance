#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  runpod_h200.sh — Fine-tuning Qwen3.8-27B on RunPod H200 (141 GB)
#
#  Why the H200 changes everything: the real Qwen3.8-27B (multimodal) does not fit
#  on the 16 GB RTX 3080 Ti, but fits comfortably on a 141 GB H200.
#  We train it in LoRA BF16 (no quantization): better quality and
#  trivial merge/GGUF export, unlike the local machine's 4-bit QLoRA.
#
#  Estimated duration: ~1.5-2 h (training) + ~20-30 min (merge + GGUF export)
#  Estimated cost: ~$7-10 (H200 SXM at ~$3.5/h on RunPod)
#
#  RunPod prerequisites:
#    - Template: recent PyTorch (NVIDIA driver >= 580 for cu132 wheels;
#                 automatic cu126 fallback otherwise)
#    - GPU      : 1× H200 SXM (141 GB)
#    - Disk     : 400 GB (27B BF16 54 GB + merged 54 GB + GGUF f16 54 GB
#                 + Q4/Q8 ~45 GB)
#    - Copy the complete repository to /workspace/code-civil (not only the
#      dataset: the script uses finetuning/train_qwen38.py and llama.cpp)
#    - Copy dataset/code_civil_qa.jsonl to /workspace/code-civil/dataset/
#
#  Usage:
#    1. Create the RunPod pod with the specs above
#    2. Copy the repository + dataset (see UPLOAD section below)
#    3. Open the RunPod terminal:
#         cd /workspace/code-civil && bash runpod_h200.sh
#    4. Retrieve the GGUF files (DOWNLOAD section at the end)
# ══════════════════════════════════════════════════════════════

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}ℹ  $*\033[0m"; }
ok()    { echo -e "${GREEN}✅ $*\033[0m"; }
warn()  { echo -e "${YELLOW}⚠️  $*\033[0m"; }
err()   { echo -e "${RED}❌ $*\033[0m"; exit 1; }
banner(){ echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
          echo -e "${GREEN}  $*${NC}"; \
          echo -e "${GREEN}══════════════════════════════════════════${NC}\n"; }

# ── Configuration ────────────────────────────────────────────
WORK_DIR="/workspace/code-civil"
BASE_MODEL="Qwen/Qwen3.8-27B"                       # official model (27B dense, multimodal)
OUTPUT_DIR="$WORK_DIR/output/qwen38-27b-code-civil"
MODEL_NAME="qwen38-27b-code-civil"
DATASET="$WORK_DIR/dataset/code_civil_qa.jsonl"

# H200 settings (LoRA BF16, no quantization)
H200_MAX_SEQ=2048
H200_LORA_R=64
H200_GRAD_ACCUM=8
H200_EPOCHS=3
H200_LR=1e-4

banner "🏛️  Fine-tuning Qwen3.8-27B — French Code Civil — H200"

# ── Preliminary checks ───────────────────────────────────────
cd "$WORK_DIR" 2>/dev/null || err "Directory not found: $WORK_DIR (copy the repository to /workspace/code-civil)"

[ -f "$WORK_DIR/run.py" ] || err "Incomplete repository: run.py missing from $WORK_DIR"
[ -f "$WORK_DIR/finetuning/train_qwen38.py" ] || err "Incomplete repository: finetuning/train_qwen38.py missing"

if [ ! -f "$DATASET" ]; then
    warn "Dataset not found: $DATASET"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │  DATASET UPLOAD (from your local machine):              │"
    echo "  │                                                         │"
    echo "  │  Option A — RunPod File Browser                         │"
    echo "  │  → dataset/code_civil_qa.jsonl                          │"
    echo "  │  → to /workspace/code-civil/dataset/                    │"
    echo "  │                                                         │"
    echo "  │  Option B — scp (replace POD_IP and PORT):              │"
    echo "  │  scp -P PORT dataset/code_civil_qa.jsonl \\             │"
    echo "  │      root@POD_IP:/workspace/code-civil/dataset/         │"
    echo "  └─────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Then rerun: bash runpod_h200.sh"
    exit 0
fi

DATASET_LINES=$(wc -l < "$DATASET")
ok "Dataset: $DATASET_LINES examples"

# ── GPU / CUDA ───────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
NVCC_VER=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | cut -d' ' -f2 || echo "N/A")

echo "  GPU       : $GPU_NAME ($GPU_VRAM)"
echo "  Driver    : $DRIVER"
echo "  nvcc      : $NVCC_VER (used to compile llama.cpp)"
echo "  Python    : $(python --version 2>&1)"
echo ""

# cu132 wheels: require a driver >= 580; otherwise automatic cu126 fallback.
if [ "$DRIVER_MAJOR" -ge 580 ] 2>/dev/null; then
    TORCH_INDEX="cu132"
else
    TORCH_INDEX="cu126"
    warn "Driver $DRIVER < 580 → installing cu126 wheels (instead of cu132)"
fi

# ══════════════════════════════════════════════════════════════
banner "STEP 1 — Dependency installation"

info "PyTorch 2.14.1+$TORCH_INDEX (embedded CUDA runtime)..."
pip install "torch==2.14.1+$TORCH_INDEX" "torchvision==0.29.1+$TORCH_INDEX" \
    --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" -q \
  || pip install "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" \
    --index-url https://download.pytorch.org/whl/cu126 -q \
  || err "PyTorch installation failed (driver too old?)"

info "Unsloth..."
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" -q
pip install unsloth-zoo --upgrade -q

info "Stack dependencies (same versions as the local machine)..."
pip install \
    transformers==5.17.0 \
    peft==0.21.2 \
    trl==1.15.0 \
    bitsandbytes==0.50.2 \
    accelerate==1.15.0 \
    datasets==4.8.5 \
    torchao==0.18.0 \
    flash-linear-attention==0.6.0 \
    python-dotenv huggingface_hub hf_transfer -q

info "Checking the stack..."
python - <<'PYCHECK'
import torch
import transformers
import unsloth
import fla
print("  torch         :", torch.__version__, "| runtime CUDA", torch.version.cuda)
print("  transformers  :", transformers.__version__)
print("  fla           :", fla.__version__)
print("  CUDA available:", torch.cuda.is_available(), "-", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("  unsloth       : OK")
PYCHECK
[ "$(python -c 'import torch; print(int(torch.cuda.is_available()))')" = "1" ] || err "CUDA unavailable"
ok "Dependencies installed"

# ── llama.cpp: only for GGUF export (CPU build, no CUDA needed)
banner "STEP 2 — llama.cpp (GGUF export tool)"

if [ ! -f "$WORK_DIR/llama.cpp/convert_hf_to_gguf.py" ]; then
    info "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$WORK_DIR/llama.cpp"
fi

if [ ! -f "$WORK_DIR/llama.cpp/build/bin/llama-quantize" ]; then
    info "Compiling llama-quantize (CPU, ~2-3 min)..."
    cd "$WORK_DIR/llama.cpp"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF >/dev/null
    cmake --build build --target llama-quantize -j "$(nproc)" >/dev/null
    cd "$WORK_DIR"
fi
[ -x "$WORK_DIR/llama.cpp/build/bin/llama-quantize" ] || err "llama-quantize not found after compilation"
ok "llama-quantize ready (qwen35 architecture support)"

# ══════════════════════════════════════════════════════════════
banner "STEP 3 — Fine-tuning Qwen3.8-27B (LoRA BF16)"

mkdir -p "$WORK_DIR/logs" "$OUTPUT_DIR"

info "Model: $BASE_MODEL (27B, ~54 GB BF16 in VRAM)"
info "Config: seq $H200_MAX_SEQ | LoRA r=$H200_LORA_R | grad-accum $H200_GRAD_ACCUM | $H200_EPOCHS epochs"
info "Logs   : $WORK_DIR/logs/train.log"
echo ""

BASE_MODEL="$BASE_MODEL" \
FINETUNED_MODEL_PATH="$OUTPUT_DIR" \
FINETUNE_LOAD_4BIT=0 \
FINETUNE_MAX_SEQ="$H200_MAX_SEQ" \
FINETUNE_LORA_R="$H200_LORA_R" \
FINETUNE_GRAD_ACCUM="$H200_GRAD_ACCUM" \
FINETUNE_EPOCHS="$H200_EPOCHS" \
FINETUNE_LR="$H200_LR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python finetuning/train_qwen38.py 2>&1 | tee "$WORK_DIR/logs/train.log"

ok "Fine-tuning complete"

# ══════════════════════════════════════════════════════════════
banner "STEP 4 — LoRA merge + GGUF export"

# MERGE_DEVICE=auto: the merge uses the GPU (141 GB, no need to go through the CPU)
BASE_MODEL="$BASE_MODEL" \
FINETUNED_MODEL_PATH="$OUTPUT_DIR" \
MERGE_DEVICE=auto \
    python finetuning/merge_and_export.py 2>&1 | tee "$WORK_DIR/logs/merge.log"

ok "GGUF export complete"

# ══════════════════════════════════════════════════════════════
banner "🎉 DONE"

GGUF_DIR="$OUTPUT_DIR/gguf"
echo "  Generated files:"
ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null || echo "  No GGUF found in $GGUF_DIR"
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  DOWNLOAD (from your local machine):                    │"
echo "  │                                                         │"
echo "  │  scp -P PORT \\                                         │"
echo "  │    root@POD_IP:$GGUF_DIR/\\*q4_k_m.gguf \\              │"
echo "  │    ./models/                                            │"
echo "  │                                                         │"
echo "  │  Then locally:                                          │"
echo "  │  ./start_llama.sh --model ./models/$MODEL_NAME-q4_k_m.gguf --mode rag"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo "  💡 The 'merged/' folder (54 GB) is no longer needed after the export:"
echo "     rm -rf $OUTPUT_DIR/merged"
