"""
train_qwen38.py
────────────────
4-bit QLoRA fine-tuning of Qwen3.8 (architecture `qwen3_5`, multimodal) on text only,
optimized for the RTX 3080 Ti 16 GB.

Recommended model: Qwen3.8-9B-Distill (empero-ai) — it is the largest Qwen3.8
that can be fine-tuned on 16 GB (real measurements taken here):
  • 4-bit loading (Unsloth FastVisionModel): ~8.6 GB of VRAM
  • forward + backward seq 512: peak ~12.3 GB  → fits in 16 GB
  • the official 27B (≈15 GB in 4-bit, multimodal) does not fit for training;
    it serves as teacher/inference model (GGUF UD-IQ3_S, ~12 GB).

Why Unsloth FastVisionModel and not AutoModelForCausalLM: the Qwen3.8 weights are
under `model.language_model.*` (VLM) + MTP layers; Unsloth correctly patches the
linear attention layers (fla) and reduces memory. The transformers-only path
exceeds 16 GB even at seq 256 (verified).

Prerequisites:
  pip install flash-linear-attention==0.6.0      # Qwen3.8 linear attention kernels
  pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

Usage:
  python finetuning/train_qwen38.py
  python finetuning/train_qwen38.py --test        # 2 steps on 100 examples
  python finetuning/train_qwen38.py --resume      # resume the LoRA adapters
"""

import os, sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import torch

# Avoid VRAM fragmentation (4-bit is close to 16 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from finetuning.train_qwen35 import format_prompt, load_dataset   # shared helpers

BASE_MODEL   = os.getenv("BASE_MODEL", "./models/Qwen3.8-9B-Distill")
DATASET_PATH = Path(os.getenv("DATASET_PATH", "./dataset/code_civil_qa.jsonl"))
OUTPUT_DIR   = Path(os.getenv("FINETUNED_MODEL_PATH", "./finetuning/output/qwen38-9b-code-civil"))
MAX_SEQ      = int(os.getenv("FINETUNE_MAX_SEQ", "512"))   # 512 = safe in 4-bit on 16 GB
LOAD_4BIT    = os.getenv("FINETUNE_LOAD_4BIT", "1") == "1" # H200: 0 (BF16 LoRA, better quality)
LORA_R       = int(os.getenv("FINETUNE_LORA_R", "16"))
LORA_ALPHA   = int(os.getenv("FINETUNE_LORA_ALPHA", str(LORA_R * 2)))
GRAD_ACCUM   = int(os.getenv("FINETUNE_GRAD_ACCUM", "16"))
EPOCHS       = int(os.getenv("FINETUNE_EPOCHS", "3"))
LR           = float(os.getenv("FINETUNE_LR", "2e-4"))


def build_model(max_seq: int):
    """Loads Qwen3.8 via Unsloth (4-bit on 16 GB, bf16 on H200) + LoRA."""
    from unsloth import FastVisionModel

    mode = "QLoRA 4-bit" if LOAD_4BIT else "LoRA BF16 (without quantization)"
    print(f"\U0001f504 Loading {BASE_MODEL} ({mode}, Unsloth)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        str(BASE_MODEL),
        load_in_4bit   = LOAD_4BIT,
        dtype          = torch.bfloat16,
        max_seq_length = max_seq,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r                        = LORA_R,
        lora_alpha               = LORA_ALPHA,
        lora_dropout             = 0.05,
        bias                     = "none",
        finetune_vision_layers   = False,   # legal text task
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules     = True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    )
    model.config.use_cache = False
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      action="store_true", help="2 steps on 100 examples")
    parser.add_argument("--resume",    action="store_true", help="resume the adapters")
    parser.add_argument("--max-steps", type=int, default=0, help="limit the number of steps (0 = full)")
    parser.add_argument("--epochs",    type=int, default=EPOCHS)
    parser.add_argument("--grad-accum",type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr",        type=float, default=LR)
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"\u274c Dataset not found: {DATASET_PATH}", file=sys.stderr)
        print("   Run first: make dataset", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\u274c CUDA unavailable: QLoRA fine-tuning requires the GPU.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model, tokenizer = build_model(MAX_SEQ)

    lora_path = OUTPUT_DIR / "lora_adapters"
    if args.resume and lora_path.exists():
        from peft import PeftModel
        print(f"\u21a9\ufe0f  Resuming from {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)

    dataset = load_dataset(DATASET_PATH, test_mode=args.test)
    examples = [
        format_prompt(item["instruction"], item["output"], tokenizer)
        for item in dataset
    ]

    # FastVisionModel returns a PROCESSOR for multimodal models:
    # processor("text") would interpret the 1st argument as IMAGES.
    # We therefore encode with the underlying tokenizer.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"\U0001f4ca {len(examples)} examples | {sum(p.numel() for p in trainable):,} trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    max_steps = args.max_steps or (2 if args.test else 0)
    grad_accum = max(1, args.grad_accum)
    model.train()

    global_step, micro, t0, losses = 0, 0, time.time(), []
    optimizer.zero_grad(set_to_none=True)
    done = False

    for epoch in range(args.epochs):
        print(f"\n\U0001f4d6 Epoch {epoch + 1}/{args.epochs}")
        for text in examples:
            enc = text_tokenizer(
                text,
                truncation = True,
                max_length = MAX_SEQ,
                return_tensors = "pt",
            ).to("cuda")
            out = model(input_ids=enc["input_ids"], labels=enc["input_ids"].clone())
            loss = out.loss / grad_accum
            loss.backward()
            losses.append(float(out.loss.detach()))
            micro += 1

            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.time() - t0
                print(f"  step {global_step:5d} | loss {sum(losses[-grad_accum:]) / grad_accum:.4f} | {global_step / elapsed:.2f} steps/s")
                if max_steps and global_step >= max_steps:
                    done = True
                    break
            del out, loss, enc
            if micro % 8 == 0:
                torch.cuda.empty_cache()
        if done:
            break

    lora_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"\n\u2705 LoRA adapters saved: {lora_path}")
    print(f"   {global_step} optimized steps — duration {(time.time() - t0) / 60:.1f} min")
    print("   Next: python run.py --step merge")


if __name__ == "__main__":
    main()
