"""
train_qwen35.py
───────────────
Fine-tuning of Qwen3.5-9B with Unsloth + 4-bit QLoRA.
Optimized for RTX 3080 Ti (16 GB VRAM).

Prerequisites:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl==1.15.0 transformers==5.17.0 peft==0.21.2 accelerate bitsandbytes

Usage:
  python finetuning/train_qwen35.py
  python finetuning/train_qwen35.py --test   # 100 examples to check
"""

import os, json, argparse
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

# Avoid VRAM fragmentation while loading the model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Base model ───────────────────────────────────────────────
# Qwen3.5-4B: ~8 GB safetensors, fits in 16 GB VRAM, fine-tuning ~2h
# Qwen3.5-9B: ~19 GB safetensors, requires CPU-first loading (slower)
# BASE_MODEL_OPTIONS = {
#     "4B" : "unsloth/Qwen3.5-4B",   # ← recommended for RTX 3080 Ti 16 GB
#     "9B" : "unsloth/Qwen3.5-9B",   # ← possible but slow to load
# }
BASE_MODEL   = os.getenv("BASE_MODEL", "unsloth/Qwen3.5-4B")
DATASET_PATH = Path(os.getenv("DATASET_PATH", "./dataset/code_civil_qa.jsonl"))
OUTPUT_DIR   = Path(os.getenv("FINETUNED_MODEL_PATH", "./finetuning/output/qwen35-9b-code-civil"))


# ── Qwen 3.5 prompt format ───────────────────────────────────
# Qwen 3.5 uses the ChatML format with optional thinking support

def format_prompt(instruction: str, output: str, tokenizer) -> str:
    """Formats an example in the Qwen 3.5 ChatML format."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal assistant with expertise in French civil law. "
                "You help users understand the French Civil Code. "
                "Always cite the numbers of the relevant articles in your answers. "
                "Answer in English."
            )
        },
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": output},
    ]
    # apply_chat_template automatically handles the Qwen 3.5 format
    return tokenizer.apply_chat_template(
        messages,
        tokenize        = False,
        add_generation_prompt = False,
    )


def load_dataset(path: Path, test_mode: bool = False):
    """Loads and prepares the JSONL dataset."""
    from datasets import Dataset

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "instruction": item["instruction"],
                "output":      item["output"],
            })

    if test_mode:
        data = data[:100]
        print(f"🧪 Test mode: {len(data)} examples")

    print(f"📊 Dataset loaded: {len(data)} examples")
    return Dataset.from_list(data)


def train(test_mode: bool = False):
    # Unsloth patching active (imported at the top of the file)
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    import torch

    print(f"\n{'═'*55}")
    print(f"  🚀 Fine-tuning Qwen3.5-9B — RTX 3080 Ti 16 GB")
    print(f"{'═'*55}\n")

    # ── 1. Model loading ─────────────────────────────────────
    # Use Unsloth's pre-quantized 4-bit model to avoid OOM
    # Qwen/Qwen3.5-9B = 19 GB bfloat16 → too large for 16 GB VRAM
    # unsloth/Qwen3.5-9B-bnb-4bit = ~5 GB, loaded directly in 4-bit
    # Choose the model according to the available VRAM:
    # LOAD_MODEL = "unsloth/Qwen3.5-9B"   # 9B: ~19 GB RAM when loading, better quality
    LOAD_MODEL = BASE_MODEL                 # 4B by default (see BASE_MODEL at the top of the file)
    print(f"🔄 Loading {LOAD_MODEL} (4-bit QLoRA via bitsandbytes)...")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Direct 4-bit loading (bitsandbytes): device_map="auto" + max_memory set
    # the GPU (14 GiB) / CPU (40 GiB) split. The historical
    # "load on CPU first" workaround (transformers 5.2.0 bug) no longer matches the code.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOAD_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        dtype               = torch.bfloat16,
        trust_remote_code   = True,
        max_memory          = {0: "14GiB", "cpu": "40GiB"},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print("✅ Model loaded")

    # ── 2. LoRA configuration ────────────────────────────────
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r              = 32,   # Reduced from 64 → 32 to save VRAM
        lora_alpha     = 64,   # alpha = 2 × r
        lora_dropout   = 0.05,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Show trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 3. Dataset ───────────────────────────────────────────
    print("\n📂 Preparing the dataset...")
    raw_dataset = load_dataset(DATASET_PATH, test_mode=test_mode)

    # Format with the Qwen 3.5 ChatML template
    def format_batch(batch):
        return {
            "text": [
                format_prompt(inst, out, tokenizer)
                for inst, out in zip(batch["instruction"], batch["output"])
            ]
        }

    dataset = raw_dataset.map(format_batch, batched=True,
                               remove_columns=raw_dataset.column_names)

    # Train/validation split 95/5
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    train_d = split["train"]
    valid_d = split["test"]
    print(f"   Train: {len(train_d)} | Validation: {len(valid_d)}")

    # ── 4. Training arguments (optimized for 3080 Ti) ────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = 1 if test_mode else 3,
        per_device_train_batch_size = 1,  # OOM fix: batch 1 + grad accum
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = 16,   # Effective batch = 16
        warmup_steps                = 50,   # ~5% of 1000 steps
        learning_rate               = 2e-4,
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        bf16                        = True,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        optim                       = "adamw_8bit",
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 200,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",   # Set "wandb" if configured
        seed                        = 42,
        # SFT-specific fields (trl >= 1.0): pass them to the constructor, not afterwards.
        # trl 1.x: max_seq_length was renamed to max_length.
        max_length                  = 1024,  # Reduced to save VRAM
        packing                     = True,
        dataset_text_field          = "text",
    )

    # ── 5. Training ──────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        processing_class   = tokenizer,  # TRL >= 0.18: tokenizer → processing_class
        train_dataset      = train_d,
        eval_dataset       = valid_d,
        args               = args,
    )

    print(f"\n🏋️  Starting training...")
    print(f"   Model     : {BASE_MODEL}")
    print(f"   Examples  : {len(train_d)}")
    print(f"   Epochs    : {args.num_train_epochs}")
    print(f"   Batch     : {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"   Output    : {OUTPUT_DIR}\n")

    trainer_stats = trainer.train()

    # ── 6. Saving ────────────────────────────────────────────
    print("\n💾 Saving the LoRA adapters...")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))

    # Final stats
    duration_min = trainer_stats.metrics["train_runtime"] / 60
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Duration    : {duration_min:.1f} minutes")
    print(f"   Final loss  : {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Adapters    : {OUTPUT_DIR}/lora_adapters")
    print(f"\n➡️  Next step: python finetuning/merge_and_export.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="100 examples to test")
    args = parser.parse_args()
    train(test_mode=args.test)
