"""
run.py  —  Main orchestrator — Civil Code RAG (Qwen3.8 + RTX 3080 Ti)

Usage:
  python run.py --step all          # Full pipeline
  python run.py --step ingest       # Step 1: Légifrance → Qdrant
  python run.py --step dataset      # Step 2: Q/A generation
  python run.py --step train        # Step 3: fine-tuning Qwen3.8-9B-Distill
  python run.py --step merge        # Step 4: merge + GGUF export
  python run.py --step rag          # Step 5: RAG pipeline test
  python run.py --step app          # Step 6: Streamlit interface

LLM backend:
  - local llama.cpp by default;
  - external OpenAI-compatible API if LLM_API_BASE_URL + LLM_API_KEY + LLM_MODEL
    are set in .env (LLM_BACKEND=auto), or if LLM_BACKEND=external;
  - LLM_BACKEND=local forces llama.cpp.

Options:
  --test        20 articles / 100 Q/A for a quick test
  --recreate    Recreate the Qdrant collection
  --from-cache  Use the local Légifrance cache
  --resume      Resume dataset generation
  --push-hub    Publish the model to HuggingFace
  --workers N   Parallel workers for the dataset (default: .env)
  --parallel N  Concurrent LLM calls (dataset and rag; default: .env)
  --sequential  Force a single LLM call at a time (dataset and rag)
"""

import argparse, os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv

# ── Fix import path ──────────────────────────────────────────
# Add the project root directory to PYTHONPATH
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Configuration (.env) ─────────────────────────────────────
# Loaded explicitly so that the checks below see the values from the
# file (and not only those exported in the shell).
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

from llm_backend import LLMConfigError, describe_backend, resolve_backend

# Create any missing __init__.py files
for _pkg in ("ingestion", "dataset", "finetuning", "rag", "interface"):
    _init = ROOT / _pkg / "__init__.py"
    _init.parent.mkdir(parents=True, exist_ok=True)
    if not _init.exists():
        _init.touch()


def banner(title):
    print(f"\n{'═'*55}\n  🏛️  {title}\n{'═'*55}")


# ── Configuration checks (hard stop if incomplete) ───
def _is_placeholder(value: str | None) -> bool:
    """True if the variable is missing or still a placeholder such as "your_…"."""
    if not value or not value.strip():
        return True
    return "your_" in value.lower()


def require_env_file(step: str) -> None:
    """Stop the process if .env is missing (actionable message)."""
    if not ENV_FILE.exists():
        print(f"\n❌ .env not found — step '{step}' is aborting.", file=sys.stderr)
        print("   → cp env.sample .env   (template provided at the project root)", file=sys.stderr)
        sys.exit(1)


def require_env(step: str, keys: list[tuple[str, str]]) -> None:
    """Stop the process if a required variable is empty/a placeholder."""
    missing = [f"{k} — {hint}" for k, hint in keys if _is_placeholder(os.getenv(k))]
    if missing:
        print(f"\n❌ Incomplete configuration for step '{step}' — aborting.", file=sys.stderr)
        for m in missing:
            print(f"   • {m}", file=sys.stderr)
        print("", file=sys.stderr)
        print("   → configuration template: env.sample", file=sys.stderr)
        print("     cp env.sample .env  then fill in the values", file=sys.stderr)
        print("   → Légifrance keys: https://piste.gouv.fr", file=sys.stderr)
        sys.exit(1)


def step_ingest(args):
    banner("STEP 1 — Légifrance ingestion → Qdrant")
    from ingestion.indexer import run_ingestion

    if args.from_cache:
        # Offline path: the article cache was built from the LEGI dump
        # (make articles-cache). No PISTE key is required.
        import json
        from ingestion.legifrance_client import CACHE_FILE
        if not CACHE_FILE.exists():
            print(f"❌ Article cache not found: {CACHE_FILE}", file=sys.stderr)
            print("   → make articles-cache", file=sys.stderr)
            sys.exit(1)
        articles = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"📂 Local cache: {len(articles)} articles")
    else:
        require_env("ingest", [
            ("LEGIFRANCE_CLIENT_ID",     "PISTE client ID (https://piste.gouv.fr)"),
            ("LEGIFRANCE_CLIENT_SECRET", "PISTE client secret"),
        ])
        from ingestion.legifrance_client import LegifranceClient
        articles = LegifranceClient().fetch_all(use_cache=False)

    if args.test: articles = articles[:20]
    run_ingestion(articles, recreate=args.recreate)
    print("✅ Ingestion complete")


def step_dataset(args):
    banner("STEP 2 — Q/A dataset generation")
    print("⚠️  Prerequisite: local llama.cpp OR an external API configured in .env")
    print("   (LLM_BACKEND=auto|external|local, LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL)\n")
    from dataset.generate_dataset import generate_dataset
    # args.max is None when --max is not provided: generate_dataset
    # then applies the DATASET_SIZE cap from .env (0 = all tasks).
    max_samples = 100 if args.test else args.max
    # --workers is the dataset option; --parallel is the global alias.
    workers = 1 if args.sequential else (
        args.workers if args.workers is not None else args.parallel
    )
    try:
        generate_dataset(max_samples=max_samples, resume=args.resume, workers=workers)
    except LLMConfigError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)
    print("✅ Dataset generated")


def step_train(args):
    banner("STEP 3 — Fine-tuning Qwen3.8-9B-Distill (Unsloth + QLoRA)")
    print("📊 RTX 3080 Ti 16 GB configuration:")
    print("   Model     : Qwen3.8-9B-Distill in 4-bit QLoRA")
    print("   Batch     : 4 × 4 = 16 (effective)")
    print("   Seq len   : 2048")
    print("   LoRA rank : 64")
    print("   Est. time : ~4-6 hours\n")
    result = subprocess.run(
        [sys.executable, "finetuning/train_qwen38.py"] +
        (["--test"] if args.test else []),
        check=False
    )
    if result.returncode != 0:
        print("❌ Training failed")
        sys.exit(1)
    print("✅ Fine-tuning complete")


def step_merge(args):
    banner("STEP 4 — LoRA merge + GGUF export")
    cmd = [sys.executable, "finetuning/merge_and_export.py"]
    if args.push_hub: cmd += ["--push-hub", args.push_hub]
    subprocess.run(cmd, check=True)
    print("✅ Model merged and exported to GGUF")


def step_rag(args):
    banner("STEP 5 — RAG pipeline test")
    print("⚠️  Prerequisite: local llama.cpp OR an external API configured in .env")
    print("   (LLM_BACKEND=auto|external|local, LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL)\n")

    try:
        from rag.rag_pipeline import CodeCivilRAG
        rag = CodeCivilRAG()
    except LLMConfigError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    tests = [
        ("What are the conditions for marriage in France?", True),
        ("What is tortious civil liability?", True),
        ("Is a verbal contract valid?", False),
    ]

    workers = 1 if args.sequential else (
        args.parallel if args.parallel is not None else rag.backend.parallel
    )
    try:
        results = rag.ask_many(
            [q for q, _ in tests],
            thinking=[thinking for _, thinking in tests],
            workers=workers,
        )
    except Exception as exc:
        print(f"\n❌ RAG batch failed: {exc}", file=sys.stderr)
        sys.exit(1)

    for (q, thinking), result in zip(tests, results):
        print(f"\n{'─'*50}")
        print(f"❓ {q}  [thinking={'ON' if thinking else 'OFF'}]")
        if result.get("thinking"):
            print(f"💭 {result['thinking'][:150]}...")
        print(f"\n🤖 {result['answer'][:300]}...")
        print("📚 " + " | ".join(f"Art.{s['num']}" for s in result["sources"][:3]))

    print(f"\n✅ RAG tests passed ({workers} concurrent LLM calls)")


def step_app(args):
    banner("STEP 6 — Streamlit interface")
    try:
        backend = resolve_backend()
        print(f"🤖 LLM backend: {describe_backend(backend)}")
    except LLMConfigError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)
    print("🌐 Available at http://localhost:8501")
    # Use the interpreter of the active environment (and its Streamlit),
    # not the streamlit binary from PATH which may come from another conda env.
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "interface/app.py"],
        check=True,
    )


def print_summary():
    print("""
╔═══════════════════════════════════════════════════════╗
║   CIVIL CODE RAG — Qwen3.8 — RTX 3080 Ti 16 GB        ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  PREREQUISITES                                        ║
║  1. LLM backend: local llama.cpp OR external          ║
║     OpenAI-compatible (LLM_API_BASE_URL/KEY/MODEL)    ║
║  2. If local: download Qwen3.8-27B-UD-IQ3_S.gguf      ║
║     from HuggingFace                                  ║
║  3. docker-compose up -d qdrant                       ║
║                                                       ║
║  PIPELINE                                             ║
║  Step 1: python run.py --step ingest   (~1h)          ║
║  Step 2: python run.py --step dataset  (~3h)          ║
║  Step 3: python run.py --step train    (~5h)          ║
║  Step 4: python run.py --step merge    (~20min)       ║
║  Step 5: python run.py --step rag                     ║
║  Step 6: python run.py --step app                     ║
║                                                       ║
║  PARALLELISM (.env)                                   ║
║  LLM_PARALLEL=4: concurrent calls (default)           ║
║  --workers N / --parallel N to override               ║
║                                                       ║
║  QUICK TEST (everything in 30 min)                    ║
║  python run.py --step all --test                      ║
╚═══════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",       default="summary",
        choices=["all","ingest","dataset","train","merge","rag","app","summary"])
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--max",        type=int, default=None,
        help="Maximum number of Q/A to generate (default: DATASET_SIZE from .env; 0 = all)")
    parser.add_argument("--workers",    type=int, default=None,
        help="Parallel workers for the dataset (default: DATASET_WORKERS/LLM_PARALLEL from .env)")
    parser.add_argument("--parallel",   type=int, default=None,
        help="Concurrent LLM calls (dataset and rag; default: LLM_PARALLEL from .env)")
    parser.add_argument("--sequential", action="store_true",
        help="Force a single LLM call at a time (dataset and rag)")
    parser.add_argument("--recreate",   action="store_true")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--push-hub",   type=str, default=None)
    args = parser.parse_args()

    steps = {
        "ingest" : step_ingest,
        "dataset": step_dataset,
        "train"  : step_train,
        "merge"  : step_merge,
        "rag"    : step_rag,
        "app"    : step_app,
    }

    if args.step == "summary":
        print_summary()
        return

    # Every step other than "summary" requires a .env file (hard stop otherwise).
    require_env_file(args.step)

    if args.step == "all":
        for name, fn in steps.items():
            if name != "app":
                fn(args)
        print("\n🎉 Full pipeline complete!")
        print("   Launch the interface: python run.py --step app")
    else:
        steps[args.step](args)


if __name__ == "__main__":
    main()
