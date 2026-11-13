"""
generate_dataset.py
───────────────────
Generates a fine-tuning dataset from articles of the French Civil Code.

LLM backend:
  - local llama.cpp by default (DATASET_GEN_MODEL);
  - external OpenAI-compatible API if LLM_API_BASE_URL + LLM_API_KEY + LLM_MODEL
    are set in .env (LLM_BACKEND=auto), or if LLM_BACKEND=external;
  - LLM_BACKEND=local forces llama.cpp.

Parallelism:
  - LLM_PARALLEL in .env (default 4) controls the number of concurrent calls;
  - --workers / --parallel override this value;
  - --sequential forces a single call at a time.

Output format: JSONL (instruction / input / output) — Alpaca style
Compatible with Unsloth for fine-tuning.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_backend import (
    LLMConfigError,
    chat_kwargs,
    describe_backend,
    extract_message,
    get_thread_client,
    parse_json_object,
    resolve_backend,
    resolve_model,
)

DATASET_SIZE = int(os.getenv("DATASET_SIZE", 5000))
DATASET_PATH = Path(os.getenv("DATASET_PATH", "./dataset/code_civil_qa.jsonl"))
CACHE_FILE = Path("./ingestion/cache_articles.json")
GEN_MODEL = os.getenv("DATASET_GEN_MODEL", "qwen3.5-9b")

_ERROR_LOCK = Lock()
_LAST_ERROR: str | None = None


def _record_error(exc: Exception) -> None:
    global _LAST_ERROR
    with _ERROR_LOCK:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"


# ── Generation templates ──────────────────────────────────────

TASK_TEMPLATES = [
    {
        "type": "qa_direct",
        "prompt": """You are an expert in French civil law.
From this article of the French Civil Code, generate ONE precise question and its complete answer.
The article text is in French; write the question and the answer in English.

Article {num}: {texte}

Answer ONLY with valid JSON, without markdown, using this exact format:
{{"question": "...", "reponse": "..."}}""",
    },
    {
        "type": "vulgarisation",
        "prompt": """You are an expert in French civil law.
Explain article {num} of the French Civil Code in simple terms, as if speaking to a non-lawyer.
The article text is in French; write your explanation in English.

Article {num}: {texte}

Answer ONLY with valid JSON, without markdown:
{{"question": "Can you explain article {num} of the French Civil Code in simple terms?", "reponse": "..."}}""",
    },
    {
        "type": "cas_pratique",
        "prompt": """You are an expert in French civil law.
From this article, invent a realistic practical case and its legal solution.
The article text is in French; write the question and the answer in English.

Article {num}: {texte}

Answer ONLY with valid JSON, without markdown:
{{"question": "Practical case: ...", "reponse": "Solution: ..."}}""",
    },
    {
        "type": "conditions",
        "prompt": """You are an expert in French civil law.
From this article, formulate a question about the legal conditions or criteria required.
The article text is in French; write the question and the answer in English.

Article {num}: {texte}

Answer ONLY with valid JSON, without markdown:
{{"question": "What are the conditions for...?", "reponse": "..."}}""",
    },
    {
        "type": "raisonnement",
        "prompt": """You are an expert in French civil law.
From this article, generate a complex question that requires multi-step legal reasoning
to be resolved.
The article text is in French; write the question and the answer in English.

Article {num}: {texte}

Answer ONLY with valid JSON, without markdown:
{{"question": "...", "reponse": "Step 1: ... Step 2: ... Conclusion: ..."}}""",
    },
]


# ── Generation of a single example (thread-safe) ──────────────

def generate_qa(article: dict, task: dict, client, backend, model: str) -> dict | None:
    prompt = task["prompt"].format(
        num=article.get("num", "unknown"),
        texte=article.get("texte", "")[:800],
    )
    try:
        params = chat_kwargs(
            backend,
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
            thinking=False,
        )
        response = client.chat.completions.create(**params)
        content, reasoning = extract_message(response)
        content = content or reasoning or ""

        parsed = parse_json_object(content)
        if not parsed:
            _record_error(ValueError("invalid JSON"))
            return None
        if "question" not in parsed or "reponse" not in parsed:
            _record_error(ValueError("missing question/reponse fields"))
            return None

        return {
            "instruction": parsed["question"],
            "input": "",
            "output": parsed["reponse"],
            "metadata": {
                "article_num": article.get("num", ""),
                "task_type": task["type"],
                "breadcrumb": article.get("breadcrumb", ""),
            },
        }
    except Exception as exc:
        _record_error(exc)
        return None


# ── Worker for ThreadPoolExecutor ─────────────────────────────

def worker(args_tuple):
    """Each thread reuses its own OpenAI client (thread-local)."""
    article, task, backend, model = args_tuple
    client = get_thread_client(backend)
    return generate_qa(article, task, client, backend, model)


# ── Main pipeline ─────────────────────────────────────────────

def _resolve_target(max_samples: int | None, n_tasks: int) -> int:
    """Number of tasks to run, depending on the requested cap.

    - None  : no --max provided → configured cap (DATASET_SIZE from .env).
              This is the case of `make dataset` / `run.py --step dataset`.
    - <= 0  : no cap, process every available task.
    - otherwise: min(max_samples, available tasks).

    The historical bug: run.py passes args.max (None when absent) and the old
    code did min(max_samples, len(tasks)) directly →
    TypeError: '<' not supported between instances of 'int' and 'NoneType'.
    """
    if max_samples is None:
        max_samples = DATASET_SIZE
    if max_samples <= 0:
        return n_tasks
    return min(int(max_samples), n_tasks)


def _resolve_workers(workers: int | None, backend) -> int:
    """Resolves the number of workers: CLI > DATASET_WORKERS > LLM_PARALLEL."""
    if workers is not None:
        try:
            resolved = int(workers)
        except (TypeError, ValueError) as exc:
            raise LLMConfigError(f"invalid workers: {workers!r}") from exc
    else:
        env_workers = os.getenv("DATASET_WORKERS", "").strip()
        if env_workers:
            try:
                resolved = int(env_workers)
            except ValueError as exc:
                raise LLMConfigError(
                    f"invalid DATASET_WORKERS: {env_workers!r}"
                ) from exc
        else:
            resolved = backend.parallel

    if resolved < 1:
        raise LLMConfigError(
            f"The number of workers must be >= 1 (received: {resolved})"
        )
    return resolved


def generate_dataset(
    max_samples: int | None = None,
    resume: bool = False,
    workers: int | None = None,
):
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"❌ Articles cache not found: {CACHE_FILE}\n"
            "   Run first: python run.py --step ingest"
        )

    backend = resolve_backend()
    client = get_thread_client(backend)
    model = resolve_model(backend, client, [GEN_MODEL])
    workers = _resolve_workers(workers, backend)

    articles = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    print(f"📚 {len(articles)} articles loaded")
    print(f"🤖 LLM backend: {describe_backend(backend)}")
    print(f"🤖 Model: {model}")
    print(f"⚡ Parallel workers: {workers}")

    if backend.mode == "local":
        try:
            local_parallel = int(os.getenv("LLAMACPP_PARALLEL", "4") or 4)
        except ValueError:
            local_parallel = 4
        if workers > local_parallel:
            print(
                f"⚠️  {workers} workers requested, but local llama.cpp is configured "
                f"for {local_parallel} (--parallel). Extra requests "
                "will be queued."
            )
    elif workers > 8:
        print(
            f"⚠️  {workers} workers against the external API — rate-limit risk; "
            "reduce LLM_PARALLEL if needed."
        )

    already_done = 0
    mode = "a" if resume else "w"
    if resume and DATASET_PATH.exists():
        already_done = sum(1 for _ in DATASET_PATH.open(encoding="utf-8"))
        print(f"↩️  Resume: {already_done} examples already generated")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # Build the task list (article, template)
    tasks = []
    shuffled = articles.copy()
    random.shuffle(shuffled)
    for article in shuffled:
        if len(article.get("texte", "")) < 50:
            continue
        for task in TASK_TEMPLATES:
            tasks.append((article, task))

    target = _resolve_target(max_samples, len(tasks))
    tasks = tasks[already_done:target]  # Skip when resuming

    print(f"\n🚀 Generating {target} Q/A examples ({workers} workers)...")
    print(f"   Estimated time: ~{target // (workers * 60) + 1}h")
    print(f"   File: {DATASET_PATH}\n")

    global _LAST_ERROR
    _LAST_ERROR = None
    generated = already_done
    errors = 0
    write_lock = Lock()
    start_time = time.time()

    with open(DATASET_PATH, mode, encoding="utf-8") as f:
        with tqdm(total=len(tasks), desc="Generating", unit="Q/A") as pbar:
            if workers == 1:
                for article, task in tasks:
                    qa = worker((article, task, backend, model))
                    if qa:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        f.flush()
                        generated += 1
                    else:
                        errors += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "ok": generated,
                        "err": errors,
                        "rate": f"{generated / max(1.0, time.time() - start_time):.1f}/s",
                    })
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(worker, (article, task, backend, model)):
                            (article, task)
                        for article, task in tasks
                    }
                    for future in as_completed(futures):
                        try:
                            qa = future.result()
                        except Exception as exc:
                            _record_error(exc)
                            qa = None
                        if qa:
                            with write_lock:
                                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                                f.flush()
                            generated += 1
                        else:
                            errors += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "ok": generated,
                            "err": errors,
                            "rate": f"{generated / max(1.0, time.time() - start_time):.1f}/s",
                        })

    print(f"\n✅ Dataset generated:")
    print(f"   Valid examples: {generated}")
    print(f"   Errors        : {errors}")
    print(f"   Success rate  : {100 * generated / (generated + errors + 1):.1f}%")
    print(f"   File          : {DATASET_PATH}")
    if _LAST_ERROR:
        print(f"   Last error    : {_LAST_ERROR}")

    # Preview
    print("\n📋 Preview (first 3 examples):")
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in list(f)[:3]:
            ex = json.loads(line)
            print(f"\n  [{ex['metadata']['task_type']}] Art. {ex['metadata']['article_num']}")
            print(f"  Q: {ex['instruction'][:100]}...")
            print(f"  A: {ex['output'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of Q/A (default: DATASET_SIZE; 0 = all)",
    )
    parser.add_argument(
        "--workers", "--parallel", type=int, default=None, dest="workers",
        help="Number of parallel workers (default: DATASET_WORKERS/LLM_PARALLEL from .env)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Force a single LLM call at a time",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume if interrupted",
    )
    args = parser.parse_args()
    workers = 1 if args.sequential else args.workers
    try:
        generate_dataset(max_samples=args.max, resume=args.resume, workers=workers)
    except LLMConfigError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
