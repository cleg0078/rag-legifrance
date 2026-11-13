"""
rag_pipeline.py
───────────────
RAG pipeline: Qdrant (retrieval) + fine-tuned Qwen3.8 (generation).

LLM backend:
  - local llama.cpp by default;
  - external OpenAI-compatible API if LLM_API_BASE_URL + LLM_API_KEY +
    LLM_MODEL are set in .env (LLM_BACKEND=auto), or if LLM_BACKEND=external;
  - LLM_BACKEND=local forces llama.cpp.

The batch mode (ask_many) parallelizes only the generation calls;
vector search remains sequential to avoid concurrent access
to the embedding model.
"""

import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    resolve_backend,
    resolve_model,
)

FINETUNED_MODEL = "qwen38-code-civil"      # Fine-tuned Qwen3.8 adapter (9B-Distill)
FALLBACK_MODEL = os.getenv(
    "RAG_INFERENCE_MODEL", "Qwen3.8-27B-UD-IQ3_S.gguf"
)  # base model if the fine-tuned one is absent
TOP_K = 5

SYSTEM_PROMPT = """You are an expert legal assistant in French civil law.

ABSOLUTE RULES:
1. Always cite the numbers of the articles used (e.g. "According to article 1240 of the Civil Code...")
2. Rely ONLY on the articles provided in the context
3. If the provided articles do not allow answering, say so clearly
4. For complex questions, reason step by step before concluding
5. Point out areas of ambiguity or divergent interpretation
6. For any personal legal advice, direct the user to a professional

LANGUAGE:
- Always answer in English, even though the legal articles are in French.

FORMAT:
- Direct answer first
- Relevant articles developed
- Conclusion with the list of cited articles: "📌 Articles: 1240, 1241..."
"""


class CodeCivilRAG:

    def __init__(self):
        from ingestion.indexer import Embedder, VectorStore

        print("🚀 Initializing the RAG pipeline...")
        self.backend = resolve_backend()
        print(f"🤖 LLM backend: {describe_backend(self.backend)}")
        self.embedder = Embedder()
        self.store = VectorStore()
        self.client = get_thread_client(self.backend)
        self.model = self._resolve_model()
        self._embed_lock = threading.Lock()
        print(f"✅ Pipeline ready — backend: {self.backend.mode}, model: {self.model}")

    def _resolve_model(self) -> str:
        """Resolve the model to use (external or local)."""
        if self.backend.mode == "external":
            print(f"✅ External model: {self.backend.model}")
            return self.backend.model

        model = resolve_model(
            self.backend,
            self.client,
            [FINETUNED_MODEL, FALLBACK_MODEL],
        )
        print(f"✅ Local model detected: {model}")
        return model

    def retrieve(self, query: str) -> list[dict]:
        # The embedding model is not necessarily thread-safe: searches are
        # serialized even though generations run in parallel.
        with self._embed_lock:
            vector = self.embedder.encode_query(query)
        return self.store.search(vector, top_k=TOP_K)

    def build_context(self, articles: list[dict]) -> str:
        parts = []
        for a in articles:
            part = f"[Article {a.get('num','?')}]"
            part += f" (score: {a.get('score',0):.2f})\n"
            if a.get("breadcrumb"):
                part += f"Section: {a['breadcrumb']}\n"
            part += a.get("text", "")
            parts.append(part)
        return "\n\n---\n\n".join(parts)

    def _build_messages(self, question: str, history: list[dict] = None):
        articles = self.retrieve(question)
        context = self.build_context(articles)

        user_msg = f"""Context — Relevant articles of the French Civil Code:

{context}

Question: {question}

Answer citing the relevant articles. Answer in English."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_msg})
        return articles, messages

    def _generate(self, messages: list[dict], thinking: bool) -> tuple[str, str | None]:
        client = get_thread_client(self.backend)
        params = chat_kwargs(
            self.backend,
            self.model,
            messages,
            temperature=0.6 if thinking else 0.3,
            top_p=0.95,
            max_tokens=1500,
            thinking=thinking,
        )
        try:
            response = client.chat.completions.create(**params)
        except Exception as exc:
            raise RuntimeError(
                f"LLM call failed ({describe_backend(self.backend)}): {exc}"
            ) from exc
        return extract_message(response)

    def _format_result(self, articles: list[dict], answer: str,
                       thinking_content: str | None) -> dict:
        sources = [{
            "num": a.get("num", "?"),
            "titre": a.get("titre", ""),
            "extrait": a.get("text", "")[:300],
            "text": a.get("text", "")[:300],
            "score": a.get("score", 0),
            "breadcrumb": a.get("breadcrumb", ""),
        } for a in articles]
        return {
            "answer": answer,
            "thinking": thinking_content,
            "sources": sources,
            "model": self.model,
        }

    def ask(self, question: str, history: list[dict] = None,
            thinking: bool = True) -> dict:
        """
        thinking=True: enables step-by-step reasoning (better for complex
                       questions, slower).
        thinking=False: direct answer (faster).
        """
        articles, messages = self._build_messages(question, history)
        answer, thinking_content = self._generate(messages, thinking)
        return self._format_result(articles, answer, thinking_content)

    def ask_many(self, questions: list[str], history: list[dict] = None,
                 thinking=True, workers: int | None = None) -> list[dict]:
        """Answer several questions by parallelizing the LLM calls.

        - vector search is performed sequentially;
        - workers=None uses LLM_PARALLEL from .env (default 4);
        - thinking can be a boolean or a list of booleans per question.
        """
        if not questions:
            return []

        if isinstance(thinking, (list, tuple)):
            think_flags = [bool(t) for t in thinking]
            if len(think_flags) != len(questions):
                raise ValueError("thinking must have the same length as questions")
        else:
            think_flags = [bool(thinking)] * len(questions)

        workers = self.backend.parallel if workers is None else int(workers)
        workers = max(1, workers)

        if self.backend.mode == "local":
            try:
                local_parallel = int(os.getenv("LLAMACPP_PARALLEL", "4") or 4)
            except ValueError:
                local_parallel = 4
            if workers > local_parallel:
                print(
                    f"⚠️  {workers} parallel calls requested, but local llama.cpp "
                    f"is configured for {local_parallel} (--parallel). Extra "
                    "requests will be queued."
                )
        elif workers > 8:
            print(
                f"⚠️  {workers} parallel calls to the external API — risk of "
                "rate limiting; lower LLM_PARALLEL if needed."
            )

        prepared = [self._build_messages(q, history) for q in questions]

        if workers == 1:
            results = []
            for (articles, messages), think in zip(prepared, think_flags):
                answer, thinking_content = self._generate(messages, think)
                results.append(self._format_result(articles, answer, thinking_content))
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._generate, messages, think)
                for (_, messages), think in zip(prepared, think_flags)
            ]
            results = []
            for (articles, _), future in zip(prepared, futures):
                answer, thinking_content = future.result()
                results.append(self._format_result(articles, answer, thinking_content))
            return results

    def interactive(self):
        print("\n" + "=" * 55)
        print("  ⚖️  Civil Code RAG — Qwen3.8 × Qdrant × Légifrance")
        print("=" * 55)
        print(f"  Backend: {describe_backend(self.backend)}")
        print(f"  Model   : {self.model}")
        print("  Commands: 'think' (toggle), 'quit'")
        print("=" * 55 + "\n")

        thinking = True
        history = []
        while True:
            try:
                q = input("❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            if not q:
                continue
            if q.lower() == "quit":
                print("👋 Goodbye!")
                break
            if q.lower() == "think":
                thinking = not thinking
                print(f"🧠 Thinking: {'ON' if thinking else 'OFF'}")
                continue

            result = self.ask(q, history=history, thinking=thinking)
            print(f"\n🤖 {result['answer']}")
            if result.get("thinking"):
                print(f"\n💭 Reasoning: {result['thinking'][:200]}...")
            print(f"\n📚 Sources: {', '.join('Art.' + s['num'] for s in result['sources'])}")
            print("-" * 55 + "\n")

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    try:
        rag = CodeCivilRAG()
    except LLMConfigError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        result = rag.ask(" ".join(sys.argv[1:]))
        print(f"\n🤖 {result['answer']}")
        print(f"\n📚 Sources: {', '.join('Art.' + s['num'] for s in result['sources'])}")
    else:
        rag.interactive()
