"""
tests/test_llm_backend.py
─────────────────────────
Unit tests for the LLM backend: resolution, request construction,
response extraction, simulated OpenAI-compatible server and parallelism.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from llm_backend import (
    LLMConfigError,
    chat_kwargs,
    describe_backend,
    extract_message,
    make_client,
    normalize_base_url,
    parse_json_object,
    resolve_backend,
    resolve_model,
)


class BackendResolutionTests(unittest.TestCase):

    def test_default_is_local(self):
        backend = resolve_backend({})
        self.assertEqual(backend.mode, "local")
        self.assertEqual(backend.base_url, "http://127.0.0.1:12080/v1")
        self.assertEqual(backend.api_key, "not-needed")
        self.assertIsNone(backend.model)
        self.assertEqual(backend.parallel, 4)
        self.assertTrue(backend.thinking_kwarg)

    def test_auto_uses_external_when_complete(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com/v1",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
        })
        self.assertEqual(backend.mode, "external")
        self.assertEqual(backend.base_url, "https://api.example.com/v1")
        self.assertEqual(backend.model, "model-test")
        self.assertFalse(backend.thinking_kwarg)

    def test_auto_requires_model_when_endpoint_and_key_present(self):
        with self.assertRaises(LLMConfigError):
            resolve_backend({
                "LLM_API_BASE_URL": "https://api.example.com/v1",
                "LLM_API_KEY": "sk-test",
            })

    def test_auto_partial_config_warns_and_falls_back_local(self):
        with self.assertWarns(UserWarning):
            backend = resolve_backend({"LLM_API_BASE_URL": "https://api.example.com/v1"})
        self.assertEqual(backend.mode, "local")

    def test_forced_external_requires_all_fields(self):
        with self.assertRaises(LLMConfigError) as ctx:
            resolve_backend({
                "LLM_BACKEND": "external",
                "LLM_API_BASE_URL": "https://api.example.com/v1",
                "LLM_API_KEY": "sk-test",
            })
        self.assertIn("LLM_MODEL", str(ctx.exception))

    def test_forced_local_ignores_external(self):
        backend = resolve_backend({
            "LLM_BACKEND": "local",
            "LLM_API_BASE_URL": "https://api.example.com/v1",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
        })
        self.assertEqual(backend.mode, "local")
        self.assertEqual(backend.api_key, "not-needed")

    def test_invalid_backend_fails(self):
        with self.assertRaises(LLMConfigError):
            resolve_backend({"LLM_BACKEND": "cloud"})

    def test_placeholders_are_ignored(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "your_url",
            "LLM_API_KEY": "your_key",
            "LLM_MODEL": "your_model",
        })
        self.assertEqual(backend.mode, "local")

    def test_parallel_default_and_override(self):
        self.assertEqual(resolve_backend({}).parallel, 4)
        self.assertEqual(resolve_backend({"LLM_PARALLEL": "7"}).parallel, 7)

    def test_parallel_invalid_fails(self):
        with self.assertRaises(LLMConfigError):
            resolve_backend({"LLM_PARALLEL": "0"})
        with self.assertRaises(LLMConfigError):
            resolve_backend({"LLM_PARALLEL": "abc"})

    def test_timeout_and_retries(self):
        backend = resolve_backend({})
        self.assertEqual(backend.timeout, 600.0)
        self.assertEqual(backend.max_retries, 2)
        backend = resolve_backend({"LLM_TIMEOUT": "12.5", "LLM_MAX_RETRIES": "0"})
        self.assertEqual(backend.timeout, 12.5)
        self.assertEqual(backend.max_retries, 0)


class UrlAndClientTests(unittest.TestCase):

    def test_normalize_base_url(self):
        self.assertEqual(
            normalize_base_url("https://api.openai.com"),
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            normalize_base_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            normalize_base_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1",
        )
        with self.assertRaises(LLMConfigError):
            normalize_base_url("api.openai.com")

    def test_make_client_arguments(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-secret",
            "LLM_MODEL": "model-test",
            "LLM_TIMEOUT": "42",
            "LLM_MAX_RETRIES": "3",
        })
        with patch("llm_backend.OpenAI") as mocked:
            make_client(backend)
            mocked.assert_called_once_with(
                base_url="https://api.example.com/v1",
                api_key="sk-secret",
                timeout=42.0,
                max_retries=3,
            )

    def test_describe_backend_hides_key(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-super-secret",
            "LLM_MODEL": "model-test",
        })
        description = describe_backend(backend)
        self.assertIn("https://api.example.com/v1", description)
        self.assertIn("model-test", description)
        self.assertNotIn("sk-super-secret", description)


class RequestAndResponseTests(unittest.TestCase):

    def test_chat_kwargs_local(self):
        backend = resolve_backend({})
        kwargs = chat_kwargs(
            backend, "local-model", [{"role": "user", "content": "x"}],
            temperature=0.5, thinking=True,
        )
        self.assertEqual(kwargs["model"], "local-model")
        self.assertEqual(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)

    def test_chat_kwargs_external_default_has_no_extra_body(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
        })
        kwargs = chat_kwargs(
            backend, "model-test", [{"role": "user", "content": "x"}],
            temperature=0.5, thinking=True,
        )
        self.assertNotIn("extra_body", kwargs)

    def test_chat_kwargs_external_thinking_on(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
            "LLM_THINKING_KWARG": "on",
        })
        kwargs = chat_kwargs(
            backend, "model-test", [{"role": "user", "content": "x"}],
            temperature=0.5, thinking=False,
        )
        self.assertEqual(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)

    def test_max_tokens_parameter_override(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
            "LLM_MAX_TOKENS_PARAM": "max_completion_tokens",
        })
        kwargs = chat_kwargs(
            backend, "model-test", [{"role": "user", "content": "x"}],
            temperature=0.5, max_tokens=123,
        )
        self.assertEqual(kwargs["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", kwargs)

    def test_extract_message_normal(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="  Hello  ", reasoning_content=None)
        )])
        self.assertEqual(extract_message(response), ("Hello", None))

    def test_extract_message_reasoning(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", reasoning_content="Reasoning")
        )])
        self.assertEqual(extract_message(response), ("Reasoning", "Reasoning"))

    def test_extract_message_think_tags(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="<think>Reasoning</think>Answer",
                reasoning_content=None,
            )
        )])
        self.assertEqual(extract_message(response), ("Answer", "Reasoning"))

    def test_extract_message_empty_raises(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", reasoning_content="")
        )])
        with self.assertRaises(RuntimeError):
            extract_message(response)

    def test_parse_json_object(self):
        fenced = chr(96) * 3 + "json\n" + '{"question": "Q", "reponse": "R"}' + chr(96) * 3
        self.assertEqual(
            parse_json_object(fenced),
            {"question": "Q", "reponse": "R"},
        )
        self.assertIsNone(parse_json_object("not json"))

    def test_resolve_model_external(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": "https://api.example.com",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "model-test",
        })
        self.assertEqual(resolve_model(backend, None, []), "model-test")

    def test_resolve_model_local_prefers_configured(self):
        backend = resolve_backend({})
        fake_client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="other"), SimpleNamespace(id="qwen38-code-civil")]
            ))
        )
        self.assertEqual(
            resolve_model(backend, fake_client, ["qwen38-code-civil", "fallback"]),
            "qwen38-code-civil",
        )


class MockHttpServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                MockHttpServerTests.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(body.decode("utf-8")),
                })
                payload = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        MockHttpServerTests.requests.clear()

    def test_external_openai_compatible_call(self):
        backend = resolve_backend({
            "LLM_API_BASE_URL": f"http://127.0.0.1:{self.port}",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
        })
        client = make_client(backend)
        response = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertEqual(response.choices[0].message.content, "Hello")
        self.assertEqual(len(MockHttpServerTests.requests), 1)
        request = MockHttpServerTests.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["auth"], "Bearer test-key")
        self.assertEqual(request["body"]["model"], "test-model")
        self.assertEqual(request["body"]["messages"][0]["content"], "Hi")


class FakeRAG:
    """Minimal subclass to test ask_many without loading Qdrant/torch."""

    def __init__(self, backend, barrier_parties=None):
        self.backend = backend
        self.model = "fake-model"
        self._embed_lock = threading.Lock()
        self._barrier = (
            threading.Barrier(barrier_parties, timeout=10)
            if barrier_parties else None
        )
        self.thread_ids = set()

    def _build_messages(self, question, history=None):
        return [{"question": question}], [{"role": "user", "content": question}]

    def _generate(self, messages, thinking):
        self.thread_ids.add(threading.get_ident())
        if self._barrier is not None:
            self._barrier.wait()
        return f"answer:{messages[0]['content']}", None

    def _format_result(self, articles, answer, thinking_content):
        return {
            "answer": answer,
            "thinking": thinking_content,
            "sources": [],
            "model": self.model,
        }


from rag.rag_pipeline import CodeCivilRAG  # noqa: E402


class FakeRAGImpl(FakeRAG, CodeCivilRAG):
    pass


class ParallelExecutionTests(unittest.TestCase):

    def test_ask_many_parallelizes_and_preserves_order(self):
        backend = resolve_backend({"LLM_PARALLEL": "3"})
        rag = FakeRAGImpl(backend, barrier_parties=3)
        results = rag.ask_many(["q1", "q2", "q3"], workers=3)
        self.assertEqual(
            [r["answer"] for r in results],
            ["answer:q1", "answer:q2", "answer:q3"],
        )
        self.assertEqual(len(rag.thread_ids), 3)

    def test_ask_many_sequential(self):
        backend = resolve_backend({})
        rag = FakeRAGImpl(backend)
        results = rag.ask_many(["a", "b"], workers=1)
        self.assertEqual([r["answer"] for r in results], ["answer:a", "answer:b"])

    def test_ask_many_thinking_list(self):
        backend = resolve_backend({})
        rag = FakeRAGImpl(backend)
        results = rag.ask_many(
            ["a", "b"], thinking=[True, False], workers=1
        )
        self.assertEqual(len(results), 2)
        with self.assertRaises(ValueError):
            rag.ask_many(["a"], thinking=[True, False], workers=1)


class DatasetGenerationTests(unittest.TestCase):

    def test_generate_qa_with_fake_client(self):
        from dataset.generate_dataset import generate_qa

        backend = resolve_backend({"LLM_BACKEND": "local"})
        article = {"num": "1240", "texte": "Test text", "breadcrumb": "Title"}
        task = {"type": "qa_direct", "prompt": "Article {num} : {texte}"}

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"question": "Q ?", "reponse": "R."}',
                        reasoning_content=None,
                    )
                )])

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        qa = generate_qa(article, task, client, backend, "fake-model")
        self.assertEqual(qa["instruction"], "Q ?")
        self.assertEqual(qa["output"], "R.")
        self.assertEqual(qa["metadata"]["article_num"], "1240")

    def test_generate_qa_invalid_json_returns_none(self):
        from dataset.generate_dataset import generate_qa

        backend = resolve_backend({"LLM_BACKEND": "local"})
        article = {"num": "1", "texte": "Text", "breadcrumb": ""}
        task = {"type": "qa_direct", "prompt": "Article {num} : {texte}"}

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="not json", reasoning_content=None)
                )])

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        self.assertIsNone(generate_qa(article, task, client, backend, "fake-model"))


if __name__ == "__main__":
    unittest.main()
