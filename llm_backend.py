"""
llm_backend.py
──────────────
LLM backend selection: external OpenAI-compatible API or local llama.cpp.

Rules:
  - LLM_BACKEND=auto (default): external API if LLM_API_BASE_URL, LLM_API_KEY
    and LLM_MODEL are set, otherwise local llama.cpp.
  - LLM_BACKEND=external: forces the external API (LLM_MODEL required).
  - LLM_BACKEND=local: forces local llama.cpp.

No API key is displayed by describe_backend().
"""

from __future__ import annotations

import json
import os
import re
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:12080/v1"
DEFAULT_PARALLEL = 4
DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_RETRIES = 2


class LLMConfigError(RuntimeError):
    """Invalid or incomplete LLM configuration."""


@dataclass(frozen=True)
class LLMBackend:
    mode: str
    base_url: str
    api_key: str
    model: str | None
    timeout: float
    max_retries: int
    parallel: int
    thinking_kwarg: bool
    max_tokens_param: str


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


def _is_placeholder(value: str | None) -> bool:
    value = _clean(value)
    if not value:
        return True
    return "your_" in value.lower()


def _env_value(env: Mapping[str, str], key: str, default: str = "") -> str:
    return _clean(env.get(key, default))


def _parse_int(value: str | None, *, name: str, default: int, minimum: int) -> int:
    raw = _clean(value)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{name} must be an integer (received value: {raw!r})") from exc
    if parsed < minimum:
        raise LLMConfigError(f"{name} must be >= {minimum} (received value: {parsed})")
    return parsed


def _parse_float(value: str | None, *, name: str, default: float, minimum: float) -> float:
    raw = _clean(value)
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{name} must be a number (received value: {raw!r})") from exc
    if parsed < minimum:
        raise LLMConfigError(f"{name} must be >= {minimum} (received value: {parsed})")
    return parsed


def normalize_base_url(url: str) -> str:
    """Normalize an OpenAI-compatible endpoint.

    - remove the trailing slash;
    - accept a full endpoint .../v1/chat/completions;
    - add /v1 if the URL has no path.
    """
    raw = _clean(url)
    if not raw:
        raise LLMConfigError("LLM_API_BASE_URL is empty")
    if "://" not in raw:
        raise LLMConfigError(
            f"Invalid LLM_API_BASE_URL: {raw!r} (expected: http(s)://host[/path])"
        )
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LLMConfigError(
            f"Invalid LLM_API_BASE_URL: {raw!r} (expected: http(s)://host[/path])"
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if not path:
        path = "/v1"
    return urlunparse(parsed._replace(path=path))


def resolve_backend(env: Mapping[str, str] | None = None) -> LLMBackend:
    """Resolve the LLM backend to use.

    env=None uses os.environ. An explicit mapping makes testing easier.
    """
    source: Mapping[str, str] = os.environ if env is None else env

    mode = _env_value(source, "LLM_BACKEND", "auto").lower() or "auto"
    if mode not in ("auto", "external", "local"):
        raise LLMConfigError(
            f"Invalid LLM_BACKEND: {mode!r} (expected: auto, external or local)"
        )

    base_url = _env_value(source, "LLM_API_BASE_URL")
    api_key = _env_value(source, "LLM_API_KEY")
    model = _env_value(source, "LLM_MODEL")

    if _is_placeholder(base_url):
        base_url = ""
    if _is_placeholder(api_key):
        api_key = ""
    if _is_placeholder(model):
        model = ""

    parallel = _parse_int(
        source.get("LLM_PARALLEL"),
        name="LLM_PARALLEL",
        default=DEFAULT_PARALLEL,
        minimum=1,
    )
    timeout = _parse_float(
        source.get("LLM_TIMEOUT"),
        name="LLM_TIMEOUT",
        default=DEFAULT_TIMEOUT,
        minimum=1.0,
    )
    max_retries = _parse_int(
        source.get("LLM_MAX_RETRIES"),
        name="LLM_MAX_RETRIES",
        default=DEFAULT_MAX_RETRIES,
        minimum=0,
    )

    thinking_mode = _env_value(source, "LLM_THINKING_KWARG", "auto").lower() or "auto"
    if thinking_mode not in ("auto", "on", "off"):
        raise LLMConfigError(
            f"Invalid LLM_THINKING_KWARG: {thinking_mode!r} (expected: auto, on or off)"
        )

    max_tokens_param = (
        _env_value(source, "LLM_MAX_TOKENS_PARAM", "max_tokens") or "max_tokens"
    )
    if max_tokens_param not in ("max_tokens", "max_completion_tokens"):
        raise LLMConfigError(
            f"Invalid LLM_MAX_TOKENS_PARAM: {max_tokens_param!r} "
            "(expected: max_tokens or max_completion_tokens)"
        )

    local_base_url = normalize_base_url(
        _env_value(source, "LLAMACPP_BASE_URL", DEFAULT_LOCAL_BASE_URL)
        or DEFAULT_LOCAL_BASE_URL
    )

    external = False
    if mode == "external":
        missing = []
        if not base_url:
            missing.append("LLM_API_BASE_URL")
        if not api_key:
            missing.append("LLM_API_KEY")
        if not model:
            missing.append("LLM_MODEL")
        if missing:
            raise LLMConfigError(
                "LLM_BACKEND=external but the configuration is incomplete: "
                + ", ".join(missing)
                + "\n   → set these variables in .env, "
                "or use LLM_BACKEND=auto / local."
            )
        external = True
    elif mode == "auto":
        if base_url and api_key:
            if not model:
                raise LLMConfigError(
                    "LLM_API_BASE_URL and LLM_API_KEY are set but LLM_MODEL is empty.\n"
                    "   → set LLM_MODEL (e.g. gpt-4o-mini) in .env."
                )
            external = True
        elif base_url or api_key:
            warnings.warn(
                "Incomplete LLM_API_BASE_URL / LLM_API_KEY: using local llama.cpp.",
                UserWarning,
                stacklevel=2,
            )

    if external:
        return LLMBackend(
            mode="external",
            base_url=normalize_base_url(base_url),
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            parallel=parallel,
            thinking_kwarg=thinking_mode == "on",
            max_tokens_param=max_tokens_param,
        )

    return LLMBackend(
        mode="local",
        base_url=local_base_url,
        api_key="not-needed",
        model=None,
        timeout=timeout,
        max_retries=max_retries,
        parallel=parallel,
        thinking_kwarg=thinking_mode != "off",
        max_tokens_param=max_tokens_param,
    )


_thread_local = threading.local()


def make_client(backend: LLMBackend) -> OpenAI:
    return OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=backend.timeout,
        max_retries=backend.max_retries,
    )


def get_thread_client(backend: LLMBackend) -> OpenAI:
    key = (
        backend.mode,
        backend.base_url,
        backend.api_key,
        backend.timeout,
        backend.max_retries,
    )
    if getattr(_thread_local, "key", None) != key:
        _thread_local.client = make_client(backend)
        _thread_local.key = key
    return _thread_local.client


def resolve_model(
    backend: LLMBackend,
    client: OpenAI,
    preferred_local_models: list[str] | None = None,
) -> str:
    if backend.mode == "external":
        if not backend.model:
            raise LLMConfigError("LLM_MODEL is empty for the external backend")
        return backend.model

    preferred = [m for m in (preferred_local_models or []) if m]
    try:
        models = [m.id for m in client.models.list().data]
    except Exception:
        return preferred[-1] if preferred else "local-model"
    for candidate in preferred:
        if candidate in models:
            return candidate
    if models:
        return models[0]
    return preferred[-1] if preferred else "local-model"


def chat_kwargs(
    backend: LLMBackend,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    top_p: float | None = None,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs[backend.max_tokens_param] = max_tokens
    if backend.thinking_kwarg:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": thinking}
        }
    return kwargs


def extract_message(response) -> tuple[str, str | None]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Empty or malformed LLM response (no choices)")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("Malformed LLM response (message missing)")
    content = (getattr(message, "content", None) or "").strip()
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    if "<think>" in content and "</think>" in content:
        _, after = content.split("<think>", 1)
        think_part, after = after.split("</think>", 1)
        reasoning = reasoning or think_part.strip()
        content = after.strip()
    if not content and reasoning:
        content = reasoning
    if not content:
        raise RuntimeError("Empty LLM response")
    return content, reasoning or None


def describe_backend(backend: LLMBackend) -> str:
    if backend.mode == "external":
        return (
            f"External OpenAI-compatible API — {backend.base_url} "
            f"— model {backend.model}"
        )
    return f"Local llama.cpp — {backend.base_url}"


def parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


__all__ = [
    "LLMBackend",
    "LLMConfigError",
    "chat_kwargs",
    "describe_backend",
    "extract_message",
    "get_thread_client",
    "make_client",
    "normalize_base_url",
    "parse_json_object",
    "resolve_backend",
    "resolve_model",
]
