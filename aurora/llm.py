"""Model backends: local llama.cpp (in-process) and optional HTTP (OpenAI-compatible).

The agent depends only on two tiny protocols: ``ChatModel`` and ``Embedder``.
This keeps the whole system testable with a scripted fake model, and lets the
same code run on a cloud endpoint later by swapping the backend.
"""

from __future__ import annotations

import json
from typing import List, Optional, Protocol, runtime_checkable

from .config import ModelConfig
from .exceptions import ModelUnavailable
from .types import Message, ToolCall, ToolSpec


@runtime_checkable
class ChatModel(Protocol):
    def chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        *,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> "ChatResponse": ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class ChatResponse:
    def __init__(self, content: str, tool_calls: List[ToolCall], finish_reason: str):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason


def _parse_arguments(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


# Some local GGUF models (notably Qwen2.5 without a tool-calling chat template)
# emit tool invocations as literal text instead of the structured `tool_calls`
# field. These regexes recover that text form so the agent loop still works.
_TOOL_CALL_BLOCK = __import__("re").compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", __import__("re").DOTALL
)


def _extract_text_tool_calls(content: str, tools: List[ToolSpec]) -> List[ToolCall]:
    """Recover tool calls the model wrote as text (Qwen/Hermes style)."""
    import re

    if not content or "<tool_call>" not in content:
        return []
    known = {t.name for t in tools} if tools else None
    calls: List[ToolCall] = []
    for i, block in enumerate(_TOOL_CALL_BLOCK.findall(content)):
        block = block.strip()
        # Blocks look like: {"name": "...", "arguments": {...}}  (possibly fenced)
        block = re.sub(r"^```(?:json)?|```$", "", block, flags=re.MULTILINE).strip()
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            # Tolerate doubled braces produced by some templates: {{ ... }}
            fixed = block
            if fixed.startswith("{{") and fixed.endswith("}}"):
                fixed = fixed[1:-1]
            try:
                obj = json.loads(fixed)
            except json.JSONDecodeError:
                continue
        name = obj.get("name") or obj.get("tool")
        if not name:
            continue
        if known is not None and name not in known:
            continue
        args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        calls.append(ToolCall(id=f"text_call_{i}", name=name, arguments=_parse_arguments(args)))
    return calls


class LocalBackend:
    """Loads the chat + embedding GGUF in-process via llama.cpp.

    Falls back to a remote OpenAI-compatible server when ``chat_api_base`` is set.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._chat_llm = None
        self._embed_llm = None
        if cfg.chat_api_base:
            self._http = _HttpBackend(cfg)
        else:
            self._load()

    def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except Exception as exc:  # pragma: no cover - env dependent
            raise ModelUnavailable(f"llama_cpp not available: {exc}") from exc

        chat_path = self.cfg.chat_model_path
        embed_path = self.cfg.embed_model_path
        if not _exists(chat_path):
            raise ModelUnavailable(f"chat model not found: {chat_path}")
        if not _exists(embed_path):
            raise ModelUnavailable(f"embed model not found: {embed_path}")

        self._chat_llm = Llama(
            model_path=chat_path,
            n_ctx=self.cfg.chat_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            verbose=False,
        )
        self._embed_llm = Llama(
            model_path=embed_path,
            n_ctx=self.cfg.embed_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            embedding=True,
            verbose=False,
        )

    # ---- ChatModel ----
    def chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        *,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        if getattr(self, "_http", None) is not None:
            return self._http.chat(messages, tools, temperature=temperature, max_tokens=max_tokens)

        payload = [m.to_openai() for m in messages]
        tool_payload = [t.to_openai() for t in tools] if tools else None
        kwargs = dict(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tool_payload:
            kwargs["tools"] = tool_payload
            kwargs["tool_choice"] = "auto"

        out = self._chat_llm.create_chat_completion(**kwargs)
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
        calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc["function"]["name"],
                arguments=_parse_arguments(tc["function"].get("arguments")),
            )
            for i, tc in enumerate(raw_calls)
        ]
        # Fallback: recover tool calls the model emitted as literal text.
        if not calls:
            calls = _extract_text_tool_calls(content, tools)
            if calls:
                content = _TOOL_CALL_BLOCK.sub("", content).strip()
        finish = out["choices"][0].get("finish_reason", "stop")
        return ChatResponse(content=content, tool_calls=calls, finish_reason=finish)

    # ---- Embedder ----
    def embed(self, texts: List[str]) -> List[List[float]]:
        if getattr(self, "_http", None) is not None:
            return self._http.embed(texts)
        # nomic-embed-text benefits from task prefixes for asymmetric retrieval.
        prepped = [_prefix(t) for t in texts]
        result = self._embed_llm.embed(prepped)
        # llama_cpp returns List[float] for a single string or List[List[float]] for many.
        if result and isinstance(result[0], float):
            return [result]
        return result


def _prefix(text: str) -> str:
    # Heuristic: document chunks are longer; queries are short. nomic uses
    # "search_document: " / "search_query: " prefixes. We prefix long texts as docs.
    if len(text) > 200:
        return "search_document: " + text
    return "search_query: " + text


class _HttpBackend:
    """OpenAI-compatible HTTP backend (cloud / hybrid deployments)."""

    def __init__(self, cfg: ModelConfig):
        import httpx

        self.cfg = cfg
        self._http = httpx.Client(timeout=120.0)
        self._chat_base = (cfg.chat_api_base or "").rstrip("/")
        self._embed_base = (cfg.embed_api_base or cfg.chat_api_base or "").rstrip("/")
        self._key = cfg.api_key

    def chat(self, messages, tools, *, temperature=0.5, max_tokens=1024):
        import httpx

        payload = {"model": "remote", "messages": [m.to_openai() for m in messages],
                   "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        r = self._http.post(
            f"{self._chat_base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json=payload,
        )
        r.raise_for_status()
        out = r.json()
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc["function"]["name"],
                arguments=_parse_arguments(tc["function"].get("arguments")),
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        if not calls:
            calls = _extract_text_tool_calls(content, tools)
            if calls:
                content = _TOOL_CALL_BLOCK.sub("", content).strip()
        return ChatResponse(content=content, tool_calls=calls,
                            finish_reason=out["choices"][0].get("finish_reason", "stop"))

    def embed(self, texts):
        r = self._http.post(
            f"{self._embed_base}/embeddings",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": "remote", "input": texts},
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


def load_backend(cfg: ModelConfig):
    return LocalBackend(cfg)


def _exists(p: str) -> bool:
    from pathlib import Path

    return Path(p).exists()
