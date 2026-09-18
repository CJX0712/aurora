"""Shared test fixtures. The agent is exercised with a scripted fake model so
the suite runs fully offline - no 5GB weights required."""

from __future__ import annotations

import json
from typing import List

import pytest

from aurora import types as T
from aurora.config import AgentConfig
from aurora.llm import ChatResponse


class ScriptedChat:
    """Returns queued responses in order; loops the last one if exhausted."""

    def __init__(self, script: List[ChatResponse]):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools, *, temperature=0.5, max_tokens=1024):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]


class FakeEmbedder:
    """Deterministic embedder: hashed bag-of-words over 256 dims.

    Coarse but collision-free enough to exercise ranking logic in tests; the
    real deployment uses nomic-embed-text via llama.cpp.
    """

    DIM = 256

    def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib

        out = []
        for t in texts:
            toks = [w.lower() for w in t.replace("search_document: ", "").replace("search_query: ", "").split()]
            vec = [0.0] * self.DIM
            for w in toks:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM
                vec[h] += 1.0
            out.append(vec)
        return out


def _tool_call(name, args, cid="c1"):
    return ChatResponse(
        content="",
        tool_calls=[T.ToolCall(id=cid, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


def _final(text):
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


@pytest.fixture
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = tmp_path / "ws"
    c.knowledge_dir = tmp_path / "kb"
    c.db_path = tmp_path / "aurora.db"
    c.ensure_dirs()
    return c


@pytest.fixture
def embedder():
    return FakeEmbedder()


def tool_call(name, args, cid="c1"):
    return _tool_call(name, args, cid)


def final(text):
    return _final(text)
