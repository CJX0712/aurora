"""Local RAG: chunking, embedding, and cosine retrieval backed by SQLite.

No external vector database is required - embeddings are stored as BLOB and
scored with numpy. This keeps the system fully self-contained and portable,
while remaining trivially upgradable to sqlite-vec / HNSW later.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from pathlib import Path
from typing import Callable, List

import numpy as np

from .config import AgentConfig

_CHUNK_SEPS = ["\n## ", "\n\n", "\n", ". ", " "]
_TEXT_EXTS = {".md", ".txt", ".json", ".py", ".csv", ".yaml", ".yml", ".log", ".rst"}


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        # step forward, trying to break on a separator near the boundary
        nxt = end
        for sep in _CHUNK_SEPS:
            pos = text.rfind(sep, start + size - overlap, end)
            if pos != -1:
                nxt = pos + len(sep)
                break
        start = max(nxt, start + 1)
    return chunks


class KnowledgeBase:
    def __init__(self, cfg: AgentConfig, embedder: Callable[[List[str]], List[List[float]]]):
        self.cfg = cfg
        self.embedder = embedder
        self.db = cfg.db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT,
                idx INTEGER,
                text TEXT,
                emb BLOB
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
        conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(str(self.db))
        conn.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()

    def count(self) -> int:
        conn = sqlite3.connect(str(self.db))
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        return n

    def ingest_path(self, path: str) -> int:
        root = Path(path)
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.suffix.lower() in _TEXT_EXTS and p.is_file()]
        total = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            total += self.ingest_text(text, str(f))
        return total

    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed(chunks)
        conn = sqlite3.connect(str(self.db))
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        rows = [
            (source, i, c, np.asarray(v, dtype=np.float32).tobytes())
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        conn.executemany("INSERT INTO chunks (source, idx, text, emb) VALUES (?,?,?,?)", rows)
        conn.commit()
        conn.close()
        return len(rows)

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        top_k = top_k or self.cfg.top_k
        conn = sqlite3.connect(str(self.db))
        rows = conn.execute("SELECT id, source, text, emb FROM chunks").fetchall()
        conn.close()
        if not rows:
            return []
        q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for _id, source, text, emb in rows:
            v = np.frombuffer(emb, dtype=np.float32)
            vn = np.linalg.norm(v) or 1.0
            cos = float(np.dot(q, v) / (qn * vn))
            scored.append({"id": _id, "source": source, "text": text, "score": cos})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
