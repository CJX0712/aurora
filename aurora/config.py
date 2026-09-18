"""Configuration for Aurora, loaded from environment / .env / explicit dict.

All paths default to sensible locations under the project or user home so the
system works out of the box after `python -m aurora.serve`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_ROOT = Path(os.environ.get("AURORA_HOME", Path.home() / ".aurora"))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class ModelConfig:
    chat_model_path: str = field(
        default_factory=lambda: _env(
            "AURORA_CHAT_MODEL",
            str(DEFAULT_ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
        )
    )
    embed_model_path: str = field(
        default_factory=lambda: _env(
            "AURORA_EMBED_MODEL",
            str(DEFAULT_ROOT / "models" / "nomic-embed-text-v1.5.Q8_0.gguf"),
        )
    )
    # Number of transformer layers to offload to GPU. 0 = pure CPU.
    n_gpu_layers: int = field(default_factory=lambda: int(_env("AURORA_GPU_LAYERS", "0")))
    n_threads: int = field(
        default_factory=lambda: int(_env("AURORA_THREADS", str(os.cpu_count() or 8)))
    )
    chat_ctx: int = field(default_factory=lambda: int(_env("AURORA_CHAT_CTX", "8192")))
    embed_ctx: int = field(default_factory=lambda: int(_env("AURORA_EMBED_CTX", "8192")))
    # Remote OpenAI-compatible backend (optional). When set, local models are ignored.
    chat_api_base: Optional[str] = field(default_factory=lambda: _env("AURORA_CHAT_API_BASE", ""))
    embed_api_base: Optional[str] = field(default_factory=lambda: _env("AURORA_EMBED_API_BASE", ""))
    api_key: str = field(default_factory=lambda: _env("AURORA_API_KEY", "not-needed"))


@dataclass
class AgentConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    workspace: Path = field(
        default_factory=lambda: Path(_env("AURORA_WORKSPACE", str(DEFAULT_ROOT / "workspace")))
    )
    knowledge_dir: Path = field(
        default_factory=lambda: Path(_env("AURORA_KNOWLEDGE", str(DEFAULT_ROOT / "knowledge")))
    )
    db_path: Path = field(
        default_factory=lambda: Path(_env("AURORA_DB", str(DEFAULT_ROOT / "aurora.db")))
    )
    max_steps: int = field(default_factory=lambda: int(_env("AURORA_MAX_STEPS", "12")))
    max_tool_retries: int = 3
    temperature: float = 0.5
    # Rolling window token budget for short-term memory.
    context_token_budget: int = 6000
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 6
    host: str = field(default_factory=lambda: _env("AURORA_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("AURORA_PORT", "7680")))

    def ensure_dirs(self) -> None:
        for p in (self.workspace, self.knowledge_dir, self.db_path.parent):
            p.mkdir(parents=True, exist_ok=True)


def load_config() -> AgentConfig:
    return AgentConfig()
