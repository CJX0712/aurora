"""End-to-end smoke test with the REAL local models (no mocks).

Loads Qwen2.5-7B + nomic-embed-text in-process, runs one generation and one
full agent task (write a file, read it back, confirm). Prints clear markers.
"""

import os
import time

os.environ.setdefault(
    "AURORA_CHAT_MODEL",
    "/workspace/aurora/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
)
os.environ.setdefault("AURORA_EMBED_MODEL", "/workspace/aurora/models/nomic-embed-text-v1.5.Q8_0.gguf")
os.environ.setdefault("AURORA_HOME", "/workspace/aurora/.aurora")

from aurora.config import load_config
from aurora.llm import LocalBackend
from aurora.rag import KnowledgeBase
from aurora.tools import Toolbox
from aurora.agent import Agent
from aurora.types import Message, Role

cfg = load_config()
cfg.ensure_dirs()

t0 = time.time()
print("[1] loading models...", flush=True)
backend = LocalBackend(cfg.model_config)
print(f"[1] models loaded in {time.time()-t0:.1f}s", flush=True)

print("[2] single generation:", flush=True)
r = backend.chat([Message(role=Role.USER, content="Reply with exactly: AURORA_OK")], [])
print("    ->", repr(r.content), flush=True)

print("[3] agent task (write+read):", flush=True)
kb = KnowledgeBase(cfg, backend)
toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, kb.retrieve)
agent = Agent(cfg, backend, toolbox)

res = agent.run(
    "Write a file named demo.txt containing the text 'Aurora is alive'. "
    "Then read it back and tell me exactly what it says."
)
print("    ANSWER:", res.answer, flush=True)
print("    TOOL_CALLS:", res.tool_calls, flush=True)
print("    STEPS:", [s.kind for s in res.steps], flush=True)

# verify the file actually exists with correct content
p = cfg.workspace / "demo.txt"
print("[4] file check:", p.exists(), repr(p.read_text()) if p.exists() else "MISSING", flush=True)
print("SMOKE_DONE", flush=True)
