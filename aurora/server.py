"""FastAPI HTTP server exposing Aurora as an OpenAI-style service.

Endpoints:
  POST /api/chat        -> run the agent on a prompt (streams JSON steps)
  POST /api/knowledge   -> ingest a file or directory into the knowledge base
  GET  /api/knowledge   -> stats (chunk count)
  GET  /api/health      -> liveness

The server loads models in-process, so a single process serves everything.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from .agent import Agent
from .config import AgentConfig, load_config
from .exceptions import AuroraError
from .llm import LocalBackend
from .memory import Memory
from .rag import KnowledgeBase
from .tools import Toolbox


def build_app(cfg: Optional[AgentConfig] = None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    cfg = cfg or load_config()
    cfg.ensure_dirs()

    backend = LocalBackend(cfg.model_config)
    kb = KnowledgeBase(cfg, backend)
    toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, kb.retrieve)
    agent = Agent(cfg, backend, toolbox)

    app = FastAPI(title="Aurora", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "chunks": kb.count()}

    # Serve the single-file web console at the root.
    from fastapi.responses import HTMLResponse

    _ui = Path(__file__).resolve().parent.parent / "ui" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index():
        if _ui.exists():
            return HTMLResponse(_ui.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Aurora</h1><p>UI not found; use the CLI.</p>", status_code=200)

    @app.get("/api/knowledge")
    def knowledge_stats():
        return {"chunks": kb.count(), "db": str(cfg.db_path)}

    @app.post("/api/knowledge")
    def ingest(body: dict = Body(...)):
        try:
            n = kb.ingest_path(str(body.get("path", "")))
            return {"ingested_chunks": n}
        except Exception as exc:  # pragma: no cover
            raise HTTPException(500, str(exc))

    @app.post("/api/chat")
    async def chat(body: dict = Body(...)):
        prompt = str(body.get("prompt", ""))
        max_steps = body.get("max_steps")
        if not prompt:
            raise HTTPException(400, "prompt is required")
        if not body.get("stream", True):
            res = agent.run(prompt, max_steps=max_steps)
            return {"answer": res.answer, "tool_calls": res.tool_calls, "steps": [s.__dict__ for s in res.steps]}

        async def event_gen():
            queue: asyncio.Queue = asyncio.Queue()

            def on_step(step):
                queue.put_nowait(step)

            async def run():
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(None, lambda: agent.run(prompt, max_steps=max_steps, on_step=on_step))
                queue.put_nowait(("__done__", res))

            task = asyncio.create_task(run())
            while True:
                item = await queue.get()
                if isinstance(item, tuple) and item[0] == "__done__":
                    res = item[1]
                    yield _sse({"type": "done", "answer": res.answer, "tool_calls": res.tool_calls})
                    break
                step = item
                yield _sse({"type": step.kind, "tool": step.tool, "text": step.text, "ok": step.ok})
            await task

        from starlette.responses import StreamingResponse

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
